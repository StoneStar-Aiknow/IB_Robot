#include <torch/extension.h>
#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/GradMode.h>

#include "torch_npu/csrc/framework/OpCommand.h"

namespace {

void check_tensor(const at::Tensor& tensor, const char* name,
                  at::ScalarType dtype, int64_t rank) {
  TORCH_CHECK(tensor.device().type() == c10::DeviceType::PrivateUse1,
              name, " must be an NPU tensor");
  TORCH_CHECK(tensor.layout() == c10::kStrided, name, " must have strided layout");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unsupported dtype");
  TORCH_CHECK(tensor.dim() == rank, name, " has unsupported rank");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void check_forward_only(const at::Tensor& features, const at::Tensor& weight) {
  TORCH_CHECK(!c10::GradMode::is_enabled() ||
                  (!features.requires_grad() && !weight.requires_grad()),
              "SubMConv3d is forward only; use torch.no_grad() or detach inputs");
}

at::Tensor build_subm_map(const at::Tensor& indices, int64_t kernel_size,
                         const c10::optional<at::Tensor>& sorted_keys,
                         const c10::optional<at::Tensor>& source_rows) {
  check_tensor(indices, "indices", at::kInt, 2);
  TORCH_CHECK(indices.size(1) == 4, "indices must have shape [N, 4]");
  TORCH_CHECK(indices.size(0) >= 1 && indices.size(0) <= 4096,
              "N must be in [1, 4096]");
  TORCH_CHECK(kernel_size == 1 || kernel_size == 3 || kernel_size == 5,
              "kernel_size must be 1, 3 or 5");
  TORCH_CHECK(sorted_keys.has_value() == source_rows.has_value(),
              "sorted_keys and source_rows must be both present or both absent");
  if (sorted_keys.has_value()) {
    check_tensor(*sorted_keys, "sorted_keys", at::kLong, 1);
    check_tensor(*source_rows, "source_rows", at::kInt, 1);
    TORCH_CHECK(sorted_keys->device() == indices.device() &&
                    source_rows->device() == indices.device(),
                "indices, sorted_keys and source_rows must be on the same device");
    TORCH_CHECK(sorted_keys->size(0) >= 1 && sorted_keys->size(0) <= indices.size(0),
                "M must be in [1, N]");
    TORCH_CHECK(source_rows->size(0) == sorted_keys->size(0),
                "source_rows must have the same M as sorted_keys");
  }
  const int64_t volume = kernel_size * kernel_size * kernel_size;
  const c10::OptionalDeviceGuard guard(indices.device());
  auto neighbors = at::empty({indices.size(0), ((volume + 7) / 8) * 8},
                            indices.options());
  at_npu::native::OpCommand command;
  command.Name("BuildSubmMap").Input(indices, "indices");
  if (sorted_keys.has_value()) {
    command.Input(*sorted_keys, "sorted_keys").Input(*source_rows, "source_rows");
  } else {
    // CANN None descriptors preserve optional IR input slots 1 and 2.
    command.Input().Input();
  }
  command.Output(neighbors, "neighbors")
      .Attr("kernel_size", kernel_size)
      .Run();
  return neighbors;
}

at::Tensor subm_conv3d(const at::Tensor& features, const at::Tensor& neighbors,
                       const at::Tensor& weight) {
  check_forward_only(features, weight);
  check_tensor(features, "features", at::kHalf, 2);
  check_tensor(neighbors, "neighbors", at::kInt, 2);
  check_tensor(weight, "weight", at::kHalf, 3);
  TORCH_CHECK(features.device() == neighbors.device() &&
                  features.device() == weight.device(),
              "features, neighbors and weight must be on the same device");
  const int64_t n = features.size(0);
  const int64_t cin = features.size(1);
  const int64_t cout = weight.size(2);
  const int64_t volume = weight.size(0);
  TORCH_CHECK(n >= 1 && n <= 4096, "N must be in [1, 4096]");
  TORCH_CHECK(cin >= 16 && cin <= 512 && cin % 16 == 0,
              "Cin must be a multiple of 16 in [16, 512]");
  TORCH_CHECK(cout >= 16 && cout <= 512 && cout % 16 == 0,
              "Cout must be a multiple of 16 in [16, 512]");
  TORCH_CHECK(volume == 1 || volume == 27 || volume == 125,
              "weight must have K^3 in {1, 27, 125}");
  TORCH_CHECK(weight.size(1) == cin, "weight Cin must match features Cin");
  TORCH_CHECK(neighbors.size(0) == n &&
                  neighbors.size(1) == ((volume + 7) / 8) * 8,
              "neighbors must have shape [N, ceil(K^3 / 8) * 8]");
  const c10::OptionalDeviceGuard guard(features.device());
  auto output = at::empty({n, cout}, features.options());
  at_npu::native::OpCommand()
      .Name("SubmConv3d")
      .Input(features, "features")
      .Input(neighbors, "neighbors")
      .Input(weight, "weight")
      .Output(output, "output")
      .Run();
  return output;
}

at::Tensor subm_conv3d_autograd(const at::Tensor& features,
                               const at::Tensor& neighbors,
                               const at::Tensor& weight) {
  check_forward_only(features, weight);
  // Redispatch, rather than calling the NPU kernel, so fake/meta stays symbolic.
  const at::AutoDispatchBelowAutograd guard;
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("graspgenx_subm::subm_conv3d", "")
                       .typed<at::Tensor(const at::Tensor&, const at::Tensor&,
                                         const at::Tensor&)>();
  return op.call(features, neighbors, weight);
}

}  // namespace

TORCH_LIBRARY(graspgenx_subm, module) {
  module.def("build_subm_map(Tensor indices, int kernel_size, "
             "Tensor? sorted_keys=None, Tensor? source_rows=None) -> Tensor");
  module.def("subm_conv3d(Tensor features, Tensor neighbors, Tensor weight) -> Tensor");
}

TORCH_LIBRARY_IMPL(graspgenx_subm, PrivateUse1, module) {
  module.impl("build_subm_map", &build_subm_map);
  module.impl("subm_conv3d", &subm_conv3d);
}

TORCH_LIBRARY_IMPL(graspgenx_subm, Autograd, module) {
  module.impl("subm_conv3d", &subm_conv3d_autograd);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
