#include <torch/extension.h>
#include <c10/core/DeviceGuard.h>

#include "torch_npu/csrc/framework/OpCommand.h"

namespace {
at::Tensor grid_encode(const at::Tensor& grid_coord, int64_t depth) {
    TORCH_CHECK(grid_coord.device().type() == c10::DeviceType::PrivateUse1,
                "grid_coord must be an NPU tensor");
    TORCH_CHECK(grid_coord.layout() == c10::kStrided,
                "grid_coord must have strided layout");
    TORCH_CHECK(grid_coord.scalar_type() == at::kInt,
                "grid_coord must have dtype int32");
    TORCH_CHECK(grid_coord.dim() == 2 && grid_coord.size(1) == 3,
                "grid_coord must have shape [N, 3]");
    TORCH_CHECK(grid_coord.size(0) >= 1 && grid_coord.size(0) <= 4096,
                "N must be in [1, 4096]");
    TORCH_CHECK(grid_coord.is_contiguous(), "grid_coord must be contiguous");
    TORCH_CHECK(depth >= 1 && depth <= 16, "depth must be in [1, 16]");
    const c10::OptionalDeviceGuard guard(grid_coord.device());
    auto spatial_codes = at::empty({grid_coord.size(0), 4}, grid_coord.options().dtype(at::kLong));
    at_npu::native::OpCommand()
        .Name("GridEncode")
        .Input(grid_coord, "grid_coord")
        .Output(spatial_codes, "spatial_codes")
        .Attr("depth", depth)
        .Run();
    return spatial_codes;
}
}

TORCH_LIBRARY(graspgenx_grid, module) {
    module.def("grid_encode(Tensor grid_coord, int depth) -> Tensor");
}

TORCH_LIBRARY_IMPL(graspgenx_grid, PrivateUse1, module) {
    module.impl("grid_encode", &grid_encode);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {}
