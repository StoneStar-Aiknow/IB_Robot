"""Forward-only NPU sparse convolution; import after sourcing the CANN environment.

Coordinates are int32 [batch, x, y, z] rows (all signed int32 values are valid).
Coordinate-only lookup requires unique rows. Optional sorted hash keys and
source rows instead preserve the reference's hash collisions/representatives.
A map belongs to the coordinate values/order AND optional table that built it.
Uniqueness and table contents are caller preconditions, not host-side scans.
No CPU execution, implicit conversion, or global topology cache is provided.
"""

import math

import torch
import torch_npu  # noqa: F401 -- registers the NPU device and bundled TorchAir

from torch_models.graspgenx_ascend_310p.custom_ops import load_bridge

# Optional for the reason given in custom_ops/grid_encode.py: a 310P rejects
# graph mode, so the converters below are unreachable there.
try:
    import torchair
    from torchair import ge
except ImportError:  # pragma: no cover - exercised only where TorchAir is absent
    torchair = None
    ge = None


__all__ = ["build_subm_map", "subm_conv3d", "SubMConv3d"]

_LIBRARY = load_bridge("graspgenx_subm", "SubMConv3d")


def _check_tensor(tensor, name, dtype, rank):
    # A real meta tensor is allowed for shape inference, never CPU execution.
    if tensor.device.type not in ("npu", "meta"):
        raise RuntimeError(f"{name} must be an NPU tensor")
    if tensor.layout != torch.strided:
        raise RuntimeError(f"{name} must have strided layout")
    if tensor.dtype != dtype:
        raise RuntimeError(f"{name} has unsupported dtype")
    if tensor.ndim != rank:
        raise RuntimeError(f"{name} has unsupported rank")
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous")


def _check_indices(indices, kernel_size, sorted_keys=None, source_rows=None):
    _check_tensor(indices, "indices", torch.int32, 2)
    if indices.shape[1] != 4:
        raise RuntimeError("indices must have shape [N, 4]")
    if not 1 <= indices.shape[0] <= 4096:
        raise RuntimeError("N must be in [1, 4096]")
    if not isinstance(kernel_size, int) or kernel_size not in (1, 3, 5):
        raise RuntimeError("kernel_size must be 1, 3 or 5")
    if (sorted_keys is None) != (source_rows is None):
        raise RuntimeError("sorted_keys and source_rows must be both present or both absent")
    if sorted_keys is not None:
        _check_tensor(sorted_keys, "sorted_keys", torch.int64, 1)
        _check_tensor(source_rows, "source_rows", torch.int32, 1)
        if sorted_keys.device != indices.device or source_rows.device != indices.device:
            raise RuntimeError("indices, sorted_keys and source_rows must be on the same device")
        if not 1 <= sorted_keys.shape[0] <= indices.shape[0]:
            raise RuntimeError("M must be in [1, N]")
        if source_rows.shape[0] != sorted_keys.shape[0]:
            raise RuntimeError("source_rows must have the same M as sorted_keys")


def _check_conv(features, neighbors, weight):
    if torch.is_grad_enabled() and (features.requires_grad or weight.requires_grad):
        raise RuntimeError("SubMConv3d is forward only; use torch.no_grad() or detach inputs")
    _check_tensor(features, "features", torch.float16, 2)
    _check_tensor(weight, "weight", torch.float16, 3)
    if features.device != weight.device:
        raise RuntimeError("features, neighbors and weight must be on the same device")
    n, cin = features.shape
    volume, weight_cin, cout = weight.shape
    if not 1 <= n <= 4096:
        raise RuntimeError("N must be in [1, 4096]")
    if not (16 <= cin <= 512 and cin % 16 == 0):
        raise RuntimeError("Cin must be a multiple of 16 in [16, 512]")
    if not (16 <= cout <= 512 and cout % 16 == 0):
        raise RuntimeError("Cout must be a multiple of 16 in [16, 512]")
    if volume not in (1, 27, 125):
        raise RuntimeError("weight must have K^3 in {1, 27, 125}")
    if weight_cin != cin:
        raise RuntimeError("weight Cin must match features Cin")
    if neighbors is not None:
        _check_tensor(neighbors, "neighbors", torch.int32, 2)
        if neighbors.device != features.device:
            raise RuntimeError("features, neighbors and weight must be on the same device")
        if neighbors.shape != (n, ((volume + 7) // 8) * 8):
            raise RuntimeError("neighbors must have shape [N, ceil(K^3 / 8) * 8]")


@torch.library.register_fake("graspgenx_subm::build_subm_map")
def _build_subm_map_fake(indices, kernel_size, sorted_keys=None, source_rows=None):
    _check_indices(indices, kernel_size, sorted_keys, source_rows)
    return indices.new_empty((indices.shape[0], ((kernel_size**3 + 7) // 8) * 8))


@torch.library.register_fake("graspgenx_subm::subm_conv3d")
def _subm_conv3d_fake(features, neighbors, weight):
    _check_conv(features, neighbors, weight)
    return features.new_empty((features.shape[0], weight.shape[2]))


if torchair is not None:

    @torchair.register_fx_node_ge_converter(torch.ops.graspgenx_subm.build_subm_map.default)
    def _build_subm_map_ge(indices, kernel_size, sorted_keys=None, source_rows=None, meta_outputs=None):
        return ge.custom_op(
            "BuildSubmMap",
            inputs={"indices": indices, "sorted_keys": sorted_keys, "source_rows": source_rows},
            outputs=["neighbors"],
            attrs={"kernel_size": ge.attr.Int(kernel_size)},
        )

    @torchair.register_fx_node_ge_converter(torch.ops.graspgenx_subm.subm_conv3d.default)
    def _subm_conv3d_ge(features, neighbors, weight, meta_outputs=None):
        return ge.custom_op(
            "SubmConv3d",
            inputs={"features": features, "neighbors": neighbors, "weight": weight},
            outputs=["output"],
        )


def build_subm_map(indices, kernel_size=3, sorted_keys=None, source_rows=None):
    """Return int32 [N, ceil(K^3/8)*8], with -1 for absent/padded neighbors.

    Column order is lexicographic (dx, dy, dz), each in [-(K//2), K//2].
    A column looks up input coordinates at output coordinates + that offset.

    Without a table, rows must be unique and matching is exact-coordinate.
    With a table, pass BOTH contiguous NPU sorted_keys int64 [M] and source_rows
    int32 [M], on the indices device, with 1 <= M <= N. Keys must be unique and
    strictly increasing; source rows in [0, N) are authoritative representatives.
    Contents are not validated. Lookup uses the signed int64 reference hash
    batch*334214467 + x*73856093 + y*19349669 + z*83492791, including collisions.
    For reference parity, select the first entry of each key from the original
    CPU int64 hash.sort(), not a stable sort or a minimum-row reduction. K=1
    still searches this table; only coordinate-only K=1 is identity.
    """
    _check_indices(indices, kernel_size, sorted_keys, source_rows)
    return torch.ops.graspgenx_subm.build_subm_map(indices, kernel_size, sorted_keys, source_rows)


def subm_conv3d(features, indices, weight, bias=None, kernel_size=3, neighbor_map=None):
    """Compute FP16 [N, Cout] from FP16 features and [K^3, Cin, Cout] weight.

    The custom op accumulates in FP32. Optional FP16 [Cout] bias is added
    separately to its FP16 output. Pass neighbor_map only for unchanged indices
    and kernel_size; its contents are not revalidated against coordinates.
    """
    _check_indices(indices, kernel_size)
    # Validate all user inputs before launching even the map-building kernel.
    _check_conv(features, neighbor_map, weight)
    if features.device != indices.device:
        raise RuntimeError("features and indices must be on the same device")
    if features.shape[0] != indices.shape[0]:
        raise RuntimeError("features and indices must have the same N")
    if weight.shape[0] != kernel_size**3:
        raise RuntimeError("weight K^3 must match kernel_size")
    if bias is not None:
        if torch.is_grad_enabled() and bias.requires_grad:
            raise RuntimeError("SubMConv3d is forward only; use torch.no_grad() or detach inputs")
        _check_tensor(bias, "bias", torch.float16, 1)
        if bias.device != features.device:
            raise RuntimeError("bias and features must be on the same device")
        if bias.shape[0] != weight.shape[2]:
            raise RuntimeError("bias must have shape [Cout]")
    if neighbor_map is None:
        neighbor_map = build_subm_map(indices, kernel_size)
    output = torch.ops.graspgenx_subm.subm_conv3d(features, neighbor_map, weight)
    return output if bias is None else output + bias


class SubMConv3d(torch.nn.Module):
    """Inference-only module. Move to NPU and call under torch.no_grad().

    Parameters start on CPU in FP16, with weight layout [K^3, Cin, Cout].
    Calling eval() alone does not disable autograd.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super().__init__()
        for name, channels in (("Cin", in_channels), ("Cout", out_channels)):
            if not isinstance(channels, int) or not (16 <= channels <= 512 and channels % 16 == 0):
                raise RuntimeError(f"{name} must be a multiple of 16 in [16, 512]")
        if not isinstance(kernel_size, int) or kernel_size not in (1, 3, 5):
            raise RuntimeError("kernel_size must be 1, 3 or 5")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.weight = torch.nn.Parameter(torch.empty(kernel_size**3, in_channels, out_channels, dtype=torch.float16))
        self.bias = torch.nn.Parameter(torch.empty(out_channels, dtype=torch.float16)) if bias else None
        bound = 1 / math.sqrt(kernel_size**3 * in_channels)
        torch.nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features, indices, neighbor_map=None):
        return subm_conv3d(features, indices, self.weight, self.bias, self.kernel_size, neighbor_map)
