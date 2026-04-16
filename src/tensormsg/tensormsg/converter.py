# tensormsg/converter.py
import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from rosidl_runtime_py.utilities import get_message
from torch import Tensor

from tensormsg.registry import DECODER_REGISTRY, ENCODER_REGISTRY, register_decoder, register_encoder
from tensormsg.utils import dot_get, dot_set, nearest_resize_any, nearest_resize_rgb

logger = logging.getLogger(__name__)


class TensorMsgConverter:
    """Central converter for ROS messages and Tensors."""

    @staticmethod
    def encode(
        ros_type: str,
        data: np.ndarray | Tensor | Sequence,
        names: list[str] | None = None,
        clamp: tuple[float, float] | None = None,
    ) -> Any:
        encoder = ENCODER_REGISTRY.get(ros_type)
        if not encoder:
            # Fallback for simple types or explicit dotted paths if names are provided
            if names:
                return _encode_via_dotted_paths(ros_type, names, data, clamp)
            raise ValueError(f"No encoder registered for {ros_type}")
        return encoder(names, data, clamp)

    @staticmethod
    def decode(msg, spec: Any = None) -> np.ndarray:
        """
        Decode a ROS message to a numpy array.
        spec can be an object with .names, .image_encoding, .image_resize attributes.
        """
        pkg_name = msg.__class__.__module__.split(".")[0]
        ros_type = f"{pkg_name}/msg/{msg.__class__.__name__}"

        decoder = DECODER_REGISTRY.get(ros_type)
        if not decoder:
            # Try to decode via names if spec has them
            if spec and hasattr(spec, "names") and spec.names:
                return _decode_via_names(msg, spec.names)
            raise ValueError(f"No decoder registered for {ros_type}")
        return decoder(msg, spec)

    @staticmethod
    def to_variant(batch: dict[str, Any]) -> Any:
        """Encode a dictionary of Tensors into a ibrobot_msgs/msg/VariantsList."""
        msg_cls = get_message("ibrobot_msgs/msg/VariantsList")
        msg = msg_cls()
        msg.variants = []

        for key, value in batch.items():
            if not any(key.startswith(p) for p in ["task", "observation", "action"]):
                continue

            variant_msg = get_message("ibrobot_msgs/msg/Variant")()
            variant_msg.key = key

            if isinstance(value, Tensor):
                _fill_variant_from_tensor(variant_msg, value)
            elif isinstance(value, list) and all(isinstance(x, str) for x in value):
                variant_msg.type = "string_array"
                variant_msg.string_array = value
            else:
                continue
            msg.variants.append(variant_msg)
        return msg

    @staticmethod
    def from_variant(msg, device: torch.device | None = None) -> dict[str, Any]:
        """Decode a ibrobot_msgs/msg/VariantsList into a dictionary of Tensors."""
        result = {}
        for variant_msg in msg.variants:
            result[str(variant_msg.key)] = _decode_variant(variant_msg, device)
        return result


# ---------- Internal Helpers ----------


def _encode_via_dotted_paths(
    ros_type: str, names: list[str], data: Any, clamp: tuple[float, float] | None = None
) -> Any:
    msg_cls = get_message(ros_type)
    msg = msg_cls()
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])
    for i, path in enumerate(names):
        if i < arr.size:
            dot_set(msg, path, float(arr[i]))
    return msg


def _decode_via_names(msg, names: list[str]) -> np.ndarray:
    out: list[float] = []
    for name in names:
        try:
            out.append(float(dot_get(msg, name)))
        except Exception:
            out.append(float("nan"))
    return np.asarray(out, dtype=np.float32)


def _fill_variant_from_tensor(variant_msg, vec: Tensor):
    if vec.dtype == torch.bool:
        variant_msg.type = "bool_array"
        variant_msg.bool_array = vec.reshape(-1).tolist()
    elif vec.dtype == torch.int32:
        variant_msg.type = "int_32_array"
        variant_msg.int_32_array = _create_multiarray_msg(vec, "Int32")
    elif vec.dtype == torch.int64:
        variant_msg.type = "int_64_array"
        variant_msg.int_64_array = _create_multiarray_msg(vec, "Int64")
    elif vec.dtype == torch.float32:
        variant_msg.type = "float_32_array"
        variant_msg.float_32_array = _create_multiarray_msg(vec, "Float32")
    elif vec.dtype == torch.float64:
        variant_msg.type = "float_64_array"
        variant_msg.float_64_array = _create_multiarray_msg(vec, "Float64")
    else:
        raise ValueError(f"Unsupported dtype {vec.dtype}")


def _create_multiarray_msg(vec: Tensor | np.ndarray, msg_type: str):
    if isinstance(vec, Tensor):
        v_np = vec.detach().cpu().numpy()
    else:
        v_np = vec

    msg_cls_name = f"std_msgs/msg/{msg_type}MultiArray"
    msg = get_message(msg_cls_name)()
    msg.data = v_np.reshape(-1).tolist()

    for size in v_np.shape:
        dim = get_message("std_msgs/msg/MultiArrayDimension")()
        dim.size = int(size)
        msg.layout.dim.append(dim)
    return msg


def _decode_variant(msg, device: torch.device | None = None) -> Any:
    v_type = str(msg.type).strip()
    if v_type == "string_array":
        return list(msg.string_array)

    type_map = {
        "bool_array": (msg.bool_array, torch.bool),
        "int_32_array": (msg.int_32_array, torch.int32),
        "int_64_array": (msg.int_64_array, torch.int64),
        "float_32_array": (msg.float_32_array, torch.float32),
        "float_64_array": (msg.float_64_array, torch.float64),
    }

    if v_type not in type_map:
        raise ValueError(f"Unsupported variant type: {v_type}")

    data_source, torch_dtype = type_map[v_type]

    if v_type == "bool_array":
        res = torch.tensor(data_source, dtype=torch_dtype).unsqueeze(0)
    else:
        res = torch.tensor(data_source.data, dtype=torch_dtype)
        if data_source.layout.dim:
            shape = tuple(dim.size for dim in data_source.layout.dim)
            res = res.reshape(shape)

    if device:
        res = res.to(device)
    return res


# ---------- Registration of Standard Types ----------


@register_encoder("geometry_msgs/msg/Twist")
def _enc_twist(names, data, clamp):
    if names:
        return _encode_via_dotted_paths("geometry_msgs/msg/Twist", names, data, clamp)
    msg = get_message("geometry_msgs/msg/Twist")()
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])
    if len(arr) >= 1:
        msg.linear.x = float(arr[0])
    if len(arr) >= 2:
        msg.angular.z = float(arr[1])
    return msg


@register_decoder("sensor_msgs/msg/Image")
def _dec_image(msg, spec):
    """
    Robust image decoder for common ROS encodings.
    Ported from rosetta/common/decoders.py
    """
    if spec and hasattr(spec, "names") and spec.names:
        return _decode_via_names(msg, spec.names)

    h, w = int(msg.height), int(msg.width)
    enc = getattr(msg, "encoding", "bgr8").lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    step = int(getattr(msg, "step", 0))

    resize_hw = spec.image_resize if spec and hasattr(spec, "image_resize") else None

    # Depth handling (simplified port)
    if enc in ("32fc1", "32fc"):
        data32 = raw.view(np.float32)
        row_elems = (step // 4) if step else w
        arr = data32.reshape(h, row_elems)[:, :w].reshape(h, w)
        hwc = arr[..., None]
        if resize_hw:
            hwc = nearest_resize_any(hwc, int(resize_hw[0]), int(resize_hw[1]))
        hwc_normalized = np.where(np.isfinite(hwc), np.clip(hwc, 0, 50) / 50, hwc)
        return np.repeat(hwc_normalized, 3, axis=-1).astype(np.float32)

    elif enc in ("16uc1", "mono16"):
        data16 = raw.view(np.uint16)
        row_elems = (step // 2) if step else w
        arr16 = data16.reshape(h, row_elems)[:, :w].reshape(h, w)
        arr_m = arr16.astype(np.float32)
        arr_m[arr16 == 0] = np.nan
        arr_m[arr16 != 0] *= 1.0 / 1000.0
        hwc = arr_m[..., None]
        if resize_hw:
            hwc = nearest_resize_any(hwc, int(resize_hw[0]), int(resize_hw[1]))
        hwc_normalized = np.where(np.isfinite(hwc), np.clip(hwc, 0, 10) / 10, hwc)
        return np.repeat(hwc_normalized, 3, axis=-1).astype(np.float32)

    # Color handling
    if enc in ("rgb8", "bgr8", "8uc3"):
        # NOTE: 8uc3 is currently a decode-only compatibility addition.
        ch = 3
        if not step:
            step = w * ch
        if step < w * ch:
            logger.warning(
                "Suspicious step=%s for %sx%s image (encoding=%r), using step=%s",
                step,
                w,
                ch,
                enc,
                w * ch,
            )
            step = w * ch
        row = raw.reshape(h, step)[:, : w * ch]
        arr = row.reshape(h, w, ch)
        hwc_rgb = arr if enc == "rgb8" else arr[..., ::-1]
    elif enc in ("rgba8", "bgra8", "8uc4"):
        # NOTE: 8uc4 is currently a decode-only compatibility addition.
        ch = 4
        if not step:
            step = w * ch
        if step < w * ch:
            logger.warning(
                "Suspicious step=%s for %sx%s image (encoding=%r), using step=%s",
                step,
                w,
                ch,
                enc,
                w * ch,
            )
            step = w * ch
        row = raw.reshape(h, step)[:, : w * ch]
        arr = row.reshape(h, w, ch)
        rgb = arr[..., :3]
        hwc_rgb = rgb if enc == "rgba8" else rgb[..., ::-1]
    elif enc in ("mono8", "8uc1"):
        if not step:
            step = w
        arr = raw.reshape(h, step)[:, :w].reshape(h, w)
        hwc_rgb = np.repeat(arr[..., None], 3, axis=-1)
    else:
        raise ValueError(f"Unsupported image encoding '{enc}'")

    if resize_hw:
        hwc_rgb = nearest_resize_rgb(hwc_rgb, int(resize_hw[0]), int(resize_hw[1]))

    return hwc_rgb.astype(np.float32) / 255.0


@register_decoder("sensor_msgs/msg/JointState")
def _dec_joint_state(msg, spec):
    if spec and hasattr(spec, "names") and spec.names:
        return _decode_via_names(msg, spec.names)
    return np.asarray(msg.position, dtype=np.float32)


@register_encoder("sensor_msgs/msg/JointState")
def _enc_joint_state(names, data, clamp):
    msg = get_message("sensor_msgs/msg/JointState")()
    if not names:
        return msg
    msg.name = [n.split(".")[1] for n in names]
    arr = np.asarray(data, dtype=np.float32).reshape(-1)
    if clamp:
        arr = np.clip(arr, clamp[0], clamp[1])
    for i, path in enumerate(names):
        dot_set(msg, path, float(arr[i]))
    return msg


@register_decoder("std_msgs/msg/Float32MultiArray")
def _dec_f32(msg, spec):
    return np.asarray(msg.data, dtype=np.float32)


@register_decoder("std_msgs/msg/Float64MultiArray")
def _dec_f64(msg, spec):
    return np.asarray(msg.data, dtype=np.float64)


@register_decoder("std_msgs/msg/Int32MultiArray")
def _dec_i32(msg, spec):
    return np.asarray(msg.data, dtype=np.int32)


@register_decoder("sensor_msgs/msg/PointCloud2")
def _dec_pointcloud2(msg, spec):
    """
    解码无序 PointCloud2（height=1, width=N_valid）。
    返回 {"xyz": (N,3) float32, "rgb": (N,3) uint8}。
    """
    import sensor_msgs_py.point_cloud2 as pc2

    N = msg.width  # height=1 for unorganized cloud

    xyz = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=False)
    xyz = xyz.reshape(N, 3).astype(np.float32)

    rgb = np.zeros((N, 3), dtype=np.uint8)
    field_names = [f.name for f in msg.fields]
    if "rgb" in field_names:
        rgb_raw = pc2.read_points_numpy(msg, field_names=("rgb",), skip_nans=False)
        rgb_packed = rgb_raw.reshape(N).view(np.uint32)
        rgb[:, 0] = (rgb_packed >> 16) & 0xFF  # R
        rgb[:, 1] = (rgb_packed >> 8) & 0xFF  # G
        rgb[:, 2] = rgb_packed & 0xFF  # B

    return {"xyz": xyz, "rgb": rgb}
