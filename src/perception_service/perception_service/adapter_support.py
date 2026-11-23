"""Model-agnostic helpers shared by perception adapters."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from inference_service.unified_runtime import ModelResult

from .perception_adapter import AdapterIdentity

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_image_tensor(image_hwc: np.ndarray, layout: str = "NCHW") -> np.ndarray:
    """Return an RGB uint8 frame as a batched contiguous float32 ImageNet tensor.

    Scale, shift and the channel-order change happen in one pass per channel, written
    straight into the destination buffer. Transcribing the contract directly --
    ``(image / 255 - mean) / std`` followed by a transpose -- allocates four full-frame
    temporaries and divides by a broadcast array; on 310P that measured 61 ms of a
    435 ms Grounding-DINO frame against 11 ms for this form, for the same arithmetic.
    """

    image = np.asarray(image_hwc)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"ImageNet preprocessing needs an HxWx3 RGB frame, got shape {image.shape}")
    height, width = image.shape[:2]
    if layout == "NCHW":
        tensor = np.empty((1, 3, height, width), dtype=np.float32)
        planes = [tensor[0, index] for index in range(3)]
    else:
        tensor = np.empty((1, height, width, 3), dtype=np.float32)
        planes = [tensor[0, ..., index] for index in range(3)]
    for index, plane in enumerate(planes):
        np.copyto(plane, image[..., index], casting="unsafe")
        plane *= np.float32(1.0 / (255.0 * IMAGENET_STD[index]))
        plane -= np.float32(IMAGENET_MEAN[index] / IMAGENET_STD[index])
    return tensor


def raw_image_tensor(image_hwc: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 frame batched and contiguous, with its values untouched.

    For a deployment whose AIPP stage performs the ImageNet normalization in hardware,
    this is the whole of the host's image preprocessing.
    """

    image = np.asarray(image_hwc)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"a raw AIPP input needs an HxWx3 uint8 RGB frame, got {image.shape} of {image.dtype}")
    return np.ascontiguousarray(image[None])


def read_adapter_identity(root: Path, expected: AdapterIdentity) -> None:
    import json

    path = root / "assets" / "adapter.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load adapter identity {path}: {exc}") from exc
    required = {
        "interface": "tensor_model",
        "model_type": expected.model_type,
        "preprocessing": expected.preprocessing,
        "postprocessing": expected.postprocessing,
    }
    if any(value.get(name) != expected_value for name, expected_value in required.items()):
        raise ValueError(f"{expected.model_type} adapter identity mismatch: expected {required}, got {value}")
    declared_operation = value.get("operation")
    if declared_operation != expected.operation:
        raise ValueError(
            f"{expected.model_type} adapter operation mismatch: expected {expected.operation!r}, got {declared_operation!r}"
        )


def output_tensor(result: ModelResult, semantic: str, dtype=np.float32) -> np.ndarray:
    try:
        value = np.asarray(result.outputs[semantic], dtype=dtype)
    except KeyError as exc:
        raise RuntimeError(f"runtime result is missing {semantic!r}") from exc
    if not np.isfinite(value).all():
        raise RuntimeError(f"runtime output {semantic!r} contains non-finite values")
    return value


__all__ = [
    "imagenet_image_tensor",
    "output_tensor",
    "raw_image_tensor",
    "read_adapter_identity",
]
