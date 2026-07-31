"""Shared validation and identity helpers for perception model services."""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_MASK_BATCH = 8
MAX_TEXT_BATCH = 16
SUPPORTED_IMAGE_ENCODINGS = {"bgr8", "rgb8"}


@dataclass(frozen=True)
class ModelManifest:
    model_name: str
    model_version: str
    weights_hash: str
    config_hash: str
    backend: str
    runtime_version: str
    preprocessing_hash: str
    embedding_dim: int = 0
    normalization: str = ""

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_image_message(image) -> None:
    if image.encoding not in SUPPORTED_IMAGE_ENCODINGS:
        raise ValueError(f"unsupported RGB image encoding: {image.encoding}")
    if image.height <= 0 or image.width <= 0:
        raise ValueError("image dimensions must be positive")
    if image.step < image.width * 3:
        raise ValueError("RGB image step is smaller than its pixel width")
    if len(image.data) < image.step * image.height:
        raise ValueError("RGB image data is truncated")


def validate_mask_batch(image, masks, max_batch_size: int = MAX_MASK_BATCH) -> None:
    validate_image_message(image)
    if not masks:
        raise ValueError("at least one mask is required")
    if len(masks) > max_batch_size:
        raise ValueError(f"mask batch exceeds limit {max_batch_size}")
    for index, mask in enumerate(masks):
        if mask.encoding != "mono8":
            raise ValueError(f"mask {index} must use mono8 encoding")
        if mask.height != image.height or mask.width != image.width:
            raise ValueError(f"mask {index} dimensions do not match the source image")
        if mask.header.stamp != image.header.stamp:
            raise ValueError(f"mask {index} timestamp does not match the source image")
        if mask.step < mask.width or len(mask.data) < mask.step * mask.height:
            raise ValueError(f"mask {index} data is truncated")


def validate_text_batch(texts, max_batch_size: int = MAX_TEXT_BATCH) -> None:
    if not texts:
        raise ValueError("at least one text is required")
    if len(texts) > max_batch_size:
        raise ValueError(f"text batch exceeds limit {max_batch_size}")
    for index, text in enumerate(texts):
        if not text.strip():
            raise ValueError(f"text {index} must not be empty")


def validate_ready(ready: bool, message: str) -> None:
    if not ready:
        raise RuntimeError(message or "model backend is not ready")
