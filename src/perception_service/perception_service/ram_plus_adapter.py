"""Canonical RAM++ preprocessing and semantic output decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from inference_service.unified_runtime import ModelResult

from .perception_adapter import AdapterIdentity, PerceptionAdapter

RAM_PLUS_CLASS_COUNT = 4585
RAM_PLUS_IMAGE_SIZE = 384
RAM_PLUS_PREPROCESSING = "resize384-rgb-imagenet-bilinear-v1"
RAM_PLUS_POSTPROCESSING = "sigmoid-per-class-threshold-score-label-v1"
RAM_PLUS_COLOR_LABELS = frozenset(
    {
        "black",
        "blue",
        "brown",
        "gray",
        "green",
        "grey",
        "orange",
        "pink",
        "purple",
        "red",
        "white",
        "yellow",
    }
)
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


@dataclass(frozen=True)
class RecognizedTag:
    label: str
    score: float


def select_mask_tags(values, *, excluded_labels=(), limit: int = 0) -> list[RecognizedTag]:
    """Apply caller policy before truncating ranked RAM++ mask candidates."""
    excluded = RAM_PLUS_COLOR_LABELS | {str(label).strip().casefold() for label in excluded_labels}
    candidates = [value for value in values if value.label.casefold() not in excluded]
    return candidates if limit <= 0 else candidates[:limit]


def masked_image_crop(image: np.ndarray, mask: np.ndarray, *, padding_ratio: float = 0.15) -> np.ndarray:
    """Crop one masked object with stable padding and a neutral background."""
    if mask.shape != image.shape[:2] or not np.any(mask):
        raise ValueError("RAM++ mask must be non-empty and match the image dimensions")
    ys, xs = np.where(mask)
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(2, int((x2 - x1) * padding_ratio))
    pad_y = max(2, int((y2 - y1) * padding_ratio))
    x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    x2, y2 = min(image.shape[1], x2 + pad_x), min(image.shape[0], y2 + pad_y)
    crop = image[y1:y2, x1:x2].copy()
    crop[~mask[y1:y2, x1:x2]] = 127
    return crop


class RAMPlusAdapter(PerceptionAdapter):
    identity = AdapterIdentity(
        model_type="ram_plus",
        preprocessing=RAM_PLUS_PREPROCESSING,
        postprocessing=RAM_PLUS_POSTPROCESSING,
        supported_deployments=frozenset({"torch_cpu", "torch_cuda", "ascend_310p", "ascend_310b"}),
        operation="recognize_tags",
    )

    compiled_abi_finalized = True

    def __init__(
        self, labels, thresholds, deleted_indices=(), *, expected_class_count: int = RAM_PLUS_CLASS_COUNT
    ) -> None:
        self.labels = np.asarray(labels).reshape(-1)
        self.thresholds = np.asarray(thresholds, dtype=np.float32).reshape(-1)
        self.deleted_indices = frozenset(int(index) for index in deleted_indices)
        self.class_count = expected_class_count
        if self.labels.shape != (expected_class_count,) or self.thresholds.shape != self.labels.shape:
            raise ValueError(f"RAM++ tag vocabulary and thresholds must each contain {expected_class_count} entries")
        if not np.isfinite(self.thresholds).all():
            raise ValueError("RAM++ class thresholds must be finite")

    @classmethod
    def from_bundle(cls, bundle_root: str | Path, _identity=None, *, model=None, deployment=None) -> RAMPlusAdapter:
        del deployment, model
        root = Path(bundle_root)
        identity = json.loads((root / "assets" / "adapter.json").read_text(encoding="utf-8"))
        expected = {
            "interface": "tensor_model",
            "model_type": cls.identity.model_type,
            "preprocessing": cls.identity.preprocessing,
            "postprocessing": cls.identity.postprocessing,
        }
        if any(identity.get(name) != value for name, value in expected.items()):
            raise ValueError(f"RAM++ adapter identity mismatch: expected {expected}, got {identity}")
        if identity.get("operation") != cls.identity.operation:
            raise ValueError(
                f"RAM++ adapter operation mismatch: expected {cls.identity.operation!r}, "
                f"got {identity.get('operation', '')!r}"
            )
        labels = (root / "assets" / "ram_tag_list.txt").read_text(encoding="utf-8").splitlines()
        thresholds = np.loadtxt(root / "assets" / "ram_tag_list_threshold.txt", dtype=np.float32)
        return cls(labels, thresholds)

    def preprocess(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        return self.preprocess_batch([image_rgb])

    def preprocess_batch(self, images_rgb) -> dict[str, np.ndarray]:
        if not images_rgb:
            raise ValueError("at least one image is required")
        tensors = []
        for image_rgb in images_rgb:
            if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
                raise ValueError("image must be an RGB uint8 HxWx3 array")
            image = Image.fromarray(image_rgb, mode="RGB").resize((RAM_PLUS_IMAGE_SIZE, RAM_PLUS_IMAGE_SIZE), _BILINEAR)
            tensors.append(np.asarray(image, dtype=np.float32) / np.float32(255.0))
        value = np.stack(tensors)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        value = (value - mean) / std
        return {"observation.image": np.ascontiguousarray(value.transpose(0, 3, 1, 2), dtype=np.float32)}

    def _scores(self, result: ModelResult) -> np.ndarray:
        try:
            logits = np.asarray(result.outputs["tag_logits"], dtype=np.float32)
        except KeyError as exc:
            raise RuntimeError("RAM++ runtime result is missing 'tag_logits'") from exc
        if logits.ndim == 1:
            logits = logits[None]
        if logits.ndim != 2 or logits.shape[1] != self.class_count:
            raise RuntimeError(f"RAM++ logits must have shape [batch, {self.class_count}], got {logits.shape}")
        if not np.isfinite(logits).all():
            raise RuntimeError("RAM++ logits contain non-finite values")
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))

    def postprocess_batch(self, result: ModelResult, *, score_threshold: float = 0.0):
        scores = self._scores(result)
        thresholds = self.thresholds
        if score_threshold > 0.0:
            thresholds = np.full_like(self.thresholds, score_threshold)
        output = []
        for row in scores:
            indices = [
                int(index) for index in np.flatnonzero(row > thresholds) if int(index) not in self.deleted_indices
            ]
            indices.sort(key=lambda index: (-float(row[index]), str(self.labels[index])))
            output.append([RecognizedTag(str(self.labels[index]), float(row[index])) for index in indices])
        return output

    def postprocess(self, result: ModelResult, *, score_threshold: float = 0.0) -> list[RecognizedTag]:
        values = self.postprocess_batch(result, score_threshold=score_threshold)
        if len(values) != 1:
            raise RuntimeError("RAM++ single-image postprocess received batched output")
        return values[0]
