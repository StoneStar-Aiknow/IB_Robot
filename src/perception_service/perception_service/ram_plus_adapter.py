"""Canonical RAM++ preprocessing and semantic output decoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from inference_service.generic_runtime import NamedTensorResult

from .perception_adapter import AdapterIdentity, PerceptionAdapter

RAM_PLUS_CLASS_COUNT = 4585
RAM_PLUS_IMAGE_SIZE = 384
RAM_PLUS_PREPROCESSING = "resize384-rgb-imagenet-bilinear-v1"
RAM_PLUS_POSTPROCESSING = "sigmoid-per-class-threshold-score-label-v1"


@dataclass(frozen=True)
class RecognizedTag:
    label: str
    score: float


class RAMPlusAdapter(PerceptionAdapter):
    identity = AdapterIdentity(
        family="ram_plus",
        preprocessing=RAM_PLUS_PREPROCESSING,
        postprocessing=RAM_PLUS_POSTPROCESSING,
        supported_deployments=frozenset({"torch_cpu", "torch_cuda", "ascend_310p", "ascend_310b"}),
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
            "family": cls.identity.family,
            "preprocessing": cls.identity.preprocessing,
            "postprocessing": cls.identity.postprocessing,
        }
        if any(identity.get(name) != value for name, value in expected.items()):
            raise ValueError(f"RAM++ adapter identity mismatch: expected {expected}, got {identity}")
        labels = (root / "assets" / "ram_tag_list.txt").read_text(encoding="utf-8").splitlines()
        thresholds = np.loadtxt(root / "assets" / "ram_tag_list_threshold.txt", dtype=np.float32)
        return cls(labels, thresholds)

    def preprocess(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image must be an RGB uint8 HxWx3 array")
        image = Image.fromarray(image_rgb, mode="RGB").resize(
            (RAM_PLUS_IMAGE_SIZE, RAM_PLUS_IMAGE_SIZE), Image.Resampling.BILINEAR
        )
        value = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
        value = (value - mean) / std
        tensor = np.ascontiguousarray(value.transpose(2, 0, 1)[None], dtype=np.float32)
        return {"observation.image": tensor}

    def postprocess(self, result: NamedTensorResult, *, score_threshold: float = 0.0) -> list[RecognizedTag]:
        try:
            logits = np.asarray(result.outputs["tag_logits"], dtype=np.float32).reshape(-1)
        except KeyError as exc:
            raise RuntimeError("RAM++ runtime result is missing 'tag_logits'") from exc
        if logits.shape != (self.class_count,):
            raise RuntimeError(f"RAM++ logits must contain {self.class_count} values, got {logits.shape}")
        if not np.isfinite(logits).all():
            raise RuntimeError("RAM++ logits contain non-finite values")
        scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        thresholds = self.thresholds
        if score_threshold > 0.0:
            thresholds = np.full_like(scores, score_threshold)
        indices = [
            int(index) for index in np.flatnonzero(scores > thresholds) if int(index) not in self.deleted_indices
        ]
        indices.sort(key=lambda index: (-float(scores[index]), str(self.labels[index])))
        return [RecognizedTag(str(self.labels[index]), float(scores[index])) for index in indices]
