"""Semantic adapters for manifest-bound perception model sessions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from inference_service.generic_runtime import NamedTensorResult

from .model_contracts import MAX_MASK_BATCH, MAX_TEXT_BATCH
from .perception_adapter import AdapterIdentity, PerceptionAdapter


def _read_adapter_identity(root: Path, expected: AdapterIdentity) -> None:
    import json

    path = root / "assets" / "adapter.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"cannot load adapter identity {path}: {exc}") from exc
    required = {
        "family": expected.family,
        "preprocessing": expected.preprocessing,
        "postprocessing": expected.postprocessing,
    }
    if any(value.get(name) != expected_value for name, expected_value in required.items()):
        raise ValueError(f"{expected.family} adapter identity mismatch: expected {required}, got {value}")


def _load_siglip2_tokenizer(model_path: Path):
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("SigLIP2 requires transformers for bundle-local tokenization") from exc
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def _output(result: NamedTensorResult, semantic: str, dtype=np.float32) -> np.ndarray:
    try:
        value = np.asarray(result.outputs[semantic], dtype=dtype)
    except KeyError as exc:
        raise RuntimeError(f"runtime result is missing {semantic!r}") from exc
    if not np.isfinite(value).all():
        raise RuntimeError(f"runtime output {semantic!r} contains non-finite values")
    return value


def _normalize(rows: np.ndarray, dimension: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows[None]
    if rows.ndim != 2 or rows.shape[1] != dimension:
        raise RuntimeError(f"embedding output must have shape (batch, {dimension}), got {rows.shape}")
    if not np.isfinite(rows).all():
        raise RuntimeError("embedding output contains non-finite values")
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise RuntimeError("embedding output contains a zero-norm vector")
    return np.ascontiguousarray(rows / norms, dtype=np.float32)


def _resize_rgb_batch(images: list[Image.Image], size: int) -> np.ndarray:
    rows = []
    for image in images:
        value = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
        rows.append(value.transpose(2, 0, 1) / np.float32(127.5) - np.float32(1.0))
    return np.ascontiguousarray(rows, dtype=np.float32)


@dataclass(frozen=True)
class SegmentationMask:
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    score: float
    stability_score: float
    area: int


class SAM2Adapter(PerceptionAdapter):
    identity = AdapterIdentity(
        "sam2", "sam2-rgb-uint8-v1", "automatic-masks-v1", frozenset({"torch_cpu", "torch_cuda"})
    )
    compiled_abi_finalized = False

    @classmethod
    def from_bundle(cls, bundle_root: str | Path, _identity=None) -> SAM2Adapter:
        _read_adapter_identity(Path(bundle_root), cls.identity)
        return cls()

    def preprocess(self, image_rgb: np.ndarray) -> dict[str, np.ndarray]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image must be an RGB uint8 HxWx3 array")
        return {"observation.image": np.ascontiguousarray(image_rgb)}

    def postprocess(self, result: NamedTensorResult, *, image_shape=None, **_options) -> list[SegmentationMask]:
        masks = _output(result, "masks", np.uint8)
        boxes = _output(result, "boxes", np.float32)
        scores = _output(result, "scores", np.float32).reshape(-1)
        stability = _output(result, "stability_scores", np.float32).reshape(-1)
        if (
            masks.ndim != 3
            or boxes.shape != (len(masks), 4)
            or scores.shape != (len(masks),)
            or stability.shape != (len(masks),)
        ):
            raise RuntimeError("SAM2 outputs have inconsistent batch dimensions")
        if image_shape is not None and masks.shape[1:] != tuple(image_shape):
            raise RuntimeError("SAM2 returned masks with dimensions different from the source image")
        return [
            SegmentationMask(
                mask=(masks[index] > 0).astype(np.uint8),
                bbox_xyxy=boxes[index].copy(),
                score=float(scores[index]),
                stability_score=float(stability[index]),
                area=int(np.count_nonzero(masks[index])),
            )
            for index in range(len(masks))
        ]


@dataclass(frozen=True)
class MaskEncoding:
    mask_index: int
    embedding: np.ndarray
    matched_label: str
    matched_score: float


class _SigLIP2Adapter(PerceptionAdapter):
    compiled_abi_finalized = False

    def __init__(self, dimension: int, tokenizer) -> None:
        self.dimension = dimension
        self.tokenizer = tokenizer

    @classmethod
    def from_bundle(cls, bundle_root: str | Path, identity) -> _SigLIP2Adapter:
        _read_adapter_identity(Path(bundle_root), cls.identity)
        if identity is None or identity.embedding is None:
            raise ValueError("SigLIP2 manifest semantic_identity must declare embedding metadata")
        embedding = identity.embedding
        if embedding.normalization != "l2":
            raise ValueError("SigLIP2 embedding normalization must be 'l2'")
        if not embedding.embedding_space_id:
            raise ValueError("SigLIP2 embedding_space_id must be non-empty")
        model_path = Path(bundle_root) / "assets" / "model"
        tokenizer = _load_siglip2_tokenizer(model_path)
        return cls(embedding.dimension, tokenizer)

    def _tokenize(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        if not texts:
            empty = np.empty((0, 64), dtype=np.int64)
            return empty, empty.copy()
        encoded = self.tokenizer(
            [f"This is a photo of {text}." for text in texts],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="np",
        )
        tokens = np.ascontiguousarray(encoded["input_ids"], dtype=np.int64)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            pad_token_id = self.tokenizer.pad_token_id
            attention_mask = np.ones_like(tokens) if pad_token_id is None else tokens != pad_token_id
        attention = np.ascontiguousarray(attention_mask, dtype=np.int64)
        return tokens, attention


class SigLIP2ImageAdapter(_SigLIP2Adapter):
    identity = AdapterIdentity(
        "siglip2",
        "siglip2-dual-encoder-v2",
        "normalized-embedding-v1",
        frozenset({"torch_cpu", "torch_cuda"}),
    )

    def preprocess(self, value: object) -> dict[str, np.ndarray]:
        image_rgb, masks, candidate_labels = value
        if not 1 <= len(masks) <= MAX_MASK_BATCH:
            raise ValueError(f"mask batch must contain between 1 and {MAX_MASK_BATCH} masks")
        crops = []
        for index, raw_mask in enumerate(masks):
            mask = np.asarray(raw_mask) > 0
            if mask.shape != image_rgb.shape[:2]:
                raise ValueError(f"mask {index} dimensions do not match the source image")
            ys, xs = np.where(mask)
            if not len(xs):
                raise ValueError(f"mask {index} is empty")
            x1, x2, y1, y2 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
            crop = image_rgb[y1:y2, x1:x2].copy()
            crop[~mask[y1:y2, x1:x2]] = 127
            crops.append(Image.fromarray(crop))
        tokens, attention = self._tokenize(list(candidate_labels))
        return {
            "masked_images": _resize_rgb_batch(crops, 384),
            "text_tokens": tokens,
            "text_attention_mask": attention,
        }

    def postprocess(self, result: NamedTensorResult, *, candidate_labels=(), **_options) -> list[MaskEncoding]:
        image_features = _normalize(_output(result, "image_embeddings"), self.dimension)
        text_features = _normalize(_output(result, "text_embeddings"), self.dimension) if candidate_labels else None
        if text_features is not None and len(text_features) != len(candidate_labels):
            raise RuntimeError("SigLIP2 returned a different embedding count than the candidate-label batch")
        records = []
        for index, embedding in enumerate(image_features):
            label = ""
            score = 0.0
            if text_features is not None:
                similarities = text_features @ embedding
                best = int(np.argmax(similarities))
                label = candidate_labels[best]
                score = float(similarities[best])
            records.append(MaskEncoding(index, embedding, label, score))
        return records


class SigLIP2TextAdapter(_SigLIP2Adapter):
    identity = AdapterIdentity(
        "siglip2",
        "siglip2-dual-encoder-v2",
        "normalized-embedding-v1",
        frozenset({"torch_cpu", "torch_cuda"}),
    )

    def preprocess(self, texts: object) -> dict[str, np.ndarray]:
        values = list(texts)
        if not 1 <= len(values) <= MAX_TEXT_BATCH:
            raise ValueError(f"text batch must contain between 1 and {MAX_TEXT_BATCH} texts")
        tokens, attention = self._tokenize(values)
        return {
            "masked_images": np.empty((0, 3, 384, 384), dtype=np.float32),
            "text_tokens": tokens,
            "text_attention_mask": attention,
        }

    def postprocess(self, result: NamedTensorResult, **_options) -> np.ndarray:
        return _normalize(_output(result, "text_embeddings"), self.dimension)


@dataclass(frozen=True)
class GroundingDetection:
    label: str
    confidence: float
    bbox_xyxy: np.ndarray
    mask: np.ndarray


class GroundingDINOAdapter(PerceptionAdapter):
    identity = AdapterIdentity(
        "grounding_dino",
        "grounded-sam2-rgb-text-thresholds-v2",
        "boxes-scores-labels-masks-v1",
        frozenset({"torch_cpu", "torch_cuda"}),
    )
    compiled_abi_finalized = False

    @classmethod
    def from_bundle(cls, bundle_root: str | Path, _identity=None) -> GroundingDINOAdapter:
        _read_adapter_identity(Path(bundle_root), cls.identity)
        return cls()

    def preprocess(self, value: object) -> dict[str, np.ndarray]:
        image_rgb, prompt, box_threshold, text_threshold = value
        prompt_bytes = prompt.encode()
        return {
            "observation.image": np.ascontiguousarray(image_rgb, dtype=np.uint8),
            "text_prompt": np.frombuffer(prompt_bytes, dtype=np.uint8).copy(),
            "box_threshold": np.asarray([box_threshold or 0.35], dtype=np.float32),
            "text_threshold": np.asarray([text_threshold or 0.25], dtype=np.float32),
        }

    def postprocess(
        self, result: NamedTensorResult, *, image_shape=None, labels=(), **_options
    ) -> list[GroundingDetection]:
        boxes = _output(result, "boxes", np.float32)
        scores = _output(result, "scores", np.float32).reshape(-1)
        masks = _output(result, "masks", np.uint8)
        label_indices = np.asarray(result.outputs.get("label_indices"), dtype=np.int32).reshape(-1)
        if (
            boxes.shape != (len(scores), 4)
            or masks.ndim != 3
            or len(masks) != len(scores)
            or len(label_indices) != len(scores)
        ):
            raise RuntimeError("Grounding DINO outputs have inconsistent batch dimensions")
        if image_shape is not None and masks.shape[1:] != tuple(image_shape):
            raise RuntimeError("Grounding DINO returned masks with dimensions different from the source image")
        vocabulary = result.metadata.get("labels", labels)
        return [
            GroundingDetection(
                label=(
                    str(vocabulary[label_indices[index]])
                    if 0 <= label_indices[index] < len(vocabulary)
                    else str(label_indices[index])
                ),
                confidence=float(scores[index]),
                bbox_xyxy=boxes[index].copy(),
                mask=(masks[index] > 0).astype(np.uint8),
            )
            for index in range(len(scores))
        ]


__all__ = [
    "GroundingDINOAdapter",
    "MaskEncoding",
    "SAM2Adapter",
    "SegmentationMask",
    "SigLIP2ImageAdapter",
    "SigLIP2TextAdapter",
]
