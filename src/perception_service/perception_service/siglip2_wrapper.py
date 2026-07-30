"""CUDA/CPU SigLIP2 masked-image encoding and candidate matching wrapper."""

from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from .ascend_om_contracts import require_om_adapter_ready
from .model_contracts import MAX_MASK_BATCH, validate_text_batch
from .model_utils import inspect_backend, resolve_model_path


@dataclass(frozen=True)
class MaskEncoding:
    mask_index: int
    embedding: np.ndarray
    matched_label: str
    matched_score: float


class SigLIP2Wrapper:
    def __init__(
        self,
        *,
        backend: str = "cuda",
        model_path: str = "siglip2_so400m_patch14_384/assets/model",
        model_dir: str | None = None,
        model=None,
        processor=None,
        image_encoder=None,
        text_encoder=None,
    ):
        if backend == "ascend_om":
            require_om_adapter_ready("siglip2")
        status = inspect_backend(backend)
        if not status.ready:
            raise RuntimeError(status.message)
        self.backend = backend
        self.runtime_version = status.runtime_version
        self.model_path = resolve_model_path(model_path, model_dir)

        if image_encoder is not None and text_encoder is not None:
            self._model = model
            self._processor = processor
            self._encode_images = image_encoder
            self._encode_texts = text_encoder
            return
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"SigLIP2 model directory not found: {self.model_path}")
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("SigLIP2 requires torch and transformers") from exc

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        self._model = AutoModel.from_pretrained(self.model_path, local_files_only=True).eval().to(backend)
        self._encode_images = self._model.get_image_features
        self._encode_texts = self._model.get_text_features

    def _inference_context(self):
        torch = getattr(self, "_torch", None)
        return nullcontext() if torch is None else torch.inference_mode()

    @staticmethod
    def _normalize(rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim == 1:
            rows = rows[None, :]
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        if np.any(norms <= 1e-12):
            raise RuntimeError("SigLIP2 returned a zero-norm embedding")
        return rows / norms

    @staticmethod
    def _masked_crops(image_rgb: np.ndarray, masks: list[np.ndarray]):
        from PIL import Image

        crops = []
        for index, mask in enumerate(masks):
            mask = np.asarray(mask) > 0
            if mask.shape != image_rgb.shape[:2]:
                raise ValueError(f"mask {index} dimensions do not match the source image")
            ys, xs = np.where(mask)
            if not len(xs):
                raise ValueError(f"mask {index} is empty")
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            crop = image_rgb[y1:y2, x1:x2].copy()
            crop[~mask[y1:y2, x1:x2]] = 127
            crops.append(Image.fromarray(crop))
        return crops

    def encode_text(self, texts: list[str]) -> np.ndarray:
        validate_text_batch(texts)
        prompts = [f"This is a photo of {text}." for text in texts]
        if self._processor is not None:
            text_inputs = self._processor(
                text=prompts,
                padding="max_length",
                max_length=64,
                truncation=True,
                return_tensors="pt",
            )
            text_inputs = {key: value.to(self.backend) for key, value in text_inputs.items()}
            with self._inference_context():
                features = self._encode_texts(**text_inputs)
        else:
            features = self._encode_texts(prompts)
        if hasattr(features, "detach"):
            features = features.detach().float().cpu().numpy()
        return self._normalize(features)

    def encode(self, image_rgb: np.ndarray, masks: list[np.ndarray], candidate_labels: list[str]) -> list[MaskEncoding]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image must be an RGB uint8 HxWx3 array")
        if not 1 <= len(masks) <= MAX_MASK_BATCH:
            raise ValueError(f"mask batch must contain between 1 and {MAX_MASK_BATCH} masks")
        crops = self._masked_crops(image_rgb, masks)

        if self._processor is not None:
            image_inputs = self._processor(images=crops, return_tensors="pt")
            image_inputs = {key: value.to(self.backend) for key, value in image_inputs.items()}
            with self._inference_context():
                image_features = self._encode_images(**image_inputs)
        else:
            image_features = self._encode_images(crops)
        if hasattr(image_features, "detach"):
            image_features = image_features.detach().float().cpu().numpy()
        image_features = self._normalize(image_features)

        text_features = None
        if candidate_labels:
            text_features = self.encode_text(candidate_labels)
            if text_features.shape[1] != image_features.shape[1]:
                raise RuntimeError("SigLIP2 image and text embedding dimensions differ")

        results = []
        for index, embedding in enumerate(image_features):
            label = ""
            score = 0.0
            if text_features is not None:
                similarities = text_features @ embedding
                best = int(np.argmax(similarities))
                label = candidate_labels[best]
                score = float(similarities[best])
            results.append(MaskEncoding(index, embedding, label, score))
        return results
