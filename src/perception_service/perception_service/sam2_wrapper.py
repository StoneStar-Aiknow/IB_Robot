"""CUDA/CPU SAM2 automatic and box-prompted segmentation wrapper."""

import threading
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

from .model_utils import DEFAULT_MODEL_DIR, inspect_backend, resolve_model_path


@dataclass(frozen=True)
class SegmentationMask:
    mask: np.ndarray
    bbox_xyxy: np.ndarray
    score: float
    stability_score: float
    area: int


class SAM2Wrapper:
    """Own one stateful SAM2 model and serialize access to its image cache."""

    def __init__(
        self,
        *,
        backend: str = "cuda",
        checkpoint: str = "sam2.1_hiera_tiny/assets/sam2.1_hiera_tiny.pt",
        config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        model_dir: str | None = None,
        points_per_batch: int = 64,
        automatic_generator=None,
        image_predictor=None,
    ):
        if backend == "ascend_om":
            raise RuntimeError("Ascend OM requires a manifest named deployment")
        status = inspect_backend(backend)
        if not status.ready:
            raise RuntimeError(status.message)

        self.backend = backend
        self.runtime_version = status.runtime_version
        self.checkpoint_path = resolve_model_path(checkpoint, model_dir or DEFAULT_MODEL_DIR)
        self.config = config
        self._lock = threading.Lock()

        if automatic_generator is not None and image_predictor is not None:
            self._automatic_generator = automatic_generator
            self._image_predictor = image_predictor
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint_path}")

        try:
            import torch
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("SAM2 requires torch and the sam2 Python package") from exc

        self._torch = torch
        model = build_sam2(config, str(self.checkpoint_path), device=backend, apply_postprocessing=False)
        self._automatic_generator = SAM2AutomaticMaskGenerator(
            model,
            points_per_batch=points_per_batch,
            output_mode="binary_mask",
        )
        self._image_predictor = SAM2ImagePredictor(model)

    def _inference_context(self):
        torch = getattr(self, "_torch", None)
        if torch is None:
            return nullcontext()
        return torch.inference_mode()

    @staticmethod
    def _validate_image(image_rgb: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image must be an RGB uint8 HxWx3 array")

    def generate(self, image_rgb: np.ndarray) -> list[SegmentationMask]:
        self._validate_image(image_rgb)
        with self._lock, self._inference_context():
            records = self._automatic_generator.generate(image_rgb)

        results = []
        for record in records:
            mask = np.asarray(record["segmentation"], dtype=np.uint8)
            if mask.shape != image_rgb.shape[:2]:
                raise RuntimeError("SAM2 returned a mask with unexpected dimensions")
            x, y, width, height = (float(value) for value in record["bbox"])
            results.append(
                SegmentationMask(
                    mask=mask,
                    bbox_xyxy=np.asarray([x, y, x + width, y + height], dtype=np.float32),
                    score=float(record["predicted_iou"]),
                    stability_score=float(record["stability_score"]),
                    area=int(record["area"]),
                )
            )
        return results

    def segment_boxes(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> list[SegmentationMask]:
        self._validate_image(image_rgb)
        boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        if not len(boxes):
            return []
        with self._lock, self._inference_context():
            self._image_predictor.set_image(image_rgb)
            masks, scores, _ = self._image_predictor.predict(box=boxes, multimask_output=False)

        masks = np.asarray(masks)
        scores = np.asarray(scores)
        if masks.ndim == 4:
            masks = masks[:, 0]
        if scores.ndim == 2:
            scores = scores[:, 0]
        if masks.shape != (len(boxes), *image_rgb.shape[:2]):
            raise RuntimeError("SAM2 returned an unexpected box-mask shape")
        return [
            SegmentationMask(
                mask=(masks[index] > 0).astype(np.uint8),
                bbox_xyxy=box.copy(),
                score=float(scores[index]),
                stability_score=0.0,
                area=int(np.count_nonzero(masks[index])),
            )
            for index, box in enumerate(boxes)
        ]
