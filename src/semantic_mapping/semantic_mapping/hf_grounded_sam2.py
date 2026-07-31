"""Hugging Face Grounding DINO + SAM2 inference backend."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    label: str
    confidence: float
    bbox_xyxy: np.ndarray
    mask: np.ndarray


class HFGroundedSAM2:
    """Run local Hugging Face Grounding DINO weights with a SAM2 predictor."""

    def __init__(self, grounding_model_path: str, sam_checkpoint: str, sam_config: str, device: str = "cuda"):
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "The Hugging Face Grounded-SAM2 backend requires torch, transformers, and sam2."
            ) from exc

        grounding_path = Path(grounding_model_path).expanduser().resolve()
        sam_checkpoint_path = Path(sam_checkpoint).expanduser().resolve()
        if not grounding_path.exists():
            raise FileNotFoundError(f"Grounding DINO model not found: {grounding_path}")
        if not sam_checkpoint_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {sam_checkpoint_path}")

        self._torch = torch
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        sam_model = build_sam2(sam_config, str(sam_checkpoint_path), device=self.device)
        self.sam_predictor = SAM2ImagePredictor(sam_model)
        self.processor = AutoProcessor.from_pretrained(grounding_path, local_files_only=True)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            grounding_path, local_files_only=True
        ).to(self.device)
        self.grounding_model.eval()

    def detect_and_segment(
        self,
        image_bgr: np.ndarray,
        text_prompt: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
    ) -> list[Detection]:
        from PIL import Image

        prompt = text_prompt.lower().strip()
        if not prompt.endswith("."):
            prompt += "."
        image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        image = Image.fromarray(image_rgb)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            outputs = self.grounding_model(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        boxes = result["boxes"].detach().cpu().numpy()
        if boxes.shape[0] == 0:
            return []

        self.sam_predictor.set_image(image_rgb)
        masks, _, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=boxes,
            multimask_output=False,
        )
        if masks.ndim == 4:
            masks = masks[:, 0]
        return [
            Detection(
                label=str(label),
                confidence=float(score),
                bbox_xyxy=box.astype(np.float32),
                mask=mask.astype(np.uint8),
            )
            for label, score, box, mask in zip(result["labels"], result["scores"], boxes, masks, strict=True)
        ]
