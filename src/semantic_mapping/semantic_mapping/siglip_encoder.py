"""Optional SigLIP image encoder used for cross-view object association."""

import os
from pathlib import Path

import numpy as np


class SigLIPEncoder:
    def __init__(self, model_path: str, device: str = "cuda"):
        if not model_path:
            raise ValueError("siglip_model_path is required when SigLIP matching is enabled")
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SigLIP matching requires torch and transformers. Install perception dependencies first."
            ) from exc

        path = Path(model_path).expanduser()
        if not path.is_absolute():
            path = Path(os.environ.get("IB_ROBOT_WORKSPACE", ".")).resolve() / path
        if not path.exists():
            raise FileNotFoundError(f"SigLIP model not found: {path}")
        self._torch = torch
        self.device = device if device != "cuda" or torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(self.device).eval()

    def encode(self, image_bgr: np.ndarray, mask: np.ndarray, bbox_xyxy: np.ndarray) -> np.ndarray | None:
        """Encode a masked object crop into a normalized SigLIP feature vector."""
        from PIL import Image

        height, width = image_bgr.shape[:2]
        x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.int32)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = image_bgr[y1:y2, x1:x2, ::-1].copy()
        crop_mask = mask[y1:y2, x1:x2] > 0
        if crop_mask.shape != crop.shape[:2] or not crop_mask.any():
            return None
        crop[~crop_mask] = 127
        inputs = self.processor(images=Image.fromarray(crop), return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            if hasattr(self.model, "vision_model"):
                outputs = self.model.vision_model(pixel_values=inputs["pixel_values"], return_dict=True)
                features = outputs.pooler_output
            elif hasattr(self.model, "get_image_features"):
                features = self.model.get_image_features(**inputs)
                if not self._torch.is_tensor(features):
                    features = getattr(features, "pooler_output", None)
            else:
                outputs = self.model(**inputs)
                features = getattr(outputs, "image_embeds", None)
            if features is None or features.ndim != 2:
                raise RuntimeError("configured model does not expose pooled SigLIP image embeddings")
        vector = features[0].detach().float().cpu().numpy()
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 1e-12 else None
