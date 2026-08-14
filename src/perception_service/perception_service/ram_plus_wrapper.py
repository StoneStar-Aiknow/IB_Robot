"""RAM++ scene-tag recognition wrapper."""

from contextlib import nullcontext

import numpy as np

from .model_utils import inspect_backend, resolve_model_path
from .ram_plus_adapter import RecognizedTag


class RAMPlusWrapper:
    def __init__(
        self,
        *,
        backend: str = "cuda",
        checkpoint: str = "ram_plus_swin_large_14m/assets/ram_plus_swin_large_14m.pth",
        model_dir: str | None = None,
        text_encoder: str = "bert-base-uncased",
        model=None,
        transform=None,
        logits_inference=None,
    ):
        if backend == "ascend_om":
            raise RuntimeError("Ascend OM requires a manifest named deployment")
        status = inspect_backend(backend)
        if not status.ready:
            raise RuntimeError(status.message)

        self.backend = backend
        self.runtime_version = status.runtime_version
        self.checkpoint_path = resolve_model_path(checkpoint, model_dir)
        if model is not None and transform is not None and logits_inference is not None:
            self._model = model
            self._transform = transform
            self._infer_logits = logits_inference
            return
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"RAM++ checkpoint not found: {self.checkpoint_path}")

        try:
            import torch
            import torch.nn.functional as functional
            from ram import get_transform
            from ram.models import ram_plus
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("RAM++ requires torch and the recognize-anything package") from exc

        self._torch = torch
        self._functional = functional
        self._model = (
            ram_plus(
                pretrained=str(self.checkpoint_path),
                image_size=384,
                vit="swin_l",
                text_encoder_type=text_encoder,
            )
            .eval()
            .to(backend)
        )
        self._transform = get_transform(image_size=384)
        self._infer_logits = self._forward_logits

    def _forward_logits(self, image):
        model = self._model
        image_embeds = model.image_proj(model.visual_encoder(image))
        image_atts = self._torch.ones(image_embeds.size()[:-1], dtype=self._torch.long, device=image.device)
        image_cls = image_embeds[:, 0]
        image_cls = image_cls / image_cls.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        descriptions_per_class = model.label_embed.shape[0] // model.num_class
        logits = model.reweight_scale.exp() * image_cls @ model.label_embed.t()
        weights = self._functional.softmax(logits.view(-1, model.num_class, descriptions_per_class), dim=2)
        descriptions = model.label_embed.view(model.num_class, descriptions_per_class, -1)
        labels = (weights.unsqueeze(-1) * descriptions).sum(dim=2)
        labels = self._functional.relu(model.wordvec_proj(labels))
        tagging = model.tagging_head(
            encoder_embeds=labels,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode="tagging",
        )
        return model.fc(tagging[0]).squeeze(-1)

    def _inference_context(self):
        torch = getattr(self, "_torch", None)
        return nullcontext() if torch is None else torch.inference_mode()

    def recognize_batch(self, images_rgb: list[np.ndarray], score_threshold: float = 0.0) -> list[list[RecognizedTag]]:
        if not images_rgb:
            raise ValueError("at least one image is required")
        for image_rgb in images_rgb:
            if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
                raise ValueError("image must be an RGB uint8 HxWx3 array")
        from PIL import Image

        tensors = [self._transform(Image.fromarray(image_rgb)) for image_rgb in images_rgb]
        if hasattr(tensors[0], "unsqueeze"):
            tensor = self._torch.stack(tensors).to(self.backend)
        else:
            tensor = np.stack(tensors)
        with self._inference_context():
            logits = self._infer_logits(tensor)
        if hasattr(logits, "detach"):
            scores = logits.detach().float().cpu().numpy()
        else:
            scores = np.asarray(logits, dtype=np.float32)
        scores = 1.0 / (1.0 + np.exp(-scores))
        if scores.ndim != 2 or scores.shape[0] != len(images_rgb):
            raise RuntimeError("RAM++ batch logits have an unexpected shape")

        model_thresholds = getattr(self._model, "class_threshold", np.zeros(scores.shape[1], dtype=np.float32))
        if hasattr(model_thresholds, "detach"):
            model_thresholds = model_thresholds.detach().float().cpu().numpy()
        thresholds = np.asarray(model_thresholds, dtype=np.float32).reshape(1, -1)
        if score_threshold > 0.0:
            thresholds = np.full_like(thresholds, score_threshold)
        labels = np.asarray(self._model.tag_list).reshape(-1)
        if scores.shape[1] != len(labels) or thresholds.shape[1] != len(labels):
            raise RuntimeError("RAM++ batch logits and tag vocabulary have different lengths")
        deleted = set(int(index) for index in getattr(self._model, "delete_tag_index", []))
        output = []
        for row in scores:
            indices = [int(index) for index in np.flatnonzero(row > thresholds[0]) if int(index) not in deleted]
            indices.sort(key=lambda index: (-float(row[index]), str(labels[index])))
            output.append([RecognizedTag(str(labels[index]), float(row[index])) for index in indices])
        return output

    def recognize(self, image_rgb: np.ndarray, score_threshold: float = 0.0) -> list[RecognizedTag]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or image_rgb.dtype != np.uint8:
            raise ValueError("image must be an RGB uint8 HxWx3 array")
        return self.recognize_batch([image_rgb], score_threshold)[0]
