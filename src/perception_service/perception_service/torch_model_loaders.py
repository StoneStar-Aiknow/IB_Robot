"""Bundle-local Torch callables for generic perception model sessions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _configuration(context) -> tuple[Path, dict]:
    root = context.validated_manifest.bundle_root
    path = root / "assets" / "adapter.json"
    return root, json.loads(path.read_text(encoding="utf-8"))


def _numpy(value, dtype=None) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    return np.asarray(value, dtype=dtype)


class _SAM2Module:
    def __init__(self, wrapper) -> None:
        self.wrapper = wrapper

    def __call__(self, inputs):
        image = np.ascontiguousarray(_numpy(inputs["observation.image"], np.uint8))
        records = self.wrapper.generate(image)
        height, width = image.shape[:2]
        return {
            "masks": np.stack([record.mask for record in records]).astype(np.uint8, copy=False)
            if records
            else np.empty((0, height, width), dtype=np.uint8),
            "boxes": np.stack([record.bbox_xyxy for record in records]).astype(np.float32, copy=False)
            if records
            else np.empty((0, 4), dtype=np.float32),
            "scores": np.asarray([record.score for record in records], dtype=np.float32),
            "stability_scores": np.asarray([record.stability_score for record in records], dtype=np.float32),
        }


def load_sam2(context):
    from .sam2_wrapper import SAM2Wrapper

    root, config = _configuration(context)
    wrapper = SAM2Wrapper(
        backend=context.deployment.device,
        checkpoint=str(root / config["checkpoint"]),
        config=config["config"],
        points_per_batch=int(config.get("points_per_batch", 64)),
        points_per_side=int(config.get("points_per_side", 32)),
        pred_iou_thresh=float(config.get("pred_iou_thresh", 0.8)),
        stability_score_thresh=float(config.get("stability_score_thresh", 0.95)),
    )
    return _SAM2Module(wrapper)


class _RAMPlusModule:
    def __init__(self, model, torch_module) -> None:
        self.model = model
        self.torch = torch_module

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, inputs):
        image = inputs["observation.image"]
        model = self.model
        image_embeds = model.image_proj(model.visual_encoder(image))
        image_atts = self.torch.ones(image_embeds.size()[:-1], dtype=self.torch.long, device=image.device)
        image_cls = image_embeds[:, 0]
        image_cls = image_cls / image_cls.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        descriptions_per_class = model.label_embed.shape[0] // model.num_class
        logits = model.reweight_scale.exp() * image_cls @ model.label_embed.t()
        weights = self.torch.nn.functional.softmax(logits.view(-1, model.num_class, descriptions_per_class), dim=2)
        descriptions = model.label_embed.view(model.num_class, descriptions_per_class, -1)
        labels = self.torch.nn.functional.relu(model.wordvec_proj((weights.unsqueeze(-1) * descriptions).sum(dim=2)))
        tagging = model.tagging_head(
            encoder_embeds=labels,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=False,
            mode="tagging",
        )
        return {"tag_logits": model.fc(tagging[0]).squeeze(-1).to(self.torch.float32)}


def load_ram_plus(context):
    root, config = _configuration(context)
    import torch
    from ram.models import ram_plus

    model = ram_plus(
        pretrained=str(root / config["checkpoint"]),
        image_size=384,
        vit="swin_l",
        text_encoder_type=str(root / config["text_encoder"]),
    )
    return _RAMPlusModule(model, torch)


class _SigLIP2Module:
    def __init__(self, model, torch_module) -> None:
        self.model = model
        self.torch = torch_module

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    @staticmethod
    def _pooled_features(value):
        pooled = getattr(value, "pooler_output", None)
        return pooled if pooled is not None else value

    def __call__(self, inputs):
        images = inputs["masked_images"]
        tokens = inputs["text_tokens"]
        attention = inputs["text_attention_mask"]
        text_config = self.model.config.text_config
        dimension = int(getattr(text_config, "projection_size", text_config.hidden_size))
        image_features = (
            self._pooled_features(self.model.get_image_features(pixel_values=images))
            if len(images)
            else self.torch.empty((0, dimension), dtype=self.torch.float32, device=images.device)
        )
        text_features = (
            self._pooled_features(self.model.get_text_features(input_ids=tokens, attention_mask=attention))
            if len(tokens)
            else self.torch.empty((0, dimension), dtype=self.torch.float32, device=tokens.device)
        )
        return {
            "image_embeddings": image_features.to(self.torch.float32),
            "text_embeddings": text_features.to(self.torch.float32),
        }


def load_siglip2(context):
    root, config = _configuration(context)
    import torch
    from transformers import AutoModel

    model = AutoModel.from_pretrained(root / config["model_path"], local_files_only=True)
    return _SigLIP2Module(model, torch)


class _GroundedSAM2Module:
    def __init__(self, wrapper) -> None:
        self.wrapper = wrapper

    def __call__(self, inputs):
        image = np.ascontiguousarray(_numpy(inputs["observation.image"], np.uint8))
        prompt = _numpy(inputs["text_prompt"], np.uint8).tobytes().decode("utf-8")
        box_threshold = float(_numpy(inputs["box_threshold"], np.float32)[0])
        text_threshold = float(_numpy(inputs["text_threshold"], np.float32)[0])
        records = self.wrapper.detect_and_segment(
            np.ascontiguousarray(image[:, :, ::-1]),
            prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        height, width = image.shape[:2]
        return {
            "boxes": np.stack([record.bbox_xyxy for record in records]).astype(np.float32, copy=False)
            if records
            else np.empty((0, 4), dtype=np.float32),
            "scores": np.asarray([record.confidence for record in records], dtype=np.float32),
            "masks": np.stack([record.mask for record in records]).astype(np.uint8, copy=False)
            if records
            else np.empty((0, height, width), dtype=np.uint8),
            "label_indices": np.zeros(len(records), dtype=np.int32),
        }


def load_grounded_sam2(context):
    from .grounded_sam2_wrapper import GroundedSAM2Wrapper

    root, config = _configuration(context)
    wrapper = GroundedSAM2Wrapper(
        device=context.deployment.device,
        sam_checkpoint=str(root / config["sam_checkpoint"]),
        sam_config=config["sam_config"],
        gdino_checkpoint=str(root / config["gdino_checkpoint"]),
        gdino_text_encoder=str(root / config["text_encoder"]),
    )
    return _GroundedSAM2Module(wrapper)


class _GraspGenModule:
    def __init__(self, sampler, config, torch_module) -> None:
        self.sampler = sampler
        self.config = config
        self.torch = torch_module

    def __call__(self, inputs):
        from grasp_gen.grasp_server import GraspGenSampler

        # The shared adapter has already centred and kappa-scaled the cloud. The upstream
        # CUDA sampler consumes centred metric points and owns its own internal scaling.
        points = inputs["observation.object_points"]
        detach = getattr(points, "detach", None)
        if callable(detach):
            points = detach()
        cpu = getattr(points, "cpu", None)
        if callable(cpu):
            points = cpu()
        numpy = getattr(points, "numpy", None)
        if callable(numpy):
            points = numpy()
        points = np.ascontiguousarray(np.asarray(points, dtype=np.float32) / float(self.config.kappa))
        poses, confidence = GraspGenSampler.run_inference(
            points,
            self.sampler,
            grasp_threshold=-1.0,
            num_grasps=int(self.config.grasp_batch_size),
            topk_num_grasps=int(self.config.grasp_batch_size),
            min_grasps=1,
            max_tries=1,
            remove_outliers=False,
        )
        if len(poses) == 0:
            poses = self.torch.empty((0, 4, 4), dtype=self.torch.float32)
            confidence = self.torch.empty((0,), dtype=self.torch.float32)
        return {
            "grasp.poses": poses.to(self.torch.float32),
            "grasp.confidence": confidence.to(self.torch.float32),
        }


def load_graspgen(context):
    from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

    from .graspgen_adapter import GraspGenConfig

    root, assets = _configuration(context)
    config = load_grasp_cfg(str(root / assets["gripper_config"]))
    config.eval.checkpoint = str(root / assets["generator_checkpoint"])
    config.discriminator.checkpoint = str(root / assets["discriminator_checkpoint"])
    return _GraspGenModule(GraspGenSampler(config), GraspGenConfig.from_assets(assets), __import__("torch"))


__all__ = ["load_graspgen", "load_grounded_sam2", "load_ram_plus", "load_sam2", "load_siglip2"]
