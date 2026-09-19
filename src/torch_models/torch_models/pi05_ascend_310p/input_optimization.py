"""Validated input reductions for the fixed-shape Ascend310P runtime."""

from __future__ import annotations

import os

import numpy as np
import torch
from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor.device_processor import DeviceProcessorStep
from lerobot.processor.pipeline import ProcessorStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

LANGUAGE_TOKEN_LIMIT_ENV = "IBROBOT_PI05_LANGUAGE_TOKEN_LIMIT"
CPU_IMAGE_RESIZE_ENV = "IBROBOT_PI05_CPU_IMAGE_RESIZE"
CPU_OUTPUT_ENV = "IBROBOT_PI05_CPU_OUTPUT"


def language_token_limit(maximum: int) -> int | None:
    """Read the opt-in static capacity; never truncate a valid token silently."""

    value = os.environ.get(LANGUAGE_TOKEN_LIMIT_ENV, "").strip()
    if not value:
        return None
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError(f"{LANGUAGE_TOKEN_LIMIT_ENV} must be an integer") from exc
    if not 1 <= limit <= maximum:
        raise ValueError(f"{LANGUAGE_TOKEN_LIMIT_ENV} must be between 1 and {maximum}, got {limit}")
    return limit


def trim_language_padding(tokens: torch.Tensor, mask: torch.Tensor, limit: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove only masked tail positions, with one stable compiled input shape."""

    if tokens.ndim != 2 or mask.ndim != 2 or tokens.shape != mask.shape:
        raise ValueError("PI05 language tokens and mask must have the same [batch, length] shape")
    if mask.dtype != torch.bool:
        raise ValueError("PI05 language attention mask must be boolean")
    if tokens.shape[1] < limit:
        raise ValueError(f"PI05 language input length {tokens.shape[1]} is shorter than static capacity {limit}")
    if tokens.shape[1] == limit:
        return tokens, mask
    # This executes outside the compiled graphs. Inspect the mask, not the
    # token IDs: valid token ID zero must not be mistaken for padding.
    if bool(mask[:, limit:].detach().to(device="cpu").any()):
        raise ValueError(
            f"PI05 language input exceeds the configured {limit}-token capacity; "
            f"increase {LANGUAGE_TOKEN_LIMIT_ENV} and prepare a separate graph cache"
        )
    return tokens[:, :limit].contiguous(), mask[:, :limit].contiguous()


class LanguageCapacityStep(ProcessorStep):
    """Validate and trim language padding on CPU before the device transfer."""

    def __init__(self, limit: int):
        self.limit = limit

    def __call__(self, transition):
        observation = dict(transition[TransitionKey.OBSERVATION])
        tokens, mask = trim_language_padding(
            observation[OBS_LANGUAGE_TOKENS], observation[OBS_LANGUAGE_ATTENTION_MASK], self.limit
        )
        observation[OBS_LANGUAGE_TOKENS] = tokens
        observation[OBS_LANGUAGE_ATTENTION_MASK] = mask
        return {**transition, TransitionKey.OBSERVATION: observation}

    def transform_features(self, features):
        observations = dict(features[PipelineFeatureType.OBSERVATION])
        for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
            observations[key] = PolicyFeature(type=FeatureType.LANGUAGE, shape=(self.limit,))
        return {**features, PipelineFeatureType.OBSERVATION: observations}


def configure_input_processors(*, preprocessor, postprocessor, config):
    if getattr(config, "ascend_cpu_output", False):
        # The policy returns CPU actions in this mode. Keep LeRobot's original
        # unnormalization math on the host and avoid uploading the result again.
        postprocessor.steps = [
            DeviceProcessorStep(device="cpu", float_dtype=step.float_dtype)
            if isinstance(step, DeviceProcessorStep)
            else step
            for step in postprocessor.steps
        ]
    limit = getattr(config, "ascend_language_token_limit", None)
    if limit is None:
        return
    steps = list(preprocessor.steps)
    indices = [i for i, step in enumerate(steps) if isinstance(step, DeviceProcessorStep)]
    if len(indices) != 1:
        raise ValueError("PI05 language compaction requires exactly one device processor")
    steps.insert(indices[0], LanguageCapacityStep(limit))
    preprocessor.steps = steps


def resize_host_images(inputs, *, config):
    """Optional CPU bilinear/pad step; its cost stays inside preprocessing."""

    if not getattr(config, "ascend_cpu_image_resize", False):
        return inputs
    import cv2

    prepared = dict(inputs)
    target_h, target_w = config.image_resolution
    for name in config.image_features:
        if name not in prepared:
            continue
        value = prepared[name]
        if torch.is_tensor(value):
            if value.device.type != "cpu":
                continue
            value = value.detach().numpy()
        if not isinstance(value, np.ndarray) or value.dtype != np.float32:
            raise ValueError("PI05 CPU resize requires float32 host images normalized to [0, 1]")
        channels_axis = 0 if value.ndim == 3 else 1
        if value.ndim in {3, 4} and value.shape[channels_axis] == 3 and value.flags.c_contiguous:
            # ROS already supplies packed CHW images. CPU resizing regresses
            # that path under callback contention; retain the NPU resize.
            continue
        unbatched = value.ndim == 3
        images = value[None] if unbatched else value
        if images.ndim != 4:
            raise ValueError("PI05 CPU resize requires CHW, HWC, NCHW or NHWC images")
        packed = []
        for image in images:
            channels_first = image.shape[0] == 3
            hwc = image.transpose(1, 2, 0) if channels_first else image
            if hwc.shape[-1] != 3:
                raise ValueError("PI05 CPU resize requires three image channels")
            height, width = hwc.shape[:2]
            if (height, width) == (target_h, target_w):
                resized = hwc
            else:
                ratio = max(width / target_w, height / target_h)
                resized_h, resized_w = int(height / ratio), int(width / ratio)
                top, left = (target_h - resized_h) // 2, (target_w - resized_w) // 2
                small = cv2.resize(hwc, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
                small = np.clip(small, 0.0, 1.0)
                resized = np.zeros((target_h, target_w, 3), dtype=np.float32)
                resized[top : top + resized_h, left : left + resized_w] = small
            packed.append(resized.transpose(2, 0, 1) if channels_first else resized)
        result = np.ascontiguousarray(np.stack(packed))
        if unbatched:
            result = result[0]
        # AddBatchDimensionProcessor handles Torch tensors, not numpy arrays.
        prepared[name] = torch.from_numpy(result) if torch.is_tensor(prepared[name]) else result
    return prepared
