# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""Lightweight PI05 image preprocessing helpers.

The resize/pad math follows LeRobot/OpenPI PI05's ``resize_with_pad_torch``
implementation.  Keep this small local copy instead of importing LeRobot's
policy/model module from the OM runtime: that import path pulls in transformers,
policy registration, and export-only model definitions, while inference only
needs the deterministic image resize contract shared with export.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def resize_with_pad_torch(
    images: torch.Tensor,
    height: int,
    width: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize image tensor to ``height`` x ``width`` without distortion.

    Supports NHWC or NCHW input. Float images are expected in ``[0, 1]`` and
    uint8 images in ``[0, 255]``; padding is black, matching LeRobot/OpenPI PI05.
    """
    channels_last = images.shape[-1] <= 4
    if channels_last:
        if images.dim() == 3:
            images = images.unsqueeze(0)
        images = images.permute(0, 3, 1, 2)
    elif images.dim() == 3:
        images = images.unsqueeze(0)

    _, _, cur_height, cur_width = images.shape
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_images = F.interpolate(
        images,
        size=(resized_height, resized_width),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )

    if images.dtype == torch.uint8:
        resized_images = torch.round(resized_images).clamp(0, 255).to(torch.uint8)
        constant_value = 0
    elif images.dtype == torch.float32:
        resized_images = resized_images.clamp(0.0, 1.0)
        constant_value = 0.0
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_images = F.pad(
        resized_images,
        (pad_w0, pad_w1, pad_h0, pad_h1),
        mode="constant",
        value=constant_value,
    )

    if channels_last:
        padded_images = padded_images.permute(0, 2, 3, 1)

    return padded_images


def resize_with_pad_nchw_numpy(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a NCHW float image batch and return contiguous ``float32`` NCHW."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"PI05 VLM image must be NCHW with 3 channels, got shape={image.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32))
    resized = resize_with_pad_torch(tensor, height, width)
    return np.ascontiguousarray(resized.numpy(), dtype=np.float32)
