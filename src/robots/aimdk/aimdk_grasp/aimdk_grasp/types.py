"""Small, framework-independent types shared by the grasp pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for the RGB image coordinate system."""

    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class RGBDFrame:
    """One RGB-D sample. Depth must be aligned to RGB pixels."""

    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: CameraIntrinsics
    frame_id: str = "rgbd_head_front"
    depth_scale: float = 1.0
    timestamp: float | None = None
