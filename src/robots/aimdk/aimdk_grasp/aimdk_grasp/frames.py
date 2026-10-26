"""Rigid transforms tied to an image time. T_A_B maps coordinates from B to A.

Camera poses come from TF at the image time; no latest-TF fallback is allowed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def rigid(matrix) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError("Expected finite 4x4 transform")
    r = value[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
        raise ValueError("Transform rotation is not proper")
    if not np.allclose(value[3], [0, 0, 0, 1]):
        raise ValueError("Invalid homogeneous row")
    return value.copy()


@dataclass(frozen=True)
class ImageTransform:
    stamp_ns: int
    camera_frame: str
    base_frame: str
    base_from_camera: np.ndarray
    calibration_verified: bool = False
