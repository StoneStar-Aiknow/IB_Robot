"""Dependency-light RGB-D localization primitives."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthMeasurement:
    depth_m: float
    valid_ratio: float
    mad_m: float
    sample_count: int


def robust_box_depth(
    depth: np.ndarray,
    bbox: tuple[float, float, float, float],
    *,
    depth_scale: float = 0.001,
    central_fraction: float = 0.6,
    min_depth_m: float = 0.15,
    max_depth_m: float = 8.0,
    min_valid_ratio: float = 0.2,
    mad_scale: float = 3.5,
) -> DepthMeasurement | None:
    """Return a median/MAD-filtered depth from the central bounding-box region."""
    if depth.ndim != 2:
        raise ValueError("depth image must be two-dimensional")
    if not 0.0 < central_fraction <= 1.0:
        raise ValueError("central_fraction must be in (0, 1]")
    x_min, y_min, x_max, y_max = bbox
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bounding box must have positive area")

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    half_width = (x_max - x_min) * central_fraction / 2.0
    half_height = (y_max - y_min) * central_fraction / 2.0
    left = max(0, int(np.floor(center_x - half_width)))
    right = min(depth.shape[1], int(np.ceil(center_x + half_width)))
    top = max(0, int(np.floor(center_y - half_height)))
    bottom = min(depth.shape[0], int(np.ceil(center_y + half_height)))
    if left >= right or top >= bottom:
        return None

    samples = np.asarray(depth[top:bottom, left:right], dtype=np.float64).reshape(-1) * depth_scale
    valid = np.isfinite(samples) & (samples >= min_depth_m) & (samples <= max_depth_m)
    valid_ratio = float(np.count_nonzero(valid) / samples.size)
    if valid_ratio < min_valid_ratio:
        return None
    samples = samples[valid]
    median = float(np.median(samples))
    deviations = np.abs(samples - median)
    mad = float(np.median(deviations))
    if mad == 0.0:
        samples = samples[deviations == 0.0]
    else:
        samples = samples[deviations <= mad_scale * mad]
    if samples.size == 0:
        return None
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    return DepthMeasurement(median, valid_ratio, mad, int(samples.size))


def back_project_pixel(u: float, v: float, depth_m: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Back-project one color pixel into its optical camera frame."""
    if depth_m <= 0.0 or not np.isfinite(depth_m):
        raise ValueError("depth_m must be finite and positive")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")
    return np.array([(u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m], dtype=np.float64)
