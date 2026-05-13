"""Utilities for summarizing RGB-D context into model-friendly structures."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sensor_msgs.msg import CameraInfo, PointCloud2


def summarize_camera_info(msg: CameraInfo | None) -> dict[str, Any]:
    if msg is None:
        return {"available": False}
    k_values = list(msg.k)
    if len(k_values) < 9:
        warnings.warn(
            f"CameraInfo K matrix has {len(k_values)} elements (expected 9); padding with zeros",
            stacklevel=2,
        )
        k = [0.0] * 9
    else:
        k = k_values
    return {
        "available": True,
        "frame_id": msg.header.frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "fx": float(k[0]),
        "fy": float(k[4]),
        "cx": float(k[2]),
        "cy": float(k[5]),
    }


def summarize_pointcloud_metadata(msg: PointCloud2 | None) -> dict[str, Any]:
    if msg is None:
        return {"available": False}
    point_count_estimate = int(msg.width) * int(msg.height)
    return {
        "available": True,
        "frame_id": msg.header.frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "is_dense": bool(msg.is_dense),
        "point_step": int(msg.point_step),
        "row_step": int(msg.row_step),
        "point_count_estimate": point_count_estimate,
    }


def summarize_depth_frame(
    depth_meters: np.ndarray,
    camera_info: CameraInfo | None = None,
    max_valid_depth_m: float = 3.0,
) -> dict[str, Any]:
    depth = np.asarray(depth_meters, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError("depth frame must be a 2D array")

    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= float(max_valid_depth_m))
    total_pixels = int(depth.size)
    valid_count = int(np.count_nonzero(valid))
    if total_pixels == 0 or valid_count == 0:
        return {
            "available": False,
            "frame_shape": [int(depth.shape[0]), int(depth.shape[1])],
            "camera_info": summarize_camera_info(camera_info),
            "notes": ["depth frame has no valid pixels"],
        }

    valid_values = depth[valid]
    h, w = depth.shape
    half_window = max(8, int(min(h, w) * 0.08))
    cy = h // 2
    cx = w // 2
    y0 = max(0, cy - half_window)
    y1 = min(h, cy + half_window)
    x0 = max(0, cx - half_window)
    x1 = min(w, cx + half_window)
    center_patch = depth[y0:y1, x0:x1]
    center_valid = center_patch[np.isfinite(center_patch) & (center_patch > 0.0) & (center_patch <= max_valid_depth_m)]

    notes = []
    valid_ratio = float(valid_count) / float(total_pixels)
    p10 = float(np.percentile(valid_values, 10))
    p90 = float(np.percentile(valid_values, 90))
    if valid_ratio < 0.35:
        notes.append("depth coverage is sparse")
    if p10 < 0.20:
        notes.append("near obstacles exist within 0.20m")
    if center_valid.size:
        center_median = float(np.median(center_valid))
        if center_median < 0.18:
            notes.append("center region is very close to the camera")
        elif center_median > 1.20:
            notes.append("center region is relatively far from the camera")
    else:
        center_median = None
        notes.append("center depth region is invalid or missing")

    return {
        "available": True,
        "frame_shape": [int(h), int(w)],
        "camera_info": summarize_camera_info(camera_info),
        "depth_valid_ratio": round(valid_ratio, 4),
        "scene_depth_range_m": [
            round(float(np.min(valid_values)), 4),
            round(float(np.max(valid_values)), 4),
        ],
        "median_depth_m": round(float(np.median(valid_values)), 4),
        "depth_percentile_10_m": round(p10, 4),
        "depth_percentile_90_m": round(p90, 4),
        "center_depth_m": round(center_median, 4) if center_median is not None else None,
        "notes": notes,
    }
