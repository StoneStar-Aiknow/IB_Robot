"""RGB-D utilities for turning 2D boxes into camera-frame 3D poses."""

from __future__ import annotations

from typing import Any

import numpy as np
from geometry_msgs.msg import PoseStamped


def _intrinsics_from_camera_info(camera_info: dict[str, Any]) -> tuple[float, float, float, float] | None:
    k = camera_info.get("k") or camera_info.get("K")
    if not isinstance(k, list | tuple) or len(k) < 6:
        return None
    fx = float(k[0])
    fy = float(k[4])
    cx = float(k[2])
    cy = float(k[5])
    if fx <= 0.0 or fy <= 0.0:
        return None
    return fx, fy, cx, cy


def bbox_to_3d_pose(
    bbox_xyxy: tuple[int, int, int, int],
    depth_meters: np.ndarray,
    camera_info: dict[str, Any],
    frame_id: str,
    depth_percentile: float = 50.0,
    max_depth_m: float = 5.0,
) -> PoseStamped | None:
    """Estimate object center pose in the camera frame from a bbox and depth image."""

    intrinsics = _intrinsics_from_camera_info(camera_info)
    if intrinsics is None:
        return None
    fx, fy, cx, cy = intrinsics

    depth = np.asarray(depth_meters, dtype=np.float32)
    if depth.ndim != 2 or depth.size == 0:
        return None
    height, width = depth.shape
    x1, y1, x2, y2 = bbox_xyxy
    x1 = max(0, min(int(x1), width - 1))
    x2 = max(0, min(int(x2), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    y2 = max(0, min(int(y2), height - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = depth[y1 : y2 + 1, x1 : x2 + 1]
    valid = roi[np.isfinite(roi) & (roi > 0.0) & (roi <= float(max_depth_m))]
    if valid.size == 0:
        return None

    z = float(np.percentile(valid, depth_percentile))
    u = float((x1 + x2) / 2.0)
    v = float((y1 + y2) / 2.0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.pose.position.x = float(x)
    pose.pose.position.y = float(y)
    pose.pose.position.z = float(z)
    pose.pose.orientation.w = 1.0
    return pose
