"""Candidate target-width estimation (ROS-free)."""

import numpy as np


def estimate_candidate_target_width(
    object_points_camera: np.ndarray,
    grasp_pose_camera: np.ndarray,
    width_axis_local: np.ndarray,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    min_width_m: float = 0.005,
    max_width_m: float = 0.14,
    min_points: int = 20,
) -> tuple[float, float, tuple[float, float, float], float, float]:
    pts = np.asarray(object_points_camera, dtype=np.float64)
    axis_local = np.asarray(width_axis_local, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis_local))
    if len(pts) < min_points or axis_norm <= 1e-9:
        return 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0

    low = float(percentile_low)
    high = float(percentile_high)
    if not (0.0 <= low < high <= 100.0):
        low, high = 5.0, 95.0

    axis_local = axis_local / axis_norm
    rotation = np.asarray(grasp_pose_camera[:3, :3], dtype=np.float64)
    axis_camera = rotation @ axis_local
    axis_camera_norm = float(np.linalg.norm(axis_camera))
    if axis_camera_norm <= 1e-9:
        return 0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0
    axis_camera = axis_camera / axis_camera_norm

    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) < min_points:
        axis = (float(axis_camera[0]), float(axis_camera[1]), float(axis_camera[2]))
        return 0.0, 0.0, axis, 0.0, 0.0

    projections = finite @ axis_camera
    p_low, p_high = np.percentile(projections, [low, high])
    origin_projection = float(np.dot(np.asarray(grasp_pose_camera[:3, 3], dtype=np.float64), axis_camera))
    min_offset = float(p_low - origin_projection)
    max_offset = float(p_high - origin_projection)
    raw_width = max(0.0, float(p_high - p_low))
    width = min(max(raw_width, float(min_width_m)), float(max_width_m))
    quality = 1.0 if abs(width - raw_width) <= 1e-6 else 0.5
    if raw_width <= 0.0:
        quality = 0.0
    axis = (float(axis_camera[0]), float(axis_camera[1]), float(axis_camera[2]))
    return width, quality, axis, min_offset, max_offset


def estimate_candidate_target_widths(
    object_points_camera: np.ndarray,
    grasp_poses_camera: np.ndarray,
    width_axis_local: np.ndarray,
    percentile_low: float = 5.0,
    percentile_high: float = 95.0,
    min_width_m: float = 0.005,
    max_width_m: float = 0.14,
    min_points: int = 20,
    chunk_size: int = 100,
) -> list[tuple[float, float, tuple[float, float, float], float, float]]:
    """Estimate candidate widths in bounded projection batches."""
    poses = np.asarray(grasp_poses_camera)
    if len(poses) == 0:
        return []

    pts = np.asarray(object_points_camera, dtype=np.float64)
    axis_local = np.asarray(width_axis_local, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis_local))
    if len(pts) < min_points or axis_norm <= 1e-9:
        return [(0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0) for _ in poses]

    low = float(percentile_low)
    high = float(percentile_high)
    if not (0.0 <= low < high <= 100.0):
        low, high = 5.0, 95.0

    axis_local = axis_local / axis_norm
    rotations = np.asarray(poses[:, :3, :3], dtype=np.float64)
    axes_camera = rotations @ axis_local
    axis_norms = np.linalg.norm(axes_camera, axis=1)
    valid_axes = axis_norms > 1e-9
    axes_camera[valid_axes] /= axis_norms[valid_axes, None]

    finite = pts[np.isfinite(pts).all(axis=1)]
    results: list[tuple[float, float, tuple[float, float, float], float, float]] = [
        (0.0, 0.0, (0.0, 0.0, 0.0), 0.0, 0.0) for _ in poses
    ]
    if len(finite) < min_points:
        for index in np.flatnonzero(valid_axes):
            axis = axes_camera[index]
            results[index] = (0.0, 0.0, (float(axis[0]), float(axis[1]), float(axis[2])), 0.0, 0.0)
        return results

    chunk_size = max(1, int(chunk_size))
    valid_indices = np.flatnonzero(valid_axes)
    for start in range(0, len(valid_indices), chunk_size):
        indices = valid_indices[start : start + chunk_size]
        axes = axes_camera[indices]
        projections = finite @ axes.T
        p_low, p_high = np.percentile(projections, [low, high], axis=0)
        origins = np.asarray(poses[indices, :3, 3], dtype=np.float64)
        origin_projections = np.einsum("ij,ij->i", origins, axes)

        raw_widths = np.maximum(0.0, p_high - p_low)
        widths = np.clip(raw_widths, float(min_width_m), float(max_width_m))
        qualities = np.where(np.abs(widths - raw_widths) <= 1e-6, 1.0, 0.5)
        qualities[raw_widths <= 0.0] = 0.0

        for offset, index in enumerate(indices):
            axis = axes[offset]
            results[index] = (
                float(widths[offset]),
                float(qualities[offset]),
                (float(axis[0]), float(axis[1]), float(axis[2])),
                float(p_low[offset] - origin_projections[offset]),
                float(p_high[offset] - origin_projections[offset]),
            )
    return results
