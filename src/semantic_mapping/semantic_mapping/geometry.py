"""RGB-D projection and rigid-transform helpers."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ObjectGeometry:
    points: np.ndarray
    centroid: np.ndarray
    size: np.ndarray


def is_ground_object(
    geometry: ObjectGeometry,
    ground_height: float | None,
    *,
    max_bottom_clearance_m: float,
    max_object_height_m: float,
    max_footprint_m: float,
    reference_position_xy: np.ndarray | None = None,
    max_horizontal_distance_m: float | None = None,
) -> bool:
    """Return whether a world-frame geometry is a bounded object supported by the floor."""
    if ground_height is None:
        return False
    lower, upper = np.percentile(geometry.points, [2.0, 98.0], axis=0)
    supported = bool(
        lower[2] <= ground_height + max_bottom_clearance_m
        and upper[2] <= ground_height + max_object_height_m
        and max(upper[0] - lower[0], upper[1] - lower[1]) <= max_footprint_m
    )
    if not supported or max_horizontal_distance_m is None:
        return supported
    if reference_position_xy is None or np.asarray(reference_position_xy).shape != (2,):
        raise ValueError("reference_position_xy must contain the reference frame x/y position")
    return bool(np.linalg.norm(geometry.centroid[:2] - reference_position_xy) <= max_horizontal_distance_m)


def select_geometry_mask_indices(
    detections,
    mask_to_array,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    depth_scale: float,
    depth_trunc_m: float,
    min_points: int,
    translation: np.ndarray,
    rotation: np.ndarray,
    ground_height: float | None,
    reference_position_xy: np.ndarray,
    *,
    enabled: bool,
    max_bottom_clearance_m: float,
    max_object_height_m: float,
    max_footprint_m: float,
    max_horizontal_distance_m: float,
) -> list[int]:
    """Select SAM masks that are valid bounded objects near the robot base."""
    if not enabled:
        return list(range(len(detections)))
    accepted = []
    for index, detection in enumerate(detections):
        geometry = project_masked_depth(
            mask_to_array(detection.mask),
            depth_image,
            camera_intrinsics,
            depth_scale,
            depth_trunc_m,
            min_points,
        )
        if geometry is None:
            continue
        world_geometry = transform_geometry(geometry, translation, rotation)
        if is_ground_object(
            world_geometry,
            ground_height,
            max_bottom_clearance_m=max_bottom_clearance_m,
            max_object_height_m=max_object_height_m,
            max_footprint_m=max_footprint_m,
            reference_position_xy=reference_position_xy,
            max_horizontal_distance_m=max_horizontal_distance_m,
        ):
            accepted.append(index)
    return accepted


def project_masked_depth(
    mask: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    depth_scale: float,
    depth_trunc_m: float,
    min_points: int,
) -> ObjectGeometry | None:
    """Back-project valid mask pixels into the camera optical frame."""
    if mask.shape != depth_image.shape[:2]:
        raise ValueError("segmentation mask and aligned depth image must have identical dimensions")
    if camera_intrinsics.shape != (3, 3):
        raise ValueError("camera_intrinsics must be a 3x3 matrix")
    if depth_scale <= 0.0:
        raise ValueError("depth_scale must be positive")

    depth_m = depth_image.astype(np.float64) / depth_scale
    valid = (mask > 0) & np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= depth_trunc_m)
    if int(valid.sum()) < min_points:
        return None

    ys, xs = np.where(valid)
    zs = depth_m[valid]
    median_z = float(np.median(zs))
    mad_z = float(np.median(np.abs(zs - median_z)))
    z_tolerance = max(3.0 * 1.4826 * mad_z, 0.02)
    inliers = np.abs(zs - median_z) <= z_tolerance
    if int(inliers.sum()) >= min_points:
        xs, ys, zs = xs[inliers], ys[inliers], zs[inliers]

    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    points = np.column_stack(((xs - cx) * zs / fx, (ys - cy) * zs / fy, zs)).astype(np.float32)
    centroid = np.median(points, axis=0)
    lower, upper = np.percentile(points, [2.0, 98.0], axis=0)
    return ObjectGeometry(points=points, centroid=centroid, size=np.maximum(upper - lower, 0.0))


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Return a 3x3 rotation matrix for a normalized quaternion."""
    norm = x * x + y * y + z * z + w * w
    if norm < 1e-12:
        raise ValueError("transform quaternion has zero norm")
    scale = 2.0 / norm
    return np.array(
        [
            [1.0 - scale * (y * y + z * z), scale * (x * y - z * w), scale * (x * z + y * w)],
            [scale * (x * y + z * w), 1.0 - scale * (x * x + z * z), scale * (y * z - x * w)],
            [scale * (x * z - y * w), scale * (y * z + x * w), 1.0 - scale * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def transform_geometry(geometry: ObjectGeometry, translation: np.ndarray, rotation: np.ndarray) -> ObjectGeometry:
    """Transform an object point cloud and recompute its world-frame AABB."""
    points = geometry.points.astype(np.float64) @ rotation.T + translation
    centroid = geometry.centroid.astype(np.float64) @ rotation.T + translation
    lower, upper = np.percentile(points, [2.0, 98.0], axis=0)
    return ObjectGeometry(points=points.astype(np.float32), centroid=centroid, size=np.maximum(upper - lower, 0.0))
