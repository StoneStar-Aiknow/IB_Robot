"""Detection-mask geometry: surface/volume centroids from mask and depth."""

import numpy as np


def volume_centroid_hull(points: np.ndarray) -> tuple[np.ndarray, float]:
    if points.shape[0] < 4:
        return points.mean(axis=0) if points.shape[0] else np.zeros(3), 0.0
    try:
        from scipy.spatial import ConvexHull, QhullError
    except ModuleNotFoundError:
        return points.mean(axis=0), 0.0
    try:
        hull = ConvexHull(points)
    except (QhullError, ValueError):
        return points.mean(axis=0), 0.0

    anchor = points[hull.vertices].mean(axis=0)
    total_volume = 0.0
    weighted_centroid = np.zeros(3)
    for simplex in hull.simplices:
        a, b, c = points[simplex] - anchor
        volume = abs(float(np.dot(a, np.cross(b, c)) / 6.0))
        total_volume += volume
        weighted_centroid += volume * (anchor + points[simplex].sum(axis=0)) / 4.0
    if total_volume < 1e-12:
        return points.mean(axis=0), 0.0
    return weighted_centroid / total_volume, total_volume


def compute_detection_geometry(
    mask: np.ndarray,
    depth_image: np.ndarray,
    camera_intrinsics: np.ndarray,
    *,
    depth_scale: float,
    depth_trunc_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    binary_mask = np.asarray(mask) > 0
    if binary_mask.shape != depth_image.shape[:2]:
        import cv2

        binary_mask = (
            cv2.resize(
                binary_mask.astype(np.uint8),
                (depth_image.shape[1], depth_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )
    depth_m = np.asarray(depth_image, dtype=np.float64) / float(depth_scale)
    valid = binary_mask & np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m <= depth_trunc_m)
    ys, xs = np.where(valid)
    zs = depth_m[valid]
    if zs.size == 0:
        zero = np.zeros(3, dtype=np.float64)
        return zero, zero.copy(), 0.0, 0

    median_z = float(np.median(zs))
    mad_z = float(np.median(np.abs(zs - median_z)))
    inliers = np.abs(zs - median_z) <= max(3.0 * 1.4826 * mad_z, 0.02)
    if int(inliers.sum()) >= max(10, int(0.2 * zs.size)):
        xs, ys, zs = xs[inliers], ys[inliers], zs[inliers]

    fx = float(camera_intrinsics[0, 0])
    fy = float(camera_intrinsics[1, 1])
    cx = float(camera_intrinsics[0, 2])
    cy = float(camera_intrinsics[1, 2])
    xs_3d = (xs - cx) * zs / fx
    ys_3d = (ys - cy) * zs / fy
    points = np.stack((xs_3d, ys_3d, zs), axis=1)
    surface_centroid = points.mean(axis=0)
    volume_centroid, volume_m3 = volume_centroid_hull(points)
    return surface_centroid, volume_centroid, volume_m3, int(len(points))
