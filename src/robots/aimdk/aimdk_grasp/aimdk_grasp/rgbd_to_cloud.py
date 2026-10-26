"""RGB-D mask to camera optical-frame point cloud conversion."""

import numpy as np

from aimdk_grasp.types import CameraIntrinsics


def mask_depth_to_points(
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
    depth_scale: float = 1.0,
    min_depth_m: float = 0.05,
    max_depth_m: float = 3.0,
) -> np.ndarray:
    """Back-project masked aligned depth into an ``N x 3`` camera cloud."""

    mask = np.asarray(mask, dtype=bool)
    depth = np.asarray(depth)
    if mask.ndim != 2 or depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError(f"Mask/depth shape mismatch: {mask.shape} vs {depth.shape}")
    if depth_scale <= 0:
        raise ValueError("depth_scale must be positive")
    if intrinsics.fx <= 0 or intrinsics.fy <= 0:
        raise ValueError("fx and fy must be positive")

    v, u = np.nonzero(mask)
    if len(u) == 0:
        return np.empty((0, 3), dtype=np.float32)
    z = depth[v, u].astype(np.float32) * float(depth_scale)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid]
    x = (u - float(intrinsics.cx)) * z / float(intrinsics.fx)
    y = (v - float(intrinsics.cy)) * z / float(intrinsics.fy)
    return np.column_stack((x, y, z)).astype(np.float32, copy=False)


def largest_cluster_mask(
    mask, depth, intrinsics, depth_scale=1.0, *, min_depth_m=0.05, max_depth_m=3.0, voxel_m=0.01, gap_m=0.03
):
    """Keep the mask pixels whose points form the largest 3-D connected cluster.

    A segmentation mask can take in a second object that touches the target in
    the image but not in space (plan9: a bottle cap on the floor 0.66 m below
    the table, 7.8% of the mask), and silhouette pixels can carry the depth of
    the background behind the target. Points are voxelized at voxel_m; voxels
    within gap_m are connected (a transparent target's depth holes do not split
    it). Pixels without valid depth stay as they were. Returns the cleaned mask
    and the number of pixels removed.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    mask = np.asarray(mask, dtype=bool)
    v, u = np.nonzero(mask)
    if len(u) == 0:
        return mask.copy(), 0
    z = np.asarray(depth)[v, u].astype(np.float64) * float(depth_scale)
    valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    if valid.sum() < 2:
        return mask.copy(), 0
    v, u, z = v[valid], u[valid], z[valid]
    points = np.column_stack(((u - intrinsics.cx) * z / intrinsics.fx, (v - intrinsics.cy) * z / intrinsics.fy, z))
    cells, inverse = np.unique(np.floor(points / voxel_m).astype(np.int64), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    pairs = cKDTree(cells * voxel_m).query_pairs(gap_m + np.sqrt(3) * voxel_m, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(len(cells), len(cells)))
    _, labels = connected_components(graph, directed=False)
    point_labels = labels[inverse]
    keep_label = np.bincount(point_labels).argmax()
    drop = point_labels != keep_label
    cleaned = mask.copy()
    cleaned[v[drop], u[drop]] = False
    return cleaned, int(drop.sum())


def validate_cloud_alignment(points_camera, mask, depth, intrinsics, depth_scale=1.0):
    """Require the saved, pre-filter cloud to match the scene's RGB-aligned depth.

    Pipeline artifacts preserve pixel scan order before GraspGenX's filtering.
    Comparing those points detects raw/registered depth mixups, changed masks,
    intrinsics and depth scales. This checks input consistency, not calibration.
    """
    expected = mask_depth_to_points(mask, depth, intrinsics, depth_scale)
    points = np.asarray(points_camera)
    if (
        not len(expected)
        or points.shape != expected.shape
        or not np.isfinite(points).all()
        or not np.allclose(points, expected, rtol=0.0, atol=1e-6)
    ):
        raise ValueError(
            "Grasp artifact point cloud does not match the registered scene depth, mask and intrinsics; "
            "regenerate grasp candidates from the same registered RGB-D frame"
        )
