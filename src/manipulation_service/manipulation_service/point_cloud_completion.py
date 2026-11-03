"""Object/scene point-cloud completion (ROS-free)."""

import logging

import numpy as np

from .point_clouds import dilate_bool_grid
from .table_plane import TablePlane, fit_table_plane_ransac

logger = logging.getLogger(__name__)


def complete_object_cloud_from_mask_depth(
    depth_m: np.ndarray,
    segmentation_mask: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    max_added_points: int = 5000,
    kernel_size: int = 5,
    min_neighbors: int = 6,
) -> np.ndarray:
    """Fill small target-mask depth holes using local depth."""
    if max_added_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    import cv2

    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1

    mask = segmentation_mask > 0
    if not mask.any():
        return np.empty((0, 3), dtype=np.float32)

    ys_mask, xs_mask = np.where(mask)
    y0 = int(ys_mask.min())
    y1 = int(ys_mask.max()) + 1
    x0 = int(xs_mask.min())
    x1 = int(xs_mask.max()) + 1
    # Pad by kernel radius (plus one) so the convolution result is identical
    # to a full-image filter2D with BORDER_CONSTANT zero padding.
    pad = k // 2 + 1
    y0 = max(0, y0 - pad)
    y1 = min(depth_m.shape[0], y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(depth_m.shape[1], x1 + pad)

    mask_roi = np.ascontiguousarray(mask[y0:y1, x0:x1])
    depth_roi = np.ascontiguousarray(depth_m[y0:y1, x0:x1])
    valid_roi = np.isfinite(depth_roi) & (depth_roi > 0) & mask_roi
    holes_roi = mask_roi & ~valid_roi
    if not holes_roi.any() or not valid_roi.any():
        return np.empty((0, 3), dtype=np.float32)

    kernel = np.ones((k, k), dtype=np.float32)
    valid_f_roi = valid_roi.astype(np.float32)
    depth_sum_roi = cv2.filter2D(
        (depth_roi * valid_f_roi).astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT
    )
    neighbor_count_roi = cv2.filter2D(valid_f_roi, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    fill_roi = holes_roi & (neighbor_count_roi >= float(min_neighbors))
    ys_roi, xs_roi = np.where(fill_roi)
    if len(ys_roi) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(ys_roi) > max_added_points:
        keep = np.linspace(0, len(ys_roi) - 1, max_added_points, dtype=np.int64)
        ys_roi = ys_roi[keep]
        xs_roi = xs_roi[keep]

    # Depths at the filled pixels only, computed in the ROI's local frame.
    zs = (depth_sum_roi[ys_roi, xs_roi] / neighbor_count_roi[ys_roi, xs_roi]).astype(np.float32)
    # Map ROI-local pixel coordinates back to the full image frame so that
    # camera-intrinsics projection matches the original full-image math.
    ys = ys_roi.astype(np.int64) + y0
    xs = xs_roi.astype(np.int64) + x0
    x3d = (xs - cx) * zs / fx
    y3d = (ys - cy) * zs / fy
    return np.stack([x3d, y3d, zs], axis=-1).astype(np.float32)


def complete_object_cloud_prismatic_extrude(
    object_points: np.ndarray,
    scene_points: np.ndarray | None,
    *,
    table_plane: TablePlane | None = None,
    max_added_points: int = 8000,
    num_layers: int = 8,
    max_object_height_m: float = 0.10,
    ransac_distance_threshold: float = 0.006,
    ransac_min_inlier_ratio: float = 0.15,
    boundary_bins: int = 144,
    boundary_quantile: float = 0.96,
    min_height_m: float = 0.003,
    seed: int = 0,
) -> np.ndarray:
    """Connect the visible surface silhouette to the table as a hollow shell."""
    pts = np.asarray(object_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 20:
        return np.empty((0, 3), dtype=np.float32)
    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) < 20:
        return np.empty((0, 3), dtype=np.float32)
    if table_plane is None:
        if scene_points is None:
            return np.empty((0, 3), dtype=np.float32)

        scene = np.asarray(scene_points, dtype=np.float64)
        scene = scene[np.isfinite(scene).all(axis=1)]
        if len(scene) < 100:
            return np.empty((0, 3), dtype=np.float32)

        fit = fit_table_plane_ransac(
            scene,
            positive_reference=finite.mean(axis=0),
            distance_threshold=float(ransac_distance_threshold),
            min_inlier_ratio=float(ransac_min_inlier_ratio),
            seed=seed,
        )
        if fit.plane is None:
            return np.empty((0, 3), dtype=np.float32)
        table_plane = fit.plane

    normal = np.asarray(table_plane.normal, dtype=np.float64)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 1e-9:
        return np.empty((0, 3), dtype=np.float32)
    normal = normal / normal_norm
    d = float(table_plane.d)

    if np.median(finite @ normal + d) < 0.0:
        normal = -normal
        d = -d

    # Build tangent basis
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = ref - normal * np.dot(ref, normal)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    signed = finite @ normal + d
    height_cap = float(max_object_height_m)
    finite = finite[(signed >= float(min_height_m)) & (signed <= height_cap)]
    signed = finite @ normal + d
    if len(finite) < 20:
        return np.empty((0, 3), dtype=np.float32)

    tan_pts = np.column_stack([finite @ t1, finite @ t2])
    centroid_tan = np.median(tan_pts, axis=0)
    relative = tan_pts - centroid_tan
    angles = np.arctan2(relative[:, 1], relative[:, 0])
    radii = np.linalg.norm(relative, axis=1)

    n_bins = max(16, int(boundary_bins))
    quantile = min(max(float(boundary_quantile), 0.5), 1.0)
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    boundary_points = []
    boundary_angles = []
    for i in range(n_bins):
        mask = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
        if not mask.any():
            continue
        local_indices = np.flatnonzero(mask)
        threshold = np.quantile(radii[mask], quantile)
        shell_indices = local_indices[radii[mask] >= threshold]
        if len(shell_indices) == 0:
            shell_indices = local_indices[[int(np.argmax(radii[mask]))]]
        weights = radii[shell_indices]
        if float(weights.sum()) <= 1e-9:
            top_point = finite[shell_indices].mean(axis=0)
        else:
            top_point = np.average(finite[shell_indices], axis=0, weights=weights)
        boundary_points.append(top_point)
        boundary_angles.append(float((bin_edges[i] + bin_edges[i + 1]) * 0.5))

    if len(boundary_points) < 5:
        return np.empty((0, 3), dtype=np.float32)
    order = np.argsort(boundary_angles)
    boundary_pts = np.asarray(boundary_points, dtype=np.float64)[order]
    boundary_signed = np.maximum(boundary_pts @ normal + d, float(min_height_m))
    projected_pts = boundary_pts - boundary_signed[:, None] * normal

    layers = max(2, int(num_layers))
    vertical_fractions = np.linspace(0.0, 1.0, layers + 2)[1:]
    wall_parts = []
    for i in range(len(boundary_pts)):
        j = (i + 1) % len(boundary_pts)
        top_a = boundary_pts[i]
        top_b = boundary_pts[j]
        foot_a = projected_pts[i]
        foot_b = projected_pts[j]
        edge_len = float(np.linalg.norm(foot_b - foot_a))
        edge_samples = max(3, int(np.ceil(edge_len / 0.0025)))
        for edge_frac in np.linspace(0.0, 1.0, edge_samples, endpoint=False):
            top = (1.0 - edge_frac) * top_a + edge_frac * top_b
            foot = (1.0 - edge_frac) * foot_a + edge_frac * foot_b
            for vertical_frac in vertical_fractions:
                wall_parts.append((1.0 - vertical_frac) * top + vertical_frac * foot)

    if not wall_parts:
        return np.empty((0, 3), dtype=np.float32)

    wall_pts = np.asarray(wall_parts, dtype=np.float64)
    max_points = int(max_added_points)
    if max_points > 0 and len(wall_pts) > max_points:
        _ = seed
        idx = np.linspace(0, len(wall_pts) - 1, max_points, dtype=np.int64)
        wall_pts = wall_pts[idx]

    return wall_pts.astype(np.float32, copy=False)


def complete_scene_cloud_table_holes(
    scene_points: np.ndarray,
    *,
    footprint_points: np.ndarray | None = None,
    table_plane: TablePlane | None = None,
    max_added_points: int = 8000,
    grid_size: float = 0.005,
    footprint_dilation_cells: int = 3,
    plane_distance_threshold: float = 0.02,
    ransac_distance_threshold: float = 0.008,
    ransac_min_inlier_ratio: float = 0.10,
    seed: int = 0,
) -> np.ndarray:
    """Generate table points around the target footprint.

    Fits the table plane from scene points via RANSAC unless ``table_plane`` is
    provided, projects table-near points onto a 2D tangent-space grid, then
    fills empty cells when no footprint is provided. With ``footprint_points``,
    it intentionally emits a dense local table patch inside the dilated target
    footprint, not only empty cells. These generated patch points may be added
    to the scene cloud and used by collision/tabletop filters.

    This covers:
    - Object occlusion (table hidden under the object)
    - Sensor depth holes (reflective/dark surfaces)
    - Local execution/debug table context near the target footprint
    Returns points in the same frame as ``scene_points``.
    """

    pts = np.asarray(scene_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 100:
        return np.empty((0, 3), dtype=np.float32)

    if table_plane is None:
        fit = fit_table_plane_ransac(
            pts,
            distance_threshold=float(ransac_distance_threshold),
            min_inlier_ratio=float(ransac_min_inlier_ratio),
            seed=seed,
        )
        if fit.plane is None:
            return np.empty((0, 3), dtype=np.float32)
        table_plane = fit.plane

    normal = np.asarray(table_plane.normal, dtype=np.float64)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-9:
        return np.empty((0, 3), dtype=np.float32)
    normal = normal / norm
    d = float(table_plane.d)
    if normal[2] > 0:
        normal = -normal
        d = -d

    # Select points near the table plane
    signed = pts @ normal + d
    on_table = pts[np.abs(signed) < float(plane_distance_threshold)]
    if len(on_table) < 50:
        return np.empty((0, 3), dtype=np.float32)

    # Build tangent basis on the table plane
    ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = ref - normal * np.dot(ref, normal)
    t1 = t1 / np.linalg.norm(t1)
    t2 = np.cross(normal, t1)

    # Project ALL table-near points onto tangent space
    u = on_table @ t1
    v = on_table @ t2
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    u_span, v_span = u_max - u_min, v_max - v_min
    if u_span < 0.02 or v_span < 0.02:
        return np.empty((0, 3), dtype=np.float32)

    # Build occupancy grid from ALL table points
    u_bins = max(2, int(u_span / float(grid_size)))
    v_bins = max(2, int(v_span / float(grid_size)))
    grid = np.zeros((u_bins, v_bins), dtype=bool)
    ui = np.clip(((u - u_min) / float(grid_size)).astype(int), 0, u_bins - 1)
    vi = np.clip(((v - v_min) / float(grid_size)).astype(int), 0, v_bins - 1)
    grid[ui, vi] = True

    fill_mask = ~grid
    if footprint_points is not None:
        fp = np.asarray(footprint_points, dtype=np.float64)
        if fp.ndim != 2 or fp.shape[1] != 3:
            return np.empty((0, 3), dtype=np.float32)
        fp = fp[np.isfinite(fp).all(axis=1)]
        if len(fp) == 0:
            return np.empty((0, 3), dtype=np.float32)

        fp_u = fp @ t1
        fp_v = fp @ t2
        fp_in_bounds = (fp_u >= u_min) & (fp_u <= u_max) & (fp_v >= v_min) & (fp_v <= v_max)
        if not fp_in_bounds.any():
            return np.empty((0, 3), dtype=np.float32)

        fp_ui = np.clip(((fp_u[fp_in_bounds] - u_min) / float(grid_size)).astype(int), 0, u_bins - 1)
        fp_vi = np.clip(((fp_v[fp_in_bounds] - v_min) / float(grid_size)).astype(int), 0, v_bins - 1)
        footprint_grid = np.zeros_like(grid)
        footprint_grid[fp_ui, fp_vi] = True
        footprint_grid = dilate_bool_grid(footprint_grid, int(footprint_dilation_cells))
        # With a target footprint, generate a dense local table patch instead
        # of only filling empty cells. The dense patch gives execution/debug
        # views enough context while staying bounded to the object area.
        fill_mask = footprint_grid

    hole_u, hole_v = np.where(fill_mask)
    if len(hole_u) == 0:
        return np.empty((0, 3), dtype=np.float32)

    if len(hole_u) > int(max_added_points):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(hole_u), int(max_added_points), replace=False)
        hole_u, hole_v = hole_u[idx], hole_v[idx]

    # Generate 3D points on the plane at all hole positions
    u_pts = u_min + (hole_u + 0.5) * float(grid_size)
    v_pts = v_min + (hole_v + 0.5) * float(grid_size)
    origin = -d * normal
    pts_3d = origin[None, :] + u_pts[:, None] * t1[None, :] + v_pts[:, None] * t2[None, :]
    return pts_3d.astype(np.float32, copy=False)
