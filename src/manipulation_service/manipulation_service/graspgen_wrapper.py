"""Wrapper for GraspGen inference pipeline (ROS-free)."""

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "manipulation_service optional dependencies are missing. Ensure IB-Robot "
        "GraspGen pip dependencies are installed with "
        "`./scripts/setup.sh --with-grasp`."
    ) from exc

logger = logging.getLogger(__name__)
_VALID_TABLETOP_FILTER_MODES = {"strict", "adaptive", "soft", "diagnostic"}
DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP = False

_LOCAL_BACKEND_REQUIRES_CUDA = (
    "GraspGen local backend requires CUDA. The upstream GraspGenSampler "
    "moves the model and point cloud tensors to CUDA internally."
)
_DEFAULT_MAX_INFERENCE_GRASPS_PER_BATCH = 1000
_EXECUTION_TABLE_SCENE_MAX_POINTS = 30000
_EXECUTION_TABLE_OBJECT_MAX_POINTS = 16000
_RANSAC_PLANE_CHUNK_SIZE = 64


def _cuda_status() -> str:
    return (
        f"torch={torch.__version__}, "
        f"torch.version.cuda={torch.version.cuda}, "
        f"torch.cuda.is_available()={torch.cuda.is_available()}"
    )


def _find_workspace_root() -> Path:
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "src").is_dir() and (parent / ".git").exists():
            return parent
        if (parent / "models").is_dir():
            return parent
    return Path(".").resolve()


_WORKSPACE_ROOT = _find_workspace_root()
_DEFAULT_MODEL_DIR = _WORKSPACE_ROOT / "models" / "grasp"


@dataclass
class GraspCandidate:
    pose_4x4: np.ndarray
    confidence: float
    collision_free: bool = True
    target_width_m: float = 0.0
    target_width_quality: float = 0.0
    width_axis_camera: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_width_min_offset_m: float = 0.0
    target_width_max_offset_m: float = 0.0


@dataclass
class GraspDiagnostic:
    mask_shape: str = ""
    depth_shape: str = ""
    mask_resized: bool = False
    mask_pixel_count: int = 0
    valid_depth_pixel_count: int = 0
    valid_depth_in_mask_count: int = 0
    valid_depth_ratio_in_mask: float = 0.0
    object_cloud_completion_enabled: bool = False
    object_cloud_completion_mode: str = "none"
    object_point_count_raw: int = 0
    object_point_count_completed: int = 0
    object_completion_added_count: int = 0
    object_prismatic_extrude_enabled: bool = False
    object_prismatic_extrude_added_count: int = 0
    object_point_count_graspgen_input: int = 0
    object_pc_raw: np.ndarray | None = None
    object_pc_after_completion: np.ndarray | None = None
    object_pc_inference_input: np.ndarray | None = None
    scene_cloud_table_holes_enabled: bool = False
    scene_point_count_raw: int = 0
    scene_table_hole_added_count: int = 0
    scene_pc_after_completion: np.ndarray | None = None
    object_point_count: int = 0
    scene_point_count: int = 0
    raw_grasp_count: int = 0
    collision_filter_before: int = 0
    collision_filter_after: int = 0
    tabletop_filter_before: int = 0
    tabletop_filter_after: int = 0
    source_gripper_tabletop_sweep_enabled: bool = False
    tabletop_plane_found: bool = False
    tabletop_plane_normal: np.ndarray | None = None
    tabletop_plane_d: float = 0.0
    tabletop_object_top_xyz: np.ndarray | None = None
    tabletop_inlier_ratio: float = 0.0
    tabletop_best_inlier_ratio: float = 0.0
    tabletop_failure_reason: str = ""
    tabletop_filter_mode: str = "strict"
    tabletop_low_profile: bool = False
    tabletop_relaxed: bool = False
    tabletop_object_height_m: float = 0.0
    tabletop_object_height_p05_m: float = 0.0
    tabletop_object_height_median_m: float = 0.0
    tabletop_object_height_p95_m: float = 0.0
    tabletop_clearance_used_m: float = 0.0
    tabletop_pregrasp_distance_used_m: float = 0.0
    tabletop_best_candidate_clearance_m: float = 0.0
    tabletop_worst_candidate_clearance_m: float = 0.0
    tabletop_auto_tuned: bool = False
    tabletop_auto_tune_attempts: int = 0
    tabletop_auto_tune_reason: str = ""
    failure_reason: str = ""
    failure_stage: str = ""
    _execution_table_fit_holder: dict[str, object] | None = None
    _execution_table_fit_thread: threading.Thread | None = None


@dataclass
class TablePlane:
    normal: np.ndarray
    d: float
    inlier_ratio: float


@dataclass
class TablePlaneFit:
    plane: TablePlane | None
    best_inlier_ratio: float = 0.0
    failure_reason: str = ""


def fit_table_plane_ransac(
    scene_pc: np.ndarray,
    positive_reference: np.ndarray | None = None,
    distance_threshold: float = 0.006,
    min_inlier_ratio: float = 0.15,
    max_iterations: int = 1000,
    max_samples: int = 30000,
    seed: int = 0,
) -> TablePlaneFit:
    pts = scene_pc[np.isfinite(scene_pc).all(axis=1)]
    if len(pts) < 100:
        logger.warning("Table plane fitting: too few scene points (%d)", len(pts))
        return TablePlaneFit(None, failure_reason=f"too few finite scene points ({len(pts)} < 100)")
    rng = np.random.default_rng(seed)
    if len(pts) > max_samples:
        pts_work = pts[rng.choice(len(pts), max_samples, replace=False)]
    else:
        pts_work = pts

    candidate_normals = []
    candidate_offsets = []
    for _ in range(max_iterations):
        tri = pts_work[rng.choice(len(pts_work), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ tri[0]
        candidate_normals.append(n)
        candidate_offsets.append(d)
    if not candidate_normals:
        return TablePlaneFit(None, failure_reason="no non-degenerate table plane sample found")

    normals = np.asarray(candidate_normals)
    offsets = np.asarray(candidate_offsets)
    counts = np.empty(len(normals), dtype=np.int64)
    for start in range(0, len(normals), _RANSAC_PLANE_CHUNK_SIZE):
        end = min(start + _RANSAC_PLANE_CHUNK_SIZE, len(normals))
        distances = np.abs(pts_work @ normals[start:end].T + offsets[start:end])
        counts[start:end] = np.sum(distances < distance_threshold, axis=0)

    best_index = int(np.argmax(counts))
    best_count = int(counts[best_index])
    if best_count <= 0:
        return TablePlaneFit(None, failure_reason="no non-degenerate table plane sample found")
    best_normal = normals[best_index]
    best_d = float(offsets[best_index])
    inlier_mask = np.abs(pts @ best_normal + best_d) < distance_threshold
    inlier_pts = pts[inlier_mask]
    inlier_ratio = len(inlier_pts) / len(pts)
    if inlier_ratio < min_inlier_ratio:
        logger.info(
            "Table plane inlier ratio %.3f < %.3f, skipping",
            inlier_ratio,
            min_inlier_ratio,
        )
        return TablePlaneFit(
            None,
            best_inlier_ratio=float(inlier_ratio),
            failure_reason=f"best inlier ratio {inlier_ratio:.3f} < minimum {min_inlier_ratio:.3f}",
        )
    center = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - center, full_matrices=False)
    normal = vh[-1]
    d = -normal @ center
    if positive_reference is not None and positive_reference @ normal + d < 0:
        normal = -normal
        d = -d
    logger.info(
        "Table plane: n=[%.4f,%.4f,%.4f] d=%.4f inlier_ratio=%.3f",
        normal[0],
        normal[1],
        normal[2],
        d,
        inlier_ratio,
    )
    return TablePlaneFit(TablePlane(normal=normal, d=d, inlier_ratio=inlier_ratio), best_inlier_ratio=inlier_ratio)


def _sample_execution_table_points(points: np.ndarray | None, max_points: int) -> np.ndarray:
    if points is None:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3 or len(arr) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    step = max(1, len(arr) // max(1, int(max_points)))
    return arr[::step]


def fit_execution_table_plane(
    scene_points: np.ndarray | None,
    object_points: np.ndarray | None,
) -> TablePlaneFit:
    scene_sample = _sample_execution_table_points(scene_points, _EXECUTION_TABLE_SCENE_MAX_POINTS)
    object_sample = _sample_execution_table_points(object_points, _EXECUTION_TABLE_OBJECT_MAX_POINTS)
    positive_reference = object_sample.mean(axis=0) if len(object_sample) else None
    return fit_table_plane_ransac(
        scene_sample,
        positive_reference=positive_reference,
        distance_threshold=0.006,
        min_inlier_ratio=0.15,
    )


def _start_execution_table_fit(diagnostic: GraspDiagnostic) -> None:
    holder: dict[str, object] = {}

    def worker() -> None:
        started = time.perf_counter()
        try:
            object_points = diagnostic.object_pc_after_completion
            if object_points is None:
                object_points = diagnostic.object_pc_raw
            holder["fit"] = fit_execution_table_plane(diagnostic.scene_pc_after_completion, object_points)
        except Exception as exc:
            holder["error"] = exc
        finally:
            holder["duration_s"] = time.perf_counter() - started

    thread = threading.Thread(target=worker, name="execution-table-fit", daemon=True)
    diagnostic._execution_table_fit_holder = holder
    diagnostic._execution_table_fit_thread = thread
    thread.start()


def _shape_string(array: np.ndarray) -> str:
    return "x".join(str(dim) for dim in array.shape)


def _tabletop_object_height_stats(object_pc: np.ndarray, table: TablePlane) -> tuple[float, float, float, float]:
    signed = object_pc @ table.normal + table.d
    signed = signed[np.isfinite(signed)]
    if len(signed) == 0:
        return 0.0, 0.0, 0.0, 0.0
    p05, median, p95 = np.quantile(signed, [0.05, 0.50, 0.95])
    return float(max(0.0, p95)), float(p05), float(median), float(p95)


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


def _resolve_tabletop_filter_params(
    *,
    mode: str,
    object_height_m: float,
    clearance: float,
    pregrasp_distance: float,
    low_profile_height: float,
    clearance_min: float,
    clearance_max: float,
    clearance_height_ratio: float,
    pregrasp_min: float,
    pregrasp_height_ratio: float,
) -> tuple[float, float, bool]:
    if mode not in {"adaptive", "soft"} or object_height_m <= 0.0 or object_height_m >= low_profile_height:
        return clearance, pregrasp_distance, False

    clearance_upper = min(clearance, clearance_max) if clearance_max > 0.0 else clearance
    adaptive_clearance = object_height_m * clearance_height_ratio
    adaptive_clearance = max(clearance_min, min(clearance_upper, adaptive_clearance))

    adaptive_pregrasp = object_height_m * pregrasp_height_ratio
    adaptive_pregrasp = max(pregrasp_min, min(pregrasp_distance, adaptive_pregrasp))
    return adaptive_clearance, adaptive_pregrasp, True


def _resolve_tabletop_retry_clearances(
    retry_clearances: tuple[float, ...] | list[float],
    *,
    initial_clearance: float,
    clearance_min: float,
) -> list[float]:
    values = []
    seen = set()
    for value in retry_clearances:
        retry = max(float(clearance_min), float(value))
        key = round(retry, 6)
        if retry <= 0.0 or retry >= initial_clearance or key in seen:
            continue
        values.append(retry)
        seen.add(key)
    if 0.0 < clearance_min < initial_clearance:
        key = round(clearance_min, 6)
        if key not in seen:
            values.append(float(clearance_min))
    return values


def _align_mask_to_depth(segmentation_mask: np.ndarray, depth_m: np.ndarray, diag: GraspDiagnostic) -> np.ndarray:
    mask = np.asarray(segmentation_mask)
    diag.mask_shape = _shape_string(mask)
    diag.depth_shape = _shape_string(depth_m)

    if depth_m.ndim != 2:
        raise ValueError(f"depth image must be 2-D, got shape {depth_m.shape}")
    if mask.ndim != 2:
        raise ValueError(f"segmentation mask must be 2-D, got shape {mask.shape}")

    if mask.shape == depth_m.shape:
        return mask

    import cv2

    diag.mask_resized = True
    return cv2.resize(mask.astype(np.uint8), (depth_m.shape[1], depth_m.shape[0]), interpolation=cv2.INTER_NEAREST)


def _downsample_points(points: np.ndarray, max_points: int, seed: int = 0) -> np.ndarray:
    if max_points <= 0 or len(points) <= max_points:
        return points
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(points), max_points, replace=False)
    return points[np.sort(idx)]


def _graspgen_inference_batches(total_grasps: int, max_batch_size: int) -> list[int]:
    total = max(1, int(total_grasps))
    batch = max(1, int(max_batch_size))
    batches = []
    remaining = total
    while remaining > 0:
        current = min(batch, remaining)
        batches.append(current)
        remaining -= current
    return batches


def _populate_depth_mask_diagnostics(mask: np.ndarray, depth_m: np.ndarray, diag: GraspDiagnostic) -> None:
    mask_bool = mask > 0
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    valid_depth_in_mask = valid_depth & mask_bool

    diag.mask_pixel_count = int(mask_bool.sum())
    diag.valid_depth_pixel_count = int(valid_depth.sum())
    diag.valid_depth_in_mask_count = int(valid_depth_in_mask.sum())
    if diag.mask_pixel_count > 0:
        diag.valid_depth_ratio_in_mask = diag.valid_depth_in_mask_count / diag.mask_pixel_count


def _points_from_pixels(
    depth_m: np.ndarray,
    ys: np.ndarray,
    xs: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
):
    zs = depth_m[ys, xs]
    x3d = (xs - cx) * zs / fx
    y3d = (ys - cy) * zs / fy
    return np.stack([x3d, y3d, zs], axis=-1).astype(np.float32)


def _dilate_bool_grid(mask: np.ndarray, iterations: int) -> np.ndarray:
    iterations = max(0, int(iterations))
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(iterations):
        src = out.copy()
        expanded = src.copy()
        for du in (-1, 0, 1):
            for dv in (-1, 0, 1):
                if du == 0 and dv == 0:
                    continue
                src_u = slice(max(0, -du), src.shape[0] - max(0, du))
                dst_u = slice(max(0, du), src.shape[0] - max(0, -du))
                src_v = slice(max(0, -dv), src.shape[1] - max(0, dv))
                dst_v = slice(max(0, dv), src.shape[1] - max(0, -dv))
                expanded[dst_u, dst_v] |= src[src_u, src_v]
        out = expanded
    return out


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

    mask = segmentation_mask > 0
    valid = np.isfinite(depth_m) & (depth_m > 0) & mask
    holes = mask & ~valid
    if not holes.any():
        return np.empty((0, 3), dtype=np.float32)

    if not valid.any():
        return np.empty((0, 3), dtype=np.float32)

    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    kernel = np.ones((k, k), dtype=np.float32)
    valid_f = valid.astype(np.float32)
    depth_sum = cv2.filter2D((depth_m * valid_f).astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    neighbor_count = cv2.filter2D(valid_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    fill = holes & (neighbor_count >= float(min_neighbors))
    ys, xs = np.where(fill)
    if len(ys) == 0:
        return np.empty((0, 3), dtype=np.float32)
    if len(ys) > max_added_points:
        keep = np.linspace(0, len(ys) - 1, max_added_points, dtype=np.int64)
        ys = ys[keep]
        xs = xs[keep]

    completed_depth = depth_m.copy()
    completed_depth[ys, xs] = depth_sum[ys, xs] / neighbor_count[ys, xs]
    return _points_from_pixels(completed_depth, ys, xs, fx, fy, cx, cy)


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
        footprint_grid = _dilate_bool_grid(footprint_grid, int(footprint_dilation_cells))
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


class GraspGenWrapper:
    def __init__(
        self,
        gripper_config: str = "graspgen_robotiq_2f_140.yml",
        model_dir: str | None = None,
        device: str = "cuda",
        collision_gripper: str | None = None,
    ):
        from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
        from grasp_gen.robot import get_gripper_info

        if device != "cuda":
            raise RuntimeError(f"{_LOCAL_BACKEND_REQUIRES_CUDA} Requested device={device!r}.")
        if not torch.cuda.is_available():
            raise RuntimeError(f"{_LOCAL_BACKEND_REQUIRES_CUDA} CUDA status: {_cuda_status()}.")
        self.device = "cuda"

        base = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        gripper_cfg_path = self._resolve_config(gripper_config, base)

        if not gripper_cfg_path.exists():
            raise FileNotFoundError(
                f"Gripper config not found: {gripper_cfg_path}\n"
                f"Download model weights with:\n"
                f'  HF_ENDPOINT=https://hf-mirror.com python3 -c "'
                f"from huggingface_hub import hf_hub_download; "
                f"hf_hub_download('adithyamurali/GraspGenModels', "
                f"'checkpoints/{gripper_config}', "
                f"local_dir='{base}')\""
            )

        logger.info("Loading GraspGen config from %s", gripper_cfg_path)
        self._cfg = load_grasp_cfg(str(gripper_cfg_path))
        self._gripper_name = self._cfg.data.gripper_name
        self._inference_point_count = int(getattr(self._cfg.data, "num_points", 2048))

        logger.info("Initializing GraspGenSampler on %s ...", self.device)
        self._sampler = GraspGenSampler(self._cfg)
        logger.info("GraspGenSampler ready (gripper=%s)", self._gripper_name)

        collision_name = collision_gripper or self._gripper_name
        self._collision_gripper_name = collision_name
        self._gripper_info = get_gripper_info(collision_name)
        self._collision_mesh = self._gripper_info.collision_mesh
        self._collision_vertices = np.asarray(self._collision_mesh.vertices)

    @staticmethod
    def _resolve_config(filename: str, model_dir: Path) -> Path:
        p = Path(filename)
        if p.is_absolute():
            return p

        candidates = [
            model_dir / "checkpoints" / filename,
            model_dir / filename,
        ]
        env_model_dir = os.environ.get("GRASPGEN_MODEL_DIR")
        if env_model_dir:
            candidates.append(Path(env_model_dir) / "checkpoints" / filename)
        for c in candidates:
            if c.exists():
                return c
        return candidates[0]

    def plan_grasps(
        self,
        depth_image: np.ndarray,
        segmentation_mask: np.ndarray,
        camera_intrinsics: np.ndarray,
        depth_scale: float = 1000.0,
        grasp_threshold: float = 0.5,
        num_grasps: int = 2000,
        topk_num_grasps: int = 300,
        min_grasps: int = 80,
        max_tries: int = 4,
        enable_collision_filter: bool = True,
        collision_threshold: float = 0.005,
        max_scene_points: int = 8192,
        enable_tabletop_filter: bool = True,
        enable_source_gripper_tabletop_sweep: bool = DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP,
        require_tabletop_filter: bool = True,
        tabletop_clearance: float = 0.003,
        tabletop_pregrasp_distance: float = 0.08,
        tabletop_pregrasp_steps: int = 5,
        tabletop_ransac_threshold: float = 0.006,
        tabletop_min_inlier_ratio: float = 0.15,
        tabletop_filter_mode: str = "strict",
        adaptive_tabletop_low_profile_height: float = 0.035,
        adaptive_tabletop_clearance_min: float = 0.001,
        adaptive_tabletop_clearance_max: float = 0.002,
        adaptive_tabletop_clearance_height_ratio: float = 0.12,
        adaptive_tabletop_pregrasp_min: float = 0.02,
        adaptive_tabletop_pregrasp_height_ratio: float = 2.0,
        adaptive_tabletop_hard_floor: float = -0.002,
        adaptive_tabletop_auto_tune: bool = True,
        adaptive_tabletop_retry_clearances: tuple[float, ...] | list[float] = (0.002, 0.001),
        adaptive_tabletop_retry_pregrasp_height_ratio: float = 1.5,
        adaptive_tabletop_retry_hard_floor: float = -0.003,
        target_width_axis_local: tuple[float, float, float] | list[float] = (1.0, 0.0, 0.0),
        target_width_percentile_low: float = 5.0,
        target_width_percentile_high: float = 95.0,
        target_width_min_m: float = 0.005,
        target_width_max_m: float = 0.14,
        enable_object_cloud_completion: bool = False,
        object_cloud_completion_mode: str = "none",
        object_cloud_completion_max_points: int = 5000,
        object_cloud_completion_kernel_size: int = 5,
        object_cloud_completion_min_neighbors: int = 6,
        enable_object_cloud_prismatic_extrude: bool = False,
        object_cloud_prismatic_extrude_max_points: int = 8000,
        object_cloud_prismatic_extrude_layers: int = 8,
        enable_scene_cloud_table_holes: bool = False,
        scene_cloud_table_holes_max_points: int = 8000,
    ) -> tuple[list[GraspCandidate], GraspDiagnostic]:
        from grasp_gen.grasp_server import GraspGenSampler as _GS
        from grasp_gen.utils.point_cloud_utils import (
            depth_and_segmentation_to_point_clouds,
            filter_colliding_grasps,
            point_cloud_outlier_removal,
        )

        diag = GraspDiagnostic()
        completion_mode = object_cloud_completion_mode.strip().lower()
        if completion_mode not in {"none", "mask_depth_inpaint"}:
            raise ValueError(
                f"invalid object_cloud_completion_mode {object_cloud_completion_mode!r}; "
                "expected one of ['none', 'mask_depth_inpaint']"
            )
        completion_enabled = bool(enable_object_cloud_completion and completion_mode == "mask_depth_inpaint")
        diag.object_cloud_completion_enabled = completion_enabled
        diag.object_cloud_completion_mode = completion_mode if completion_enabled else "none"
        mode = tabletop_filter_mode.strip().lower()
        if mode not in _VALID_TABLETOP_FILTER_MODES:
            raise ValueError(
                f"invalid tabletop_filter_mode {tabletop_filter_mode!r}; "
                f"expected one of {sorted(_VALID_TABLETOP_FILTER_MODES)}"
            )
        diag.tabletop_filter_mode = mode

        depth_m = depth_image.astype(np.float64) / depth_scale

        mask_for_pc = _align_mask_to_depth(segmentation_mask, depth_m, diag)
        _populate_depth_mask_diagnostics(mask_for_pc, depth_m, diag)

        if diag.mask_pixel_count == 0:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = "segmentation mask is empty after alignment to depth image"
            logger.warning(diag.failure_reason)
            return [], diag
        if diag.valid_depth_pixel_count == 0:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = "depth image contains no finite positive depth pixels"
            logger.warning(diag.failure_reason)
            return [], diag
        if diag.valid_depth_in_mask_count == 0:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = (
                "segmentation mask has no valid depth pixels — check RGB/depth alignment, "
                "depth holes on the target, or depth encoding/scale"
            )
            logger.warning(diag.failure_reason)
            return [], diag

        fx = float(camera_intrinsics[0, 0])
        fy = float(camera_intrinsics[1, 1])
        cx = float(camera_intrinsics[0, 2])
        cy = float(camera_intrinsics[1, 2])

        scene_pc, object_pc, scene_colors, object_colors = depth_and_segmentation_to_point_clouds(
            depth_image=depth_m,
            segmentation_mask=mask_for_pc,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            target_object_id=1,
            remove_object_from_scene=True,
        )

        diag.object_point_count = len(object_pc)
        diag.object_point_count_raw = len(object_pc)
        diag.scene_point_count = len(scene_pc) if scene_pc is not None else 0
        diag.scene_point_count_raw = diag.scene_point_count
        diag.object_pc_raw = object_pc.astype(np.float32, copy=True)
        logger.info("Point clouds: scene=%d, object=%d", diag.scene_point_count, diag.object_point_count)

        if len(object_pc) == 0:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = (
                "object point cloud is empty after conversion despite non-empty mask/depth diagnostics — "
                "check mask labels and RGB-depth registration"
            )
            logger.warning(diag.failure_reason)
            return [], diag

        shared_table_fit = TablePlaneFit(None)
        if scene_pc is not None and len(scene_pc) > 0:
            shared_min_inlier_ratio = min(float(tabletop_min_inlier_ratio), 0.10)
            shared_table_fit = fit_table_plane_ransac(
                scene_pc,
                positive_reference=object_pc.mean(axis=0),
                distance_threshold=float(tabletop_ransac_threshold),
                min_inlier_ratio=shared_min_inlier_ratio,
            )
        shared_table = shared_table_fit.plane

        if completion_enabled:
            completed_points = complete_object_cloud_from_mask_depth(
                depth_m,
                mask_for_pc,
                fx,
                fy,
                cx,
                cy,
                max_added_points=int(object_cloud_completion_max_points),
                kernel_size=int(object_cloud_completion_kernel_size),
                min_neighbors=int(object_cloud_completion_min_neighbors),
            )
            if len(completed_points):
                object_pc = np.vstack([object_pc, completed_points.astype(object_pc.dtype, copy=False)])
            diag.object_completion_added_count = int(len(completed_points))
            diag.object_point_count_completed = int(len(object_pc))
            diag.object_point_count = int(len(object_pc))
            logger.info(
                "Object PC completion: mode=%s raw=%d depth_added=%d completed=%d",
                completion_mode,
                diag.object_point_count_raw,
                len(completed_points),
                diag.object_point_count_completed,
            )
        else:
            diag.object_point_count_completed = int(len(object_pc))

        diag.object_prismatic_extrude_enabled = bool(enable_object_cloud_prismatic_extrude)
        object_pc_shell = object_pc
        if enable_object_cloud_prismatic_extrude and scene_pc is not None and len(scene_pc) > 0:
            prismatic_points = complete_object_cloud_prismatic_extrude(
                object_pc,
                scene_pc,
                table_plane=shared_table,
                max_added_points=int(object_cloud_prismatic_extrude_max_points),
                num_layers=int(object_cloud_prismatic_extrude_layers),
            )
            if len(prismatic_points):
                object_pc = np.vstack([object_pc, prismatic_points.astype(object_pc.dtype, copy=False)])
            diag.object_prismatic_extrude_added_count = int(len(prismatic_points))
            diag.object_point_count_completed = int(len(object_pc))
            object_pc_shell = object_pc
            logger.info(
                "Object PC prismatic extrude: added=%d completed=%d",
                len(prismatic_points),
                diag.object_point_count_completed,
            )
        diag.object_pc_after_completion = object_pc_shell.astype(np.float32, copy=True)

        # Optionally generate a local dense table patch around the target footprint.
        diag.scene_cloud_table_holes_enabled = bool(enable_scene_cloud_table_holes)
        if scene_pc is not None and len(scene_pc) > 0:
            if enable_scene_cloud_table_holes:
                scene_hole_pts = complete_scene_cloud_table_holes(
                    scene_pc,
                    footprint_points=object_pc,
                    table_plane=shared_table,
                    max_added_points=int(scene_cloud_table_holes_max_points),
                )
                if len(scene_hole_pts):
                    scene_pc = np.vstack([scene_pc, scene_hole_pts.astype(scene_pc.dtype, copy=False)])
                    logger.info("Scene cloud table patch: added=%d total=%d", len(scene_hole_pts), len(scene_pc))
                diag.scene_table_hole_added_count = int(len(scene_hole_pts))
            diag.scene_point_count = int(len(scene_pc))
            diag.scene_pc_after_completion = scene_pc.astype(np.float32, copy=True)

        _start_execution_table_fit(diag)

        # GraspGen gets a hollow surface shell: visible object points, optional
        # mask-depth surface inpaint, and silhouette side walls to the table. We
        # never add interior volume samples.
        object_pc_tensor = torch.from_numpy(object_pc_shell).float()
        pc_filtered, _ = point_cloud_outlier_removal(object_pc_tensor)
        object_pc_clean = pc_filtered.numpy()
        diag.object_point_count = len(object_pc_clean)
        logger.info("Object PC for GraspGen inference (hollow shell): %d points", len(object_pc_clean))

        if len(object_pc_clean) < 20:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = (
                f"object point cloud too sparse after outlier removal ({len(object_pc_clean)} points < 20)"
            )
            logger.warning(diag.failure_reason)
            return [], diag

        object_pc_inference = _downsample_points(object_pc_clean, self._inference_point_count)
        diag.object_point_count_graspgen_input = len(object_pc_inference)
        diag.object_pc_inference_input = object_pc_inference.astype(np.float32, copy=True)
        if len(object_pc_inference) < len(object_pc_clean):
            logger.info(
                "Object PC downsampled for GraspGen inference: %d -> %d points",
                len(object_pc_clean),
                len(object_pc_inference),
            )

        grasps, confidences = self._run_batched_inference(
            _GS,
            object_pc_inference,
            grasp_threshold=grasp_threshold,
            num_grasps=num_grasps,
            topk_num_grasps=topk_num_grasps,
            min_grasps=min_grasps,
            max_tries=max_tries,
        )

        diag.raw_grasp_count = len(grasps)

        if len(grasps) == 0:
            diag.failure_stage = "graspgen_inference"
            diag.failure_reason = (
                f"GraspGen produced zero grasps (threshold={grasp_threshold}, "
                f"num_grasps={num_grasps}, object_points={len(object_pc_clean)}) — "
                f"object geometry may be unsuitable for this gripper or threshold too high"
            )
            logger.warning(diag.failure_reason)
            return [], diag

        grasps_np = grasps.cpu().numpy()
        confidences_np = confidences.cpu().numpy()
        grasps_np[:, 3, 3] = 1.0

        if enable_collision_filter and scene_pc is not None and len(scene_pc) > 0:
            pc_mean = scene_pc.mean(axis=0)
            scene_centered = scene_pc - pc_mean
            grasps_centered = grasps_np.copy()
            grasps_centered[:, :3, 3] -= pc_mean

            if len(scene_centered) > max_scene_points:
                idx = np.random.choice(len(scene_centered), max_scene_points, replace=False)
                scene_for_collision = scene_centered[idx]
            else:
                scene_for_collision = scene_centered

            collision_free_mask = filter_colliding_grasps(
                scene_pc=scene_for_collision,
                grasp_poses=grasps_centered,
                gripper_collision_mesh=self._collision_mesh,
                collision_threshold=collision_threshold,
            )

            diag.collision_filter_before = len(grasps_np)
            diag.collision_filter_after = int(collision_free_mask.sum())

            grasps_final = grasps_centered[collision_free_mask].copy()
            grasps_final[:, :3, 3] += pc_mean
            confidences_final = confidences_np[collision_free_mask]
            logger.info(
                "Collision filter: %d/%d grasps remaining",
                diag.collision_filter_after,
                diag.collision_filter_before,
            )
            if diag.collision_filter_after == 0:
                diag.failure_stage = "collision_filter"
                diag.failure_reason = (
                    f"all {diag.collision_filter_before} grasps filtered by collision — "
                    f"object is likely too close to scene obstacles for this gripper"
                )
                logger.warning(diag.failure_reason)
                return [], diag
        else:
            diag.collision_filter_before = len(grasps_np)
            diag.collision_filter_after = len(grasps_np)
            grasps_final = grasps_np
            confidences_final = confidences_np

        if enable_tabletop_filter:
            if scene_pc is None or len(scene_pc) == 0:
                if require_tabletop_filter:
                    diag.failure_stage = "tabletop_filter"
                    diag.failure_reason = "tabletop filter required but scene point cloud is empty"
                    logger.warning(diag.failure_reason)
                    return [], diag
                table = None
            else:
                fit = shared_table_fit
                if fit.plane is not None and fit.best_inlier_ratio < float(tabletop_min_inlier_ratio):
                    fit = TablePlaneFit(
                        None,
                        best_inlier_ratio=fit.best_inlier_ratio,
                        failure_reason=(
                            f"best inlier ratio {fit.best_inlier_ratio:.3f} < "
                            f"minimum {float(tabletop_min_inlier_ratio):.3f}"
                        ),
                    )
                table = fit.plane
                diag.tabletop_best_inlier_ratio = fit.best_inlier_ratio
                diag.tabletop_failure_reason = fit.failure_reason

            if table is None:
                if require_tabletop_filter:
                    diag.failure_stage = "tabletop_filter"
                    reason = diag.tabletop_failure_reason or "no acceptable table plane found"
                    diag.failure_reason = (
                        "tabletop filter required but no table plane found "
                        f"({reason}; min_inlier_ratio={tabletop_min_inlier_ratio:.3f})"
                    )
                    logger.warning(diag.failure_reason)
                    return [], diag
            else:
                diag.tabletop_plane_found = True
                diag.tabletop_plane_normal = np.asarray(table.normal, dtype=np.float64).copy()
                diag.tabletop_plane_d = float(table.d)
                if len(object_pc_clean) > 0:
                    object_heights = object_pc_clean @ table.normal + table.d
                    diag.tabletop_object_top_xyz = np.asarray(
                        object_pc_clean[int(np.argmax(object_heights))],
                        dtype=np.float64,
                    ).copy()
                diag.tabletop_inlier_ratio = table.inlier_ratio
                height, p05, median, p95 = _tabletop_object_height_stats(object_pc_clean, table)
                diag.tabletop_object_height_m = height
                diag.tabletop_object_height_p05_m = p05
                diag.tabletop_object_height_median_m = median
                diag.tabletop_object_height_p95_m = p95
                clearance_used, pregrasp_used, low_profile = _resolve_tabletop_filter_params(
                    mode=mode,
                    object_height_m=height,
                    clearance=tabletop_clearance,
                    pregrasp_distance=tabletop_pregrasp_distance,
                    low_profile_height=adaptive_tabletop_low_profile_height,
                    clearance_min=adaptive_tabletop_clearance_min,
                    clearance_max=adaptive_tabletop_clearance_max,
                    clearance_height_ratio=adaptive_tabletop_clearance_height_ratio,
                    pregrasp_min=adaptive_tabletop_pregrasp_min,
                    pregrasp_height_ratio=adaptive_tabletop_pregrasp_height_ratio,
                )
                diag.tabletop_low_profile = low_profile
                diag.tabletop_clearance_used_m = clearance_used
                diag.tabletop_pregrasp_distance_used_m = pregrasp_used
                diag.tabletop_filter_before = len(grasps_final)
                if enable_source_gripper_tabletop_sweep:
                    diag.source_gripper_tabletop_sweep_enabled = True
                    filtered_grasps, filtered_confidences, clearances = self._filter_tabletop_clearance(
                        grasps_final,
                        confidences_final,
                        table=table,
                        clearance=clearance_used,
                        pregrasp_distance=pregrasp_used,
                        pregrasp_steps=tabletop_pregrasp_steps,
                    )
                else:
                    filtered_grasps = grasps_final
                    filtered_confidences = confidences_final
                    clearances = np.zeros(0, dtype=np.float64)
                if len(clearances) > 0:
                    diag.tabletop_best_candidate_clearance_m = float(np.max(clearances))
                    diag.tabletop_worst_candidate_clearance_m = float(np.min(clearances))
                if (
                    len(filtered_grasps) == 0
                    and mode == "adaptive"
                    and low_profile
                    and adaptive_tabletop_auto_tune
                    and len(clearances) > 0
                ):
                    retry_clearances = _resolve_tabletop_retry_clearances(
                        adaptive_tabletop_retry_clearances,
                        initial_clearance=clearance_used,
                        clearance_min=adaptive_tabletop_clearance_min,
                    )
                    initial_best_clearance = diag.tabletop_best_candidate_clearance_m
                    if initial_best_clearance < adaptive_tabletop_retry_hard_floor:
                        diag.tabletop_auto_tune_reason = (
                            f"initial_best_clearance {initial_best_clearance:.3f}m < "
                            f"retry_hard_floor {adaptive_tabletop_retry_hard_floor:.3f}m"
                        )
                    elif not retry_clearances:
                        diag.tabletop_auto_tune_reason = "no retry clearance below current adaptive clearance"
                    else:
                        retry_pregrasp = height * adaptive_tabletop_retry_pregrasp_height_ratio
                        retry_pregrasp = max(adaptive_tabletop_pregrasp_min, min(pregrasp_used, retry_pregrasp))
                        for retry_clearance in retry_clearances:
                            diag.tabletop_auto_tune_attempts += 1
                            retry_grasps, retry_confidences, retry_candidate_clearances = (
                                self._filter_tabletop_clearance(
                                    grasps_final,
                                    confidences_final,
                                    table=table,
                                    clearance=retry_clearance,
                                    pregrasp_distance=retry_pregrasp,
                                    pregrasp_steps=tabletop_pregrasp_steps,
                                )
                            )
                            clearance_used = retry_clearance
                            pregrasp_used = retry_pregrasp
                            diag.tabletop_clearance_used_m = clearance_used
                            diag.tabletop_pregrasp_distance_used_m = pregrasp_used
                            if len(retry_candidate_clearances) > 0:
                                diag.tabletop_best_candidate_clearance_m = float(np.max(retry_candidate_clearances))
                                diag.tabletop_worst_candidate_clearance_m = float(np.min(retry_candidate_clearances))
                            if len(retry_grasps) > 0:
                                diag.tabletop_auto_tuned = True
                                diag.tabletop_auto_tune_reason = (
                                    f"accepted {len(retry_grasps)} grasps after retry "
                                    f"clearance={retry_clearance:.3f}m pregrasp={retry_pregrasp:.3f}m "
                                    f"initial_best_clearance={initial_best_clearance:.3f}m"
                                )
                                filtered_grasps = retry_grasps
                                filtered_confidences = retry_confidences
                                break
                        if not diag.tabletop_auto_tuned and not diag.tabletop_auto_tune_reason:
                            diag.tabletop_auto_tune_reason = "all adaptive retry clearances kept zero grasps"
                if len(filtered_grasps) == 0 and mode == "soft" and low_profile and len(clearances) > 0:
                    relaxed_mask = clearances >= adaptive_tabletop_hard_floor
                    if relaxed_mask.any():
                        diag.tabletop_relaxed = True
                        relaxed_indices = np.flatnonzero(relaxed_mask)
                        order = np.lexsort((-confidences_final[relaxed_indices], -clearances[relaxed_indices]))
                        relaxed_indices = relaxed_indices[order]
                        filtered_grasps = grasps_final[relaxed_indices]
                        filtered_confidences = confidences_final[relaxed_indices]
                        logger.warning(
                            "Tabletop soft fallback accepted %d/%d low-profile grasps "
                            "(hard_floor=%.3fm, best_clearance=%.3fm)",
                            len(filtered_grasps),
                            len(grasps_final),
                            adaptive_tabletop_hard_floor,
                            diag.tabletop_best_candidate_clearance_m,
                        )
                if mode == "diagnostic":
                    diag.tabletop_relaxed = len(filtered_grasps) < len(grasps_final)
                    diag.tabletop_filter_after = len(grasps_final)
                    if enable_source_gripper_tabletop_sweep:
                        logger.info(
                            "Tabletop diagnostic mode: kept %d grasps while %d/%d passed Robotiq tabletop clearance",
                            len(grasps_final),
                            len(filtered_grasps),
                            diag.tabletop_filter_before,
                        )
                    else:
                        logger.info(
                            "Source-gripper tabletop sweep disabled: kept %d grasps with table/object diagnostics",
                            len(grasps_final),
                        )
                else:
                    grasps_final = filtered_grasps
                    confidences_final = filtered_confidences
                    diag.tabletop_filter_after = len(grasps_final)
                if diag.tabletop_filter_after == 0 and diag.tabletop_filter_before > 0:
                    diag.failure_stage = "tabletop_filter"
                    diag.failure_reason = (
                        f"all {diag.tabletop_filter_before} grasps filtered by tabletop clearance "
                        f"(mode={mode}, clearance={clearance_used:.3f}m, pregrasp={pregrasp_used:.3f}m, "
                        f"object_height={height:.3f}m, best_clearance={diag.tabletop_best_candidate_clearance_m:.3f}m) — "
                        f"object is too close to or below the table surface"
                    )
                    logger.warning(diag.failure_reason)
                    return [], diag

        width_axis_local = np.asarray(target_width_axis_local, dtype=np.float64)
        width_estimates = estimate_candidate_target_widths(
            object_pc_clean,
            grasps_final,
            width_axis_local,
            percentile_low=target_width_percentile_low,
            percentile_high=target_width_percentile_high,
            min_width_m=target_width_min_m,
            max_width_m=target_width_max_m,
        )
        results = []
        for i, width_estimate in enumerate(width_estimates):
            target_width_m, target_width_quality, width_axis_camera, min_offset, max_offset = width_estimate
            results.append(
                GraspCandidate(
                    pose_4x4=grasps_final[i].astype(np.float32),
                    confidence=float(confidences_final[i]),
                    target_width_m=target_width_m,
                    target_width_quality=target_width_quality,
                    width_axis_camera=width_axis_camera,
                    target_width_min_offset_m=min_offset,
                    target_width_max_offset_m=max_offset,
                )
            )

        results.sort(key=lambda g: g.confidence, reverse=True)
        return results, diag

    def _run_batched_inference(
        self,
        grasp_server_cls,
        object_pc: np.ndarray,
        *,
        grasp_threshold: float,
        num_grasps: int,
        topk_num_grasps: int,
        min_grasps: int,
        max_tries: int,
    ):
        batches = _graspgen_inference_batches(num_grasps, _DEFAULT_MAX_INFERENCE_GRASPS_PER_BATCH)
        if len(batches) > 1:
            logger.info(
                "Running GraspGen inference in %d batches: total=%d max_batch=%d",
                len(batches),
                num_grasps,
                _DEFAULT_MAX_INFERENCE_GRASPS_PER_BATCH,
            )

        all_grasps = []
        all_confidences = []
        for index, batch_size in enumerate(batches, start=1):
            per_batch_topk = -1
            if grasp_threshold == -1.0 and topk_num_grasps > 0:
                per_batch_topk = min(topk_num_grasps, batch_size)

            grasps, confidences = grasp_server_cls.run_inference(
                object_pc,
                self._sampler,
                grasp_threshold=grasp_threshold,
                num_grasps=batch_size,
                topk_num_grasps=per_batch_topk,
                min_grasps=min_grasps,
                max_tries=max_tries,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if len(grasps) == 0:
                logger.info("GraspGen batch %d/%d returned zero grasps", index, len(batches))
                continue
            all_grasps.append(grasps)
            all_confidences.append(confidences)
            logger.info(
                "GraspGen batch %d/%d accepted %d grasps, total accepted=%d",
                index,
                len(batches),
                len(grasps),
                sum(len(g) for g in all_grasps),
            )

        if not all_grasps:
            return torch.tensor([], device="cuda"), torch.tensor([], device="cuda")

        grasps = torch.cat(all_grasps, dim=0)
        confidences = torch.cat(all_confidences, dim=0)
        if topk_num_grasps > 0 and len(grasps) > topk_num_grasps:
            order = torch.argsort(confidences, descending=True)[:topk_num_grasps]
            grasps = grasps[order]
            confidences = confidences[order]
        return grasps, confidences

    @property
    def gripper_name(self) -> str:
        return self._gripper_name

    @property
    def collision_gripper_name(self) -> str:
        return self._collision_gripper_name

    def _filter_tabletop_clearance(
        self,
        grasps: np.ndarray,
        confidences: np.ndarray,
        table: TablePlane,
        clearance: float,
        pregrasp_distance: float,
        pregrasp_steps: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        verts_local = self._collision_vertices
        tabletop_mask = np.ones(len(grasps), dtype=bool)
        min_clearances = np.full(len(grasps), np.inf, dtype=np.float64)
        steps = max(0, int(pregrasp_steps))
        offsets = [0.0]
        if pregrasp_distance > 0.0 and steps > 0:
            offsets.extend(np.linspace(0.0, pregrasp_distance, steps + 1)[1:])

        for i, T in enumerate(grasps):
            approach_axis = T[:3, 2]
            min_signed_dist = np.inf
            for offset in offsets:
                swept_translation = T[:3, 3] - approach_axis * offset
                verts_world = (verts_local @ T[:3, :3].T) + swept_translation
                signed_dist = verts_world @ table.normal + table.d
                min_signed_dist = min(min_signed_dist, float(signed_dist.min()))
            min_clearances[i] = min_signed_dist
            if min_signed_dist < clearance:
                tabletop_mask[i] = False

        before = len(grasps)
        filtered_grasps = grasps[tabletop_mask]
        filtered_confidences = confidences[tabletop_mask]
        logger.info(
            "Tabletop filter: %d/%d grasps remaining (clearance=%.3fm, pregrasp=%.3fm/%d steps)",
            len(filtered_grasps),
            before,
            clearance,
            pregrasp_distance,
            steps,
        )
        return filtered_grasps, filtered_confidences, min_clearances
