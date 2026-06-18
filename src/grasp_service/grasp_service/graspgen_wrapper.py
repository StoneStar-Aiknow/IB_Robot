"""Wrapper for GraspGen inference pipeline (ROS-free)."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "grasp_service optional dependencies are missing. Ensure IB-Robot "
        "GraspGen pip dependencies are installed with "
        "`./scripts/setup.sh --with-grasp`."
    ) from exc

logger = logging.getLogger(__name__)
_VALID_TABLETOP_FILTER_MODES = {"strict", "adaptive", "soft"}

_LOCAL_BACKEND_REQUIRES_CUDA = (
    "GraspGen local backend requires CUDA. The upstream GraspGenSampler "
    "moves the model and point cloud tensors to CUDA internally."
)


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


@dataclass
class GraspDiagnostic:
    mask_shape: str = ""
    depth_shape: str = ""
    mask_resized: bool = False
    mask_pixel_count: int = 0
    valid_depth_pixel_count: int = 0
    valid_depth_in_mask_count: int = 0
    valid_depth_ratio_in_mask: float = 0.0
    object_point_count: int = 0
    scene_point_count: int = 0
    raw_grasp_count: int = 0
    collision_filter_before: int = 0
    collision_filter_after: int = 0
    tabletop_filter_before: int = 0
    tabletop_filter_after: int = 0
    tabletop_plane_found: bool = False
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
    best_count = 0
    best_normal = None
    best_d = 0.0
    for _ in range(max_iterations):
        tri = pts_work[rng.choice(len(pts_work), 3, replace=False)]
        n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ tri[0]
        count = int((np.abs(pts_work @ n + d) < distance_threshold).sum())
        if count > best_count:
            best_count = count
            best_normal = n
            best_d = d
    if best_normal is None:
        return TablePlaneFit(None, failure_reason="no non-degenerate table plane sample found")
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
) -> tuple[float, float, tuple[float, float, float]]:
    pts = np.asarray(object_points_camera, dtype=np.float64)
    axis_local = np.asarray(width_axis_local, dtype=np.float64)
    axis_norm = float(np.linalg.norm(axis_local))
    if len(pts) < min_points or axis_norm <= 1e-9:
        return 0.0, 0.0, (0.0, 0.0, 0.0)

    low = float(percentile_low)
    high = float(percentile_high)
    if not (0.0 <= low < high <= 100.0):
        low, high = 5.0, 95.0

    axis_local = axis_local / axis_norm
    rotation = np.asarray(grasp_pose_camera[:3, :3], dtype=np.float64)
    axis_camera = rotation @ axis_local
    axis_camera_norm = float(np.linalg.norm(axis_camera))
    if axis_camera_norm <= 1e-9:
        return 0.0, 0.0, (0.0, 0.0, 0.0)
    axis_camera = axis_camera / axis_camera_norm

    finite = pts[np.isfinite(pts).all(axis=1)]
    if len(finite) < min_points:
        return 0.0, 0.0, tuple(float(v) for v in axis_camera)

    projections = finite @ axis_camera
    p_low, p_high = np.percentile(projections, [low, high])
    raw_width = max(0.0, float(p_high - p_low))
    width = min(max(raw_width, float(min_width_m)), float(max_width_m))
    quality = 1.0 if abs(width - raw_width) <= 1e-6 else 0.5
    if raw_width <= 0.0:
        quality = 0.0
    return width, quality, tuple(float(v) for v in axis_camera)


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


def _populate_depth_mask_diagnostics(mask: np.ndarray, depth_m: np.ndarray, diag: GraspDiagnostic) -> None:
    mask_bool = mask > 0
    valid_depth = np.isfinite(depth_m) & (depth_m > 0)
    valid_depth_in_mask = valid_depth & mask_bool

    diag.mask_pixel_count = int(mask_bool.sum())
    diag.valid_depth_pixel_count = int(valid_depth.sum())
    diag.valid_depth_in_mask_count = int(valid_depth_in_mask.sum())
    if diag.mask_pixel_count > 0:
        diag.valid_depth_ratio_in_mask = diag.valid_depth_in_mask_count / diag.mask_pixel_count


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
    ) -> tuple[list[GraspCandidate], GraspDiagnostic]:
        from grasp_gen.grasp_server import GraspGenSampler as _GS
        from grasp_gen.utils.point_cloud_utils import (
            depth_and_segmentation_to_point_clouds,
            filter_colliding_grasps,
            point_cloud_outlier_removal,
        )

        diag = GraspDiagnostic()
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
        diag.scene_point_count = len(scene_pc) if scene_pc is not None else 0
        logger.info("Point clouds: scene=%d, object=%d", diag.scene_point_count, diag.object_point_count)

        if len(object_pc) == 0:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = (
                "object point cloud is empty after conversion despite non-empty mask/depth diagnostics — "
                "check mask labels and RGB-depth registration"
            )
            logger.warning(diag.failure_reason)
            return [], diag

        object_pc_tensor = torch.from_numpy(object_pc).float()
        pc_filtered, _ = point_cloud_outlier_removal(object_pc_tensor)
        object_pc_clean = pc_filtered.numpy()
        diag.object_point_count = len(object_pc_clean)
        logger.info("Object PC after outlier removal: %d points", len(object_pc_clean))

        if len(object_pc_clean) < 20:
            diag.failure_stage = "point_cloud"
            diag.failure_reason = (
                f"object point cloud too sparse after outlier removal ({len(object_pc_clean)} points < 20)"
            )
            logger.warning(diag.failure_reason)
            return [], diag

        grasps, confidences = _GS.run_inference(
            object_pc_clean,
            self._sampler,
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
                fit = fit_table_plane_ransac(
                    scene_pc,
                    positive_reference=object_pc_clean.mean(axis=0),
                    distance_threshold=tabletop_ransac_threshold,
                    min_inlier_ratio=tabletop_min_inlier_ratio,
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
                filtered_grasps, filtered_confidences, clearances = self._filter_tabletop_clearance(
                    grasps_final,
                    confidences_final,
                    table=table,
                    clearance=clearance_used,
                    pregrasp_distance=pregrasp_used,
                    pregrasp_steps=tabletop_pregrasp_steps,
                )
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
                        filtered_grasps = grasps_final[relaxed_mask]
                        filtered_confidences = confidences_final[relaxed_mask]
                        logger.warning(
                            "Tabletop soft fallback accepted %d/%d low-profile grasps "
                            "(hard_floor=%.3fm, best_clearance=%.3fm)",
                            len(filtered_grasps),
                            len(grasps_final),
                            adaptive_tabletop_hard_floor,
                            diag.tabletop_best_candidate_clearance_m,
                        )
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

        results = []
        width_axis_local = np.asarray(target_width_axis_local, dtype=np.float64)
        for i in range(len(grasps_final)):
            target_width_m, target_width_quality, width_axis_camera = estimate_candidate_target_width(
                object_pc_clean,
                grasps_final[i],
                width_axis_local,
                percentile_low=target_width_percentile_low,
                percentile_high=target_width_percentile_high,
                min_width_m=target_width_min_m,
                max_width_m=target_width_max_m,
            )
            results.append(
                GraspCandidate(
                    pose_4x4=grasps_final[i].astype(np.float32),
                    confidence=float(confidences_final[i]),
                    target_width_m=target_width_m,
                    target_width_quality=target_width_quality,
                    width_axis_camera=width_axis_camera,
                )
            )

        results.sort(key=lambda g: g.confidence, reverse=True)
        return results, diag

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
