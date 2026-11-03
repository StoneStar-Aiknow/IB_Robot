"""Table-plane fitting and the tabletop clearance filter (ROS-free).

``TablePlane`` here is the planning-side, camera-frame RANSAC result
(``normal: ndarray``, ``d``) used for prismatic extrusion, scene hole filling
and the tabletop clearance filter, serialised into the ``PlanGrasp``
``table_plane_*`` / ``execution_table_plane_*`` response fields. It is not
``manipulation_execution.geometry.TablePlane``: that type is the base-frame
execution-side plane rebuilt from those message fields. The two packages
meet at the message, not at a shared Python type.
"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np

from .grasp_types import GraspDiagnostic

logger = logging.getLogger(__name__)
VALID_TABLETOP_FILTER_MODES = {"strict", "adaptive", "soft", "diagnostic"}
DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP = False

_EXECUTION_TABLE_SCENE_MAX_POINTS = 30000
_EXECUTION_TABLE_OBJECT_MAX_POINTS = 16000
_RANSAC_PLANE_CHUNK_SIZE = 64
_RANSAC_CONFIDENCE = 0.999


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
    iterations_evaluated: int = 0


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

    attempts = 0
    iterations_evaluated = 0
    best_count = 0
    best_normal = None
    best_d = 0.0
    while attempts < max_iterations:
        candidate_normals = []
        candidate_offsets = []
        while attempts < max_iterations and len(candidate_normals) < _RANSAC_PLANE_CHUNK_SIZE:
            attempts += 1
            tri = pts_work[rng.choice(len(pts_work), 3, replace=False)]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n = n / norm
            candidate_normals.append(n)
            candidate_offsets.append(-n @ tri[0])
        if not candidate_normals:
            continue

        normals = np.asarray(candidate_normals)
        offsets = np.asarray(candidate_offsets)
        distances = np.abs(pts_work @ normals.T + offsets)
        counts = np.sum(distances < distance_threshold, axis=0)
        batch_best_index = int(np.argmax(counts))
        batch_best_count = int(counts[batch_best_index])
        if batch_best_count > best_count:
            best_count = batch_best_count
            best_normal = normals[batch_best_index]
            best_d = float(offsets[batch_best_index])
        iterations_evaluated += len(normals)

        best_ratio = best_count / len(pts_work)
        all_inlier_sample_probability = best_ratio**3
        if all_inlier_sample_probability >= 1.0:
            break
        if all_inlier_sample_probability > 0.0:
            required_iterations = int(np.ceil(np.log1p(-_RANSAC_CONFIDENCE) / np.log1p(-all_inlier_sample_probability)))
            if iterations_evaluated >= required_iterations:
                break

    if best_normal is None or best_count <= 0:
        return TablePlaneFit(
            None,
            failure_reason="no non-degenerate table plane sample found",
            iterations_evaluated=iterations_evaluated,
        )
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
            iterations_evaluated=iterations_evaluated,
        )
    center = inlier_pts.mean(axis=0)
    _, _, vh = np.linalg.svd(inlier_pts - center, full_matrices=False)
    normal = vh[-1]
    d = -normal @ center
    if positive_reference is not None and positive_reference @ normal + d < 0:
        normal = -normal
        d = -d
    logger.info(
        "Table plane: n=[%.4f,%.4f,%.4f] d=%.4f inlier_ratio=%.3f ransac_iterations=%d/%d",
        normal[0],
        normal[1],
        normal[2],
        d,
        inlier_ratio,
        iterations_evaluated,
        max_iterations,
    )
    return TablePlaneFit(
        TablePlane(normal=normal, d=d, inlier_ratio=inlier_ratio),
        best_inlier_ratio=inlier_ratio,
        iterations_evaluated=iterations_evaluated,
    )


def sample_execution_table_points(points: np.ndarray | None, max_points: int) -> np.ndarray:
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
    scene_sample = sample_execution_table_points(scene_points, _EXECUTION_TABLE_SCENE_MAX_POINTS)
    object_sample = sample_execution_table_points(object_points, _EXECUTION_TABLE_OBJECT_MAX_POINTS)
    positive_reference = object_sample.mean(axis=0) if len(object_sample) else None
    return fit_table_plane_ransac(
        scene_sample,
        positive_reference=positive_reference,
        distance_threshold=0.006,
        min_inlier_ratio=0.15,
    )


def start_execution_table_fit(diagnostic: GraspDiagnostic) -> None:
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


def join_execution_table_fit(diagnostic: GraspDiagnostic) -> TablePlaneFit:
    thread = getattr(diagnostic, "_execution_table_fit_thread", None)
    holder = getattr(diagnostic, "_execution_table_fit_holder", None)
    if thread is not None and isinstance(holder, dict):
        thread.join()
        error = holder.get("error")
        if isinstance(error, Exception):
            raise error
        fit = holder.get("fit")
        if isinstance(fit, TablePlaneFit):
            return fit

    object_source = diagnostic.object_pc_after_completion
    if object_source is None:
        object_source = diagnostic.object_pc_raw
    return fit_execution_table_plane(diagnostic.scene_pc_after_completion, object_source)


def tabletop_object_height_stats(object_pc: np.ndarray, table: TablePlane) -> tuple[float, float, float, float]:
    signed = object_pc @ table.normal + table.d
    signed = signed[np.isfinite(signed)]
    if len(signed) == 0:
        return 0.0, 0.0, 0.0, 0.0
    p05, median, p95 = np.quantile(signed, [0.05, 0.50, 0.95])
    return float(max(0.0, p95)), float(p05), float(median), float(p95)


def resolve_tabletop_filter_params(
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


def resolve_tabletop_retry_clearances(
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


def tabletop_clearance_sweep(
    grasps: np.ndarray,
    confidences: np.ndarray,
    *,
    table: TablePlane,
    clearance: float,
    pregrasp_distance: float,
    pregrasp_steps: int,
    collision_vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    verts_local = collision_vertices
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


@dataclass(frozen=True)
class TabletopFilterParams:
    """Every parameter the tabletop stage needs, frozen by the caller (D2)."""

    mode: str
    enable_source_gripper_tabletop_sweep: bool
    require_tabletop_filter: bool
    tabletop_min_inlier_ratio: float
    tabletop_clearance: float
    tabletop_pregrasp_distance: float
    tabletop_pregrasp_steps: int
    adaptive_tabletop_low_profile_height: float
    adaptive_tabletop_clearance_min: float
    adaptive_tabletop_clearance_max: float
    adaptive_tabletop_clearance_height_ratio: float
    adaptive_tabletop_pregrasp_min: float
    adaptive_tabletop_pregrasp_height_ratio: float
    adaptive_tabletop_hard_floor: float
    adaptive_tabletop_auto_tune: bool
    adaptive_tabletop_retry_clearances: tuple[float, ...] | list[float]
    adaptive_tabletop_retry_pregrasp_height_ratio: float
    adaptive_tabletop_retry_hard_floor: float


def run_tabletop_filter_stage(
    *,
    grasps_final: np.ndarray,
    confidences_final: np.ndarray,
    scene_pc: np.ndarray | None,
    shared_table_fit: TablePlaneFit,
    object_pc_clean: np.ndarray,
    diag: GraspDiagnostic,
    params: TabletopFilterParams,
    sweep_fn,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Run the tabletop clearance stage; the tabletop-filter body of plan_grasps.

    Returns ``(None, None)`` exactly where plan_grasps returned ``[], diag``
    early (the caller then returns ``[], diag``); otherwise the filtered
    (grasps, confidences). ``sweep_fn`` is the wrapper's
    ``_filter_tabletop_clearance`` method, bound to the wrapper instance.
    """
    mode = params.mode
    if scene_pc is None or len(scene_pc) == 0:
        if params.require_tabletop_filter:
            diag.failure_stage = "tabletop_filter"
            diag.failure_reason = "tabletop filter required but scene point cloud is empty"
            logger.warning(diag.failure_reason)
            return None, None
        table = None
    else:
        fit = shared_table_fit
        if fit.plane is not None and fit.best_inlier_ratio < float(params.tabletop_min_inlier_ratio):
            fit = TablePlaneFit(
                None,
                best_inlier_ratio=fit.best_inlier_ratio,
                failure_reason=(
                    f"best inlier ratio {fit.best_inlier_ratio:.3f} < "
                    f"minimum {float(params.tabletop_min_inlier_ratio):.3f}"
                ),
            )
        table = fit.plane
        diag.tabletop_best_inlier_ratio = fit.best_inlier_ratio
        diag.tabletop_failure_reason = fit.failure_reason

    if table is None:
        if params.require_tabletop_filter:
            diag.failure_stage = "tabletop_filter"
            reason = diag.tabletop_failure_reason or "no acceptable table plane found"
            diag.failure_reason = (
                "tabletop filter required but no table plane found "
                f"({reason}; min_inlier_ratio={params.tabletop_min_inlier_ratio:.3f})"
            )
            logger.warning(diag.failure_reason)
            return None, None
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
        height, p05, median, p95 = tabletop_object_height_stats(object_pc_clean, table)
        diag.tabletop_object_height_m = height
        diag.tabletop_object_height_p05_m = p05
        diag.tabletop_object_height_median_m = median
        diag.tabletop_object_height_p95_m = p95
        clearance_used, pregrasp_used, low_profile = resolve_tabletop_filter_params(
            mode=mode,
            object_height_m=height,
            clearance=params.tabletop_clearance,
            pregrasp_distance=params.tabletop_pregrasp_distance,
            low_profile_height=params.adaptive_tabletop_low_profile_height,
            clearance_min=params.adaptive_tabletop_clearance_min,
            clearance_max=params.adaptive_tabletop_clearance_max,
            clearance_height_ratio=params.adaptive_tabletop_clearance_height_ratio,
            pregrasp_min=params.adaptive_tabletop_pregrasp_min,
            pregrasp_height_ratio=params.adaptive_tabletop_pregrasp_height_ratio,
        )
        diag.tabletop_low_profile = low_profile
        diag.tabletop_clearance_used_m = clearance_used
        diag.tabletop_pregrasp_distance_used_m = pregrasp_used
        diag.tabletop_filter_before = len(grasps_final)
        if params.enable_source_gripper_tabletop_sweep:
            diag.source_gripper_tabletop_sweep_enabled = True
            filtered_grasps, filtered_confidences, clearances = sweep_fn(
                grasps_final,
                confidences_final,
                table=table,
                clearance=clearance_used,
                pregrasp_distance=pregrasp_used,
                pregrasp_steps=params.tabletop_pregrasp_steps,
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
            and params.adaptive_tabletop_auto_tune
            and len(clearances) > 0
        ):
            retry_clearances = resolve_tabletop_retry_clearances(
                params.adaptive_tabletop_retry_clearances,
                initial_clearance=clearance_used,
                clearance_min=params.adaptive_tabletop_clearance_min,
            )
            initial_best_clearance = diag.tabletop_best_candidate_clearance_m
            if initial_best_clearance < params.adaptive_tabletop_retry_hard_floor:
                diag.tabletop_auto_tune_reason = (
                    f"initial_best_clearance {initial_best_clearance:.3f}m < "
                    f"retry_hard_floor {params.adaptive_tabletop_retry_hard_floor:.3f}m"
                )
            elif not retry_clearances:
                diag.tabletop_auto_tune_reason = "no retry clearance below current adaptive clearance"
            else:
                retry_pregrasp = height * params.adaptive_tabletop_retry_pregrasp_height_ratio
                retry_pregrasp = max(params.adaptive_tabletop_pregrasp_min, min(pregrasp_used, retry_pregrasp))
                for retry_clearance in retry_clearances:
                    diag.tabletop_auto_tune_attempts += 1
                    retry_grasps, retry_confidences, retry_candidate_clearances = sweep_fn(
                        grasps_final,
                        confidences_final,
                        table=table,
                        clearance=retry_clearance,
                        pregrasp_distance=retry_pregrasp,
                        pregrasp_steps=params.tabletop_pregrasp_steps,
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
            relaxed_mask = clearances >= params.adaptive_tabletop_hard_floor
            if relaxed_mask.any():
                diag.tabletop_relaxed = True
                relaxed_indices = np.flatnonzero(relaxed_mask)
                order = np.lexsort((-confidences_final[relaxed_indices], -clearances[relaxed_indices]))
                relaxed_indices = relaxed_indices[order]
                filtered_grasps = grasps_final[relaxed_indices]
                filtered_confidences = confidences_final[relaxed_indices]
                logger.warning(
                    "Tabletop soft fallback accepted %d/%d low-profile grasps (hard_floor=%.3fm, best_clearance=%.3fm)",
                    len(filtered_grasps),
                    len(grasps_final),
                    params.adaptive_tabletop_hard_floor,
                    diag.tabletop_best_candidate_clearance_m,
                )
        if mode == "diagnostic":
            diag.tabletop_relaxed = len(filtered_grasps) < len(grasps_final)
            diag.tabletop_filter_after = len(grasps_final)
            if params.enable_source_gripper_tabletop_sweep:
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
            return None, None
    return grasps_final, confidences_final
