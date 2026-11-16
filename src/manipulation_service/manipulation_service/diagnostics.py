"""GraspDiagnostic serialisation to text lines and JSON dicts."""

import numpy as np

from .grasp_types import GraspDiagnostic


def diagnostic_to_lines(diag: GraspDiagnostic, *, detection_confidence: float | None = None) -> list[str]:
    lines = []
    if diag.failure_stage:
        lines.extend([f"failure_stage: {diag.failure_stage}", f"failure_reason: {diag.failure_reason}"])
    lines.append("detection_status: ok")
    if detection_confidence is not None:
        lines.append(f"detection_confidence: {detection_confidence:.4f}")
    lines.extend(
        [
            f"mask_shape: {diag.mask_shape}",
            f"depth_shape: {diag.depth_shape}",
            f"mask_resized_to_depth: {diag.mask_resized}",
            f"mask_pixel_count: {diag.mask_pixel_count}",
            f"valid_depth_pixel_count: {diag.valid_depth_pixel_count}",
            f"valid_depth_in_mask_count: {diag.valid_depth_in_mask_count}",
            f"valid_depth_ratio_in_mask: {diag.valid_depth_ratio_in_mask:.3f}",
            f"object_cloud_completion_enabled: {diag.object_cloud_completion_enabled}",
            f"object_cloud_completion_mode: {diag.object_cloud_completion_mode}",
            f"object_point_count_raw: {diag.object_point_count_raw}",
            f"object_point_count_completed: {diag.object_point_count_completed}",
            f"object_completion_added_count: {diag.object_completion_added_count}",
            f"object_prismatic_extrude_enabled: {diag.object_prismatic_extrude_enabled}",
            f"object_prismatic_extrude_added_count: {diag.object_prismatic_extrude_added_count}",
            f"object_point_count_graspgen_input: {diag.object_point_count_graspgen_input}",
            f"scene_cloud_table_holes_enabled: {diag.scene_cloud_table_holes_enabled}",
            f"scene_point_count_raw: {diag.scene_point_count_raw}",
            f"scene_table_hole_added_count: {diag.scene_table_hole_added_count}",
            f"object_point_count: {diag.object_point_count}",
            f"scene_point_count: {diag.scene_point_count}",
            f"raw_grasp_count: {diag.raw_grasp_count}",
            f"collision_filter: {diag.collision_filter_after}/{diag.collision_filter_before}",
            f"tabletop_plane_found: {diag.tabletop_plane_found}",
            "tabletop_plane_normal: "
            + (
                "unavailable"
                if diag.tabletop_plane_normal is None
                else ",".join(f"{float(value):.6f}" for value in diag.tabletop_plane_normal)
            ),
            f"tabletop_plane_d: {diag.tabletop_plane_d:.6f}",
            f"tabletop_inlier_ratio: {diag.tabletop_inlier_ratio:.3f}",
            f"tabletop_best_inlier_ratio: {diag.tabletop_best_inlier_ratio:.3f}",
            f"tabletop_filter_mode: {diag.tabletop_filter_mode}",
            f"source_gripper_tabletop_sweep_enabled: {diag.source_gripper_tabletop_sweep_enabled}",
            f"tabletop_low_profile: {diag.tabletop_low_profile}",
            f"tabletop_relaxed: {diag.tabletop_relaxed}",
            f"tabletop_object_height_m: {diag.tabletop_object_height_m:.4f}",
            f"tabletop_clearance_used_m: {diag.tabletop_clearance_used_m:.4f}",
            f"tabletop_pregrasp_distance_used_m: {diag.tabletop_pregrasp_distance_used_m:.4f}",
            f"tabletop_best_candidate_clearance_m: {diag.tabletop_best_candidate_clearance_m:.4f}",
            f"tabletop_auto_tuned: {diag.tabletop_auto_tuned}",
            f"tabletop_auto_tune_attempts: {diag.tabletop_auto_tune_attempts}",
            f"tabletop_auto_tune_reason: {diag.tabletop_auto_tune_reason}",
            f"tabletop_ransac_iterations: {diag.tabletop_ransac_iterations}",
            f"timing_tabletop_ransac_ms: {diag.tabletop_ransac_duration_ms:.3f}",
            f"tabletop_filter: {diag.tabletop_filter_after}/{diag.tabletop_filter_before}",
            f"timing_point_cloud_preprocess_ms: {diag.point_cloud_preprocess_ms:.3f}",
            f"timing_outlier_removal_ms: {diag.outlier_removal_ms:.3f}",
            f"timing_model_inference_ms: {diag.model_inference_ms:.3f}",
            f"timing_grasp_postprocess_ms: {diag.grasp_postprocess_ms:.3f}",
            f"timing_plan_total_ms: {diag.plan_total_ms:.3f}",
        ]
    )
    if diag.tabletop_failure_reason:
        lines.append(f"tabletop_failure_reason: {diag.tabletop_failure_reason}")
    return lines


def diagnostic_to_dict(diag: GraspDiagnostic, *, detection_confidence: float | None = None) -> dict:
    data = {
        "failure_stage": diag.failure_stage,
        "failure_reason": diag.failure_reason,
        "mask_shape": diag.mask_shape,
        "depth_shape": diag.depth_shape,
        "mask_resized_to_depth": bool(diag.mask_resized),
        "mask_pixel_count": int(diag.mask_pixel_count),
        "valid_depth_pixel_count": int(diag.valid_depth_pixel_count),
        "valid_depth_in_mask_count": int(diag.valid_depth_in_mask_count),
        "valid_depth_ratio_in_mask": float(diag.valid_depth_ratio_in_mask),
        "object_cloud_completion_enabled": bool(diag.object_cloud_completion_enabled),
        "object_cloud_completion_mode": diag.object_cloud_completion_mode,
        "object_point_count_raw": int(diag.object_point_count_raw),
        "object_point_count_completed": int(diag.object_point_count_completed),
        "object_completion_added_count": int(diag.object_completion_added_count),
        "object_prismatic_extrude_enabled": bool(diag.object_prismatic_extrude_enabled),
        "object_prismatic_extrude_added_count": int(diag.object_prismatic_extrude_added_count),
        "object_point_count_graspgen_input": int(diag.object_point_count_graspgen_input),
        "scene_cloud_table_holes_enabled": bool(diag.scene_cloud_table_holes_enabled),
        "scene_point_count_raw": int(diag.scene_point_count_raw),
        "scene_table_hole_added_count": int(diag.scene_table_hole_added_count),
        "object_point_count": int(diag.object_point_count),
        "scene_point_count": int(diag.scene_point_count),
        "raw_grasp_count": int(diag.raw_grasp_count),
        "collision_filter_before": int(diag.collision_filter_before),
        "collision_filter_after": int(diag.collision_filter_after),
        "tabletop_filter_before": int(diag.tabletop_filter_before),
        "tabletop_filter_after": int(diag.tabletop_filter_after),
        "tabletop_plane_found": bool(diag.tabletop_plane_found),
        "tabletop_plane_normal": (
            None if diag.tabletop_plane_normal is None else [float(value) for value in diag.tabletop_plane_normal]
        ),
        "tabletop_plane_d": float(diag.tabletop_plane_d),
        "tabletop_object_top_xyz": (
            None if diag.tabletop_object_top_xyz is None else [float(value) for value in diag.tabletop_object_top_xyz]
        ),
        "tabletop_inlier_ratio": float(diag.tabletop_inlier_ratio),
        "tabletop_best_inlier_ratio": float(diag.tabletop_best_inlier_ratio),
        "tabletop_failure_reason": diag.tabletop_failure_reason,
        "tabletop_filter_mode": diag.tabletop_filter_mode,
        "source_gripper_tabletop_sweep_enabled": bool(diag.source_gripper_tabletop_sweep_enabled),
        "tabletop_low_profile": bool(diag.tabletop_low_profile),
        "tabletop_relaxed": bool(diag.tabletop_relaxed),
        "tabletop_object_height_m": float(diag.tabletop_object_height_m),
        "tabletop_object_height_p05_m": float(diag.tabletop_object_height_p05_m),
        "tabletop_object_height_median_m": float(diag.tabletop_object_height_median_m),
        "tabletop_object_height_p95_m": float(diag.tabletop_object_height_p95_m),
        "tabletop_clearance_used_m": float(diag.tabletop_clearance_used_m),
        "tabletop_pregrasp_distance_used_m": float(diag.tabletop_pregrasp_distance_used_m),
        "tabletop_best_candidate_clearance_m": float(diag.tabletop_best_candidate_clearance_m),
        "tabletop_worst_candidate_clearance_m": float(diag.tabletop_worst_candidate_clearance_m),
        "tabletop_auto_tuned": bool(diag.tabletop_auto_tuned),
        "tabletop_auto_tune_attempts": int(diag.tabletop_auto_tune_attempts),
        "tabletop_auto_tune_reason": diag.tabletop_auto_tune_reason,
        "tabletop_ransac_iterations": int(diag.tabletop_ransac_iterations),
        "timing_tabletop_ransac_ms": float(diag.tabletop_ransac_duration_ms),
        "timing_point_cloud_preprocess_ms": float(diag.point_cloud_preprocess_ms),
        "timing_outlier_removal_ms": float(diag.outlier_removal_ms),
        "timing_model_inference_ms": float(diag.model_inference_ms),
        "timing_grasp_postprocess_ms": float(diag.grasp_postprocess_ms),
        "timing_plan_total_ms": float(diag.plan_total_ms),
    }
    if detection_confidence is not None:
        data["detection_confidence"] = float(detection_confidence)
    return data


def candidate_debug_record(index: int, candidate) -> dict:
    return {
        "index": index,
        "confidence": round(float(candidate.confidence), 6),
        "collision_free": bool(candidate.collision_free),
        "target_width_m": round(float(candidate.target_width_m), 6),
        "target_width_quality": round(float(candidate.target_width_quality), 3),
        "width_axis_camera": [round(float(v), 6) for v in candidate.width_axis_camera],
        "target_width_min_offset_m": round(float(candidate.target_width_min_offset_m), 6),
        "target_width_max_offset_m": round(float(candidate.target_width_max_offset_m), 6),
        "position_xyz": [round(float(v), 6) for v in candidate.pose_4x4[:3, 3]],
        "pose_4x4_rowmajor": candidate.pose_4x4.flatten().tolist(),
    }


def execution_table_diagnostic_details(
    diag: GraspDiagnostic,
    *,
    execution_table_fit,
    execution_table_parallel: bool,
    execution_table_wait_s: float,
    execution_table_duration_s: float,
    final_grasp_count: int,
    debug_output_mode: str,
    detection_stamp_ns: int,
    depth_stamp_ns: int,
    logger,
) -> list[str]:
    """Build the execution-table diagnostic lines and log the fit summary.

    The line-assembly part of ``GraspPlannerNode._srv_cb``'s execution-table
    stage: the node joins the background fit (re-raising worker errors) and
    resolves the wait/compute durations before calling this; the log text and
    level are unchanged and ``logger`` is the node's ROS logger.
    """
    lines = [
        f"final_grasp_count: {final_grasp_count}",
        f"debug_output_mode: {debug_output_mode}",
        f"detection_stamp_ns: {detection_stamp_ns}",
        f"depth_stamp_ns: {depth_stamp_ns}",
        f"depth_detection_delta_ms: {abs(depth_stamp_ns - detection_stamp_ns) / 1_000_000.0:.3f}",
        f"execution_table_plane_found: {execution_table_fit.plane is not None}",
        f"execution_table_plane_best_inlier_ratio: {execution_table_fit.best_inlier_ratio:.3f}",
        f"execution_table_plane_fit_duration_s: {execution_table_duration_s:.6f}",
        f"execution_table_plane_wait_duration_s: {execution_table_wait_s:.6f}",
        f"execution_table_plane_parallel: {execution_table_parallel}",
        f"execution_table_plane_ransac_iterations: {execution_table_fit.iterations_evaluated}",
    ]
    if execution_table_fit.failure_reason:
        lines.append(f"execution_table_plane_failure_reason: {execution_table_fit.failure_reason}")
    if execution_table_fit.plane is not None and diag.tabletop_plane_found and diag.tabletop_plane_normal is not None:
        normal_dot = float(np.clip(diag.tabletop_plane_normal @ execution_table_fit.plane.normal, -1.0, 1.0))
        plane_angle_deg = float(np.degrees(np.arccos(normal_dot)))
        plane_offset_delta_m = abs(float(diag.tabletop_plane_d) - float(execution_table_fit.plane.d))
        lines.append(f"execution_table_plane_shadow_angle_deg: {plane_angle_deg:.6f}")
        lines.append(f"execution_table_plane_shadow_offset_delta_m: {plane_offset_delta_m:.6f}")
        logger.info(
            f"Execution table plane fit: compute={execution_table_duration_s * 1000.0:.3f}ms "
            f"wait={execution_table_wait_s * 1000.0:.3f}ms parallel={execution_table_parallel}, "
            f"planning delta angle={plane_angle_deg:.4f}deg offset={plane_offset_delta_m * 1000.0:.3f}mm"
        )
    else:
        logger.info(
            f"Execution table plane fit: compute={execution_table_duration_s * 1000.0:.3f}ms "
            f"wait={execution_table_wait_s * 1000.0:.3f}ms parallel={execution_table_parallel}"
        )
    return lines
