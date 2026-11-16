"""Planning-side grasp data types shared across the manipulation_service modules."""

import threading
from dataclasses import dataclass

import numpy as np


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
    tabletop_ransac_iterations: int = 0
    tabletop_ransac_duration_ms: float = 0.0
    point_cloud_preprocess_ms: float = 0.0
    outlier_removal_ms: float = 0.0
    model_inference_ms: float = 0.0
    grasp_postprocess_ms: float = 0.0
    plan_total_ms: float = 0.0
    failure_reason: str = ""
    failure_stage: str = ""
    _execution_table_fit_holder: dict[str, object] | None = None
    _execution_table_fit_thread: threading.Thread | None = None
