"""Schema validation for the robot.grasp_execution configuration block."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _Rule:
    kind: str
    minimum: float | None = None
    maximum: float | None = None
    exclusive_minimum: bool = False
    choices: frozenset[str] = frozenset()
    length: int | None = None
    allow_empty: bool = False
    nonzero: bool = False


def _bool() -> _Rule:
    return _Rule("bool")


def _string(*, allow_empty: bool = False) -> _Rule:
    return _Rule("string", allow_empty=allow_empty)


def _number(
    minimum: float | None = None,
    maximum: float | None = None,
    *,
    exclusive_minimum: bool = False,
) -> _Rule:
    return _Rule(
        "number",
        minimum=minimum,
        maximum=maximum,
        exclusive_minimum=exclusive_minimum,
    )


def _integer(minimum: int | None = None, maximum: int | None = None) -> _Rule:
    return _Rule("integer", minimum=minimum, maximum=maximum)


def _enum(*choices: str) -> _Rule:
    return _Rule("enum", choices=frozenset(choices))


def _vector(*, nonzero: bool = False) -> _Rule:
    return _Rule("vector", length=3, nonzero=nonzero)


_POSITIVE = _number(0.0, exclusive_minimum=True)
_NON_NEGATIVE = _number(0.0)
_UNIT_INTERVAL = _number(0.0, 1.0)
_VELOCITY = _number(0.0, 1.0, exclusive_minimum=True)

_SCHEMA: dict[str, Any] = {
    "enabled": _bool(),
    "action_name": _string(),
    "auto_start_dependencies": _bool(),
    "model_bundle_path": _string(),
    "model_deployment": _string(),
    "planner_service": _string(),
    "verifier_service": _string(),
    "detect_service": _string(),
    "fallback_detect_service": _string(),
    "segment_service": _string(allow_empty=True),
    "ik_service": _string(),
    "fk_service": _string(),
    "joint_state_topic": _string(),
    "move_configuration_service": _string(),
    "base_frame": _string(),
    "ee_frame": _string(),
    "observe_pose": _string(allow_empty=True),
    "timeout_sec": _POSITIVE,
    "ready_timeout_sec": _POSITIVE,
    "verification": _enum("disabled", "optional", "required"),
    "verification_timeout_sec": _POSITIVE,
    "verification_wait_sec": _NON_NEGATIVE,
    "recover_after_close_failure": _bool(),
    "recover_after_retention_failure": _bool(),
    "max_execution_attempts": _integer(0),
    "approach_distance_m": _NON_NEGATIVE,
    "lift_distance_m": _NON_NEGATIVE,
    "observe_velocity_scaling": _VELOCITY,
    "approach_velocity_scaling": _VELOCITY,
    "descend_velocity_scaling": _VELOCITY,
    "probe_lift_velocity_scaling": _VELOCITY,
    "lift_velocity_scaling": _VELOCITY,
    "release_velocity_scaling": _VELOCITY,
    # Robot-specific candidate correction expressed in the planning base frame.
    "candidate_target_offset_base_m": _vector(),
    "observe_settle_sec": _NON_NEGATIVE,
    "open_settle_sec": _NON_NEGATIVE,
    "hold_sec": _NON_NEGATIVE,
    "release_settle_sec": _NON_NEGATIVE,
    "state_wait": {
        "enabled": _bool(),
        "minimum_sec": _NON_NEGATIVE,
        "stable_sec": _NON_NEGATIVE,
        "joint_delta_rad": _NON_NEGATIVE,
        "gripper_tolerance_rad": _NON_NEGATIVE,
        "gripper_joint": _string(),
    },
    "probe_lift_height_m": _NON_NEGATIVE,
    "descend_duration_sec": _POSITIVE,
    "camera": {
        "frame_id": _string(),
        "rgb_topic": _string(),
        "depth_topic": _string(),
        "camera_info_topic": _string(),
    },
    "planner": {
        "confidence_threshold": _UNIT_INTERVAL,
        "grasp_threshold": _UNIT_INTERVAL,
        "timeout_sec": _POSITIVE,
        "debug_output_mode": _enum("none", "diagnostic", "full"),
    },
    "candidate_selection": {
        "min_confidence": _UNIT_INTERVAL,
        "min_point_count": _integer(0),
        "require_collision_free": _bool(),
        "min_contact_z": _number(),
        "min_approach_z": _number(),
        "min_topdown_score": _UNIT_INTERVAL,
        "topdown_min_z": _number(-1.0, 1.0),
        "max_candidates": _integer(0),
        "selection_attempts": _integer(1),
        "retry_settle_sec": _NON_NEGATIVE,
        "confidence_weight": _NON_NEGATIVE,
        "topdown_weight": _NON_NEGATIVE,
    },
    "ik": {
        "group_name": _string(),
        # MoveIt solver budget; keep it small so LMA cannot discard the seed.
        "timeout_sec": _POSITIVE,
        # Service wait budget, independent of the solver budget above.
        "rpc_timeout_sec": _POSITIVE,
        "avoid_collisions": _bool(),
        "check_orientation": _bool(),
        "worker_count": _integer(0, 8),
        "worker_namespace_prefix": _string(allow_empty=True),
        "auto_start_workers": _bool(),
    },
    "contact_compensation": {
        "enabled": _bool(),
        "xy_tolerance_m": _NON_NEGATIVE,
        "max_iterations": _integer(0),
        "max_correction_m": _NON_NEGATIVE,
        "max_z_error_m": _NON_NEGATIVE,
    },
    "prepared_candidate_scoring": {
        "enabled": _bool(),
        "fixed_finger_envelope_weight": _NON_NEGATIVE,
        "contact_xy_weight": _NON_NEGATIVE,
        "contact_z_weight": _NON_NEGATIVE,
        "confidence_weight": _NON_NEGATIVE,
        "centroid_distance_weight": _NON_NEGATIVE,
        "robust_gap_headroom_weight": _NON_NEGATIVE,
        "contact_xy_scale_m": _POSITIVE,
        "contact_z_scale_m": _POSITIVE,
        "centroid_distance_scale_m": _POSITIVE,
        "robust_gap_headroom_scale_m": _POSITIVE,
        "fixed_finger_gap_sigma_m": _POSITIVE,
        "missing_fixed_finger_envelope_score": _UNIT_INTERVAL,
        "missing_robust_gap_headroom_score": _UNIT_INTERVAL,
        "fixed_finger_score_weight": _UNIT_INTERVAL,
        # Read from this block by the preparation phase, not from target_gripper.
        "reliable_max_opening_m": _POSITIVE,
        "moving_finger_min_clearance_m": _NON_NEGATIVE,
    },
    "contact_realign": {
        "enabled": _bool(),
        "tolerance_m": _NON_NEGATIVE,
        "max_iterations": _integer(0),
        "max_correction_m": _NON_NEGATIVE,
        "settle_sec": _NON_NEGATIVE,
        "pregrasp_enabled": _bool(),
        "pregrasp_clearance_m": _NON_NEGATIVE,
    },
    "pose_diagnostics": {
        "enabled": _bool(),
        "settle_sec": _NON_NEGATIVE,
        "grasp_warn_threshold_m": _NON_NEGATIVE,
        "grasp_realign_log_threshold_m": _NON_NEGATIVE,
        "grasp_abort_log_threshold_m": _NON_NEGATIVE,
    },
    "frame_diagnostics": {"enabled": _bool()},
    "planner_node": {
        "device": _string(),
        "inference_backend": _enum("local_cuda", "remote_310p", "ascend_local"),
        "remote_310p_host": _string(),
        "remote_310p_port": _integer(1, 65535),
        "remote_310p_username": _string(),
        "remote_310p_password_env": _string(),
        "remote_310p_root": _string(),
        "remote_310p_timeout_sec": _POSITIVE,
        "ascend_local_manifest_path": _string(),
        "ascend_local_deployment_name": _string(),
        "ascend_local_device_id": _integer(0),
        "ascend_local_random_seed": _integer(-1),
        "startup_warmup": _bool(),
        "save_debug_outputs": _bool(),
        "debug_output_dir": _string(),
        "enable_collision_filter": _bool(),
        "enable_tabletop_filter": _bool(),
        "enable_source_gripper_tabletop_sweep": _bool(),
        "require_tabletop_filter": _bool(),
        "tabletop_filter_mode": _enum("disabled", "diagnostic", "strict"),
        "tabletop_clearance": _number(),
        "tabletop_pregrasp_distance": _NON_NEGATIVE,
        "enable_object_cloud_completion": _bool(),
        "object_cloud_completion_mode": _string(),
        "enable_object_cloud_prismatic_extrude": _bool(),
        "sync_max_age_sec": _POSITIVE,
        "input_buffer_size": _integer(1),
        "num_grasps": _integer(1),
        # Non-positive means retain the complete threshold-qualified pool.
        "topk_num_grasps": _integer(-1),
        "host_runtime": {
            "omp_threads": _integer(1),
            "blas_threads": _integer(1),
        },
    },
    "verifier_node": {
        "gripper_joint": _string(),
        "joint_state_topic": _string(),
        "joint_current_topic": _string(),
        "wrist_depth_topic": _string(),
        "gripper_closed_position": _number(),
        "gripper_contact_min_opening": _NON_NEGATIVE,
        "gripper_no_contact_max_opening": _NON_NEGATIVE,
        "current_contact_threshold_a": _NON_NEGATIVE,
    },
    "source_gripper": _string(),
    "source_contact_point": _vector(),
    "adapter": {"source_to_ee_rpy": _vector()},
    "execution_scoring": {
        "centroid_source": _enum("surface", "volume"),
        "confidence_weight": _NON_NEGATIVE,
        "contact_distance_weight": _NON_NEGATIVE,
        "contact_distance_scale_m": _POSITIVE,
        "topdown_weight": _NON_NEGATIVE,
    },
    "target_geometry": {
        "tabletop_filter": _bool(),
        "tabletop_clearance_m": _number(),
        "tabletop_sweep_steps": _integer(1),
        "mesh_package": _string(),
        "mesh_directory": _string(),
    },
    "target_gripper": {
        "type": _string(),
        "ee_frame": _string(),
        "fixed_finger_contact_ee": _vector(),
        "closing_axis_ee": _vector(nonzero=True),
        "ik_orientation_guard": {
            "enabled": _bool(),
            "approach_axis_ee": _vector(nonzero=True),
            "closing_axis_180_symmetric": _bool(),
            "joint5_constraints_enabled": _bool(),
            "joint5_home_max_delta_rad": _number(0.0, math.pi, exclusive_minimum=True),
            "joint5_limit_epsilon_rad": _NON_NEGATIVE,
            "joint5_stage_continuity": _bool(),
            "joint5_stage_max_delta_rad": _number(0.0, math.pi, exclusive_minimum=True),
            "max_approach_error_deg": _number(0.0, 180.0),
            "max_closing_error_deg": _number(0.0, 180.0),
            "moveit_orientation_search": {
                "enabled": _bool(),
                "approach_tolerance_deg": _number(0.0, 180.0),
                "free_rotation_tolerance_deg": _number(0.0, 180.0),
                "constraint_weight": _POSITIVE,
                "max_attempts": _integer(1, 8),
            },
        },
        "fixed_finger_margin_m": _NON_NEGATIVE,
        "fixed_finger_margin_max_m": _NON_NEGATIVE,
        "fixed_finger_margin_width_ref_m": _NON_NEGATIVE,
        "fixed_finger_margin_width_gain": _NON_NEGATIVE,
        "fixed_finger_base_side": {
            "enabled": _bool(),
            "reference_point_base": _vector(),
            "min_alignment_cos": _number(-1.0, 1.0),
        },
        "fixed_finger_robust_gap": {
            "enabled": _bool(),
            "max_target_gap_deficit_m": _NON_NEGATIVE,
            "measurement_tolerance_m": _NON_NEGATIVE,
        },
        "width_clearance_m": _NON_NEGATIVE,
        "min_width_m": _NON_NEGATIVE,
        "max_width_m": _POSITIVE,
        "fallback_width_m": _NON_NEGATIVE,
        "width_quality_min": _UNIT_INTERVAL,
    },
}


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int | float) and math.isfinite(float(value))


def _number_value(value: Any) -> float | None:
    return float(value) if _is_number(value) else None


def _validate_rule(value: Any, path: str, rule: _Rule, errors: list[str]) -> None:
    if rule.kind == "bool":
        if not isinstance(value, bool):
            errors.append(f"{path} must be a boolean")
        return
    if rule.kind == "string":
        if not isinstance(value, str) or (not rule.allow_empty and not value.strip()):
            errors.append(f"{path} must be a{' non-empty' if not rule.allow_empty else ''} string")
        return
    if rule.kind == "enum":
        if not isinstance(value, str) or value not in rule.choices:
            errors.append(f"{path} must be one of {sorted(rule.choices)}")
        return
    if rule.kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append(f"{path} must be an integer")
            return
        numeric = float(value)
    elif rule.kind == "number":
        if not _is_number(value):
            errors.append(f"{path} must be a finite number")
            return
        numeric = float(value)
    elif rule.kind == "vector":
        if (
            not isinstance(value, list | tuple)
            or len(value) != rule.length
            or not all(_is_number(item) for item in value)
        ):
            errors.append(f"{path} must be a {rule.length}-element finite numeric vector")
            return
        if rule.nonzero and math.sqrt(sum(float(item) ** 2 for item in value)) <= 1e-9:
            errors.append(f"{path} must not be a zero vector")
        return
    else:
        raise AssertionError(f"unsupported validation rule: {rule.kind}")

    if rule.minimum is not None:
        below_minimum = numeric <= rule.minimum if rule.exclusive_minimum else numeric < rule.minimum
        if below_minimum:
            comparator = "greater than" if rule.exclusive_minimum else "greater than or equal to"
            errors.append(f"{path} must be {comparator} {rule.minimum}")
    if rule.maximum is not None and numeric > rule.maximum:
        errors.append(f"{path} must be less than or equal to {rule.maximum}")


def _validate_mapping(value: Any, path: str, schema: dict[str, Any], errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{path} must be a mapping")
        return
    unknown = sorted(set(value) - set(schema))
    if unknown:
        errors.append(f"{path} contains unknown key(s): {', '.join(unknown)}")
    for key, item in value.items():
        rule = schema.get(key)
        if rule is None:
            continue
        item_path = f"{path}.{key}"
        if isinstance(rule, dict):
            _validate_mapping(item, item_path, rule, errors)
        else:
            _validate_rule(item, item_path, rule, errors)


def validate_grasp_execution_config(value: Any) -> list[str]:
    """Return schema and range errors for a grasp_execution mapping."""
    if value is None:
        return []
    errors: list[str] = []
    _validate_mapping(value, "grasp_execution", _SCHEMA, errors)
    if not isinstance(value, dict):
        return errors

    ik = value.get("ik")
    if isinstance(ik, dict):
        worker_count = ik.get("worker_count")
        positive_worker_count = (
            isinstance(worker_count, int) and not isinstance(worker_count, bool) and worker_count > 0
        )
        if positive_worker_count and not str(ik.get("worker_namespace_prefix", "")).strip("/"):
            errors.append("grasp_execution.ik.worker_namespace_prefix must not be empty when worker_count is positive")

    prepared_scoring = value.get("prepared_candidate_scoring")
    if isinstance(prepared_scoring, dict):
        reliable_opening = _number_value(prepared_scoring.get("reliable_max_opening_m"))
        moving_clearance = _number_value(prepared_scoring.get("moving_finger_min_clearance_m"))
        if reliable_opening is not None and moving_clearance is not None and moving_clearance >= reliable_opening:
            errors.append(
                "grasp_execution.prepared_candidate_scoring.moving_finger_min_clearance_m "
                "must be less than reliable_max_opening_m"
            )

    target = value.get("target_gripper")
    if isinstance(target, dict):
        minimum = _number_value(target.get("min_width_m"))
        maximum = _number_value(target.get("max_width_m"))
        fallback = _number_value(target.get("fallback_width_m"))
        if minimum is not None and maximum is not None and minimum > maximum:
            errors.append("grasp_execution.target_gripper.min_width_m must not exceed max_width_m")
        if fallback is not None and minimum is not None and fallback < minimum:
            errors.append("grasp_execution.target_gripper.fallback_width_m must be at least min_width_m")
        if fallback is not None and maximum is not None and fallback > maximum:
            errors.append("grasp_execution.target_gripper.fallback_width_m must not exceed max_width_m")
        base_margin = _number_value(target.get("fixed_finger_margin_m"))
        max_margin = _number_value(target.get("fixed_finger_margin_max_m"))
        if base_margin is not None and max_margin is not None and base_margin > max_margin:
            errors.append(
                "grasp_execution.target_gripper.fixed_finger_margin_m must not exceed fixed_finger_margin_max_m"
            )
        orientation_guard = target.get("ik_orientation_guard")
        if isinstance(orientation_guard, dict):
            search = orientation_guard.get("moveit_orientation_search")
            if isinstance(search, dict) and bool(search.get("enabled", False)):
                approach_tolerance = _number_value(search.get("approach_tolerance_deg"))
                hard_approach_limit = _number_value(orientation_guard.get("max_approach_error_deg"))
                if (
                    approach_tolerance is not None
                    and hard_approach_limit is not None
                    and approach_tolerance > hard_approach_limit
                ):
                    errors.append(
                        "grasp_execution.target_gripper.ik_orientation_guard.moveit_orientation_search."
                        "approach_tolerance_deg must not exceed max_approach_error_deg"
                    )
                approach_axis = orientation_guard.get("approach_axis_ee")
                if (
                    isinstance(approach_axis, list | tuple)
                    and len(approach_axis) == 3
                    and all(_is_number(item) for item in approach_axis)
                ):
                    norm = math.sqrt(sum(float(item) ** 2 for item in approach_axis))
                    normalized = [float(item) / norm for item in approach_axis] if norm > 1e-9 else []
                    if normalized:
                        dominant = max(range(3), key=lambda index: abs(normalized[index]))
                        cardinal = abs(abs(normalized[dominant]) - 1.0) <= 1e-6 and all(
                            abs(item) <= 1e-6 for index, item in enumerate(normalized) if index != dominant
                        )
                        if not cardinal:
                            errors.append(
                                "grasp_execution.target_gripper.ik_orientation_guard.moveit_orientation_search "
                                "requires approach_axis_ee to be an EE-frame cardinal axis"
                            )

    planner_node = value.get("planner_node")
    if isinstance(planner_node, dict):
        planner_backend = planner_node.get("inference_backend")
        if planner_backend == "ascend_local" and not str(planner_node.get("ascend_local_manifest_path", "")).strip():
            errors.append(
                "grasp_execution.planner_node.ascend_local_manifest_path must not be empty when using ascend_local"
            )
        total = _number_value(planner_node.get("num_grasps"))
        topk = _number_value(planner_node.get("topk_num_grasps"))
        if total is not None and topk is not None and topk > total:
            errors.append("grasp_execution.planner_node.topk_num_grasps must not exceed num_grasps")

    return errors
