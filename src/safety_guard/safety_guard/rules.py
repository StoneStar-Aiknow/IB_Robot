"""Pure safety rules for the embodied minimal closure."""

import math
from typing import Any

from embodied_common.json_utils import load_json_mapping
from embodied_common.skill_templates import SUPPORTED_PRIMITIVES, get_skill_templates

__all__ = [
    "SUPPORTED_PRIMITIVES",
    "get_skill_templates",
    "load_json_mapping",
    "validate_xyz_within_workspace",
    "validate_pose_within_workspace",
    "validate_skill_request",
    "validate_primitive_request",
]


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _extract_pose_xyz(pose: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    position = pose.get("position", {})
    return position.get("x"), position.get("y"), position.get("z")


def validate_xyz_within_workspace(x: Any, y: Any, z: Any, workspace: dict[str, Any]) -> tuple[bool, str]:
    for axis, value in (("x", x), ("y", y), ("z", z)):
        if not _is_finite_number(value):
            return False, f"pose {axis} must be finite"
        limits = workspace.get(axis)
        if limits is None:
            continue
        if not isinstance(limits, list) or len(limits) != 2:
            return False, f"workspace axis {axis} must be [min, max]"
        if not all(_is_finite_number(limit) for limit in limits):
            return False, f"workspace axis {axis} limits must be finite"
        min_value = float(limits[0])
        max_value = float(limits[1])
        if min_value > max_value:
            return False, f"workspace axis {axis} minimum must not exceed maximum"
        if float(value) < min_value or float(value) > max_value:
            return False, f"pose is outside workspace on {axis}: {value}"
    return True, ""


def validate_pose_within_workspace(pose: dict[str, Any], workspace: dict[str, Any]) -> tuple[bool, str]:
    x, y, z = _extract_pose_xyz(pose)
    if x is None or y is None or z is None:
        return False, "pose is missing x/y/z position"

    return validate_xyz_within_workspace(x, y, z, workspace)


def _validate_joint_targets(
    joint_positions: dict[str, Any],
    arm_joint_names: list[str] | None,
    joint_limits: dict[str, Any] | None,
    label: str,
) -> tuple[bool, str]:
    for joint_name, joint_position in joint_positions.items():
        if not _is_finite_number(joint_position):
            return False, f"joint target for {joint_name} in {label} must be finite"

    resolved_arm_joint_names = list(arm_joint_names or [])
    if resolved_arm_joint_names:
        unknown_joint_names = sorted(set(joint_positions) - set(resolved_arm_joint_names))
        if unknown_joint_names:
            return False, f"unknown arm joints in {label}: {', '.join(unknown_joint_names)}"

    resolved_joint_limits = joint_limits or {}
    if resolved_arm_joint_names and resolved_joint_limits:
        missing_joint_limits = sorted(joint for joint in resolved_arm_joint_names if joint not in resolved_joint_limits)
        if missing_joint_limits:
            return False, f"missing joint limits for arm joints: {', '.join(missing_joint_limits)}"
        for joint_name, joint_position in joint_positions.items():
            limits = resolved_joint_limits[joint_name]
            if not _is_finite_number(limits.get("min")) or not _is_finite_number(limits.get("max")):
                return False, f"joint limits for {joint_name} must be finite"
            min_position = float(limits["min"])
            max_position = float(limits["max"])
            if min_position > max_position:
                return False, f"joint limits for {joint_name} are reversed"
            position = float(joint_position)
            if position < min_position or position > max_position:
                return False, f"joint target outside limits for {joint_name}: {position}"

    return True, ""


def validate_skill_request(
    skill_name: str,
    target_name: str,
    place_name: str,
    motion_direction: str,
    motion_distance: float,
    named_poses: dict[str, Any],
    named_targets: dict[str, Any],
    skill_templates: dict[str, Any] | None = None,
    arm_joint_names: list[str] | None = None,
    joint_limits: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    templates = get_skill_templates(skill_templates)
    template = templates.get(skill_name)
    if template is None:
        return False, f"unsupported skill: {skill_name}"

    primitive_sequence = template.get("primitive_sequence", [])
    if not isinstance(primitive_sequence, list) or not primitive_sequence:
        return False, f"skill template '{skill_name}' has no primitive_sequence"

    valid_directions = {"forward", "backward", "left", "right", "up", "down"}
    for step in primitive_sequence:
        if not isinstance(step, dict):
            return False, f"skill template '{skill_name}' contains a non-object step"

        primitive_name = str(step.get("primitive_name", "")).strip()
        if primitive_name not in SUPPORTED_PRIMITIVES:
            return False, f"skill template '{skill_name}' uses unsupported primitive: {primitive_name}"

        if primitive_name == "move_to_named_pose":
            pose_name = str(step.get("pose_name", "")).strip()
            if step.get("place_name_from_request"):
                if not place_name:
                    return False, "place_name is required"
                if place_name not in named_poses:
                    return False, f"unknown place pose: {place_name}"
                continue

            target_pose_key = str(step.get("target_pose_key", "")).strip()
            if target_pose_key:
                if target_name not in named_targets:
                    return False, f"unknown target: {target_name}"
                target_cfg = named_targets[target_name]
                resolved_pose_name = str(target_cfg.get(target_pose_key, "")).strip()
                if not resolved_pose_name:
                    return False, f"target '{target_name}' is missing pose key '{target_pose_key}'"
                if resolved_pose_name not in named_poses:
                    return False, f"target pose '{resolved_pose_name}' is undefined"
                continue

            if not pose_name:
                return False, f"skill template '{skill_name}' is missing pose reference"
            if pose_name not in named_poses:
                return False, f"unknown pose: {pose_name}"
            continue

        if primitive_name == "move_relative_ee":
            resolved_direction = str(step.get("motion_direction", "")).strip()
            if step.get("motion_direction_from_request"):
                resolved_direction = motion_direction
            if resolved_direction not in valid_directions and resolved_direction != "":
                return False, f"unsupported motion direction: {resolved_direction}"

            resolved_distance = step.get("motion_distance", 0.0)
            if step.get("motion_distance_from_request"):
                resolved_distance = motion_distance
            if not _is_finite_number(resolved_distance):
                return False, "motion_distance must be finite"
            resolved_distance = float(resolved_distance)
            if resolved_distance < 0.0:
                return False, "motion_distance must be non-negative"
            if resolved_direction and resolved_distance == 0.0:
                return False, "motion_distance must be greater than zero"

        if primitive_name in {"rotate_gripper_cw", "rotate_gripper_ccw"}:
            # Rotation angle is carried on motion_distance for these primitives.
            # Reject NaN/inf/negative so an LLM or rule parser cannot dispatch a
            # harmful wrist rotation through the safety layer. Absolute upper
            # bound is enforced by the controller's joint limits, not here.
            resolved_distance = step.get("motion_distance", 0.0)
            if step.get("motion_distance_from_request"):
                resolved_distance = motion_distance
            if not _is_finite_number(resolved_distance):
                return False, "rotation angle must be finite"
            resolved_distance = float(resolved_distance)
            if resolved_distance < 0.0:
                return False, "rotation angle must be non-negative"
            if resolved_distance == 0.0:
                return False, "rotation angle must be greater than zero"

        if primitive_name == "move_to_joint_positions":
            joint_position_offsets = step.get("joint_position_offsets", {})
            joint_position_targets = step.get("joint_positions", {})
            has_offsets = bool(joint_position_offsets)
            has_targets = bool(joint_position_targets)
            if has_offsets and has_targets:
                return False, "move_to_joint_positions cannot define both joint_position_offsets and joint_positions"
            if not has_offsets and not has_targets:
                return False, "move_to_joint_positions requires joint_position_offsets or joint_positions"
            joint_map = joint_position_targets if has_targets else joint_position_offsets
            if not isinstance(joint_map, dict):
                return False, "joint target map must be an object"
            allowed, reason = _validate_joint_targets(joint_map, arm_joint_names, joint_limits, "joint target")
            if not allowed:
                return allowed, reason
            duration_sec = step.get("duration_sec", 0.0)
            if not _is_finite_number(duration_sec):
                return False, "duration_sec must be finite"
            if float(duration_sec) < 0.0:
                return False, "duration_sec must be non-negative"

        if primitive_name == "move_through_joint_positions":
            raw_waypoints = step.get("joint_waypoints", [])
            if not isinstance(raw_waypoints, list) or not raw_waypoints:
                return False, "move_through_joint_positions requires joint_waypoints"
            waypoint_duration_sec = step.get("waypoint_duration_sec", 0.0)
            if not _is_finite_number(waypoint_duration_sec):
                return False, "waypoint_duration_sec must be finite"
            if float(waypoint_duration_sec) <= 0.0:
                return False, "waypoint_duration_sec must be greater than zero"
            for waypoint_index, waypoint in enumerate(raw_waypoints):
                if not isinstance(waypoint, dict):
                    return False, f"joint waypoint {waypoint_index} must be an object"
                joint_positions = waypoint.get("joint_positions", {})
                if not isinstance(joint_positions, dict) or not joint_positions:
                    return False, f"joint waypoint {waypoint_index} must define joint_positions"
                allowed, reason = _validate_joint_targets(
                    joint_positions,
                    arm_joint_names,
                    joint_limits,
                    f"joint waypoint {waypoint_index}",
                )
                if not allowed:
                    return allowed, reason

    return True, ""


def validate_primitive_request(
    primitive_name: str,
    pose_name: str,
    relative_dx: float,
    relative_dy: float,
    relative_dz: float,
    target_x: float,
    target_y: float,
    target_z: float,
    gripper_position: float,
    named_poses: dict[str, Any],
    workspace: dict[str, Any],
    joint_names: list[str] | None = None,
    joint_positions: list[float] | None = None,
    joint_waypoints: list[float] | None = None,
    joint_waypoint_count: int = 0,
    arm_joint_names: list[str] | None = None,
    joint_limits: dict[str, Any] | None = None,
    primitive_duration_sec: float = 0.0,
    waypoint_duration_sec: float = 0.0,
) -> tuple[bool, str]:
    if primitive_name not in {
        "move_to_named_pose",
        "move_relative_ee",
        "move_to_joint_positions",
        "move_through_joint_positions",
        "open_gripper",
        "close_gripper",
        "rotate_gripper_cw",
        "rotate_gripper_ccw",
    }:
        return False, f"unsupported primitive: {primitive_name}"

    if primitive_name == "move_to_named_pose":
        pose = named_poses.get(pose_name)
        if pose is None:
            return False, f"unknown pose: {pose_name}"
        return validate_pose_within_workspace(pose, workspace)

    if primitive_name == "move_relative_ee":
        for field_name, value in (
            ("relative_dx", relative_dx),
            ("relative_dy", relative_dy),
            ("relative_dz", relative_dz),
        ):
            if not _is_finite_number(value):
                return False, f"{field_name} must be finite"
        return validate_xyz_within_workspace(target_x, target_y, target_z, workspace)

    if primitive_name == "move_to_joint_positions":
        resolved_joint_names = list(joint_names or [])
        resolved_joint_positions = list(joint_positions or [])
        if not resolved_joint_names:
            return False, "joint_names are required for move_to_joint_positions"
        if len(resolved_joint_names) != len(resolved_joint_positions):
            return False, "joint_names and joint_positions must have the same length"
        if not _is_finite_number(primitive_duration_sec):
            return False, "primitive_duration_sec must be finite"
        resolved_duration_sec = float(primitive_duration_sec) or 0.4
        if float(resolved_duration_sec) <= 0.0:
            return False, "primitive_duration_sec must be greater than zero"
        if arm_joint_names and resolved_joint_names != list(arm_joint_names):
            return False, "joint target primitive must command the full arm joint list in configured order"
        joint_map = dict(zip(resolved_joint_names, resolved_joint_positions, strict=False))
        return _validate_joint_targets(joint_map, arm_joint_names, joint_limits, "joint target")

    if primitive_name == "move_through_joint_positions":
        resolved_joint_names = list(joint_names or [])
        resolved_joint_waypoints = list(joint_waypoints or [])
        resolved_waypoint_count = int(joint_waypoint_count or 0)
        if not resolved_joint_names:
            return False, "joint_names are required for move_through_joint_positions"
        if resolved_waypoint_count <= 0:
            return False, "joint_waypoint_count must be greater than zero"
        if len(resolved_joint_waypoints) != len(resolved_joint_names) * resolved_waypoint_count:
            return False, "joint_waypoints length must equal joint_names length times joint_waypoint_count"
        if not _is_finite_number(waypoint_duration_sec):
            return False, "waypoint_duration_sec must be finite"
        if float(waypoint_duration_sec) <= 0.0:
            return False, "waypoint_duration_sec must be greater than zero"
        if arm_joint_names and resolved_joint_names != list(arm_joint_names):
            return False, "joint waypoint primitive must command the full arm joint list in configured order"
        for waypoint_index in range(resolved_waypoint_count):
            start = waypoint_index * len(resolved_joint_names)
            waypoint_positions = resolved_joint_waypoints[start : start + len(resolved_joint_names)]
            joint_map = dict(zip(resolved_joint_names, waypoint_positions, strict=False))
            allowed, reason = _validate_joint_targets(joint_map, arm_joint_names, joint_limits, "joint waypoint")
            if not allowed:
                return allowed, reason
        return True, ""

    if primitive_name in {"rotate_gripper_cw", "rotate_gripper_ccw"} and not _is_finite_number(relative_dz):
        return False, "relative_dz must be finite for gripper rotation"

    if not _is_finite_number(gripper_position):
        return False, "gripper_position must be finite"
    if gripper_position < 0.0 or gripper_position > 1.0:
        return False, "gripper_position must be in [0.0, 1.0]"

    return True, ""
