"""Pure safety rules for the embodied minimal closure."""

import json
from typing import Any

from skill_library.skill_templates import SUPPORTED_PRIMITIVES, get_skill_templates


def load_json_mapping(raw_value: str) -> dict[str, Any]:
    if not raw_value:
        return {}
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON mapping: {exc}") from exc


def _extract_pose_xyz(pose: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
    position = pose.get("position", {})
    return position.get("x"), position.get("y"), position.get("z")


def validate_xyz_within_workspace(x: float, y: float, z: float, workspace: dict[str, Any]) -> tuple[bool, str]:
    for axis, value in (("x", x), ("y", y), ("z", z)):
        limits = workspace.get(axis)
        if limits is None:
            continue
        if not isinstance(limits, list) or len(limits) != 2:
            return False, f"workspace axis {axis} must be [min, max]"
        if value < float(limits[0]) or value > float(limits[1]):
            return False, f"pose is outside workspace on {axis}: {value}"
    return True, ""


def validate_pose_within_workspace(pose: dict[str, Any], workspace: dict[str, Any]) -> tuple[bool, str]:
    x, y, z = _extract_pose_xyz(pose)
    if x is None or y is None or z is None:
        return False, "pose is missing x/y/z position"

    return validate_xyz_within_workspace(float(x), float(y), float(z), workspace)


def validate_skill_request(
    skill_name: str,
    target_name: str,
    place_name: str,
    motion_direction: str,
    motion_distance: float,
    named_poses: dict[str, Any],
    named_targets: dict[str, Any],
    skill_templates: dict[str, Any] | None = None,
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

            resolved_distance = float(step.get("motion_distance", 0.0) or 0.0)
            if step.get("motion_distance_from_request"):
                resolved_distance = motion_distance
            if resolved_distance < 0.0:
                return False, "motion_distance must be non-negative"
            if resolved_direction and resolved_distance == 0.0:
                return False, "motion_distance must be greater than zero"

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
) -> tuple[bool, str]:
    if primitive_name not in {
        "move_to_named_pose",
        "move_relative_ee",
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
        return validate_xyz_within_workspace(target_x, target_y, target_z, workspace)

    if gripper_position < 0.0 or gripper_position > 1.0:
        return False, "gripper_position must be in [0.0, 1.0]"

    return True, ""
