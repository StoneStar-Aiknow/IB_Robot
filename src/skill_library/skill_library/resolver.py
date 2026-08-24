"""Helpers for resolving skills, targets, and named poses."""

import math
from dataclasses import dataclass, field
from typing import Any

from embodied_common.json_utils import load_json_mapping
from embodied_common.skill_templates import DEFAULT_WAYPOINT_DURATION_SEC, get_skill_templates
from ibrobot_msgs.action import ExecuteNavigation

__all__ = ["PrimitiveSpec", "load_json_mapping", "direction_to_delta", "resolve_skill_primitives"]


@dataclass
class PrimitiveSpec:
    primitive_name: str
    pose_name: str = ""
    gripper_position: float = 0.0
    relative_dx: float = 0.0
    relative_dy: float = 0.0
    relative_dz: float = 0.0
    joint_names: list[str] = field(default_factory=list)
    joint_positions: list[float] = field(default_factory=list)
    joint_waypoints: list[list[float]] = field(default_factory=list)
    duration_sec: float = 0.0
    waypoint_duration_sec: float = 0.0
    navigation_goal: ExecuteNavigation.Goal | None = None


def direction_to_delta(
    direction: str,
    distance: float,
    direction_mapping: dict[str, Any] | None = None,
) -> tuple[float, float, float]:
    mapping = direction_mapping or {
        "forward": [1.0, 0.0, 0.0],
        "backward": [-1.0, 0.0, 0.0],
        "left": [0.0, 1.0, 0.0],
        "right": [0.0, -1.0, 0.0],
        "up": [0.0, 0.0, 1.0],
        "down": [0.0, 0.0, -1.0],
    }
    if not direction:
        return (0.0, 0.0, 0.0)
    if direction not in mapping:
        raise KeyError(f"unsupported motion direction: {direction}")
    axis = mapping[direction]
    return (float(axis[0]) * distance, float(axis[1]) * distance, float(axis[2]) * distance)


def _resolve_pose_name(
    step: dict[str, Any],
    target_name: str,
    place_name: str,
    named_targets: dict[str, Any],
) -> str:
    pose_name = str(step.get("pose_name", "")).strip()
    if pose_name:
        return pose_name

    if step.get("place_name_from_request"):
        normalized_place_name = place_name.strip()
        if not normalized_place_name:
            raise KeyError("place_name is required for this skill template")
        return normalized_place_name

    target_pose_key = str(step.get("target_pose_key", "")).strip()
    if target_pose_key:
        if target_name not in named_targets:
            raise KeyError(f"unknown target: {target_name}")
        resolved_pose_name = str(named_targets[target_name].get(target_pose_key, "")).strip()
        if not resolved_pose_name:
            raise KeyError(f"target '{target_name}' is missing pose key '{target_pose_key}'")
        return resolved_pose_name

    raise KeyError("skill template move_to_named_pose step is missing pose reference")


def _positive_navigation_value(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be a positive finite number")
    return result


def _navigation_coordinate(value: Any, field_name: str) -> float:
    if value is None:
        raise ValueError(f"{field_name} is required for absolute navigation")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return 0.0 if result == 0.0 else result


def resolve_skill_primitives(
    skill_name: str,
    target_name: str,
    place_name: str,
    motion_direction: str,
    motion_distance: float,
    named_targets: dict[str, Any],
    open_position: float,
    closed_position: float,
    skill_templates: dict[str, Any] | None = None,
    direction_mapping: dict[str, Any] | None = None,
    current_joint_positions: dict[str, float] | None = None,
    arm_joint_names: list[str] | None = None,
    *,
    direction: str = "",
    distance: float = 0.0,
    degree: float = 0.0,
    x: float | None = None,
    y: float | None = None,
    yaw: float | None = None,
) -> list[PrimitiveSpec]:
    templates = get_skill_templates(skill_templates)
    template = templates.get(skill_name)
    if template is None:
        raise KeyError(f"unsupported skill: {skill_name}")

    primitive_sequence = template.get("primitive_sequence", [])
    if not isinstance(primitive_sequence, list) or not primitive_sequence:
        raise ValueError(f"skill template '{skill_name}' must define a non-empty primitive_sequence")

    primitives: list[PrimitiveSpec] = []
    initial_gripper_state = str(template.get("initial_gripper_state", "")).strip().lower()
    if initial_gripper_state:
        if initial_gripper_state == "open":
            primitives.append(PrimitiveSpec(primitive_name="open_gripper", gripper_position=open_position))
        elif initial_gripper_state == "closed":
            primitives.append(PrimitiveSpec(primitive_name="close_gripper", gripper_position=closed_position))
        elif initial_gripper_state not in {"hold", "none"}:
            raise ValueError(f"skill template '{skill_name}' initial_gripper_state must be open, closed, hold, or none")

    for step in primitive_sequence:
        if not isinstance(step, dict):
            raise ValueError(f"skill template '{skill_name}' contains a non-object step")

        primitive_name = str(step.get("primitive_name", "")).strip()
        if not primitive_name:
            raise ValueError(f"skill template '{skill_name}' contains a step without primitive_name")

        if primitive_name == "move_to_named_pose":
            primitives.append(
                PrimitiveSpec(
                    primitive_name="move_to_named_pose",
                    pose_name=_resolve_pose_name(step, target_name, place_name, named_targets),
                )
            )
            continue

        if primitive_name == "move_relative_ee":
            resolved_direction = str(step.get("motion_direction", "")).strip()
            if step.get("motion_direction_from_request"):
                resolved_direction = motion_direction
            resolved_distance = float(step.get("motion_distance", 0.0) or 0.0)
            if step.get("motion_distance_from_request"):
                resolved_distance = motion_distance
            delta_x, delta_y, delta_z = direction_to_delta(
                resolved_direction,
                resolved_distance,
                direction_mapping=direction_mapping,
            )
            primitives.append(
                PrimitiveSpec(
                    primitive_name="move_relative_ee",
                    relative_dx=delta_x,
                    relative_dy=delta_y,
                    relative_dz=delta_z,
                )
            )
            continue

        if primitive_name == "nav_straight":
            resolved_direction = step.get("direction", "")
            if step.get("direction_from_request"):
                resolved_direction = direction
            if not isinstance(resolved_direction, str):
                raise ValueError("direction must be a string")
            normalized_direction = resolved_direction.strip().lower()
            command_types = {
                "forward": ExecuteNavigation.Goal.FORWARD,
                "backward": ExecuteNavigation.Goal.BACKWARD,
                "left": ExecuteNavigation.Goal.STRAFE_LEFT,
                "right": ExecuteNavigation.Goal.STRAFE_RIGHT,
            }
            if normalized_direction not in command_types:
                raise ValueError(f"unsupported navigation direction: {normalized_direction}")
            resolved_distance = step.get("distance", 0.0)
            if step.get("distance_from_request"):
                resolved_distance = distance
            navigation_goal = ExecuteNavigation.Goal()
            navigation_goal.command_type = command_types[normalized_direction]
            navigation_goal.value = _positive_navigation_value(resolved_distance, "distance")
            primitives.append(PrimitiveSpec(primitive_name=primitive_name, navigation_goal=navigation_goal))
            continue

        if primitive_name == "nav_turn":
            resolved_direction = step.get("direction", "")
            if step.get("direction_from_request"):
                resolved_direction = direction
            if not isinstance(resolved_direction, str):
                raise ValueError("direction must be a string")
            normalized_direction = resolved_direction.strip().lower()
            command_types = {
                "left": ExecuteNavigation.Goal.TURN_LEFT,
                "right": ExecuteNavigation.Goal.TURN_RIGHT,
            }
            if normalized_direction not in command_types:
                raise ValueError(f"unsupported navigation direction: {normalized_direction}")
            resolved_degree = step.get("degree", 0.0)
            if step.get("degree_from_request"):
                resolved_degree = degree
            navigation_goal = ExecuteNavigation.Goal()
            navigation_goal.command_type = command_types[normalized_direction]
            navigation_goal.value = math.radians(_positive_navigation_value(resolved_degree, "degree"))
            primitives.append(PrimitiveSpec(primitive_name=primitive_name, navigation_goal=navigation_goal))
            continue

        if primitive_name == "nav_abs_coordinate":
            resolved_x = x if step.get("x_from_request") else step.get("x")
            resolved_y = y if step.get("y_from_request") else step.get("y")
            resolved_yaw = yaw if step.get("yaw_from_request") else step.get("yaw")
            target_x = _navigation_coordinate(resolved_x, "x")
            target_y = _navigation_coordinate(resolved_y, "y")
            target_yaw = _navigation_coordinate(resolved_yaw, "yaw")
            yaw_radians = math.radians(target_yaw)
            navigation_goal = ExecuteNavigation.Goal()
            navigation_goal.command_type = ExecuteNavigation.Goal.ABSOLUTE_POSE
            navigation_goal.target_pose.header.frame_id = "map"
            navigation_goal.target_pose.pose.position.x = target_x
            navigation_goal.target_pose.pose.position.y = target_y
            navigation_goal.target_pose.pose.orientation.z = math.sin(yaw_radians / 2.0)
            navigation_goal.target_pose.pose.orientation.w = math.cos(yaw_radians / 2.0)
            primitives.append(PrimitiveSpec(primitive_name=primitive_name, navigation_goal=navigation_goal))
            continue

        if primitive_name == "open_gripper":
            primitives.append(
                PrimitiveSpec(
                    primitive_name="open_gripper",
                    gripper_position=open_position,
                )
            )
            continue

        if primitive_name == "close_gripper":
            primitives.append(
                PrimitiveSpec(
                    primitive_name="close_gripper",
                    gripper_position=closed_position,
                )
            )
            continue

        if primitive_name in {"rotate_gripper_cw", "rotate_gripper_ccw"}:
            resolved_angle = float(step.get("motion_distance", 45.0) or 45.0)
            if step.get("motion_distance_from_request"):
                resolved_angle = motion_distance if motion_distance > 0.0 else 45.0
            primitives.append(
                PrimitiveSpec(
                    primitive_name=primitive_name,
                    relative_dz=resolved_angle,  # angle in degrees
                )
            )
            continue

        if primitive_name in {"move_to_configuration", "move_to_joint_positions"}:
            resolved_arm_joint_names = list(arm_joint_names or [])
            if not resolved_arm_joint_names:
                raise ValueError(f"arm_joint_names are required for {primitive_name}")
            joint_position_offsets = step.get("joint_position_offsets", {})
            joint_position_targets = step.get("joint_positions", {})
            has_offsets = bool(joint_position_offsets)
            has_absolute_targets = bool(joint_position_targets)
            if has_offsets and has_absolute_targets:
                raise ValueError(
                    f"skill template '{skill_name}' move_to_joint_positions step cannot define both "
                    "joint_position_offsets and joint_positions"
                )
            if not has_offsets and not has_absolute_targets:
                raise ValueError(
                    f"skill template '{skill_name}' move_to_joint_positions step must define "
                    "joint_position_offsets or joint_positions"
                )

            if has_absolute_targets:
                if not isinstance(joint_position_targets, dict):
                    raise ValueError(
                        f"skill template '{skill_name}' move_to_joint_positions step joint_positions must be an object"
                    )
                unknown_joint_names = sorted(set(joint_position_targets) - set(resolved_arm_joint_names))
                if unknown_joint_names:
                    raise KeyError(
                        f"skill template '{skill_name}' references unknown arm joints: {', '.join(unknown_joint_names)}"
                    )
                joint_positions = [
                    round(float(joint_position_targets.get(joint_name, 0.0) or 0.0), 6)
                    for joint_name in resolved_arm_joint_names
                ]
            else:
                if not isinstance(joint_position_offsets, dict):
                    raise ValueError(
                        f"skill template '{skill_name}' move_to_joint_positions step joint_position_offsets must be an object"
                    )
                if current_joint_positions is None:
                    raise KeyError("current_joint_positions are required for move_to_joint_positions offsets")
                unknown_joint_names = sorted(set(joint_position_offsets) - set(resolved_arm_joint_names))
                if unknown_joint_names:
                    raise KeyError(
                        f"skill template '{skill_name}' references unknown arm joints: {', '.join(unknown_joint_names)}"
                    )
                joint_positions = []
                for joint_name in resolved_arm_joint_names:
                    if joint_name not in current_joint_positions:
                        raise KeyError(f"current joint position unavailable: {joint_name}")
                    base_position = float(current_joint_positions[joint_name])
                    joint_offset = float(joint_position_offsets.get(joint_name, 0.0) or 0.0)
                    joint_positions.append(round(base_position + joint_offset, 6))

            duration_sec = float(step.get("duration_sec", 0.4))
            if duration_sec < 0.0:
                raise ValueError(f"skill template '{skill_name}' move_to_joint_positions duration must be >= 0")
            primitives.append(
                PrimitiveSpec(
                    primitive_name=primitive_name,
                    joint_names=resolved_arm_joint_names,
                    joint_positions=joint_positions,
                    duration_sec=duration_sec,
                )
            )
            continue

        if primitive_name == "move_through_joint_positions":
            resolved_arm_joint_names = list(arm_joint_names or [])
            if not resolved_arm_joint_names:
                raise ValueError("arm_joint_names are required for move_through_joint_positions")

            raw_waypoints = step.get("joint_waypoints", [])
            if not isinstance(raw_waypoints, list) or not raw_waypoints:
                raise ValueError(
                    f"skill template '{skill_name}' move_through_joint_positions step must define joint_waypoints"
                )

            joint_waypoints: list[list[float]] = []
            for waypoint_index, waypoint in enumerate(raw_waypoints):
                if not isinstance(waypoint, dict):
                    raise ValueError(
                        f"skill template '{skill_name}' move_through_joint_positions waypoint "
                        f"{waypoint_index} must be an object"
                    )
                joint_position_targets = waypoint.get("joint_positions", {})
                if not isinstance(joint_position_targets, dict) or not joint_position_targets:
                    raise ValueError(
                        f"skill template '{skill_name}' move_through_joint_positions waypoint "
                        f"{waypoint_index} must define joint_positions"
                    )
                unknown_joint_names = sorted(set(joint_position_targets) - set(resolved_arm_joint_names))
                if unknown_joint_names:
                    raise KeyError(
                        f"skill template '{skill_name}' references unknown arm joints: "
                        + ", ".join(unknown_joint_names)
                    )
                joint_waypoints.append(
                    [
                        round(float(joint_position_targets.get(joint_name, 0.0) or 0.0), 6)
                        for joint_name in resolved_arm_joint_names
                    ]
                )

            waypoint_duration_sec = float(step.get("waypoint_duration_sec", DEFAULT_WAYPOINT_DURATION_SEC))
            if waypoint_duration_sec <= 0.0:
                raise ValueError(
                    f"skill template '{skill_name}' move_through_joint_positions waypoint_duration_sec must be > 0"
                )
            primitives.append(
                PrimitiveSpec(
                    primitive_name="move_through_joint_positions",
                    joint_names=resolved_arm_joint_names,
                    joint_waypoints=joint_waypoints,
                    waypoint_duration_sec=waypoint_duration_sec,
                )
            )
            continue

        raise KeyError(f"unsupported primitive in skill template '{skill_name}': {primitive_name}")

    return primitives
