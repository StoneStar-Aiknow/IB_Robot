"""Helpers for resolving skills, targets, and named poses."""

import json
from dataclasses import dataclass
from typing import Any

from skill_library.skill_templates import get_skill_templates


def load_json_mapping(raw_value: str) -> dict[str, Any]:
    if not raw_value:
        return {}
    return json.loads(raw_value)


@dataclass
class PrimitiveSpec:
    primitive_name: str
    pose_name: str = ""
    gripper_position: float = 0.0
    relative_dx: float = 0.0
    relative_dy: float = 0.0
    relative_dz: float = 0.0


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
) -> list[PrimitiveSpec]:
    templates = get_skill_templates(skill_templates)
    template = templates.get(skill_name)
    if template is None:
        raise KeyError(f"unsupported skill: {skill_name}")

    primitive_sequence = template.get("primitive_sequence", [])
    if not isinstance(primitive_sequence, list) or not primitive_sequence:
        raise ValueError(f"skill template '{skill_name}' must define a non-empty primitive_sequence")

    primitives: list[PrimitiveSpec] = []
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

        raise KeyError(f"unsupported primitive in skill template '{skill_name}': {primitive_name}")

    return primitives
