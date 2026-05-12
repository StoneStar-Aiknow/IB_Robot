"""Shared embodied skill template definitions."""

from __future__ import annotations

import copy
from typing import Any

SUPPORTED_PRIMITIVES = {
    "move_to_named_pose",
    "move_relative_ee",
    "open_gripper",
    "close_gripper",
    "rotate_gripper_cw",
    "rotate_gripper_ccw",
}

DEFAULT_SKILL_TEMPLATES: dict[str, dict[str, Any]] = {
    "inspect_scene": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "pose_name": "observe_table"},
        ]
    },
    "recover_safe_pose": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "pose_name": "home"},
        ]
    },
    "observe_target_area": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "target_pose_key": "observe_pose"},
        ]
    },
    "approach_named_target": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "target_pose_key": "pregrasp_pose"},
        ]
    },
    "hover_named_target": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "target_pose_key": "hover_pose"},
        ]
    },
    "pick_named_target": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "target_pose_key": "pregrasp_pose"},
            {"primitive_name": "move_to_named_pose", "target_pose_key": "grasp_pose"},
            {"primitive_name": "close_gripper"},
            {"primitive_name": "move_to_named_pose", "target_pose_key": "lift_pose"},
        ]
    },
    "lift_named_target": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "target_pose_key": "lift_pose"},
        ]
    },
    "retreat_from_target": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "target_pose_key": "retreat_pose"},
        ]
    },
    "place_named_pose": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "place_name_from_request": True},
            {"primitive_name": "open_gripper"},
        ]
    },
    "release_at_named_pose": {
        "primitive_sequence": [
            {"primitive_name": "move_to_named_pose", "place_name_from_request": True},
            {"primitive_name": "open_gripper"},
        ]
    },
    "open_gripper_skill": {
        "primitive_sequence": [
            {"primitive_name": "open_gripper"},
        ]
    },
    "close_gripper_skill": {
        "primitive_sequence": [
            {"primitive_name": "close_gripper"},
        ]
    },
    "move_relative_ee": {
        "primitive_sequence": [
            {
                "primitive_name": "move_relative_ee",
                "motion_direction_from_request": True,
                "motion_distance_from_request": True,
            },
        ]
    },
}


def get_skill_templates(raw_templates: dict[str, Any] | None) -> dict[str, Any]:
    """Return provided skill templates or the default minimum-closure set."""
    if raw_templates:
        return copy.deepcopy(raw_templates)
    return copy.deepcopy(DEFAULT_SKILL_TEMPLATES)
