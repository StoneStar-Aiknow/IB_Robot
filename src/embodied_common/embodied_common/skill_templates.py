"""Default neutral skill template definitions for the embodied pipeline."""

from __future__ import annotations

import copy
from typing import Any

SUPPORTED_PRIMITIVES = {
    "move_to_named_pose",
    "move_relative_ee",
    "move_to_joint_positions",
    "move_through_joint_positions",
    "open_gripper",
    "close_gripper",
    "rotate_gripper_cw",
    "rotate_gripper_ccw",
}

DEFAULT_SKILL_TEMPLATES: dict[str, dict[str, Any]] = {
    "inspect_scene": {"primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "observe_table"}]},
    "recover_safe_pose": {"primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "home"}]},
    "recover_zero_pose": {"primitive_sequence": [{"primitive_name": "move_to_named_pose", "pose_name": "zero"}]},
    "open_gripper_skill": {"primitive_sequence": [{"primitive_name": "open_gripper"}]},
    "close_gripper_skill": {"primitive_sequence": [{"primitive_name": "close_gripper"}]},
    "move_relative_ee": {
        "primitive_sequence": [
            {
                "primitive_name": "move_relative_ee",
                "motion_direction_from_request": True,
                "motion_distance_from_request": True,
            }
        ]
    },
    "rotate_gripper_cw": {
        "primitive_sequence": [{"primitive_name": "rotate_gripper_cw", "motion_distance_from_request": True}]
    },
    "rotate_gripper_ccw": {
        "primitive_sequence": [{"primitive_name": "rotate_gripper_ccw", "motion_distance_from_request": True}]
    },
}

SUPPORTED_SKILLS = set(DEFAULT_SKILL_TEMPLATES.keys()).union({"dance_basic"})
DEFAULT_ALLOWED_SKILLS = list(DEFAULT_SKILL_TEMPLATES.keys())


def get_skill_templates(raw_templates: dict[str, Any] | None) -> dict[str, Any]:
    if raw_templates:
        return copy.deepcopy(raw_templates)
    return copy.deepcopy(DEFAULT_SKILL_TEMPLATES)
