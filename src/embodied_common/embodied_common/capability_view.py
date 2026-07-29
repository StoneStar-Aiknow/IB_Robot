"""Build the public, ROS-independent view of normalized robot capabilities."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from embodied_common.skill_templates import get_skill_templates

_PUBLIC_CAPABILITY_FIELDS = (
    "summary",
    "domain",
    "moves_robot",
    "required_control_mode",
)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def build_capability_view(robot_config: Mapping[str, Any], *, timeout_policy: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable public capability document from normalized config data only."""
    config = _mapping(robot_config, "robot_config")
    embodied = _mapping(config.get("embodied", {}), "robot_config.embodied")
    robot_name = str(config.get("name", "")).strip()
    if not robot_name:
        raise ValueError("robot_config.name must be a non-empty string")

    raw_skill_templates = embodied.get("skill_templates")
    if raw_skill_templates is None:
        skill_templates = {}
    else:
        skill_templates = get_skill_templates(
            dict(_mapping(raw_skill_templates, "robot_config.embodied.skill_templates"))
        )
    named_poses = _mapping(embodied.get("named_poses", {}), "robot_config.embodied.named_poses")
    resolved_timeout_policy = _mapping(timeout_policy, "timeout_policy")
    pose_names = sorted(str(name) for name in named_poses)

    skills: list[dict[str, Any]] = []
    for skill_name in sorted(skill_templates):
        template = _mapping(skill_templates[skill_name], f"skill_templates.{skill_name}")
        capability = _mapping(template.get("capability"), f"skill_templates.{skill_name}.capability")
        for field in _PUBLIC_CAPABILITY_FIELDS + ("parameters", "recovery_policy"):
            if field not in capability:
                raise ValueError(f"skill_templates.{skill_name}.capability.{field} is required")
        skill_view: dict[str, Any] = {
            "name": str(skill_name),
            **{field: copy.deepcopy(capability[field]) for field in _PUBLIC_CAPABILITY_FIELDS},
            "parameters": copy.deepcopy(capability["parameters"]),
            "recovery_policy": copy.deepcopy(capability["recovery_policy"]),
        }
        skills.append(skill_view)

    public_view = {
        "robot_name": robot_name,
        "skills": skills,
        "pose_names": pose_names,
        "timeout_policy": {
            name: copy.deepcopy(resolved_timeout_policy[name]) for name in sorted(resolved_timeout_policy)
        },
    }
    public_view["capability_digest"] = hashlib.sha256(_canonical_json(public_view)).hexdigest()
    return public_view
