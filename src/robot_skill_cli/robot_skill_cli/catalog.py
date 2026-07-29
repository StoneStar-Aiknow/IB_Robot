"""ROS-independent loading and querying of robot skill capabilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embodied_common.capability_view import build_capability_view
from robot_config.config_path import resolve_robot_config_path
from robot_config.loader import load_robot_config_dict
from robot_config.timeout_policy import resolve_embodied_timeout_policy

_LIST_CAPABILITY_FIELDS = (
    "summary",
    "domain",
    "moves_robot",
    "required_control_mode",
)


class UnknownSkillError(ValueError):
    """Raised when a catalog query names a skill that is not enabled."""

    code = "UNKNOWN_SKILL"

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        super().__init__(f"unknown skill: {skill_name}")


@dataclass(frozen=True)
class CatalogContext:
    view: dict[str, Any]


@dataclass(frozen=True)
class GatewayTransport:
    status_service: str
    validate_skill_service: str
    skill_action_name: str


def load_catalog_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> CatalogContext:
    """Load one normalized config for public catalog use (ROS-independent)."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    embodied = robot_config.get("embodied", {})
    timeout_policy = resolve_embodied_timeout_policy(embodied)
    return CatalogContext(
        view=build_capability_view(robot_config, timeout_policy=timeout_policy),
    )


def load_runtime_context(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> tuple[CatalogContext, GatewayTransport]:
    """Load both catalog view and ROS transport for runtime Gateway commands."""
    resolved_path = resolve_robot_config_path(config_name=config_name, config_path=config_path)
    robot_config = load_robot_config_dict(resolved_path)
    embodied = robot_config.get("embodied", {})
    timeout_policy = resolve_embodied_timeout_policy(embodied)
    return (
        CatalogContext(
            view=build_capability_view(robot_config, timeout_policy=timeout_policy),
        ),
        GatewayTransport(
            status_service=embodied.get("skill_gateway_status_service", "/embodied/get_skill_gateway_status"),
            validate_skill_service=embodied.get("validate_skill_service", "/embodied/validate_skill"),
            skill_action_name=embodied.get("skill_action_name", "/embodied/execute_skill"),
        ),
    )


def load_capability_catalog(
    *,
    config_name: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load the normalized config and return its public capability view."""
    return load_catalog_context(config_name=config_name, config_path=config_path).view


def _skill_entry(view: dict[str, Any], skill_name: str) -> dict[str, Any]:
    normalized_name = skill_name.strip()
    for skill in view["skills"]:
        if skill["name"] == normalized_name:
            return skill
    raise UnknownSkillError(normalized_name)


def list_skills(view: dict[str, Any]) -> dict[str, Any]:
    skills = []
    for skill in view["skills"]:
        entry = {
            "name": skill["name"],
            **{field: copy.deepcopy(skill[field]) for field in _LIST_CAPABILITY_FIELDS},
        }
        skills.append(entry)
    return {
        "robot_name": view["robot_name"],
        "config_digest": view["capability_digest"],
        "skills": skills,
    }


def describe_skill(view: dict[str, Any], skill_name: str) -> dict[str, Any]:
    skill = _skill_entry(view, skill_name)
    result = {
        "robot_name": view["robot_name"],
        "name": skill["name"],
        **{field: copy.deepcopy(skill[field]) for field in _LIST_CAPABILITY_FIELDS},
        "parameters": copy.deepcopy(skill["parameters"]),
        "recovery_policy": copy.deepcopy(skill["recovery_policy"]),
        "timeout_policy": copy.deepcopy(view["timeout_policy"]),
        "config_digest": view["capability_digest"],
    }
    return result


def list_poses(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "robot_name": view["robot_name"],
        "config_digest": view["capability_digest"],
        "poses": list(view["pose_names"]),
    }
