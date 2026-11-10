"""Neutral online catalog verification and public projection helpers."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from typing import Any

from embodied_common.capability_view import project_capability_timeout_policy
from embodied_common.catalog_digest import (
    derive_capability_digest,
    derive_provenance_digest,
    derive_registry_digest,
    to_canonical_json,
)


class UnknownSkillError(ValueError):
    """Raised when a catalog query names a skill that is not enabled."""

    code = "UNKNOWN_SKILL"

    def __init__(self, skill_name: str) -> None:
        self.skill_name = skill_name
        super().__init__(f"unknown skill: {skill_name}")


class UnknownGameError(ValueError):
    """Raised when a visual game is absent or disabled in a catalog view."""

    code = "UNKNOWN_GAME"

    def __init__(self, game_name: str) -> None:
        self.game_name = game_name
        super().__init__(f"unknown or disabled visual game: {game_name}")


def capability_view_from_snapshot(snapshot: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """Verify one exact Gateway snapshot and return its public catalog view."""
    expected_identity = (
        status["registry_epoch"],
        int(status["registry_generation"]),
        status["registry_digest"],
    )
    actual_identity = (
        snapshot["registry_epoch"],
        int(snapshot["generation"]),
        snapshot["registry_digest"],
    )
    if not snapshot["success"] or actual_identity != expected_identity:
        raise ValueError("SKILL_REGISTRY_VERSION_MISMATCH: snapshot identity does not match Gateway status")
    try:
        payload = json.loads(snapshot["snapshot_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot payload is not valid JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot schema is invalid")
    if to_canonical_json(payload) != snapshot["snapshot_json"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot payload is not canonical")
    registry_preimage = payload.get("registry_preimage")
    capability_preimage = payload.get("capability_preimage")
    provenance_preimage = payload.get("provenance_preimage")
    if not all(isinstance(value, dict) for value in (registry_preimage, capability_preimage, provenance_preimage)):
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: snapshot preimages are invalid")
    if derive_registry_digest(registry_preimage) != snapshot["registry_digest"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: registry digest does not match")
    if derive_capability_digest(capability_preimage) != snapshot["capability_digest"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: capability digest does not match")
    if derive_provenance_digest(provenance_preimage) != snapshot["provenance_digest"]:
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: provenance digest does not match")
    capability_mapping = capability_preimage.get("capability_view")
    if not isinstance(capability_mapping, dict):
        raise ValueError("SKILL_SNAPSHOT_DIGEST_MISMATCH: capability view is invalid")
    planner_visible = {str(name) for name in capability_preimage.get("planner_visible_skill_names", [])}
    semantic_levels = {
        str(entry["name"]): str(entry["semantic_level"])
        for entry in registry_preimage.get("skills", [])
        if isinstance(entry, Mapping) and "name" in entry and "semantic_level" in entry
    }
    timeout_caps = {
        str(entry["name"]): float(entry["template"]["timeout_sec"])
        for entry in registry_preimage.get("skills", [])
        if isinstance(entry, Mapping)
        and isinstance(entry.get("template"), Mapping)
        and "timeout_sec" in entry["template"]
    }
    skills = [copy.deepcopy(capability_mapping[name]) for name in sorted(capability_mapping)]
    for skill in skills:
        skill["planner_visible"] = skill["name"] in planner_visible
        skill["semantic_level"] = semantic_levels.get(skill["name"], "")
        if skill["name"] in timeout_caps:
            skill["timeout_sec"] = timeout_caps[skill["name"]]
    return {
        "robot_name": capability_preimage["robot_name"],
        "skills": skills,
        "pose_names": list(capability_preimage["named_pose_names"]),
        "timeout_policy": project_capability_timeout_policy(capability_preimage["timeout_policy"]),
        "capability_digest": snapshot["capability_digest"],
        "profile_name": capability_preimage["profile_name"],
    }


def _skill_entry(view: dict[str, Any], skill_name: str) -> dict[str, Any]:
    normalized_name = skill_name.strip()
    for skill in view["skills"]:
        if skill["name"] == normalized_name:
            return skill
    raise UnknownSkillError(normalized_name)


_LIST_CAPABILITY_FIELDS = (
    "summary",
    "domain",
    "moves_robot",
    "required_control_mode",
)
_DISPLAY_CAPABILITY_FIELDS = ("display_name", "tags")

_PLANNER_SKILL_FIELDS = (
    "summary",
    "domain",
    "semantic_level",
    "moves_robot",
    "required_control_mode",
    "parameters",
    "schema_version",
)


def planner_catalog_view(view: dict[str, Any]) -> dict[str, Any]:
    """Return the smallest catalog a planner prompt needs."""
    skills = [
        {
            "name": skill["name"],
            **{field: copy.deepcopy(skill[field]) for field in _PLANNER_SKILL_FIELDS if field in skill},
        }
        for skill in view.get("skills", [])
        if skill.get("planner_visible", True)
    ]
    projected = {
        "robot_name": view.get("robot_name", ""),
        "skills": skills,
        "pose_names": list(view.get("pose_names", [])),
        "capability_digest": view.get("capability_digest", ""),
    }
    for field in ("control_plane_ready", "active_control_mode", "motion_authorized"):
        if field in view:
            projected[field] = view[field]
    return projected


def list_skills(view: dict[str, Any]) -> dict[str, Any]:
    skills = []
    for skill in view["skills"]:
        entry = {
            "name": skill["name"],
            "contract_schema_version": int(skill.get("schema_version", 1)),
            **{field: copy.deepcopy(skill[field]) for field in _LIST_CAPABILITY_FIELDS},
            **{field: copy.deepcopy(skill[field]) for field in _DISPLAY_CAPABILITY_FIELDS if field in skill},
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
        "schema_version": int(skill.get("schema_version", 1)),
        **{field: copy.deepcopy(skill[field]) for field in _LIST_CAPABILITY_FIELDS},
        **{field: copy.deepcopy(skill[field]) for field in _DISPLAY_CAPABILITY_FIELDS if field in skill},
        "parameters": copy.deepcopy(skill["parameters"]),
        "recovery_policy": copy.deepcopy(skill["recovery_policy"]),
        "timeout_policy": copy.deepcopy(view["timeout_policy"]),
        "config_digest": view["capability_digest"],
    }
    if "timeout_sec" in skill:
        result["timeout_sec"] = float(skill["timeout_sec"])
    return result


def list_poses(view: dict[str, Any]) -> dict[str, Any]:
    return {
        "robot_name": view["robot_name"],
        "config_digest": view["capability_digest"],
        "poses": list(view["pose_names"]),
    }


def list_games(game_view: dict[str, Any]) -> dict[str, Any]:
    return {
        "robot_name": game_view["robot_name"],
        "config_digest": game_view["config_digest"],
        "games": [
            {
                "name": game["name"],
                "summary": game["summary"],
                "result_field": game["result_schema"]["field"],
            }
            for game in game_view["games"]
        ],
    }


def require_enabled_game(game_view: dict[str, Any], game_name: str) -> dict[str, Any]:
    normalized_name = game_name.strip()
    for game in game_view["games"]:
        if game["name"] == normalized_name:
            return copy.deepcopy(game)
    raise UnknownGameError(normalized_name)


def describe_game(game_view: dict[str, Any], game_name: str) -> dict[str, Any]:
    game = require_enabled_game(game_view, game_name)
    return {
        "robot_name": game_view["robot_name"],
        "name": game["name"],
        "summary": game["summary"],
        "required_inputs": game["required_inputs"],
        "result_schema": game["result_schema"],
        "timeout_sec": game_view["timeout_sec"],
        "result_retention_sec": game_view["result_retention_sec"],
        "result_capacity": game_view["result_capacity"],
        "config_digest": game_view["config_digest"],
    }
