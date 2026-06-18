"""Config-driven skill/pose catalog.

This module resolves the active robot configuration (following the same
convention as ``robot_config/launch/robot.launch.py``) and derives the skill
and named-pose catalog straight from the single source of truth (the YAML
parsed into ``RobotConfig``). No skill knowledge is hard-coded here: changing
the YAML changes the catalog the agent sees.

Resolution precedence (highest first):
    1. explicit ``config_path``
    2. ``ROBOT_CONFIG`` environment variable (full path)
    3. ``config_name`` resolved against the installed robot_config share
    4. ``ROBOT_NAME`` env var, then the built-in default ``so101_single_arm``
    5. workspace ``src/robot_config/config/robots/<name>.yaml`` fallback (dev)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_ROBOT_NAME = "so101_single_arm"
DEFAULT_JOINT_STATE_TOPIC = "/joint_states"
_VALID_MOTION_DIRECTIONS = {"forward", "backward", "left", "right", "up", "down"}


@dataclass
class Catalog:
    """Immutable snapshot of the agent-facing skill/pose catalog."""

    robot_name: str
    config_path: str
    skills: list[dict[str, Any]] = field(default_factory=list)
    poses: list[dict[str, Any]] = field(default_factory=list)
    workspace: dict[str, Any] = field(default_factory=dict)
    ros_interfaces: dict[str, str] = field(default_factory=dict)

    def to_summary(self) -> dict[str, Any]:
        """Compact representation (used in logs / status payloads)."""
        return {
            "robot_name": self.robot_name,
            "config_path": self.config_path,
            "skill_count": len(self.skills),
            "pose_count": len(self.poses),
        }


def resolve_config_path(
    config_name: str | None = None,
    config_path: str | None = None,
) -> Path:
    """Resolve the robot_config YAML path following launch conventions."""
    if config_path:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"ROBOT_CONFIG path not found: {path}")
        return path

    env_path = os.environ.get("ROBOT_CONFIG", "").strip()
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path
        logger.warning("ROBOT_CONFIG env points to missing path: %s", path)

    name = (config_name or os.environ.get("ROBOT_NAME") or DEFAULT_ROBOT_NAME).strip()

    # 1) installed share directory
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory("robot_config"))
        candidate = share / "config" / "robots" / f"{name}.yaml"
        if candidate.exists():
            return candidate
    except Exception as exc:  # noqa: BLE001 - share resolution is best-effort
        logger.debug("robot_config share lookup failed: %s", exc)

    # 2) dev workspace fallback: <repo>/src/robot_config/config/robots/<name>.yaml
    here = Path(__file__).resolve()
    repo_src = here.parents[3]  # src/robot_mcp/robot_mcp/catalog.py -> src
    candidate = repo_src / "robot_config" / "config" / "robots" / f"{name}.yaml"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Could not resolve robot config '{name}'. Pass --config-path, set ROBOT_CONFIG, "
        f"or ensure robot_config is installed."
    )


def _build_skill_entry(skill_name: str, template: dict[str, Any]) -> dict[str, Any]:
    """Derive an agent-facing description of a single skill from its template."""
    primitive_sequence = template.get("primitive_sequence") or []
    primitives: list[str] = []
    pose_targets: list[str] = []
    accepts_motion = False
    for step in primitive_sequence:
        if not isinstance(step, dict):
            continue
        primitive_name = str(step.get("primitive_name", "")).strip()
        if primitive_name:
            primitives.append(primitive_name)
        pose_name = str(step.get("pose_name", "") or "").strip()
        if pose_name:
            pose_targets.append(pose_name)
        if step.get("motion_direction_from_request") or step.get("motion_distance_from_request"):
            accepts_motion = True

    return {
        "name": skill_name,
        "primitives": primitives,
        "pose_targets": pose_targets,
        "accepts_motion": accepts_motion,
        "vision_only": skill_name == "inspect_scene" or not primitives,
        "doc": _SKILL_DOCS.get(skill_name, ""),
    }


def _build_pose_entry(pose_name: str, raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a named pose entry to {position, orientation} float dicts."""
    position = raw.get("position", {}) if isinstance(raw, dict) else {}
    orientation = raw.get("orientation", {}) if isinstance(raw, dict) else {}

    def _round_map(keys: tuple[str, ...], src: Any) -> dict[str, float]:
        if not isinstance(src, dict):
            return {}
        return {k: round(float(src[k]), 6) for k in keys if k in src and src[k] is not None}

    return {
        "name": pose_name,
        "position": _round_map(("x", "y", "z"), position),
        "orientation": _round_map(("x", "y", "z", "w"), orientation),
    }


# Short human-readable hints for the canonical skills (loader.py valid set).
# Pure documentation aid; the source of truth remains robot_config YAML.
_SKILL_DOCS: dict[str, str] = {
    "inspect_scene": "Acquire a scene observation (vision only, no motion).",
    "open_gripper_skill": "Open the gripper to its open position.",
    "close_gripper_skill": "Close the gripper to its closed position.",
    "recover_safe_pose": "Move to the 'home' safe pose.",
    "recover_zero_pose": "Move to the 'zero' pose.",
    "move_relative_ee": "Translate the end-effector by a direction/distance.",
    "rotate_gripper_cw": "Rotate the gripper clockwise.",
    "rotate_gripper_ccw": "Rotate the gripper counter-clockwise.",
    "dance_basic": "Play the basic dance sequence.",
}


def build_catalog(config_name: str | None = None, config_path: str | None = None) -> Catalog:
    """Load robot_config and derive the skill/pose catalog (SSOT-driven)."""
    from robot_config import load_robot_config

    resolved = resolve_config_path(config_name=config_name, config_path=config_path)
    logger.info("robot_mcp loading robot_config from: %s", resolved)

    cfg = load_robot_config(str(resolved))
    embodied = cfg.embodied

    skill_templates: dict[str, Any] = embodied.skill_templates or {}
    skills = [_build_skill_entry(name, tpl) for name, tpl in skill_templates.items()]
    skills.sort(key=lambda s: s["name"])

    named_poses: dict[str, Any] = embodied.named_poses or {}
    poses = [_build_pose_entry(name, raw) for name, raw in named_poses.items()]
    poses.sort(key=lambda p: p["name"])

    workspace = _normalize_workspace(embodied.workspace or {})

    # ROS interface names come straight from the SSOT config so the agent sees
    # exactly what the running stack uses (no hard-coding).
    ros_interfaces = {
        "status_topic": embodied.status_topic,
        "skill_action": embodied.skill_action_name,
        "primitive_action": embodied.primitive_action_name,
        "validate_skill_service": embodied.validate_skill_service,
        "validate_primitive_service": embodied.validate_primitive_service,
        "joint_state_topic": DEFAULT_JOINT_STATE_TOPIC,
    }

    return Catalog(
        robot_name=cfg.name,
        config_path=str(resolved),
        skills=skills,
        poses=poses,
        workspace=workspace,
        ros_interfaces=ros_interfaces,
    )


def _normalize_workspace(workspace: dict[str, Any]) -> dict[str, Any]:
    """Round workspace bounds to plain {axis: [min, max]} for agent readability."""
    out: dict[str, Any] = {}
    for axis in ("x", "y", "z"):
        bounds = workspace.get(axis)
        if isinstance(bounds, list | tuple) and len(bounds) == 2:
            out[axis] = [round(float(bounds[0]), 4), round(float(bounds[1]), 4)]
    return out


def valid_motion_directions() -> list[str]:
    """Expose the legal motion directions (mirrors loader.py validation)."""
    return sorted(_VALID_MOTION_DIRECTIONS)
