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


def _synthesize_skill_doc(skill_name: str, description: dict[str, Any]) -> str:
    """Build a dense, disambiguation-rich doc string from the SSOT description block.

    The synthesized doc is what an MCP/LLM caller reads to pick between
    near-synonym skills, so it leads with category+summary, lists when-to-use,
    and -- critically -- spells out do-not-use redirects toward the right
    alternative. Falls back to the legacy ``_SKILL_DOCS`` one-liners only when
    the SSOT description is absent.
    """
    summary = str(description.get("summary", "")).strip()
    category = str(description.get("category", "")).strip()
    when_to_use = description.get("when_to_use") or []
    do_not_use = description.get("do_not_use") or []
    aliases_zh = description.get("aliases_zh") or []
    motion_scope = description.get("motion_scope") or []
    intensity = str(description.get("intensity", "")).strip()
    anchor_pose = str(description.get("anchor_pose", "")).strip()

    parts: list[str] = []
    if category:
        parts.append(f"[{category}]")
    if summary:
        parts.append(summary)
    if when_to_use:
        parts.append("Use: " + ", ".join(str(item) for item in when_to_use) + ".")
    redirects = [
        f"{entry.get('condition', '')} -> {entry.get('instead_use', '')}"
        for entry in do_not_use
        if isinstance(entry, dict) and entry.get("instead_use")
    ]
    if redirects:
        parts.append("Do NOT use for: " + "; ".join(redirects) + ".")
    if aliases_zh:
        parts.append("中文: " + "/".join(str(alias) for alias in aliases_zh) + ".")
    scope_bits = []
    if motion_scope:
        scope_bits.append("scope=" + "+".join(str(token) for token in motion_scope))
    if anchor_pose and anchor_pose != "none":
        scope_bits.append(f"anchor={anchor_pose}")
    if intensity:
        scope_bits.append(f"intensity={intensity}")
    if scope_bits:
        parts.append(" ".join(scope_bits) + ".")
    synthesized = " ".join(part for part in parts if part).strip()
    return synthesized or _SKILL_DOCS.get(skill_name, "")


def _build_skill_entry(skill_name: str, template: dict[str, Any]) -> dict[str, Any]:
    """Derive an agent-facing description of a single skill from its template.

    The structured ``description`` block (SSOT YAML) is surfaced verbatim so an
    LLM can filter deterministically (category / aliases / motion_scope /
    intensity / anchor_pose / requires_motion_params), while ``doc`` carries the
    synthesized, human-readable disambiguation text. The legacy ``_SKILL_DOCS``
    map is only consulted as a last-resort fallback when the SSOT block is
    missing, so no skill is ever left without a hint.
    """
    primitive_sequence = template.get("primitive_sequence") or []
    primitives: list[str] = []
    pose_targets: list[str] = []
    accepts_motion = False
    initial_gripper_state = str(template.get("initial_gripper_state", "")).strip().lower()
    if initial_gripper_state == "open":
        primitives.append("open_gripper")
    elif initial_gripper_state == "closed":
        primitives.append("close_gripper")
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

    description = template.get("description") if isinstance(template.get("description"), dict) else {}
    executor_name = str(template.get("executor", "")).strip()

    entry: dict[str, Any] = {
        "name": skill_name,
        "primitives": primitives,
        "pose_targets": pose_targets,
        "accepts_motion": accepts_motion,
        "required_args": [str(arg) for arg in template.get("required_args", [])],
        "executor": executor_name or "primitive_sequence",
        "timeout_sec": float(template.get("timeout_sec", 0.0) or 0.0),
        "vision_only": skill_name == "inspect_scene" or (not primitives and not executor_name),
        "doc": _synthesize_skill_doc(skill_name, description),
    }

    if description:
        entry["category"] = description.get("category", "")
        entry["aliases_zh"] = list(description.get("aliases_zh") or [])
        entry["aliases_en"] = list(description.get("aliases_en") or [])
        entry["motion_scope"] = list(description.get("motion_scope") or [])
        entry["anchor_pose"] = description.get("anchor_pose", "")
        entry["intensity"] = description.get("intensity", "")
        entry["duration_sec_estimate"] = description.get("duration_sec_estimate")
        entry["requires_motion_params"] = bool(description.get("requires_motion_params", False))
        entry["when_to_use"] = list(description.get("when_to_use") or [])
        entry["do_not_use"] = list(description.get("do_not_use") or [])

    return entry


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
    "wave_hello": "Wave hello with the wrist, then return to the gesture base pose.",
    "nod_yes": "Nod yes with small shoulder/elbow motion, then return to the gesture base pose.",
    "shake_no": "Shake no with a small base joint motion, then return to the gesture base pose.",
    "celebrate": "Move to observe_table, then celebrate by moving up/down/left/right around that observation pose.",
    "greet_observe_raise": "Move to observe_table, then greet by raising and lowering the end-effector.",
    "act_cute": "Play a cute attention-seeking wiggle with gripper open-close.",
    "happy_spin_upright": "Keep an upright gesture base while spinning the base joint with a cheerful wrist wiggle.",
    "pick_object": "Detect, grasp, verify, and lift the requested object.",
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
    if workspace.get("max_radius_m") is not None:
        out["max_radius_m"] = round(float(workspace["max_radius_m"]), 4)
    return out


def valid_motion_directions() -> list[str]:
    """Expose the legal motion directions (mirrors loader.py validation)."""
    return sorted(_VALID_MOTION_DIRECTIONS)
