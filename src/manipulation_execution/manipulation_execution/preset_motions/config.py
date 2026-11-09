"""Strict configuration for the optional overhead-heart observer.

This module stays standard-library-only so launch registry discovery does not
import the body model or NumPy. Motion output requires a separately validated
template and is deliberately unavailable in this observer entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

MATCH_TOPIC = "/preset_motions/matched"
STATUS_TOPIC = "/preset_motions/status"
STATE_TOPIC = "/mirror_session/state"
MATCH_STATE_TOPIC = "/preset_motions/state"
_KEYS = {
    "enabled",
    "motion_enabled",
    "match_topic",
    "status_topic",
    "state_topic",
    "matcher",
    "motion_name",
    "template_file",
    "geometry_file",
}
_TIMING = {
    "enter_confirm_s",
    "exit_confirm_s",
    "min_enter_frames",
    "max_frame_age_s",
    "max_receive_gap_s",
    "actor_max_age_s",
    "future_tolerance_s",
    "clock_jump_tolerance_s",
}
_BOUNDS = {
    "wrist_height_min",
    "elbow_height_min",
    "wrist_span_max",
    "wrist_center_offset_max",
    "forearm_inward_min",
    "elbow_angle_min_deg",
    "elbow_angle_max_deg",
}


def topic(value, label):
    if not isinstance(value, str) or not value.startswith("/") or any(c.isspace() for c in value):
        raise ValueError(f"{label} must be an absolute ROS topic")
    return value


def matcher_settings(value):
    if not isinstance(value, Mapping):
        raise ValueError("preset_motions.matcher must be a mapping")
    unknown = set(value) - (_TIMING | {"rule"})
    if unknown:
        raise ValueError(f"preset_motions.matcher has unknown keys {sorted(unknown)}")
    result = dict(value)
    for key in _TIMING & result.keys():
        number = result[key]
        if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
            raise ValueError(f"preset_motions.matcher.{key} must be finite and non-negative")
        if key == "min_enter_frames" and (type(number) is not int or number < 2):
            raise ValueError("min_enter_frames must be an integer >= 2")
    if "rule" in result:
        rule = result["rule"]
        if not isinstance(rule, Mapping) or set(rule) - {"enter", "keep"}:
            raise ValueError("matcher.rule accepts enter and keep mappings only")
        for phase, bounds in rule.items():
            if not isinstance(bounds, Mapping) or set(bounds) - _BOUNDS:
                raise ValueError(f"matcher.rule.{phase} contains invalid bounds")
            for key, number in bounds.items():
                if type(number) not in (int, float) or not math.isfinite(number):
                    raise ValueError(f"matcher.rule.{phase}.{key} must be finite")
        result["rule"] = {phase: dict(bounds) for phase, bounds in rule.items()}
    return result


def launch_parameters(robot_config, stream):
    """Return observer parameters, or None for the unchanged disabled baseline."""
    section = robot_config.get("preset_motions")
    if section is None:
        return None
    if not isinstance(section, Mapping) or set(section) - _KEYS:
        raise ValueError("preset_motions must contain only its documented keys")
    for key in ("enabled", "motion_enabled"):
        if key in section and type(section[key]) is not bool:
            raise ValueError(f"preset_motions.{key} must be a boolean")
    if not section.get("enabled", False):
        return None
    if section.get("motion_enabled", False) and (not section.get("template_file") or not section.get("geometry_file")):
        raise ValueError("preset_motions motion output requires a validated template and geometry artifact")
    return {
        "pear_frame_topic": topic(stream.get("frame_topic", ""), "PEAR frame topic"),
        "actor_state_topic": topic(stream.get("actor_state_topic", ""), "actor state topic"),
        "session_state_topic": STATE_TOPIC,
        "match_topic": topic(section.get("match_topic", MATCH_TOPIC), "match topic"),
        "status_topic": topic(section.get("status_topic", STATUS_TOPIC), "status topic"),
        "match_state_topic": topic(section.get("state_topic", MATCH_STATE_TOPIC), "match state topic"),
        "matcher_json": json.dumps(matcher_settings(section.get("matcher", {})), sort_keys=True),
        "motion_enabled": False,
    }


def motion_parameters(robot_config, resolver=lambda value: value):
    """Require independent safety evidence before creating any runtime client.

    Visual approval alone cannot activate motion. The runtime checks the full
    geometry again; these launch checks only reject an invalid deployment early.
    """
    section = robot_config.get("preset_motions") or {}
    if not isinstance(section, Mapping):
        raise ValueError("preset_motions must be a mapping")
    if not section.get("enabled", False) or not section.get("motion_enabled", False):
        return None
    if type(section["enabled"]) is not bool or type(section["motion_enabled"]) is not bool:
        raise ValueError("preset_motions enable flags must be booleans")
    import yaml

    def path(key):
        value = section.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"preset_motions.{key} is required for motion")
        resolved = Path(resolver(value))
        if not resolved.is_absolute() or not resolved.is_file():
            raise ValueError(f"preset_motions.{key} must resolve to an existing absolute file")
        return resolved

    template_path, geometry_path = path("template_file"), path("geometry_file")
    record = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    name = section.get("motion_name", "overhead_heart")
    if name != "overhead_heart" or not isinstance(record, Mapping) or record.get("name") != name:
        raise ValueError("only the configured overhead_heart template is admitted")
    if record.get("validated") is not True or not isinstance(geometry, dict) or geometry.get("validated") is not True:
        raise ValueError("visual-only or unvalidated assets cannot enable preset motion")
    approval = record.get("visual_approval")
    if not isinstance(approval, Mapping) or approval.get("approved") is not True:
        raise ValueError("the fixed visual pose has not been approved")
    for item in (record, geometry):
        clearance = item.get("required_head_clearance_m")
        if type(clearance) not in (int, float) or not math.isfinite(clearance) or clearance < 0.20:
            raise ValueError("preset motion requires the unchanged 0.20 m head clearance")
    payload = {key: value for key, value in geometry.items() if key != "sha256"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if geometry.get("name") != name or geometry.get("sha256") != digest:
        raise ValueError("preset geometry identity or digest mismatch")
    if not all(
        geometry.get(key) is True
        for key in (
            "head_envelope_covers_allowed_motion",
            "complete_articulated_hands",
            "continuous_motion_bounds_verified",
        )
    ):
        raise ValueError("preset geometry lacks complete continuous-motion safety evidence")
    if record.get("stop_bridge_certified") is not True:
        raise ValueError("preset motion requires a separately certified stop bridge")
    if record.get("motion_enabled") is not True:
        raise ValueError("preset template motion output must be explicitly enabled")
    return {
        "motion_name": name,
        "match_state_topic": topic(section.get("state_topic", MATCH_STATE_TOPIC), "match state topic"),
        "template_file": str(template_path),
        "geometry_file": str(geometry_path),
    }
