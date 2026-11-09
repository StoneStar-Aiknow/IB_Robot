"""Strict configuration for the optional overhead-heart observer.

This module stays standard-library-only so launch registry discovery does not
import the body model or NumPy. Motion output requires a separately validated
template and is deliberately unavailable in this observer entry point.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

MATCH_TOPIC = "/preset_motions/matched"
STATUS_TOPIC = "/preset_motions/status"
STATE_TOPIC = "/mirror_session/state"
_KEYS = {"enabled", "motion_enabled", "match_topic", "status_topic", "matcher"}
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
    if section.get("motion_enabled", False):
        raise ValueError("preset_motions motion output is not accepted without a validated session handoff")
    return {
        "pear_frame_topic": topic(stream.get("frame_topic", ""), "PEAR frame topic"),
        "actor_state_topic": topic(stream.get("actor_state_topic", ""), "actor state topic"),
        "session_state_topic": STATE_TOPIC,
        "match_topic": topic(section.get("match_topic", MATCH_TOPIC), "match topic"),
        "status_topic": topic(section.get("status_topic", STATUS_TOPIC), "status topic"),
        "matcher_json": json.dumps(matcher_settings(section.get("matcher", {})), sort_keys=True),
        "motion_enabled": False,
    }
