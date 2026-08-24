"""Lightweight shared contracts for perception requests and results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

PRIMARY_IMAGE_INPUT = "primary_image"
EE_POSE_INPUT = "ee_pose"
JOINT_STATE_INPUT = "joint_state"

KNOWN_REQUIRED_INPUTS = frozenset({PRIMARY_IMAGE_INPUT, EE_POSE_INPUT, JOINT_STATE_INPUT})
SCENE_ANALYSIS_RESULT_FIELD_KINDS = {
    "scene_summary": {"enum", "string"},
    "visible_objects": {"string_array"},
    "robot_state_summary": {"enum", "string"},
    "ee_pose_interpretation": {"enum", "string"},
    "risks": {"string_array"},
    "confidence": {"number"},
}


def validate_result_schema(contract: Any, result: Mapping[str, Any]) -> str | None:
    """Validate a result mapping against a generic response schema."""
    if not isinstance(contract, Mapping):
        return "response contract must be a mapping"
    field = contract.get("field")
    kind = contract.get("kind")
    if not isinstance(field, str) or not field:
        return "result contract field must be a non-empty string"
    if field not in result:
        return f"result is missing contract field {field!r}"
    value = result[field]
    if kind == "enum":
        allowed_values = contract.get("allowed_values")
        if not isinstance(allowed_values, list) or not allowed_values:
            return "enum result contract must declare allowed_values"
        if value not in allowed_values:
            return f"{field}={value!r} is outside the result contract"
    elif kind == "string":
        if not isinstance(value, str):
            return f"{field} must be a string"
    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            return f"{field} must be a finite number"
    elif kind == "string_array":
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            return f"{field} must be an array of strings"
    else:
        return f"unsupported result contract kind: {kind!r}"
    return None
