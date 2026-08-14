"""Canonical payload helpers shared by Gateway skill requests."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from typing import Any

# ROS 2 Humble float32 setters reject values beyond the IEEE-754 single-precision maximum,
# so motion_distance and timeout_sec must stay within float32 range to match ROS action fields.
_FLOAT32_MAX = 3.402823466e38


def _normalized_string(value: Any, field_name: str, *, required: bool = False, lowercase: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    normalized = value.strip()
    if lowercase:
        normalized = normalized.lower()
    if required and not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return normalized


def _finite_float(value: Any, field_name: str, *, positive: bool, float32: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    if number < 0.0 or (positive and number == 0.0):
        raise ValueError(
            f"{field_name} must be greater than zero" if positive else f"{field_name} must not be negative"
        )
    if float32 and number > _FLOAT32_MAX:
        raise ValueError(f"{field_name} must not exceed float32 maximum")
    if number == 0.0:
        return 0.0
    return number


def canonical_skill_payload(
    skill_name: Any,
    target_name: Any = None,
    container_name: Any = None,
    place_name: Any = None,
    motion_direction: Any = None,
    motion_distance: Any = None,
    timeout_sec: Any = None,
    *,
    default_timeout_sec: Any,
) -> dict[str, str | float | None]:
    """Normalize a skill request into the exact payload hashed by the Gateway."""
    effective_timeout = default_timeout_sec if timeout_sec is None else timeout_sec
    normalized_distance = None
    if motion_distance is not None:
        normalized_distance = _finite_float(motion_distance, "motion_distance", positive=False, float32=True)

    return {
        "skill_name": _normalized_string(skill_name, "skill_name", required=True),
        "target_name": _normalized_string(target_name, "target_name"),
        "container_name": _normalized_string(container_name, "container_name"),
        "place_name": _normalized_string(place_name, "place_name"),
        "motion_direction": _normalized_string(motion_direction, "motion_direction", lowercase=True),
        "motion_distance": normalized_distance,
        "timeout_sec": _finite_float(effective_timeout, "timeout_sec", positive=True, float32=True),
    }


def skill_payload_hash(payload: dict[str, Any]) -> str:
    """Return the SHA256 digest of canonical JSON payload bytes."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def skill_goal_uuid(task_id: Any) -> uuid.UUID:
    """Create the deterministic UUID used to correlate a Gateway task goal."""
    normalized_task_id = _normalized_string(task_id, "task_id", required=True)
    return uuid.uuid5(uuid.NAMESPACE_URL, f"ibrobot:{normalized_task_id}")


def derive_skill_task_id(parent_task_id: str, skill_index: int) -> str:
    """Return the deterministic child action ID for one planned skill.

    Args:
        parent_task_id: Task-level ID supplied by the planned task.
        skill_index: Zero-based position of the skill in the plan.

    Returns:
        A normalized child ID in ``<parent>/skill/<1-based index>`` form.

    Raises:
        TypeError: If the skill index is not an integer or is a boolean.
        ValueError: If the parent ID is empty after trimming or the index is negative.
    """
    if not isinstance(parent_task_id, str) or not (parent_id := parent_task_id.strip()):
        raise ValueError("parent_task_id must be a non-empty string")
    if isinstance(skill_index, bool) or not isinstance(skill_index, int):
        raise TypeError("skill_index must be a nonnegative integer")
    if skill_index < 0:
        raise ValueError("skill_index must be a nonnegative integer")
    return f"{parent_id}/skill/{skill_index + 1:04d}"
