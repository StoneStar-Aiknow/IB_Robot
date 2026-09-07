"""Canonical payload helpers shared by Gateway skill requests."""

from __future__ import annotations

import hashlib
import json
import math
import struct
import uuid
from typing import Any

# ROS 2 Humble float32 setters reject values beyond the IEEE-754 single-precision maximum,
# so motion_distance and timeout_sec must stay within float32 range to match ROS action fields.
_FLOAT32_MAX = 3.402823466e38
_SUPPORTED_REQUEST_SCHEMA_VERSIONS = frozenset({1, 2})
_SCHEMA_VERSION_ERROR = "SKILL_SCHEMA_INVALID: schema_version must be 1 or 2"
_ALLOWED_ARM_SIDES = frozenset({"left", "right", "auto"})


def validate_request_schema_version(value: Any) -> int:
    """Return a supported public request schema version or fail closed."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in _SUPPORTED_REQUEST_SCHEMA_VERSIONS:
        raise ValueError(_SCHEMA_VERSION_ERROR)
    return value


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
    if float32:
        number = struct.unpack("!f", struct.pack("!f", number))[0]
    if number == 0.0:
        return 0.0
    return number


def _optional_coordinate(value: Any, provided: Any, field_name: str) -> tuple[bool, float]:
    if provided is None:
        is_provided = value is not None
    elif isinstance(provided, bool):
        is_provided = provided
    else:
        raise ValueError(f"has_{field_name} must be a boolean")

    if not is_provided:
        if value is not None and _signed_finite_float(value, field_name) != 0.0:
            raise ValueError(f"{field_name} must be zero when has_{field_name} is false")
        return False, 0.0
    if value is None:
        raise ValueError(f"{field_name} must be provided when has_{field_name} is true")
    return True, _signed_finite_float(value, field_name)


def _signed_finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be a finite number")
    return 0.0 if number == 0.0 else number


def canonical_skill_payload(
    skill_name: Any,
    target_name: Any = None,
    container_name: Any = None,
    place_name: Any = None,
    motion_direction: Any = None,
    motion_distance: Any = None,
    arm_side: Any = None,
    imitation_duration_sec: Any = None,
    timeout_sec: Any = None,
    *,
    schema_version: Any,
    direction: Any = None,
    distance: Any = None,
    degree: Any = None,
    x: Any = None,
    y: Any = None,
    yaw: Any = None,
    has_x: Any = None,
    has_y: Any = None,
    has_yaw: Any = None,
    default_timeout_sec: Any,
) -> dict[str, str | float | bool | int | None]:
    """Normalize a skill request into the exact payload hashed by the Gateway."""
    effective_timeout = default_timeout_sec if timeout_sec is None else timeout_sec
    normalized_distance = None
    if motion_distance is not None:
        normalized_distance = _finite_float(motion_distance, "motion_distance", positive=False, float32=True)
    normalized_arm_side = _normalized_string(arm_side, "arm_side", lowercase=True)
    if normalized_arm_side and normalized_arm_side not in _ALLOWED_ARM_SIDES:
        raise ValueError("arm_side must be left, right, or auto")
    normalized_imitation_duration = 0.0
    if imitation_duration_sec is not None:
        normalized_imitation_duration = _finite_float(
            imitation_duration_sec, "imitation_duration_sec", positive=True, float32=True
        )
    coordinate_x = _optional_coordinate(x, has_x, "x")
    coordinate_y = _optional_coordinate(y, has_y, "y")
    coordinate_yaw = _optional_coordinate(yaw, has_yaw, "yaw")
    normalized_direction = _normalized_string(direction, "direction", lowercase=True)
    normalized_nav_distance = _finite_float(0.0 if distance is None else distance, "distance", positive=False)
    normalized_degree = _finite_float(0.0 if degree is None else degree, "degree", positive=False)

    payload: dict[str, str | float | bool | int | None] = {
        "schema_version": validate_request_schema_version(schema_version),
        "skill_name": _normalized_string(skill_name, "skill_name", required=True),
        "target_name": _normalized_string(target_name, "target_name"),
        "container_name": _normalized_string(container_name, "container_name"),
        "place_name": _normalized_string(place_name, "place_name"),
        "motion_direction": _normalized_string(motion_direction, "motion_direction", lowercase=True),
        "motion_distance": normalized_distance,
    }
    if normalized_arm_side or normalized_imitation_duration:
        payload.update(
            {
                "arm_side": normalized_arm_side,
                "imitation_duration_sec": normalized_imitation_duration,
            }
        )
    if (
        normalized_direction
        or normalized_nav_distance
        or normalized_degree
        or any(coordinate[0] for coordinate in (coordinate_x, coordinate_y, coordinate_yaw))
    ):
        payload.update(
            {
                "direction": normalized_direction,
                "distance": normalized_nav_distance,
                "degree": normalized_degree,
                "has_x": coordinate_x[0],
                "x": coordinate_x[1],
                "has_y": coordinate_y[0],
                "y": coordinate_y[1],
                "has_yaw": coordinate_yaw[0],
                "yaw": coordinate_yaw[1],
            }
        )
    payload["timeout_sec"] = _finite_float(effective_timeout, "timeout_sec", positive=True, float32=True)
    return payload


def skill_payload_hash(payload: dict[str, Any]) -> str:
    """Return the SHA256 digest of canonical JSON payload bytes."""
    validate_request_schema_version(payload.get("schema_version"))
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
