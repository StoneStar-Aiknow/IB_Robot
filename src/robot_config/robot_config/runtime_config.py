"""Synthesize a host-specific runtime config from the repository SSOT."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

RUNTIME_OWNED_ROBOT_KEYS = (
    "ros2_control",
    "peripherals",
    "contract",
    "teleoperation",
)


def _robot_mapping(config: dict[str, Any], label: str) -> dict[str, Any]:
    robot = config.get("robot")
    if not isinstance(robot, dict):
        raise ValueError(f"{label} config must contain a top-level 'robot' mapping")
    return robot


def synthesize_runtime_config(
    base_config: dict[str, Any],
    runtime_config: dict[str, Any],
    *,
    runtime_owned_keys: tuple[str, ...] = RUNTIME_OWNED_ROBOT_KEYS,
) -> dict[str, Any]:
    """Overlay host/calibration state onto the latest repository config.

    The repository config owns capabilities and safety policy. The runtime file
    owns only hardware enumeration, camera calibration/shape, the matching data
    contract, and optional leader-arm teleoperation.
    """
    base_robot = _robot_mapping(base_config, "base")
    runtime_robot = _robot_mapping(runtime_config, "runtime")
    base_name = str(base_robot.get("name", "")).strip()
    runtime_name = str(runtime_robot.get("name", "")).strip()
    if base_name and runtime_name and base_name != runtime_name:
        raise ValueError(f"robot name mismatch: base={base_name!r}, runtime={runtime_name!r}")

    merged = deepcopy(base_config)
    merged_robot = _robot_mapping(merged, "merged")
    for key in runtime_owned_keys:
        if key in runtime_robot:
            merged_robot[key] = deepcopy(runtime_robot[key])
    return merged
