"""Synthesize a host-specific runtime config from the repository SSOT."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

RUNTIME_OWNED_ROBOT_KEYS = ("ros2_control", "peripherals", "contract", "teleoperation")
_ROS2_CONTROL_FIELDS = ("port", "calib_file", "reset_positions")
_PERIPHERAL_FIELDS = (
    "serial_number",
    "device_index",
    "device_name",
    "port",
    "width",
    "height",
    "fps",
    "pixel_format",
    "camera_info_url",
    "calibration_url",
    "transform",
)
_TELEOPERATION_FIELDS = ("active_device", "devices")


def _robot_mapping(config: dict[str, Any], label: str) -> dict[str, Any]:
    robot = config.get("robot")
    if not isinstance(robot, dict):
        raise ValueError(f"{label} config must contain a top-level 'robot' mapping")
    return robot


def _copy_fields(target: dict[str, Any], source: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if field in source:
            target[field] = deepcopy(source[field])


def _merge_peripherals(base: Any, runtime: Any) -> Any:
    if not isinstance(base, list) or not isinstance(runtime, list):
        return deepcopy(base)
    runtime_by_name = {
        str(item.get("name", "")): item for item in runtime if isinstance(item, dict) and str(item.get("name", ""))
    }
    merged = deepcopy(base)
    for peripheral in merged:
        if not isinstance(peripheral, dict):
            continue
        runtime_peripheral = runtime_by_name.get(str(peripheral.get("name", "")))
        if runtime_peripheral is not None:
            _copy_fields(peripheral, runtime_peripheral, _PERIPHERAL_FIELDS)
    return merged


def _merge_contract(base: Any, runtime: Any) -> Any:
    if not isinstance(base, dict) or not isinstance(runtime, dict):
        return deepcopy(base)
    merged = deepcopy(base)
    runtime_observations = {
        str(item.get("key", "")): item
        for item in runtime.get("observations", [])
        if isinstance(item, dict) and str(item.get("key", ""))
    }
    for observation in merged.get("observations", []):
        if not isinstance(observation, dict):
            continue
        runtime_observation = runtime_observations.get(str(observation.get("key", "")))
        runtime_image = runtime_observation.get("image") if isinstance(runtime_observation, dict) else None
        if isinstance(runtime_image, dict) and "resize" in runtime_image:
            image = observation.setdefault("image", {})
            if isinstance(image, dict):
                image["resize"] = deepcopy(runtime_image["resize"])
    return merged


def synthesize_runtime_config(base_config: dict[str, Any], runtime_config: dict[str, Any]) -> dict[str, Any]:
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
    base_ros2_control = merged_robot.get("ros2_control")
    runtime_ros2_control = runtime_robot.get("ros2_control")
    if isinstance(base_ros2_control, dict) and isinstance(runtime_ros2_control, dict):
        _copy_fields(base_ros2_control, runtime_ros2_control, _ROS2_CONTROL_FIELDS)

    merged_robot["peripherals"] = _merge_peripherals(merged_robot.get("peripherals"), runtime_robot.get("peripherals"))
    merged_robot["contract"] = _merge_contract(merged_robot.get("contract"), runtime_robot.get("contract"))

    base_teleoperation = merged_robot.get("teleoperation")
    runtime_teleoperation = runtime_robot.get("teleoperation")
    if isinstance(base_teleoperation, dict) and isinstance(runtime_teleoperation, dict):
        _copy_fields(base_teleoperation, runtime_teleoperation, _TELEOPERATION_FIELDS)
    return merged
