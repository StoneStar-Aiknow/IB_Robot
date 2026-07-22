"""Utility functions for robot_config package.

This module contains common utility functions used across the robot_config package:
- Path resolution (ROS-style substitutions)
- Boolean parsing
- Type conversion helpers
- Joint configuration validation
"""

import hashlib
import json
import logging
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def resolve_ros_path(path):
    """Resolve ROS-style path substitutions like $(find pkg) and $(env VAR).

    Handles ROS path substitution syntax:
    - $(find package_name): Resolves to package share directory
    - $(env VAR_NAME): Resolves to environment variable value

    Args:
        path: Path string that may contain $(find package) or $(env VAR)

    Returns:
        Resolved path string. Returns original path if it's None or empty.

    Example:
        >>> resolve_ros_path("$(find so101_hardware)/config/controllers.yaml")
        "/home/user/workspace/install/share/so101_hardware/config/controllers.yaml"

        >>> resolve_ros_path("$(env HOME)/.config/robot.yaml")
        "/home/user/.config/robot.yaml"
    """
    if not path:
        return path

    # Resolve $(find package)
    find_pattern = re.compile(r"\$\(find\s+(\w+)\)")
    for match in find_pattern.finditer(path):
        pkg_name = match.group(1)
        try:
            from ament_index_python.packages import get_package_share_directory

            pkg_path = get_package_share_directory(pkg_name)
            path = path.replace(f"$(find {pkg_name})", pkg_path)
        except Exception as e:
            logger.warning(f"Could not find package '{pkg_name}': {e}")

    # Resolve $(env VAR)
    env_pattern = re.compile(r"\$\(env\s+(\w+)\)")
    for match in env_pattern.finditer(path):
        var_name = match.group(1)
        var_value = os.environ.get(var_name, "")
        path = path.replace(f"$(env {var_name})", var_value)
        if not var_value:
            logger.info(f"WARNING: Environment variable '{var_name}' is not set or empty")

    return path


def prepare_writable_file_path(path):
    """Resolve a file path and ensure its parent directory exists.

    This is intended for output paths that will be created by downstream tools,
    such as RTAB-Map databases or exported artifacts.

    Args:
        path: File path that may contain ROS-style substitutions.

    Returns:
        Resolved path string. Returns original path if it's None or empty.
    """
    resolved_path = resolve_ros_path(path)
    if not resolved_path:
        return resolved_path

    parent_dir = Path(resolved_path).expanduser().parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    return resolved_path


def parse_bool(value, default=False):
    """Parse various value types to boolean with robust handling.

    Handles multiple input formats:
    - Strings: "true", "TRUE", "True", "1", "yes", "on" -> True
    - Strings: "false", "FALSE", "False", "0", "no", "off" -> False
    - Booleans: True/False -> as-is
    - Numbers: 1/0 -> True/False
    - None: -> default value

    Args:
        value: Input value to parse (string, bool, int, or None)
        default: Default value if input is None or unparseable

    Returns:
        Boolean value

    Example:
        >>> parse_bool("true")
        True
        >>> parse_bool("FALSE")
        False
        >>> parse_bool(True)
        True
        >>> parse_bool(None, default=False)
        False
    """
    if value is None:
        return default

    # Handle boolean types directly
    if isinstance(value, bool):
        return value

    # Convert to string and normalize
    str_value = str(value).strip().lower()

    # Check for true-like values
    if str_value in ("true", "1", "yes", "on"):
        return True

    # Check for false-like values
    if str_value in ("false", "0", "no", "off", ""):
        return False

    # Unknown value, return default
    return default


def validate_joint_config(robot_config):
    """Validate joint configuration across controllers and robot config.

    Implements DRY principle by checking that joint definitions are consistent
    between robot_config and controller configuration files.

    Args:
        robot_config: Robot configuration dict with joints and ros2_control sections

    Returns:
        True if validation passes, False otherwise

    Raises:
        Prints warnings/errors but does not raise exceptions to avoid blocking startup
    """
    logger.info("[robot_config] ========== Joint Configuration Validation ==========")

    joints_config = robot_config.get("joints", {})
    if not joints_config:
        logger.info("[robot_config] WARNING: No 'joints' configuration found")
        return True

    expected_arm_joints = set(joints_config.get("arm", []))
    expected_gripper_joints = set(joints_config.get("gripper", []))
    expected_base_joints = set(joints_config.get("base", []))
    expected_all_joints = set(joints_config.get("all", []))
    expected_broadcaster_joints = expected_all_joints | expected_base_joints

    logger.info("Canonical joints from robot_config:")
    logger.info(f"  arm: {sorted(expected_arm_joints)}")
    logger.info(f"  gripper: {sorted(expected_gripper_joints)}")
    logger.info(f"  base: {sorted(expected_base_joints)}")
    logger.info(f"  all: {sorted(expected_all_joints)}")

    # Load controllers configuration
    ros2_control_config = robot_config.get("ros2_control", {})
    controllers_config_path = ros2_control_config.get("controllers_config", "")

    if not controllers_config_path:
        logger.info("[robot_config] WARNING: No controllers_config path specified")
        return True

    controllers_config_path = resolve_ros_path(controllers_config_path)

    if not Path(controllers_config_path).exists():
        logger.info(f"WARNING: Controllers config not found at {controllers_config_path}")
        return True

    # Load controllers YAML
    try:
        with open(controllers_config_path) as f:
            controllers_yaml = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load controllers config: {e}")
        return False

    validation_passed = True
    controllers_checked = 0

    # Check arm_position_controller
    arm_pos_ctrl = controllers_yaml.get("arm_position_controller", {}).get("ros__parameters", {})
    if arm_pos_ctrl:
        ctrl_joints = set(arm_pos_ctrl.get("joints", []))
        if ctrl_joints != expected_arm_joints:
            logger.error("[robot_config] ERROR: arm_position_controller joints mismatch!")
            validation_passed = False
        else:
            logger.info("✓ arm_position_controller joints match")
        controllers_checked += 1

    # Check gripper_position_controller
    grip_pos_ctrl = controllers_yaml.get("gripper_position_controller", {}).get("ros__parameters", {})
    if grip_pos_ctrl:
        ctrl_joints = set(grip_pos_ctrl.get("joints", []))
        if ctrl_joints != expected_gripper_joints:
            logger.error("[robot_config] ERROR: gripper_position_controller joints mismatch!")
            validation_passed = False
        else:
            logger.info("✓ gripper_position_controller joints match")
        controllers_checked += 1

    # Check joint_state_broadcaster
    jsb_ctrl = controllers_yaml.get("joint_state_broadcaster", {}).get("ros__parameters", {})
    if jsb_ctrl:
        ctrl_joints = set(jsb_ctrl.get("joints", []))
        if ctrl_joints != expected_broadcaster_joints:
            logger.error("[robot_config] ERROR: joint_state_broadcaster joints mismatch!")
            validation_passed = False
        else:
            logger.info("✓ joint_state_broadcaster joints match")
        controllers_checked += 1

    logger.info(f"Validated {controllers_checked} controller configurations")

    if validation_passed:
        logger.info("[robot_config] ✓ All joint configurations are consistent")
    else:
        logger.info("[robot_config] ✗ Joint configuration validation FAILED")

    logger.info("[robot_config] =========================================================")

    return validation_passed


def prepare_lerobot_env():
    """Prepare environment with lerobot PYTHONPATH."""
    env = os.environ.copy()
    workspace_path = os.environ.get("WORKSPACE", os.getcwd())
    lerobot_src = os.path.join(workspace_path, "libs/lerobot/src")

    if os.path.exists(lerobot_src):
        current_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{lerobot_src}:{current_pp}" if current_pp else lerobot_src

    return env


# ---------------------------------------------------------------------------
# Joint unit-conversion helpers  (LeRobot percentage  ↔  ros2_control radians)
# ---------------------------------------------------------------------------

# Each entry: (rad_min, rad_max, pct_span, pct_offset)
JointConversionEntry = tuple[float, float, float, float]
CalibrationSnapshot = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CalibrationSourceSpec:
    """Resolved calibration file with the namespace needed for numeric keys."""

    resolved_path: str
    namespace: str


CalibrationSource = str | os.PathLike[str] | CalibrationSourceSpec | dict[str, Any] | list[Any] | tuple[Any, ...]

_TICKS_PER_RAD = 4096.0 / (2.0 * math.pi)


# Supported LeRobot motor normalization modes.
# Keep in sync with the YAML ``lerobot_norm_mode`` option.
NORM_MODE_RANGE = "range_m100_100"  # arm [-100,+100], gripper [0,100]
NORM_MODE_DEGREES = "degrees"  # arm centred degrees, gripper [0,100]
NORM_MODE_NONE = "none"  # pass-through (no conversion)

_MODEL_RESOLUTION = 4096  # Feetech STS3215 12-bit encoder
_CALIBRATION_SNAPSHOT_FIELDS = (
    "id",
    "model",
    "drive_mode",
    "homing_offset",
    "range_min",
    "range_max",
)


def normalize_lerobot_norm_mode(norm_mode: str) -> str:
    """Normalize and validate LeRobot motor normalization mode."""
    mode = str(norm_mode or NORM_MODE_RANGE).strip().lower()
    if mode not in (NORM_MODE_RANGE, NORM_MODE_DEGREES, NORM_MODE_NONE):
        raise ValueError(
            f"Unsupported LeRobot normalization mode '{norm_mode}'. "
            f"Expected one of: {NORM_MODE_RANGE}, {NORM_MODE_DEGREES}, {NORM_MODE_NONE}"
        )
    return mode


def resolve_joint_names_from_config(robot_config: dict[str, Any]) -> list[str]:
    """Resolve ordered joint names from raw robot_config YAML content."""
    ros2_control = robot_config.get("ros2_control", {}) or {}
    joints_cfg = robot_config.get("joints", {}) or {}
    joint_names = ros2_control.get("joint_names") or joints_cfg.get("all") or []
    return [str(name) for name in joint_names]


def _dedupe_strings(values: list[Any]) -> list[str]:
    """Return non-empty string values in first-seen order."""
    seen = set()
    result: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def resolve_gripper_joints_from_config(robot_config: dict[str, Any]) -> list[str]:
    """Resolve gripper joint names from raw robot_config YAML content."""
    ros2_control = robot_config.get("ros2_control", {}) or {}
    joints_cfg = robot_config.get("joints", {}) or {}
    explicit = ros2_control.get("gripper_joints")
    if explicit:
        return _dedupe_strings(list(explicit))

    gripper_joints = list(joints_cfg.get("gripper") or [])
    gripper_joints.extend(joints_cfg.get("left_gripper") or [])
    gripper_joints.extend(joints_cfg.get("right_gripper") or [])
    return _dedupe_strings(gripper_joints)


def _make_calibration_source_spec(
    path: Any,
    namespace: Any,
) -> CalibrationSourceSpec:
    resolved_path = resolve_ros_path(str(path or "").strip())
    if not resolved_path:
        raise ValueError("Calibration source path must be non-empty")

    namespace_text = str(namespace or "").strip()
    if not namespace_text:
        raise ValueError("Calibration source namespace must be non-empty")

    return CalibrationSourceSpec(
        resolved_path=resolved_path,
        namespace=namespace_text,
    )


def _coerce_calibration_source_spec(value: Any) -> CalibrationSourceSpec:
    if isinstance(value, CalibrationSourceSpec):
        return value
    if isinstance(value, dict):
        raw_path = value.get("resolved_path") or value.get("path") or value.get("file")
        return _make_calibration_source_spec(
            path=raw_path,
            namespace=value.get("namespace"),
        )
    raise TypeError(f"Unsupported calibration source spec: {value!r}")


def _is_calibration_source_spec(value: Any) -> bool:
    if isinstance(value, CalibrationSourceSpec):
        return True
    return (
        isinstance(value, dict)
        and bool(value.get("namespace"))
        and bool(value.get("resolved_path") or value.get("path") or value.get("file"))
    )


def _validate_calibration_source_namespaces(specs: list[CalibrationSourceSpec]) -> list[CalibrationSourceSpec]:
    seen: set[str] = set()
    for spec in specs:
        if spec.namespace in seen:
            raise ValueError(f"Duplicate calibration source namespace: {spec.namespace}")
        seen.add(spec.namespace)
    return specs


def resolve_calibration_source_specs_from_config(robot_config: dict[str, Any]) -> list[CalibrationSourceSpec]:
    """Resolve calibration source specs from raw robot_config.

    ``ros2_control.calib_file`` is the legacy single-arm source and maps to
    ``arm``.  Multi-source configs use namespace-suffixed
    ``ros2_control.xacro_args.calib_file_<namespace>`` keys, for example
    ``calib_file_left``, ``calib_file_right``, and ``calib_file_1``.
    """
    ros2_control = robot_config.get("ros2_control", {}) or {}
    calib_file = str(ros2_control.get("calib_file", "") or "").strip()

    xacro_args = ros2_control.get("xacro_args", {}) or {}
    named_calib_files: list[tuple[str, Any]] = []
    for key, value in xacro_args.items():
        match = re.fullmatch(r"calib_file_([A-Za-z0-9_]+)", str(key))
        if match and str(value or "").strip():
            named_calib_files.append((match.group(1), value))
    named_calib_files.sort(key=lambda item: item[0])

    if calib_file and named_calib_files:
        raise ValueError(
            "ros2_control.calib_file cannot be combined with ros2_control.xacro_args.calib_file_<namespace>"
        )
    if calib_file:
        return [_make_calibration_source_spec(calib_file, "arm")]

    specs: list[CalibrationSourceSpec] = []
    for namespace, path in named_calib_files:
        specs.append(_make_calibration_source_spec(path, namespace))
    return _validate_calibration_source_namespaces(specs)


def resolve_calibration_paths_from_config(robot_config: dict[str, Any]) -> list[str]:
    """Resolve calibration file path list from raw robot_config."""
    return [spec.resolved_path for spec in resolve_calibration_source_specs_from_config(robot_config)]


def resolve_calibration_path_from_config(robot_config: dict[str, Any]) -> str:
    """Resolve calibration file path(s) as a legacy pathsep string.

    Path strings lose explicit source namespaces; runtime conversion callers
    should prefer ``resolve_calibration_source_specs_from_config``.
    """
    return os.pathsep.join(resolve_calibration_paths_from_config(robot_config))


def resolve_lerobot_norm_mode(
    robot_config: dict[str, Any],
    preferred_control_mode: str | None = None,
) -> str:
    """Resolve the LeRobot normalization mode from robot_config semantics."""
    recording_cfg = robot_config.get("recording", {}) or {}
    explicit_mode = recording_cfg.get("lerobot_norm_mode")
    if explicit_mode:
        return normalize_lerobot_norm_mode(str(explicit_mode))

    return NORM_MODE_RANGE


def _normalize_legacy_calibration_paths(calib_file: CalibrationSource) -> list[str]:
    if isinstance(calib_file, list | tuple):
        return [resolve_ros_path(str(path)) for path in calib_file if str(path or "").strip()]
    text = str(calib_file or "").strip()
    if not text:
        return []
    parts = [part for part in text.split(os.pathsep) if part]
    return [resolve_ros_path(part) for part in parts]


def _normalize_explicit_calibration_source_specs(calib_file: CalibrationSource) -> list[CalibrationSourceSpec]:
    if _is_calibration_source_spec(calib_file):
        return [_coerce_calibration_source_spec(calib_file)]
    if isinstance(calib_file, list | tuple) and all(_is_calibration_source_spec(source) for source in calib_file):
        return _validate_calibration_source_namespaces(
            [_coerce_calibration_source_spec(source) for source in calib_file]
        )
    return []


def _normalize_legacy_calibration_source_specs(calib_file: CalibrationSource) -> list[CalibrationSourceSpec]:
    resolved_paths = _normalize_legacy_calibration_paths(calib_file)
    if len(resolved_paths) > 2:
        raise ValueError(
            "Legacy calibration path inputs support at most two calibration sources; "
            "use explicit calibration source specs with namespace for more sources"
        )
    namespaces = ("arm",) if len(resolved_paths) == 1 else ("left", "right")
    return [_make_calibration_source_spec(path, namespaces[index]) for index, path in enumerate(resolved_paths)]


def _normalize_calibration_source_specs(calib_file: CalibrationSource) -> list[CalibrationSourceSpec]:
    explicit_specs = _normalize_explicit_calibration_source_specs(calib_file)
    if explicit_specs:
        return explicit_specs
    if isinstance(calib_file, list | tuple) and any(_is_calibration_source_spec(source) for source in calib_file):
        raise TypeError("Calibration source list must not mix explicit specs with legacy path strings")
    return _normalize_legacy_calibration_source_specs(calib_file)


def _load_single_calibration_file(resolved_path: str) -> dict[str, Any]:
    calib_path = Path(resolved_path).expanduser().resolve()
    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with calib_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Calibration file must contain a JSON object: {calib_path}")
    return data


def _namespace_numeric_calibration_keys(
    calibration: dict[str, Any],
    namespace: str,
) -> dict[str, Any]:
    namespaced: dict[str, Any] = {}
    numeric_items: list[tuple[int, str, Any]] = []
    for key, value in calibration.items():
        if str(key).isdigit():
            numeric_items.append((int(str(key)), str(key), value))
        else:
            namespaced[str(key)] = value

    for index, key, value in sorted(numeric_items, key=lambda item: item[0]):
        namespaced_key = f"joint{index}_{namespace}"
        if namespaced_key in namespaced:
            raise ValueError(f"Calibration key collision while namespacing numeric key '{key}' as '{namespaced_key}'")
        namespaced[namespaced_key] = value
    return namespaced


def _merge_calibration_data(merged: dict[str, Any], data: dict[str, Any], resolved_path: str) -> None:
    for key, value in data.items():
        if key in merged:
            raise ValueError(f"Calibration key collision for '{key}' while loading {resolved_path}")
        merged[key] = value


def load_calibration_data(calib_file: CalibrationSource) -> dict[str, Any]:
    """Load one or more calibration JSON files from disk.

    Spec inputs preserve explicit namespaces. Plain strings remain legacy mode
    and infer namespaces from the number of pathsep-separated files.
    """
    return _load_calibration_data_from_specs(_normalize_calibration_source_specs(calib_file))


def _resolve_calibration_key(calibration: dict[str, Any], joint_name: str) -> str:
    if joint_name in calibration:
        return joint_name
    if joint_name.isdigit() and str(int(joint_name)) == joint_name:
        arm_key = f"joint{int(joint_name)}_arm"
        if arm_key in calibration:
            return arm_key
    raise KeyError(f"Joint '{joint_name}' missing from calibration data")


def extract_calibration_snapshot(
    calibration: dict[str, Any],
    joint_names: list[str],
) -> CalibrationSnapshot:
    """Extract a canonical calibration snapshot for the selected joints."""
    snapshot: CalibrationSnapshot = {}
    for joint_name in [str(name) for name in joint_names]:
        calibration_key = _resolve_calibration_key(calibration, joint_name)
        entry = calibration[calibration_key]
        if not isinstance(entry, dict):
            raise ValueError(f"Calibration entry for joint '{joint_name}' must be an object")

        joint_snapshot: dict[str, Any] = {}
        for field in _CALIBRATION_SNAPSHOT_FIELDS:
            if field not in entry:
                continue
            if field == "model":
                joint_snapshot[field] = str(entry[field])
            else:
                joint_snapshot[field] = int(entry[field])

        if "range_min" not in joint_snapshot or "range_max" not in joint_snapshot:
            raise KeyError(f"Calibration entry for joint '{joint_name}' must contain range_min/range_max")
        joint_snapshot.setdefault("drive_mode", 0)
        snapshot[joint_name] = joint_snapshot

    return snapshot


def lerobot_conversion_fingerprint(
    calibration: CalibrationSnapshot,
    joint_names: list[str],
    gripper_joints: list[str] | None = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> str:
    """Compute a stable fingerprint for LeRobot conversion semantics."""
    mode = normalize_lerobot_norm_mode(norm_mode)
    ordered_joints = [str(name) for name in joint_names]
    ordered_gripper_joints = [str(name) for name in (gripper_joints or [])]

    payload: dict[str, Any] = {
        "norm_mode": mode,
        "joint_names": ordered_joints,
        "gripper_joints": ordered_gripper_joints,
        "joints": {},
    }

    for joint_name in ordered_joints:
        entry = calibration.get(joint_name, {})
        joint_payload: dict[str, int] = {}
        if mode != NORM_MODE_NONE:
            if "range_min" not in entry or "range_max" not in entry:
                raise KeyError(f"Calibration snapshot for joint '{joint_name}' must contain range_min/range_max")
            joint_payload["range_min"] = int(entry["range_min"])
            joint_payload["range_max"] = int(entry["range_max"])
            joint_payload["drive_mode"] = int(entry.get("drive_mode", 0))
        payload["joints"][joint_name] = joint_payload

    json_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(json_str.encode("utf-8")).hexdigest()[:16]


def _load_calibration_data_from_specs(specs: list[CalibrationSourceSpec]) -> dict[str, Any]:
    """Load calibration data from already-normalized source specs."""
    if not specs:
        raise FileNotFoundError("Calibration file path is empty")

    merged: dict[str, Any] = {}
    for spec in specs:
        data = _load_single_calibration_file(spec.resolved_path)
        data = _namespace_numeric_calibration_keys(data, spec.namespace)
        _merge_calibration_data(merged, data, spec.resolved_path)
    return merged


def build_lerobot_conversion_metadata(
    calib_file: CalibrationSource,
    joint_names: list[str],
    gripper_joints: list[str] | None = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> dict[str, Any]:
    """Build a dataset-storable snapshot of LeRobot conversion semantics."""
    mode = normalize_lerobot_norm_mode(norm_mode)
    ordered_joints = [str(name) for name in joint_names]
    ordered_gripper_joints = [str(name) for name in (gripper_joints or [])]

    metadata: dict[str, Any] = {
        "norm_mode": mode,
        "joint_names": ordered_joints,
        "gripper_joints": ordered_gripper_joints,
    }

    if mode == NORM_MODE_NONE:
        metadata["conversion_fingerprint"] = lerobot_conversion_fingerprint(
            calibration={},
            joint_names=ordered_joints,
            gripper_joints=ordered_gripper_joints,
            norm_mode=mode,
        )
        return metadata

    calibration_specs = _normalize_calibration_source_specs(calib_file)
    calibration = _load_calibration_data_from_specs(calibration_specs)
    snapshot = extract_calibration_snapshot(calibration, ordered_joints)
    resolved_sources = [str(Path(spec.resolved_path).expanduser().resolve()) for spec in calibration_specs]
    metadata["calibration_source"] = resolved_sources[0] if resolved_sources else ""
    metadata["calibration_sources"] = resolved_sources
    metadata["calibration"] = snapshot
    metadata["conversion_fingerprint"] = lerobot_conversion_fingerprint(
        calibration=snapshot,
        joint_names=ordered_joints,
        gripper_joints=ordered_gripper_joints,
        norm_mode=mode,
    )
    return metadata


def build_joint_conversion_table_from_calibration(
    calibration: dict[str, Any],
    joint_names: list[str],
    gripper_joints: list[str] | None = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> list[JointConversionEntry]:
    """Build a conversion table from calibration content already in memory."""
    mode = normalize_lerobot_norm_mode(norm_mode)
    if mode == NORM_MODE_NONE:
        return []

    ordered_joints = [str(name) for name in joint_names]
    gripper_joint_set = {str(name) for name in (gripper_joints or [])}
    table: list[JointConversionEntry] = []

    for joint_name in ordered_joints:
        calibration_key = _resolve_calibration_key(calibration, joint_name)
        entry = calibration[calibration_key]
        if not isinstance(entry, dict):
            raise ValueError(f"Calibration entry for joint '{joint_name}' must be an object")

        tick_min = int(entry["range_min"])
        tick_max = int(entry["range_max"])

        if mode == NORM_MODE_DEGREES and joint_name not in gripper_joint_set:
            mid = (tick_min + tick_max) / 2.0
            max_res = _MODEL_RESOLUTION - 1  # 4095
            deg_at_tick_min = (tick_min - mid) * 360.0 / max_res
            deg_at_tick_max = (tick_max - mid) * 360.0 / max_res

            rad_min = (tick_min - 2048.0) / _TICKS_PER_RAD
            rad_max = (tick_max - 2048.0) / _TICKS_PER_RAD
            span = deg_at_tick_max - deg_at_tick_min
            offset = deg_at_tick_min
        else:
            rad_min = (tick_min - 2048.0) / _TICKS_PER_RAD
            rad_max = (tick_max - 2048.0) / _TICKS_PER_RAD

            if joint_name in gripper_joint_set:
                span = 100.0
                offset = 0.0
            else:
                span = 200.0
                offset = -100.0

        table.append((rad_min, rad_max, span, offset))

    return table


def _joint_limits_from_urdf(root: ET.Element, joint_name: str) -> tuple[float, float]:
    """Resolve lower/upper position limits (radians) for one joint from URDF XML.

    Looks first at ``<joint><limit lower=... upper=.../></joint>`` and falls
    back to the ``ros2_control`` command-interface ``min``/``max`` params when
    the joint is not declared as a plain URDF joint (typical for grippers).
    """
    for joint in root.findall(".//joint"):
        if joint.get("name") != joint_name:
            continue

        limit = joint.find("limit")
        if limit is not None and limit.get("lower") is not None and limit.get("upper") is not None:
            return float(limit.get("lower")), float(limit.get("upper"))

    for joint in root.findall(".//ros2_control/joint"):
        if joint.get("name") != joint_name:
            continue

        command = joint.find("./command_interface[@name='position']")
        if command is None:
            continue

        params = {param.get("name"): param.text for param in command.findall("param")}
        if params.get("min") is not None and params.get("max") is not None:
            return float(params["min"]), float(params["max"])

    raise KeyError(f"Joint '{joint_name}' missing lower/upper limits in URDF")


def build_joint_conversion_table_from_urdf(
    urdf_xml: str,
    joint_names: list[str],
    gripper_joints: list[str] | None = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> list[JointConversionEntry]:
    """Build a conversion table from URDF joint limits (sim mode).

    Sister function of :func:`build_joint_conversion_table_from_calibration`.
    Used in simulation where there is no physical calibration file: the URDF
    joint ``limit`` (or the ``ros2_control`` command-interface min/max)
    provides the runtime position range in radians.

    The output contract is identical to
    :func:`build_joint_conversion_table_from_calibration`, so callers can use
    either table transparently. In ``degrees`` mode the radian range is scaled
    by ``_MODEL_RESOLUTION / (_MODEL_RESOLUTION - 1)`` (= 4096/4095) so the
    resulting ``span`` matches the tick-based formula used on real hardware.
    This keeps sim-trained and real-deployed models numerically aligned.
    """
    mode = normalize_lerobot_norm_mode(norm_mode)
    if mode == NORM_MODE_NONE:
        return []

    root = ET.fromstring(urdf_xml)
    gripper_joint_set = {str(name) for name in (gripper_joints or [])}
    table: list[JointConversionEntry] = []

    for joint_name in [str(name) for name in joint_names]:
        rad_min, rad_max = _joint_limits_from_urdf(root, joint_name)
        if rad_max <= rad_min:
            raise ValueError(f"Joint '{joint_name}' has invalid URDF limits: lower={rad_min}, upper={rad_max}")

        if joint_name in gripper_joint_set:
            span = 100.0
            offset = 0.0
        elif mode == NORM_MODE_DEGREES:
            span = math.degrees(rad_max - rad_min) * _MODEL_RESOLUTION / (_MODEL_RESOLUTION - 1)
            offset = -span / 2.0
        else:
            span = 200.0
            offset = -100.0

        table.append((rad_min, rad_max, span, offset))

    return table


def build_joint_conversion_table(
    calib_file: CalibrationSource,
    joint_names: list[str],
    gripper_joints: list[str] | None = None,
    norm_mode: str = NORM_MODE_RANGE,
) -> list[JointConversionEntry]:
    """Build per-joint ``(rad_min, rad_max, lerobot_span, lerobot_offset)``.

    The C++ hardware layer converts  ticks ↔ radians  using a fixed formula::

        rad  = (ticks - 2048) / (4096 / 2π)
        ticks = rad * (4096 / 2π) + 2048

    LeRobot normalises ticks differently depending on *norm_mode*:

    **range_m100_100** (default)
        * Arm joints  ``RANGE_M100_100``:  ``pct = (t-tmin)/(tmax-tmin)*200 - 100``
        * Gripper      ``RANGE_0_100``:     ``pct = (t-tmin)/(tmax-tmin)*100``

    **degrees**
        * Arm joints: ``deg = (t - mid) * 360 / 4095``  where ``mid=(tmin+tmax)/2``
        * Gripper joints keep ``RANGE_0_100`` semantics so model action ``0``
          maps to the closed calibration end and ``100`` maps to open.

    **none**
        No conversion – returns an empty table so the caller does a pass-through.

    Parameters
    ----------
    calib_file : CalibrationSource
        Explicit specs or legacy path input. Specs preserve namespaces; path
        strings infer namespaces from file count for compatibility.
    joint_names : list[str]
        Ordered joint identifiers (e.g. ``["1","2",…,"6"]``).
    gripper_joints : list[str] | None
        Selects RANGE_0_100 semantics for these joints in ``range_m100_100``
        and ``degrees`` modes.
    norm_mode : str
        One of ``"range_m100_100"``, ``"degrees"``, ``"none"``.

    Returns
    -------
    list[JointConversionEntry]
        One ``(rad_min, rad_max, span, offset)`` per joint.  The linear
        mapping is::

            lerobot_val  = (rad - rad_min) / (rad_max - rad_min) * span + offset
            rad          = (lerobot_val - offset) / span * (rad_max - rad_min) + rad_min

        For *degrees* mode ``rad_min / rad_max`` are the rad equivalents of the
        degree endpoints, and ``span / offset`` encode the degree range so that
        the same linear formula works.
    """
    calibration = load_calibration_data(calib_file)
    return build_joint_conversion_table_from_calibration(
        calibration=calibration,
        joint_names=joint_names,
        gripper_joints=gripper_joints,
        norm_mode=norm_mode,
    )
