"""Camera ISP override loader.

Loads per-camera ISP overrides from disk so that color calibration produced by
``camera_isp_calibrator`` (in dataset_tools) can be applied automatically at
launch without modifying the YAML SSOT.

Storage convention (mirrors ``~/.ros/ibrobot/sim_camera_overrides/``):
    ``~/.ros/ibrobot/camera_isp_overrides/{camera_name}.json``

Schema: a flat JSON object mapping a subset of the 11 known V4L2/usb_cam ISP
parameters to their values. Unknown keys are dropped with a WARN log; missing
file → empty dict (logged at INFO). Range / type errors per-key are logged
at WARN and the offending key is dropped (other valid keys still applied).

The loader is intentionally fail-safe: any exception returns ``{}`` so a
corrupt or unreadable override file never breaks launch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.camera_isp_overrides")


# Mapping: param name -> (expected python type, valid range or None)
# Ranges are conservative bounds spanning typical UVC device caps; usb_cam
# itself / V4L2 will further clip to the actual device range at apply time.
_BOOL_KEYS = {"auto_white_balance", "autoexposure", "autofocus"}
_INT_KEYS = {
    "brightness": (-255, 255),
    "contrast": (0, 255),
    "saturation": (0, 255),
    "sharpness": (0, 255),
    "gain": (0, 255),
    "white_balance": (2000, 10000),  # kelvin
    "exposure": (0, 20000),          # v4l2 absolute exposure units
    "focus": (0, 1023),
}

_ALLOWED_KEYS = set(_INT_KEYS) | set(_BOOL_KEYS)


def _override_path(camera_name: str) -> Path:
    """Return path to the per-camera override JSON file.

    Honours $ROS_HOME if set, otherwise falls back to ``~/.ros``.
    """
    ros_home = os.environ.get("ROS_HOME")
    base = Path(ros_home) if ros_home else Path.home() / ".ros"
    return base / "ibrobot" / "camera_isp_overrides" / f"{camera_name}.json"


def _validate_entry(key: str, value: Any) -> Any | None:
    """Validate one (key, value) pair. Return cleaned value or None to drop."""
    if key in _BOOL_KEYS:
        if isinstance(value, bool):
            return value
        logger.warning(
            f"  ISP override key '{key}' expects bool, got {type(value).__name__}; dropped"
        )
        return None

    if key in _INT_KEYS:
        if isinstance(value, bool):  # bool is subclass of int — reject
            logger.warning(f"  ISP override key '{key}' expects int, got bool; dropped")
            return None
        if not isinstance(value, (int, float)):
            logger.warning(
                f"  ISP override key '{key}' expects number, got {type(value).__name__}; dropped"
            )
            return None
        ivalue = int(value)
        lo, hi = _INT_KEYS[key]
        if not (lo <= ivalue <= hi):
            logger.warning(
                f"  ISP override key '{key}'={ivalue} outside [{lo},{hi}]; dropped"
            )
            return None
        return ivalue

    return None  # unknown key — caller already filtered, but be safe


def load_isp_override(camera_name: str) -> dict:
    """Load and validate an ISP override file for one camera.

    Args:
        camera_name: The peripheral ``name`` (e.g. ``"top"``, ``"wrist"``).

    Returns:
        A dict of validated ISP parameters (possibly empty). Always safe to
        ``params.update(...)`` directly into a usb_cam parameter dict.
    """
    path = _override_path(camera_name)
    if not path.exists():
        logger.info(f"No ISP override for camera '{camera_name}' at {path}")
        return {}

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"Failed to read ISP override {path}: {exc}; ignoring")
        return {}

    if not isinstance(raw, dict):
        logger.warning(f"ISP override {path} root is not an object; ignoring")
        return {}

    cleaned: dict = {}
    unknown: list = []
    for key, value in raw.items():
        # Ignore underscore-prefixed metadata keys (e.g. _warning, _timestamp)
        if isinstance(key, str) and key.startswith("_"):
            continue
        if key not in _ALLOWED_KEYS:
            unknown.append(key)
            continue
        validated = _validate_entry(key, value)
        if validated is not None:
            cleaned[key] = validated

    if unknown:
        logger.warning(
            f"ISP override for '{camera_name}' contains unknown keys "
            f"{unknown}; ignored. Allowed: {sorted(_ALLOWED_KEYS)}"
        )

    if cleaned:
        logger.info(f"Applied ISP override for '{camera_name}': {cleaned}")
    else:
        logger.info(f"ISP override for '{camera_name}' present but yielded no valid keys")

    return cleaned
