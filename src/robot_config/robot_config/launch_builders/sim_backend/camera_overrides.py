"""Runtime loading of user-saved simulation camera overrides."""

from __future__ import annotations

from pathlib import Path

import yaml

from robot_config.launch_builders.sim_backend.camera_presets import get_preset

_OVERRIDE_BASE = Path.home() / ".ros" / "ibrobot" / "sim_camera_overrides"


def load_gazebo_override(camera_name: str) -> dict | None:
    """Load a real Gazebo calibration override for one camera.

    Returns None when the override file is absent, unreadable, or still a
    placeholder stub. Unlike ``load_with_override()``, this function never falls
    back to hardcoded presets; callers can use it to distinguish user-saved
    Gazebo calibration from default values.
    """
    try:
        override_path = _OVERRIDE_BASE / f"{camera_name}.yaml"
        if not override_path.exists():
            return None
        data = yaml.safe_load(override_path.read_text(encoding="utf-8")) or {}
        pose = data.get("pose", {}) or {}
        # Skip stub overrides (placeholder files written by camera_alignment
        # before any real calibration was performed). This mirrors the same
        # check that sim_camera_adjuster does when loading existing overrides:
        # see src/sim_models/sim_models/sim_camera_adjuster.py around the
        # "_is_stub" / all-zero-pose handling. Loading an all-zero pose into
        # a camera URDF joint origin would silently break the simulation.
        pose_keys = ("x", "y", "z", "roll", "pitch", "yaw")
        is_stub = bool(data.get("_is_stub"))
        is_all_zero = all(float(pose.get(k, 0.0)) == 0.0 for k in pose_keys)
        if is_stub or is_all_zero:
            return None
        # Inherit parent_frame from override; fall back to Gazebo preset default.
        fallback = get_preset("gazebo", camera_name) or {}
        return {
            "parent_frame": data.get(
                "parent_frame",
                fallback.get("parent_frame", "base"),
            ),
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "z": float(pose.get("z", 0.0)),
            "roll": float(pose.get("roll", 0.0)),
            "pitch": float(pose.get("pitch", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
            "fovy": float(data.get("fovy_deg", fallback.get("fovy", 60))),
        }
    except Exception:
        return None


def load_with_override(platform: str, camera_name: str) -> dict | None:
    """Load platform preset, applying a user Gazebo override when appropriate.

    Override files are written by ``sim_camera_adjuster`` (Gazebo tool) to
    ``~/.ros/ibrobot/sim_camera_overrides/{camera_name}.yaml``. They are only
    directly applied for platform == "gazebo" because the calibration is saved
    in Gazebo convention; MuJoCo callers should use ``load_gazebo_override()``
    only when they intend to convert that real calibration into MuJoCo's camera
    convention.
    """
    if platform == "gazebo":
        override = load_gazebo_override(camera_name)
        if override is not None:
            return override
    return get_preset(platform, camera_name)
