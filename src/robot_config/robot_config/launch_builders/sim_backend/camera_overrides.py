"""Runtime loading of user-saved simulation camera overrides.

Thin re-import layer (openspec ``so101-gazebo-runtime-transport`` D6.1/D6.3):
the override reader and the override-aware pose resolution live in
``robot_runtime.gazebo`` so runtime providers and generic launch share one
implementation; the generic preset fallbacks stay here.
"""

from __future__ import annotations

from robot_config.launch_builders.sim_backend.camera_presets import get_preset
from robot_runtime.gazebo import load_camera_override


def load_gazebo_override(camera_name: str) -> dict | None:
    """Load a real Gazebo calibration override for one camera.

    Returns None when the override file is absent, unreadable, or still a
    placeholder stub. Unlike ``load_with_override()``, this function never falls
    back to hardcoded presets; callers can use it to distinguish user-saved
    Gazebo calibration from default values.
    """
    return load_camera_override(camera_name, fallback=get_preset("gazebo", camera_name) or {})


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
