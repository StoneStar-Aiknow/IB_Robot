"""Simulation backend registry and factory.

Usage:
    from robot_config.launch_builders.sim_backend import get_sim_backend

    adapter = get_sim_backend("gazebo")      # GazeboAdapter instance
    adapter = get_sim_backend("mujoco")      # MujocoAdapter instance (T6)
    adapter = get_sim_backend("mock")        # MockAdapter instance

The platform string comes from robot_config['simulation']['platform'] in
the robot YAML (e.g., so101_single_arm.yaml).
"""

from .base_adapter import SimBackendAdapter
from .gazebo_adapter import GazeboAdapter
from .mock_adapter import MockAdapter
from .mujoco_adapter import MujocoAdapter

__all__ = ["SimBackendAdapter", "GazeboAdapter", "MockAdapter", "MujocoAdapter", "get_backend_caps", "get_sim_backend"]

_BACKEND_REGISTRY: dict[str, type[SimBackendAdapter]] = {
    "gazebo": GazeboAdapter,
    "mock": MockAdapter,
    "mujoco": MujocoAdapter,
}

_BACKEND_CAPS: dict[str, dict[str, bool]] = {
    "gazebo": {"provides_clock": True, "needs_ros2_control": True},
    "mock": {"provides_clock": False, "needs_ros2_control": False},
    "mujoco": {"provides_clock": True, "needs_ros2_control": True},
}


def get_backend_caps(platform: str) -> dict[str, bool]:
    """Return backend capabilities used before backend actions are built."""
    caps = _BACKEND_CAPS.get(platform)
    if caps is None:
        available = list(_BACKEND_CAPS.keys())
        raise ValueError(
            f"Unknown sim platform: '{platform}'. "
            f"Available platforms: {available}. "
            f"Check simulation.platform in your robot YAML."
        )
    return dict(caps)


def get_sim_backend(platform: str) -> SimBackendAdapter:
    """Instantiate a simulation backend adapter by platform name.

    Args:
        platform: Backend identifier, e.g. 'gazebo', 'mujoco', or 'mock'.
                  Must match a key in _BACKEND_REGISTRY.

    Returns:
        A fresh SimBackendAdapter instance for the requested platform.

    Raises:
        ValueError: If platform is not registered.
    """
    cls = _BACKEND_REGISTRY.get(platform)
    if cls is None:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"Unknown sim platform: '{platform}'. "
            f"Available platforms: {available}. "
            f"Check simulation.platform in your robot YAML."
        )
    return cls()
