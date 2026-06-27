"""Simulation launch builders — thin dispatch layer.

The actual simulation logic lives in sim_backend/:
  sim_backend/gazebo_adapter.py  — Gazebo Ignition implementation
  sim_backend/mujoco_adapter.py  — MuJoCo implementation
  sim_backend/mock_adapter.py    — hardware_mock contract backend

robot.launch.py uses get_sim_backend() directly, reading
'simulation.platform' from the robot YAML.

generate_gazebo_nodes() is kept as a convenience helper for
debugging and unit testing; it always uses the Gazebo backend
regardless of YAML platform.
"""

from .sim_backend import get_sim_backend


def generate_gazebo_nodes(robot_config: dict) -> list:
    """Debug/test helper: directly invoke GazeboAdapter.

    Always uses the Gazebo backend. In normal operation, robot.launch.py
    calls get_sim_backend(platform) instead of this function so that the
    YAML simulation.platform field is respected.

    Args:
        robot_config: Full robot configuration dict from YAML.

    Returns:
        List of launch actions for Gazebo simulation.
    """
    adapter = get_sim_backend("gazebo")
    actions, _ = adapter.start_backend(robot_config)
    actions += adapter.spawn_peripheral_bridges(robot_config.get("peripherals", []))
    return actions
