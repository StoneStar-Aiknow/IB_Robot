"""Mock simulation backend adapter.

This lets ``use_sim:=true`` route ``simulation.platform: mock`` through the
same backend registry as Gazebo and MuJoCo while reusing hardware_mock's
contract-level ROS topic surface.
"""

from typing import Any

from robot_config.launch_builders.hardware_mock import generate_hardware_mock_nodes
from robot_config.logger_utils import get_colored_logger

from .base_adapter import SimBackendAdapter

logger = get_colored_logger("robot_config.sim_backend.mock")


class MockAdapter(SimBackendAdapter):
    """Contract-only backend backed by hardware_mock.contract_mock."""

    provides_clock = False
    needs_ros2_control = False

    def start_backend(self, robot_config: dict) -> tuple[list, Any]:
        """Start contract_mock and return no simulator spawn node."""
        logger.info("Starting hardware_mock contract backend")
        scene_name = robot_config.get("simulation", {}).get("scene")
        if scene_name:
            self.load_scene(str(scene_name))
        return generate_hardware_mock_nodes(robot_config), None

    def load_scene(self, scene_file_path: str) -> list:
        """Scenes are outside contract mock scope."""
        logger.warning(
            f"simulation.scene='{scene_file_path}' is ignored by mock backend; "
            "contract mock does not model scene objects"
        )
        return []

    def ensure_controller_manager(self, robot_config: dict) -> list:
        """contract_mock owns the loop directly; no controller_manager exists."""
        return []

    def spawn_peripheral_bridges(self, peripherals: list) -> list:
        """contract_mock publishes ROS contract topics directly."""
        return []

    def update_object_pose(self, object_name: str, pose) -> None:
        """No runtime scene objects exist in the mock backend."""
        pass
