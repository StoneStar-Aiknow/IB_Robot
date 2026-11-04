"""Abstract base class for simulation backend adapters.

IB-Robot's provider-less simulation backends (Gazebo, mock) are selectable
via the 'simulation.platform' field in the robot YAML configuration. A
runtime provider never reaches this registry: generic launch hands the world
to the provider through the platform's world hook (sim-platform-registry D5)
and the provider owns its own embodiment.

Architecture:
    YAML simulation.platform
        └── get_sim_backend(platform)          # factory in __init__.py
                └── SimBackendAdapter          # this abstract class
                        ├── GazeboAdapter      # ros_gz_bridge, Gazebo launch
                        └── MockAdapter        # contract_mock topic backend
"""

from abc import ABC, abstractmethod
from typing import Any


class SimBackendAdapter(ABC):
    """Abstract adapter for provider-less simulation backends.

    All methods return a flat list of ROS 2 launch actions that are appended
    to the main launch description by robot.launch.py.
    """

    provides_clock: bool = True
    needs_ros2_control: bool = True

    @abstractmethod
    def start_backend(self, robot_config: dict) -> tuple[list, Any]:
        """Launch the simulator process and core infrastructure.

        Includes: simulator server + GUI, entity spawning, clock bridge,
        joint state bridge, and any environment variable setup required
        by the simulator.

        Args:
            robot_config: Full robot configuration dict loaded from YAML.

        Returns:
            (actions, gz_create_entity_node)
            - actions: launch actions (Node, IncludeLaunchDescription, …)
            - gz_create_entity_node: the ros_gz_sim ``create`` Node (Gazebo only),
              for OnProcessExit → delayed controller spawners; ``None`` if N/A.
        """

    @abstractmethod
    def spawn_peripheral_bridges(self, peripherals: list) -> list:
        """Create topic bridge nodes for each peripheral device.

        For Gazebo, ros_gz_bridge nodes remap the raw Gazebo sensor path
        (e.g. /world/demo/model/.../sensor/top_camera/top_camera/image) to
        the ROS topic naming contract:
          /camera/{name}/image_raw
          /camera/{name}/camera_info
        where {name} is YAML peripherals[].name.

        Args:
            peripherals: List of peripheral dicts from
                         robot_config['peripherals'].

        Returns:
            List of Node actions for bridge processes, or [].
        """
