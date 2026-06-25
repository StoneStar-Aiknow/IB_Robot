"""Abstract base class for simulation backend adapters.

IB-Robot supports multiple simulation backends (Gazebo, MuJoCo, mock) selectable
via the 'simulation.platform' field in the robot YAML configuration.

Each backend must implement all five methods. Methods that are not applicable
to a particular backend should return an empty list (e.g., spawn_peripheral_bridges
for MuJoCo, which publishes ROS topics directly without bridge nodes).

Architecture:
    YAML simulation.platform
        └── get_sim_backend(platform)          # factory in __init__.py
                └── SimBackendAdapter          # this abstract class
                        ├── GazeboAdapter      # ros_gz_bridge, Gazebo launch
                        ├── MujocoAdapter      # mujoco_ros2_control (T6)
                        └── MockAdapter        # contract_mock topic backend
"""

from abc import ABC, abstractmethod
from typing import Any


class SimBackendAdapter(ABC):
    """Abstract adapter for simulation backends.

    All methods return a flat list of ROS 2 launch actions that are appended
    to the main launch description by robot.launch.py.
    """

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
    def load_scene(self, scene_file_path: str) -> list:
        """Inject a scene file into the simulator.

        Called after start_backend when robot_config['simulation']['scene']
        is not null. Implementation is deferred to T5.

        Args:
            scene_file_path: Absolute path to the resolved scene file.

        Returns:
            List of launch actions to load the scene.
        """

    @abstractmethod
    def ensure_controller_manager(self, robot_config: dict) -> list:
        """Ensure controller_manager is available inside the simulation.

        For Gazebo, controller_manager is embedded in the gz_ros2_control
        plugin loaded with the URDF, so this returns [].
        For MuJoCo, this starts the mujoco_ros2_control node which acts
        as the hardware interface AND controller_manager host.

        Args:
            robot_config: Full robot configuration dict loaded from YAML.

        Returns:
            List of launch actions, or [] if no action needed.
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

        For MuJoCo, mujoco_ros2_control publishes ROS topics directly and
        is configured to use the same naming contract, so this returns [].

        Args:
            peripherals: List of peripheral dicts from
                         robot_config['peripherals'].

        Returns:
            List of Node actions for bridge processes, or [].
        """

    @abstractmethod
    def update_object_pose(self, object_name: str, pose) -> None:
        """[Reserved for v1] Update a scene object's pose at runtime.

        This method is reserved for dynamic scene manipulation (e.g.,
        randomizing object positions for data collection). Not called
        during launch; intended for future runtime use.

        Args:
            object_name: Name of the object in the simulator scene.
            pose: Target pose (type TBD in v1 design).
        """
