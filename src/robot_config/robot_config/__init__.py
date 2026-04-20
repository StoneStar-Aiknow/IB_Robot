"""robot_config package - Unified robot configuration system."""

from robot_config.config import (
    RobotConfig,
    Ros2ControlConfig,
    PeripheralConfig,
    ContractExtensionConfig,
    CameraConfig,
)
from robot_config.loader import (
    load_robot_config,
    load_robot_config_dict,
    build_contract_from_robot_config_dict,
    validate_config,
)

# Import launch builders
from robot_config.launch_builders import (
    generate_ros2_control_nodes,
    generate_camera_nodes,
    generate_tf_nodes,
    generate_virtual_camera_relays,
    generate_gazebo_nodes,
    validate_joint_config,
)

# Import utilities
from robot_config.utils import resolve_ros_path, parse_bool

__all__ = [
    # Config classes
    "RobotConfig",
    "Ros2ControlConfig",
    "PeripheralConfig",
    "ContractExtensionConfig",
    "CameraConfig",
    # Loaders
    "load_robot_config",
    "load_robot_config_dict",
    "build_contract_from_robot_config_dict",
    "validate_config",
    # Launch builders
    "generate_ros2_control_nodes",
    "generate_camera_nodes",
    "generate_tf_nodes",
    "generate_virtual_camera_relays",
    "generate_gazebo_nodes",
    "validate_joint_config",
    # Utilities
    "resolve_ros_path",
    "parse_bool",
]
