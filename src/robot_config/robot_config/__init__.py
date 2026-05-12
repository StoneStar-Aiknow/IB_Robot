"""robot_config package - Unified robot configuration system."""

from robot_config.config import (
    CameraConfig,
    ContractExtensionConfig,
    PeripheralConfig,
    RobotConfig,
    Ros2ControlConfig,
)
from robot_config.loader import (
    build_contract_from_robot_config_dict,
    load_robot_config,
    load_robot_config_dict,
    validate_config,
)

# Import utilities
from robot_config.utils import parse_bool, resolve_ros_path

_LAZY_EXPORTS = {
    "generate_ros2_control_nodes": (
        "robot_config.launch_builders",
        "generate_ros2_control_nodes",
    ),
    "generate_camera_nodes": (
        "robot_config.launch_builders",
        "generate_camera_nodes",
    ),
    "generate_tf_nodes": (
        "robot_config.launch_builders",
        "generate_tf_nodes",
    ),
    "generate_virtual_camera_relays": (
        "robot_config.launch_builders",
        "generate_virtual_camera_relays",
    ),
    "generate_gazebo_nodes": (
        "robot_config.launch_builders",
        "generate_gazebo_nodes",
    ),
    "validate_joint_config": (
        "robot_config.launch_builders",
        "validate_joint_config",
    ),
}


def __getattr__(name: str):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


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
