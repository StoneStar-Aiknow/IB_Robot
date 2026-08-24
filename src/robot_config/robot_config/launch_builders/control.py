"""Control system launch builders.

This module handles:
- ros2_control node generation
- Controller spawner creation
- Joint configuration validation (delegates to utils.py)

URDF building (xacro processing + camera injection) is in description.py.
"""

import os
import tempfile
from pathlib import Path

import yaml
from launch_ros.actions import Node

from robot_config.launch_builders.description import generate_robot_description
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import (
    parse_bool,
    resolve_calibration_paths_from_config,
    resolve_ros_path,
    validate_joint_config,
)

logger = get_colored_logger("robot_config.control")


def generate_controller_spawners(
    controller_names,
    use_sim=True,
    controller_manager_name="controller_manager",
    controller_manager_timeout=None,
    inactive_controller_names=(),
):
    """Generate one timeout-aware controller group spawner.

    Args:
        controller_names: Controller names to load, configure, and activate
        use_sim: Simulation mode (affects timeout and use_sim_time)
        controller_manager_name: Name of controller manager service
        controller_manager_timeout: Timeout for manager discovery and service calls
        inactive_controller_names: Controller names to load/configure but leave inactive

    Returns:
        Empty list or a single Node action for atomic controller group activation
    """
    is_sim = parse_bool(use_sim, default=True)

    controller_names = list(controller_names)
    inactive_controller_names = list(inactive_controller_names)
    if not controller_names and not inactive_controller_names:
        return []
    if len(set(controller_names)) != len(controller_names):
        raise ValueError("Active controller names must be unique")
    if len(set(inactive_controller_names)) != len(inactive_controller_names):
        raise ValueError("Inactive controller names must be unique")
    overlap = set(controller_names) & set(inactive_controller_names)
    if overlap:
        raise ValueError(f"Controllers cannot be both active and inactive: {sorted(overlap)}")

    timeout = float(controller_manager_timeout) if controller_manager_timeout is not None else (60 if is_sim else 10)
    if timeout <= 0.0:
        raise ValueError("controller_manager_timeout must be greater than zero")
    service_call_timeout = min(timeout, 10.0)
    switch_timeout = timeout
    arguments = [
        *controller_names,
        "--controller-manager",
        controller_manager_name,
        "--controller-manager-timeout",
        str(timeout),
        "--service-call-timeout",
        str(service_call_timeout),
        "--switch-timeout",
        str(switch_timeout),
    ]
    for controller_name in inactive_controller_names:
        arguments.extend(["--inactive-controller", controller_name])

    return [
        Node(
            package="robot_config",
            executable="controller_spawner",
            name="spawner_controller_group",
            parameters=[{"use_sim_time": is_sim}],
            arguments=arguments,
            output="screen",
        )
    ]


def generate_ros2_control_nodes(
    robot_config,
    use_sim,
    auto_start_controllers="true",
    controller_startup_timeout=None,
):
    """Generate ros2_control nodes from configuration.

    Args:
        robot_config: Robot configuration dict
        use_sim: Simulation mode flag (string or bool)
        auto_start_controllers: Whether to automatically start controllers (string or bool)
        controller_startup_timeout: Timeout used by controller-manager spawners

    Returns:
        Tuple: (nodes, controller_names, deferred_spawners, robot_description)
        Controller spawners are returned in ``deferred_spawners`` so launch can
        gate them on hardware or simulator readiness.
    """
    is_sim = parse_bool(use_sim, default=False)
    is_auto_start = parse_bool(auto_start_controllers, default=True)

    nodes = []
    deferred_spawners = []
    ros2_control_config = robot_config.get("ros2_control")

    if not ros2_control_config:
        logger.warning("No ros2_control configuration found")
        return nodes, [], deferred_spawners, {}

    logger.info("Creating ros2_control nodes")

    # Pre-flight check: calibration files must exist for real hardware.
    if not is_sim:
        try:
            calibration_paths = resolve_calibration_paths_from_config(robot_config)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid ros2_control calibration configuration: {exc}") from exc
        if not calibration_paths:
            logger.warning(
                "No calibration files configured for real hardware mode; calibrated joint conversion is unavailable"
            )
        for calib_file_resolved in calibration_paths:
            if not Path(calib_file_resolved).exists():
                logger.error("Calibration file not found!")
                logger.error(f"  Resolved path: {calib_file_resolved}")
                logger.error(f"  HOME=$HOME -> {os.environ.get('HOME', '(unset)')}")
                logger.error("")
                logger.error("  Please run calibration first:")
                calib_port = ros2_control_config.get("port", "/dev/ttyACM0")
                logger.error("    ros2 run so101_hardware calibrate_arm --arm follower --port " + calib_port)
                raise RuntimeError(
                    f"Calibration file not found: {calib_file_resolved}. "
                    f"Run: ros2 run so101_hardware calibrate_arm --arm follower --port " + calib_port
                )

    # Validate joint configuration
    validate_joint_config(robot_config)

    # Build URDF (xacro processing + camera injection) via description layer
    _desc_result = generate_robot_description(robot_config, use_sim)
    if _desc_result is None:
        return nodes, [], deferred_spawners, {}

    robot_description_str, robot_description = _desc_result

    # Robot State Publisher
    nodes.append(
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[robot_description],
        )
    )

    # Get control mode configuration
    control_mode_name = robot_config.get("default_control_mode", "model_inference")
    control_modes = robot_config.get("control_modes", {})

    if control_modes:
        if control_mode_name not in control_modes:
            available_modes = list(control_modes.keys())
            logger.error(f"Control mode '{control_mode_name}' not found")
            logger.info(f"Available modes: {available_modes}")
            if available_modes:
                control_mode_name = available_modes[0]

        if control_mode_name:
            mode_config = control_modes[control_mode_name]
            if is_sim and mode_config.get("sim_controllers") is not None:
                controller_names = mode_config.get("sim_controllers", [])
                inactive_controller_names = mode_config.get("sim_inactive_controllers", [])
            elif not is_sim and mode_config.get("hardware_controllers") is not None:
                controller_names = mode_config.get("hardware_controllers", [])
                inactive_controller_names = mode_config.get("hardware_inactive_controllers", [])
            else:
                controller_names = mode_config.get("controllers", [])
                inactive_controller_names = mode_config.get("inactive_controllers", [])
            mode_description = mode_config.get("description", "No description")
            logger.info(f"Using control mode: {control_mode_name}")
            logger.info(f"  Description: {mode_description}")
            logger.info(f"  Controllers: {controller_names}")
            logger.info(f"  Inactive controllers: {inactive_controller_names}")
        else:
            controller_names = []
            inactive_controller_names = []
    else:
        controller_names = ros2_control_config.get("controllers", [])
        inactive_controller_names = []

    controllers_config = resolve_ros_path(ros2_control_config.get("controllers_config"))

    if not is_sim:
        # Real hardware mode
        logger.info("Real hardware mode")

        if controllers_config and Path(controllers_config).exists():
            logger.info(f"Controllers config: {controllers_config}")

            # Write robot_description to a temp YAML under the 'controller_manager'
            # node name.  ros2_control_node internally creates a node called
            # 'controller_manager', but launch writes dict params under the
            # executable name ('ros2_control_node') — a namespace mismatch.
            # Using a file with the correct key avoids the mismatch WITHOUT
            # setting name= on the Node (which would add a global __node
            # remapping that breaks child controller nodes).
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".yaml",
                delete=False,
                prefix="cm_robot_desc_",
            ) as cm_params_file:
                yaml.dump(
                    {
                        "controller_manager": {
                            "ros__parameters": {
                                "robot_description": robot_description_str,
                            }
                        }
                    },
                    cm_params_file,
                    default_flow_style=False,
                )
            logger.info(f"Controller manager params: {cm_params_file.name}")

            nodes.append(
                Node(
                    package="controller_manager",
                    executable="ros2_control_node",
                    parameters=[cm_params_file.name, controllers_config],
                    remappings=[
                        ("~/robot_description", "/robot_description"),
                    ],
                    output="screen",
                )
            )

            if is_auto_start and (controller_names or inactive_controller_names):
                deferred_spawners = generate_controller_spawners(
                    controller_names,
                    use_sim=False,
                    controller_manager_timeout=controller_startup_timeout,
                    inactive_controller_names=inactive_controller_names,
                )
                logger.info(f"Deferring {len(deferred_spawners)} controller spawners until hardware is active")
    else:
        # Simulation mode
        # gz_ros2_control plugin provides controller_manager, but spawners
        # must wait until the Gazebo entity is fully created and the plugin
        # has initialized the hardware interface.
        logger.info("Simulation mode: Gazebo provides controller_manager")
        logger.info(f"Controllers to spawn (deferred until after gz spawn): {controller_names}")

        if is_auto_start and (controller_names or inactive_controller_names):
            deferred_spawners = generate_controller_spawners(
                controller_names,
                use_sim=True,
                controller_manager_timeout=controller_startup_timeout,
                inactive_controller_names=inactive_controller_names,
            )
            logger.info(f"Deferring {len(deferred_spawners)} controller spawners (handled by caller)")

    return nodes, controller_names, deferred_spawners, robot_description
