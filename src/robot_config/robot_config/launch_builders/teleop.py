"""Teleoperation node generation for robot_config.

This module provides utilities to generate teleoperation nodes
for integration with the robot_config launch system.
"""

import json
import os
import re
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import prepare_lerobot_env, resolve_ros_path

logger = get_colored_logger("robot_config.teleop")


def generate_teleop_nodes(robot_config: dict, robot_description_dict: dict = None) -> list[Node]:
    """
    Generate teleoperation nodes based on robot configuration.

    This function creates ROS 2 nodes for teleoperation based on the
    robot configuration YAML. It integrates with the robot_config launch system.

    Args:
        robot_config: Robot configuration dictionary loaded from YAML
        robot_description_dict: Dictionary containing robot_description (URDF)

    Returns:
        List of Node actions for teleoperation

    Example:
        >>> from robot_config.launch_builders.teleop import generate_teleop_nodes
        >>> config = load_robot_config('so101_single_arm')
        >>> nodes = generate_teleop_nodes(config, {'robot_description': '...'})
        >>> ld.add_action(nodes[0])

    Configuration Format:
        ```yaml
        robot:
          teleoperation:
            enabled: true
            active_device: "so101_leader"
            devices:
              - name: "so101_leader"
                type: "leader_arm"
                port: "/dev/ttyUSB0"
                calib_file: "~/.calibrate/so101_leader_calibrate.json"
            safety:
              joint_limits:
                "1": {"min": -3.14, "max": 3.14}
        ```
    """
    nodes = []

    # Get teleoperation config
    teleop_config = robot_config.get("teleoperation", {})
    if not teleop_config.get("enabled", False):
        logger.info("Teleoperation not enabled, skipping")
        return nodes

    active_device_names = teleop_config.get("active_devices")
    if active_device_names:
        device_configs = {device.get("name"): device for device in teleop_config.get("devices", [])}
        for active_device_name in active_device_names:
            device_config = device_configs.get(active_device_name)
            if not device_config:
                logger.error(f"Active device '{active_device_name}' not found")
                continue
            nodes.extend(_generate_device_nodes(robot_config, device_config, robot_description_dict))
        return nodes

    active_device_name = teleop_config.get("active_device", "")
    if not active_device_name:
        logger.warning("No active_device specified")
        return nodes

    # Find device config
    device_config = None
    for device in teleop_config.get("devices", []):
        if device.get("name") == active_device_name:
            device_config = device
            break

    if not device_config:
        logger.error(f"Active device '{active_device_name}' not found")
        return nodes

    return _generate_device_nodes(robot_config, device_config, robot_description_dict)


def _generate_device_nodes(robot_config: dict, device_config: dict, robot_description_dict: dict = None) -> list[Node]:
    nodes = []
    teleop_config = robot_config.get("teleoperation", {})

    # Get joint limits from safety config
    safety_config = teleop_config.get("safety", robot_config.get("safety", {}))
    joint_limits = safety_config.get("joint_limits", {})

    # Get joint names from robot config
    joints_config = robot_config.get("joints", {})
    target_config = device_config.get("target", {})
    arm_joint_names = target_config.get("arm_joint_names", joints_config.get("arm", []))
    gripper_joint_names = target_config.get("gripper_joint_names", joints_config.get("gripper", []))

    # Build device config for node parameter
    device_param = {
        "type": device_config.get("type", ""),
        "name": device_config.get("name", ""),
    }

    # Add optional device parameters
    if "port" in device_config:
        device_param["port"] = device_config["port"]
    if "calib_file" in device_config:
        # Expand environment variables in calib_file path
        calib_file_raw = device_config["calib_file"]
        calib_file_expanded = resolve_ros_path(device_config["calib_file"])
        logger.debug(f"calib_file_raw: {calib_file_raw}")
        logger.debug(f"calib_file_expanded: {calib_file_expanded}")
        device_param["calib_file"] = calib_file_expanded
        if not Path(calib_file_expanded).exists():
            logger.error("=" * 60)
            logger.error("Leader arm calibration file not found!")
            logger.error(f"  Resolved path: {calib_file_expanded}")
            logger.error(f"  Raw path:      {calib_file_raw}")
            logger.error(f"  HOME=$HOME -> {os.environ.get('HOME', '(unset)')}")
            calib_port = device_config.get("port", "/dev/ttyACM0")
            logger.error("")
            logger.error("  Please run calibration first:")
            logger.error("    ros2 run so101_hardware calibrate_arm --arm leader --port " + calib_port)
            logger.error("=" * 60)
            raise RuntimeError(
                f"Calibration file not found: {calib_file_expanded}. "
                f"Run: ros2 run so101_hardware calibrate_arm --arm leader --port " + calib_port
            )
    if "joint_mapping" in device_config:
        device_param["joint_mapping"] = device_config["joint_mapping"]

    device_param["arm_joint_names"] = arm_joint_names
    device_param["gripper_joint_names"] = gripper_joint_names

    # Add joint limits for proper scaling
    if joint_limits:
        device_param["joint_limits"] = joint_limits

    # Add any extra device-specific parameters
    known_keys = {
        "name",
        "type",
        "port",
        "calib_file",
        "joint_mapping",
        "phone_config",
        "group_name",
        "base_link_name",
        "ee_frame_name",
        "target_frame_name",
        "ik_timeout",
        "target",
    }
    for key, value in device_config.items():
        if key not in known_keys:
            device_param[key] = value

    if "phone_config" in device_config:
        device_param["phone_config"] = device_config["phone_config"]

    for moveit_key in ("group_name", "base_link_name", "ee_frame_name", "target_frame_name", "ik_timeout"):
        if moveit_key in device_config:
            device_param[moveit_key] = device_config[moveit_key]

    # ----- Cartesian solver selection -----
    # SSOT lives in robot.teleoperation.cartesian.{solver,tool_frame}.
    # Default is 'safe_servo' for low-cost arms with noticeable servo error.
    cart_cfg = teleop_config.get("cartesian", {}) or {}
    cart_solver = cart_cfg.get("solver", "safe_servo")
    if cart_solver not in ("velocity_servo", "safe_servo"):
        raise ValueError(
            f"teleoperation.cartesian.solver must be 'velocity_servo', or 'safe_servo', got {cart_solver!r}"
        )
    moveit_cfg = robot_config.get("moveit", {}) or {}
    cart_tool_frame = (
        cart_cfg.get("tool_frame") or device_config.get("ee_frame_name") or moveit_cfg.get("ee_link") or "gripper"
    )
    _validate_tool_frame(cart_tool_frame, robot_config)
    device_param["cartesian_solver"] = cart_solver
    device_param["tool_frame"] = cart_tool_frame
    device_param["base_link_name"] = device_param.get("base_link_name", moveit_cfg.get("base_link", "base"))

    # When solver=safe_servo, route device-side speed knobs from the safe_servo
    # config block.
    if cart_solver == "safe_servo":
        safe_cfg = cart_cfg.get("safe_servo", {}) or {}
        safe_linear_speed = safe_cfg.get("linear_speed", 0.3)
        safe_angular_speed = safe_cfg.get("angular_speed", 0.7)
        cp = device_param.setdefault("control_params", {})
        cp["cartesian_linear_speed"] = safe_linear_speed
        cp["cartesian_angular_speed"] = safe_angular_speed
        logger.info(f"safe_servo speed override: linear={safe_linear_speed} (0~1), angular={safe_angular_speed} (0~1)")

    logger.info(f"Cartesian solver={cart_solver}, tool_frame={cart_tool_frame}")

    device_type = device_config.get("type", "")

    control_frequency = device_config.get("control_frequency", 50.0)

    if device_type == "joy_teleop":
        return _create_joy_teleop_nodes(device_config)

    # Prepare lerobot environment
    env = prepare_lerobot_env()

    # Convert dicts to JSON strings for ROS 2 parameter passing
    device_param_json = json.dumps(device_param)
    joint_limits_json = json.dumps(joint_limits)
    logger.debug(f"device_param_json: {device_param_json}")
    logger.debug(f"device_param dict: {device_param}")

    moveit_config = robot_config.get("moveit", {})

    # For phone devices: inject extra params into device_config so PhoneDevice
    # can read arm/gripper joint names, home positions, and servo frame at runtime.
    if device_type == "phone":
        base_link_name = device_param.get("base_link_name", moveit_config.get("base_link", "base_link"))
        reset_positions = robot_config.get("ros2_control", {}).get("reset_positions", {})
        home_positions_list = [reset_positions.get(n, 0.0) for n in arm_joint_names]

        device_param_ext = dict(device_param)
        device_param_ext.update(
            {
                "home_joint_positions": home_positions_list,
                "base_link_name": base_link_name,
                "control_frequency": 50.0,
            }
        )
        device_param_json = json.dumps(device_param_ext)
        control_frequency = 50.0

    node_name = "robot_teleop_node"
    if device_config.get("name"):
        node_name = f"robot_teleop_{re.sub(r'[^A-Za-z0-9_]+', '_', device_config['name']).strip('_')}"

    teleop_node = Node(
        package="robot_teleop",
        executable="teleop_node",
        name=node_name,
        output="screen",
        env=env,
        parameters=[
            {
                "control_frequency": control_frequency,
                "device_config": device_param_json,
                "joint_limits": joint_limits_json,
                "arm_joint_names": arm_joint_names,
                "gripper_joint_names": gripper_joint_names,
                "arm_command_topic": target_config.get("arm_command_topic", "/arm_position_controller/commands"),
                "gripper_command_topic": target_config.get(
                    "gripper_command_topic", "/gripper_position_controller/commands"
                ),
            }
        ],
    )
    nodes.append(teleop_node)
    logger.info(f"Generated teleop_node for device: {device_config.get('name', '')} (type: {device_type})")

    # Add joy_node to read physical joystick and publish to /joy
    if device_config.get("type") == "xbox_controller":
        input_dev = device_config.get("input_device", "/dev/input/js0")
        joy_node = Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            parameters=[
                {
                    "device_id": 0,
                    "device_name": "",
                    "deadzone": 0.1,
                    "autorepeat_rate": 20.0,
                    "sticky_buttons": False,
                }
            ],
            output="screen",
        )
        nodes.append(joy_node)
        logger.info(f"Added joy_node for input device: {input_dev}")

    # Add MoveIt Servo or Safe Servo node for Xbox controller and phone.
    # Selection is driven by robot.teleoperation.cartesian.solver.
    if device_config.get("type") in ("xbox_controller", "phone"):
        if cart_solver == "velocity_servo":
            servo_node = _create_servo_node(robot_config, device_config, robot_description_dict)
            nodes.append(servo_node)
            logger.info("Generated servo_node for Cartesian velocity_servo control")
        elif cart_solver == "safe_servo":
            safe_node = _create_so101_safe_servo_node(robot_config, device_config, robot_description_dict)
            nodes.append(safe_node)
            logger.info("Generated so101_safe_servo_node for SO101 robust Cartesian control")

    return nodes


def _create_joy_teleop_nodes(device_config: dict) -> list[Node]:
    """Create joy_node + joy_teleop launch actions for mobile-base teleoperation."""
    nodes = []

    input_dev = device_config.get("input_device", "/dev/input/js0")
    joy_params = {
        "dev": input_dev,
        "deadzone": float(device_config.get("deadzone", 0.1)),
        "autorepeat_rate": float(device_config.get("autorepeat_rate", 20.0)),
        "sticky_buttons": bool(device_config.get("sticky_buttons", False)),
    }
    nodes.append(
        Node(
            package="joy",
            executable="joy_node",
            name="joy_node",
            parameters=[joy_params],
            output="screen",
        )
    )

    config_path = device_config.get("config_path")
    if not config_path:
        raise ValueError("joy_teleop device requires 'config_path'")

    nodes.append(
        Node(
            package="joy_teleop",
            executable="joy_teleop",
            name="joy_teleop",
            parameters=[resolve_ros_path(config_path)],
            output="screen",
        )
    )
    print(f"[teleop_builder] Generated joy_teleop stack using {config_path}")
    return nodes


def _create_servo_node(robot_config: dict, device_config: dict, robot_description_dict: dict = None) -> Node:
    """Create MoveIt Servo node."""
    import yaml

    from robot_config.utils import resolve_ros_path

    # 1. Load servo parameters
    servo_config_name = device_config.get("servo_config", "so101_servo")
    servo_params_path = resolve_ros_path(f"$(find robot_moveit)/config/{servo_config_name}.yaml")

    with open(servo_params_path) as f:
        servo_params = yaml.safe_load(f)

    # 2. Build MoveIt configuration manually for robustness
    # MoveItConfigsBuilder can be finicky with relative paths in different environments
    robot_type = robot_config.get("type", "so101")

    moveit_params = {}
    try:
        # Load SRDF (Semantic Robot Description Format)
        srdf_path = resolve_ros_path(f"$(find robot_moveit)/config/lerobot/{robot_type}/{robot_type}.srdf")
        if os.path.exists(srdf_path):
            with open(srdf_path) as f:
                moveit_params["robot_description_semantic"] = f.read()
            logger.info(f"Loaded SRDF from {srdf_path}")
        else:
            logger.warning(f"SRDF not found at {srdf_path}")

        # Load Kinematics
        kinematics_path = resolve_ros_path(f"$(find robot_moveit)/config/lerobot/{robot_type}/kinematics.yaml")
        if os.path.exists(kinematics_path):
            with open(kinematics_path) as f:
                moveit_params["robot_description_kinematics"] = yaml.safe_load(f)
            logger.info(f"Loaded kinematics from {kinematics_path}")

        # Load Joint Limits
        joint_limits_path = resolve_ros_path(f"$(find robot_moveit)/config/lerobot/{robot_type}/joint_limits.yaml")
        if os.path.exists(joint_limits_path):
            with open(joint_limits_path) as f:
                joint_limits_data = yaml.safe_load(f)
                moveit_params.update(joint_limits_data)
                # MoveIt 2 nodes also look for joint_limits under robot_description_planning
                moveit_params["robot_description_planning"] = joint_limits_data
            logger.info(f"Loaded joint limits from {joint_limits_path}")
    except Exception as e:
        logger.warning(f"Failed to manually load MoveIt configs: {e}")

    # Merge robot_description_dict to ensure robot_description
    # and use_sim_time are always present (from description layer)
    if robot_description_dict:
        moveit_params.update(robot_description_dict)
        # If use_sim_time is true, also tell moveit_servo to use gazebo if configured
        if robot_description_dict.get("use_sim_time"):
            servo_params["use_gazebo"] = True

    # Create the servo node
    servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="servo_node",
        output="screen",
        parameters=[
            {"moveit_servo": servo_params},
            moveit_params,
        ],
    )
    return servo_node


# ---------------------------------------------------------------------------
# Cartesian helpers: tool_frame validation + safe_servo launcher
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when the SSOT YAML is internally inconsistent."""


def _validate_tool_frame(tool_frame: str, robot_config: dict) -> None:
    """Confirm ``tool_frame`` resolves at launch time.

    Accepts: the configured ``base_link``, any virtual frame declared under
    ``kinematics.frames``, or anything the user typed (URDF links are not
    enumerated here — they are confirmed when the Servo node first runs).
    The strict check is intentionally narrow: we hard-fail on the obvious
    mistake of referencing a virtual frame name that was never declared.
    """
    from robot_config.launch_builders.virtual_frames import (
        collect_virtual_frame_names,
    )

    if not tool_frame:
        raise ConfigError("tool_frame is empty")
    moveit_cfg = robot_config.get("moveit", {}) or {}
    base_link = moveit_cfg.get("base_link", "base")
    if tool_frame == base_link:
        return
    if tool_frame == moveit_cfg.get("ee_link"):
        return
    virtual = collect_virtual_frame_names((robot_config.get("kinematics", {}) or {}).get("frames"))
    # If user references a virtual frame, it must exist.
    if tool_frame in virtual:
        return
    # Otherwise we assume the user supplied a URDF link name; warn but allow,
    # since runtime TF will be the final authority (and the Servo node
    # will fail-fast with a clear log if the link doesn't exist).
    logger.warning(
        f"tool_frame={tool_frame!r} is neither base_link, ee_link, nor a declared "
        f"virtual frame; assuming it is a URDF link. Runtime TF will validate."
    )


def _create_so101_safe_servo_node(
    robot_config: dict,
    device_config: dict,
    robot_description_dict: dict = None,
) -> Node:
    """Launch ``so101_safe_servo_node``.

    Loads the solver YAML from ``moveit.so101_safe_servo_config_path`` and
    appends arm joint names from ``robot.joints.arm`` so the node knows
    the controller output order.
    """
    import yaml as _yaml

    moveit_cfg = robot_config.get("moveit", {}) or {}
    yaml_ref = moveit_cfg.get("so101_safe_servo_config_path")
    if not yaml_ref:
        raise ConfigError(
            "solver=safe_servo requires moveit.so101_safe_servo_config_path "
            "(e.g. package://robot_moveit/config/so101_safe_servo.yaml)"
        )
    if yaml_ref.startswith("package://"):
        rest = yaml_ref[len("package://") :]
        pkg_name, _, rel = rest.partition("/")
        yaml_path = os.path.join(get_package_share_directory(pkg_name), rel)
    else:
        yaml_path = resolve_ros_path(yaml_ref)
    with open(yaml_path) as f:
        params = _yaml.safe_load(f) or {}

    joints_cfg = robot_config.get("joints", {}) or {}
    arm_joint_names = joints_cfg.get("arm", [])
    if not arm_joint_names:
        raise ConfigError(
            "solver=safe_servo requires robot.joints.arm to list the arm "
            "joint names (used to order the position command output)"
        )
    params["arm_joint_names"] = arm_joint_names

    # The node needs robot_description so MoveIt's compute_ik can resolve
    # the configured ik_link_name and collision model.
    extra = {}
    if robot_description_dict:
        extra.update(robot_description_dict)

    return Node(
        package="robot_moveit",
        executable="so101_safe_servo_node.py",
        name="so101_safe_servo_node",
        output="screen",
        parameters=[params, extra],
    )


def validate_teleop_config(teleop_config: dict[str, object]) -> list[str]:
    """
    Validate teleoperation configuration.

    Args:
        teleop_config: Teleoperation configuration dictionary

    Returns:
        List of validation error messages (empty if valid)

    Example:
        >>> errors = validate_teleop_config(config['teleoperation'])
        >>> if errors:
        ...     for error in errors:
        ...         print(f"Error: {error}")
    """
    errors = []

    if not teleop_config.get("enabled", False):
        return errors  # Not enabled, skip validation

    devices = teleop_config.get("devices", [])
    if not devices:
        errors.append("devices list is empty")
        return errors

    active_devices = teleop_config.get("active_devices")
    active_device = teleop_config.get("active_device")
    if active_devices:
        if not isinstance(active_devices, list):
            errors.append("active_devices must be a list when specified")
            return errors
        active_device_names = active_devices
    elif active_device:
        active_device_names = [active_device]
    else:
        errors.append("active_device or active_devices must be specified when teleoperation is enabled")
        return errors

    devices_by_name = {device.get("name"): device for device in devices}
    for active_device_name in active_device_names:
        device = devices_by_name.get(active_device_name)
        if not device:
            errors.append(f"active device '{active_device_name}' not found in devices list")
            continue

        # Validate device type
        if not device.get("type"):
            errors.append(f"Device '{active_device_name}': missing 'type' field")

        # Type-specific validation
        if device.get("type") == "leader_arm" and not device.get("port"):
            errors.append(f"Device '{active_device_name}': leader_arm requires 'port' field")

        if device.get("type") == "phone":
            phone_config = device.get("phone_config", {})
            if not phone_config:
                errors.append(f"Device '{active_device_name}': phone requires 'phone_config' field")

    # Validate safety config
    safety = teleop_config.get("safety", {})
    joint_limits = safety.get("joint_limits", {})

    if not joint_limits:
        errors.append("safety.joint_limits not specified (recommended for safe operation)")
    else:
        # Validate joint limit format
        for joint_name, limits in joint_limits.items():
            if "min" not in limits or "max" not in limits:
                errors.append(f"Joint '{joint_name}': limits must have 'min' and 'max' fields")
            elif limits["min"] >= limits["max"]:
                errors.append(f"Joint '{joint_name}': min must be less than max")

    return errors


def get_recording_topics(robot_config: dict) -> list[str]:
    """
    Get list of topics to record for teleoperation sessions.

    Args:
        robot_config: Robot configuration dictionary

    Returns:
        List of topic names for rosbag recording

    Example:
        >>> topics = get_recording_topics(config)
        >>> cmd = ['ros2', 'bag', 'record'] + topics
    """
    topics = []

    # Always record joint states
    topics.append("/joint_states")

    # Add controller command topics
    topics.append("/arm_position_controller/commands")
    topics.append("/gripper_position_controller/commands")

    # Add teleop diagnostics
    topics.append("/diagnostics")

    # Add camera topics from peripherals
    peripherals = robot_config.get("peripherals", [])
    for peripheral in peripherals:
        if peripheral.get("type") == "camera":
            name = peripheral.get("name", "camera")
            # Add common camera topics
            topics.append(f"/camera/{name}/image_raw")
            topics.append(f"/camera/{name}/camera_info")

    return topics
