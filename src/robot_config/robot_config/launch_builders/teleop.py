"""Teleoperation node generation for robot_config.

This module provides utilities to generate teleoperation nodes
for integration with the robot_config launch system.
"""

import json
import os
import re
from pathlib import Path

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
    # Cartesian solver selection lives in robot.teleoperation.cartesian.solver.
    # The default tool frame follows robot.moveit.ee_link so arm/gripper and
    # arm_tcp/tcp remain a single MoveIt-owned SSOT; cartesian.tool_frame is
    # only an explicit override.
    cart_cfg = teleop_config.get("cartesian", {}) or {}
    cart_solver = cart_cfg.get("solver", "placo_servo")
    if cart_solver not in ("moveit_servo", "placo_servo"):
        raise ValueError(f"teleoperation.cartesian.solver must be 'moveit_servo' or 'placo_servo', got {cart_solver!r}")
    moveit_cfg = robot_config.get("moveit", {}) or {}
    cart_tool_frame = (
        cart_cfg.get("tool_frame") or device_config.get("ee_frame_name") or moveit_cfg.get("ee_link") or "gripper"
    )
    _validate_tool_frame(cart_tool_frame, robot_config)
    device_param["cartesian_solver"] = cart_solver
    device_param["tool_frame"] = cart_tool_frame
    device_param["base_link_name"] = device_param.get("base_link_name", moveit_cfg.get("base_link", "base"))

    if cart_solver == "placo_servo":
        placo_cfg = cart_cfg.get("placo_servo", {}) or {}
        placo_linear_speed = placo_cfg.get("linear_speed", 0.3)
        placo_angular_speed = placo_cfg.get("angular_speed", 0.7)
        cp = device_param.setdefault("control_params", {})
        cp["cartesian_linear_speed"] = placo_linear_speed
        cp["cartesian_angular_speed"] = placo_angular_speed
        logger.info(
            f"placo_servo speed override: linear={placo_linear_speed} (0~1), angular={placo_angular_speed} (0~1)"
        )

    logger.info(f"Cartesian solver={cart_solver}, tool_frame={cart_tool_frame}")

    device_type = device_config.get("type", "")

    control_frequency = device_config.get("control_frequency", 50.0)

    if device_type == "joy_teleop":
        return _create_joy_teleop_nodes(device_config)

    if device_type == "vr_teleop":
        return _generate_vr_teleop_nodes(
            robot_config,
            device_config,
            robot_description_dict,
            cart_solver=cart_solver,
            base_link_name=device_param["base_link_name"],
            tool_frame=cart_tool_frame,
        )

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

    # Add the selected Cartesian backend node for Xbox controller and phone.
    if device_config.get("type") in ("xbox_controller", "phone"):
        if cart_solver == "moveit_servo":
            servo_node = _create_servo_node(robot_config, device_config, robot_description_dict)
            nodes.append(servo_node)
            logger.info("Generated servo_node for Cartesian servo (MoveIt Servo) control")
        elif cart_solver == "placo_servo":
            placo_node = _create_so101_placo_servo_node(robot_config, device_config, robot_description_dict)
            nodes.append(placo_node)
            logger.info("Generated so101_placo_servo_node for SO101 Placo QP Cartesian control")

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


def _generate_vr_teleop_nodes(
    robot_config: dict,
    device_config: dict,
    robot_description_dict: dict = None,
    *,
    cart_solver: str = "placo_servo",
    base_link_name: str = "base",
    tool_frame: str = "gripper",
) -> list[Node]:
    """Launch the standalone VR teleop server + (for so101) the placo servo.

    The VR server (``vr_teleop``) is a self-contained rclpy node — it
    does NOT go through ``teleop_node`` / device_factory. Its so101 output
    profile drives the placo cartesian servo, so downstream topic/service names
    default to ``so101_placo_servo_node`` and this builder also spawns that
    servo node when ``output_profile == 'so101'``.
    """
    nodes = []
    vr_cfg = device_config.get("vr_config", {}) or {}
    output_profile = str(vr_cfg.get("output_profile", "humanoid"))
    # so101 input mode: "velocity" (differential twist → placo integrates) or
    # "pose" (absolute EE pose passthrough → placo sets its reference). This
    # single source of truth in vr_config drives BOTH the VR node and the placo
    # servo node (the placo node's input_mode is overridden below to match).
    so101_input_mode = str(vr_cfg.get("so101_input_mode", "velocity"))
    so101_pose_topic = vr_cfg.get("so101_pose_topic", "/so101_placo_servo_node/pose_cmd_base")

    env = prepare_lerobot_env()

    # control_frequency is declared at the device layer (sibling of vr_config)
    # to match every other teleop device; accept it there as the default and
    # still allow a vr_config override. Reading only vr_config.control_frequency
    # silently ignored the device-layer value.
    control_frequency = float(
        vr_cfg.get("control_frequency", device_config.get("control_frequency", 50.0))
    )

    # Resolve the placo start/stop/home service names ONCE (with defaults) so
    # both the VR node and the placo node below are wired to the same names.
    # A user override of any of these in vr_config must reach BOTH sides.
    so101_start_service = vr_cfg.get(
        "so101_start_service", "/so101_placo_servo_node/start"
    )
    so101_stop_service = vr_cfg.get(
        "so101_stop_service", "/so101_placo_servo_node/stop"
    )
    so101_home_service = vr_cfg.get(
        "so101_home_service", "/so101_placo_servo_node/home"
    )

    # Downstream placo wiring (overridable via vr_config). base/tool frames come
    # from the shared cartesian SSOT so the tool→base angular conversion inside
    # the VR node uses the same frames as the xbox/phone placo path.
    vr_params = {
        "host": vr_cfg.get("host", "0.0.0.0"),
        "port": int(vr_cfg.get("port", 8889)),
        "control_frequency": control_frequency,
        "output_profile": output_profile,
        "controller_side": vr_cfg.get("controller_side", "right"),
        "base_link_name": vr_cfg.get("base_link_name", base_link_name),
        "tool_frame": vr_cfg.get("tool_frame", tool_frame),
        "so101_linear_topic": vr_cfg.get(
            "so101_linear_topic", "/so101_placo_servo_node/linear_cmd_base"
        ),
        "so101_angular_topic": vr_cfg.get(
            "so101_angular_topic", "/so101_placo_servo_node/angular_cmd_base"
        ),
        "so101_gripper_topic": vr_cfg.get(
            "so101_gripper_topic", "/gripper_position_controller/commands"
        ),
        "so101_start_service": so101_start_service,
        "so101_stop_service": so101_stop_service,
        "so101_home_service": so101_home_service,
        "so101_input_mode": so101_input_mode,
        "so101_pose_topic": so101_pose_topic,
    }
    # Forward any remaining tuning knobs verbatim (speed scales, deadzones,
    # gripper open/closed, tf_stale_threshold_s, position_scale, etc.).
    _passthrough = {
        "linear_speed_scale",
        "angular_speed_scale",
        "max_linear_speed",
        "max_angular_speed",
        "velocity_ema_alpha",
        "linear_deadzone",
        "angular_deadzone",
        "so101_gripper_open",
        "so101_gripper_closed",
        "tf_stale_threshold_s",
        "position_scale",
        "so101_position_only",
        "so101_command_stale_s",
        "so101_home_settle_s",
    }
    for key in _passthrough:
        if key in vr_cfg:
            vr_params[key] = vr_cfg[key]

    nodes.append(
        Node(
            package="robot_teleop",
            executable="vr_teleop",
            name="vr_teleop",
            output="screen",
            env=env,
            parameters=[vr_params],
        )
    )
    logger.info(f"Generated vr_teleop (output_profile={output_profile})")

    # so101 profile drives the cartesian servo; spawn placo alongside.
    if output_profile == "so101":
        if cart_solver != "placo_servo":
            raise ValueError(
                "vr_teleop output_profile=so101 requires cartesian.solver="
                f"'placo_servo', got {cart_solver!r}"
            )
        placo_node = _create_so101_placo_servo_node(
            robot_config,
            device_config,
            robot_description_dict,
            input_mode=so101_input_mode,
            pose_cmd_topic=so101_pose_topic,
            start_service=so101_start_service,
            stop_service=so101_stop_service,
            home_service=so101_home_service,
        )
        nodes.append(placo_node)
        logger.info(
            f"Generated so101_placo_servo_node for VR so101 Cartesian control "
            f"(input_mode={so101_input_mode})"
        )

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
# Cartesian helpers: tool_frame validation + placo_servo launcher
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Raised when the SSOT YAML is internally inconsistent."""


def _validate_tool_frame(tool_frame: str, robot_config: dict) -> None:
    """Confirm ``tool_frame`` resolves at launch time.

    Accepts the configured ``base_link`` / ``ee_link`` and otherwise lets URDF
    links through to runtime TF validation. Launch-time code does not enumerate
    URDF links, so the Servo node is the final authority for custom link names.
    """
    if not tool_frame:
        raise ConfigError("tool_frame is empty")
    moveit_cfg = robot_config.get("moveit", {}) or {}
    base_link = moveit_cfg.get("base_link", "base")
    if tool_frame == base_link:
        return
    if tool_frame == moveit_cfg.get("ee_link"):
        return
    # Assume the user supplied a URDF link name; runtime TF will validate it
    # and the Servo node will fail fast if the link does not exist.
    logger.warning(
        f"tool_frame={tool_frame!r} is neither base_link nor ee_link; "
        f"assuming it is a URDF link. Runtime TF will validate."
    )


def _create_so101_placo_servo_node(
    robot_config: dict,
    device_config: dict,  # noqa: ARG001 — kept for signature parity with siblings
    robot_description_dict: dict = None,
    *,
    input_mode: str = None,
    pose_cmd_topic: str = None,
    start_service: str = None,
    stop_service: str = None,
    home_service: str = None,
) -> Node:
    """Launch ``so101_placo_servo_node``.

    Loads the solver YAML from ``moveit.so101_placo_servo_config_path`` and
    appends arm joint names from ``robot.joints.arm`` so the node knows the
    controller output order.
    """
    import yaml as _yaml

    moveit_cfg = robot_config.get("moveit", {}) or {}
    yaml_ref = moveit_cfg.get("so101_placo_servo_config_path")
    if not yaml_ref:
        raise ConfigError(
            "solver=placo_servo requires moveit.so101_placo_servo_config_path "
            "(e.g. $(find robot_moveit)/config/so101_placo_servo.yaml)"
        )
    yaml_path = resolve_ros_path(yaml_ref)
    with open(yaml_path) as f:
        params = _yaml.safe_load(f) or {}

    joints_cfg = robot_config.get("joints", {}) or {}
    arm_joint_names = joints_cfg.get("arm", [])
    if not arm_joint_names:
        raise ConfigError(
            "solver=placo_servo requires robot.joints.arm to list the arm "
            "joint names (used to order the position command output)"
        )
    params["arm_joint_names"] = arm_joint_names

    # Drop-in TCP support: the IK tip frame is the SSOT moveit.ee_link
    # (gripper | tcp). Inject it so selecting tcp re-targets placo's frame
    # task without editing the solver YAML. Defaults to the YAML value (gripper)
    # when ee_link is absent, so the gripper path is byte-for-byte unchanged.
    ee_link = moveit_cfg.get("ee_link")
    if ee_link:
        params["ik_link_name"] = ee_link

    # VR pose passthrough: when the caller (VR builder) selects pose mode, override
    # the servo node's input_mode and pose topic to match vr_config — single
    # source of truth. Left as the YAML default ("velocity") for xbox/phone.
    if input_mode is not None:
        params["input_mode"] = input_mode
    if pose_cmd_topic is not None:
        params["pose_cmd_topic"] = pose_cmd_topic

    # Service names MUST match whatever the VR node was told to call. The VR
    # builder resolves these from vr_config (with the same defaults) and passes
    # them here so a user override of so101_start/stop/home_service re-targets
    # BOTH nodes. Forwarding only to the VR node (as before) left placo serving
    # the default names while VR called the overridden ones — the handshake and
    # the deadman stop would silently miss.
    if start_service is not None:
        params["start_service"] = start_service
    if stop_service is not None:
        params["stop_service"] = stop_service
    if home_service is not None:
        params["home_service"] = home_service

    # Home pose for the B-button go-home service: inject the EE pose from
    # embodied.named_poses.home (base frame). The placo node drives the EE here
    # smoothly when its home service is called. Absent/malformed => left as the
    # YAML default (zeros), which the node treats as "home disabled".
    named_poses = ((robot_config.get("embodied", {}) or {}).get("named_poses", {}) or {})
    home_pose = named_poses.get("home", {}) or {}
    hp = home_pose.get("position") or {}
    ho = home_pose.get("orientation") or {}
    if hp and ho:
        params["home_position"] = [float(hp.get("x", 0.0)), float(hp.get("y", 0.0)), float(hp.get("z", 0.0))]
        params["home_orientation"] = [
            float(ho.get("x", 0.0)), float(ho.get("y", 0.0)),
            float(ho.get("z", 0.0)), float(ho.get("w", 1.0)),
        ]

    # The node expands the so101 xacro in-memory at runtime via the
    # robot_description package share dir, so robot_description_dict is not
    # required for kinematics; it is still forwarded for parity / use_sim_time.
    extra = {}
    if robot_description_dict:
        extra.update(robot_description_dict)

    return Node(
        package="robot_moveit",
        executable="so101_placo_servo_node.py",
        name="so101_placo_servo_node",
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
