"""MoveIt 2 launch builders.

This module handles:
- MoveIt 2 core node generation (move_group)
- RViz visualization for MoveIt
- Inclusion of external MoveIt launch files
"""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.moveit")


def generate_moveit_nodes(robot_config, control_mode, use_sim=False, display=True, force=False):
    """Generate MoveIt 2 nodes.

    Args:
        robot_config: Robot configuration dict
        control_mode: Active control mode
        use_sim: Simulation mode flag
        display: Whether to launch RViz visualization
        force: When True, skip the ``'moveit' in control_mode`` guard. Used
            when callers explicitly request MoveIt for a non-MoveIt control
            mode, such as a Cartesian backend that needs ``move_group``
            services.

    Returns:
        List of launch actions for MoveIt 2
    """
    actions = []

    # Check if MoveIt is needed for this control mode
    # Usually enabled for 'moveit_planning' or any mode with 'moveit' in name
    with_moveit = force or ("moveit" in control_mode.lower())

    if not with_moveit:
        return actions

    logger.info(f"Generating MoveIt Nodes (mode: {control_mode}, display: {display})")

    # Find MoveIt launch file
    try:
        moveit_package_dir = get_package_share_directory("robot_moveit")
        moveit_launch_file = Path(moveit_package_dir) / "launch" / "so101_moveit.launch.py"

        if moveit_launch_file.exists():
            # Get joint_names from robot_config to pass to MoveIt launch
            joint_names = robot_config["joints"]["arm"]
            # Convert list to space-separated string for launch argument
            joint_names_str = " ".join(joint_names)

            # Get MoveIt gateway parameters from robot_config
            arm_group_name = robot_config["moveit"]["arm_group_name"]
            base_link = robot_config["moveit"]["base_link"]
            ee_link = robot_config["moveit"]["ee_link"]
            shoulder_link = robot_config["moveit"]["shoulder_link"]
            motion_status_hold_s = max(float(robot_config["moveit"].get("motion_status_hold_s", 0.0)), 0.0)
            motion_feedback_timeout_s = max(float(robot_config["moveit"].get("motion_feedback_timeout_s", 0.3)), 0.0)
            motion_feedback_tolerance_rad = max(
                float(robot_config["moveit"].get("motion_feedback_tolerance_rad", 0.12)), 0.0
            )
            motion_require_tf_sync = bool(robot_config["moveit"].get("motion_require_tf_sync", True))
            motion_hardware_feedback_topic = (
                "" if use_sim else str(robot_config["moveit"].get("motion_hardware_feedback_topic", "")).strip()
            )
            joint_state_topic = str(robot_config["moveit"].get("joint_state_topic", "/joint_states")).strip()
            motion_mode = robot_config.get("motion_mode", {})
            motion_mode_enabled = bool(motion_mode.get("enabled", False))
            if motion_mode_enabled and "navigation_enabled_on_startup" not in motion_mode:
                raise ValueError(
                    "robot.motion_mode.navigation_enabled_on_startup is required when motion_mode is enabled"
                )

            # Include MoveIt launch file
            moveit_launch = IncludeLaunchDescription(
                PythonLaunchDescriptionSource(str(moveit_launch_file)),
                launch_arguments={
                    "is_sim": "True" if use_sim else "False",
                    "display": "True" if display else "False",
                    "joint_names": joint_names_str,
                    "arm_group_name": arm_group_name,
                    "base_link": base_link,
                    "ee_link": ee_link,
                    "shoulder_link": shoulder_link,
                    "motion_status_hold_s": str(motion_status_hold_s),
                    "motion_feedback_timeout_s": str(motion_feedback_timeout_s),
                    "motion_feedback_tolerance_rad": str(motion_feedback_tolerance_rad),
                    "motion_require_tf_sync": str(motion_require_tf_sync),
                    "motion_hardware_feedback_topic": motion_hardware_feedback_topic,
                    "joint_state_topic": joint_state_topic,
                    "motion_mode_enabled": str(motion_mode_enabled),
                    "navigation_enabled_on_startup": str(bool(motion_mode.get("navigation_enabled_on_startup", False))),
                    "navigation_enabled_topic": str(
                        motion_mode.get("navigation_enabled_topic", "motion_mode/navigation_enabled")
                    ),
                    "navigation_mode_ack_topic": str(
                        motion_mode.get("navigation_mode_ack_topic", "motion_mode/base_navigation_enabled")
                    ),
                    "set_navigation_enabled_service": str(
                        motion_mode.get("set_navigation_enabled_service", "motion_mode/set_navigation_enabled")
                    ),
                    "controller_switch_service": str(
                        motion_mode.get("controller_switch_service", "controller_manager/switch_controller")
                    ),
                    "motion_mode_manipulation_controllers": " ".join(
                        motion_mode.get(
                            "manipulation_controllers",
                            ["arm_trajectory_controller", "gripper_trajectory_controller"],
                        )
                    ),
                    "motion_mode_navigation_controllers": " ".join(
                        motion_mode.get("navigation_controllers", ["base_velocity_controller"])
                    ),
                    "motion_mode_transition_timeout_s": str(
                        max(float(motion_mode.get("transition_timeout_s", 2.0)), 0.0)
                    ),
                    "motion_mode_bridge_heartbeat_timeout_s": str(
                        max(float(motion_mode.get("bridge_heartbeat_timeout_s", 1.0)), 0.0)
                    ),
                }.items(),
            )
            actions.append(moveit_launch)
            logger.info(f"Added MoveIt launch (is_sim={use_sim}, display={display})")
        else:
            logger.warning(f"MoveIt launch file not found at {moveit_launch_file}")
    except Exception as e:
        logger.warning(f"Could not find robot_moveit package: {e}")
        logger.info("Continuing without MoveIt...")

    return actions
