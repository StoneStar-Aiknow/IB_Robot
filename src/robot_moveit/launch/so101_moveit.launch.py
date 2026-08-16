import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # create a runtime lauch argument
    is_sim_arg = DeclareLaunchArgument(name="is_sim", default_value="True")

    # 2. 声明 "display" 启动参数，默认为 True
    display_arg = DeclareLaunchArgument(
        name="display", default_value="True", description="Launch RViz for visualization if True"
    )

    # 3. 声明 "joint_names" 启动参数（从 robot_config 传入）
    joint_names_arg = DeclareLaunchArgument(
        name="joint_names", description="Joint names for the arm group (space-separated, required)"
    )
    joint_state_topic_arg = DeclareLaunchArgument(
        name="joint_state_topic",
        default_value="/joint_states",
        description="Arm-only joint state topic consumed by MoveIt",
    )

    # 4. 声明 MoveIt gateway 参数（从 robot_config 传入）
    arm_group_name_arg = DeclareLaunchArgument(
        name="arm_group_name", description="MoveIt planning group name (required)"
    )
    base_link_arg = DeclareLaunchArgument(name="base_link", description="Base link frame id (required)")
    ee_link_arg = DeclareLaunchArgument(name="ee_link", description="End effector link frame id (required)")
    shoulder_link_arg = DeclareLaunchArgument(name="shoulder_link", description="Shoulder link frame id (required)")
    motion_status_hold_arg = DeclareLaunchArgument(
        name="motion_status_hold_s", default_value="0.0", description="Legacy terminal motion-status hold"
    )
    motion_feedback_timeout_arg = DeclareLaunchArgument(
        name="motion_feedback_timeout_s",
        default_value="0.3",
        description="Maximum wait for fresh post-motion joint feedback",
    )
    motion_feedback_tolerance_arg = DeclareLaunchArgument(
        name="motion_feedback_tolerance_rad",
        default_value="0.12",
        description="Maximum arm joint error accepted by the post-motion feedback barrier",
    )
    motion_require_tf_sync_arg = DeclareLaunchArgument(
        name="motion_require_tf_sync",
        default_value="True",
        description="Require the end-effector TF to include the accepted post-motion joint sample",
    )
    motion_hardware_feedback_topic_arg = DeclareLaunchArgument(
        name="motion_hardware_feedback_topic",
        default_value="",
        description="Hardware-read heartbeat topic required by the post-motion barrier",
    )
    motion_mode_enabled_arg = DeclareLaunchArgument(name="motion_mode_enabled", default_value="False")
    navigation_enabled_on_startup_arg = DeclareLaunchArgument(
        name="navigation_enabled_on_startup", default_value="False"
    )
    navigation_enabled_topic_arg = DeclareLaunchArgument(
        name="navigation_enabled_topic", default_value="motion_mode/navigation_enabled"
    )
    navigation_mode_ack_topic_arg = DeclareLaunchArgument(
        name="navigation_mode_ack_topic", default_value="motion_mode/base_navigation_enabled"
    )
    set_navigation_enabled_service_arg = DeclareLaunchArgument(
        name="set_navigation_enabled_service", default_value="motion_mode/set_navigation_enabled"
    )
    controller_switch_service_arg = DeclareLaunchArgument(
        name="controller_switch_service", default_value="controller_manager/switch_controller"
    )
    motion_mode_manipulation_controllers_arg = DeclareLaunchArgument(
        name="motion_mode_manipulation_controllers",
        default_value="arm_trajectory_controller gripper_trajectory_controller",
    )
    motion_mode_navigation_controllers_arg = DeclareLaunchArgument(
        name="motion_mode_navigation_controllers", default_value="base_velocity_controller"
    )
    motion_mode_transition_timeout_arg = DeclareLaunchArgument(
        name="motion_mode_transition_timeout_s", default_value="2.0"
    )
    motion_mode_bridge_heartbeat_timeout_arg = DeclareLaunchArgument(
        name="motion_mode_bridge_heartbeat_timeout_s", default_value="1.0"
    )

    # get the argument value at runtime
    is_sim = LaunchConfiguration("is_sim")
    display = LaunchConfiguration("display")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    arm_group_name = LaunchConfiguration("arm_group_name")
    base_link = LaunchConfiguration("base_link")
    ee_link = LaunchConfiguration("ee_link")
    shoulder_link = LaunchConfiguration("shoulder_link")
    motion_status_hold_s = LaunchConfiguration("motion_status_hold_s")
    motion_feedback_timeout_s = LaunchConfiguration("motion_feedback_timeout_s")
    motion_feedback_tolerance_rad = LaunchConfiguration("motion_feedback_tolerance_rad")
    motion_require_tf_sync = LaunchConfiguration("motion_require_tf_sync")
    motion_hardware_feedback_topic = LaunchConfiguration("motion_hardware_feedback_topic")
    motion_mode_enabled = LaunchConfiguration("motion_mode_enabled")
    navigation_enabled_on_startup = LaunchConfiguration("navigation_enabled_on_startup")
    navigation_enabled_topic = LaunchConfiguration("navigation_enabled_topic")
    navigation_mode_ack_topic = LaunchConfiguration("navigation_mode_ack_topic")
    set_navigation_enabled_service = LaunchConfiguration("set_navigation_enabled_service")
    controller_switch_service = LaunchConfiguration("controller_switch_service")
    motion_mode_manipulation_controllers = LaunchConfiguration("motion_mode_manipulation_controllers")
    motion_mode_navigation_controllers = LaunchConfiguration("motion_mode_navigation_controllers")
    motion_mode_transition_timeout_s = LaunchConfiguration("motion_mode_transition_timeout_s")
    motion_mode_bridge_heartbeat_timeout_s = LaunchConfiguration("motion_mode_bridge_heartbeat_timeout_s")

    # URDF
    robot_description_dir = get_package_share_directory("robot_description")
    so101_urdf_path = os.path.join(robot_description_dir, "urdf", "lerobot", "so101", "so101.urdf.xacro")

    moveit_config = (
        MoveItConfigsBuilder("so101", package_name="robot_moveit")
        .robot_description(file_path=so101_urdf_path)
        .robot_description_semantic(file_path="config/lerobot/so101/so101.srdf")
        .robot_description_kinematics(file_path="config/lerobot/so101/kinematics.yaml")
        .joint_limits(file_path="config/lerobot/so101/joint_limits.yaml")
        .pilz_cartesian_limits(file_path="config/lerobot/so101/pilz_cartesian_limits.yaml")
        .trajectory_execution(file_path="config/lerobot/so101/moveit_controllers.yaml")
        .planning_pipelines(pipelines=["ompl"])
        .to_moveit_configs()
    )

    # moveit core
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            {"use_sim_time": is_sim},
            {"publish_robot_description_semantic": True},
        ],
        remappings=[("joint_states", joint_state_topic)],
        arguments=["--ros-args", "--log-level", "info"],
    )

    rviz_config_path = os.path.join(
        get_package_share_directory("robot_moveit"), "config", "lerobot", "so101", "moveit.rviz"
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        remappings=[("joint_states", joint_state_topic)],
        condition=IfCondition(display),
    )

    moveit_gateway_node = Node(
        package="robot_moveit",
        executable="moveit_gateway.py",
        name="moveit_gateway",
        output="screen",
        parameters=[
            {"arm_group_name": arm_group_name},
            {"base_link": base_link},
            {"ee_link": ee_link},
            {"shoulder_link": shoulder_link},
            {"joint_names": PythonExpression(["'", LaunchConfiguration("joint_names"), "'.split()"])},
            {"motion_status_hold_s": motion_status_hold_s},
            {"motion_feedback_timeout_s": motion_feedback_timeout_s},
            {"motion_feedback_tolerance_rad": motion_feedback_tolerance_rad},
            {"motion_require_tf_sync": motion_require_tf_sync},
            {"motion_hardware_feedback_topic": motion_hardware_feedback_topic},
            {"joint_state_topic": joint_state_topic},
            {"motion_mode_enabled": motion_mode_enabled},
            {"navigation_enabled_on_startup": navigation_enabled_on_startup},
            {"navigation_enabled_topic": navigation_enabled_topic},
            {"navigation_mode_ack_topic": navigation_mode_ack_topic},
            {"set_navigation_enabled_service": set_navigation_enabled_service},
            {"controller_switch_service": controller_switch_service},
            {
                "motion_mode_manipulation_controllers": PythonExpression(
                    ["'", motion_mode_manipulation_controllers, "'.split()"]
                )
            },
            {
                "motion_mode_navigation_controllers": PythonExpression(
                    ["'", motion_mode_navigation_controllers, "'.split()"]
                )
            },
            {"motion_mode_transition_timeout_s": motion_mode_transition_timeout_s},
            {"motion_mode_bridge_heartbeat_timeout_s": motion_mode_bridge_heartbeat_timeout_s},
            {"use_sim_time": is_sim},
        ],
        remappings=[("joint_states", joint_state_topic)],
    )

    return LaunchDescription(
        [
            is_sim_arg,
            display_arg,
            joint_names_arg,
            joint_state_topic_arg,
            arm_group_name_arg,
            base_link_arg,
            ee_link_arg,
            shoulder_link_arg,
            motion_status_hold_arg,
            motion_feedback_timeout_arg,
            motion_feedback_tolerance_arg,
            motion_require_tf_sync_arg,
            motion_hardware_feedback_topic_arg,
            motion_mode_enabled_arg,
            navigation_enabled_on_startup_arg,
            navigation_enabled_topic_arg,
            navigation_mode_ack_topic_arg,
            set_navigation_enabled_service_arg,
            controller_switch_service_arg,
            motion_mode_manipulation_controllers_arg,
            motion_mode_navigation_controllers_arg,
            motion_mode_transition_timeout_arg,
            motion_mode_bridge_heartbeat_timeout_arg,
            move_group_node,
            rviz_node,
            moveit_gateway_node,
        ]
    )
