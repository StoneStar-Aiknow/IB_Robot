"""Minimal Gazebo + Nav2 test launch for Layer 3 E2E tests.

gpu_lidar is disabled due to EGL rendering issues on hybrid GPU systems.
Instead, a static TF (map->odom) is used for localization, and costmaps
use only the static map layer (no dynamic obstacle detection).

Starts:
  1. robot_state_publisher (URDF with use_sim:=true)
  2. Gazebo server (nav_test.world)
  3. ros_gz_bridge: /clock, joint_state
  4. wait_for_clock -> spawn entity -> controller spawner
  5. cmd_vel relay (/cmd_vel -> /base_controller/cmd_vel_unstamped)
  6. Static TF: map -> odom (at spawn position)
  7. Nav2 bringup (Planner + DWB, use_sim_time:=True, no AMCL)

Usage:
    # Headless (default, for CI/testing)
    ros2 launch robot_navigation sim_e2e.launch.py

    # With Gazebo GUI (for debugging)
    ros2 launch robot_navigation sim_e2e.launch.py gui:=true
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    # -- Constants ---------------------------------------------------------
    SPAWN_X = "-1.5"
    SPAWN_Y = "-1.5"

    # -- Package directories -----------------------------------------------
    robot_nav_dir = get_package_share_directory("robot_navigation")
    lekiwi_desc_dir = get_package_share_directory("lekiwi_description")
    robot_config_dir = get_package_share_directory("robot_config")
    ros_gz_sim_dir = get_package_share_directory("ros_gz_sim")
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    # -- Launch arguments --------------------------------------------------
    gui_arg = DeclareLaunchArgument(
        "gui",
        default_value="false",
        description="Launch Gazebo GUI (true) or run headless (false)",
    )
    world_file_arg = DeclareLaunchArgument(
        "world_file",
        default_value=os.path.join(robot_config_dir, "config", "worlds", "nav_test.world"),
        description="Absolute path to Gazebo world file",
    )
    nav2_params_arg = DeclareLaunchArgument(
        "nav2_params",
        default_value=os.path.join(robot_nav_dir, "config", "nav2_sim_params.yaml"),
        description="Absolute path to Nav2 params file",
    )
    sim_map_arg = DeclareLaunchArgument(
        "sim_map",
        default_value=os.path.join(robot_nav_dir, "config", "maps", "sim_map.yaml"),
        description="Absolute path to simulation map yaml",
    )
    controllers_config_arg = DeclareLaunchArgument(
        "controllers_config",
        default_value=os.path.join(robot_config_dir, "config", "lekiwi", "lekiwi_sim_controllers.yaml"),
        description="Absolute path to controller manager config",
    )

    # -- GZ_SIM_RESOURCE_PATH so Gazebo can find mesh files ----------------
    gazebo_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH",
        value=[str(Path(lekiwi_desc_dir).parent.resolve())],
    )

    # -- 1. robot_state_publisher ------------------------------------------
    urdf_path = os.path.join(lekiwi_desc_dir, "urdf", "lekiwi_sim.urdf.xacro")
    robot_description = ParameterValue(
        Command(
            [
                "xacro ",
                urdf_path,
                " use_sim:=true",
                " gz_ros2_control_parameters_file:=",
                LaunchConfiguration("controllers_config"),
            ]
        ),
        value_type=str,
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description, "use_sim_time": True, "frame_prefix": ""}],
        output="screen",
    )

    # -- 2. Gazebo (condition resolved via OpaqueFunction) ----------------
    def gazebo_launch(context):
        """Resolve gui flag at launch time and return the correct action."""
        gui_flag = context.launch_configurations.get("gui", "false")
        if gui_flag == "true":
            gz_args = " -v 4 -r " + context.launch_configurations["world_file"]
        else:
            gz_args = " -v 4 -r -s " + context.launch_configurations["world_file"]
        return [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(ros_gz_sim_dir, "launch", "gz_sim.launch.py")),
                launch_arguments=[("gz_args", [gz_args])],
            )
        ]

    gazebo_server = OpaqueFunction(function=gazebo_launch)

    # -- 3. ros_gz_bridge: /clock ------------------------------------------
    clock_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    # -- 4. ros_gz_bridge: joint_state --------------------------------------
    joint_state_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/world/nav_test/model/lekiwi/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model"],
        output="screen",
    )

    # -- 5. Static TF: map -> odom -----------------------------------------
    # gpu_lidar is disabled; use static TF at spawn position instead of AMCL.
    # Ground truth odom->base_footprint TF published by Gazebo OdometryPublisher + bridge.
    map_to_odom_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments=[
            SPAWN_X,
            SPAWN_Y,
            "0.0",  # x, y, z (spawn position)
            "0",
            "0",
            "0",  # roll, pitch, yaw
            "map",
            "odom",  # parent, child (map -> odom)
        ],
        output="screen",
    )

    # -- 5b. Ground truth odometry bridge (Gazebo -> ROS) ------------------
    # Bridges gz::sim::systems::OdometryPublisher output to ROS Odometry.
    odom_gt_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/model/lekiwi/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry"],
        output="screen",
    )

    # -- 5c. Ground truth odometry relay (subscribes Gazebo world pose, publishes /odom)
    gt_odom_script = os.path.join(robot_nav_dir, "test", "e2e", "gt_odom_node.py")
    gt_odom_node = ExecuteProcess(
        cmd=["python3", gt_odom_script, "--ros-args", "-p", f"spawn_x:={SPAWN_X}", "-p", f"spawn_y:={SPAWN_Y}"],
        output="screen",
    )

    # -- 6. wait_for_clock (blocks until /clock is published) ---------------
    wait_for_clock = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "robot_config",
            "wait_for_clock",
            "--topic",
            "/clock",
            "--timeout",
            "60",
        ],
        output="screen",
    )

    # -- 7. Spawn entity (after clock is ready) -----------------------------
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic",
            "robot_description",
            "-name",
            "lekiwi",
            "-x",
            SPAWN_X,
            "-y",
            SPAWN_Y,
            "-z",
            "0.01",
        ],
        output="screen",
    )

    # -- 8. Controller spawner (uses spawner node, not CLI) ------------------
    # ros2 control CLI uses XML-RPC which fails with gz_ros2_control
    # (xmlrpc.client.Fault: !rclpy.ok()). The spawner node uses direct
    # ROS 2 service calls and works correctly.
    controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "base_controller",
            "--controller-manager",
            "controller_manager",
            "--controller-manager-timeout",
            "60",
            "--switch-timeout",
            "30",
            "--activate-as-group",
        ],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    # -- 9. cmd_vel relay (/cmd_vel -> /base_controller/cmd_vel_unstamped) --
    cmd_vel_relay_script = os.path.join(robot_nav_dir, "test", "e2e", "cmd_vel_relay.py")
    cmd_vel_relay = ExecuteProcess(
        cmd=["python3", cmd_vel_relay_script],
        output="screen",
    )

    # -- 10. Nav2: map_server + lifecycle manager + navigation (no AMCL) ----
    # Start map_server with its own lifecycle manager (avoids ros2 lifecycle
    # CLI which uses XML-RPC and fails with gz_ros2_control).
    map_server_node = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        parameters=[
            {"use_sim_time": True, "yaml_filename": LaunchConfiguration("sim_map")},
        ],
        output="screen",
    )

    map_server_lifecycle = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="map_server_lifecycle_manager",
        parameters=[
            {"use_sim_time": True, "autostart": True, "node_names": ["map_server"]},
        ],
        output="screen",
    )

    nav2_navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")),
        launch_arguments={
            "use_sim_time": "True",
            "use_composition": "False",
            "autostart": "True",
            "params_file": LaunchConfiguration("nav2_params"),
        }.items(),
    )

    # -- Event handlers for sequential startup ------------------------------
    spawn_on_clock = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=wait_for_clock,
            on_exit=[spawn_entity],
        )
    )

    controller_on_spawn = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_entity,
            on_exit=[controller_spawner, cmd_vel_relay],
        )
    )

    nav2_on_controllers = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=controller_spawner,
            on_exit=[map_server_node, map_server_lifecycle, nav2_navigation],
        )
    )

    return LaunchDescription(
        [
            # Arguments
            gui_arg,
            world_file_arg,
            nav2_params_arg,
            sim_map_arg,
            controllers_config_arg,
            # Environment
            gazebo_resource_path,
            # Always-running nodes
            robot_state_publisher,
            gazebo_server,
            clock_bridge,
            joint_state_bridge,
            map_to_odom_tf,
            odom_gt_bridge,
            gt_odom_node,
            # Sequential startup chain
            wait_for_clock,
            spawn_on_clock,
            controller_on_spawn,
            nav2_on_controllers,
        ]
    )
