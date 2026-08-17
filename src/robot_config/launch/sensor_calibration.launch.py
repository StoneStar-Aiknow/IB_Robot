"""Sensor launch with manual base control for extrinsic calibration.

This entry point owns the configured sensors, FAST-LIO, and the minimum
ros2_control chain needed for manual base teleoperation. It does not construct
Nav2, SLAM Toolbox, inference, or autonomous motion components.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction

from robot_config.launch_builders.cmd_vel import generate_cmd_vel_nodes
from robot_config.launch_builders.control import generate_ros2_control_nodes
from robot_config.launch_builders.fast_lio import generate_fast_lio_nodes
from robot_config.launch_builders.perception import (
    generate_camera_nodes,
    generate_lidar_nodes,
    generate_tf_nodes,
    generate_virtual_camera_relays,
)
from robot_config.launch_builders.static_tf import generate_static_tf_nodes
from robot_config.loader import load_robot_config_dict
from robot_config.utils import resolve_ros_path


def launch_setup(context, *args, **kwargs):
    config_path = context.launch_configurations.get("config_path", "")
    if not config_path:
        config_path = f"$(find robot_config)/config/robots/{context.launch_configurations['robot_config']}.yaml"
    robot_config = load_robot_config_dict(resolve_ros_path(config_path))
    actions = []
    actions.extend(generate_camera_nodes(robot_config, use_sim=False))
    actions.extend(generate_lidar_nodes(robot_config, use_sim=False))
    actions.extend(generate_tf_nodes(robot_config, use_sim=False))
    actions.extend(generate_virtual_camera_relays(robot_config))
    navigation_config = robot_config.get("navigation", {})
    actions.extend(generate_static_tf_nodes(navigation_config, use_sim=False))
    actions.extend(generate_fast_lio_nodes(navigation_config, use_sim=False))

    control_nodes, _controller_names, controller_spawners, _robot_description = generate_ros2_control_nodes(
        robot_config,
        use_sim=False,
        auto_start_controllers="true",
        controller_startup_timeout=300.0,
    )
    actions.extend(control_nodes)
    actions.extend(controller_spawners)
    actions.extend(generate_cmd_vel_nodes(navigation_config, use_sim=False))
    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_config", default_value="lekiwi_sensor_calib"),
            DeclareLaunchArgument("config_path", default_value=""),
            OpaqueFunction(function=launch_setup),
        ]
    )
