"""LeKiwi runtime launch entry (robot-runtime-packaging spec).

robot_state_publisher + ros2_control (lekiwi_hardware, shared bus, real or
SDK simulated transport) + controllers + so101_motion for the arm + the
LeKiwi base node (cmd_vel, odometry, navigation gating) + sensor peripherals
(D12: profile defaults, deployment overrides via peripherals:=<file>) with
the fast_lio lidar odometry chain + runtime facade.

    ros2 launch lekiwi_robot runtime.launch.py profile:=<path-or-name> \
        [simulated:=true] [display:=true] [peripherals:=<fragment.yaml>]
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_runtime.launch_support import (
    RuntimeProfileError,
    motion_launch_arguments,
    peripheral_stack_actions,
    resolve_profile_argument,
    resolve_simulated,
    runtime_stack_actions,
)
from robot_runtime.peripherals import fast_lio_nodes, load_peripherals_file
from robot_runtime.profile import load_profile


def _launch_setup(context, *_args, **_kwargs):
    profile_path = resolve_profile_argument("lekiwi_robot", LaunchConfiguration("profile").perform(context))
    profile = load_profile(profile_path)
    simulated = resolve_simulated(profile, LaunchConfiguration("simulated").perform(context))
    display = LaunchConfiguration("display").perform(context).strip().lower() in ("1", "true")

    base = profile.get("base") or {}
    if not base.get("wheel_joints"):
        raise RuntimeProfileError(f"runtime profile missing required key 'base.wheel_joints': {profile_path}")
    base_params = {
        "wheel_joints": [str(j) for j in base["wheel_joints"]],
        "wheel_radius": float(base.get("wheel_radius", 0.05)),
        "base_radius": float(base.get("base_radius", 0.125)),
        "max_wheel_radps": float(base.get("max_wheel_radps", 4.602)),
        "control_frequency": float(base.get("control_frequency", 50.0)),
        "staleness_s": float((profile["capabilities"].get("base.cmd_vel") or {}).get("staleness_s", 0.5)),
        "joint_states_topic": str(profile["joint_state_topic"]),
        "odom_frame": str(base.get("odom_frame", "odom")),
        "base_frame": str(base.get("base_frame", "base_link")),
        "publish_tf": bool(base.get("publish_tf", True)),
        "navigation_enabled_on_startup": bool(base.get("navigation_enabled_on_startup", False)),
        "base_modes": [
            name for name, spec in profile["modes"].items() if name != "initial" and (spec or {}).get("allows_base")
        ],
    }
    base_node = Node(
        package="lekiwi_robot",
        executable="base_node",
        name="lekiwi_base",
        output="screen",
        parameters=[base_params],
    )
    motion = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("so101_motion"), "launch", "motion.launch.py")
        ),
        launch_arguments=motion_launch_arguments(profile, profile_path, simulated, display).items(),
    )
    peripherals_file = LaunchConfiguration("peripherals").perform(context).strip()
    fragment = load_peripherals_file(peripherals_file) if peripherals_file else {}
    fast_lio_config = {**(profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}
    peripherals = [
        *peripheral_stack_actions(profile, peripherals_file, simulated),
        *fast_lio_nodes(fast_lio_config, bridge_package="lekiwi_robot", use_sim=simulated),
    ]
    return [
        *runtime_stack_actions(
            profile,
            profile_path,
            simulated,
            peripherals_file,
            LaunchConfiguration("instance_id").perform(context).strip(),
        ),
        base_node,
        motion,
        *peripherals,
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", description="Runtime profile path or shipped profile name (required)"),
            DeclareLaunchArgument("simulated", default_value="", description="Override profile.simulated"),
            DeclareLaunchArgument("display", default_value="false", description="Launch RViz"),
            DeclareLaunchArgument("instance_id", default_value="", description="Public robot instance identity"),
            DeclareLaunchArgument(
                "peripherals",
                default_value="",
                description="Deployment peripherals fragment {peripherals: [...], fast_lio: {...}} "
                "(overrides profile defaults by (type, name))",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
