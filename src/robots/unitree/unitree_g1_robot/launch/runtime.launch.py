"""Unitree G1 runtime launch entry (robot-runtime-packaging spec).

Brings the G1 up from a runtime profile alone: one bridge process that serves
the public runtime contract over the vendor DDS tier, plus — in simulated
transport — the in-repo vendor mock that stands in for the robot's own stack.

    ros2 launch unitree_g1_robot runtime.launch.py profile:=<path-or-name> \
        [simulated:=true] [instance_id:=<id>] [initial_mode:=<mode>]

No controller manager, no ros2_control hardware component, no
robot_state_publisher and no sensor driver is launched: the vendor owns
balance, whole-body control and the low-level tier (design D1/D11), and the
composed peripheral drivers arrive with the phase-3 bring-up. The interface
description is projected here from the profile alone — the URDF is
deliberately not passed, so no public model is published (D6) — and handed to
the runtime, which serves it verbatim in RuntimeStatus.
"""

from __future__ import annotations

import json

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from robot_runtime.interface_description import build_description
from robot_runtime.launch_support import resolve_profile_argument, resolve_simulated
from robot_runtime.peripherals import synthetic_perception_nodes
from robot_runtime.profile import load_profile


def _launch_setup(context, *_args, **_kwargs):
    profile_path = resolve_profile_argument("unitree_g1_robot", LaunchConfiguration("profile").perform(context))
    profile = load_profile(profile_path)
    simulated = resolve_simulated(profile, LaunchConfiguration("simulated").perform(context))
    instance_id = LaunchConfiguration("instance_id").perform(context).strip()
    initial_mode = LaunchConfiguration("initial_mode").perform(context).strip()

    if instance_id:
        profile["runtime"]["instance_id"] = instance_id
    description = build_description(profile, simulated=simulated)
    runtime = Node(
        package="unitree_g1_robot",
        executable="runtime_node",
        name="unitree_g1_runtime",
        output="screen",
        parameters=[
            {
                "profile": str(profile_path),
                "initial_mode": initial_mode,
                "interface_description_json": json.dumps(description),
            }
        ],
    )
    if not simulated:
        return [runtime]

    # Simulated transport: the in-repo vendor mock stands in for the robot's
    # own stack on the vendor DDS tier — the runtime's D10 startup wait
    # tolerates the discovery race, so the nodes need no stagger — and the
    # contract layer's synthetic publishers stand in for the composed
    # peripheral drivers (D8) on exactly the declared camera and lidar
    # endpoints. The bridged IMU needs no stand-in: the runtime republishes
    # /imu/data from the (mock) LowState feed, so the synthetic set covers the
    # peripherals only.
    mock = Node(
        package="unitree_g1_robot",
        executable="vendor_mock_node",
        name="unitree_vendor_mock",
        output="screen",
        parameters=[{"profile": str(profile_path)}],
    )
    return [mock, runtime, *synthetic_perception_nodes(profile.get("peripherals"))]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("profile", description="Runtime profile path or shipped profile name (required)"),
            DeclareLaunchArgument(
                "simulated",
                default_value="",
                description="Override profile.simulated: run against the in-repo vendor mock",
            ),
            DeclareLaunchArgument("instance_id", default_value="", description="Public robot instance identity"),
            DeclareLaunchArgument(
                "initial_mode",
                default_value="",
                description="Initial runtime mode override (the G1 boots in idle only)",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
