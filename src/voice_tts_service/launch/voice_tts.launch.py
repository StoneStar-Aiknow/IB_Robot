#!/usr/bin/env python3
"""Standalone launch entry for debugging the Voice TTS node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("bundle_path", default_value=""),
            DeclareLaunchArgument("deployment", default_value=""),
            DeclareLaunchArgument("service_name", default_value="/voice_tts/synthesize"),
            LogInfo(
                msg=(
                    "[voice_tts_service] Standalone launch is for debugging only. "
                    "Use robot_config/robot.launch.py and robot.voice_tts for the system entry."
                )
            ),
            Node(
                package="voice_tts_service",
                executable="voice_tts_node",
                name="voice_tts_node",
                output="screen",
                parameters=[
                    {
                        "bundle_path": LaunchConfiguration("bundle_path"),
                        "deployment": LaunchConfiguration("deployment"),
                        "service_name": LaunchConfiguration("service_name"),
                    }
                ],
            ),
        ]
    )
