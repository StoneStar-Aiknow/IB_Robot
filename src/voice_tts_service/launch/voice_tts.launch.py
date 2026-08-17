#!/usr/bin/env python3
"""Standalone launch entry for debugging the shared ZipVoice model service."""

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
            DeclareLaunchArgument("playback_service_name", default_value="/voice_tts/play"),
            DeclareLaunchArgument("playback_timeout_sec", default_value="300.0"),
            LogInfo(
                msg=(
                    "[voice_tts_service] Standalone launch is for debugging only. "
                    "Use robot_config/robot.launch.py and robot.voice_tts for the system entry."
                )
            ),
            Node(
                package="inference_service",
                executable="model_service_node",
                name="model_service_voice_tts",
                output="screen",
                parameters=[
                    {
                        "bundle_path": LaunchConfiguration("bundle_path"),
                        "deployment": LaunchConfiguration("deployment"),
                        "instance_id": "voice_tts",
                        "adapter_class": "voice_tts_service.model_service_plugin:ZipVoiceSynthesizePlugin",
                        "service_type": "ibrobot_msgs/srv/SynthesizeSpeech",
                        "service_endpoint": LaunchConfiguration("service_name"),
                        "runtime_options_json": "{}",
                    }
                ],
            ),
            Node(
                package="voice_tts_service",
                executable="audio_playback_node",
                name="voice_tts_audio_player",
                output="screen",
                parameters=[
                    {
                        "service_name": LaunchConfiguration("playback_service_name"),
                        "timeout_sec": LaunchConfiguration("playback_timeout_sec"),
                    }
                ],
            ),
        ]
    )
