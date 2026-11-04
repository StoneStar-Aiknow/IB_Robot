#!/usr/bin/env python3
"""Standalone launch entry for debugging the shared ZipVoice model service."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("bundle_path", default_value=""),
            DeclareLaunchArgument("deployment", default_value=""),
            DeclareLaunchArgument("service_name", default_value="/voice_tts/synthesize"),
            DeclareLaunchArgument("playback_service_name", default_value="/voice_tts/play"),
            DeclareLaunchArgument("playback_timeout_sec", default_value="300.0"),
            DeclareLaunchArgument("audio_topic", default_value="/audio/play"),
            DeclareLaunchArgument("playback_sample_rate", default_value="24000"),
            DeclareLaunchArgument("playback_channels", default_value="1"),
            DeclareLaunchArgument("resample_to_playback_rate", default_value="false"),
            DeclareLaunchArgument("audio_message_type", default_value="audio_common_msgs/msg/AudioData"),
            DeclareLaunchArgument("chunk_ms", default_value="50"),
            # JSON object of adapter runtime options, e.g. '{"num_steps": 4}'.
            DeclareLaunchArgument("runtime_options_json", default_value="{}"),
            DeclareLaunchArgument("prebuffer_ms", default_value="0"),
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
                        "runtime_options_json": ParameterValue(
                            LaunchConfiguration("runtime_options_json"), value_type=str
                        ),
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
                        "audio_topic": LaunchConfiguration("audio_topic"),
                        "playback_sample_rate": LaunchConfiguration("playback_sample_rate"),
                        "playback_channels": LaunchConfiguration("playback_channels"),
                        "resample_to_playback_rate": LaunchConfiguration("resample_to_playback_rate"),
                        "audio_message_type": LaunchConfiguration("audio_message_type"),
                        "chunk_ms": LaunchConfiguration("chunk_ms"),
                        "prebuffer_ms": LaunchConfiguration("prebuffer_ms"),
                    }
                ],
            ),
        ]
    )
