"""Launch the cross-platform audio_common device owners."""

from typing import Any

from launch_ros.actions import Node

from robot_config.audio_contract import find_microphone_params, is_audio_io_enabled
from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.audio_io")


def generate_audio_io_actions(robot_config: dict[str, Any]) -> list[Node]:
    config = robot_config.get("audio_io", {})
    if not isinstance(config, dict) or not is_audio_io_enabled(config):
        return []
    capture_enabled = bool(robot_config.get("voice_asr", {}).get("enabled", False)) or bool(
        robot_config.get("speech_direction", {}).get("enabled", False)
    )
    playback_enabled = bool(robot_config.get("voice_tts", {}).get("enabled", False))
    if not capture_enabled and not playback_enabled:
        return []
    microphone_name = str(config.get("microphone", "")).strip()
    params = find_microphone_params(robot_config.get("peripherals", []), microphone_name)
    capture_topic = str(config.get("capture_topic", "/audio/capture"))
    capture_stamped_topic = str(config.get("capture_stamped_topic", "/audio/capture_stamped"))
    audio_info_topic = str(config.get("audio_info_topic", "/audio/info"))
    playback_topic = str(config.get("playback_topic", "/audio/play"))
    capture_device = str(params.get("device", "default"))
    playback_device = str(config.get("playback_device", ""))
    capture_channels = int(params.get("channels", 6))
    capture_sample_rate = int(params.get("sample_rate", 16000))
    capture_sample_format = str(params.get("sample_format", "S16LE"))
    playback_channels = int(config.get("playback_channels", 1))
    playback_sample_rate = int(config.get("playback_sample_rate", 24000))
    playback_sample_format = str(config.get("playback_sample_format", "S16LE"))
    logger.info("Launching shared audio_common capture/playback topics")
    actions = []
    if capture_enabled:
        actions.append(
            Node(
                package="audio_capture",
                executable="audio_capture_node",
                name="audio_capture",
                output="screen",
                parameters=[
                    {
                        "src": "alsasrc",
                        "dst": "appsink",
                        "format": "wave",
                        "device": capture_device,
                        "channels": capture_channels,
                        "sample_rate": capture_sample_rate,
                        "sample_format": capture_sample_format,
                        "depth": 16,
                    }
                ],
                remappings=[
                    ("audio", capture_topic),
                    ("audio_stamped", capture_stamped_topic),
                    ("audio_info", audio_info_topic),
                ],
            )
        )
    if playback_enabled:
        actions.append(
            Node(
                package="audio_play",
                executable="audio_play_node",
                name="audio_play",
                output="screen",
                parameters=[
                    {
                        "dst": "alsasink",
                        "device": playback_device,
                        "format": "wave",
                        "channels": playback_channels,
                        "depth": 16,
                        "sample_rate": playback_sample_rate,
                        "sample_format": playback_sample_format,
                    }
                ],
                remappings=[("audio", playback_topic)],
            )
        )
    return actions
