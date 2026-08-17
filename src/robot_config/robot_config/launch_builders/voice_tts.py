"""Voice TTS launch builder for robot_config."""

import json
import os
from pathlib import Path
from typing import Any

from launch_ros.actions import Node

from robot_config.logger_utils import get_colored_logger
from robot_config.utils import resolve_ros_path

logger = get_colored_logger("robot_config.voice_tts")
_VOICE_TTS_MISSING_ERROR = "voice_tts.enabled=true requires the voice_tts_service package to be installed"


def _load_voice_tts_service():
    try:
        from voice_tts_service.defaults import VOICE_TTS_DEFAULTS
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("voice_tts_service"):
            raise ModuleNotFoundError(_VOICE_TTS_MISSING_ERROR) from exc
        raise
    return VOICE_TTS_DEFAULTS


def resolve_voice_tts_path(path: str) -> str:
    """Resolve a bundle path; relative paths use the explicit WORKSPACE root."""

    resolved = resolve_ros_path(path)
    if not resolved:
        return resolved
    candidate = Path(resolved).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    workspace_value = os.environ.get("WORKSPACE")
    if not workspace_value:
        raise ValueError("voice_tts.bundle_path is relative but WORKSPACE is unset")
    workspace = Path(workspace_value).expanduser()
    if not workspace.is_absolute():
        raise ValueError("WORKSPACE must be absolute for a relative voice_tts.bundle_path")
    return str((workspace / candidate).resolve())


def generate_voice_tts_nodes(robot_config: dict[str, Any]) -> list[Node]:
    """Generate the ZipVoice model host and local audio playback service."""

    config = robot_config.get("voice_tts", {})
    if not config.get("enabled", False):
        logger.info("Voice TTS disabled, skipping")
        return []

    defaults = _load_voice_tts_service()
    bundle_path = resolve_voice_tts_path(str(config.get("bundle_path", "")))
    deployment = str(config.get("deployment", ""))
    if not bundle_path or not deployment:
        raise ValueError("voice_tts.bundle_path and voice_tts.deployment are required when enabled")
    runtime_options = {
        name: config.get(name, default)
        for name, default in defaults.items()
        if name
        in {
            "device_id",
            "prompt_profile",
            "segment_max_chars",
            "segment_pause_ms",
            "max_request_chars",
            "max_prompt_audio_bytes",
            "max_prompt_duration_sec",
            "max_segments",
            "max_response_audio_bytes",
        }
    }
    node_params = {
        "instance_id": str(config.get("instance_id", "voice_tts")),
        "bundle_path": bundle_path,
        "deployment": deployment,
        "adapter_class": "voice_tts_service.model_service_plugin:ZipVoiceSynthesizePlugin",
        "service_type": "ibrobot_msgs/srv/SynthesizeSpeech",
        "service_endpoint": str(config.get("service_name", defaults.get("service_name", "/voice_tts/synthesize"))),
        "required": bool(config.get("exit_on_init_failure", defaults.get("exit_on_init_failure", True))),
        "runtime_options_json": json.dumps(runtime_options, sort_keys=True),
    }
    node_name = str(config.get("node_name", "model_service_voice_tts"))
    logger.info(f"Voice TTS enabled, launching node {node_name!r} with deployment {deployment!r}")
    return [
        Node(
            package="inference_service",
            executable="model_service_node",
            name=node_name,
            output="screen",
            parameters=[node_params],
        ),
        Node(
            package="voice_tts_service",
            executable="audio_playback_node",
            name=str(config.get("playback_node_name", "voice_tts_audio_player")),
            output="screen",
            parameters=[
                {
                    "service_name": str(
                        config.get(
                            "playback_service_name",
                            defaults.get("playback_service_name", "/voice_tts/play"),
                        )
                    ),
                    "timeout_sec": float(
                        config.get("playback_timeout_sec", defaults.get("playback_timeout_sec", 300.0))
                    ),
                }
            ],
        ),
    ]
