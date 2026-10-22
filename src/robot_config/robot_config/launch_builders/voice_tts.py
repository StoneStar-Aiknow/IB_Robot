"""Voice TTS launch builder for robot_config."""

import json
import os
from pathlib import Path
from typing import Any

from launch_ros.actions import Node

from robot_config.audio_contract import is_audio_io_enabled
from robot_config.cann_runtime import ascend_node_environment, resolve_cann_root
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


def _ascend_node_environment(bundle_path: str, deployment: str) -> dict[str, str]:
    """CANN paths for an Ascend deployment, chosen from the ABI its OM files were built against.

    The ZipVoice OM files declare ``cann-8.1.RC1``; loading them under a different CANN runtime
    crashes inside ACL (observed on the 310P, 2026-10-16), so the node must not inherit whichever
    toolkit the caller sourced. Non-Ascend deployments get no extra environment.
    """
    from inference_manifest import load_inference_manifest

    try:
        validated = load_inference_manifest(bundle_path, deployment)
    except Exception:  # The node validates the bundle itself and reports a usable error.
        return {}
    profiles = dict(validated.deployment.role_runtime_profiles or {})
    if not profiles and validated.deployment.runtime_profile is not None:
        profiles = {"": validated.deployment.runtime_profile}
    ascend = [profile for profile in profiles.values() if profile.backend == "ascend"]
    if not ascend:
        return {}
    abis = {profile.target.runtime_abi for profile in ascend if profile.target is not None}
    abis.discard(None)
    if len(abis) > 1:
        raise ValueError(f"voice_tts deployment {deployment!r} declares conflicting CANN ABIs: {sorted(abis)}")
    return ascend_node_environment(resolve_cann_root(next(iter(abis), None)))


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
    audio_io = robot_config.get("audio_io", {})
    if not is_audio_io_enabled(audio_io):
        raise ValueError("voice_tts.enabled=true requires audio_io.enabled=true")
    audio_topic = str(audio_io.get("playback_topic", "/audio/play"))
    audio_message_type = "audio_common_msgs/msg/AudioData"
    audio_output = config.get("audio_output")
    if audio_output is not None:
        # Runtime-owned speaker: publish where the runtime description says it listens.
        if not isinstance(audio_output, dict) or not audio_output.get("_interface_source"):
            raise ValueError("voice_tts.audio_output must be bound to a runtime description before launch")
        explicit = audio_io.get("playback_topic")
        if explicit and explicit != audio_output["topic"]:
            raise ValueError("audio_io.playback_topic conflicts with the bound runtime endpoint")
        audio_topic = str(audio_output["topic"])
        audio_message_type = str(audio_output["type"])
    playback_sample_rate = int(audio_io.get("playback_sample_rate", 24000))
    playback_channels = int(audio_io.get("playback_channels", 1))
    logger.info(f"Voice TTS enabled, launching node {node_name!r} with deployment {deployment!r}")
    return [
        Node(
            package="inference_service",
            executable="model_service_node",
            name=node_name,
            output="screen",
            parameters=[node_params],
            additional_env=_ascend_node_environment(bundle_path, deployment),
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
                    "audio_topic": audio_topic,
                    "playback_sample_rate": playback_sample_rate,
                    "playback_channels": playback_channels,
                    "audio_message_type": audio_message_type,
                    "resample_to_playback_rate": bool(config.get("resample_to_playback_rate", False)),
                    "chunk_ms": int(config.get("chunk_ms", 50)),
                }
            ],
        ),
    ]
