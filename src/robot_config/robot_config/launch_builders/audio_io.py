"""Launch the cross-platform audio_common device owners (legacy provider-less path).

``runtime-owned-audio-io``: with ``runtime.provider`` set the robot runtime owns
the audio devices (its peripheral list launches the same ALSA nodes), so this
entry steps aside exactly like the perception entry does. The node generation
itself moved to ``robot_runtime.peripherals``; this legacy path calls that
single source, so defaults, node names, parameters and topics cannot drift
between the two paths.
"""

from typing import Any

from launch_ros.actions import Node

from robot_config.audio_contract import find_microphone_params, is_audio_io_enabled
from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext
from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.audio_io")


def generate_audio_io_actions(robot_config: dict[str, Any]) -> list[Node]:
    config = robot_config.get("audio_io", {})
    if not isinstance(config, dict) or not is_audio_io_enabled(config):
        return []
    provider = str((robot_config.get("runtime") or {}).get("provider", "") or "").strip()
    if provider:
        logger.info(
            f"runtime.provider={provider}: audio devices are runtime-owned (peripherals), "
            "skipping the robot_config audio_io entry"
        )
        return []
    capture_enabled = bool(robot_config.get("voice_asr", {}).get("enabled", False)) or bool(
        robot_config.get("speech_direction", {}).get("enabled", False)
    )
    playback_enabled = bool(robot_config.get("voice_tts", {}).get("enabled", False))
    if not capture_enabled and not playback_enabled:
        return []
    from robot_runtime.peripherals import audio_capture_node, audio_play_node

    microphone_name = str(config.get("microphone", "")).strip()
    params = find_microphone_params(robot_config.get("peripherals", []), microphone_name)
    actions = []
    if capture_enabled:
        actions.append(
            audio_capture_node(
                params,
                capture_topic=str(config.get("capture_topic", "/audio/capture")),
                capture_stamped_topic=str(config.get("capture_stamped_topic", "/audio/capture_stamped")),
                audio_info_topic=str(config.get("audio_info_topic", "/audio/info")),
            )
        )
    if playback_enabled:
        actions.append(
            audio_play_node(
                {
                    "device": str(config.get("playback_device", "")),
                    "channels": int(config.get("playback_channels", 1)),
                    "sample_rate": int(config.get("playback_sample_rate", 24000)),
                    "sample_format": str(config.get("playback_sample_format", "S16LE")),
                },
                playback_topic=str(config.get("playback_topic", "/audio/play")),
            )
        )
    if actions:
        logger.info("Launching shared audio_common capture/playback topics")
    return actions


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``):
# the BuilderSpec constant and build function moved here from the retired
# ``adapters.py``; the body is a verbatim migration.
# ---------------------------------------------------------------------------


def build_audio_io(context: LaunchBuildContext) -> BuildResult:
    from robot_config.launch_builders.audio_io import generate_audio_io_actions

    return BuildResult(actions=list(generate_audio_io_actions(context.robot_config)))


AUDIO_IO_BUILDER = BuilderSpec(
    id="audio_io",
    factory=f"{__name__}:build_audio_io",
)
