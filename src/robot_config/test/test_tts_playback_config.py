"""TTS playback cases taken from the voice group's test_external_vad_config.py; its ASR cases were not taken."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from robot_config.config import RobotConfig, Ros2ControlConfig  # noqa: E402
from robot_config.launch_builders import audio_io  # noqa: E402
from robot_config.loader import load_audio_io_config, validate_config  # noqa: E402


def _audio_io_config(**overrides) -> dict:
    config = {"enabled": True}
    config.update(overrides)
    return config


def _robot_config_dict(audio_io_overrides=None) -> dict:
    return {
        "audio_io": _audio_io_config(**(audio_io_overrides or {})),
        "voice_tts": {"enabled": False},
        "peripherals": [],
    }


def _robot_config(audio_io_overrides=None) -> RobotConfig:
    data = _robot_config_dict(audio_io_overrides)
    config = RobotConfig(
        name="x2_test",
        type="x2_ultra",
        robot_type="agibot_x2",
        ros2_control=Ros2ControlConfig(hardware_plugin="mock/hardware", params={}),
        peripherals=[],
    )
    config.audio_io = load_audio_io_config(data["audio_io"])
    return config


# --- validation ---------------------------------------------------------------


def test_invalid_audio_io_playback_source_is_rejected():
    config = _robot_config(audio_io_overrides={"playback_source": "spdif"})
    errors = validate_config(config)
    assert any("audio_io.playback_source must be 'alsa' or 'runtime'" in error for error in errors)


# --- builders -----------------------------------------------------------------


def test_audio_io_builder_skips_playback_node_for_the_runtime_playback_axis():
    # The speaker belongs to the runtime (audio.play): no local ALSA sink may
    # subscribe to the playback topic alongside it.
    config = _robot_config_dict(audio_io_overrides={"playback_source": "runtime"})
    config["voice_tts"] = {"enabled": True, "audio_output": {"interface": "audio.play"}}
    assert audio_io.generate_audio_io_actions(config) == []


def test_audio_io_builder_keeps_local_playback_for_the_alsa_playback_axis():
    config = _robot_config_dict(audio_io_overrides={"microphone": ""})
    config["voice_tts"] = {"enabled": True}
    actions = audio_io.generate_audio_io_actions(config)
    assert [action._Node__node_name for action in actions] == ["audio_play"]


def test_runtime_playback_without_a_bound_sink_is_rejected():
    config = _robot_config(audio_io_overrides={"playback_source": "runtime"})
    config.voice_tts.enabled = True
    errors = validate_config(config)
    assert any("playback_source=runtime requires voice_tts.audio_output" in error for error in errors)


def test_bound_speaker_with_an_alsa_playback_axis_is_rejected():
    config = _robot_config()
    config.voice_tts.enabled = True
    config.voice_tts.audio_output = {"interface": "audio.play", "type": "audio_common_msgs/msg/AudioDataStamped"}
    errors = validate_config(config)
    assert any("audio_io.playback_source must be 'runtime'" in error for error in errors)


def test_runtime_bound_tts_playback_is_valid():
    config = _robot_config(audio_io_overrides={"playback_source": "runtime"})
    config.voice_tts.enabled = True
    config.voice_tts.deployment = "ascend_310p"
    config.voice_tts.audio_output = {"interface": "audio.play", "type": "audio_common_msgs/msg/AudioDataStamped"}
    config.voice_tts.resample_to_playback_rate = True
    assert validate_config(config) == []


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("audio_output", {"interface": "audio.play", "type": "std_msgs/msg/String"}, "audio_output.type must be"),
        ("audio_output", {"interface": "audio.play"}, "must be a mapping with interface and type"),
        ("chunk_ms", 5, "chunk_ms must be an integer between 20 and 200"),
    ],
)
def test_invalid_tts_playback_settings_are_rejected(field, value, error):
    config = _robot_config(audio_io_overrides={"playback_source": "runtime"})
    config.voice_tts.enabled = True
    config.voice_tts.audio_output = {"interface": "audio.play", "type": "audio_common_msgs/msg/AudioDataStamped"}
    setattr(config.voice_tts, field, value)
    assert any(error in item for item in validate_config(config))
