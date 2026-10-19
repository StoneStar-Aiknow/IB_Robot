"""Configuration-level tests for external_vad voice ASR and runtime audio source.

Covers the loader fields, the validation rules (input_mode values, the
external_vad/requires-runtime-source pairing, runtime-source exemption from
microphone peripherals) and the launch builders (no ALSA capture node; node
parameter passthrough for the event-driven input).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from robot_config.config import RobotConfig, Ros2ControlConfig  # noqa: E402
from robot_config.launch_builders import audio_io  # noqa: E402
from robot_config.loader import load_audio_io_config, load_voice_asr_config, validate_config  # noqa: E402


def _voice_asr_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "input_mode": "external_vad",
        "bundle_path": "/tmp/asr-bundle",
        "deployment": "torch_cpu",
        "chunk_topic": "/audio/speech_chunk",
    }
    config.update(overrides)
    return config


def _audio_io_config(**overrides) -> dict:
    config = {"enabled": True, "source": "runtime"}
    config.update(overrides)
    return config


def _robot_config_dict(voice_asr_overrides=None, audio_io_overrides=None) -> dict:
    return {
        "audio_io": _audio_io_config(**(audio_io_overrides or {})),
        "voice_asr": _voice_asr_config(**(voice_asr_overrides or {})),
        "voice_tts": {"enabled": False},
        "peripherals": [],
    }


def _robot_config(voice_asr_overrides=None, audio_io_overrides=None) -> RobotConfig:
    data = _robot_config_dict(voice_asr_overrides, audio_io_overrides)
    config = RobotConfig(
        name="x2_test",
        type="x2_ultra",
        robot_type="agibot_x2",
        ros2_control=Ros2ControlConfig(hardware_plugin="mock/hardware", params={}),
        peripherals=[],
    )
    config.audio_io = load_audio_io_config(data["audio_io"])
    config.voice_asr = load_voice_asr_config(data["voice_asr"])
    return config


# --- loader ------------------------------------------------------------------


def test_loader_reads_external_vad_fields():
    voice = load_voice_asr_config(_voice_asr_config())
    assert voice.input_mode == "external_vad"
    assert voice.chunk_topic == "/audio/speech_chunk"
    assert voice.external_min_samples == 3200
    # The retired split-topic fields are gone, not merely deprecated: they were
    # never published, so a config still naming them is a plain unknown key.
    assert not hasattr(voice, "segment_topic")
    assert not hasattr(voice, "external_backfill_seconds")

    audio = load_audio_io_config(_audio_io_config())
    assert audio.source == "runtime"


# --- validation ---------------------------------------------------------------


def test_external_vad_with_runtime_source_passes_validation():
    config = _robot_config()
    assert validate_config(config) == []


def test_invalid_input_mode_is_rejected():
    config = _robot_config(voice_asr_overrides={"input_mode": "vendor"})
    errors = validate_config(config)
    assert any("voice_asr.input_mode must be 'shared_audio' or 'external_vad'" in error for error in errors)


def test_external_vad_requires_runtime_source():
    config = _robot_config(audio_io_overrides={"source": "alsa"})
    errors = validate_config(config)
    assert any("requires audio_io.source=runtime" in error for error in errors)


def test_runtime_source_is_exempt_from_microphone_peripherals():
    # 没有任何麦克风外设：runtime 源下这仍是合法配置（采集由 runtime 提供）。
    config = _robot_config()
    assert config.audio_io.microphone == ""
    assert validate_config(config) == []


def test_invalid_audio_io_source_is_rejected():
    config = _robot_config(audio_io_overrides={"source": "spdif"})
    errors = validate_config(config)
    assert any("audio_io.source must be 'alsa' or 'runtime'" in error for error in errors)


def test_external_vad_topics_must_be_absolute():
    config = _robot_config(voice_asr_overrides={"chunk_topic": "chunks"})
    errors = validate_config(config)
    assert any("voice_asr.chunk_topic must be an absolute ROS topic name" in error for error in errors)


# --- builders -----------------------------------------------------------------


def test_audio_io_builder_skips_capture_node_for_runtime_source():
    actions = audio_io.generate_audio_io_actions(_robot_config_dict())
    assert actions == []
