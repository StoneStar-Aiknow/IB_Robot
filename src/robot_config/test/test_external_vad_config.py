"""Configuration-level tests for external_vad voice ASR and runtime audio source.

Covers the loader fields, the validation rules (input_mode values, the
external_vad/requires-runtime-source pairing, runtime-source exemption from
microphone peripherals) and the launch builders (no ALSA capture node; node
parameter passthrough for the event-driven input).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from robot_config.config import RobotConfig, Ros2ControlConfig  # noqa: E402
from robot_config.launch_builders import audio_io, voice_asr  # noqa: E402
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


def _bound_audio_input(topic="/audio/speech_chunk"):
    return {
        "interface": "audio.speech",
        "type": "ibrobot_msgs/msg/SpeechAudioChunk",
        "topic": topic,
        "_interface_source": {"id": "audio.speech"},
    }


def _parameters(node):
    parameters = {}
    for raw_key, raw_value in node._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if isinstance(raw_value, tuple):
            raw_value = "".join(getattr(item, "text", str(item)) for item in raw_value).strip()
            if raw_value.endswith("\n..."):
                raw_value = raw_value[:-4].rstrip()
        parameters[key] = raw_value
    return parameters


def test_external_vad_builder_accepts_an_explicit_standalone_topic(monkeypatch, tmp_path):
    # A node deployed without a runtime description may still name its chunk
    # topic explicitly; that is the only input this mode has.
    monkeypatch.setattr(voice_asr, "validate_voice_asr_model_config", lambda **_kwargs: [])
    config = _robot_config_dict(voice_asr_overrides={"bundle_path": str(tmp_path)})
    parameters = _parameters(voice_asr.generate_voice_asr_nodes(config)[0])
    assert parameters["chunk_topic"] == "/audio/speech_chunk"


def test_external_vad_builder_rejects_an_unbound_audio_input(monkeypatch, tmp_path):
    # Declaring the logical input but launching before the runtime resolved it
    # must fail here rather than subscribe to a guessed endpoint.
    monkeypatch.setattr(voice_asr, "validate_voice_asr_model_config", lambda **_kwargs: [])
    config = _robot_config_dict(
        voice_asr_overrides={
            "bundle_path": str(tmp_path),
            "audio_input": {"interface": "audio.speech", "type": "ibrobot_msgs/msg/SpeechAudioChunk"},
        }
    )
    with pytest.raises(ValueError, match="must be bound to a runtime description"):
        voice_asr.generate_voice_asr_nodes(config)


def test_external_vad_builder_rejects_a_topic_conflicting_with_the_binding(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_asr, "validate_voice_asr_model_config", lambda **_kwargs: [])
    config = _robot_config_dict(
        voice_asr_overrides={
            "bundle_path": str(tmp_path),
            "audio_input": _bound_audio_input(),
            "chunk_topic": "/audio/other_chunk",
        }
    )
    with pytest.raises(ValueError, match="conflicts with the bound runtime endpoint"):
        voice_asr.generate_voice_asr_nodes(config)


def test_voice_asr_builder_passes_external_vad_parameters(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_asr, "validate_voice_asr_model_config", lambda **_kwargs: [])
    config = _robot_config_dict(voice_asr_overrides={"bundle_path": str(tmp_path), "audio_input": _bound_audio_input()})

    node = voice_asr.generate_voice_asr_nodes(config)[0]
    parameters = {}
    for raw_key, raw_value in node._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if isinstance(raw_value, tuple):
            raw_value = "".join(getattr(item, "text", str(item)) for item in raw_value).strip()
            if raw_value.endswith("\n..."):
                raw_value = raw_value[:-4].rstrip()
        parameters[key] = raw_value

    assert parameters["input_mode"] == "external_vad"
    assert parameters["chunk_topic"] == "/audio/speech_chunk"
    # 外部分段输入不装配本地 Silero VAD：VAD 参数不得进入节点配置。
    assert "vad_bundle_path" not in parameters
    assert "vad_deployment" not in parameters
    # 单条原子消息就是全部输入：不向节点注入任何音频/分段/回填参数。
    for retired in ("segment_topic", "external_backfill_seconds", "audio_topic"):
        assert retired not in parameters, retired
