"""Launch-builder tests for the voice_asr entry (voice_asr_service's contribution).

Migrated from ``robot_config/test/test_launch_readiness.py`` (the missing-
package robustness cases), ``robot_config/test/test_external_vad_config.py``
(the external-VAD node-parameter cases) and ``robot_config/test/
test_config.py`` (the workspace-root path resolution and the defaults
alignment) — openspec change ``migrate-business-launch-builders``, task
B.4/B.9: the tests follow the voice_asr module to its owning package.
Assertions are verbatim; only the import paths changed.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

import voice_asr_service.launch_builders.voice_asr as voice_asr
from robot_config.config import VoiceASRConfig
from voice_asr_service.defaults import VOICE_ASR_DEFAULTS
from voice_asr_service.launch_builders.voice_asr import (
    generate_voice_asr_nodes,
    resolve_voice_asr_path,
)


def _block_voice_asr_service_import(monkeypatch):
    """Simulate the runtime defaults being unavailable.

    The builder lives in this package now, so an absent package means the
    entry is never registered; the blocked import still pins the module's
    own failure mode — a named, actionable error instead of a raw
    ModuleNotFoundError from the defaults import.
    """
    real_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("voice_asr_service.defaults"):
            raise ModuleNotFoundError(f"No module named '{name}'", name=name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)


def test_voice_asr_builder_skips_missing_package_when_disabled(monkeypatch):
    _block_voice_asr_service_import(monkeypatch)
    nodes = generate_voice_asr_nodes({"voice_asr": {"enabled": False}})

    assert nodes == []


def test_voice_asr_builder_reports_missing_package_when_enabled(monkeypatch):
    _block_voice_asr_service_import(monkeypatch)

    with pytest.raises(
        ModuleNotFoundError,
        match="voice_asr.enabled=true requires the voice_asr_service package to be installed",
    ):
        generate_voice_asr_nodes({"voice_asr": {"enabled": True}})


def test_resolve_voice_asr_path_uses_workspace_root_for_relative_paths():
    """Test voice ASR relative paths resolve from the workspace root."""
    resolved = resolve_voice_asr_path("models/voice_asr/demo-bundle")

    assert Path(resolved).is_absolute()
    assert resolved.endswith("models/voice_asr/demo-bundle")


def test_voice_asr_runtime_defaults_match_robot_config_defaults():
    """Test robot_config's loader defaults stay aligned with the runtime defaults."""
    config_defaults = VoiceASRConfig()

    assert config_defaults.enabled == VOICE_ASR_DEFAULTS["enabled"]
    assert config_defaults.active_mode == VOICE_ASR_DEFAULTS["active_mode"]
    assert config_defaults.language == VOICE_ASR_DEFAULTS["language"]
    assert config_defaults.bundle_path == VOICE_ASR_DEFAULTS["bundle_path"]
    assert config_defaults.deployment == VOICE_ASR_DEFAULTS["deployment"]
    assert config_defaults.max_recording_duration == VOICE_ASR_DEFAULTS["max_recording_duration"]
    assert config_defaults.vad_sensitivity == VOICE_ASR_DEFAULTS["vad_sensitivity"]
    assert config_defaults.vad_bundle_path == VOICE_ASR_DEFAULTS["vad_bundle_path"]
    assert config_defaults.vad_deployment == VOICE_ASR_DEFAULTS["vad_deployment"]
    assert config_defaults.realtime_pre_roll_seconds == VOICE_ASR_DEFAULTS["realtime_pre_roll_seconds"]
    assert config_defaults.final_tail_padding_seconds == VOICE_ASR_DEFAULTS["final_tail_padding_seconds"]
    assert config_defaults.publish_partial == VOICE_ASR_DEFAULTS["publish_partial"]
    assert config_defaults.output_topic == VOICE_ASR_DEFAULTS["output_topic"]
    assert config_defaults.sample_rate == VOICE_ASR_DEFAULTS["sample_rate"]
    assert config_defaults.chunk_size == VOICE_ASR_DEFAULTS["chunk_size"]
    assert config_defaults.buffer_seconds == VOICE_ASR_DEFAULTS["buffer_seconds"]
    assert config_defaults.exit_on_init_failure == VOICE_ASR_DEFAULTS["exit_on_init_failure"]
    assert config_defaults.input_mode == VOICE_ASR_DEFAULTS["input_mode"]
    assert config_defaults.external_min_samples == VOICE_ASR_DEFAULTS["external_min_samples"]


# --- external-VAD node parameters (from robot_config's test_external_vad_config) ---


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


# --- shared stamped audio topics (from robot_config's test_audio_io_launch_builder) ---


def _audio_config(enabled=True):
    """The shared audio_io section the voice builders read (verbatim from
    robot_config's test_audio_io_launch_builder.py)."""
    return {
        "audio_io": {
            "enabled": enabled,
            "microphone": "respeaker",
            "capture_topic": "/shared/capture",
            "capture_stamped_topic": "/shared/capture_stamped",
            "audio_info_topic": "/shared/info",
            "playback_topic": "/shared/play",
            "playback_device": "plughw:1,0",
            "playback_sample_rate": 24000,
            "playback_channels": 1,
            "playback_sample_format": "S16LE",
        },
        "voice_asr": {"enabled": True},
        "voice_tts": {"enabled": True},
        "peripherals": [
            {
                "name": "respeaker",
                "type": "microphone",
                "params": {
                    "device": "hw:2,0",
                    "channels": 6,
                    "sample_rate": 16000,
                    "sample_format": "S16LE",
                },
            }
        ],
    }


import voice_asr_service.launch_builders.speech_direction as speech_direction  # noqa: E402


def test_speech_direction_uses_shared_stamped_topic(monkeypatch, tmp_path):
    launch_file = tmp_path / "launch" / "speech_direction.launch.py"
    launch_file.parent.mkdir()
    launch_file.touch()
    monkeypatch.setattr(speech_direction, "_voice_asr_service_share", lambda: Path(tmp_path))
    config = _audio_config()
    config["speech_direction"] = {
        "enabled": True,
        "profile": "ubuntu",
        "microphone": "respeaker",
        "parameters": {},
    }

    actions = speech_direction.generate_speech_direction_actions(config)

    arguments = dict(actions[0]._IncludeLaunchDescription__launch_arguments)
    assert arguments["speech_direction_audio_topic"] == "/shared/capture_stamped"
    assert arguments["microphone_channels"] == "6"


def test_voice_asr_uses_shared_stamped_topic_and_microphone_channels(monkeypatch, tmp_path):
    monkeypatch.setattr(
        voice_asr,
        "_load_voice_asr_service",
        lambda: {
            "bundle_path": "models/voice_asr/demo",
            "deployment": "torch_cpu",
            "language": "zh",
            "max_recording_duration": 10.0,
            "vad_sensitivity": 0.6,
            "vad_bundle_path": "models/silero-vad",
            "vad_deployment": "torch_cpu",
            "realtime_pre_roll_seconds": 0.5,
            "final_tail_padding_seconds": 1.5,
            "publish_partial": True,
            "output_topic": "/voice_command",
            "sample_rate": 16000,
            "chunk_size": 512,
            "buffer_seconds": 5.0,
            "exit_on_init_failure": False,
            "input_mode": "shared_audio",
            "external_min_samples": 3200,
        },
    )
    monkeypatch.setattr(voice_asr, "validate_voice_asr_model_config", lambda **_kwargs: [])
    config = _audio_config()
    config["voice_asr"] = {
        "enabled": True,
        "bundle_path": str(tmp_path),
        "deployment": "torch_cpu",
        "active_mode": "manual",
        "vad_input_channel": 0,
    }

    node = voice_asr.generate_voice_asr_nodes(config)[0]
    parameters = {}
    for raw_key, raw_value in node._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if isinstance(raw_value, tuple):
            raw_value = "".join(getattr(item, "text", str(item)) for item in raw_value).strip()
            if raw_value.endswith("\n..."):
                raw_value = raw_value[:-4].rstrip()
        parameters[key] = raw_value

    assert parameters["audio_topic"] == "/shared/capture_stamped"
    assert parameters["audio_channels"] == 6
    assert parameters["vad_input_channel"] == 0
