"""Configuration-level tests for external_vad voice ASR and runtime-owned audio.

Covers the loader fields, the validation rules (input_mode values, the
external_vad/requires-audio_input-binding pairing, the removed
``audio_io.source`` key, provider deployments rejecting an ``audio_io``
section) and the launch builders (no ALSA nodes under a runtime provider;
node parameter passthrough for the event-driven input).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

import pytest  # noqa: E402

from robot_config.config import RobotConfig, Ros2ControlConfig  # noqa: E402
from robot_config.launch_builders import audio_io  # noqa: E402
from robot_config.loader import (  # noqa: E402
    load_audio_io_config,
    load_voice_asr_config,
    validate_config,
    validate_runtime_owned_audio_config,
)

_AUDIO_INPUT_BINDING = {
    "interface": "audio.speech",
    "type": "ibrobot_msgs/msg/SpeechAudioChunk",
    "_interface_source": {"id": "audio.speech", "state": "unknown"},
    "topic": "/audio/speech_chunk",
}


def _voice_asr_config(**overrides) -> dict:
    config = {
        "enabled": True,
        "input_mode": "external_vad",
        "bundle_path": "/tmp/asr-bundle",
        "deployment": "torch_cpu",
        "chunk_topic": "/audio/speech_chunk",
        "audio_input": dict(_AUDIO_INPUT_BINDING),
    }
    config.update(overrides)
    return config


def _robot_config(voice_asr_overrides=None, audio_io_overrides=None) -> RobotConfig:
    data = {
        "audio_io": {"enabled": True, **(audio_io_overrides or {})},
        "voice_asr": _voice_asr_config(**(voice_asr_overrides or {})),
        "voice_tts": {"enabled": False},
        "peripherals": [],
    }
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


def test_loader_rejects_the_removed_audio_io_source_key():
    with pytest.raises(ValueError, match="audio_io.source was removed"):
        load_audio_io_config({"enabled": True, "source": "runtime"})


# --- validation ---------------------------------------------------------------


def test_external_vad_with_audio_input_binding_passes_validation():
    assert validate_config(_robot_config()) == []


def test_invalid_input_mode_is_rejected():
    config = _robot_config(voice_asr_overrides={"input_mode": "vendor"})
    errors = validate_config(config)
    assert any("voice_asr.input_mode must be 'shared_audio' or 'external_vad'" in error for error in errors)


def test_external_vad_requires_the_audio_input_binding():
    config = _robot_config(voice_asr_overrides={"audio_input": None})
    errors = validate_config(config)
    assert any(
        "voice_asr.input_mode=external_vad requires the voice_asr.audio_input interface binding" in error
        for error in errors
    )


def test_external_vad_topics_must_be_absolute():
    config = _robot_config(voice_asr_overrides={"chunk_topic": "chunks"})
    errors = validate_config(config)
    assert any("voice_asr.chunk_topic must be an absolute ROS topic name" in error for error in errors)


def test_streaming_capture_still_requires_the_microphone_peripheral():
    config = _robot_config(voice_asr_overrides={"input_mode": "shared_audio", "audio_input": None})
    errors = validate_config(config)
    assert any("audio_io.microphone is required when shared audio capture is enabled" in error for error in errors)


# --- provider-owned audio validation (dict level) ------------------------------


def test_provider_config_with_audio_io_section_is_rejected():
    errors = validate_runtime_owned_audio_config(
        {"runtime": {"provider": "aimdk_robot"}, "audio_io": {"enabled": False}}
    )
    assert any("audio_io section is not allowed with runtime.provider=aimdk_robot" in error for error in errors)


def test_provider_config_audio_consumers_require_bindings():
    config = {
        "runtime": {"provider": "lekiwi_robot"},
        "speech_direction": {"enabled": True},
        "voice_tts": {"enabled": True},
        "voice_asr": {"enabled": True, "input_mode": "shared_audio"},
    }
    errors = validate_runtime_owned_audio_config(config)
    assert any("speech_direction.audio_input" in error for error in errors)
    assert any("voice_tts.audio_output" in error for error in errors)
    assert any("voice_asr.audio_input" in error for error in errors)


def test_providerless_configs_are_not_checked_for_runtime_owned_audio():
    assert validate_runtime_owned_audio_config({"audio_io": {"enabled": True}}) == []


# --- builders -----------------------------------------------------------------


def test_audio_io_builder_steps_aside_under_a_runtime_provider():
    robot_config = {
        "runtime": {"provider": "aimdk_robot"},
        "audio_io": {"enabled": True},
        "voice_asr": _voice_asr_config(),
        "peripherals": [],
    }
    assert audio_io.generate_audio_io_actions(robot_config) == []


# --- bound speech direction (runtime-owned-audio-io) ----------------------------


def _bound_speech_direction_robot(*, audio_input) -> dict:
    return {
        "name": "bound_direction",
        "type": "lekiwi",
        "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp", "target": "simulation"},
        "peripherals": [],  # require_model-style deployment: the microphone lives in the runtime profile
        "speech_direction": {
            "enabled": True,
            "profile": "ascend_310p",
            "audio_input": audio_input,
            "config_file": "$(find voice_asr_service)/config/speech_direction.yaml",
            "profiles_file": "$(find voice_asr_service)/config/speech_direction_profiles.yaml",
            "models_root": "$(env WORKSPACE)/models",
        },
    }


_DECLARED_CAPTURE_BINDING = {"interface": "audio.capture", "type": "audio_common_msgs/msg/AudioDataStamped"}
_BOUND_CAPTURE_BINDING = {
    **_DECLARED_CAPTURE_BINDING,
    "_interface_source": {"id": "audio.capture", "state": "unknown"},
    "topic": "/audio/capture_stamped",
    "sample_rate_hz": 16000,
    "channels": 6,
    "sample_format": "S16LE",
    "channel_indices": [1, 2, 3, 4],
}


def test_bound_speech_direction_loads_and_validates_through_the_canonical_loader(tmp_path):
    """A microphone owned by the runtime profile needs no deployment-side entry.

    The declaration stage (defer_interface_binding) and the bound stage (the
    resolved topic plus the runtime's array format) both pass the canonical
    loader; the legacy microphone requirement no longer runs ahead of the
    builder for a bound section.
    """
    import yaml

    from robot_config.loader import load_robot_config_dict, validate_robot_config_dict

    config_path = tmp_path / "bound_direction.yaml"
    config_path.write_text(
        yaml.safe_dump({"robot": _bound_speech_direction_robot(audio_input=dict(_DECLARED_CAPTURE_BINDING))}),
        encoding="utf-8",
    )
    config = load_robot_config_dict(config_path, defer_interface_binding=True)
    validate_robot_config_dict(config, deferred_interfaces=True)

    config["speech_direction"]["audio_input"] = dict(_BOUND_CAPTURE_BINDING)
    validate_robot_config_dict(config, deferred_interfaces=True)
    assert config["speech_direction"]["audio_input"]["topic"] == "/audio/capture_stamped"


def test_legacy_speech_direction_still_requires_the_deployment_microphone():
    from robot_config.loader import validate_robot_config_dict

    config = _bound_speech_direction_robot(audio_input=None)
    with pytest.raises(ValueError) as excinfo:
        validate_robot_config_dict(config, deferred_interfaces=True)
    assert "speech_direction.microphone must be a non-empty string when enabled" in str(excinfo.value)


@pytest.mark.parametrize(
    ("audio_input", "error"),
    [
        ({"interface": "", "type": "audio_common_msgs/msg/AudioDataStamped"}, "audio_input.interface"),
        ({"interface": "audio.capture", "type": ""}, "audio_input.type"),
        (
            {"interface": "audio.capture", "type": "ibrobot_msgs/msg/SpeechAudioChunk"},
            "audio_input.type must be audio_common_msgs/msg/AudioDataStamped",
        ),
        (
            {**_BOUND_CAPTURE_BINDING, "topic": ""},
            "audio_input.topic must be a non-empty string once bound",
        ),
        ({**_BOUND_CAPTURE_BINDING, "channels": 0}, "audio_input.channels must be a positive integer"),
        ({**_BOUND_CAPTURE_BINDING, "sample_rate_hz": 0}, "audio_input.sample_rate_hz must be a positive number"),
        (
            {**_BOUND_CAPTURE_BINDING, "channel_indices": [1, 2, 3, 3]},
            "audio_input.channel_indices must contain 4 unique non-negative integers",
        ),
        (
            {**_BOUND_CAPTURE_BINDING, "channel_indices": [1, 2, 3, 6]},
            "audio_input.channel_indices must be smaller than channels",
        ),
    ],
)
def test_bound_speech_direction_rejects_malformed_bindings(audio_input, error):
    from robot_config.loader import validate_robot_config_dict

    config = _bound_speech_direction_robot(audio_input=audio_input)
    with pytest.raises(ValueError) as excinfo:
        validate_robot_config_dict(config, deferred_interfaces=True)
    assert error in str(excinfo.value)


# --- bound consumers in the typed configuration ---------------------------------


def test_typed_loaders_preserve_the_audio_bindings():
    from robot_config.loader import load_speech_direction_config, load_voice_tts_config

    direction = load_speech_direction_config({"enabled": True, "audio_input": dict(_BOUND_CAPTURE_BINDING)})
    assert direction.audio_input["topic"] == "/audio/capture_stamped"
    tts = load_voice_tts_config(
        {"enabled": True, "audio_output": {"interface": "audio.play", "type": "audio_common_msgs/msg/AudioDataStamped"}}
    )
    assert tts.audio_output["interface"] == "audio.play"


def test_bound_consumer_snapshot_validates_through_the_typed_loader(tmp_path):
    """A bound provider snapshot survives load_robot_config + validate_config.

    The runtime-ready callback publishes exactly this shape (the bound fields
    plus the embedded runtime description); the typed path must not demand the
    audio_io section that provider deployments are forbidden to carry.
    """
    import yaml

    from robot_config.interface_binding import bind_robot_interfaces
    from robot_config.loader import load_robot_config, validate_config
    from robot_runtime.interface_description import description_digest, validate_description

    qos = {"reliability": "reliable", "durability": "volatile", "history": "keep_last", "depth": 10}
    description = {
        "schema_version": 1,
        "robot": {"id": "unit-1", "type": "lekiwi", "runtime_name": "lekiwi_robot", "runtime_version": "0.1.0"},
        "execution": "simulated",
        "interfaces": {
            "audio.play": {
                "capability": "interaction.audio_playback",
                "kind": "topic",
                "direction": "subscribe",
                "endpoint": "/audio/play",
                "message_type": "audio_common_msgs/msg/AudioDataStamped",
                "qos": dict(qos),
                "sample_rate_hz": 24000,
                "channels": 1,
                "sample_format": "S16LE",
            },
        },
        "states": {},
    }
    description["digest"] = description_digest(description)
    validate_description(description)

    config = {
        "name": "bound_tts_consumer",
        "type": "lekiwi",
        "runtime": {"provider": "lekiwi_robot", "profile": "lekiwi_nav_grasp", "target": "simulation"},
        "voice_tts": {
            "enabled": True,
            "bundle_path": "models/zipvoice",
            "deployment": "ascend_310p",
            "audio_output": {"interface": "audio.play", "type": "audio_common_msgs/msg/AudioDataStamped"},
        },
    }
    bound = bind_robot_interfaces(config, description)
    snapshot = {key: value for key, value in bound.items() if key != "_config_path"}
    snapshot_path = tmp_path / "bound_tts_snapshot.yaml"
    snapshot_path.write_text(yaml.safe_dump({"robot": snapshot}, sort_keys=False), encoding="utf-8")

    typed = load_robot_config(snapshot_path)
    assert validate_config(typed) == []
    assert typed.voice_tts.audio_output["topic"] == "/audio/play"
    assert typed.voice_tts.audio_output["sample_rate_hz"] == 24000


def test_unbound_typed_consumers_still_require_audio_io():
    """The provider-less legacy path keeps its audio_io requirement."""
    from robot_config.config import RobotConfig, Ros2ControlConfig
    from robot_config.loader import load_voice_tts_config, validate_config

    config = RobotConfig(
        name="legacy_tts",
        type="so101",
        robot_type="so101",
        ros2_control=Ros2ControlConfig(hardware_plugin="mock/hardware", params={}),
        peripherals=[],
    )
    config.voice_tts = load_voice_tts_config(
        {"enabled": True, "bundle_path": "/tmp/zipvoice", "deployment": "torch_cpu"}
    )
    errors = validate_config(config)
    assert any(
        "audio_io.enabled must be true when voice ASR, speech direction, or voice TTS is enabled" in e for e in errors
    )
