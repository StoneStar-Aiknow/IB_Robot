"""The X2 Agent voice deployment: ibrobot_agent entry with offline ZipVoice on the runtime speaker."""

from __future__ import annotations

from pathlib import Path

import pytest

from robot_config.config_path import resolve_robot_config_path
from robot_config.interface_binding import InterfaceBindingError, bind_robot_interfaces, required_interface_ids
from robot_config.loader import load_robot_config_dict, validate_robot_config_dict

CONFIG = resolve_robot_config_path(config_name="aimdk_x2_voice_agent")
# src/robot_config/test/ -> src/
PROFILE = Path(__file__).resolve().parents[2] / "robots" / "aimdk" / "aimdk_robot" / "profiles" / "x2_ultra.yaml"


@pytest.fixture(scope="module")
def config():
    return load_robot_config_dict(CONFIG, defer_interface_binding=True)


@pytest.fixture(scope="module")
def description():
    """The description the real X2 runtime would publish, built from its profile."""
    from robot_runtime.interface_description import build_description
    from robot_runtime.profile import load_profile

    return build_description(load_profile(PROFILE), use_sim=True)


def test_configuration_is_valid(config):
    validate_robot_config_dict(config, deferred_interfaces=True)


def test_it_layers_on_the_voice_profile_without_renaming_the_robot(config):
    # The contracts against the catalog profile (its robot_name and enabled skills) are
    # pinned in skill_catalog, which owns the profiles.
    assert config["_config_sources"][-2].endswith("aimdk_x2_voice.yaml")
    assert config["name"] == "aimdk_x2_skills"
    assert config["embodied"]["skill_catalog_profile"] == "aimdk_x2_skills"


def test_the_runtime_owns_every_audio_device(config):
    # A provider deployment carries no audio_io: capture and playback both bind runtime interfaces.
    assert "audio_io" not in config
    assert set(required_interface_ids(config)) == {"audio.speech", "audio.play", "motion.named"}


def test_agent_executes_only_allowlisted_gestures(config):
    agent = config["embodied"]["agent"]
    assert agent["execution_enabled"] is True
    assert agent["test_allowlist"]


def test_speech_output_uses_zipvoice_on_the_runtime_speaker(config):
    tts = config["voice_tts"]
    assert tts["enabled"] is True
    assert tts["deployment"] == "ascend_310p"
    assert tts["bundle_path"] == "models/zipvoice"
    # ZipVoice synthesizes 24 kHz; the runtime speaker takes 16 kHz.
    assert tts["resample_to_playback_rate"] is True
    assert tts["audio_output"]["interface"] == "audio.play"


def test_playback_binds_the_runtime_speaker_with_its_declared_format(config, description):
    bound = bind_robot_interfaces(config, description, require_ready=False)

    output = bound["voice_tts"]["audio_output"]
    assert output["_interface_source"]["id"] == "audio.play"
    assert output["topic"] == description["interfaces"]["audio.play"]["endpoint"]
    assert (output["sample_rate_hz"], output["channels"]) == (16000, 1)
    # The X2 runtime subscribes to AudioDataStamped, so that is what the playback node publishes.
    assert output["type"] == bound["voice_tts"]["audio_message_type"] == "audio_common_msgs/msg/AudioDataStamped"


def test_playback_binding_fails_closed_without_the_speaker_interface(config, description):
    from robot_runtime.interface_description import description_digest

    stripped = dict(description)
    stripped["interfaces"] = {k: v for k, v in description["interfaces"].items() if k != "audio.play"}
    stripped["digest"] = description_digest(stripped)
    with pytest.raises(InterfaceBindingError, match="unknown_interface: voice_tts.audio_output"):
        bind_robot_interfaces(config, stripped, require_ready=False)


def test_voice_bridge_speaks_with_a_timeout_covering_a_full_segment(config):
    bridge = config["embodied"]["agent"]["voice_bridge"]
    assert bridge["enabled"] is True
    assert bridge["tts_timeout_sec"] >= 90.0
    assert "tts_enabled" not in bridge  # follows voice_tts.enabled
