"""The commissioned external microphone uses Oula wake and speaker checking."""

from copy import deepcopy
from pathlib import Path

from robot_config.loader import (
    _load_robot_section,
    _validate_agent_voice_bridge,
    validate_speaker_embedding_config,
    validate_voice_bridge_segment_inputs,
)


def _profile():
    robots = Path(__file__).resolve().parents[1] / "config" / "robots"
    return _load_robot_section(robots / "aimdk_x2_voice_mirror_grasp.yaml")[1]


def test_grasp_mirror_defaults_to_external_microphone_and_verified_oula_wake():
    config = _profile()
    voice = config["voice_asr"]
    bridge = config["embodied"]["agent"]["voice_bridge"]
    assert voice["input_mode"] == "shared_audio"
    assert voice["audio_input_channel"] == 0
    assert voice["speaker_embedding"]["enabled"] is True
    assert bridge["addressee_trigger"] == "wake_phrase"
    forms = ("欧拉", "欧啦", "鸥拉", "偶拉")
    assert bridge["wake_phrases"] == [a + b for a in forms for b in forms]
    assert bridge["speaker_check_enabled"] is True
    assert bridge["require_verified_speech"] is True
    assert not bridge.get("wake_event")
    assert _validate_agent_voice_bridge(bridge) == []
    assert validate_speaker_embedding_config(config) == []
    assert validate_voice_bridge_segment_inputs(config) == []


def test_grasp_mirror_can_disable_the_speaker_gate_without_changing_audio_input():
    config = deepcopy(_profile())
    bridge = config["embodied"]["agent"]["voice_bridge"]
    bridge["speaker_check_enabled"] = False
    bridge["speaker_split_enabled"] = False
    bridge["require_verified_speech"] = False
    assert config["voice_asr"]["speaker_embedding"]["bundle_path"] == "models/campplus-speaker"
    assert bridge["speaker_reject_short"] is True
    assert bridge["wake_window_sec"] == 60.0
    assert _validate_agent_voice_bridge(bridge) == []
    assert validate_speaker_embedding_config(config) == []
    assert validate_voice_bridge_segment_inputs(config) == []
