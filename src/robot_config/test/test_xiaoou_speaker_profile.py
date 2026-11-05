"""Optional text wake/speaker settings stay disabled in the shipped profile."""

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


def test_grasp_mirror_defaults_to_no_wake_or_speaker_check():
    config = _profile()
    voice = config["voice_asr"]
    bridge = config["embodied"]["agent"]["voice_bridge"]
    assert voice["input_mode"] == "external_vad"
    assert voice["speaker_embedding"]["enabled"] is False
    assert bridge["addressee_trigger"] == "wake_phrase"
    assert "wake_phrases" not in bridge
    assert bridge["speaker_check_enabled"] is False
    assert not bridge.get("wake_event")
    assert _validate_agent_voice_bridge(bridge) == []
    assert validate_speaker_embedding_config(config) == []
    assert validate_voice_bridge_segment_inputs(config) == []


def test_grasp_mirror_can_opt_in_to_text_wake_and_voiceprints():
    config = deepcopy(_profile())
    config["voice_asr"]["speaker_embedding"]["enabled"] = True
    bridge = config["embodied"]["agent"]["voice_bridge"]
    bridge["speaker_check_enabled"] = True
    bridge["wake_phrases"] = [f"小{a}小{b}" for a in "欧鸥偶" for b in "欧鸥偶"]
    assert config["voice_asr"]["speaker_embedding"]["bundle_path"] == "models/campplus-speaker"
    assert bridge["speaker_reject_short"] is True
    assert bridge["wake_window_sec"] == 25.0
    assert _validate_agent_voice_bridge(bridge) == []
    assert validate_speaker_embedding_config(config) == []
    assert validate_voice_bridge_segment_inputs(config) == []
