"""Config rules for an addressee gate that pairs transcripts by speech segment."""

from __future__ import annotations

import pytest

from robot_config.loader import validate_voice_bridge_segment_inputs

WAKE_EVENT = {"interface": "audio.wake_event", "type": "ibrobot_msgs/msg/SpeechWakeEvent"}


def _config(voice_bridge, *, input_mode="external_vad") -> dict:
    return {
        "voice_asr": {"enabled": True, "input_mode": input_mode},
        "embodied": {"agent": {"voice_bridge": {"enabled": True, **voice_bridge}}},
    }


def test_the_x2_shape_is_valid():
    voice_bridge = {"addressee_trigger": "vendor_session", "speaker_check_enabled": True, "wake_event": WAKE_EVENT}

    assert validate_voice_bridge_segment_inputs(_config(voice_bridge)) == []


def test_a_gate_that_does_not_pair_by_segment_is_not_constrained():
    plain = {"wake_phrases": ["灵犀灵犀"], "input_topic": "/asr/final_text"}

    assert validate_voice_bridge_segment_inputs(_config(plain, input_mode="shared_audio")) == []
    assert validate_voice_bridge_segment_inputs({}) == []


def test_the_session_trigger_needs_a_wake_event_because_nothing_else_opens_a_session():
    errors = validate_voice_bridge_segment_inputs(_config({"addressee_trigger": "vendor_session"}))

    assert any("only a wake event opens a session" in error for error in errors)


@pytest.mark.parametrize(
    "voice_bridge",
    [
        pytest.param({"wake_phrases": ["灵犀灵犀"], "speaker_check_enabled": True}, id="voiceprint check"),
        pytest.param({"addressee_trigger": "vendor_session", "wake_event": WAKE_EVENT}, id="session trigger"),
    ],
)
def test_pairing_by_segment_needs_a_segmenting_source(voice_bridge):
    """Only an external, segmenting source names the segment a transcript came from."""
    errors = validate_voice_bridge_segment_inputs(_config(voice_bridge, input_mode="shared_audio"))

    assert any("voice_asr.input_mode=external_vad" in error for error in errors)


def test_the_plain_text_topic_is_not_configurable_once_the_gate_pairs_by_segment():
    """The bridge reads voice_asr.transcript_topic then; an input_topic would be ignored."""
    voice_bridge = {"wake_phrases": ["灵犀灵犀"], "speaker_check_enabled": True, "input_topic": "/asr/final_text"}

    errors = validate_voice_bridge_segment_inputs(_config(voice_bridge))

    assert any("it follows voice_asr.transcript_topic" in error for error in errors)
