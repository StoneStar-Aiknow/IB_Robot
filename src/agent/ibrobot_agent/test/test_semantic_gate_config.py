"""Semantic gate parameters: defaults, validation, and the JSON-serialized phrase lists."""

from __future__ import annotations

import json

import pytest

from ibrobot_agent.voice_bridge import (
    VoiceBridgeConfig,
    config_from_parameters,
    node_parameter_defaults,
)

SCOPE = {"channel_id": "voice_asr", "principal_id": "local_operator", "robot_scope": "aimdk_x2"}


def _values(**overrides):
    values = node_parameter_defaults()
    values.update(SCOPE)
    values.update(overrides)
    return values


def test_the_gate_is_off_by_default():
    config = VoiceBridgeConfig(**SCOPE)
    assert config.semantic_gate_mode == "off"
    assert config.semantic_gate_model_path == ""
    assert config.semantic_gate_threshold == pytest.approx(0.10)
    assert config.semantic_gate_lexicon == ()
    assert config.semantic_gate_wake_prefixes == ()
    assert config.semantic_gate_fail_open is True


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="semantic_gate_mode"):
        VoiceBridgeConfig(**SCOPE, semantic_gate_mode="monitor")


@pytest.mark.parametrize("mode", ["shadow", "enforce"])
def test_a_mode_other_than_off_needs_a_model_path(mode):
    with pytest.raises(ValueError, match="semantic_gate_model_path"):
        VoiceBridgeConfig(**SCOPE, semantic_gate_mode=mode)


def test_threshold_must_be_a_probability():
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="semantic_gate_threshold"):
            VoiceBridgeConfig(**SCOPE, semantic_gate_threshold=bad)


def test_phrase_lists_are_declared_as_json_parameters():
    defaults = node_parameter_defaults()
    assert json.loads(defaults["semantic_gate_lexicon_json"]) == []
    assert json.loads(defaults["semantic_gate_wake_prefixes_json"]) == []
    assert defaults["semantic_gate_mode"] == "off"


def test_parameters_round_trip_into_the_config():
    config = config_from_parameters(
        _values(
            semantic_gate_mode="shadow",
            semantic_gate_model_path="/var/lib/ibrobot/addressee/model.json",
            semantic_gate_threshold=0.25,
            semantic_gate_lexicon_json=json.dumps(["挥挥手", "抓取"], ensure_ascii=False),
            semantic_gate_wake_prefixes_json=json.dumps(["灵犀", "林星"], ensure_ascii=False),
            semantic_gate_log_path="/var/log/ibrobot/semantic.jsonl",
            semantic_gate_fail_open=False,
        )
    )
    assert config.semantic_gate_mode == "shadow"
    assert config.semantic_gate_model_path == "/var/lib/ibrobot/addressee/model.json"
    assert config.semantic_gate_threshold == pytest.approx(0.25)
    assert config.semantic_gate_lexicon == ("挥挥手", "抓取")
    assert config.semantic_gate_wake_prefixes == ("灵犀", "林星")
    assert config.semantic_gate_log_path == "/var/log/ibrobot/semantic.jsonl"
    assert config.semantic_gate_fail_open is False


def test_lexicon_entries_are_normalized_like_transcripts():
    config = VoiceBridgeConfig(**SCOPE, semantic_gate_lexicon=(" 握 手 ", "", "抓取"))
    assert config.semantic_gate_lexicon == ("握手", "抓取")
