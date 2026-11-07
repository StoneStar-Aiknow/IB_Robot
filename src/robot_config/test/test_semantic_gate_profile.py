"""The L3 semantic gate section of a robot profile must fail at load time when it cannot run."""

from __future__ import annotations

from robot_config.loader import _validate_agent_voice_bridge


def _bridge(**gate):
    return {"enabled": True, "stop_phrases": ["停止"], **gate}


def test_gate_is_optional_and_off_by_default():
    assert _validate_agent_voice_bridge(_bridge()) == []


def test_a_shadow_gate_with_a_model_is_valid():
    errors = _validate_agent_voice_bridge(
        _bridge(
            semantic_gate_mode="shadow",
            semantic_gate_model_path="/var/lib/ibrobot/addressee/model.json",
            semantic_gate_threshold=0.1,
            semantic_gate_lexicon=["挥挥手", "抓取"],
            semantic_gate_wake_prefixes=["灵犀"],
            semantic_gate_log_path="/var/log/ibrobot/semantic_gate.jsonl",
            semantic_gate_fail_open=True,
        )
    )
    assert errors == []


def test_shadow_without_a_model_is_rejected():
    errors = _validate_agent_voice_bridge(_bridge(semantic_gate_mode="shadow"))
    assert any("semantic_gate_model_path is required" in error for error in errors)


def test_an_unknown_mode_is_rejected():
    errors = _validate_agent_voice_bridge(
        _bridge(semantic_gate_mode="monitor", semantic_gate_model_path="/tmp/model.json")
    )
    assert any("semantic_gate_mode must be off, shadow or enforce" in error for error in errors)


def test_the_model_only_threshold_must_not_underbid_the_threshold():
    errors = _validate_agent_voice_bridge(
        _bridge(
            semantic_gate_mode="enforce",
            semantic_gate_model_path="/tmp/model.json",
            semantic_gate_threshold=0.30,
            semantic_gate_model_only_threshold=0.20,
        )
    )
    assert any("must not be below semantic_gate_threshold" in error for error in errors)
    for bad in (0, 1.5, True):
        errors = _validate_agent_voice_bridge(
            _bridge(
                semantic_gate_mode="enforce",
                semantic_gate_model_path="/tmp/model.json",
                semantic_gate_model_only_threshold=bad,
            )
        )
        assert any("semantic_gate_model_only_threshold must be in (0, 1]" in error for error in errors), bad


def test_the_threshold_must_stay_inside_the_open_unit_interval():
    for bad in (0, 1, 2.5):
        errors = _validate_agent_voice_bridge(
            _bridge(
                semantic_gate_mode="enforce",
                semantic_gate_model_path="/tmp/model.json",
                semantic_gate_threshold=bad,
            )
        )
        assert any("semantic_gate_threshold must be in (0, 1)" in error for error in errors), bad


def test_lexicon_entries_must_be_non_empty_strings():
    errors = _validate_agent_voice_bridge(
        _bridge(
            semantic_gate_mode="shadow",
            semantic_gate_model_path="/tmp/model.json",
            semantic_gate_lexicon=["挥挥手", ""],
        )
    )
    assert any("semantic_gate_lexicon must be a list of non-empty strings" in error for error in errors)


def test_fail_open_must_be_a_boolean():
    errors = _validate_agent_voice_bridge(
        _bridge(
            semantic_gate_mode="shadow",
            semantic_gate_model_path="/tmp/model.json",
            semantic_gate_fail_open="yes",
        )
    )
    assert any("semantic_gate_fail_open must be a boolean" in error for error in errors)
