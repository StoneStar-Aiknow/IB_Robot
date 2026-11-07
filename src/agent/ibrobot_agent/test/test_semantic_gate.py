"""The L3 semantic gate: scoring, the veto rule, and how the bridge applies it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ibrobot_agent import semantic_gate as gate_module
from ibrobot_agent.semantic_gate import SEMANTIC_GATE_MODES, SemanticGate, normalize_text, strip_wake_words
from ibrobot_agent.voice_bridge import Log, PublishRequest, VoiceBridgeConfig, VoiceBridgeCore

SCOPE = {"channel_id": "voice_asr", "principal_id": "local_operator", "robot_scope": "aimdk_x2"}
REPO_ROOT = Path(__file__).resolve().parents[3].parent
REPO_MODEL = REPO_ROOT / "models" / "addressee" / "addressee_zh_v1.json"
EMIT_LEXICON = REPO_ROOT / "tools" / "addressee" / "emit_lexicon.py"


def _weigh(text: str, weight: float, *, bias: float = 0.0, dim: int = 4096) -> list[list[float]]:
    """One 1-gram weight, so a test model is readable instead of a fixture blob."""
    return [[gate_module._fnv1a32(char) % dim, weight] for char in normalize_text(text)]


def _payload(weights: list[list[float]], *, bias: float = 0.0, wake_words=("灵犀灵犀", "灵犀"), dim: int = 4096):
    return {
        "schema_version": 1,
        "created_at": "2026-11-07T12:00:00+0800",
        "feature": {"ngrams": [1], "dim": dim, "hash": "fnv1a32", "pad": "<>"},
        "wake_words": list(wake_words),
        "bias": bias,
        "weights": weights,
    }


def _toy_gate(tmp_path: Path, **kwargs) -> SemanticGate:
    """A model that knows "挥" and hates meeting words, so both branches are reachable."""
    weights = _weigh("挥", 4.0) + _weigh("今天开会", -3.0)
    payload = _payload(weights)
    gate = SemanticGate(payload, threshold=kwargs.pop("threshold", 0.10), **kwargs)
    return gate


def _quiet_gate(**kwargs) -> SemanticGate:
    """Only "挥" logits up; everything else, including two-character fragments, sits near 0."""
    payload = _payload(_weigh("挥", 6.0), bias=-4.0)
    return SemanticGate(payload, threshold=kwargs.pop("threshold", 0.10), **kwargs)


# --- format ------------------------------------------------------------------------------


def test_model_format_is_versioned_and_refuses_unknown_hashes():
    payload = _payload(_weigh("挥", 1.0))
    payload["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        SemanticGate(payload, threshold=0.1)
    payload = _payload(_weigh("挥", 1.0))
    payload["feature"]["hash"] = "python-hash"
    with pytest.raises(ValueError, match="hash"):
        SemanticGate(payload, threshold=0.1)


def test_python_hash_is_never_used_because_it_is_random_per_process():
    # The regression this guards: a model exported with Python's hash() scores 0 everywhere
    # in the process that loads it, which looks exactly like a gate that never fires.
    assert gate_module._fnv1a32("挥挥手") == gate_module._fnv1a32("挥挥手")
    payload = _payload([[gate_module._fnv1a32("挥") % 4096, 4.0]])
    gate = SemanticGate(payload, threshold=0.1)
    assert gate.score("挥挥手") > 0.9


# --- scoring and the veto rule ------------------------------------------------------------


def test_score_ranks_a_skill_word_above_meeting_talk(tmp_path):
    gate = _toy_gate(tmp_path)
    assert gate.score("挥挥手") > 0.9
    assert gate.score("今天开会") < 0.05


def test_low_score_without_a_skill_word_is_rejected(tmp_path):
    decision = _toy_gate(tmp_path).judge("今天开会")
    assert decision.rejected
    assert not decision.lexicon_hit
    assert decision.reason.startswith("score")


def test_a_skill_word_only_counts_in_a_command_sized_sentence():
    # 2026-11-07 field session: "这不是没开始吗智商没开始吗" carries 开始 and 镜子 and
    # scored 0.000, yet the catalog word alone let it through to the planner.
    gate = _quiet_gate(lexicon=("开始", "镜子"))
    decision = gate.judge("这不是没开始吗智商没开始吗")
    assert decision.lexicon_hit
    assert decision.rejected
    # The same word in a command still passes without the model's help.
    assert not gate.judge("开始照镜子").rejected


def test_naming_someone_else_in_the_room_is_not_addressed_to_the_robot():
    # "周哥举手的动作" was said to a colleague and scored 1.000 on the skill word.
    gate = _quiet_gate(lexicon=("举手",))
    decision = gate.judge("周哥举手的动作")
    assert decision.rejected
    assert decision.address_term == "哥"
    assert "someone else" in decision.reason


def test_a_bare_fragment_needs_a_session_the_addressee_gate_opened():
    # "怎么" scored 0.999: two characters carry no grammar, so the score is mostly the bias.
    gate = _quiet_gate()
    assert gate.judge("挥").rejected
    assert not gate.judge("挥", session_open=True).rejected


def test_skill_word_wins_over_the_model(tmp_path):
    # "抓取" is never in the corpus with this phrasing; the catalog's alias must win anyway.
    gate = _toy_gate(tmp_path, lexicon=("抓取",))
    decision = gate.judge("帮我抓取橘子")
    assert decision.lexicon_hit
    assert not decision.rejected


def test_wake_prefix_wins_over_the_model(tmp_path):
    gate = _toy_gate(tmp_path, wake_prefixes=("年轻人",))
    decision = gate.judge("年轻人继续")
    assert decision.wake_prefix
    assert not decision.rejected


def test_swallowing_the_wake_word_does_not_change_the_verdict(tmp_path):
    gate = _toy_gate(tmp_path)
    assert gate.score("灵犀灵犀挥挥手") == pytest.approx(gate.score("挥挥手"), abs=1e-6)
    assert strip_wake_words("灵犀灵犀挥挥手", ("灵犀灵犀", "灵犀")) == "挥挥手"


def test_a_transcript_that_is_only_the_wake_word_is_let_through(tmp_path):
    # Nothing left to judge: the user woke it deliberately, and dropping that would look dead.
    decision = _toy_gate(tmp_path).judge("灵犀")
    assert not decision.rejected
    assert "empty" in decision.reason


# --- configuration -----------------------------------------------------------------------


def test_off_builds_no_gate():
    config = VoiceBridgeConfig(**SCOPE, semantic_gate_mode="off")
    gate, status = SemanticGate.from_config(config)
    assert gate is None
    assert "off" in status


def test_missing_model_fails_open_and_says_why(tmp_path):
    config = VoiceBridgeConfig(
        **SCOPE, semantic_gate_mode="shadow", semantic_gate_model_path=str(tmp_path / "absent.json")
    )
    gate, status = SemanticGate.from_config(config)
    assert gate is None
    assert "disabled" in status


def test_missing_model_raises_when_failing_open_is_off(tmp_path):
    config = VoiceBridgeConfig(
        **SCOPE,
        semantic_gate_mode="enforce",
        semantic_gate_model_path=str(tmp_path / "absent.json"),
        semantic_gate_fail_open=False,
    )
    with pytest.raises(OSError):
        SemanticGate.from_config(config)


def test_a_loaded_model_reports_its_threshold(tmp_path):
    model = tmp_path / "model.json"
    model.write_text(json.dumps(_payload(_weigh("挥", 4.0))), encoding="utf-8")
    gate, status = SemanticGate.from_config(
        VoiceBridgeConfig(**SCOPE, semantic_gate_mode="shadow", semantic_gate_model_path=str(model))
    )
    assert gate is not None
    assert "threshold=0.10" in status


# --- shadow log --------------------------------------------------------------------------


def test_record_appends_json_lines(tmp_path):
    gate = _toy_gate(tmp_path)
    log_path = tmp_path / "semantic.jsonl"
    decision = gate.judge("今天开会")
    gate.record(decision, str(log_path), mode="shadow", after_wake=False)
    gate.record(gate.judge("挥挥手"), str(log_path), mode="shadow", after_wake=True)
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [line["rejected"] for line in lines] == [True, False]
    assert lines[0]["text"] == "今天开会"
    assert lines[1]["after_wake"] is True
    assert lines[0]["mode"] == "shadow"


def test_record_never_raises_when_the_log_cannot_be_written(tmp_path):
    gate = _toy_gate(tmp_path)
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    gate.record(gate.judge("今天开会"), str(blocker / "semantic.jsonl"), mode="shadow")  # must not raise
    gate.record(gate.judge("今天开会"), "", mode="shadow")


# --- bridge integration ------------------------------------------------------------------


def _core(tmp_path: Path, mode: str, **overrides) -> VoiceBridgeCore:
    model = tmp_path / "model.json"
    model.write_text(json.dumps(_payload(_weigh("挥", 4.0) + _weigh("今天开会", -3.0))), encoding="utf-8")
    ids = iter(f"voice-{index}" for index in range(100))
    config = VoiceBridgeConfig(
        **{
            "confirm_mode": "auto",
            **SCOPE,
            "semantic_gate_mode": mode,
            "semantic_gate_model_path": str(model),
            "semantic_gate_log_path": str(tmp_path / "semantic.jsonl"),
            **overrides,
        }
    )
    return VoiceBridgeCore(config, id_factory=lambda: next(ids))


def _requests(actions):
    return [action for action in actions if isinstance(action, PublishRequest)]


def test_shadow_records_but_still_publishes(tmp_path):
    core = _core(tmp_path, "shadow")
    actions = core.on_transcript("今天开会", now=1.0)
    assert len(_requests(actions)) == 1
    logged = [json.loads(line) for line in (tmp_path / "semantic.jsonl").read_text().splitlines()]
    assert logged[0]["rejected"] is True
    assert logged[0]["mode"] == "shadow"


def test_enforce_drops_the_utterance_and_says_why(tmp_path):
    core = _core(tmp_path, "enforce")
    actions = core.on_transcript("今天开会", now=1.0)
    assert not _requests(actions)
    assert any("not addressed to me (semantic" in action.message for action in actions if isinstance(action, Log))
    assert core.active_request_id == ""


def test_enforce_still_publishes_a_skill_word(tmp_path):
    core = _core(tmp_path, "enforce", semantic_gate_lexicon=("抓取",))
    actions = core.on_transcript("帮我抓取橘子", now=1.0)
    assert len(_requests(actions)) == 1


def test_enforce_never_swallows_a_stop(tmp_path):
    core = _core(tmp_path, "enforce")
    actions = core.on_transcript("停止", now=1.0)
    # Nothing is running, so a bare stop is not routed as a request, but it is never
    # treated as chatter either: the stop branch runs before the gate.
    assert not any("not addressed to me" in action.message for action in actions if isinstance(action, Log))


@pytest.mark.skipif(not REPO_MODEL.exists(), reason="trained model not present")
def test_the_shipped_model_scores_field_sentences_as_trained():
    gate = SemanticGate.load(str(REPO_MODEL), threshold=0.10)
    assert gate.score("灵犀灵犀挥挥手") > 0.9
    assert gate.score("帮我抓取橘子") > 0.9
    assert gate.score("第二件事情呢就是我们成立了全域安全部专门设计") < 0.05
    assert gate.judge("这个要问吗").rejected


def test_modes_are_the_documented_set():
    assert SEMANTIC_GATE_MODES == ("off", "shadow", "enforce")


@pytest.mark.skipif(not EMIT_LEXICON.exists(), reason="lexicon tool not present")
def test_the_generated_lexicon_holds_no_bare_negation():
    # "不是" is an answer, and as a skill word it let the question "是不是" through the gate.
    import importlib.util

    spec = importlib.util.spec_from_file_location("emit_lexicon", EMIT_LEXICON)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    words = module.catalog_lexicon(module.CATALOG)

    assert "不是" not in words
    assert "挥手" in words
