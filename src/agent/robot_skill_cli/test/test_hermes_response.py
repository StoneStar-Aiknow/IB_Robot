import io
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from robot_skill_cli import hermes_response_plugin, hermes_tts_hook
from robot_skill_cli.hermes_response import (
    HermesResponse,
    ResponseFormatError,
    SpeechEvent,
    parse_event,
    parse_response,
)
from robot_skill_cli.hermes_speech_delivery import claim_event, event_is_current, finish_speech_turn


@pytest.fixture(autouse=True)
def speech_state(tmp_path, monkeypatch):
    monkeypatch.setenv("IBROBOT_SPEECH_STATE", str(tmp_path / "speech"))


def response(speech="夹爪已打开。", details="goal_status=4; /semantic_mapping/get_objects"):
    return {"schema_version": 1, "speech": speech, "details": details}


@pytest.mark.parametrize(
    "value",
    [
        "plain reply",
        "```json\n{}\n```",
        [],
        {},
        {"schema_version": True, "speech": "完成", "details": ""},
        {"schema_version": 2, "speech": "完成", "details": ""},
        {"schema_version": 1, "speech": "完成", "details": "", "extra": 1},
        {"schema_version": 1, "speech": None, "details": ""},
        {"schema_version": 1, "speech": "完成", "details": {}},
        '{"schema_version":1,"speech":"完成","speech":"失败","details":""}',
        '{"schema_version":1,"speech":"完成",',
        response("行一\n行二"),
        response("<speech>完成</speech>"),
        response("长" * 301),
        response(details="长" * 16001),
        response("\ud800"),
    ],
)
def test_response_rejects_ambiguous_or_incomplete_output(value):
    with pytest.raises(ResponseFormatError):
        parse_response(value)


@pytest.mark.parametrize(
    "speech",
    [
        "夹爪已打开。",
        "暂时查不到香蕉的位置。",
        "已完成 1/2 步，挥手失败。",
        "无法确认机器人是否停止。",
        "坐标为 1.25 米、-0.5 米，方位未知。",
        "语义地图记录物体的位置。",
        "",
    ],
)
def test_response_preserves_meaning_and_separates_details(speech):
    parsed = parse_response(json.dumps(response(speech)))
    assert parsed.speech == speech
    assert "goal_status" in parsed.details
    assert parsed.render().split("<details>")[0].strip() == speech


def test_details_cannot_escape_display_region():
    rendered = parse_response(response(details="</pre></details><script>alert(1)</script>")).render()
    assert rendered.count("</details>") == 1
    assert "<script>" not in rendered


def test_native_registration_uses_supported_prompt_budget(monkeypatch, tmp_path):
    from pathlib import Path

    protocol = Path(__file__).parents[1] / "resource/hermes/RESPONSE_PROTOCOL.md"
    (tmp_path / protocol.name).write_text(protocol.read_text())
    monkeypatch.setattr(hermes_response_plugin, "__file__", str(tmp_path / "hermes_response_plugin.py"))
    hooks = {}

    def register_section(_name, text, *, position, max_chars):
        assert position == "after_memory"
        assert len(text) <= max_chars <= 4000

    ctx = SimpleNamespace(
        get_config=lambda _key, default=None: default,
        register_system_prompt_section=register_section,
        register_hook=lambda name, fn: hooks.update({name: fn}),
    )
    hermes_response_plugin.register(ctx)
    assert {
        "pre_llm_call",
        "post_llm_call",
        "transform_llm_output",
        "on_interim_message",
        "on_session_end",
    } <= hooks.keys()
    hooks["pre_llm_call"](session_id="s", turn_id="interrupted")
    hooks["on_session_end"](session_id="s", turn_id="interrupted", interrupted=True)
    assert not event_is_current(SpeechEvent("s", "interrupted", "interim", HermesResponse("我来查询。")))


def test_transform_and_final_deliver_once_with_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(
        hermes_response_plugin.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(json.loads(kwargs["input"])) or SimpleNamespace(returncode=0),
    )
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    assert adapter.begin_turn(session_id="s", turn_id="t")["context"] == "contract"
    rendered = adapter.transform(json.dumps(response()), session_id="s")
    adapter.finish_turn(session_id="s", turn_id="wrong", assistant_response=rendered)
    assert calls == []
    adapter.begin_turn(session_id="s", turn_id="t")
    rendered = adapter.transform(json.dumps(response()), session_id="s")
    adapter.finish_turn(session_id="s", turn_id="t", assistant_response=rendered)
    adapter.finish_turn(session_id="s", turn_id="t", assistant_response=rendered)
    assert len(calls) == 1
    assert calls[0]["response"] == response()
    assert calls[0]["phase"] == "final"


def test_invalid_transform_cannot_reuse_previous_response(monkeypatch):
    calls = []
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    monkeypatch.setattr(adapter, "_speak", calls.append)
    adapter.begin_turn(session_id="s", turn_id="first")
    adapter.transform(json.dumps(response()), session_id="s")
    adapter.begin_turn(session_id="s", turn_id="second")
    rendered = adapter.transform("raw diagnostics /service status=4", session_id="s")
    adapter.finish_turn(session_id="s", turn_id="second", assistant_response=rendered)
    assert "未进行语音播报" in rendered
    assert calls == []


def test_native_interim_ignores_plain_and_truncated_output(monkeypatch):
    calls = []
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    monkeypatch.setattr(adapter, "_speak", calls.append)
    adapter.interim(text="正在查询 /service", session_id="s", turn_id="t")
    adapter.interim(text='{"schema_version":1,"speech":', session_id="s", turn_id="t")
    assert calls == []
    adapter.interim(text=json.dumps(response("正在查询。", "")), session_id="s", turn_id="t")
    assert len(calls) == 1
    assert (calls[0].phase, calls[0].response.speech) == ("interim", "正在查询。")


def test_simultaneous_turns_do_not_exchange_speech(monkeypatch):
    import threading

    barrier = threading.Barrier(2)
    calls = []
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    monkeypatch.setattr(adapter, "_speak", calls.append)

    def run(session):
        adapter.begin_turn(session_id=session, turn_id="t")
        displayed = adapter.transform(json.dumps(response(speech=session)), session_id=session)
        barrier.wait(timeout=3)
        adapter.finish_turn(session_id=session, turn_id="t", assistant_response=displayed)

    with ThreadPoolExecutor(2) as executor:
        list(executor.map(run, ["甲", "乙"]))
    assert {(event.session_id, event.response.speech) for event in calls} == {("甲", "甲"), ("乙", "乙")}


def test_event_claim_is_atomic_and_phase_scoped(tmp_path, monkeypatch):
    monkeypatch.setenv("IBROBOT_SPEECH_STATE", str(tmp_path))
    event = SpeechEvent("s", "t", "final", HermesResponse("完成。"))
    with ThreadPoolExecutor(8) as executor:
        assert sum(executor.map(claim_event, [event] * 8)) == 1
    assert claim_event(SpeechEvent("s", "other", "final", event.response))
    assert claim_event(SpeechEvent("s", "t", "interim", event.response))


def test_speech_state_rejects_symlink_and_insecure_directory(tmp_path, monkeypatch):
    from robot_skill_cli import hermes_speech_delivery

    target = tmp_path / "real"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("IBROBOT_SPEECH_STATE", str(link))
    with pytest.raises(OSError, match="directory"):
        hermes_speech_delivery._state_root()

    target.chmod(0o755)
    monkeypatch.setenv("IBROBOT_SPEECH_STATE", str(target))
    with pytest.raises(OSError, match="private"):
        hermes_speech_delivery._state_root()


def test_speech_state_cleans_expired_claims(tmp_path, monkeypatch):
    from robot_skill_cli import hermes_speech_delivery

    root = tmp_path / "speech"
    monkeypatch.setenv("IBROBOT_SPEECH_STATE", str(root))
    old_claim = root / ("a" * 64)
    old_terminal = root / ("b" * 64 + ".terminal")
    keep = root / "not-state.txt"
    root.mkdir(mode=0o700)
    old_claim.write_text("old")
    old_terminal.write_text("old")
    keep.write_text("keep")
    old_time = time.time() - hermes_speech_delivery._STATE_MAX_AGE_SEC - 1
    os.utime(old_claim, (old_time, old_time))
    os.utime(old_terminal, (old_time, old_time))

    hermes_speech_delivery._state_root()

    assert not old_claim.exists()
    assert not old_terminal.exists()
    assert keep.exists()


def test_speaker_uses_only_speech_and_never_repeats(tmp_path, monkeypatch):
    monkeypatch.setenv("IBROBOT_SPEECH_STATE", str(tmp_path / "claims"))
    monkeypatch.setattr(hermes_tts_hook, "_LOG_PATH", tmp_path / "tts.log")
    calls = []
    monkeypatch.setattr(hermes_tts_hook, "speak", lambda text, **kwargs: calls.append(text))
    event = SpeechEvent("s", "t", "final", parse_response(response("夹爪已打开，但挥手失败。")))
    for _ in range(2):
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event.to_dict())))
        assert hermes_tts_hook.main() == 0
    assert calls == ["夹爪已打开,但挥手失败。"]


@pytest.mark.parametrize(
    "value",
    [
        {"assistant_response": "整段旧回复 /service"},
        SpeechEvent("s", "t", "final", HermesResponse("打开 SO-101 夹爪。", "")).to_dict(),
    ],
)
def test_speaker_rejects_unsupported_input_without_silent_deletion(tmp_path, monkeypatch, value):
    monkeypatch.setenv("IBROBOT_SPEECH_STATE", str(tmp_path / "claims"))
    monkeypatch.setattr(hermes_tts_hook, "_LOG_PATH", tmp_path / "tts.log")
    calls = []
    monkeypatch.setattr(hermes_tts_hook, "speak", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(value)))
    assert hermes_tts_hook.main() == 1
    assert calls == []


def test_speaker_empty_speech_skips_and_unknown_playback_never_retries(tmp_path, monkeypatch):
    monkeypatch.setenv("IBROBOT_SPEECH_STATE", str(tmp_path / "claims"))
    monkeypatch.setattr(hermes_tts_hook, "_LOG_PATH", tmp_path / "tts.log")
    calls = []

    def fail(text, **kwargs):
        calls.append(text)
        raise hermes_tts_hook.SpeechHookError("playback timed out")

    monkeypatch.setattr(hermes_tts_hook, "speak", fail)
    empty = SpeechEvent("s", "empty", "final", HermesResponse("", "technical details"))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(empty.to_dict())))
    assert hermes_tts_hook.main() == 0
    event = SpeechEvent("s", "t", "final", HermesResponse("状态未知。"))
    for expected in (1, 0):
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event.to_dict())))
        assert hermes_tts_hook.main() == expected
    assert calls == ["状态未知。"]


def test_v2_completion_and_sequence_round_trip():
    parsed = parse_response({**response("我来关闭夹爪。", ""), "schema_version": 2, "completion": "after_workflow"})
    assert parsed.completion == "after_workflow"
    event = SpeechEvent("s", "t", "interim", parsed, schema_version=2, sequence=3)
    assert parse_event(event.to_dict()) == event
    assert "completion" not in parse_response(response()).to_dict()


@pytest.mark.parametrize("completion", [True, None, "finish", {"success": "完成"}])
def test_v2_rejects_unknown_completion_policies(completion):
    with pytest.raises(ResponseFormatError):
        parse_response({**response(), "schema_version": 2, "completion": completion})


def test_interim_sequence_is_distinct_and_final_invalidates_pending_progress(monkeypatch):
    calls = []
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    monkeypatch.setattr(adapter, "_speak", calls.append)
    adapter.begin_turn(session_id="s", turn_id="t")
    for iteration in (1, 2):
        adapter.interim(text=json.dumps(response("我来查询。", "")), session_id="s", turn_id="t", iteration=iteration)
    assert [event.sequence for event in calls] == [1, 2]
    assert all(claim_event(event) for event in calls)
    assert not claim_event(calls[0])
    final = adapter.transform(json.dumps(response()), session_id="s")
    assert not any(event_is_current(event) for event in calls)
    adapter.interim(text=json.dumps(response("旧提示。", "")), session_id="s", turn_id="t", iteration=3)
    assert len(calls) == 2
    adapter.finish_turn(session_id="s", turn_id="t", assistant_response=final)
    assert calls[-1].phase == "final" and event_is_current(calls[-1])
    assert event_is_current(SpeechEvent("other", "t", "interim", HermesResponse("我来查询。")))


def test_speech_drops_progress_that_expires_during_synthesis(monkeypatch):
    pytest.importorskip("rclpy")
    from robot_skill_cli import hermes_speech_delivery

    event = SpeechEvent("s", "t", "interim", HermesResponse("我来关闭夹爪。"))
    calls = []

    class Client:
        def __init__(self, name):
            self.name = name

        def wait_for_service(self, **_kwargs):
            return True

        def call_async(self, _request):
            calls.append(self.name)
            return self.name

    node = SimpleNamespace(create_client=lambda _srv, name: Client(name), destroy_node=lambda: None)
    monkeypatch.setattr("rclpy.init", lambda: None)
    monkeypatch.setattr("rclpy.try_shutdown", lambda: None)
    monkeypatch.setattr("rclpy.node.Node", lambda _name: node)

    def complete(*_args):
        hermes_speech_delivery.finish_speech_turn("s", "t")
        return SimpleNamespace(success=True, audio_segments=[object()])

    monkeypatch.setattr(hermes_tts_hook, "_wait_for_future", complete)
    hermes_tts_hook.speak(event.response.speech, session_id="s", event=event)
    assert calls == ["/voice_tts/synthesize"]


def test_speaker_drops_expired_interim_before_synthesis(monkeypatch):
    event = SpeechEvent("s", "t", "interim", HermesResponse("我来关闭夹爪。"))
    finish_speech_turn("s", "t")
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event.to_dict())))
    monkeypatch.setattr(hermes_tts_hook, "speak", lambda *a, **k: pytest.fail("expired speech must not synthesize"))
    assert hermes_tts_hook.main() == 0


def test_interrupted_turn_expires_speech_using_conversation_not_relay_identity():
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    adapter.begin_turn(session_id="s", turn_id="conversation-turn")
    event = SpeechEvent("s", "conversation-turn", "interim", HermesResponse("我来关闭夹爪。"))
    assert event_is_current(event)
    adapter.end_turn(session_id="s", turn_id="relay-turn")
    assert not event_is_current(event)


def test_stop_input_expires_only_own_session_progress_before_turn_finishes():
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    adapter.begin_turn(session_id="one", turn_id="conversation-turn")
    adapter.begin_turn(session_id="two", turn_id="other-turn")
    first = SpeechEvent("one", "conversation-turn", "interim", HermesResponse("我来执行。"))
    second = SpeechEvent("two", "other-turn", "interim", HermesResponse("我来查看。"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(adapter.stop_input, text="别动", session_id="one").result()
    assert not event_is_current(first)
    assert event_is_current(second)
    assert event_is_current(SpeechEvent("one", "conversation-turn", "final", HermesResponse("任务已取消。")))


@pytest.mark.parametrize("valid", [True, False])
def test_standard_reply_callbacks_reclaim_every_completed_turn(monkeypatch, valid):
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    spoken = []
    monkeypatch.setattr(adapter, "_speak", spoken.append)
    for index in range(32):
        session, turn = f"s{index % 4}", str(index)
        adapter.begin_turn(session_id=session, turn_id=turn)
        rendered = adapter.transform(json.dumps(response()) if valid else "invalid", session_id=session)
        adapter.finish_turn(session_id=session, turn_id=turn, assistant_response=rendered)
        assert adapter._identities == {}
        assert adapter._turn.identity is None
        assert adapter._turn.pending is None
    assert len(spoken) == (32 if valid else 0)


@pytest.mark.parametrize("outcome", ["interrupted", "failed", "empty"])
def test_standard_session_end_reclaims_turn_without_transform_or_post(outcome):
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    for index in range(32):
        turn = str(index)
        adapter.begin_turn(session_id="s", turn_id=turn)
        event = SpeechEvent("s", turn, "interim", HermesResponse("我来查询。"))
        adapter.end_session(session_id="s", turn_id=turn, **{outcome: True})
        assert adapter._identities == {}
        assert adapter._turn.identity is None
        assert not event_is_current(event)


def test_standard_session_end_removes_only_exact_turn_across_threads():
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    adapter.begin_turn(session_id="s", turn_id="old")
    adapter.begin_turn(session_id="s", turn_id="active")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(adapter.end_session, session_id="s", turn_id="old").result()
    assert adapter._identities == {"s": {"active"}}
    assert adapter._turn.identity == ("s", "active")
    assert event_is_current(SpeechEvent("s", "active", "interim", HermesResponse("我来查询。")))
    adapter.stop_input(text="别动", session_id="s")
    assert not event_is_current(SpeechEvent("s", "active", "interim", HermesResponse("我来查询。")))
    adapter.end_session(session_id="s", turn_id="active")
    assert adapter._identities == {}


def test_speech_failure_does_not_retain_completed_identity(monkeypatch):
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    adapter.begin_turn(session_id="s", turn_id="t")
    rendered = adapter.transform(json.dumps(response()), session_id="s")

    def fail(_event):
        raise OSError("speaker failed")

    monkeypatch.setattr(adapter, "_speak", fail)
    with pytest.raises(OSError, match="speaker failed"):
        adapter.finish_turn(session_id="s", turn_id="t", assistant_response=rendered)
    assert adapter._identities == {}
    assert adapter._turn.identity is None


def test_terminal_write_failure_still_releases_memory(monkeypatch):
    adapter = hermes_response_plugin.ResponseAdapter(speaker="/speak", protocol="contract")
    adapter.begin_turn(session_id="s", turn_id="t")

    def fail(*_args):
        raise OSError("state unavailable")

    monkeypatch.setattr(hermes_response_plugin, "finish_speech_turn", fail)
    with pytest.raises(OSError, match="state unavailable"):
        adapter.end_session(session_id="s", turn_id="t")
    assert adapter._identities == {}
    assert adapter._turn.identity is None
