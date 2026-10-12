from __future__ import annotations

import pytest

from ibrobot_agent.presentation import presentation_digest
from ibrobot_agent.voice_bridge import Log, PublishControl, PublishRequest, Speak, VoiceBridgeConfig, VoiceBridgeCore

SCOPE = {"channel_id": "voice_asr", "principal_id": "local_operator", "robot_scope": "so101_single_arm"}


def _core(**overrides) -> VoiceBridgeCore:
    ids = iter(f"voice-{index}" for index in range(100))
    return VoiceBridgeCore(VoiceBridgeConfig(**{**SCOPE, **overrides}), id_factory=lambda: next(ids))


def _of(actions, kind):
    return [action for action in actions if isinstance(action, kind)]


def _event(request_id, event_type, state, message="", **extra):
    key = {**SCOPE, "request_id": request_id}
    return {"request_key": key, "event_type": event_type, "state": state, "user_message": message, **extra}


def _presentation_event(request_id):
    presentation = {"summary": "nod_yes", "steps": [{"index": 1, "skill_name": "nod_yes"}]}
    detail = {"presentation": presentation, "presentation_digest": presentation_digest(presentation)}
    return _event(request_id, "presentation", "MAY_EXECUTE", detail={**detail, "receipt_token": "token-1"})


def test_transcript_becomes_agent_request_and_stop_phrase_bypasses_planner():
    core = _core()

    (request,) = _of(core.on_transcript("机器人请点头", now=0.0), PublishRequest)
    stop = core.on_transcript("停止！", now=1.0)

    assert request.payload["text"] == "机器人请点头"
    assert request.payload["request_id"] == "voice-0"
    assert _of(stop, PublishRequest) == []
    assert [action.payload for action in _of(stop, PublishControl)] == [{"operation": "stop", "request_id": "voice-0"}]


def test_busy_channel_drops_new_utterance_and_tells_user():
    core = _core()
    core.on_transcript("挥手", now=0.0)

    actions = core.on_transcript("点头", now=1.0)

    assert _of(actions, PublishRequest) == []
    assert [action.text for action in _of(actions, Speak)] == [VoiceBridgeConfig.busy_message]


def test_auto_mode_executes_without_second_confirmation_once_plan_is_rendered():
    core = _core()
    core.on_transcript("点头", now=0.0)

    actions = core.on_event(_presentation_event("voice-0"), now=1.0)

    assert _of(actions, PublishControl) == []
    (speak,) = _of(actions, Speak)
    (ack,) = speak.then
    assert ack.payload["operation"] == "presentation_rendered"
    assert ack.payload["receipt_token"] == "token-1"


def test_tampered_presentation_is_never_acknowledged():
    core = _core()
    core.on_transcript("点头", now=0.0)
    event = _presentation_event("voice-0")
    event["detail"]["presentation"]["summary"] = "wave_hello"

    actions = core.on_event(event, now=1.0)

    assert _of(actions, Speak) == [] and _of(actions, PublishControl) == []


def test_opt_in_voice_confirm_waits_for_confirmation_and_times_out():
    core = _core(confirm_mode="voice_confirm", confirm_timeout_sec=5.0)
    core.on_transcript("点头", now=0.0)
    core.on_event(_presentation_event("voice-0"), now=1.0)

    assert _of(core.on_transcript("今天天气不错", now=2.0), PublishControl) == []
    (ack,) = _of(core.on_transcript("确认执行", now=3.0), PublishControl)
    assert ack.payload["operation"] == "presentation_rendered"

    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=5.0)
    core.on_transcript("挥手", now=6.0)
    core.on_event(_presentation_event("voice-1"), now=7.0)
    (stop,) = _of(core.tick(12.0), PublishControl)
    assert stop.payload == {"operation": "stop", "request_id": "voice-1"}


@pytest.mark.parametrize(
    ("event_type", "state", "message", "spoken"),
    [
        ("conversation", "ANSWERED", "你好，我可以帮助你控制 SO-101。", "你好，我可以帮助你控制 SO-101。"),
        ("terminal", "SUCCEEDED", "Execution succeeded", "执行完成。"),
        ("terminal", "CANCELLED", "Execution cancelled", "已停止。"),
    ],
)
def test_terminal_reply_is_spoken_once_in_user_facing_words(event_type, state, message, spoken):
    core = _core()
    core.on_transcript("点头", now=0.0)

    first = core.on_event(_event("voice-0", event_type, state, message), now=1.0)
    duplicate = core.on_event(_event("voice-0", event_type, state, message), now=1.1)

    assert [action.text for action in _of(first, Speak)] == [spoken]
    assert duplicate == []
    assert core.active_request_id == ""


def test_domain_corrections_rewrite_homophones_before_the_agent():
    core = _core(text_corrections=(("夹爪", ("家长", "家找")),))

    (request,) = _of(core.on_transcript("请打开家长", now=0.0), PublishRequest)

    assert request.payload["text"] == "请打开夹爪"
    with pytest.raises(ValueError, match="text_corrections"):
        VoiceBridgeConfig(text_corrections=(("夹爪", ()),))


def test_bounded_id_sets_forget_oldest_without_losing_the_active_request():
    """A long-running channel must not grow its id sets without bound."""
    from ibrobot_agent.voice_bridge import _REMEMBERED_REQUEST_IDS, _BoundedIdSet

    ids = _BoundedIdSet(limit=3)
    for value in ("a", "b", "c", "d"):
        ids.add(value)

    assert len(ids) == 3
    assert "a" not in ids and "d" in ids
    assert _REMEMBERED_REQUEST_IDS >= 2  # room for more than the single in-flight request


def test_failed_terminal_is_spoken_without_technical_english():
    core = _core()
    core.on_transcript("机械臂点头", now=0.0)

    actions = core.on_event(
        _event("voice-0", "failed", "FAILED", "skill is outside the incubation allowlist: nod_yes"), now=1.0
    )

    (speak,) = _of(actions, Speak)
    assert speak.text == "执行失败，详情见日志。"
    # The technical reason stays available for the operator.
    assert any("allowlist" in log.message for log in _of(actions, Log))


def test_speech_thread_survives_a_tts_error_and_still_acknowledges_later_plans():
    """One raising render must not end the only consumer of the speech queue."""
    import json
    import logging
    import threading

    from ibrobot_agent.voice_bridge import _build_node_class

    node_class = _build_node_class()
    acknowledged = threading.Event()

    class _Publisher:
        def __init__(self, published: threading.Event | None = None) -> None:
            self.messages: list[str] = []
            self._published = published

        def publish(self, message) -> None:
            self.messages.append(message.data)
            if self._published is not None:
                self._published.set()

    class _Speaker:
        def __init__(self) -> None:
            self.attempts: list[str] = []

        def speak(self, text: str) -> bool:
            self.attempts.append(text)
            if len(self.attempts) == 1:
                raise RuntimeError("TTS service died mid-call")
            return True

    node = node_class.__new__(node_class)
    node.get_logger = lambda: logging.getLogger("voice-bridge-test")
    node._mute_asr = True
    node._asr_state = "listening"
    node._deaf_until = 0.0
    node._resume_settle_sec = 0.0
    node._speaker = _Speaker()
    node._control_publisher = _Publisher(acknowledged)
    node._asr_control_publisher = _Publisher()
    node._speech_wakeup = threading.Condition()
    first = Speak("准备执行：点头。", then=(PublishControl({"control_type": "presentation_rendered", "n": 1}),))
    second = Speak("准备执行：摇头。", then=(PublishControl({"control_type": "presentation_rendered", "n": 2}),))
    node._speech_queue = [first, second]

    threading.Thread(target=node._speech_loop, daemon=True).start()

    assert acknowledged.wait(timeout=5.0), "speech thread died on the first render"
    assert node._speaker.attempts == ["准备执行：点头。", "准备执行：摇头。"]
    # The plan whose speech failed is never acknowledged; the next one still is.
    assert [json.loads(message)["n"] for message in node._control_publisher.messages] == [2]
    # ASR is resumed after the failed render too, so the microphone is not left muted.
    assert node._asr_control_publisher.messages == ["stop", "start", "stop", "start"]


@pytest.mark.parametrize("utterance", ["帮我停一下", "别动了谢谢", "快停下来", "STOP please"])
def test_stop_keyword_inside_a_sentence_interrupts_the_active_request(utterance):
    core = _core()
    core.on_transcript("挥挥手", now=0.0)

    actions = core.on_transcript(utterance, now=1.0)

    assert _of(actions, PublishRequest) == []
    assert [action.payload for action in _of(actions, PublishControl)] == [
        {"operation": "stop", "request_id": "voice-0"}
    ]


def test_keyword_sentence_without_anything_to_interrupt_is_still_a_command():
    core = _core()

    (request,) = _of(core.on_transcript("不停地挥手", now=0.0), PublishRequest)
    lone_stop = core.on_transcript("停止", now=1.0)

    assert request.payload["text"] == "不停地挥手"
    # The keyword sentence is a running request now, so a lone stop phrase targets it.
    assert [action.payload for action in _of(lone_stop, PublishControl)] == [
        {"operation": "stop", "request_id": "voice-0"}
    ]
    idle = _core()
    assert _of(idle.on_transcript("停止", now=0.0), PublishRequest) == []


def test_partial_transcript_stops_once_before_the_utterance_ends():
    core = _core()
    core.on_transcript("挥挥手", now=0.0)

    first = core.on_partial("帮我停", now=1.0)
    grown = core.on_partial("帮我停一下", now=1.2)
    unrelated = core.on_partial("今天天气", now=1.3)

    assert [action.payload for action in _of(first, PublishControl)] == [{"operation": "stop", "request_id": "voice-0"}]
    assert grown == [] and unrelated == []
    # The final transcript of the same utterance belongs to that stop: no second stop, no request.
    final = core.on_transcript("帮我停一下", now=2.0)
    assert _of(final, PublishRequest) == [] and _of(final, PublishControl) == []
    # A new utterance may stop again, e.g. when the first stop has not taken effect yet.
    retry = core.on_partial("停", now=3.0)
    assert [action.payload for action in _of(retry, PublishControl)] == [{"operation": "stop", "request_id": "voice-0"}]


def test_final_of_a_partial_stop_does_not_become_a_command_after_the_request_ends():
    # Seen on the real SO-101: the stop finished the request before the final arrived, so
    # "帮我停一下" was sent to the Agent as a new request and "继续" was then dropped as busy.
    core = _core()
    core.on_transcript("挥手然后点头再摇头", now=0.0)
    core.on_partial("帮我停", now=1.0)
    core.on_event(_event("voice-0", "terminal", "CANCELLED", "skill cancelled"), now=1.1)

    final = core.on_transcript("帮我停一下", now=1.8)
    (resume,) = _of(core.on_transcript("继续", now=3.0), PublishRequest)

    assert _of(final, PublishRequest) == []
    assert resume.payload["text"] == "继续"


def test_partial_without_a_running_request_is_silent():
    assert _core().on_partial("停", now=0.0) == []


def test_request_timeout_stops_the_robot_and_keeps_it_stoppable():
    core = _core(request_timeout_sec=10.0)
    core.on_transcript("挥挥手", now=0.0)

    timed_out = core.tick(11.0)

    assert [action.payload for action in _of(timed_out, PublishControl)] == [
        {"operation": "stop", "request_id": "voice-0"}
    ]
    assert core.active_request_id == ""
    # The channel accepts new speech, and a stop keyword still reaches the timed-out request.
    stop = core.on_transcript("别动", now=12.0)
    assert [action.payload for action in _of(stop, PublishControl)] == [{"operation": "stop", "request_id": "voice-0"}]
    # Its late terminal event is still spoken once and clears the stop target.
    terminal = core.on_event(_event("voice-0", "terminal", "CANCELLED"), now=13.0)
    assert [action.text for action in _of(terminal, Speak)] == ["已停止。"]
    assert _of(core.on_transcript("停止", now=14.0), PublishControl) == []


def test_late_presentation_after_timeout_is_not_acknowledged():
    core = _core(request_timeout_sec=10.0)
    core.on_transcript("点头", now=0.0)
    core.tick(11.0)

    actions = core.on_event(_presentation_event("voice-0"), now=12.0)

    assert _of(actions, Speak) == [] and _of(actions, PublishControl) == []
    finished = _core()
    finished.on_transcript("点头", now=0.0)
    finished.on_event(_event("voice-0", "terminal", "CANCELLED"), now=1.0)
    assert _of(finished.on_event(_presentation_event("voice-0"), now=2.0), PublishControl) == []


def test_continuation_event_is_spoken_without_releasing_the_channel():
    core = _core()
    core.on_transcript("继续", now=0.0)

    actions = core.on_event(
        _event("voice-0", "continuation", "PLANNING", "上次完整完成 1/3 步后停止，现在继续剩余部分"), now=1.0
    )

    assert [action.text for action in _of(actions, Speak)] == ["上次完整完成 1/3 步后停止，现在继续剩余部分"]
    assert core.active_request_id == "voice-0"


@pytest.mark.parametrize(
    ("response", "spoken"),
    [
        (
            {"accepted": False, "state": "BUSY", "reason_code": "BUSY", "message": "robot scope has an active request"},
            VoiceBridgeConfig.busy_message,
        ),
        (
            {
                "accepted": False,
                "state": "UNKNOWN",
                "reason_code": "ROBOT_QUARANTINED",
                "message": "robot scope is quarantined",
            },
            "执行结果未知，请人工检查机器人状态。",
        ),
    ],
)
def test_rejected_admission_is_spoken_in_chinese(response, spoken):
    core = _core()
    core.on_transcript("点头", now=0.0)

    actions = core.on_response({"request_id": "voice-0", **response}, now=1.0)

    assert [action.text for action in _of(actions, Speak)] == [spoken]
