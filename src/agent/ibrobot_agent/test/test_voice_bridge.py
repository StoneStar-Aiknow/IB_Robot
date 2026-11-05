from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

from ibrobot_agent.addressee import Voiceprint
from ibrobot_agent.contracts import ASYNC_PRESENTATION_MODE
from ibrobot_agent.presentation import presentation_digest
from ibrobot_agent.voice_bridge import (
    Log,
    PublishControl,
    PublishRequest,
    Speak,
    VoiceBridgeConfig,
    VoiceBridgeCore,
)

SCOPE = {"channel_id": "voice_asr", "principal_id": "local_operator", "robot_scope": "so101_single_arm"}


def _core(*, checks=None, **overrides) -> VoiceBridgeCore:
    ids = iter(f"voice-{index}" for index in range(100))
    # The launch profiles run with TTS and therefore resolve to auto; voice_confirm is opted into per test.
    config = VoiceBridgeConfig(**{"confirm_mode": "auto", **SCOPE, **overrides})
    return VoiceBridgeCore(config, id_factory=lambda: next(ids), checks=checks)


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


def test_busy_channel_drops_new_utterance_and_briefly_names_current_task():
    core = _core()
    core.on_transcript("挥手", now=0.0)

    actions = core.on_transcript("点头", now=1.0)

    assert _of(actions, PublishRequest) == []
    assert [a.text for a in _of(actions, Speak)] == ["正在做挥手，请稍等。"]
    assert core.active_request_id == "voice-0"


def test_busy_speech_is_rate_limited_and_expires_without_controlling_execution():
    core = _core()
    core.on_transcript("双手交叉放在胸前", now=0.0)
    first = core.on_transcript("你好", now=1.0)
    assert [a.text for a in _of(first, Speak)] == ["正在做双手交叉放在胸前，请稍等。"]
    assert not _of(first, PublishControl) and not _of(first, PublishRequest)
    assert not _of(core.on_transcript("你好", now=2.0), Speak)
    assert not core.speech_is_current(_of(first, Speak)[0], now=7.0)
    later = core.on_transcript("你好", now=12.0)
    assert len(_of(later, Speak)) == 1
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=13.0)
    assert not core.speech_is_current(_of(later, Speak)[0], now=13.1)


def test_acceptance_gets_one_async_receipt_and_never_waits_for_playback():
    core = _core()
    core.on_transcript("双手侧平举", now=0.0)
    response = {"request_id": "voice-0", "accepted": True, "state": "RECEIVED"}
    actions = core.on_response(response, now=0.01)
    assert [a.text for a in _of(actions, Speak)] == ["收到，正在准备。"]
    assert not _of(actions, Speak)[0].then
    assert not core.on_response(response, now=0.02)
    stop = core.on_partial("停", now=0.03)
    assert _of(stop, PublishControl)[0].payload["operation"] == "stop"


def test_delayed_playback_text_and_captured_audio_cannot_trigger_busy_loop():
    core = _core()
    core.on_transcript("双手交叉放在胸前", now=0.0)
    core.note_speech("上次任务还没有完成任何步骤，现在重新执行。", now=1.0)
    core.note_audio(("mic", 10), captured_at=1.2)
    core.note_speech("上次任务还没有完成任何步骤，现在重新执行。", now=5.0, finished=True)
    # ASR final arrives after the old playback+1s window, with a misrecognized first word.
    actions = core.on_transcript("这个任务还没有完成任何步骤现在重新执行", now=6.24)
    assert not _of(actions, Speak) and not _of(actions, PublishRequest)
    actions = core.on_transcript("识别完全错误的内容", now=25.0, segment=("mic", 10))
    assert not _of(actions, Speak) and not _of(actions, PublishRequest)
    stop = core.on_transcript("停", now=25.1, segment=("mic", 10))
    assert _of(stop, PublishControl)[0].payload["operation"] == "stop"


def test_internal_english_summary_is_not_sent_to_chinese_tts():
    core = _core()
    core.on_transcript("双手交叉放在胸前", now=0.0)
    presentation = {
        "summary": "continue_cross_arms_on_chest",
        "steps": [{"skill_name": "free_pose", "target_name": "双手交叉放在胸前"}],
        "execution_mode": ASYNC_PRESENTATION_MODE,
    }
    event = _event(
        "voice-0",
        "presentation",
        "MAY_EXECUTE",
        detail={
            "presentation": presentation,
            "presentation_digest": presentation_digest(presentation),
            "async_display": True,
        },
    )
    actions = core.on_event(event, now=1.0)
    assert [a.text for a in _of(actions, Speak)] == ["正在执行：双手交叉放在胸前。"]


@pytest.mark.parametrize(
    "text",
    [
        "正在执行：free_pose: 双手交叉放在胸前。",
        "正在执行：continue_cross_arms_on_chest。",
        "正在执行：free_pose: 双手侧平举 (remembered)。",
        "做一个V字动作。",
    ],
)
def test_deployed_chinese_tts_never_receives_internal_english(text):
    import re

    from ibrobot_agent.voice_bridge import _tts_text

    spoken = _tts_text(text)
    assert re.search(r"[\u4e00-\u9fff]", spoken)
    assert not re.search("[A-Za-z_]", spoken)


@pytest.mark.parametrize("structured", [False, True])
def test_save_question_is_spoken_only_after_success(structured):
    core = _core()
    core.on_transcript("双手交叉放在胸前", now=0.0)
    detail = {"save_prompt": "刚才这个动作到位吗？可以保存吗？"}
    if structured:
        detail.update(execution_id="pose-execution", gateway_sequence=1)
    actions = core.on_event(_event("voice-0", "terminal", "SUCCEEDED", detail=detail), now=1.0)
    assert [a.text for a in _of(actions, Speak)] == ["执行完成。刚才这个动作到位吗？可以保存吗？"]


def test_auto_mode_executes_without_second_confirmation_once_plan_is_rendered():
    core = _core()
    core.on_transcript("点头", now=0.0)

    actions = core.on_event(_presentation_event("voice-0"), now=1.0)

    assert _of(actions, PublishControl) == []
    (speak,) = _of(actions, Speak)
    (ack,) = speak.then
    assert ack.payload["operation"] == "presentation_rendered"
    assert ack.payload["receipt_token"] == "token-1"


@pytest.mark.parametrize("confirm_mode", ["auto", "voice_confirm"])
def test_async_presentation_is_announced_without_a_receipt_or_a_confirmation_prompt(confirm_mode):
    # Seen on the real SO-101: typed Programs move the robot before any client renders the plan,
    # so "请说确认或取消" would ask for a confirmation that holds nothing back.
    core = _core(confirm_mode=confirm_mode)
    core.on_transcript("点头", now=0.0)
    presentation = {"summary": "nod_yes", "execution_mode": "direct_execute_async_presentation"}
    detail = {"presentation": presentation, "presentation_digest": presentation_digest(presentation)}
    event = _event(
        "voice-0", "presentation", "MAY_EXECUTE", detail={**detail, "receipt_token": "", "async_display": True}
    )

    actions = core.on_event(event, now=1.0)

    assert _of(actions, PublishControl) == []
    (speak,) = _of(actions, Speak)
    assert speak.text == "正在执行：当前动作。" and speak.then == ()
    assert not core.awaiting_confirmation
    assert any("cannot gate" in log.message for log in _of(actions, Log)) == (confirm_mode == "voice_confirm")
    # The robot is moving: a stop word still reaches the request.
    (stop,) = _of(core.on_transcript("停", now=2.0), PublishControl)
    assert stop.payload == {"operation": "stop", "request_id": "voice-0"}


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


@pytest.mark.parametrize("refusal", ["不执行", "我不确定", "别执行了", "先别确认"])
def test_negated_answer_never_acknowledges_the_plan(refusal):
    # Each refusal contains a confirm word (执行/确定/确认) and no cancel word.
    core = _core(confirm_mode="voice_confirm")
    core.on_transcript("点头", now=0.0)
    core.on_event(_presentation_event("voice-0"), now=1.0)

    actions = core.on_transcript(refusal, now=2.0)

    assert _of(actions, PublishControl) == []
    assert [action.text for action in _of(actions, Speak)] == ["请说确认或取消。"]
    assert core.awaiting_confirmation
    # A plain confirmation afterwards still releases the plan.
    (ack,) = _of(core.on_transcript("确认", now=3.0), PublishControl)
    assert ack.payload["operation"] == "presentation_rendered"


def test_cancel_phrase_wins_over_a_confirm_word():
    core = _core(confirm_mode="voice_confirm")
    core.on_transcript("点头", now=0.0)
    core.on_event(_presentation_event("voice-0"), now=1.0)

    (stop,) = _of(core.on_transcript("不要执行", now=2.0), PublishControl)

    assert stop.payload == {"operation": "stop", "request_id": "voice-0"}
    assert not core.awaiting_confirmation


def test_robot_scope_is_required_and_the_unset_confirm_mode_is_the_safe_one():
    with pytest.raises(ValueError, match="robot_scope"):
        VoiceBridgeConfig()
    assert VoiceBridgeConfig(**SCOPE).confirm_mode == "voice_confirm"


def test_busy_message_must_not_stop_the_request_it_is_spoken_during():
    with pytest.raises(ValueError, match="busy_message"):
        VoiceBridgeConfig(**SCOPE, busy_message="正在执行，要中断请说停止。")
    with pytest.raises(ValueError, match="busy_message"):
        VoiceBridgeConfig(**SCOPE, busy_message="请稍等", stop_phrases=("稍等",))


def test_replies_spoken_while_interruptible_carry_no_stop_word():
    """The microphone stays open while a request can be stopped; the bridge must not stop itself."""
    core = _core(request_timeout_sec=10.0)
    assert not core.interruptible
    core.on_transcript("挥挥手", now=0.0)
    assert core.interruptible

    busy = core.on_transcript("点头", now=1.0)
    timed_out = core.tick(11.0)

    assert core.interruptible  # timed out, still the stop target until its terminal event
    spoken = [action.text for action in _of(busy + timed_out, Speak)]
    assert spoken and not any(core.is_stop_text(text) for text in spoken)
    assert core.is_stop_text("帮我停一下") and not core.is_stop_text("准备执行：点头。")


def test_node_parameter_defaults_come_from_the_config_dataclass():
    import json

    from ibrobot_agent.voice_bridge import node_parameter_defaults

    defaults = node_parameter_defaults()
    config = VoiceBridgeConfig(**SCOPE)

    assert defaults["robot_scope"] == ""  # required from launch, never a robot-specific default
    assert defaults["confirm_mode"] == ""  # derived from tts_enabled
    assert defaults["request_timeout_sec"] == config.request_timeout_sec
    assert defaults["busy_message"] == config.busy_message
    assert tuple(json.loads(defaults["stop_phrases_json"])) == config.stop_phrases
    assert tuple(json.loads(defaults["confirm_phrases_json"])) == config.confirm_phrases
    assert json.loads(defaults["text_corrections_json"]) == {}


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
        VoiceBridgeConfig(**SCOPE, text_corrections=(("夹爪", ()),))


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
    assert speak.text == "执行失败，请查看执行记录或联系操作员。"
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
    # Nothing to interrupt: speaking mutes ASR.
    node._core = _core()
    node._core_lock = threading.Lock()
    node._mute_asr = True
    node._asr_state = "listening"
    node._deaf_until = 0.0
    node._echo_until = 0.0
    node._resume_settle_sec = 0.0
    node._speaker = _Speaker()
    node._control_publisher = _Publisher(acknowledged)
    node._asr_control_publisher = _Publisher()
    node._speech_wakeup = threading.Condition()
    first = Speak("准备执行：点头。", then=(PublishControl({"control_type": "presentation_rendered", "n": 1}),))
    second = Speak("准备执行：摇头。", then=(PublishControl({"control_type": "presentation_rendered", "n": 2}),))
    node._speech_queue = [first, second]
    node._speech_closed = False

    speech_thread = threading.Thread(target=node._speech_loop, daemon=True)
    speech_thread.start()

    assert acknowledged.wait(timeout=5.0), "speech thread died on the first render"
    assert node._speaker.attempts == ["准备执行：点头。", "准备执行：摇头。"]
    # The plan whose speech failed is never acknowledged; the next one still is.
    assert [json.loads(message)["n"] for message in node._control_publisher.messages] == [2]
    # ASR is resumed after the failed render too, so the microphone is not left muted.
    assert node._asr_control_publisher.messages == ["stop", "start", "stop", "start"]
    with node._speech_wakeup:
        node._speech_closed = True
        node._speech_wakeup.notify_all()
    speech_thread.join(timeout=1.0)
    assert not speech_thread.is_alive()


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

    first = core.on_partial("停", now=1.0)
    grown = core.on_partial("停下", now=1.2)
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
    core.on_partial("停", now=1.0)
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
        _event("voice-0", "continuation", "PLANNING", "上次任务已完整完成 1/3 步，现在继续剩余部分"), now=1.0
    )

    assert [action.text for action in _of(actions, Speak)] == ["从当前位置继续准备。"]
    assert core.active_request_id == "voice-0"


@pytest.mark.parametrize(
    ("response", "spoken"),
    [
        (
            {"accepted": False, "state": "BUSY", "reason_code": "BUSY", "message": "robot scope has an active request"},
            None,
        ),
        (
            {
                "accepted": False,
                "state": "UNKNOWN",
                "reason_code": "ROBOT_QUARANTINED",
                "message": "robot scope is quarantined",
            },
            "执行结果未知，请操作员确认机器人状态。",
        ),
    ],
)
def test_rejected_admission_is_spoken_in_chinese(response, spoken):
    core = _core()
    core.on_transcript("点头", now=0.0)

    actions = core.on_response({"request_id": "voice-0", **response}, now=1.0)

    assert [action.text for action in _of(actions, Speak)] == ([] if spoken is None else [spoken])


@pytest.mark.parametrize(
    ("configured", "tts_enabled", "resolved"),
    [
        ("", True, "auto"),
        # Without TTS the plan is only logged: an unset mode must not acknowledge it on its own.
        ("", False, "voice_confirm"),
        # auto without TTS stays possible, but only as an explicit choice.
        ("auto", False, "auto"),
        ("voice_confirm", True, "voice_confirm"),
    ],
)
def test_unset_confirm_mode_requires_a_spoken_confirmation_without_tts(configured, tts_enabled, resolved):
    from ibrobot_agent.voice_bridge import resolve_confirm_mode

    assert resolve_confirm_mode(configured, tts_enabled=tts_enabled) == resolved


def test_default_stop_words_are_the_shared_control_vocabulary():
    from embodied_common.control_phrases import STOP_PHRASES

    assert VoiceBridgeConfig(**SCOPE).stop_phrases == STOP_PHRASES


def _speaking_node(core, heard_during_playback):
    """A bridge node whose speaker feeds ``heard_during_playback`` back as ASR finals while it speaks."""
    import logging
    import threading

    from ibrobot_agent.voice_bridge import _build_node_class

    node_class = _build_node_class()

    class _Publisher:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def publish(self, message) -> None:
            self.messages.append(message.data)

    class _Message:
        def __init__(self, data: str) -> None:
            self.data = data

    class _Speaker:
        def speak(self, text: str) -> bool:
            for heard in heard_during_playback:
                node._on_transcript(_Message(heard))
            return True

    node = node_class.__new__(node_class)
    node.get_logger = lambda: logging.getLogger("voice-bridge-test")
    node._core = core
    node._core_lock = threading.Lock()
    node._mute_asr = True
    node._asr_state = "listening"
    node._deaf_until = 0.0
    node._echo_until = 0.0
    node._resume_settle_sec = 30.0
    node._speaker = _Speaker()
    node._request_publisher = _Publisher()
    node._control_publisher = _Publisher()
    node._asr_control_publisher = _Publisher()
    return node


def test_stop_word_is_heard_while_the_bridge_speaks_during_motion():
    import json

    core = _core()
    core.on_transcript("挥挥手", now=0.0)
    # The robot hears its own busy reply and then the user's stop word.
    node = _speaking_node(core, [VoiceBridgeConfig.busy_message, "停"])

    node._render_speech(Speak(VoiceBridgeConfig.busy_message))

    assert node._asr_control_publisher.messages == []  # ASR was never muted
    assert [json.loads(data) for data in node._control_publisher.messages] == [
        {"operation": "stop", "request_id": "voice-0"}
    ]
    assert node._request_publisher.messages == []  # the echo was not routed as a command


def test_terminal_feedback_discards_queued_busy_speech_without_delaying_stop_control():
    import threading

    core = _core()
    core.on_transcript("双手侧平举", now=0.0)
    node = _speaking_node(core, [])
    node._speech_queue = []
    node._speech_wakeup = threading.Condition()
    node._execute([Speak("正在做双手侧平举，请稍等。", kind="busy"), Speak("收到，正在准备。", kind="receipt")])
    # _execute queues TTS, it never invokes the blocking speaker in the input callback.
    node._execute(core.on_partial("停", now=1.0))
    assert node._control_publisher.messages
    node._execute(core.on_event(_event("voice-0", "terminal", "CANCELLED"), now=2.0))
    assert [action.text for action in node._speech_queue] == ["已停止。"]


def test_reply_with_a_stop_word_still_mutes_asr():
    core = _core()
    core.on_transcript("挥挥手", now=0.0)
    node = _speaking_node(core, ["已停止"])

    node._render_speech(Speak("已停止。"))

    assert node._asr_control_publisher.messages == ["stop", "start"]
    assert node._control_publisher.messages == []


def test_reply_preview_is_spoken_once_and_not_repeated_by_the_answer():
    # A streaming planner hands the reply over before the plan is complete; the answer that
    # carries the same text must not make the robot say it twice.
    core = _core()
    core.on_transcript("你好", now=0.0)

    preview = core.on_event(_event("voice-0", "reply_preview", "PLANNING", "你好呀"), now=1.0)
    again = core.on_event(_event("voice-0", "reply_preview", "PLANNING", "你好呀"), now=1.1)
    answer = core.on_event(_event("voice-0", "conversation", "ANSWERED", "你好呀"), now=2.0)

    assert [action.text for action in _of(preview, Speak)] == ["你好呀"]
    assert _of(again, Speak) == []
    assert _of(answer, Speak) == []
    # The answer still released the channel.
    assert _of(core.on_transcript("挥手", now=3.0), PublishRequest)


def test_an_answer_that_differs_from_its_preview_is_still_spoken():
    core = _core()
    core.on_transcript("现在状态怎么样", now=0.0)
    core.on_event(_event("voice-0", "reply_preview", "PLANNING", "我查一下。"), now=1.0)

    answer = core.on_event(_event("voice-0", "read_only", "ANSWERED", "机器人已就绪。"), now=2.0)

    assert [action.text for action in _of(answer, Speak)] == ["机器人已就绪。"]


def test_a_refused_plan_after_its_preview_is_reported_as_a_failure():
    # The documented trade-off: the reply was said before validation, so a refused plan must
    # be followed by the failure rather than by silence.
    core = _core()
    core.on_transcript("挥手", now=0.0)
    core.on_event(_event("voice-0", "reply_preview", "PLANNING", "准备执行：挥手。"), now=1.0)

    failed = core.on_event(_event("voice-0", "failed", "FAILED", "SKILL_SCHEMA_INVALID"), now=2.0)

    assert [action.text for action in _of(failed, Speak)] == ["执行失败，请查看执行记录或联系操作员。"]


def _async_presentation_event(request_id):
    presentation = {"summary": "nod_yes", "execution_mode": "direct_execute_async_presentation"}
    detail = {"presentation": presentation, "presentation_digest": presentation_digest(presentation)}
    return _event(
        request_id, "presentation", "MAY_EXECUTE", detail={**detail, "receipt_token": "", "async_display": True}
    )


def test_a_previewed_typed_program_is_not_announced_a_second_time():
    # Seen in review: preview, then "正在执行：nod_yes。", then "执行完成。" -- the same intent twice,
    # the second one an identifier the speech front end cannot even say.
    core = _core()
    core.on_transcript("点个头", now=0.0)
    core.on_event(_event("voice-0", "reply_preview", "PLANNING", "好的，点个头。"), now=1.0)

    presented = core.on_event(_async_presentation_event("voice-0"), now=1.5)
    done = core.on_event(_event("voice-0", "workflow", "SUCCEEDED", "done"), now=5.0)

    assert _of(presented, Speak) == [] and _of(presented, PublishControl) == []
    assert any("already announced" in log.message for log in _of(presented, Log))
    assert [action.text for action in _of(done, Speak)] == ["执行完成。"]


def test_a_previewed_plan_on_the_receipt_path_is_still_rendered_before_its_receipt():
    # That receipt means the user heard the exact plan; the preview is only the model's text.
    core = _core()
    core.on_transcript("点个头", now=0.0)
    core.on_event(_event("voice-0", "reply_preview", "PLANNING", "好的，点个头。"), now=1.0)

    actions = core.on_event(_presentation_event("voice-0"), now=1.5)

    assert _of(actions, PublishControl) == []
    (speak,) = _of(actions, Speak)
    assert speak.text == "准备执行：当前动作。"
    (ack,) = speak.then
    assert ack.payload["operation"] == "presentation_rendered"


def test_reply_preview_for_a_request_this_channel_is_not_running_is_ignored():
    core = _core()
    core.on_transcript("你好", now=0.0)
    core.on_event(_event("voice-0", "conversation", "ANSWERED", "你好呀"), now=1.0)

    late = core.on_event(_event("voice-0", "reply_preview", "PLANNING", "你好呀"), now=2.0)

    assert _of(late, Speak) == []


def test_reply_preview_after_a_stop_is_not_spoken():
    # The stop is on its way to the Agent, which may publish the preview before it sees it;
    # "好的，我来挥手" followed by "已停止" would contradict the user who just said stop.
    core = _core()
    core.on_transcript("挥手", now=0.0)
    assert _of(core.on_transcript("停止", now=0.5), PublishControl)

    late = core.on_event(_event("voice-0", "reply_preview", "PLANNING", "好的，我来挥手。"), now=0.6)

    assert _of(late, Speak) == []


def test_reply_preview_is_not_spoken_when_replies_are_off():
    core = _core(speak_replies=False)
    core.on_transcript("你好", now=0.0)

    actions = core.on_event(_event("voice-0", "reply_preview", "PLANNING", "你好呀"), now=1.0)

    assert _of(actions, Speak) == []


# --- addressee gate: wake phrase plus the window it opens ------------------------

WAKE = {"wake_phrases": ("灵犀灵犀", "灵犀"), "wake_window_sec": 12.0}


def test_an_utterance_without_the_wake_phrase_is_ignored():
    """The platform arms its VAD once, so unaddressed speech still arrives here."""
    core = _core(**WAKE)

    actions = core.on_transcript("我跟你说昨天那个事", now=1.0)

    assert _of(actions, PublishRequest) == []
    assert any("not addressed to me" in action.message for action in _of(actions, Log))


@pytest.mark.parametrize(
    "heard,expected",
    [
        ("灵犀灵犀，挥个手", "挥个手"),
        ("灵犀灵犀挥个手", "挥个手"),
        ("灵犀挥个手", "挥个手"),
        ("灵犀，你能做什么", "你能做什么"),
    ],
)
def test_every_wake_phrasing_yields_the_bare_command(heard, expected):
    core = _core(**WAKE)

    actions = core.on_transcript(heard, now=1.0)

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == [expected]


def test_a_wake_phrase_in_the_middle_does_not_address_the_robot():
    """Prefix matching is the point: a mention is not an address."""
    core = _core(**WAKE)

    actions = core.on_transcript("昨天我跟灵犀说过这个事", now=1.0)

    assert _of(actions, PublishRequest) == []


def test_a_follow_up_inside_the_window_needs_no_wake_phrase():
    core = _core(**WAKE)
    core.on_transcript("灵犀灵犀挥个手", now=1.0)
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=2.0)

    actions = core.on_transcript("再来一次", now=5.0)

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["再来一次"]


def test_the_window_closes_after_wake_window_sec():
    core = _core(**WAKE)
    core.on_transcript("灵犀灵犀挥个手", now=1.0)
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=2.0)

    actions = core.on_transcript("再来一次", now=1.0 + 12.0 + 0.1)

    assert _of(actions, PublishRequest) == []


def test_ignored_speech_does_not_hold_the_window_open():
    """Only an utterance the robot actually handled may extend the window.

    Otherwise a conversation happening nearby keeps renewing it and the gate
    degrades into always-on.
    """
    core = _core(**WAKE)
    core.on_transcript("灵犀灵犀挥个手", now=1.0)
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=2.0)
    # Unaddressed chatter arrives repeatedly inside the window, on an idle channel.
    for moment in (4.0, 8.0, 11.0):
        core.on_transcript("我跟你说昨天那个事", now=moment)

    actions = core.on_transcript("再来一次", now=13.5)

    assert _of(actions, PublishRequest) == []


def test_the_wake_phrase_alone_is_not_a_command():
    """The wake phrase with nothing after it leaves an empty command."""
    core = _core(**WAKE)

    actions = core.on_transcript("灵犀灵犀", now=1.0)

    assert _of(actions, PublishRequest) == []
    assert any("short utterance" in action.message for action in _of(actions, Log))


def test_the_wake_phrase_opens_the_window_even_with_no_command():
    """Saying only the wake phrase is how a user asks for attention first."""
    core = _core(**WAKE)
    core.on_transcript("灵犀灵犀", now=1.0)

    actions = core.on_transcript("挥个手", now=3.0)

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["挥个手"]


def test_homophones_are_corrected_before_the_gate_sees_them():
    """X-ASR renders the wake word as 林夕/林溪; corrections run first."""
    core = _core(**WAKE, text_corrections=(("灵犀", ("林夕", "林溪")),))

    actions = core.on_transcript("林夕林夕挥个手", now=1.0)

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["挥个手"]


def test_wake_window_sec_must_be_positive():
    with pytest.raises(ValueError, match="wake_window_sec"):
        VoiceBridgeConfig(**{**SCOPE, "confirm_mode": "auto", "wake_window_sec": 0.0})


# --- addressee gate: the anchor-check chain the optional signals plug into ------

#: The source every test segment comes from (``SpeechAudioChunk.source_id``).
SOURCE = "runtime-1"


def _seg(sequence):
    """The speech segment opened by BEGIN chunk ``sequence``."""
    return (SOURCE, sequence)


@dataclass(frozen=True)
class _FakeCheck:
    """A check whose agreement is plain equality, so the chain is testable without a sensor."""

    name: str = "fake"
    max_age_sec: float = 2.0

    def verdict(self, anchor, measurement):
        return anchor == measurement, f"{measurement} is not {anchor}"


def _wake_then(core, measurements, *, wake_at=1.0, then_at=3.0, wake="灵犀灵犀", text="挥个手"):
    """Say the wake phrase (segment 1), then a follow-up (segment 2).

    ``measurements`` are (name, value, at, sequence); each is fed at its time.
    """
    for name, value, at, sequence in measurements:
        if at <= wake_at:
            core.on_measurement(name, value, now=at, segment=_seg(sequence))
    core.on_transcript(wake, now=wake_at, segment=_seg(1))
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=wake_at + 0.1)
    for name, value, at, sequence in measurements:
        if at > wake_at:
            core.on_measurement(name, value, now=at, segment=_seg(sequence))
    return core.on_transcript(text, now=then_at, segment=_seg(2))


def test_every_enabled_check_must_agree():
    core = _core(**WAKE, checks=[_FakeCheck(name="speaker"), _FakeCheck(name="second")])

    agreeing = [("speaker", "alice", 1.0, 1), ("second", "front", 1.0, 1)]
    actions = _wake_then(core, [*agreeing, ("speaker", "alice", 2.9, 2), ("second", "left", 2.9, 2)])

    assert _of(actions, PublishRequest) == []
    assert any("second: left is not front" in action.message for action in _of(actions, Log))


def test_a_missing_measurement_accepts():
    """A sensor that is down must not make the robot deaf; a false reject is the worse failure."""
    core = _core(**WAKE, checks=[_FakeCheck()])

    actions = _wake_then(core, [])

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["挥个手"]


def test_an_anchor_is_dropped_by_the_next_wake_phrase_even_without_a_measurement():
    """Re-waking must not keep judging against the old voice."""
    core = _core(**WAKE, checks=[_FakeCheck()])
    core.on_measurement("fake", "alice", now=1.0, segment=_seg(1))
    core.on_transcript("灵犀灵犀", now=1.0, segment=_seg(1))
    core.on_transcript("灵犀灵犀", now=30.0, segment=_seg(2))
    core.on_measurement("fake", "bob", now=31.0, segment=_seg(3))

    actions = core.on_transcript("挥个手", now=31.0, segment=_seg(3))

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["挥个手"]


def test_the_next_wake_phrase_re_anchors_on_whoever_said_it():
    """Alice opens the window, Bob takes over by naming the robot himself."""
    core = _core(**WAKE, checks=[_FakeCheck()])
    core.on_measurement("fake", "alice", now=1.0, segment=_seg(1))
    core.on_transcript("灵犀灵犀", now=1.0, segment=_seg(1))
    core.on_measurement("fake", "bob", now=3.0, segment=_seg(2))
    core.on_transcript("灵犀灵犀", now=3.0, segment=_seg(2))
    core.on_measurement("fake", "bob", now=4.0, segment=_seg(3))

    actions = core.on_transcript("挥个手", now=4.0, segment=_seg(3))

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["挥个手"]


def test_a_measurement_only_judges_the_transcript_of_its_own_segment():
    """Bob's voiceprint belongs to a segment the ASR dropped, so no transcript of
    it ever comes. Paired by arrival, it would have judged Alice's next one."""
    core = _core(**WAKE, checks=[_FakeCheck()])
    core.on_measurement("fake", "alice", now=1.0, segment=_seg(1))
    core.on_transcript("灵犀灵犀", now=1.0, segment=_seg(1))
    core.on_measurement("fake", "bob", now=2.0, segment=_seg(2))

    actions = core.on_transcript("握个手", now=2.5, segment=_seg(3))

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["握个手"]


def test_a_stop_keyword_is_never_checked_against_the_anchor():
    """Anyone in the room must be able to stop the robot, including a bystander."""
    core = _core(**WAKE, checks=[_FakeCheck()])
    core.on_measurement("fake", "alice", now=1.0, segment=_seg(1))
    core.on_transcript("灵犀灵犀挥个手", now=1.0, segment=_seg(1))
    core.on_measurement("fake", "bob", now=2.0, segment=_seg(2))

    actions = core.on_transcript("停止", now=2.0, segment=_seg(2))

    assert [action.payload for action in _of(actions, PublishControl)] == [
        {"operation": "stop", "request_id": "voice-0"}
    ]


# --- addressee gate: the voiceprint check (stage B) ----------------------------

#: The vector space every voiceprint in these tests comes from.
MODEL = "campplus@test"
#: Two unit vectors far enough apart to be a different speaker at any sane threshold.
ALICE = Voiceprint((1.0, 0.0, 0.0), MODEL)
BOB = Voiceprint((0.0, 1.0, 0.0), MODEL)
SPEAKER = {**WAKE, "speaker_check_enabled": True, "speaker_min_similarity": 0.5, "speaker_max_age_sec": 2.5}


def test_verified_speech_renews_the_window_even_while_the_robot_is_busy():
    core = _core(**SPEAKER)
    core.on_measurement("speaker", ALICE, now=1.0, segment=_seg(1))
    core.on_transcript("灵犀灵犀挥个手", now=1.0, segment=_seg(1))
    for index, moment in enumerate((10.0, 20.0, 30.0), start=2):
        core.on_measurement("speaker", ALICE, now=moment, segment=_seg(index))
        actions = core.on_transcript("继续聊一会儿", now=moment, segment=_seg(index))
        assert not _of(actions, PublishRequest)
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=31.0)
    core.on_measurement("speaker", ALICE, now=40.0, segment=_seg(5))
    actions = core.on_transcript("举个手", now=40.0, segment=_seg(5))
    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["举个手"]


@pytest.mark.parametrize("voice", [BOB, None, Voiceprint(ALICE.vector, "other-model")])
def test_unverified_speech_does_not_renew_the_window_while_busy(voice):
    core = _core(**SPEAKER)
    core.on_measurement("speaker", ALICE, now=1.0, segment=_seg(1))
    core.on_transcript("灵犀灵犀挥个手", now=1.0, segment=_seg(1))
    core.on_measurement("speaker", voice, now=10.0, segment=_seg(2))
    core.on_transcript("继续聊一会儿", now=10.0, segment=_seg(2))
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=11.0)
    core.on_measurement("speaker", ALICE, now=14.0, segment=_seg(3))
    assert not _of(core.on_transcript("举个手", now=14.0, segment=_seg(3)), PublishRequest)


def test_enabling_the_check_makes_the_node_subscribe_to_the_voiceprints():
    assert _core(**SPEAKER).required_measurements() == ("speaker",)
    assert _core(**WAKE).required_measurements() == ()


def test_the_check_and_the_session_trigger_read_segment_tagged_transcripts():
    assert _core(**SPEAKER).config.pairs_by_segment
    assert _core(**SESSION).config.pairs_by_segment
    assert not _core(**WAKE).config.pairs_by_segment


def test_a_follow_up_in_the_wake_window_from_another_voice_is_ignored():
    core = _core(**SPEAKER)

    actions = _wake_then(core, [("speaker", ALICE, 1.0, 1), ("speaker", BOB, 2.9, 2)])

    assert _of(actions, PublishRequest) == []
    assert any("speaker: similarity 0.00 < 0.50" in action.message for action in _of(actions, Log))


def test_a_follow_up_in_the_wake_window_from_the_same_voice_is_forwarded():
    core = _core(**SPEAKER)

    actions = _wake_then(core, [("speaker", ALICE, 1.0, 1), ("speaker", ALICE, 2.9, 2)])

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["挥个手"]


def test_another_voice_can_still_take_over_by_saying_the_wake_phrase():
    core = _core(**SPEAKER)
    core.on_measurement("speaker", ALICE, now=1.0, segment=_seg(1))
    core.on_transcript("灵犀灵犀", now=1.0, segment=_seg(1))
    core.on_measurement("speaker", BOB, now=3.0, segment=_seg(2))

    actions = core.on_transcript("灵犀灵犀挥个手", now=3.0, segment=_seg(2))

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["挥个手"]


def test_the_check_needs_a_wake_phrase_to_anchor_on():
    """Without one, nothing would ever capture an anchor and the check would be a no-op."""
    with pytest.raises(ValueError, match="speaker_check_enabled requires an enabled gate"):
        VoiceBridgeConfig(**{**SCOPE, "confirm_mode": "auto", "speaker_check_enabled": True})


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        pytest.param("speaker_min_similarity", 0.0, "speaker_min_similarity", id="similarity zero"),
        pytest.param("speaker_min_similarity", 1.5, "speaker_min_similarity", id="similarity above one"),
        pytest.param("speaker_max_age_sec", 0.0, "speaker_max_age_sec", id="age zero"),
        pytest.param("speaker_merge_margin", -0.1, "speaker_merge_margin", id="margin negative"),
        pytest.param("speaker_merge_margin", 1.0, "speaker_merge_margin", id="margin one"),
    ],
)
def test_the_speaker_thresholds_are_validated(field, value, match):
    with pytest.raises(ValueError, match=match):
        VoiceBridgeConfig(**{**SCOPE, "confirm_mode": "auto", **WAKE, field: value})


# --- addressee gate: the vendor_session trigger ---------------------------------

#: The X2 source consumes its own wake word, so no phrase list can be set.
SESSION = {"addressee_trigger": "vendor_session", "session_gap_sec": 10.0}


def _open_session(core, text, sequence, now, value="alice", name="fake"):
    """The platform reports its wake word for ``sequence``, then that utterance arrives."""
    core.on_wake(now=now - 1.0, segment=_seg(sequence))
    if name is not None:
        core.on_measurement(name, value, now=now, segment=_seg(sequence))
    actions = core.on_transcript(text, now=now, segment=_seg(sequence))
    for request in _of(actions, PublishRequest):
        core.on_event(_event(request.payload["request_id"], "terminal", "SUCCEEDED"), now=now + 0.5)
    return actions


def _speak(core, text, sequence, now, value=None, name="fake"):
    """A transcript with its measurement, no wake event."""
    if value is not None:
        core.on_measurement(name, value, now=now, segment=_seg(sequence))
    actions = core.on_transcript(text, now=now, segment=_seg(sequence))
    for request in _of(actions, PublishRequest):
        core.on_event(_event(request.payload["request_id"], "terminal", "SUCCEEDED"), now=now + 0.5)
    return actions


def test_the_utterance_a_wake_event_names_opens_the_session_with_no_wake_phrase_in_the_text():
    """Measured on the X2: the source strips the wake word before releasing."""
    core = _core(**SESSION)

    actions = _open_session(core, "挥个手", 1, now=100.0, name=None)

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["挥个手"]


def test_without_a_wake_event_nothing_opens_a_session():
    core = _core(**SESSION)

    actions = core.on_transcript("挥个手", now=100.0, segment=_seg(1))

    assert _of(actions, PublishRequest) == []
    assert any("no wake event opened a session" in action.message for action in _of(actions, Log))


def test_after_a_silence_a_command_needs_the_wake_word_again():
    """The counterexample to the old silence rule: past session_gap_sec the first
    utterance became the session opener, so a bystander's command after a quiet
    spell was accepted and anchored on. The platform releases speech nobody woke
    it for, so silence proves nothing; only a wake event opens a session."""
    core = _core(**SESSION, checks=[_FakeCheck()])
    _open_session(core, "挥个手", 1, now=100.0, value="alice")

    bystander = _speak(core, "举个手", 2, now=130.0, value="bob")
    woken = _open_session(core, "举个手", 3, now=140.0, value="bob")

    assert _of(bystander, PublishRequest) == []
    assert any("no wake event opened a session" in action.message for action in _of(bystander, Log))
    assert [action.payload["text"] for action in _of(woken, PublishRequest)] == ["举个手"]
    assert core._gate.anchors["fake"] == "bob"


def test_a_rejected_utterance_still_keeps_the_session_alive():
    """The source's session is kept alive by any speech in the room. Bob's second
    utterance is 12 s after Alice's, past session_gap_sec, but only 7 s after his
    own first, so the session is still Alice's and he is still checked."""
    core = _core(**SESSION, checks=[_FakeCheck()])
    _open_session(core, "挥个手", 1, now=100.0, value="alice")
    _speak(core, "举个手", 2, now=105.0, value="bob")

    actions = _speak(core, "鼓个掌", 3, now=112.0, value="bob")

    assert _of(actions, PublishRequest) == []
    assert any("fake: bob is not alice" in action.message for action in _of(actions, Log))


def test_the_wake_words_own_residue_segment_does_not_lose_the_wake():
    """The X2 sometimes opens a segment for the wake word's residue alone and
    drops it, so the acknowledgement pairs with a segment that never yields a
    transcript. The next segment of that source is still what it woke."""
    core = _core(**SESSION, checks=[_FakeCheck()])
    core.on_wake(now=99.0, segment=_seg(5))
    core.on_measurement("fake", "alice", now=100.0, segment=_seg(6))

    actions = core.on_transcript("挥个手", now=100.0, segment=_seg(6))

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["挥个手"]
    assert core._gate.anchors["fake"] == "alice"


def test_a_sentence_transcribed_after_someone_else_woke_the_robot_is_not_their_opener():
    """Bob wakes the robot while a sentence of Carol's, begun earlier, is still being
    transcribed. Paired by arrival, Carol's late final would have taken Bob's wake
    and the session. By segment it is judged in the session it was spoken in."""
    core = _core(**SESSION, checks=[_FakeCheck()])
    _open_session(core, "挥个手", 1, now=100.0, value="alice")
    core.on_wake(now=104.0, segment=_seg(4))

    carol = _speak(core, "我跟你说个事", 3, now=104.5, value="carol")
    bob = _speak(core, "鼓个掌", 4, now=106.0, value="bob")

    assert _of(carol, PublishRequest) == []
    assert any("fake: carol is not alice" in action.message for action in _of(carol, Log))
    assert [action.payload["text"] for action in _of(bob, PublishRequest)] == ["鼓个掌"]
    assert core._gate.anchors["fake"] == "bob"


def test_a_wake_whose_utterance_never_came_expires():
    """A wake word with nothing usable after it must not hand the session to
    whoever speaks long afterwards."""
    core = _core(**SESSION, wake_window_sec=5.0)
    core.on_wake(now=100.0, segment=_seg(5))

    actions = core.on_transcript("举个手", now=106.0, segment=_seg(9))

    assert _of(actions, PublishRequest) == []


def test_a_session_opened_by_a_stop_word_still_re_anchors():
    """The stop branch returns before the gate; the new session must not keep the old anchor."""
    core = _core(**SESSION, checks=[_FakeCheck()])
    _open_session(core, "挥个手", 1, now=100.0, value="alice")

    _open_session(core, "停", 2, now=120.0, value="bob")
    actions = _speak(core, "举个手", 3, now=123.0, value="bob")

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["举个手"]


def test_a_session_opened_by_a_confirmation_answer_still_re_anchors():
    core = _core(**SESSION, confirm_mode="voice_confirm", checks=[_FakeCheck()])
    core.on_wake(now=99.0, segment=_seg(1))
    core.on_measurement("fake", "alice", now=100.0, segment=_seg(1))
    core.on_transcript("挥个手", now=100.0, segment=_seg(1))
    core.on_event(_presentation_event("voice-0"), now=100.5)

    # Woken again, the pending plan is answered by whoever opened the new session.
    core.on_wake(now=119.0, segment=_seg(2))
    core.on_measurement("fake", "bob", now=120.0, segment=_seg(2))
    core.on_transcript("取消", now=120.0, segment=_seg(2))
    core.on_event(_event("voice-0", "terminal", "CANCELLED"), now=120.5)
    actions = _speak(core, "举个手", 3, now=123.0, value="bob")

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["举个手"]


def test_the_window_is_the_session_not_wake_window_sec():
    """Under vendor_session wake_window_sec only bounds a pending wake; the session
    lasts while the room keeps talking."""
    core = _core(**SESSION, wake_window_sec=2.0, checks=[_FakeCheck()])
    _open_session(core, "挥个手", 1, now=100.0, value="alice")

    actions = _speak(core, "举个手", 2, now=108.0, value="alice")

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["举个手"]


def test_an_opener_with_no_voiceprint_leaves_the_window_open_until_the_next_wake():
    """The wake-stripped opener is often ~1 s, below the voiceprint floor. No anchor
    forms, and no later utterance is promoted to one: a bystander speaking first
    would otherwise take the session from the user."""
    core = _core(**SESSION, checks=[_FakeCheck()])
    core.on_wake(now=99.0, segment=_seg(1))
    core.on_measurement("fake", None, now=100.0, segment=_seg(1))
    core.on_transcript("挥个手", now=100.0, segment=_seg(1))
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=100.5)

    for sequence, (voice, text, at) in enumerate((("bob", "举个手", 105.0), ("alice", "鼓个掌", 110.0)), start=2):
        actions = _speak(core, text, sequence, now=at, value=voice)
        assert [action.payload["text"] for action in _of(actions, PublishRequest)] == [text]
    assert "fake" not in core._gate.anchors

    _open_session(core, "跳个舞", 4, now=115.0, value="carol")
    assert core._gate.anchors["fake"] == "carol"


def test_a_session_opener_is_routed_before_its_voiceprint_and_anchored_when_it_lands():
    """The opener proves itself, so it never waits; its voiceprint anchors whenever it lands."""
    core = _core(**SESSION, measurement_grace_sec=1.5, checks=[_FakeCheck()])
    core.on_wake(now=99.0, segment=_seg(1))

    assert not core.should_hold("挥个手", now=100.0, segment=_seg(1))
    actions = core.on_transcript("挥个手", now=100.0, segment=_seg(1))
    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["挥个手"]
    core.on_measurement("fake", "alice", now=103.0, segment=_seg(1))
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=103.5)

    rejected = _speak(core, "举个手", 2, now=105.0, value="bob")
    assert any("fake: bob is not alice" in action.message for action in _of(rejected, Log))


def test_another_segments_voiceprint_is_never_taken_for_the_openers():
    core = _core(**SESSION, checks=[_FakeCheck()])
    core.on_wake(now=99.0, segment=_seg(1))
    core.on_transcript("挥个手", now=100.0, segment=_seg(1))

    core.on_measurement("fake", "bob", now=100.5, segment=_seg(2))

    assert "fake" not in core._gate.anchors


def test_only_a_follow_up_compared_against_an_anchor_waits_for_its_measurement():
    core = _core(**SESSION, confirm_mode="voice_confirm", checks=[_FakeCheck()])
    core.on_wake(now=99.0, segment=_seg(1))
    assert not core.should_hold("挥个手", now=100.0, segment=_seg(1)), "a session opener proves itself"
    _open_session(core, "挥个手", 1, now=100.0, value="alice")

    assert core.should_hold("举个手", now=105.0, segment=_seg(2))
    assert not core.should_hold("停", now=105.0, segment=_seg(2)), "a stop never waits"
    core.on_measurement("fake", "bob", now=105.1, segment=_seg(3))
    assert core.should_hold("举个手", now=105.0, segment=_seg(2)), "another segment's voiceprint is not its own"
    core.on_measurement("fake", "alice", now=105.1, segment=_seg(2))
    assert not core.should_hold("举个手", now=105.0, segment=_seg(2)), "its voiceprint has landed"


def test_a_follow_up_waits_for_the_openers_voiceprint_too():
    """Judged before the anchor lands, the follow-up would go through unchecked."""
    core = _core(**SESSION, measurement_grace_sec=1.5, checks=[_FakeCheck()])
    core.on_wake(now=99.0, segment=_seg(1))
    core.on_heard("挥个手", now=100.0, segment=_seg(1))
    core.on_event(_event("voice-0", "terminal", "SUCCEEDED"), now=100.2)
    core.on_measurement("fake", "bob", now=100.5, segment=_seg(2))

    held = core.on_heard("举个手", now=100.6, segment=_seg(2))
    judged = core.on_measurement("fake", "alice", now=100.9, segment=_seg(1))

    assert _of(held, PublishRequest) == []
    assert any("fake: bob is not alice" in action.message for action in _of(judged, Log))


def test_nothing_waits_when_there_is_no_anchor_to_compare_against():
    core = _core(**SESSION, checks=[_FakeCheck()])
    _open_session(core, "挥个手", 1, now=100.0, value=None)

    assert not core.should_hold("举个手", now=105.0, segment=_seg(2))


def test_outside_a_session_a_transcript_waits_for_its_own_wake_event():
    """The wake event travels on another topic and may land after the transcript."""
    core = _core(**SESSION, checks=[_FakeCheck()])

    assert core.should_hold("举个手", now=105.0, segment=_seg(2))
    assert not core.should_hold("停", now=105.0, segment=_seg(2)), "a stop never waits"
    core.on_wake(now=105.1, segment=_seg(3))
    assert not core.should_hold("举个手", now=105.0, segment=_seg(2)), "a later segment was woken instead"


def test_a_wake_event_that_lands_after_its_transcript_still_opens_the_session():
    core = _core(**SESSION, measurement_grace_sec=1.5, checks=[_FakeCheck()])
    core.on_measurement("fake", "bob", now=100.0, segment=_seg(10))

    held = core.on_heard("鼓个掌", now=100.2, segment=_seg(10))
    opened = core.on_wake(now=100.5, segment=_seg(10))

    assert _of(held, PublishRequest) == []
    assert [action.payload["text"] for action in _of(opened, PublishRequest)] == ["鼓个掌"]
    assert core._gate.anchors["fake"] == "bob"


def test_a_wake_event_for_a_segment_already_routed_is_refused_not_passed_on():
    """Bob's transcript waited its grace period for a wake event that came later
    still. Left pending, that wake would have opened the session for whoever
    spoke next -- here Alice, who never said the wake word."""
    core = _core(**SESSION, measurement_grace_sec=1.5, checks=[_FakeCheck()])
    core.on_measurement("fake", "bob", now=100.0, segment=_seg(10))
    core.on_heard("鼓个掌", now=100.2, segment=_seg(10))
    rejected = core.tick(now=101.8)

    late = core.on_wake(now=102.0, segment=_seg(10))
    core.on_measurement("fake", "alice", now=103.0, segment=_seg(20))
    alice = core.on_heard("握个手", now=103.2, segment=_seg(20))
    alice += core.tick(now=104.8)

    assert _of(rejected, PublishRequest) == []
    assert any("no wake event opened a session" in action.message for action in _of(rejected, Log))
    assert any("wake event for segment" in action.message for action in _of(late, Log))
    assert _of(alice, PublishRequest) == []


def test_the_woken_segments_final_opens_the_session_even_after_a_partial_stop():
    """Bob wakes the robot and stops Alice's request from a partial; his final is
    then the stop already sent. It still opens Bob's session and spends the wake:
    otherwise the wake stayed pending and Carol, who never woke the robot, opened
    the session with her next sentence and anchored it on her voice."""
    core = _core(**SESSION, checks=[_FakeCheck()])
    core.on_wake(now=99.0, segment=_seg(1))
    core.on_measurement("fake", "alice", now=100.0, segment=_seg(1))
    core.on_heard("挥个手", now=100.0, segment=_seg(1))
    core.on_wake(now=104.0, segment=_seg(2))
    core.on_measurement("fake", "bob", now=104.8, segment=_seg(2))

    stopped = core.on_partial("停", now=104.5)
    core.on_heard("停一下", now=105.0, segment=_seg(2))
    core.on_event(_event("voice-0", "terminal", "CANCELLED"), now=105.2)
    core.on_measurement("fake", "carol", now=106.0, segment=_seg(3))
    carol = core.on_heard("跳个舞", now=106.2, segment=_seg(3))

    assert [action.payload for action in _of(stopped, PublishControl)] == [
        {"operation": "stop", "request_id": "voice-0"}
    ]
    assert _of(carol, PublishRequest) == []
    assert any("fake: carol is not bob" in action.message for action in _of(carol, Log))


def test_an_empty_final_of_the_woken_segment_still_spends_the_wake():
    core = _core(**SESSION, checks=[_FakeCheck()])
    core.on_wake(now=99.0, segment=_seg(1))
    core.on_measurement("fake", "bob", now=100.0, segment=_seg(1))
    core.on_heard("。", now=100.0, segment=_seg(1))
    core.on_measurement("fake", "carol", now=101.0, segment=_seg(2))

    carol = core.on_heard("跳个舞", now=101.2, segment=_seg(2))

    assert _of(carol, PublishRequest) == []


def test_a_pending_confirmation_answer_never_waits():
    core = _core(**SESSION, confirm_mode="voice_confirm", checks=[_FakeCheck()])
    core.on_wake(now=99.0, segment=_seg(1))
    core.on_measurement("fake", "alice", now=100.0, segment=_seg(1))
    core.on_transcript("挥个手", now=100.0, segment=_seg(1))
    core.on_event(_presentation_event("voice-0"), now=100.5)

    assert not core.should_hold("确认", now=103.0, segment=_seg(2))


def test_vendor_session_must_not_also_set_wake_phrases():
    """Two judgements of the same thing; the phrase list could only reject everything."""
    with pytest.raises(ValueError, match="must not set wake_phrases"):
        VoiceBridgeConfig(**{**SCOPE, "confirm_mode": "auto", **SESSION, "wake_phrases": ("灵犀灵犀",)})


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"addressee_trigger": "whenever"}, "addressee_trigger", id="unknown trigger"),
        pytest.param({"session_gap_sec": 0.0}, "session_gap_sec", id="gap zero"),
    ],
)
def test_the_trigger_configuration_is_validated(overrides, match):
    with pytest.raises(ValueError, match=match):
        VoiceBridgeConfig(**{**SCOPE, "confirm_mode": "auto", **overrides})


def test_measurement_grace_sec_must_not_be_negative():
    with pytest.raises(ValueError, match="measurement_grace_sec"):
        VoiceBridgeConfig(**{**SCOPE, "confirm_mode": "auto", **SESSION, "measurement_grace_sec": -0.1})


# --- the voiceprint anchor accumulates over the session's accepted utterances ---

#: Session trigger plus a real speaker check, as the X2 test profile configures it.
SESSION_SPEAKER = {
    **SESSION,
    "speaker_check_enabled": True,
    "speaker_min_similarity": 0.6,
    "speaker_max_age_sec": 6.0,
}


def _say(core, text, vector, now, *, wake=False):
    """One utterance of the next segment: its voiceprint (or none), then its transcript.

    ``wake`` makes the platform report its wake word for that segment first.
    """
    sequence = getattr(core, "_test_sequence", 0) + 1
    core._test_sequence = sequence
    if wake:
        core.on_wake(now=now - 1.0, segment=_seg(sequence))
    core.on_measurement(
        "speaker", None if vector is None else Voiceprint(vector, MODEL), now=now, segment=_seg(sequence)
    )
    actions = core.on_transcript(text, now=now, segment=_seg(sequence))
    for request in _of(actions, PublishRequest):
        core.on_event(_event(request.payload["request_id"], "terminal", "SUCCEEDED"), now=now + 0.1)
    return actions


def test_an_accepted_follow_up_sharpens_the_anchor():
    """Seen on the X2: the platform cut the anchor sentence to 3 s, and the
    speaker's own second half then scored 0.59 against it and was turned away.
    Folding the accepted first follow-up into the anchor is what lets the later
    utterance through; against the single-utterance anchor it scores 0.50."""
    core = _core(**SESSION_SPEAKER)
    _say(core, "挥个手", (1.0, 0.0, 0.0), now=100.0, wake=True)
    _say(core, "握个手", (0.8, 0.6, 0.0), now=110.0)

    actions = _say(core, "顺便让大家看看", (0.5, math.sqrt(0.75), 0.0), now=120.0)

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["顺便让大家看看"]


def test_a_follow_up_just_over_the_threshold_is_accepted_but_not_merged():
    """Measured on the X2 with two lecture recordings: a 0.66 follow-up folded into
    the anchor lifted another speaker from 0.59 to past the threshold. A barely
    accepted voice is weak evidence and must not move the anchor."""
    core = _core(**SESSION_SPEAKER)
    _say(core, "挥个手", (1.0, 0.0, 0.0), now=100.0, wake=True)

    borderline = _say(core, "握个手", (0.65, math.sqrt(1 - 0.65**2), 0.0), now=110.0)
    other = _say(core, "我插一句", (0.59, 0.5, math.sqrt(1 - 0.59**2 - 0.25)), now=120.0)

    assert [action.payload["text"] for action in _of(borderline, PublishRequest)] == ["握个手"]
    assert core._gate.anchors["speaker"] == Voiceprint((1.0, 0.0, 0.0), MODEL)
    assert _of(other, PublishRequest) == []


def test_without_a_merge_margin_the_borderline_voice_drags_the_anchor():
    """The same sequence with the margin off: the drift the margin exists to stop."""
    core = _core(**SESSION_SPEAKER, speaker_merge_margin=0.0)
    _say(core, "挥个手", (1.0, 0.0, 0.0), now=100.0, wake=True)
    _say(core, "握个手", (0.65, math.sqrt(1 - 0.65**2), 0.0), now=110.0)

    other = _say(core, "我插一句", (0.59, 0.5, math.sqrt(1 - 0.59**2 - 0.25)), now=120.0)

    assert [action.payload["text"] for action in _of(other, PublishRequest)] == ["我插一句"]


def test_a_rejected_voice_never_moves_the_anchor():
    core = _core(**SESSION_SPEAKER)
    _say(core, "挥个手", (1.0, 0.0, 0.0), now=100.0, wake=True)

    rejected = _say(core, "我插一句", (0.0, 1.0, 0.0), now=110.0)
    later = _say(core, "我再插一句", (0.3, 0.954, 0.0), now=120.0)

    assert _of(rejected, PublishRequest) == [] and _of(later, PublishRequest) == []


def test_a_follow_up_too_short_for_a_voiceprint_fails_open_by_default():
    core = _core(**SESSION_SPEAKER)
    _say(core, "你会干什么你能握手吗", (1.0, 0.0, 0.0), now=100.0, wake=True)

    actions = _say(core, "握手握手", None, now=110.0)

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["握手握手"]


def test_speaker_reject_short_turns_away_a_follow_up_with_no_voiceprint():
    """Measured on the X2 with two real people (2026-10-26): the commands they gave
    were about a second of speech, so none had a voiceprint and the second person's
    all went through. With an anchor set, such a follow-up is now rejected."""
    core = _core(**SESSION_SPEAKER, speaker_reject_short=True)
    _say(core, "你会干什么你能握手吗", (1.0, 0.0, 0.0), now=100.0, wake=True)

    actions = _say(core, "握手握手", None, now=110.0)

    assert _of(actions, PublishRequest) == []
    assert any("speaker: too short to compare" in action.message for action in _of(actions, Log))
    assert core._gate.anchors["speaker"] == Voiceprint((1.0, 0.0, 0.0), MODEL)


def test_speaker_reject_short_leaves_the_opener_and_stops_alone():
    """The opener proves itself whatever its length, a short opener still leaves the
    window open until the next wake, and a stop never needs addressing."""
    core = _core(**SESSION_SPEAKER, speaker_reject_short=True)

    opener = _say(core, "握手", None, now=100.0, wake=True)
    unanchored = _say(core, "挥手", None, now=105.0)
    _say(core, "麻烦你现在帮我挥一下手", (1.0, 0.0, 0.0), now=110.0, wake=True)
    short = _say(core, "鼓掌", None, now=115.0)
    stop = _say(core, "停", None, now=120.0)

    assert [action.payload["text"] for action in _of(opener, PublishRequest)] == ["握手"]
    assert [action.payload["text"] for action in _of(unanchored, PublishRequest)] == ["挥手"]
    assert _of(short, PublishRequest) == []
    assert any("stop keyword" in action.message for action in _of(stop, Log))


def test_speaker_reject_short_fails_open_when_no_voiceprint_arrives_at_all():
    """A voiceprint node that is down must not make the robot deaf."""
    core = _core(**SESSION_SPEAKER, speaker_reject_short=True)
    _say(core, "你会干什么你能握手吗", (1.0, 0.0, 0.0), now=100.0, wake=True)

    actions = core.on_transcript("握手握手", now=110.0, segment=_seg(2))

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["握手握手"]
    assert core._gate.anchors["speaker"] == Voiceprint((1.0, 0.0, 0.0), MODEL)


def test_a_voiceprint_from_another_model_is_not_compared_or_merged():
    """Same length, other vector space: the numbers would look reasonable and mean nothing.

    The model_id published with every vector is what tells them apart, so such a
    voiceprint counts as no measurement -- the follow-up fails open, the anchor
    stays as it was -- rather than as a confident judgement against the anchor.
    """
    core = _core(**SESSION_SPEAKER)
    _say(core, "挥个手", (1.0, 0.0, 0.0), now=100.0, wake=True)
    core.on_measurement("speaker", Voiceprint((0.0, 1.0, 0.0), "campplus@another"), now=110.0, segment=_seg(2))

    actions = core.on_transcript("握个手", now=110.0, segment=_seg(2))

    assert [action.payload["text"] for action in _of(actions, PublishRequest)] == ["握个手"]
    assert core._gate.anchors["speaker"] == Voiceprint((1.0, 0.0, 0.0), MODEL)


@dataclass(frozen=True)
class _AgreeingCheck:
    """Accepts everything and defines no merge."""

    name: str = "agreeing"
    max_age_sec: float = 2.0

    def verdict(self, anchor, measurement):
        return True, ""


def test_a_check_without_merge_keeps_the_anchor_the_session_opened_with():
    """Only checks that define merge accumulate; any other anchor stays where
    the session opened."""
    core = _core(**SESSION, checks=[_AgreeingCheck()])
    _open_session(core, "挥个手", 1, now=100.0, value="first", name="agreeing")

    actions = _speak(core, "握个手", 2, now=110.0, value="later", name="agreeing")

    assert _of(actions, PublishRequest)
    assert core._gate.anchors["agreeing"] == "first"


# --- a new speaker naming the robot mid-session takes the window over ---

ALICE_VOICE = (1.0, 0.0, 0.0)
BOB_VOICE = (0.0, 1.0, 0.0)


def test_a_wake_event_hands_the_window_to_whoever_it_woke():
    """Scenario D: A opens the session; B names the robot inside it. The source
    consumes the wake word, so only its wake event says B's utterance is not a
    follow-up; B is accepted and re-anchors, and A is now the one checked."""
    core = _core(**SESSION_SPEAKER)
    _say(core, "挥个手", ALICE_VOICE, now=100.0, wake=True)

    taken_over = _say(core, "鼓个掌", BOB_VOICE, now=108.0, wake=True)
    alice_again = _say(core, "握个手", ALICE_VOICE, now=115.0)
    bob_again = _say(core, "挥个手", BOB_VOICE, now=122.0)

    assert [a.payload["text"] for a in _of(taken_over, PublishRequest)] == ["鼓个掌"]
    assert _of(alice_again, PublishRequest) == []
    assert [a.payload["text"] for a in _of(bob_again, PublishRequest)] == ["挥个手"]


def test_without_a_wake_event_the_new_voice_is_still_an_intruder():
    """The control for scenario D: the takeover rests on the wake event, not on a looser check."""
    core = _core(**SESSION_SPEAKER)
    _say(core, "挥个手", ALICE_VOICE, now=100.0, wake=True)

    actions = _say(core, "鼓个掌", BOB_VOICE, now=108.0)

    assert _of(actions, PublishRequest) == []


def test_a_wake_event_is_not_read_under_the_wake_phrase_trigger():
    """There the wake word is in the text; the event must not let an unaddressed utterance through."""
    core = _core(**WAKE)
    core.on_wake(now=99.0, segment=_seg(1))

    actions = core.on_transcript("我跟你说昨天那个事", now=100.0, segment=_seg(1))

    assert _of(actions, PublishRequest) == []


# --- transcripts and voiceprints are paired by segment, held ones keep their place ---


def _requests(actions):
    return [action.payload["text"] for action in _of(actions, PublishRequest)]


def _finish(core, actions, now):
    for request in _of(actions, PublishRequest):
        core.on_event(_event(request.payload["request_id"], "terminal", "SUCCEEDED"), now=now)


def test_a_stop_leaves_no_voiceprint_for_the_next_utterance():
    """Paired by arrival, the stop's voiceprint stayed behind: a bystander whose
    transcript beat its own voiceprint was judged by the user's and went out."""
    core = _core(**SESSION_SPEAKER, measurement_grace_sec=1.5)
    _say(core, "挥个手", ALICE_VOICE, now=100.0, wake=True)
    core.on_measurement("speaker", Voiceprint(ALICE_VOICE, MODEL), now=103.0, segment=_seg(2))
    core.on_heard("停", now=103.5, segment=_seg(2))

    heard = core.on_heard("鼓个掌", now=104.5, segment=_seg(3))
    judged = core.on_measurement("speaker", Voiceprint(BOB_VOICE, MODEL), now=104.7, segment=_seg(3))

    assert _requests(heard) == [], "the bystander waits for its own voiceprint"
    assert _requests(judged) == []
    assert any("not addressed to me" in log.message for log in _of(judged, Log))


def test_a_lost_voiceprint_costs_only_its_own_utterance():
    """Under arrival pairing a lost voiceprint was either taken from the next
    utterance or left one of its own behind. By segment, the transcript whose
    voiceprint was lost fails open after the grace period, and the bystander
    after it is still judged by the bystander's own voice."""
    core = _core(**SESSION_SPEAKER, measurement_grace_sec=1.5)
    _say(core, "挥个手", ALICE_VOICE, now=100.0, wake=True)
    core.on_heard("握个手", now=105.0, segment=_seg(2))
    released = core.tick(now=106.6)
    _finish(core, released, now=106.7)

    core.on_measurement("speaker", Voiceprint(BOB_VOICE, MODEL), now=107.0, segment=_seg(3))
    heard = core.on_heard("鼓个掌", now=107.2, segment=_seg(3))

    assert _requests(released) == ["握个手"], "fail-open after the grace period"
    assert _requests(heard) == []
    assert any("not addressed to me" in log.message for log in _of(heard, Log))


def test_a_voiceprint_whose_transcript_never_came_judges_nobody():
    """The ASR dropped Bob's segment; its voiceprint still arrived. Alice's next
    transcript waits for her own voiceprint instead of taking his."""
    core = _core(**SESSION_SPEAKER, measurement_grace_sec=1.5)
    _say(core, "挥个手", ALICE_VOICE, now=100.0, wake=True)
    core.on_measurement("speaker", Voiceprint(BOB_VOICE, MODEL), now=104.0, segment=_seg(2))

    heard = core.on_heard("握个手", now=105.0, segment=_seg(3))
    judged = core.on_measurement("speaker", Voiceprint(ALICE_VOICE, MODEL), now=105.2, segment=_seg(3))

    assert _requests(heard) == [] and _requests(judged) == ["握个手"]


def test_a_held_follow_up_does_not_take_a_wake_word_said_while_it_waited():
    """The previous user's follow-up is held for its voiceprint when a new speaker
    wakes the robot. The wake names the new speaker's segment, so the held one is
    judged in the old session and the new speaker opens the next."""
    core = _core(**SESSION_SPEAKER, measurement_grace_sec=1.5)
    _say(core, "挥个手", ALICE_VOICE, now=100.0, wake=True)
    held = core.on_heard("握个手", now=108.0, segment=_seg(2))
    core.on_wake(now=108.5, segment=_seg(3))
    released = core.on_measurement("speaker", Voiceprint(ALICE_VOICE, MODEL), now=108.8, segment=_seg(2))
    _finish(core, released, now=108.9)

    core.on_measurement("speaker", Voiceprint(BOB_VOICE, MODEL), now=110.0, segment=_seg(3))
    taken_over = core.on_heard("鼓个掌", now=110.2, segment=_seg(3))

    assert _requests(held) == [] and _requests(released) == ["握个手"]
    assert _requests(taken_over) == ["鼓个掌"]


def test_a_held_bystander_is_still_judged_when_a_wake_word_comes_while_it_waits():
    """The other half of the same race: what was held is checked against the
    session it was heard in, not taken as the opener of the next one."""
    core = _core(**SESSION_SPEAKER, measurement_grace_sec=1.5)
    _say(core, "挥个手", ALICE_VOICE, now=100.0, wake=True)
    core.on_heard("鼓个掌", now=108.0, segment=_seg(2))
    core.on_wake(now=108.5, segment=_seg(3))
    released = core.on_measurement("speaker", Voiceprint(BOB_VOICE, MODEL), now=108.8, segment=_seg(2))

    assert _requests(released) == []
    assert any("not addressed to me" in log.message for log in _of(released, Log))


def test_a_stop_cancels_the_transcripts_held_for_their_voiceprint():
    """ "挥手" is waiting for its voiceprint when the user says "停": it must not
    go out after the stop, even with nothing running for the stop to reach."""
    core = _core(**SESSION_SPEAKER, measurement_grace_sec=1.5)
    _say(core, "挥个手", ALICE_VOICE, now=100.0, wake=True)
    core.on_heard("挥挥手", now=105.0, segment=_seg(2))

    stopped = core.on_heard("停一下", now=105.3, segment=_seg(3))
    later = [
        *core.on_measurement("speaker", Voiceprint(ALICE_VOICE, MODEL), now=105.6, segment=_seg(2)),
        *core.tick(now=107.0),
    ]

    assert _requests(stopped) == [], "a stop that cancelled something is not a command"
    assert any("cancelled '挥挥手'" in log.message for log in _of(stopped, Log))
    assert _requests(later) == []


def test_the_stops_own_voiceprint_does_not_release_the_held_command():
    """The common timing on the X2: the stop's voiceprint lands before its
    transcript. Paired by arrival it released the held command ahead of the stop;
    by segment it is only the stop's, and the stop then cancels the command."""
    core = _core(**SESSION_SPEAKER, measurement_grace_sec=1.5)
    _say(core, "挥个手", ALICE_VOICE, now=100.0, wake=True)
    core.on_heard("挥挥手", now=105.0, segment=_seg(2))

    early = core.on_measurement("speaker", Voiceprint(BOB_VOICE, MODEL), now=105.66, segment=_seg(3))
    stopped = core.on_heard("停下来别动", now=105.8, segment=_seg(3))
    later = [
        *core.on_measurement("speaker", Voiceprint(ALICE_VOICE, MODEL), now=106.0, segment=_seg(2)),
        *core.tick(now=107.0),
    ]

    assert _requests(early) == [] and _requests(stopped) == [] and _requests(later) == []
    assert any("cancelled '挥挥手'" in log.message for log in _of(stopped, Log))


def test_a_stop_in_a_partial_cancels_the_held_transcripts_and_owns_its_final():
    core = _core(**SESSION_SPEAKER, measurement_grace_sec=1.5)
    _say(core, "挥个手", ALICE_VOICE, now=100.0, wake=True)
    core.on_heard("挥挥手", now=105.0, segment=_seg(2))

    partial = core.on_partial("停", now=105.2)
    final = core.on_heard("停一下", now=105.8, segment=_seg(3))
    later = [
        *core.on_measurement("speaker", Voiceprint(ALICE_VOICE, MODEL), now=105.9, segment=_seg(2)),
        *core.tick(now=107.0),
    ]

    assert any("cancelled '挥挥手'" in log.message for log in _of(partial, Log))
    assert _requests(final) == [] and _requests(later) == []


def test_the_node_routes_a_held_follow_up_the_moment_its_voiceprint_lands():
    """No fixed grace period: the opener goes straight out, a compared follow-up waits
    only until its own voiceprint (or the notice that it has none) arrives."""
    import json
    import logging
    import threading
    import time

    from ibrobot_agent.voice_bridge import _build_node_class

    node_class = _build_node_class()

    class _Publisher:
        def __init__(self) -> None:
            self.messages: list[str] = []

        def publish(self, message) -> None:
            self.messages.append(message.data)

    class _Transcript:
        def __init__(self, text: str, sequence: int) -> None:
            self.text, self.source_id, self.segment_sequence = text, SOURCE, sequence

    class _Embedding:
        def __init__(self, vector, sequence: int, model_id=MODEL) -> None:
            self.embedding = list(vector)
            self.model_id = model_id
            self.source_id, self.segment_sequence = SOURCE, sequence

    class _Wake:
        def __init__(self, sequence: int) -> None:
            self.source_id, self.segment_sequence = SOURCE, sequence

    node = node_class.__new__(node_class)
    node.get_logger = lambda: logging.getLogger("voice-bridge-test")
    # A grace period far beyond the test: only the voiceprint may release it.
    node._core = _core(**SESSION_SPEAKER, measurement_grace_sec=30.0)
    node._core_lock = threading.Lock()
    node._deaf_until = node._echo_until = 0.0
    node._request_publisher = _Publisher()
    node._control_publisher = _Publisher()
    node._speech_queue = []
    node._speech_wakeup = threading.Condition()
    requests = lambda: [json.loads(data)["text"] for data in node._request_publisher.messages]  # noqa: E731

    def finish_last() -> None:
        request_id = json.loads(node._request_publisher.messages[-1])["request_id"]
        node._core.on_event(_event(request_id, "terminal", "SUCCEEDED"), now=time.monotonic())

    node._on_wake_event(_Wake(1))
    node._on_segment_transcript(_Transcript("挥个手", 1))
    assert requests() == ["挥个手"], "the session opener is never held"
    node._on_speaker_embedding(_Embedding((1.0, 0.0, 0.0), 1))
    finish_last()

    node._on_segment_transcript(_Transcript("握个手", 2))
    assert requests() == ["挥个手"], "the follow-up waits for its voiceprint"
    node._on_speaker_embedding(_Embedding((0.0, 1.0, 0.0), 9))
    assert requests() == ["挥个手"], "another segment's voiceprint does not release it"
    node._on_speaker_embedding(_Embedding((1.0, 0.0, 0.0), 2))
    assert requests() == ["挥个手", "握个手"]
    finish_last()

    node._on_segment_transcript(_Transcript("鼓个掌", 3))
    node._on_speaker_embedding(_Embedding((), 3))
    assert requests() == ["挥个手", "握个手", "鼓个掌"], "a known-missing voiceprint releases at once"
