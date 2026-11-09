"""Voice interaction boundaries, with local evidence and isolated ROS transports."""

from __future__ import annotations

import itertools
import os
import threading
import time

import pytest

from ibrobot_agent.contracts import ASYNC_PRESENTATION_MODE
from ibrobot_agent.presentation import presentation_digest
from ibrobot_agent.voice_bridge import (
    CancelSpeech,
    PublishControl,
    PublishRequest,
    Speak,
    VoiceBridgeConfig,
    VoiceBridgeCore,
    _build_node_class,
)

SCOPE = {"channel_id": "voice_asr", "principal_id": "local_operator", "robot_scope": "interaction_mock"}
SAFE_CANCEL = {
    "schema_version": 1,
    "phase": "cancel",
    "phase_state": "CANCELED",
    "reason_code": "SKILL_CANCELLED",
    "recovery_state": "NOT_NEEDED",
}


def core(**overrides):
    ids = itertools.count()
    return VoiceBridgeCore(
        VoiceBridgeConfig(**{**SCOPE, "confirm_mode": "auto", **overrides}),
        id_factory=lambda: f"r{next(ids)}",
    )


def of(actions, kind):
    return [action for action in actions if isinstance(action, kind)]


def event(request="r0", *, state="ACTIVE", kind="execution_progress", detail=None, sequence=1):
    return {
        "request_key": {**SCOPE, "request_id": request},
        "event_type": kind,
        "state": state,
        "sequence": sequence,
        "detail": detail or {},
    }


def present(bridge, steps, now=0.1, request="r0", valid=True):
    presentation = {"steps": steps, "summary": "当前任务", "execution_mode": ASYNC_PRESENTATION_MODE}
    digest = presentation_digest(presentation) if valid else "tampered"
    return bridge.on_event(
        event(request, kind="presentation", detail={"presentation": presentation, "presentation_digest": digest}), now
    )


def intro(bridge):
    bridge.on_heard("详细介绍一下", 0)
    present(bridge, [{"skill_name": "introduce_in_detail"}])


def terminal(bridge, state="CANCELLED", facts=None, now=2, request="r0"):
    return bridge.on_event(event(request, state=state, kind="terminal", detail={"execution_detail": facts or {}}), now)


def gift_detail(sequence=1, skill="pick_object", index=0, state="RUNNING", **facts):
    return {
        "execution_id": "e0",
        "gateway_sequence": sequence,
        "workflow_step_index": index,
        "current_skill": skill,
        "state": state,
        "execution_detail": {
            "schema_version": 1,
            "phase": "approach",
            "phase_state": "RUNNING",
            "recovery_state": "NOT_NEEDED",
            "retry_count": 0,
            **facts,
        },
    }


@pytest.mark.parametrize("phrase", ["不用说了", "别讲了", "停止播报", "别说了", "不用讲了"])
def test_speech_only_interrupt_keeps_business_request_running(phrase):
    bridge = core()
    bridge.on_heard("帮我拿柠檬", 0)
    old = of(bridge.on_heard("拿苹果", 0.2), Speak)[0]
    actions = bridge.on_heard(phrase, 1)
    assert of(actions, CancelSpeech)
    assert not of(actions, PublishControl) and not of(actions, PublishRequest)
    assert bridge.active_request_id == "r0"
    assert not bridge.speech_is_current(old, 1)


@pytest.mark.parametrize("phrase", ["停", "停止", "停一下"])
def test_ambiguous_stop_uses_task_control(phrase):
    bridge = core()
    bridge.on_heard("帮我拿柠檬", 0)
    actions = bridge.on_heard(phrase, 1)
    assert of(actions, PublishControl)[0].payload == {"operation": "stop", "request_id": "r0"}
    assert of(actions, CancelSpeech)


def test_new_choice_during_business_pick_never_automatically_stops_or_regrabs():
    bridge = core()
    bridge.on_heard("帮我拿柠檬", 0)
    actions = bridge.on_heard("改成苹果", 1)
    assert not of(actions, PublishControl) and not of(actions, PublishRequest)
    assert bridge.active_request_id == "r0"


@pytest.mark.parametrize(
    "phrase,remainder",
    [
        ("别讲了，我想拿一个柠檬", "我想拿一个柠檬"),
        ("好，别讲了，我想拿一个柠檬", "我想拿一个柠檬"),
        ("不用介绍了，给我柠檬", "给我柠檬"),
    ],
)
def test_intro_interrupt_retains_new_intent_until_safe_root_terminal_and_next_tick(phrase, remainder):
    bridge = core()
    intro(bridge)
    actions = bridge.on_heard(phrase, 1)
    assert of(actions, PublishControl)[0].payload["request_id"] == "r0"
    assert not of(actions, PublishRequest)
    assert not of(bridge.tick(1.9), PublishRequest)
    terminal(bridge, facts=SAFE_CANCEL)
    assert not of(bridge.tick(2.1), PublishRequest)
    request = of(bridge.tick(2.5), PublishRequest)[0].payload
    assert request["text"] == remainder
    assert request["request_id"] == "r1"


@pytest.mark.parametrize(
    "state,facts",
    [
        ("UNKNOWN", SAFE_CANCEL),
        ("FAILED", SAFE_CANCEL),
        ("CANCELLED", {}),
        ("CANCELLED", {**SAFE_CANCEL, "recovery_state": "FAILED"}),
        ("CANCELLED", {**SAFE_CANCEL, "recovery_state": "UNKNOWN"}),
    ],
)
def test_unsafe_intro_terminal_cannot_be_bypassed_by_end_session(state, facts):
    bridge = core()
    intro(bridge)
    bridge.on_heard("别讲了，我想拿一个柠檬", 1)
    terminal(bridge, state, facts)
    assert not of(bridge.tick(20), PublishRequest)
    bridge.end_session(21)
    assert not of(bridge.on_heard("帮我拿苹果", 22), PublishRequest)


def test_plain_intro_speech_cancel_stops_its_started_gesture_too():
    bridge = core()
    intro(bridge)
    actions = bridge.on_heard("不用说了", 1)
    assert of(actions, PublishControl)[0].payload == {"operation": "stop", "request_id": "r0"}
    assert not of(actions, PublishRequest)
    terminal(bridge, facts=SAFE_CANCEL)
    assert not of(bridge.tick(10), PublishRequest)


def test_staff_session_reset_waits_for_motion_safety():
    bridge = core()
    first = of(bridge.on_heard("帮我拿柠檬", 0), PublishRequest)[0].payload
    bridge.on_response({"request_id": "r0", "accepted": True}, 0.1)
    actions = bridge.end_session(1)
    assert of(actions, PublishControl) and of(actions, CancelSpeech)
    assert not of(bridge.on_heard("帮我拿苹果", 1.5), PublishRequest)
    terminal(bridge, facts=SAFE_CANCEL)
    request = of(bridge.on_heard("帮我拿苹果", 3), PublishRequest)[0].payload
    assert request["session_id"] != first["session_id"]
    assert not of(bridge.on_event(event(state="SUCCEEDED", kind="terminal"), 4), Speak)
    assert not of(bridge.tick(5), Speak)


def test_new_request_invalidates_queued_terminal_speech():
    bridge = core()
    bridge.on_heard("点点头", 0)
    old = of(terminal(bridge, "SUCCEEDED"), Speak)[0]
    assert bridge.speech_is_current(old, 2.1)
    bridge.on_heard("挥挥手", 3)
    assert not bridge.speech_is_current(old, 3)


def test_timeout_latch_rejects_new_motion_and_survives_session_reset():
    bridge = core(request_timeout_sec=2)
    bridge.on_heard("帮我拿柠檬", 0)
    assert of(bridge.tick(3), PublishControl)
    bridge.end_session(4)
    assert not of(bridge.on_heard("帮我拿苹果", 5), PublishRequest)
    assert of(bridge.on_heard("停止", 6), PublishControl)[0].payload["request_id"] == "r0"


def test_echo_similarity_remains_but_capture_overlap_does_not_drop_user_intent():
    bridge = core()
    bridge.note_speech("我是具身机器人，可以向你介绍我的能力。", 0)
    bridge.note_audio(("mic", 1), 0.1)
    assert not of(bridge.on_heard("我是具身机器人可以向你介绍我的能力", 1, ("mic", 1)), PublishRequest)
    assert of(bridge.on_heard("我想拿一个柠檬", 1.1, ("mic", 1)), PublishRequest)


def test_deferred_admission_retries_busy_after_agent_finally_release():
    bridge = core()
    intro(bridge)
    bridge.on_heard("别讲了，我想拿一个柠檬", 1)
    terminal(bridge, facts=SAFE_CANCEL)
    assert of(bridge.tick(2.5), PublishRequest)
    bridge.on_response({"request_id": "r1", "accepted": False, "state": "REJECTED", "reason_code": "BUSY"}, 2.6)
    assert not of(bridge.tick(2.8), PublishRequest)
    assert of(bridge.tick(3.1), PublishRequest)[0].payload["text"] == "我想拿一个柠檬"


def test_validated_adjacent_gift_steps_replace_old_single_pick_speech():
    bridge = core()
    bridge.on_heard("拿个苹果放到礼盒里", 0)
    steps = [
        {"skill_name": name, "target_name": "an apple", "container_name": "gift box"}
        for name in ("pick_object", "place_in_container")
    ]
    assert not of(present(bridge, steps), Speak)
    start = of(bridge.on_event(event(detail=gift_detail()), 1), Speak)[0]
    assert start.text == "好，我来拿苹果，再放到指定容器。请稍等。"
    step_success = event(
        detail=gift_detail(2, "place_in_container", 1, "SUCCEEDED", release_status="RELEASED"), sequence=2
    )
    assert not of(bridge.on_event(step_success, 2), Speak)
    result = bridge.on_event(
        event(
            state="SUCCEEDED", kind="terminal", sequence=3, detail=gift_detail(3, "place_in_container", 1, "SUCCEEDED")
        ),
        3,
    )
    spoken = "".join(a.text for a in of(result, Speak))
    assert "领取" not in spoken and "送达" not in spoken and "整体成功" not in spoken
    assert not bridge.speech_is_current(start, 3)


def test_tampered_presentation_never_registers_gift_policy():
    bridge = core()
    bridge.on_heard("拿苹果放到礼盒", 0)
    steps = [
        {"skill_name": name, "target_name": "an apple", "container_name": "gift box"}
        for name in ("pick_object", "place_in_container")
    ]
    present(bridge, steps, valid=False)
    assert not bridge._gift.owns("r0")


def test_late_gift_progress_and_old_session_events_do_not_revive_speech():
    bridge = core()
    bridge.on_heard("拿苹果放到礼盒", 0)
    steps = [
        {"skill_name": name, "target_name": "an apple", "container_name": "gift box"}
        for name in ("pick_object", "place_in_container")
    ]
    present(bridge, steps)
    old = of(bridge.on_event(event(detail=gift_detail()), 1), Speak)[0]
    bridge.end_session(2)
    terminal(bridge, facts=SAFE_CANCEL, now=3)
    bridge.on_heard("挥挥手", 4)
    assert not bridge.speech_is_current(old, 4)
    assert not of(bridge.on_event(event(detail=gift_detail(5), sequence=5), 5), Speak)


def test_two_thread_ros_executor_cancels_playback_without_blocking_inbound():
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID")
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.context import Context
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from std_msgs.msg import String
    from std_srvs.srv import Trigger

    from ibrobot_msgs.msg import SynthesizedAudio
    from ibrobot_msgs.srv import PlayAudioFile, SynthesizeSpeech

    context = Context()
    rclpy.init(context=context)
    server = Node("voice_interaction_mock_audio", context=context)
    group = ReentrantCallbackGroup()
    started = threading.Event()
    ended = threading.Event()
    plays = []
    stops = []

    def synthesize(request, response):
        segment = SynthesizedAudio()
        segment.audio_data = list(b"RIFF-mock")
        response.audio_segments = [segment]
        response.success = True
        return response

    def play(request, response):
        plays.append(request.file_path)
        started.set()
        assert ended.wait(4), "stop callback could not run under two-thread executor"
        response.success = False
        response.error_code = "PLAYBACK_CANCELED"
        return response

    def stop(request, response):
        stops.append(time.monotonic())
        ended.set()
        response.success = True
        return response

    server.create_service(SynthesizeSpeech, "/voice_tts/synthesize", synthesize, callback_group=group)
    server.create_service(PlayAudioFile, "/voice_tts/play", play, callback_group=group)
    server.create_service(Trigger, "/voice_tts/stop", stop, callback_group=group)
    parameters = [
        Parameter("robot_scope", value=SCOPE["robot_scope"]),
        Parameter("tts_enabled", value=True),
        Parameter("confirm_mode", value="auto"),
    ]
    node = _build_node_class()(context=context, parameter_overrides=parameters)
    executor = MultiThreadedExecutor(num_threads=2, context=context)
    executor.add_node(node)
    executor.add_node(server)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        with node._core_lock:
            node._core.on_heard("挥挥手", time.monotonic())
            action = node._core._speech("正在执行挥手，请稍等。")
        node._execute([action])
        assert started.wait(3)
        publisher = server.create_publisher(String, "/ibrobot_agent/voice_control", 10)
        deadline = time.monotonic() + 3
        while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        publisher.publish(String(data="cancel_speech"))
        assert ended.wait(3)
        assert plays and stops
        assert node._core.active_request_id
        # Service callback completion precedes client future completion. Keep the
        # executor alive until both the stop and play responses have drained.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if node._speaker._pending_play is None and node._speaker._pending_stop is None:
                break
            time.sleep(0.01)
        assert node._speaker._pending_play is None
        assert node._speaker._pending_stop is None
        assert not node._speech_transport_failed
    finally:
        ended.set()
        node.destroy_node()
        executor.shutdown(timeout_sec=3)
        spin.join(3)
        server.destroy_node()
        context.shutdown()


def test_stop_confirmation_timeout_emits_one_e06_and_preserves_latch():
    bridge = core()
    intro(bridge)
    bridge.on_heard("不用说了", 1)
    assert not of(bridge.tick(5.9), Speak)
    warning = of(bridge.tick(6.1), Speak)
    assert len(warning) == 1 and "工作人员检查" in warning[0].text
    assert bridge.speech_is_current(warning[0], 6.1)
    assert not of(bridge.tick(7), Speak)
    bridge.end_session(8)
    assert not of(bridge.on_heard("帮我拿一个柠檬", 9), PublishRequest)


def test_confirmation_timeout_says_canceling_until_terminal():
    bridge = core(confirm_mode="voice_confirm", confirm_timeout_sec=1)
    bridge.on_heard("挥挥手", 0)
    presentation = {"steps": [{"skill_name": "wave_hand"}], "summary": "挥手"}
    bridge.on_event(
        event(
            kind="presentation",
            detail={
                "presentation": presentation,
                "presentation_digest": presentation_digest(presentation),
            },
        ),
        0.1,
    )
    actions = bridge.tick(2)
    assert [a.text for a in of(actions, Speak)] == ["确认超时，正在取消。"]
    assert not of(bridge.on_heard("点点头", 3), PublishRequest)


@pytest.mark.parametrize("kind", ["continuation", "reply_preview"])
def test_late_old_request_dialogue_cannot_speak_or_change_new_request_label(kind):
    bridge = core()
    bridge.on_heard("挥挥手", 0)
    terminal(bridge, "SUCCEEDED", now=1)
    bridge.on_heard("帮我拿一个柠檬", 2)
    label = bridge._state.task_label
    actions = bridge.on_event(
        event(
            kind=kind,
            detail={
                "continuation": {
                    "interrupted_step": {"target_name": "苹果"},
                }
            },
        ),
        3,
    )
    assert not of(actions, Speak)
    assert bridge._state.task_label == label
    assert bridge.active_request_id == "r1"


def test_stop_first_response_then_old_play_drains_before_new_play_and_no_late_stop():
    from concurrent.futures import Future
    from types import SimpleNamespace

    from ibrobot_agent.speech_pipeline import SpeechPipeline
    from ibrobot_agent.voice_bridge import _Speaker

    old_play = Future()
    play_sent = threading.Event()
    stop_sent = threading.Event()
    new_play = threading.Event()
    order = []

    class Client:
        def __init__(self, kind):
            self.kind = kind

        def wait_for_service(self, **kwargs):
            return True

        def call_async(self, request):
            if self.kind == "play" and not play_sent.is_set():
                play_sent.set()
                order.append("old_play")
                return old_play
            result = Future()
            result.set_result(SimpleNamespace(success=True))
            if self.kind == "stop":
                order.append("stop")
                stop_sent.set()
            else:
                order.append("new_play")
                new_play.set()
            return result

    class Node:
        def create_client(self, kind, name):
            return Client("stop" if name.endswith("stop") else "play")

        def get_logger(self):
            return SimpleNamespace(error=lambda text: None)

    speaker = _Speaker(Node(), synthesis_service="synthesize", playback_service="play", timeout_sec=1)
    completed = []
    segments = [SimpleNamespace(audio_data=b"RIFF")] * 2
    pipeline = SpeechPipeline(
        synthesize=lambda action: segments,
        play=lambda action, audio: speaker.play(audio),
        is_current=lambda action: True,
        completed=completed.append,
    )
    cancellation_errors = []

    def cancel():
        try:
            pipeline.cancel(stop=speaker.stop, timeout=2)
        except Exception as error:
            cancellation_errors.append(error)

    try:
        pipeline.submit(Speak("旧的播报"))
        assert play_sent.wait(1)
        worker = threading.Thread(target=cancel)
        worker.start()
        assert stop_sent.wait(1)
        assert worker.is_alive(), "stop response must not imply old play response arrived"
        old_play.set_result(SimpleNamespace(success=False, error_code="PLAYBACK_CANCELED"))
        worker.join(2)
        assert not worker.is_alive() and not cancellation_errors
        pipeline.submit(Speak("新的播报"))
        assert new_play.wait(1)
        deadline = time.monotonic() + 1
        while len(completed) < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert [a.text for a in completed] == ["新的播报"]
        assert "stop" not in order[order.index("new_play") + 1 :]
    finally:
        if not old_play.done():
            old_play.set_result(SimpleNamespace(success=False))
        pipeline.close(stop=speaker.close)
