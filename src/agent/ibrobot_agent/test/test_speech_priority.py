"""Priority scheduling and scoped Agent feedback contracts."""

import itertools
import threading

import pytest

from ibrobot_agent.speech_pipeline import SpeechPipeline
from ibrobot_agent.voice_bridge import Speak, VoiceBridgeConfig, VoiceBridgeCore


def test_high_priority_stops_audio_and_stale_stop_never_hits_new_audio():
    changed = threading.Condition()
    started, released, stopping, stop_done = (threading.Event() for _ in range(4))
    played, completed, stops, errors = [], [], [], []

    def play(action, audio):
        with changed:
            played.append(action.text)
            changed.notify_all()
        if action.text == "low":
            started.set()
            assert released.wait(3)
        return True

    def stop():
        stops.append(tuple(played))
        stopping.set()
        released.set()
        assert stop_done.wait(3)

    pipeline = SpeechPipeline(
        synthesize=lambda action: action.text,
        play=play,
        stop=stop,
        completed=lambda action: completed.append(action.text),
    )
    try:
        assert pipeline.submit(Speak("low", request_id="r", priority=10))
        assert started.wait(3)
        assert pipeline.submit(Speak("retry", request_id="r", priority=10))

        def submit():
            try:
                assert pipeline.submit(Speak("protection", request_id="r", priority=40, event_key="unknown"))
            except Exception as error:
                errors.append(error)

        thread = threading.Thread(target=submit)
        thread.start()
        assert stopping.wait(3)
        assert played == ["low"]
        stop_done.set()
        thread.join(3)
        assert not thread.is_alive() and not errors
        with changed:
            assert changed.wait_for(lambda: "protection" in played, 3)
        assert played == ["low", "protection"]
        assert "low" not in completed
        assert stops and all(snapshot == ("low",) for snapshot in stops)
    finally:
        stop_done.set()
        released.set()
        pipeline.close()


def test_equal_priority_and_duplicate_do_not_interrupt():
    started, release, second = threading.Event(), threading.Event(), threading.Event()
    stops = []

    def play(action, audio):
        if action.text == "first":
            started.set()
            assert release.wait(3)
        else:
            second.set()
        return True

    pipeline = SpeechPipeline(synthesize=lambda action: action.text, play=play, stop=lambda: stops.append(1))
    try:
        first = Speak("first", priority=30, event_key="guide", request_id="r")
        assert pipeline.submit(first) and started.wait(3)
        assert pipeline.submit(Speak("second", priority=30, event_key="timeup", request_id="r"))
        assert not pipeline.submit(first)
        assert not stops and not second.is_set()
        release.set()
        assert second.wait(3)
    finally:
        release.set()
        pipeline.close()


def test_expired_event_cannot_clear_current_question():
    started, release = threading.Event(), threading.Event()
    stops = []

    def play(action, audio):
        started.set()
        assert release.wait(3)
        return True

    pipeline = SpeechPipeline(
        synthesize=lambda action: action.text,
        play=play,
        stop=lambda: stops.append(1),
        is_current=lambda action: action.text != "expired",
    )
    try:
        assert pipeline.submit(Speak("question", priority=10)) and started.wait(3)
        assert not pipeline.submit(Speak("expired", priority=30))
        pipeline.preempt(Speak("expired", priority=40))
        assert not stops
    finally:
        release.set()
        pipeline.close()


def test_result_only_clears_own_retry_and_preserves_protection():
    release, started = threading.Event(), threading.Event()

    def synthesize(action):
        if action.text == "protection":
            started.set()
            assert release.wait(3)
        return action.text

    pipeline = SpeechPipeline(synthesize=synthesize, play=lambda action, audio: True)
    try:
        assert pipeline.submit(Speak("protection", priority=40, request_id="other"))
        assert started.wait(3)
        assert pipeline.submit(Speak("other retry", priority=10, request_id="other"))
        assert pipeline.submit(Speak("own retry", priority=10, request_id="r"))
        assert pipeline.submit(Speak("result", priority=20, request_id="r", terminal=True, kind="terminal"))
        with pipeline._condition:
            names = [entry.action.text for entry in pipeline._entries]
        assert "protection" in names and "other retry" in names and "own retry" not in names
    finally:
        release.set()
        pipeline.close()


SCOPE = {"channel_id": "voice_asr", "principal_id": "local_operator", "robot_scope": "interaction_mock"}


def core():
    ids = itertools.count()
    bridge = VoiceBridgeCore(VoiceBridgeConfig(**SCOPE, confirm_mode="auto"), id_factory=lambda: f"r{next(ids)}")
    bridge.on_heard("开始任务", 0)
    return bridge


def event(sequence, code="", *, state="RUNNING", skill="mirror_start", round_id=0, recovery="NOT_NEEDED"):
    return {
        "request_key": {**SCOPE, "request_id": "r0"},
        "state": state,
        "sequence": sequence,
        "event_type": "execution_progress" if state == "RUNNING" else "terminal",
        "detail": {
            "execution_id": "e",
            "gateway_sequence": sequence,
            "current_skill": skill,
            "execution_detail": {
                "schema_version": 1,
                "phase": "approach" if skill == "pick_object" else "MIRRORING",
                "phase_state": "RUNNING" if state == "RUNNING" else "SUCCEEDED",
                "failed_phase": "",
                "reason_code": code,
                "retry_count": round_id,
                "recovery_state": recovery,
                "suggested_action": "NONE",
            },
        },
    }


def spoken(actions):
    return [action for action in actions if isinstance(action, Speak)]


def test_agent_event_mirror_dedupe_expiry_and_same_session_rounds():
    bridge = core()
    warning = spoken(bridge.on_event(event(1, "MIRROR_LAST_TEN_SECONDS"), 50))[0]
    assert warning.priority == 30 and warning.expires_at == 60
    assert bridge.speech_is_current(warning, 59.9)
    assert not bridge.speech_is_current(warning, 60)
    assert not spoken(bridge.on_event(event(2, "MIRROR_LAST_TEN_SECONDS"), 51))
    done = spoken(bridge.on_event(event(3, "MIRROR_TIME_UP"), 60))[0]
    assert done.priority == 30 and bridge.speech_is_current(done, 60)
    assert not bridge.speech_is_current(warning, 59)
    assert not spoken(bridge.on_event(event(4, "MIRROR_LAST_TEN_SECONDS"), 61))
    session = bridge._state.session_id
    bridge.on_event(event(5, "MIRROR_READY", round_id=1), 65)
    assert bridge._state.session_id == session
    assert not bridge.speech_is_current(done, 65)


@pytest.mark.parametrize("state,recovery", [("UNKNOWN", "NOT_NEEDED"), ("SUCCEEDED", "FAILED"), ("FAILED", "UNKNOWN")])
def test_grasp_unsafe_facts_override_success_and_preempt(state, recovery):
    bridge = core()
    speech = spoken(bridge.on_event(event(1, state=state, skill="pick_object", recovery=recovery), 1))[0]
    assert speech.priority == 40
    assert "检查" in speech.text and "拿起来" not in speech.text


def test_grasp_retry_is_low_and_result_invalidates_it():
    bridge = core()
    retry = spoken(bridge.on_event(event(1, skill="pick_object", round_id=1), 1))[0]
    assert retry.priority == 10
    result = spoken(bridge.on_event(event(2, state="SUCCEEDED", skill="pick_object", round_id=1), 2))[0]
    assert result.priority == 20
    assert not bridge.speech_is_current(retry, 2)


def test_gift_e06_maps_to_protection_and_retry_stays_low():
    from ibrobot_agent.contracts import ASYNC_PRESENTATION_MODE
    from ibrobot_agent.presentation import presentation_digest

    bridge = core()
    presentation = {
        "steps": [
            {"skill_name": "pick_object", "target_name": "a lemon"},
            {"skill_name": "place_in_container", "target_name": "a lemon"},
        ],
        "execution_mode": ASYNC_PRESENTATION_MODE,
    }
    payload = event(1)
    payload["event_type"] = "presentation"
    payload["detail"] = {"presentation": presentation, "presentation_digest": presentation_digest(presentation)}
    bridge.on_event(payload, 0.1)
    retry = event(2, skill="pick_object", round_id=1)
    retry["detail"].update(workflow_step_index=0, state="RUNNING")
    assert spoken(bridge.on_event(retry, 1))[0].priority == 10
    unknown = event(3, skill="pick_object", round_id=1, recovery="UNKNOWN")
    unknown["detail"].update(workflow_step_index=0, state="RUNNING")
    speech = spoken(bridge.on_event(unknown, 2))[0]
    assert speech.priority == 40 and speech.notice.token.startswith("E06:")


def test_mirror_protection_never_claims_pause_and_unsafe_motion_requires_event():
    bridge = core()
    protection = spoken(bridge.on_event(event(1, "MIRROR_PROTECTION"), 1))[0]
    assert protection.priority == 40 and "已暂停" not in protection.text
    bridge = core()
    assert not spoken(bridge.on_event(event(1, "UNRELATED"), 1))
    warning = spoken(bridge.on_event(event(2, "MIRROR_UNSAFE_MOTION"), 2))[0]
    assert warning.priority == 40 and warning.text == "动作再小一点，动作太危险了，我怕疼"


def test_knowledge_answer_stays_low_priority():
    bridge = core()
    payload = event(1, state="ANSWERED")
    payload.update(event_type="conversation", detail={}, user_message="这是一个知识回答。")
    assert spoken(bridge.on_event(payload, 1))[0].priority == 10
