"""Behaviour of runtime_play_audio, alone and with a bound motion, without a ROS graph."""

from threading import Event, Lock, Thread
from types import SimpleNamespace

import pytest
from test_runtime_named_motion_executor import _Client, _Future, _GoalHandle, _motion_result, _MotionHandle, _node

from ibrobot_msgs.action import ExecuteNamedMotion
from skill_library import runtime_named_motion as named_motion
from skill_library import skill_executor_node as executor_module
from skill_library.skill_executor_node import PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, SkillExecutorNode

RESULT = ExecuteNamedMotion.Result
CLIP_TEMPLATE = {"executor": "runtime_play_audio", "binding": {"clip": "self_intro"}, "timeout_sec": 5.0}
WAVE_TEMPLATE = {
    "executor": "runtime_play_audio",
    "binding": {"clip": "self_intro", "motion": "wave"},
    "timeout_sec": 5.0,
}


@pytest.fixture(autouse=True)
def _rclpy_running(monkeypatch):
    monkeypatch.setattr(executor_module.rclpy, "ok", lambda: True)


class _Playback(_Future):
    """A playback call that counts how often it was polled before it finished."""

    def __init__(self, response=None, *, done_after_polls=None, error=None):
        super().__init__(response, done_after_polls=done_after_polls)
        self.polls = 0
        self._error = error

    def done(self) -> bool:
        self.polls += 1
        return super().done()

    def result(self):
        if self._error is not None:
            raise self._error
        return super().result()


def _played(success=True, error_code="", message=""):
    return SimpleNamespace(success=success, error_code=error_code, message=message)


class _AudioClient:
    def __init__(self, events, playback, *, ready=True, error=None):
        self.events = events
        self.playback = playback
        self.ready = ready
        self.error = error
        self.requests = []

    def wait_for_service(self, timeout_sec):
        return self.ready

    def call_async(self, request):
        if self.error is not None:
            raise self.error
        self.requests.append(request.file_path)
        self.events.append(("audio", request.file_path))
        return self.playback


class _StopClient:
    def __init__(self, playback, *, ready=True, response=None, future=None):
        self.playback = playback
        self.ready = ready
        self.calls = 0
        self.response = response or SimpleNamespace(success=True, message="publication stopped")
        self.future = future

    def wait_for_service(self, timeout_sec):
        return self.ready

    def call_async(self, request):
        self.calls += 1
        if self.future is not None:
            return self.future
        if self.response.success:
            self.playback.complete(_played(False, "PLAYBACK_CANCELED", "publication stopped"))
        return _Future(self.response)


class _MotionClient(_Client):
    def __init__(self, events, send_future, playback=None, **kwargs):
        super().__init__(send_future, **kwargs)
        self.events = events
        self.playback = playback

    def send_goal_async(self, goal, feedback_callback=None):
        # How often playback had been polled when the motion went out: zero means
        # the motion did not wait for the playback call.
        self.events.append(("motion", goal.name, self.playback.polls if self.playback else None))
        return super().send_goal_async(goal, feedback_callback)


class _CountingGoalHandle(_GoalHandle):
    def __init__(self, **kwargs):
        super().__init__(skill_name="self_introduction", **kwargs)
        self.terminal_calls = 0

    def succeed(self):
        self.terminal_calls += 1
        super().succeed()

    def abort(self):
        self.terminal_calls += 1
        super().abort()

    def canceled(self):
        self.terminal_calls += 1
        super().canceled()


def _audio_node(tmp_path, audio_client, motion_client):
    clip = tmp_path / "self_introduction.wav"
    clip.write_bytes(b"RIFF")
    node = _node(motion_client, required=("motion.named",))
    node._wait_for_future = SkillExecutorNode._wait_for_future
    node._play_audio_client = audio_client
    node._play_audio_clips = {"self_intro": str(clip)}
    node._play_audio_service = "/voice_tts/play"
    node._play_audio_stop_client = _StopClient(audio_client.playback)
    node._play_audio_lock = Lock()
    node._play_audio_pending_cleanup = None
    node._audit_cancel_propagated = lambda: node.audits.append("cancel_propagated")
    node.asr = []
    node._asr_control_pub = SimpleNamespace(publish=lambda message: node.asr.append(message.data))
    return node


def _setup(tmp_path, playback, motion_handle=None, **motion_kwargs):
    events = []
    audio = _AudioClient(events, playback)
    motion = _MotionClient(events, _Future(motion_handle), playback, **motion_kwargs)
    return events, audio, motion, _audio_node(tmp_path, audio, motion)


def _run(node, handle, template=WAVE_TEMPLATE, timeout=5.0):
    return node._execute_play_audio_skill(handle, template, effective_timeout_sec=timeout)


def test_playback_goes_out_first_and_the_wave_follows_without_waiting(tmp_path):
    playback = _Playback(_played(), done_after_polls=40)
    events, _audio, motion, node = _setup(
        tmp_path, playback, _MotionHandle(_Future(_motion_result(), done_after_polls=3))
    )
    goal_handle = _CountingGoalHandle()

    result = _run(node, goal_handle)

    assert [event[0] for event in events] == ["audio", "motion"]
    assert events[1] == ("motion", "wave", 0)
    sent = motion.goals[0]
    assert (sent.name, sent.target, sent.interrupt, sent.resume_token) == ("wave", "", False, "")
    assert (result.success, goal_handle.terminal, goal_handle.terminal_calls) == (True, "succeeded", 1)
    assert result.executed_primitives == ["runtime_play_audio:self_intro", "runtime_named_motion:wave"]
    assert node.asr == []
    assert set(node.confirmed) == set(node.registered)


def test_a_playback_request_that_cannot_be_sent_sends_no_wave(tmp_path):
    events = []
    audio = _AudioClient(events, None, error=RuntimeError("service client destroyed"))
    motion = _MotionClient(events, _Future(_MotionHandle(_Future(_motion_result()))))
    node = _audio_node(tmp_path, audio, motion)
    goal_handle = _CountingGoalHandle()

    result = _run(node, goal_handle)

    assert (result.error_code, goal_handle.terminal) == ("PLAY_AUDIO_FAILED", "aborted")
    assert motion.goals == []
    assert result.executed_primitives == []
    assert node.asr == []


@pytest.mark.parametrize(
    ("motion_kwargs", "status", "code"),
    [
        ({"server_ready": False}, None, named_motion.RUNTIME_UNAVAILABLE),
        ({}, {"motion.named": {"names": ["handshake"]}}, named_motion.UNKNOWN),
    ],
)
def test_a_wave_the_runtime_cannot_run_now_starts_nothing(tmp_path, motion_kwargs, status, code):
    from test_runtime_named_motion_executor import _status

    playback = _Playback(_played(), done_after_polls=0)
    events, audio, motion, node = _setup(tmp_path, playback, None, **motion_kwargs)
    if status is not None:
        node._runtime_status_for_admission = lambda: (_status(status), "")

    result = _run(node, _CountingGoalHandle())

    assert result.error_code == code
    assert (events, audio.requests, motion.goals, node.asr) == ([], [], [], [])


def test_a_refused_wave_fails_the_skill_while_the_clip_was_already_requested(tmp_path):
    playback = _Playback(_played(), done_after_polls=0)
    events, _audio, _motion, node = _setup(tmp_path, playback, _MotionHandle(_Future(None), accepted=False))
    goal_handle = _CountingGoalHandle()

    result = _run(node, goal_handle)

    assert [event[0] for event in events] == ["audio", "motion"]
    assert (result.error_code, goal_handle.terminal, goal_handle.terminal_calls) == (
        named_motion.REJECTED,
        "aborted",
        1,
    )
    assert node.asr == []
    assert set(node.confirmed) == set(node.registered)


class _CancellableMotion(_MotionHandle):
    """A replayed wave: it runs until cancelled, then stops part-way."""

    def __init__(self):
        super().__init__(_Future(None, done_after_polls=None), cancel_accepted=True)

    def cancel_goal_async(self):
        future = super().cancel_goal_async()
        self.result_future.complete(
            _motion_result(
                success=False, error_code=RESULT.CANCELLED, message="wave cancelled", resume_token="t", progress=0.3
            )
        )
        return future


def test_a_playback_failure_while_waving_cancels_the_wave_and_waits_for_it(tmp_path):
    playback = _Playback(_played(False, "AUDIO_FOCUS_REJECTED", "speaker busy"), done_after_polls=5)
    wave = _CancellableMotion()
    _events, _audio, _motion, node = _setup(tmp_path, playback, wave)
    goal_handle = _CountingGoalHandle()

    result = _run(node, goal_handle)

    assert wave.cancel_calls == 1
    assert (result.error_code, result.message) == ("AUDIO_FOCUS_REJECTED", "speaker busy")
    assert (goal_handle.terminal, goal_handle.terminal_calls) == ("aborted", 1)
    assert not hasattr(goal_handle, "motion_checkpoint")
    assert node.asr == []
    assert set(node.confirmed) == set(node.registered)


def test_a_playback_failure_after_the_wave_fails_the_skill(tmp_path):
    playback = _Playback(None, done_after_polls=None)
    wave = _MotionHandle(_Future(_motion_result()))
    _events, _audio, _motion, node = _setup(tmp_path, playback, wave)
    original_done = playback.done

    def finish_after_wave():
        if wave.result_future.done() and not playback._value:
            playback.complete(_played(False, "", "player stopped"))
        return original_done()

    playback.done = finish_after_wave
    goal_handle = _CountingGoalHandle()

    result = _run(node, goal_handle)

    assert wave.cancel_calls == 0
    assert (result.error_code, result.message, goal_handle.terminal) == (
        "PLAY_AUDIO_FAILED",
        "player stopped",
        "aborted",
    )


def test_a_skill_cancel_stops_audio_and_wave_before_reporting_canceled(tmp_path):
    playback = _Playback(None, done_after_polls=None)
    wave = _CancellableMotion()
    _events, _audio, motion, node = _setup(tmp_path, playback, wave)
    goal_handle = _CountingGoalHandle()
    original_send = motion.send_goal_async

    def cancel_after_dispatch(*args, **kwargs):
        sent = original_send(*args, **kwargs)
        goal_handle.is_cancel_requested = True
        return sent

    motion.send_goal_async = cancel_after_dispatch
    result = _run(node, goal_handle)

    assert wave.cancel_calls == 1
    assert node._play_audio_stop_client.calls == 1
    assert node.audits == ["cancel_propagated"]
    assert (result.error_code, goal_handle.terminal, goal_handle.terminal_calls) == ("SKILL_CANCELLED", "canceled", 1)
    assert "buffered hardware audio may still play" in result.message
    assert result.execution_detail.phase_state == "CANCELED"
    assert result.execution_detail.recovery_state == "NOT_NEEDED"
    # The step restarts as a whole: no wave checkpoint for a resume.
    assert not hasattr(goal_handle, "motion_checkpoint")
    assert node.asr == []


@pytest.mark.parametrize(
    ("runtime_code", "public_code", "terminal"),
    [
        (RESULT.CANCELLED, "SKILL_CANCELLED", "canceled"),
        (RESULT.REJECTED_BY_PLATFORM, named_motion.REJECTED, "aborted"),
        (RESULT.RUNTIME_UNAVAILABLE, named_motion.PLATFORM_NOT_READY, "aborted"),
        (RESULT.STOP_LATCHED, named_motion.STOP_LATCHED, "aborted"),
    ],
)
def test_intro_waits_for_return_confirmation_and_preserves_return_failures(
    tmp_path, runtime_code, public_code, terminal
):
    """Accepted cancel is not terminal: the runtime still owns the returning arm."""
    playback = _Playback(None, done_after_polls=None)
    runtime_result = _Future(None, done_after_polls=None)
    cancel_seen = Event()
    wave = _MotionHandle(runtime_result, cancel_accepted=True)
    original_cancel = wave.cancel_goal_async

    def cancel():
        response = original_cancel()
        cancel_seen.set()
        return response

    wave.cancel_goal_async = cancel
    _events, _audio, motion, node = _setup(tmp_path, playback, wave)
    handle = _CountingGoalHandle()
    original_send = motion.send_goal_async

    def cancel_after_dispatch(*args, **kwargs):
        response = original_send(*args, **kwargs)
        handle.is_cancel_requested = True
        return response

    motion.send_goal_async = cancel_after_dispatch
    outcomes, errors = [], []

    def execute():
        try:
            outcomes.append(_run(node, handle))
        except BaseException as error:
            errors.append(error)

    worker = Thread(target=execute, daemon=True)
    worker.start()
    try:
        assert cancel_seen.wait(2.0), "motion cancel was not forwarded"
        # The mock has accepted cancel but deliberately withholds its result
        # until the runtime would have verified the measured starting pose.
        assert worker.is_alive()
        assert (handle.terminal, handle.terminal_calls) == (None, 0)
        assert node._play_audio_stop_client.calls == 1
        assert len(node.confirmed) < len(node.registered)
        assert outcomes == [] and errors == []
    finally:
        runtime_result.complete(
            _motion_result(
                success=False,
                error_code=runtime_code,
                message="returned to measured start" if terminal == "canceled" else "return not confirmed",
                resume_token="",
                progress=0.3,
            )
        )
        worker.join(2.0)
    assert not worker.is_alive() and errors == []
    (result,) = outcomes
    assert (result.success, result.error_code) == (False, public_code)
    assert (handle.terminal, handle.terminal_calls) == (terminal, 1)
    assert wave.cancel_calls == 1
    assert set(node.confirmed) == set(node.registered)
    assert not hasattr(handle, "motion_checkpoint")
    if terminal == "canceled":
        assert result.execution_detail.phase_state == "CANCELED"
        assert result.execution_detail.recovery_state == "NOT_NEEDED"
    else:
        assert "return not confirmed" in result.message
        assert result.execution_detail.phase_state != "CANCELED"


def test_a_wave_still_running_at_the_deadline_is_reported_unknown_and_keeps_its_admission(tmp_path):
    playback = _Playback(None, done_after_polls=None)
    result_future = _Future(None, done_after_polls=None)
    _events, _audio, _motion, node = _setup(tmp_path, playback, _MotionHandle(result_future))
    goal_handle = _CountingGoalHandle()

    result = _run(node, goal_handle, timeout=0.1)

    assert (result.error_code, goal_handle.terminal, goal_handle.terminal_calls) == (
        PRIMITIVE_CANCEL_CLEANUP_TIMEOUT,
        "aborted",
        1,
    )
    assert result.execution_detail.phase_state == "UNKNOWN"
    assert result.execution_detail.recovery_state == "UNKNOWN"
    # Audio has ended, but the motion's delegated admission is still held.
    assert len(node.confirmed) + 1 == len(node.registered)
    result_future.complete(_motion_result())
    assert set(node.confirmed) == set(node.registered)
    assert node.asr == []


def test_a_clip_without_a_motion_plays_as_before(tmp_path):
    events = []
    audio = _AudioClient(events, _Playback(_played(), done_after_polls=2))
    node = _audio_node(tmp_path, audio, None)
    goal_handle = _CountingGoalHandle()

    result = _run(node, goal_handle, template=CLIP_TEMPLATE)

    assert (result.success, goal_handle.terminal, goal_handle.terminal_calls) == (True, "succeeded", 1)
    assert result.executed_primitives == ["runtime_play_audio:self_intro"]
    assert result.message == "self_introduction: played self_intro"
    assert node.asr == []


def _cancel_when_audio_is_sent(audio, handle):
    original_call = audio.call_async

    def send(request):
        future = original_call(request)
        handle.is_cancel_requested = True
        return future

    audio.call_async = send


def test_cancel_before_dispatch_does_not_stop_unrelated_audio(tmp_path):
    _events, audio, _motion, node = _setup(tmp_path, _Playback(None, done_after_polls=None))
    handle = _CountingGoalHandle(cancel_requested=True)

    result = _run(node, handle, CLIP_TEMPLATE)

    assert result.error_code == "SKILL_CANCELLED"
    assert audio.requests == []
    assert node._play_audio_stop_client.calls == 0


def test_clip_only_cancel_stops_playback_and_preserves_listening(tmp_path):
    playback = _Playback(None, done_after_polls=None)
    _events, audio, _motion, node = _setup(tmp_path, playback)
    handle = _CountingGoalHandle()
    _cancel_when_audio_is_sent(audio, handle)

    result = _run(node, handle, CLIP_TEMPLATE)

    assert (result.error_code, handle.terminal, handle.terminal_calls) == ("SKILL_CANCELLED", "canceled", 1)
    assert node._play_audio_stop_client.calls == 1
    assert playback.done()
    assert set(node.confirmed) == set(node.registered)
    assert node.asr == []


@pytest.mark.parametrize("pending_response", ["play", "stop"])
def test_cancel_waits_for_both_responses_and_blocks_new_playback(tmp_path, pending_response):
    playback = _Playback(None, done_after_polls=None)
    stop_future = _Future(None, done_after_polls=None)
    _events, audio, _motion, node = _setup(tmp_path, playback)
    handle = _CountingGoalHandle()
    _cancel_when_audio_is_sent(audio, handle)
    node._play_audio_stop_client = _StopClient(playback, future=stop_future)
    original_stop = node._play_audio_stop_client.call_async

    def stop(request):
        future = original_stop(request)
        if pending_response == "play":
            stop_future.complete(SimpleNamespace(success=True, message="stopped"))
        else:
            playback.complete(_played(False, "PLAYBACK_CANCELED"))
        return future

    node._play_audio_stop_client.call_async = stop
    result = _run(node, handle, CLIP_TEMPLATE)

    assert (result.error_code, handle.terminal) == (PRIMITIVE_CANCEL_CLEANUP_TIMEOUT, "aborted")
    assert result.execution_detail.phase_state == "UNKNOWN"
    assert result.execution_detail.recovery_state == "UNKNOWN"
    assert node.confirmed == []
    audio.call_async = lambda _request: pytest.fail("new audio sent while old stop/play pending")
    blocked = _run(node, _CountingGoalHandle(), CLIP_TEMPLATE)
    assert blocked.error_code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT
    stop_calls = node._play_audio_stop_client.calls
    stop_future.complete(SimpleNamespace(success=True, message="stopped"))
    playback.complete(_played(False, "PLAYBACK_CANCELED"))
    assert set(node.confirmed) == set(node.registered)
    next_play = _Playback(_played(), done_after_polls=0)
    audio.call_async = lambda _request: next_play
    assert _run(node, _CountingGoalHandle(), CLIP_TEMPLATE).success
    assert node._play_audio_stop_client.calls == stop_calls
    if pending_response == "stop":
        assert stop_calls == 1  # Never queue another global stop behind a pending one.


@pytest.mark.parametrize("unavailable", [False, True])
def test_unconfirmed_or_failed_stop_never_reports_safe_cancel(tmp_path, unavailable):
    playback = _Playback(None, done_after_polls=None)
    _events, audio, _motion, node = _setup(tmp_path, playback)
    handle = _CountingGoalHandle()
    _cancel_when_audio_is_sent(audio, handle)
    node._play_audio_stop_client = _StopClient(
        playback, ready=not unavailable, response=SimpleNamespace(success=False, message="stop failed")
    )
    result = _run(node, handle, CLIP_TEMPLATE)
    assert result.error_code == PRIMITIVE_CANCEL_CLEANUP_TIMEOUT
    assert handle.terminal == "aborted"
    assert node.confirmed == []


def test_stop_is_repeated_when_play_is_accepted_after_the_first_stop(tmp_path):
    playback = _Playback(None, done_after_polls=None)
    _events, audio, _motion, node = _setup(tmp_path, playback)
    handle = _CountingGoalHandle()
    _cancel_when_audio_is_sent(audio, handle)
    calls = []

    def stop(_request):
        calls.append("stop")
        if len(calls) == 1:
            # The stop callback ran before the play callback entered its queue.
            return _Future(SimpleNamespace(success=True, message="no active publication"))
        playback.complete(_played(False, "PLAYBACK_CANCELED"))
        return _Future(SimpleNamespace(success=True, message="late play stopped"))

    node._play_audio_stop_client.call_async = stop
    result = _run(node, handle, CLIP_TEMPLATE)
    assert len(calls) == 2
    assert (result.error_code, handle.terminal) == ("SKILL_CANCELLED", "canceled")
    assert set(node.confirmed) == set(node.registered)
    audio.call_async = lambda _request: _Playback(_played(), done_after_polls=0)
    assert _run(node, _CountingGoalHandle(), CLIP_TEMPLATE).success
    assert len(calls) == 2  # No late global stop after ownership was returned.


def test_stop_refusal_with_finished_playback_is_failed_not_canceled(tmp_path):
    playback = _Playback(None, done_after_polls=None)
    _events, audio, _motion, node = _setup(tmp_path, playback)
    handle = _CountingGoalHandle()
    _cancel_when_audio_is_sent(audio, handle)
    stop = node._play_audio_stop_client

    def refuse(_request):
        playback.complete(_played())
        return _Future(SimpleNamespace(success=False, message="stop refused"))

    stop.call_async = refuse
    result = _run(node, handle, CLIP_TEMPLATE)
    assert (result.error_code, handle.terminal) == ("PLAY_AUDIO_STOP_FAILED", "aborted")
    assert result.execution_detail.recovery_state == "FAILED"


def test_timeout_stops_playback_and_waits_for_terminal(tmp_path):
    playback = _Playback(None, done_after_polls=None)
    _events, _audio, _motion, node = _setup(tmp_path, playback)
    result = _run(node, _CountingGoalHandle(), CLIP_TEMPLATE, timeout=0.01)
    assert result.error_code == "PLAY_AUDIO_TIMEOUT"
    assert node._play_audio_stop_client.calls == 1
    assert set(node.confirmed) == set(node.registered)


def test_a_clip_with_a_motion_counts_as_a_named_motion_catalog_entry():
    node = object.__new__(SkillExecutorNode)
    node._runtime_named_motion_enabled = False
    node._play_audio_enabled = True
    snapshot = SimpleNamespace(templates={"self_introduction": dict(WAVE_TEMPLATE)}, enabled_skill_names=())

    with pytest.raises(ValueError, match="runtime_named_motion configuration mismatch"):
        node._validate_hri_runtime_catalog_consistency(snapshot)
