"""Who a long cover and the session timeout call: the Gateway, and the session only when it gives no answer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy
import pytest

from ibrobot_msgs.srv import ValidateSkill
from manipulation_execution.mirror_operator.config import OperatorConfig
from manipulation_execution.mirror_operator.core import GATEWAY_TIMEOUT_S, STOP_TIMEOUT_S, TIMEOUT_RETRY_S, Operator
from manipulation_execution.mirror_operator.gateway import MIRROR_START, MIRROR_STOP, GatewayError, MirrorSkillRequest
from manipulation_execution.mirror_operator.gesture import DOUBLE_COVER, LONG_COVER, GestureRecognizer
from manipulation_execution.mirror_operator.judge import CoverJudge, GrayFrame
from manipulation_execution.mirror_operator_node import RosOperatorCalls

LIMIT = 60.0
FRAME = GrayFrame.of(numpy.zeros((2, 2), dtype=numpy.uint8), 0.0)
BUSY = (False, "SKILL_BUSY: another root execution is active")
NO_SERVICE = (None, "Gateway status service is not available")


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Judge(CoverJudge):
    def __init__(self):
        super().__init__()
        self.is_covered = False

    def covered(self, frame):
        return self.is_covered


class Calls:
    """Records every call; ``answers`` answers one at once, otherwise the test answers the held ``done``."""

    def __init__(self):
        self.starts, self.stops, self.direct, self.touches = [], [], [], []
        self.answers = {}

    def _record(self, name, done):
        getattr(self, name).append(done)
        if name in self.answers:
            done(*self.answers[name])

    def mirror_start(self, done):
        self._record("starts", done)

    def mirror_stop(self, done):
        self._record("stops", done)

    def stop_session(self, done):
        self._record("direct", done)

    def operator_touch(self, done):
        self._record("touches", done)

    def set_led(self, color, priority, done):
        pass


class Rig:
    def __init__(self, limit=0.0, send_enabled=True):
        self.clock, self.calls, self.judge, self.logs = Clock(), Calls(), Judge(), []
        self.config = OperatorConfig.from_mapping({"max_session_s": limit})
        self.operator = Operator(
            self.config,
            self.judge,
            self.calls,
            send_enabled=send_enabled,
            led_unavailable="no light in the test",
            clock=self.clock,
            log=lambda level, message: self.logs.append((level, message)),
        )
        self.frames(self.config.startup_clear_s + 0.2, "WAIT", covered=False)
        assert self.operator.status()["armed"] is True

    def frames(self, seconds, session, *, covered, every=0.1):
        """Frames and session state messages for ``seconds``; ticks too."""
        self.judge.is_covered = covered
        end = self.clock.now + seconds
        while self.clock.now < end:
            self.clock.now += every
            if session is not None:
                self.operator.on_state({"state": session})
            self.operator.on_frame(FRAME)
            self.operator.tick()

    def long_cover(self, session):
        """A cover up to the frame that fires, then clear long enough to count again."""
        gesture = self.operator.status()["gesture"]
        self.frames(self.config.long_cover_s + 0.2, session, covered=True)
        assert self.operator.status()["gesture"] != gesture, "the long cover did not fire"
        self.frames(self.config.rearm_clear_s + 0.2, session, covered=False)

    def double_cover(self, session):
        double = self.config.double_cover
        cover, gap = (double.cover_min_s + double.cover_max_s) / 2, (double.gap_min_s + double.gap_max_s) / 2
        self.frames(cover, session, covered=True)
        self.frames(gap, session, covered=False)
        self.frames(cover, session, covered=True)
        self.frames(self.config.rearm_clear_s + 0.2, session, covered=False)

    def status(self):
        return self.operator.status()


# ---------------------------------------------------------------------------
# long cover
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("session", ["STANDBY", "MIRRORING"])
def test_long_cover_goes_through_the_gateway(session):
    rig = Rig()
    rig.long_cover(session)
    assert len(rig.calls.stops) == 1
    assert rig.calls.direct == []
    assert rig.status()["in_flight"]["gateway"] == "mirror_stop"
    assert rig.status()["last_gateway"]["call"] == "mirror_stop"
    rig.calls.stops[0](True, "stopped")
    assert rig.status()["last_gateway"]["result"] == "ok"
    assert rig.status()["stop_fallback"] is None
    assert rig.calls.direct == []


def test_long_cover_in_wait_sends_nothing():
    rig = Rig()
    rig.long_cover("WAIT")
    assert rig.calls.stops == [] and rig.calls.direct == []
    assert rig.status()["gesture"]["action"] == "ignored: session WAIT"


def test_a_refusal_is_obeyed():
    rig = Rig()
    rig.calls.answers["stops"] = BUSY
    rig.long_cover("MIRRORING")
    assert rig.status()["last_gateway"]["result"] == "failed"
    assert "SKILL_BUSY" in rig.status()["last_gateway"]["message"]
    rig.frames(20.0, "MIRRORING", covered=False)
    assert len(rig.calls.stops) == 1  # a long cover is one try
    assert rig.calls.direct == []
    assert rig.status()["stop_fallback"] is None


def test_no_gateway_stops_the_session_directly_at_once():
    rig = Rig()
    rig.calls.answers["stops"] = NO_SERVICE
    rig.long_cover("MIRRORING")
    assert len(rig.calls.stops) == 1
    assert len(rig.calls.direct) == 1
    assert rig.status()["last_session_call"]["call"] == "stop"
    fallback = rig.status()["stop_fallback"]
    assert "no answer from the Gateway" in fallback["reason"]
    assert fallback["action"] == "sent stop"
    assert any(level == "warning" and "direct session stop" in message for level, message in rig.logs)


def test_hung_gateway_stops_the_session_directly_after_its_timeout():
    rig = Rig()
    rig.long_cover("MIRRORING")
    assert rig.calls.direct == []
    rig.frames(GATEWAY_TIMEOUT_S, "MIRRORING", covered=False)
    assert rig.status()["last_gateway"]["result"] == "timeout"
    assert len(rig.calls.direct) == 1
    rig.calls.direct[0](True, "stopped")
    rig.calls.stops[0](True, "late")  # ignored
    rig.frames(10.0, "WAIT", covered=False)
    assert rig.status()["last_gateway"]["result"] == "timeout"
    assert len(rig.calls.stops) == 1 and len(rig.calls.direct) == 1


def test_no_direct_stop_when_the_session_is_already_back_in_wait():
    rig = Rig()
    rig.long_cover("MIRRORING")
    rig.frames(GATEWAY_TIMEOUT_S, "WAIT", covered=False)
    assert rig.status()["last_gateway"]["result"] == "timeout"
    assert rig.calls.direct == []
    assert rig.status()["stop_fallback"]["action"].startswith("not sent")


def test_no_direct_stop_into_a_later_round():
    # WAIT came after the stop was sent: whatever is on now is not what the stop was for
    rig = Rig()
    rig.long_cover("MIRRORING")
    rig.frames(1.0, "WAIT", covered=False)
    rig.frames(GATEWAY_TIMEOUT_S, "STANDBY", covered=False)
    assert rig.status()["last_gateway"]["result"] == "timeout"
    assert rig.calls.direct == []
    assert rig.status()["stop_fallback"]["action"].startswith("not sent")


def test_no_direct_stop_when_the_session_is_silent():
    rig = Rig()
    rig.long_cover("MIRRORING")
    rig.frames(GATEWAY_TIMEOUT_S, None, covered=False)  # no state message: no session to stop
    assert rig.status()["last_gateway"]["result"] == "timeout"
    assert rig.calls.direct == []
    assert rig.status()["stop_fallback"]["action"] == "not sent: session none"


def test_touch_in_flight_drops_the_direct_stop():
    rig = Rig()
    assert rig.operator._send("session", "touch", rig.clock.now) == "sent touch"
    rig.calls.answers["stops"] = NO_SERVICE
    rig.clock.now += 0.1
    rig.operator.on_state({"state": "MIRRORING"})
    rig.operator._on_gesture("long_cover", rig.clock.now)
    assert len(rig.calls.stops) == 1
    assert rig.calls.direct == []
    assert rig.operator.dropped["session"] == 1
    assert rig.status()["stop_fallback"]["action"] == "dropped: touch in flight"


def test_send_disabled_sends_neither():
    rig = Rig(send_enabled=False)
    rig.long_cover("MIRRORING")
    assert rig.calls.stops == [] and rig.calls.direct == []
    assert (rig.status()["last_gateway"]["call"], rig.status()["last_gateway"]["result"]) == ("mirror_stop", "not_sent")
    assert rig.status()["stop_fallback"] is None


# ---------------------------------------------------------------------------
# session timeout
# ---------------------------------------------------------------------------


def test_timeout_goes_through_the_gateway():
    rig = Rig(limit=LIMIT)
    rig.frames(LIMIT - 0.5, "STANDBY", covered=False)
    assert rig.calls.stops == []
    rig.frames(1.0, "STANDBY", covered=False)
    assert len(rig.calls.stops) == 1
    assert rig.calls.direct == []
    rig.calls.stops[0](True, "stopped")
    rig.frames(30.0, "STANDBY", covered=False)
    assert len(rig.calls.stops) == 1
    assert rig.status()["session_timeout"]["done"] is True


def test_timeout_without_a_gateway_stops_directly_and_counts_that_answer():
    rig = Rig(limit=LIMIT)
    rig.calls.answers["stops"] = NO_SERVICE
    rig.frames(LIMIT + 0.5, "MIRRORING", covered=False)
    assert len(rig.calls.stops) == 1
    assert len(rig.calls.direct) == 1
    assert rig.status()["session_timeout"]["last"]["call"] == "stop"
    assert rig.status()["session_timeout"]["last"]["result"] == ""
    rig.frames(STOP_TIMEOUT_S - 1.0, "MIRRORING", covered=False)  # the direct stop is in flight: nothing more
    assert len(rig.calls.stops) == 1 and len(rig.calls.direct) == 1
    rig.calls.direct[0](True, "stopped")
    sent = len(rig.calls.stops)
    rig.frames(10.0, "MIRRORING", covered=False)
    assert len(rig.calls.stops) == sent
    assert rig.status()["session_timeout"]["done"] is True


def test_timeout_without_a_gateway_retries_every_retry_s_when_the_session_declines():
    rig = Rig(limit=LIMIT)
    rig.calls.answers["stops"] = NO_SERVICE
    rig.calls.answers["direct"] = (False, "stop in progress")
    rig.frames(LIMIT + 0.5, "MIRRORING", covered=False)
    first = len(rig.calls.direct)
    assert first == 1
    rig.frames(10.0, "MIRRORING", covered=False)
    assert 4 <= len(rig.calls.direct) - first <= 6
    assert len(rig.calls.stops) == len(rig.calls.direct)  # the Gateway is asked first every time


def test_retry_interval_is_two_seconds():
    assert TIMEOUT_RETRY_S == 2.0


# ---------------------------------------------------------------------------
# double cover and touch: not changed
# ---------------------------------------------------------------------------


def test_double_cover_in_wait_is_mirror_start():
    rig = Rig()
    rig.double_cover("WAIT")
    assert len(rig.calls.starts) == 1
    assert rig.calls.stops == [] and rig.calls.direct == []


def test_double_cover_while_on_sends_nothing():
    rig = Rig()
    rig.double_cover("MIRRORING")
    assert rig.calls.starts == [] and rig.calls.stops == []


def test_failed_mirror_start_does_not_stop_anything():
    rig = Rig()
    rig.calls.answers["starts"] = NO_SERVICE
    rig.double_cover("WAIT")
    rig.frames(1.0, "MIRRORING", covered=False)
    assert rig.calls.direct == []
    assert rig.status()["stop_fallback"] is None


def test_touch_still_goes_to_the_session():
    rig = Rig()
    touch = rig.config.touch
    rig.operator.on_state({"state": "MIRRORING"})
    rig.operator.on_touch(0)
    for _ in range(int(touch.hold_s / 0.1) + 2):
        rig.clock.now += 0.1
        rig.operator.on_touch(touch.pressed_event_types[0])
    assert len(rig.calls.touches) == 1
    assert rig.calls.stops == [] and rig.calls.direct == []


# ---------------------------------------------------------------------------
# a covered camera drops frames: the cover goes on, other gaps still clear
# ---------------------------------------------------------------------------

DARK_GAPS = (0.4, 0.7, 0.5, 0.6, 0.4, 0.7)  # about 2-3 Hz once the exposure is long


def armed_recognizer(config=None):
    """A recognizer armed by clear frames at 10 Hz; returns it and the last frame time."""
    config = config or OperatorConfig()
    gestures = GestureRecognizer(config)
    now = 0.0
    while not gestures.armed:
        now = round(now + 0.1, 6)
        gestures.update(False, now)
    return gestures, now


def feed(gestures, now, covered, gaps):
    """One frame after each gap; the (time, gesture) of each and the last time."""
    seen = []
    for gap in gaps:
        now = round(now + gap, 6)
        seen.append((now, gestures.update(covered, now)))
    return seen, now


def test_long_cover_fires_at_dark_frame_rates():
    config = OperatorConfig()
    gestures, now = armed_recognizer(config)
    start = round(now + 0.1, 6)
    assert gestures.update(True, start) is None
    seen, _ = feed(gestures, start, True, DARK_GAPS)
    fired = [at for at, gesture in seen if gesture == LONG_COVER]
    assert fired, "the long cover never fired at 2-3 Hz"
    # Real time: the first covered frame at or after long_cover_s, not earlier.
    expected = next(at for at, _ in seen if at - start >= config.long_cover_s)
    assert fired == [expected]
    assert all(gesture is None for at, gesture in seen if at < expected)


def test_long_cover_at_dark_frame_rates_stops_the_session():
    rig = Rig()
    rig.judge.is_covered = True
    for gap in (0.1,) + DARK_GAPS:
        rig.clock.now += gap
        rig.operator.on_state({"state": "MIRRORING"})
        rig.operator.on_frame(FRAME)
        rig.operator.tick()
    assert len(rig.calls.stops) == 1


def test_bridge_off_when_cover_frame_gap_is_not_over_the_reset_gap():
    config = OperatorConfig.from_mapping({"cover_frame_gap_s": 0.5})
    gestures, now = armed_recognizer(config)
    seen, _ = feed(gestures, now, True, (0.1,) + DARK_GAPS)
    assert [gesture for _, gesture in seen] == [None] * len(seen)
    assert gestures.armed is False


def test_a_clear_stream_gap_still_disarms():
    gestures, now = armed_recognizer()
    seen, now = feed(gestures, now, False, (0.8,))
    assert gestures.armed is False
    # and a cover right after it is not timed: the startup clear comes first
    seen, _ = feed(gestures, now, True, (0.1,) * 30)
    assert [gesture for _, gesture in seen] == [None] * 30


@pytest.mark.parametrize(("before", "after"), [(False, True), (True, False)])
def test_a_gap_into_or_out_of_a_cover_still_disarms(before, after):
    gestures, now = armed_recognizer()
    seen, now = feed(gestures, now, before, (0.1, 0.1))
    seen, now = feed(gestures, now, after, (0.8,))
    assert [gesture for _, gesture in seen] == [None]
    assert gestures.armed is False


def test_a_cover_gap_over_cover_frame_gap_still_disarms():
    config = OperatorConfig()
    gestures, now = armed_recognizer(config)
    seen, now = feed(gestures, now, True, (0.1, 0.4, 0.6))
    assert gestures.armed is True
    seen, now = feed(gestures, now, True, (config.cover_frame_gap_s + 0.1,))
    assert gestures.armed is False
    seen, _ = feed(gestures, now, True, (0.4,) * 10)
    assert [gesture for _, gesture in seen] == [None] * 10


def test_double_cover_still_fires_at_full_frame_rate():
    gestures, now = armed_recognizer()
    seen, now = feed(gestures, now, True, (0.1,) * 5)
    seen, now = feed(gestures, now, False, (0.1,) * 4)
    seen, now = feed(gestures, now, True, (0.1,) * 5)
    seen, now = feed(gestures, now, False, (0.1,))
    assert seen[-1][1] == DOUBLE_COVER


def test_a_bridged_cover_is_never_half_a_double_cover():
    """0.7 s from cover to clear fits cover_max_s, but a hand may have lifted unseen in the 0.6 s gap."""
    gestures, now = armed_recognizer()
    seen, now = feed(gestures, now, True, (0.1, 0.6))  # bridged first cover
    seen, now = feed(gestures, now, False, (0.1,) * 4)
    assert gestures.armed is True
    seen, now = feed(gestures, now, True, (0.1,) * 5)
    seen, _ = feed(gestures, now, False, (0.1,) * 5)
    assert [gesture for _, gesture in seen] == [None] * 5
    gestures, now = armed_recognizer()
    seen, now = feed(gestures, now, True, (0.1,) * 5)
    seen, now = feed(gestures, now, False, (0.1,) * 4)
    seen, now = feed(gestures, now, True, (0.1, 0.6))  # bridged second cover
    seen, _ = feed(gestures, now, False, (0.1,) * 5)
    assert [gesture for _, gesture in seen] == [None] * 5


def test_a_double_cover_hidden_in_a_bridge_is_not_a_long_cover():
    """Cover, an unseen lift inside a bridged gap, cover, then clear: no gesture at all."""
    gestures, now = armed_recognizer()
    seen, now = feed(gestures, now, True, (0.1, 0.1, 0.1, 1.2, 0.1, 0.1))  # 1.7 s of cover frames
    assert [gesture for _, gesture in seen] == [None] * 6
    seen, _ = feed(gestures, now, False, (0.1,) * 15)
    assert [gesture for _, gesture in seen] == [None] * 15
    assert gestures.armed is True


# ---------------------------------------------------------------------------
# the request and the ROS calls
# ---------------------------------------------------------------------------


class Answered:
    """A future already answered."""

    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        callback(self)


class Service:
    def __init__(self, answer=None, ready=True):
        self.answer, self.ready, self.requests = answer, ready, []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)
        return Answered(self.answer)


class Action:
    def __init__(self, handle=None, ready=True):
        self.handle, self.ready, self.goals = handle, ready, []

    def server_is_ready(self):
        return self.ready

    def send_goal_async(self, goal, goal_uuid=None):
        self.goals.append(goal)
        return Answered(self.handle)


def status_answer(skill, ready=True, reason=""):
    return SimpleNamespace(
        capabilities=[SimpleNamespace(name=skill, ready=ready, reason=reason)],
        registry_epoch="epoch",
        registry_generation=3,
        registry_digest="digest",
        default_skill_timeout_sec=30.0,
    )


def validate_answer():
    return SimpleNamespace(
        allowed=True,
        error_code="",
        reason="",
        actual_registry_epoch="epoch",
        actual_registry_generation=3,
        actual_registry_digest="digest",
    )


def handle_answer(success=True, message="stopped", error_code=""):
    result = SimpleNamespace(success=success, message=message, error_code=error_code)
    return SimpleNamespace(accepted=True, get_result_async=lambda: Answered(SimpleNamespace(result=result)))


def ros_calls(*, status=None, validate=None, skill=None, session=None):
    return RosOperatorCalls(
        status=status or Service(status_answer(MIRROR_STOP)),
        validate=validate or Service(validate_answer()),
        skill=skill or Action(handle_answer()),
        session=session or Service(SimpleNamespace(success=True, message="stopped")),
        touch=Service(SimpleNamespace(success=True, message="")),
    )


def answer_of(call):
    answers = []
    call(lambda success, message: answers.append((success, message)))
    assert len(answers) == 1
    return answers[0]


def test_ros_mirror_stop_sends_the_mirror_stop_goal():
    skill = Action(handle_answer())
    validate = Service(validate_answer())
    calls = ros_calls(skill=skill, validate=validate)
    assert answer_of(calls.mirror_stop) == (True, "stopped")
    assert [goal.skill_name for goal in skill.goals] == [MIRROR_STOP]
    assert skill.goals[0].timeout_sec == 5.0
    assert [request.skill_name for request in validate.requests] == [MIRROR_STOP]


def test_ros_mirror_start_still_sends_the_mirror_start_goal():
    skill = Action(handle_answer(message="started"))
    calls = ros_calls(status=Service(status_answer(MIRROR_START)), skill=skill)
    assert answer_of(calls.mirror_start) == (True, "started")
    assert [goal.skill_name for goal in skill.goals] == [MIRROR_START]


def test_ros_mirror_stop_with_no_gateway_is_no_answer():
    assert answer_of(ros_calls(status=Service(ready=False)).mirror_stop)[0] is None
    assert answer_of(ros_calls(validate=Service(ready=False)).mirror_stop)[0] is None
    assert answer_of(ros_calls(skill=Action(ready=False)).mirror_stop)[0] is None


def test_ros_mirror_stop_busy_is_a_refusal():
    # the Gateway reports the lease in its status: the code comes back, no goal goes
    skill = Action(handle_answer())
    reason = "SKILL_BUSY: another root execution is active"
    calls = ros_calls(status=Service(status_answer(MIRROR_STOP, ready=False, reason=reason)), skill=skill)
    success, message = answer_of(calls.mirror_stop)
    assert success is False
    assert "SKILL_BUSY" in message
    assert skill.goals == []


def test_ros_mirror_stop_failed_result_is_a_refusal():
    skill = Action(handle_answer(success=False, message="busy", error_code="SKILL_BUSY"))
    success, message = answer_of(ros_calls(skill=skill).mirror_stop)
    assert success is False
    assert "SKILL_BUSY" in message


def test_request_is_for_the_named_skill_only():
    request = MirrorSkillRequest(MIRROR_STOP)
    with pytest.raises(GatewayError):
        request.accept_status(status_answer(MIRROR_START))
    request.accept_status(status_answer(MIRROR_STOP))
    assert request.payload["skill_name"] == MIRROR_STOP
    assert isinstance(request.fill_validate_request(ValidateSkill.Request()), ValidateSkill.Request)


def test_each_request_is_a_new_task():
    assert MirrorSkillRequest(MIRROR_STOP).task_id != MirrorSkillRequest(MIRROR_STOP).task_id
