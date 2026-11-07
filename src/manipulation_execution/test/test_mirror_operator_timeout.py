"""The operator's session timeout (``max_session_s``): when the stop goes, and when it does not. No ROS."""

from __future__ import annotations

import pytest

from manipulation_execution.mirror_operator.config import OperatorConfig
from manipulation_execution.mirror_operator.core import GATEWAY_TIMEOUT_S, STOP_TIMEOUT_S, TIMEOUT_RETRY_S, Operator
from manipulation_execution.mirror_operator.judge import CoverJudge

LIMIT = 60.0


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Calls:
    """Records every call; the test answers the held ``done`` callbacks itself."""

    def __init__(self):
        self.stops = []  # mirror_stop through the Gateway
        self.direct = []  # set_session(false), only when the Gateway gave no answer
        self.touches = []

    def mirror_start(self, done):
        raise AssertionError("no mirror_start expected")

    def mirror_stop(self, done):
        self.stops.append(done)

    def stop_session(self, done):
        self.direct.append(done)

    def operator_touch(self, done):
        self.touches.append(done)

    def set_led(self, color, priority, done):
        pass


def make(limit=LIMIT, send_enabled=True):
    clock, calls, logs = Clock(), Calls(), []
    config = OperatorConfig.from_mapping({"max_session_s": limit})
    operator = Operator(
        config,
        CoverJudge(),
        calls,
        send_enabled=send_enabled,
        led_unavailable="no light in the test",
        clock=clock,
        log=lambda level, message: logs.append((level, message)),
    )
    return operator, clock, calls, logs


def state(operator, clock, value, advance=0.0):
    clock.now += advance
    operator.on_state({"state": value})


def run(operator, clock, seconds, *, every=1.0, keep=None):
    """Tick for ``seconds``; ``keep`` is a state sent with every tick (the session's periodic message)."""
    end = clock.now + seconds
    while clock.now < end:
        clock.now = min(end, clock.now + every)
        if keep is not None:
            operator.on_state({"state": keep})
        operator.tick()


def hold_touch(operator, clock):
    """A release, then a press held past ``hold_s``: one touch call."""
    touch = OperatorConfig().touch
    operator.on_touch(0)
    for _ in range(int(touch.hold_s / 0.1) + 2):
        clock.now += 0.1
        operator.on_touch(touch.pressed_event_types[0])


def timeout(operator):
    return operator.status()["session_timeout"]


def test_config_default_is_off_and_zero_is_accepted():
    assert OperatorConfig().max_session_s == 0.0
    assert OperatorConfig.from_mapping({"max_session_s": 0}).max_session_s == 0.0
    assert OperatorConfig.from_mapping({"max_session_s": 60}).max_session_s == 60.0
    with pytest.raises(ValueError, match="max_session_s"):
        OperatorConfig.from_mapping({"max_session_s": -1})


def test_default_off_never_stops():
    operator, clock, calls, _ = make(limit=0.0)
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, 600.0, keep="MIRRORING")
    assert calls.stops == []
    assert timeout(operator)["enabled"] is False
    assert timeout(operator)["fired"] is False


@pytest.mark.parametrize("started_as", ["STANDBY", "MIRRORING"])
def test_due_sends_stop_once(started_as):
    # robot-skill mirror_start and the double cover both reach the operator as WAIT -> STANDBY
    operator, clock, calls, logs = make()
    state(operator, clock, "WAIT")
    state(operator, clock, started_as, advance=1.0)
    run(operator, clock, LIMIT - 0.5, every=0.1, keep=started_as)
    assert calls.stops == []
    assert timeout(operator)["elapsed_s"] == pytest.approx(LIMIT - 0.5)
    run(operator, clock, 1.0, every=0.1, keep=started_as)
    assert len(calls.stops) == 1
    assert timeout(operator)["fired"] is True
    calls.stops[0](True, "stopped")
    run(operator, clock, 30.0, every=0.1, keep=started_as)  # the answer came, WAIT not yet: no second stop
    assert len(calls.stops) == 1
    assert timeout(operator)["done"] is True
    assert timeout(operator)["last"]["result"] == "ok"
    assert any("session timeout" in message for _, message in logs)


@pytest.mark.parametrize("first", ["STANDBY", "MIRRORING"])
def test_first_state_on_without_wait_counts(first):
    # the operator (re)started mid-round: the first message it sees is not WAIT
    operator, clock, calls, _ = make()
    state(operator, clock, first)
    run(operator, clock, LIMIT - 1.0, keep=first)
    assert calls.stops == []
    run(operator, clock, 1.0, keep=first)
    assert len(calls.stops) == 1


def test_standby_mirroring_back_and_forth_does_not_clear():
    operator, clock, calls, _ = make()
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    for value in ("MIRRORING", "STANDBY", "MIRRORING", "STANDBY", "MIRRORING", "STANDBY"):
        run(operator, clock, 9.0, keep=value)  # a head touch takes the session back and forth
    assert calls.stops == []
    run(operator, clock, 7.0, keep="STANDBY")
    assert len(calls.stops) == 1


def test_head_touch_does_not_clear():
    operator, clock, calls, _ = make()
    state(operator, clock, "WAIT")
    state(operator, clock, "MIRRORING")
    run(operator, clock, 30.0, keep="MIRRORING")
    hold_touch(operator, clock)
    assert len(calls.touches) == 1
    calls.touches[0](True, "cancelled")
    state(operator, clock, "STANDBY")
    run(operator, clock, LIMIT - 30.0, keep="STANDBY")
    assert len(calls.stops) == 1


def test_silent_state_topic_does_not_clear():
    operator, clock, calls, _ = make()
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, 20.0, keep="STANDBY")
    run(operator, clock, 30.0)  # no state message for 30 s (> session_silent_s)
    assert timeout(operator)["elapsed_s"] == pytest.approx(50.0)
    run(operator, clock, 10.0, keep="STANDBY")
    assert len(calls.stops) == 1


def test_silent_when_due_waits_for_the_session():
    operator, clock, calls, _ = make()
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, LIMIT + 10.0)  # silent from the start: no session to stop
    assert calls.stops == []
    state(operator, clock, "STANDBY")
    operator.tick()
    assert len(calls.stops) == 1


def test_busy_gateway_channel_delays_the_stop_without_drops():
    operator, clock, calls, logs = make()
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, LIMIT - 1.0, keep="STANDBY")
    assert operator._send("gateway", "mirror_stop", clock.now) == "sent mirror_stop"  # a long cover's stop
    run(operator, clock, 1.5, every=0.1, keep="STANDBY")  # due, channel still busy
    assert len(calls.stops) == 1
    assert timeout(operator)["attempts"] == 0
    assert operator.dropped["gateway"] == 0
    assert not any("dropped" in message for _, message in logs)
    calls.stops[0](False, "SKILL_BUSY: another root execution is active")
    operator.tick()
    assert len(calls.stops) == 2
    assert timeout(operator)["attempts"] == 1


def test_touch_in_flight_does_not_delay_the_stop():
    # a touch holds the session channel; the stop goes on the Gateway channel
    operator, clock, calls, _ = make()
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, LIMIT - 1.0, keep="STANDBY")
    hold_touch(operator, clock)
    assert len(calls.touches) == 1
    run(operator, clock, 0.5, every=0.1, keep="STANDBY")
    assert len(calls.stops) == 1


def test_refused_stop_is_retried_every_retry_s():
    operator, clock, calls, _ = make()
    answer = (False, "SKILL_BUSY: another root execution is active")
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, LIMIT, keep="STANDBY")
    assert len(calls.stops) == 1
    calls.mirror_stop = lambda done: (calls.stops.append(done), done(*answer))  # answered at once from now on
    calls.stops[0](*answer)
    run(operator, clock, TIMEOUT_RETRY_S * 0.5, every=0.1, keep="STANDBY")
    assert len(calls.stops) == 1
    run(operator, clock, 10.0, every=0.1, keep="STANDBY")
    # 10 s: about one stop per TIMEOUT_RETRY_S
    assert 4 <= len(calls.stops) - 1 <= 6
    assert timeout(operator)["attempts"] == len(calls.stops)
    assert timeout(operator)["last"]["result"] == "failed"
    assert calls.direct == []  # a refusal is obeyed
    calls.mirror_stop = lambda done: (calls.stops.append(done), done(True, "stopped"))
    run(operator, clock, TIMEOUT_RETRY_S * 1.5, every=0.1, keep="STANDBY")
    sent = len(calls.stops)
    run(operator, clock, 10.0, every=0.1, keep="STANDBY")  # answered: no more
    assert len(calls.stops) == sent
    assert timeout(operator)["done"] is True


def test_unanswered_stop_goes_direct_then_retries_through_the_gateway():
    operator, clock, calls, _ = make()
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, LIMIT, keep="STANDBY")
    assert len(calls.stops) == 1
    run(operator, clock, GATEWAY_TIMEOUT_S + 0.5, every=0.1, keep="STANDBY")  # no answer from the Gateway
    assert len(calls.direct) == 1
    assert timeout(operator)["last"]["call"] == "stop"
    run(operator, clock, STOP_TIMEOUT_S + 0.5, every=0.1, keep="STANDBY")  # no answer from the session either
    assert len(calls.stops) == 2
    assert timeout(operator)["attempts"] == 2


def test_wait_clears_and_next_round_counts_again():
    operator, clock, calls, _ = make()
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, LIMIT, keep="STANDBY")
    calls.stops[0](True, "stopped")
    state(operator, clock, "WAIT", advance=1.0)
    assert timeout(operator)["elapsed_s"] is None
    assert timeout(operator)["fired"] is False
    run(operator, clock, 120.0, keep="WAIT")
    assert len(calls.stops) == 1
    state(operator, clock, "STANDBY")
    run(operator, clock, LIMIT - 1.0, keep="STANDBY")
    assert len(calls.stops) == 1
    run(operator, clock, 1.0, keep="STANDBY")
    assert len(calls.stops) == 2


def test_wait_before_due_clears():
    operator, clock, calls, _ = make()
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, 40.0, keep="MIRRORING")
    state(operator, clock, "WAIT")  # a robot-skill stop or a long cover
    state(operator, clock, "STANDBY", advance=1.0)
    run(operator, clock, 40.0, keep="STANDBY")
    assert calls.stops == []


def test_unknown_state_message_does_not_clear():
    operator, clock, calls, _ = make()
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, 30.0, keep="STANDBY")
    operator.on_state("not json")
    operator.on_state({"state": "BOGUS"})
    run(operator, clock, 30.0, keep="STANDBY")
    assert len(calls.stops) == 1


def test_send_disabled_records_once():
    operator, clock, calls, logs = make(send_enabled=False)
    state(operator, clock, "WAIT")
    state(operator, clock, "STANDBY")
    run(operator, clock, LIMIT + 30.0, every=0.1, keep="STANDBY")
    assert calls.stops == []
    assert timeout(operator)["attempts"] == 1
    assert timeout(operator)["done"] is True
    assert timeout(operator)["last"]["result"] == "not_sent"
    assert sum("session timeout" in message for _, message in logs) == 1
