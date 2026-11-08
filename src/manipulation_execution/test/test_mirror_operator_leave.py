"""The operator's leave stop (``leave_stop_min_round_s``): a long round the person walked away from. No ROS."""

from __future__ import annotations

import pytest

from manipulation_execution.mirror_operator.config import OperatorConfig
from manipulation_execution.mirror_operator.core import TIMEOUT_RETRY_S, Operator
from manipulation_execution.mirror_operator.judge import CoverJudge

MIN_ROUND = 10.0
SESSION_LIMIT = 60.0

WAIT = {"state": "WAIT", "phase": "none", "cancel_reason": ""}
APPROACH = {"state": "STANDBY", "phase": "approach", "cancel_reason": ""}
READY = {"state": "STANDBY", "phase": "ready", "cancel_reason": ""}
RUNNING = {"state": "MIRRORING", "phase": "running", "cancel_reason": ""}


def cancelled(reason):
    """The messages of a cancel: the wrap-up, then STANDBY/cooldown and ready; cancel_reason set on all."""
    return [
        {"state": "MIRRORING", "phase": "wrap_up", "cancel_reason": reason},
        {"state": "STANDBY", "phase": "cooldown", "cancel_reason": reason},
        {"state": "STANDBY", "phase": "ready", "cancel_reason": reason},
    ]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Calls:
    """Records every call; the test answers the held ``done`` callbacks itself."""

    def __init__(self):
        self.stops = []  # mirror_stop through the Gateway
        self.direct = []

    def mirror_start(self, done):
        raise AssertionError("no mirror_start expected")

    def mirror_stop(self, done):
        self.stops.append(done)

    def stop_session(self, done):
        self.direct.append(done)

    def operator_touch(self, done):
        raise AssertionError("no touch expected")

    def set_led(self, color, priority, done):
        pass


def make(min_round=MIN_ROUND, limit=SESSION_LIMIT, send_enabled=True):
    clock, calls, logs = Clock(), Calls(), []
    config = OperatorConfig.from_mapping({"max_session_s": limit, "leave_stop_min_round_s": min_round})
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


def run(operator, clock, seconds, keep, *, every=0.1):
    """Tick for ``seconds`` with ``keep`` sent every tick (the session's heartbeat)."""
    end = clock.now + seconds
    while clock.now < end:
        clock.now = min(end, clock.now + every)
        operator.on_state(keep)
        operator.tick()


def start_round(operator, clock):
    """mirror_start, the approach, and a person who starts a round."""
    run(operator, clock, 1.0, WAIT)
    run(operator, clock, 3.0, APPROACH)
    run(operator, clock, 2.0, READY)


def play(operator, clock, seconds, reason):
    """A round of ``seconds`` cancelled for ``reason``, then 2 s of each cancel message."""
    run(operator, clock, seconds, RUNNING)
    for message in cancelled(reason):
        run(operator, clock, 2.0, message)


def leave(operator):
    return operator.status()["leave_stop"]


def test_config_default_is_off_and_zero_is_accepted():
    assert OperatorConfig().leave_stop_min_round_s == 0.0
    assert OperatorConfig.from_mapping({"leave_stop_min_round_s": 0}).leave_stop_min_round_s == 0.0
    assert OperatorConfig.from_mapping({"leave_stop_min_round_s": 10}).leave_stop_min_round_s == 10.0
    with pytest.raises(ValueError, match="leave_stop_min_round_s"):
        OperatorConfig.from_mapping({"leave_stop_min_round_s": -1})


@pytest.mark.parametrize("reason", ["no_actor", "too_far"])
def test_a_long_round_the_person_left_stops_once_back_in_standby(reason):
    operator, clock, calls, logs = make()
    start_round(operator, clock)
    run(operator, clock, MIN_ROUND + 1.0, RUNNING)
    wrap_up, cooldown, _ = cancelled(reason)
    run(operator, clock, 2.0, wrap_up)
    assert calls.stops == []  # not during the wrap-up: the arm is still on its way back
    operator.on_state(cooldown)
    operator.tick()
    assert len(calls.stops) == 1
    assert leave(operator)["last"]["stop"] is True
    assert leave(operator)["last"]["round_s"] == pytest.approx(MIN_ROUND + 1.0, abs=0.2)
    assert operator.status()["session_timeout"]["cause"] == "leave"
    calls.stops[0](True, "stopped")
    run(operator, clock, 10.0, cooldown)  # answered, WAIT not yet: no second stop
    assert len(calls.stops) == 1
    assert any("leave stop" in message for _, message in logs)


def test_a_short_round_stays_in_standby():
    # a round that ended within MIN_ROUND (a flash): the person may still be there
    operator, clock, calls, _ = make()
    start_round(operator, clock)
    play(operator, clock, MIN_ROUND - 1.0, "no_actor")
    run(operator, clock, 20.0, cancelled("no_actor")[2])
    assert calls.stops == []
    assert leave(operator)["last"]["stop"] is False


@pytest.mark.parametrize("reason", ["stale", "no_pose", "operator_touch", "criteria"])
def test_other_cancel_reasons_stay_in_standby(reason):
    operator, clock, calls, _ = make()
    start_round(operator, clock)
    play(operator, clock, MIN_ROUND + 5.0, reason)
    run(operator, clock, 20.0, cancelled(reason)[2])
    assert calls.stops == []
    assert leave(operator)["due"] is False


def test_the_first_standby_without_a_round_waits_for_the_session_timeout():
    operator, clock, calls, logs = make()
    start_round(operator, clock)
    run(operator, clock, SESSION_LIMIT - 7.0, READY)  # the timeout started with APPROACH, 5 s ago
    assert calls.stops == []
    run(operator, clock, 3.0, READY)
    assert len(calls.stops) == 1
    assert operator.status()["session_timeout"]["cause"] == "session_timeout"
    assert any("session timeout" in message for _, message in logs)


def test_a_stop_during_a_long_round_is_not_a_leave():
    # mirror_stop from MIRRORING goes through STANDBY/wrap_up to WAIT; the cancel_reason of an
    # earlier round is still in the message
    operator, clock, calls, _ = make()
    start_round(operator, clock)
    play(operator, clock, 3.0, "no_actor")  # a short round: stays
    run(operator, clock, MIN_ROUND + 1.0, RUNNING)
    run(operator, clock, 2.0, {"state": "STANDBY", "phase": "wrap_up", "cancel_reason": "no_actor"})
    run(operator, clock, 2.0, WAIT)
    assert calls.stops == []


def test_a_new_round_before_the_stop_is_answered_calls_it_off():
    operator, clock, calls, _ = make()
    start_round(operator, clock)
    play(operator, clock, MIN_ROUND + 1.0, "no_actor")
    assert len(calls.stops) == 1
    calls.stops[0](False, "SKILL_BUSY: another root execution is active")
    run(operator, clock, TIMEOUT_RETRY_S + 0.5, cancelled("no_actor")[2])
    assert len(calls.stops) == 2  # a refusal is retried, as the timeout's
    calls.stops[1](False, "SKILL_BUSY: another root execution is active")
    run(operator, clock, 5.0, RUNNING)  # the person came back and a round started
    assert len(calls.stops) == 2
    assert leave(operator)["due"] is False


def test_wait_clears_and_the_next_session_counts_again():
    operator, clock, calls, _ = make()
    start_round(operator, clock)
    play(operator, clock, MIN_ROUND + 1.0, "no_actor")
    calls.stops[0](True, "stopped")
    run(operator, clock, 5.0, WAIT)
    assert leave(operator)["due"] is False
    assert operator.status()["session_timeout"]["cause"] == ""
    start_round(operator, clock)
    play(operator, clock, MIN_ROUND + 1.0, "no_actor")
    assert len(calls.stops) == 2


def test_off_never_stops_on_a_leave():
    operator, clock, calls, _ = make(min_round=0.0, limit=0.0)
    start_round(operator, clock)
    play(operator, clock, MIN_ROUND + 20.0, "no_actor")
    run(operator, clock, 30.0, READY)
    assert calls.stops == []
    assert leave(operator)["enabled"] is False


def test_messages_without_a_phase_never_stop_on_a_leave():
    # an older session publishes the state only
    operator, clock, calls, _ = make(limit=0.0)
    run(operator, clock, 1.0, {"state": "WAIT"})
    run(operator, clock, 2.0, {"state": "STANDBY"})
    run(operator, clock, MIN_ROUND + 5.0, {"state": "MIRRORING"})
    run(operator, clock, 10.0, {"state": "STANDBY", "cancel_reason": "no_actor"})
    assert calls.stops == []


def test_send_disabled_records_once():
    operator, clock, calls, logs = make(send_enabled=False)
    start_round(operator, clock)
    play(operator, clock, MIN_ROUND + 1.0, "no_actor")
    run(operator, clock, 10.0, READY)
    assert calls.stops == []
    assert operator.status()["session_timeout"]["last"]["result"] == "not_sent"
    assert sum("leave stop (attempt" in message for _, message in logs) == 1
