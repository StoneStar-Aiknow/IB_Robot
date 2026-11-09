"""Characterization tests for the runtime-ownership boundary (design D5/D10).

These pin the behaviour of ``assess`` / ``takeover_refusal`` /
``release_to_idle`` and ``observation_from_status`` on the moved-but-unsplit
code, BEFORE the D5 split moves them to ``shared/runtime_ownership.py``. The
split may change only these tests' imports (openspec
reorganize-manipulation-execution-layout, task 2.4).
"""

from __future__ import annotations

import json

import pytest

from ibrobot_msgs.msg import RuntimeStatus
from manipulation_execution.shared.runtime_ownership import (
    IDLE,
    RuntimeObservation,
    WaitReason,
    assess,
    observation_from_status,
    release_to_idle,
    takeover_refusal,
)

POLICY_MODE = "policy_stream"
IDENTITY = {
    "id": "x2-ultra-001",
    "type": "aimdk_x2",
    "runtime_name": "x2_runtime",
    "runtime_version": "1.4.2",
}


def healthy_observation(
    *,
    active_mode: str = POLICY_MODE,
    stop_latched: bool = False,
    stamp_age_s: float | None = 0.5,
    robot: dict | None = None,
    runtime_name: str = IDENTITY["runtime_name"],
    runtime_version: str = IDENTITY["runtime_version"],
) -> RuntimeObservation:
    return RuntimeObservation(
        lifecycle="ACTIVE",
        active_mode=active_mode,
        stop_latched=stop_latched,
        faults=(),
        runtime_name=runtime_name,
        runtime_version=runtime_version,
        robot=dict(robot if robot is not None else IDENTITY),
        stamp_age_s=stamp_age_s,
    )


class TestObservationFromStatus:
    def test_a_parsable_description_and_valid_stamp_produce_the_observation(self):
        status = RuntimeStatus()
        status.lifecycle = "ACTIVE"
        status.active_mode = POLICY_MODE
        status.stop_latched = False
        status.faults = []
        status.runtime_name = IDENTITY["runtime_name"]
        status.runtime_version = IDENTITY["runtime_version"]
        status.interface_description_json = json.dumps({"robot": IDENTITY})
        status.stamp.sec = 100
        status.stamp.nanosec = 500_000_000

        observation = observation_from_status(status, now_ns=101_000_000_000)

        assert observation.lifecycle == "ACTIVE"
        assert observation.active_mode == POLICY_MODE
        assert observation.stop_latched is False
        assert observation.runtime_name == IDENTITY["runtime_name"]
        assert observation.runtime_version == IDENTITY["runtime_version"]
        assert observation.robot == IDENTITY
        assert observation.stamp_age_s == 0.5

    def test_malformed_json_yields_no_robot_but_keeps_the_other_fields(self):
        status = RuntimeStatus()
        status.lifecycle = "DEGRADED"
        status.active_mode = "trajectory"
        status.interface_description_json = "{not json"
        status.stamp.sec = 10
        status.stamp.nanosec = 0

        observation = observation_from_status(status, now_ns=15_000_000_000)

        assert observation.robot is None
        assert observation.lifecycle == "DEGRADED"
        assert observation.stamp_age_s == 5.0

    def test_non_object_json_yields_no_robot(self):
        status = RuntimeStatus()
        status.interface_description_json = "[1, 2, 3]"
        status.stamp.sec = 0
        status.stamp.nanosec = 0

        observation = observation_from_status(status, now_ns=1_000_000_000)

        assert observation.robot is None

    def test_robot_key_that_is_not_an_object_yields_no_robot(self):
        status = RuntimeStatus()
        status.interface_description_json = json.dumps({"robot": "x2"})
        status.stamp.sec = 0
        status.stamp.nanosec = 0

        observation = observation_from_status(status, now_ns=0)

        assert observation.robot is None

    @pytest.mark.parametrize("nanosec", [1_000_000_000, 4_294_967_295])
    def test_malformed_stamp_nanosec_yields_no_stamp_age(self, nanosec):
        # The uint32 field accepts values beyond one second; the observation
        # treats those stamps as malformed and reports no age.
        status = RuntimeStatus()
        status.stamp.sec = 5
        status.stamp.nanosec = nanosec

        observation = observation_from_status(status, now_ns=6_000_000_000)

        assert observation.stamp_age_s is None


class TestAssess:
    def test_a_missing_observation_is_runtime_lost(self):
        assert assess(None, IDENTITY) == (WaitReason.RUNTIME_LOST, "runtime status unavailable")

    def test_a_stale_stamp_is_runtime_lost(self):
        problem = assess(healthy_observation(stamp_age_s=2.5), IDENTITY)
        assert problem == (WaitReason.RUNTIME_LOST, "runtime status stamp is not fresh")

    def test_a_malformed_stamp_is_runtime_lost(self):
        problem = assess(healthy_observation(stamp_age_s=None), IDENTITY)
        assert problem == (WaitReason.RUNTIME_LOST, "runtime status stamp is not fresh")

    def test_a_future_stamp_is_runtime_lost(self):
        problem = assess(healthy_observation(stamp_age_s=-0.5), IDENTITY)
        assert problem == (WaitReason.RUNTIME_LOST, "runtime status stamp is not fresh")

    def test_a_different_robot_identity_is_revoked(self):
        other_robot = {**IDENTITY, "id": "another-unit"}
        problem = assess(healthy_observation(robot=other_robot), IDENTITY)
        assert problem == (WaitReason.REVOKED, "runtime identity does not match the bound description")

    def test_a_different_runtime_name_is_revoked(self):
        problem = assess(healthy_observation(runtime_name="other_runtime"), IDENTITY)
        assert problem == (WaitReason.REVOKED, "runtime identity does not match the bound description")

    def test_a_different_runtime_version_is_revoked(self):
        problem = assess(healthy_observation(runtime_version="0.0.1"), IDENTITY)
        assert problem == (WaitReason.REVOKED, "runtime identity does not match the bound description")

    def test_a_null_identity_skips_the_identity_check(self):
        assert assess(healthy_observation(robot=None, runtime_name="any", runtime_version="any"), None) is None

    def test_a_stop_latch_beats_a_healthy_status(self):
        problem = assess(healthy_observation(stop_latched=True), IDENTITY)
        assert problem == (WaitReason.STOP_LATCHED, "runtime stop is latched")

    def test_a_healthy_status_in_any_mode_assesses_clean(self):
        # The active mode is judged by takeover_refusal, not by assess.
        assert assess(healthy_observation(active_mode="trajectory"), IDENTITY) is None


class TestTakeoverRefusal:
    def test_a_healthy_status_in_the_policy_mode_may_be_taken_over(self):
        assert takeover_refusal(healthy_observation(), IDENTITY, POLICY_MODE) is None

    def test_an_identity_mismatch_refuses_the_takeover(self):
        reason = takeover_refusal(healthy_observation(runtime_name="other_runtime"), IDENTITY, POLICY_MODE)
        assert reason == "runtime identity does not match the bound description"

    def test_a_stop_latch_refuses_the_takeover(self):
        reason = takeover_refusal(healthy_observation(stop_latched=True), IDENTITY, POLICY_MODE)
        assert reason == "runtime stop is latched"

    def test_an_idle_runtime_is_never_taken_over(self):
        reason = takeover_refusal(healthy_observation(active_mode=IDLE), IDENTITY, POLICY_MODE)
        assert reason == "runtime is idle: the session only takes over a mode the Gateway switched to"

    def test_a_mode_owned_by_another_producer_is_refused(self):
        reason = takeover_refusal(healthy_observation(active_mode="trajectory"), IDENTITY, POLICY_MODE)
        assert reason == "runtime mode 'trajectory' is owned by another producer"

    def test_a_stale_status_refuses_the_takeover(self):
        reason = takeover_refusal(healthy_observation(stamp_age_s=30.0), IDENTITY, POLICY_MODE)
        assert reason == "runtime status stamp is not fresh"


class FakePort:
    """A ``RuntimePort`` scripted per call, recording every request and the
    timeout each call received."""

    def __init__(self, statuses, set_mode_result=None):
        self._statuses = list(statuses)
        self._set_mode_result = set_mode_result
        self.set_mode_calls: list[str] = []
        self.get_status_timeouts: list[float] = []
        self.set_mode_timeouts: list[float] = []

    def get_status(self, timeout_s: float) -> RuntimeObservation | None:
        self.get_status_timeouts.append(timeout_s)
        return self._statuses.pop(0) if self._statuses else None

    def set_mode(self, mode: str, timeout_s: float) -> tuple[bool, str]:
        self.set_mode_calls.append(mode)
        self.set_mode_timeouts.append(timeout_s)
        return self._set_mode_result


class TestReleaseToIdle:
    def test_an_unavailable_status_sends_nothing(self):
        port = FakePort([None])
        outcome = release_to_idle(port, POLICY_MODE, budget_s=1.0)
        assert (outcome.idle, outcome.sent_idle, outcome.message) == (
            False,
            False,
            "runtime status unavailable; idle not requested",
        )
        assert port.set_mode_calls == []

    def test_an_idle_runtime_is_already_released(self):
        port = FakePort([healthy_observation(active_mode=IDLE)])
        outcome = release_to_idle(port, POLICY_MODE, budget_s=1.0)
        assert (outcome.idle, outcome.sent_idle, outcome.message) == (True, False, "runtime already idle")
        assert port.set_mode_calls == []

    def test_a_stop_latch_never_receives_an_idle_request(self):
        port = FakePort([healthy_observation(stop_latched=True)])
        outcome = release_to_idle(port, POLICY_MODE, budget_s=1.0)
        assert (outcome.idle, outcome.sent_idle, outcome.message) == (
            False,
            False,
            "runtime stop is latched; idle is never requested over a latch",
        )
        assert port.set_mode_calls == []

    def test_a_foreign_mode_is_not_ours_to_release(self):
        port = FakePort([healthy_observation(active_mode="trajectory")])
        outcome = release_to_idle(port, POLICY_MODE, budget_s=1.0)
        assert (outcome.idle, outcome.sent_idle, outcome.message) == (
            False,
            False,
            "runtime mode 'trajectory' is not 'policy_stream'; not ours to release",
        )
        assert port.set_mode_calls == []

    def test_a_refused_idle_request_reports_the_refusal(self):
        port = FakePort([healthy_observation()], set_mode_result=(False, "stop latched (HOLD)"))
        outcome = release_to_idle(port, POLICY_MODE, budget_s=1.0)
        assert (outcome.idle, outcome.sent_idle, outcome.message) == (
            False,
            True,
            "idle request refused: stop latched (HOLD)",
        )
        assert port.set_mode_calls == [IDLE]

    def test_an_idle_request_that_times_out_on_the_reread_is_not_success(self):
        port = FakePort([healthy_observation(), None], set_mode_result=(True, "ok"))
        outcome = release_to_idle(port, POLICY_MODE, budget_s=1.0)
        assert (outcome.idle, outcome.sent_idle, outcome.message) == (
            False,
            True,
            "idle requested but runtime reports None",
        )

    def test_a_successful_release_observes_idle_after_the_request(self):
        port = FakePort(
            [healthy_observation(), healthy_observation(active_mode=IDLE)],
            set_mode_result=(True, "ok"),
        )
        outcome = release_to_idle(port, POLICY_MODE, budget_s=1.0)
        assert (outcome.idle, outcome.sent_idle, outcome.message) == (True, True, "runtime returned to idle")
        assert port.set_mode_calls == [IDLE]

    def test_each_subsequent_rpc_gets_a_shrinking_share_of_the_budget(self):
        # clock() is read once for the deadline and once per RPC; advancing it
        # 0.1 s per read makes every later call receive less budget than the one
        # before it.
        ticks = iter([0.0, 0.1, 0.2, 0.3])

        def clock() -> float:
            return next(ticks)

        port = FakePort(
            [healthy_observation(), healthy_observation(active_mode=IDLE)],
            set_mode_result=(True, "ok"),
        )

        outcome = release_to_idle(port, POLICY_MODE, budget_s=1.0, clock=clock)

        assert outcome.idle
        assert port.get_status_timeouts[0] == pytest.approx(0.9)
        assert port.set_mode_timeouts == [pytest.approx(0.8)]
        assert port.get_status_timeouts[1] == pytest.approx(0.7)
        assert port.get_status_timeouts[0] > port.set_mode_timeouts[0] > port.get_status_timeouts[1]
