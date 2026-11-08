"""Placement contract: release once, stop on faults, preserve custody facts."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_grasp_pipeline import JOINTS, Effector, Executor, Source, trajectory

from manipulation_execution.grasp_pipeline.contracts import (
    ContainerEstimate,
    Effect,
    EffectorAction,
    GraspFailure,
    HeldObject,
    Move,
    PlaceBackends,
    PlacePlan,
    PlaceVerdict,
)
from manipulation_execution.grasp_pipeline.place_pipeline import PlacePipeline


def fixture(*, verdict=None, planning_failure=None, release_confirmed=True, fail_on=None, cancel_on=None):
    held = HeldObject("grip", "a red apple", "left", JOINTS, (0.0,) * 3, (0.0,) * 3, 0.04, (0.0,) * 3, "{}")
    executor, effector = Executor(fail_on=fail_on, cancel_on=cancel_on), Effector()
    effector.released = lambda: release_confirmed
    container = ContainerEstimate(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([0.0, 0.0, 1.0]), 0)
    plan = PlacePlan(
        (Move(trajectory("transfer", np.zeros(3), np.ones(3))),),
        (Effect(EffectorAction.RELEASE),),
        (Move(trajectory("retreat", np.ones(3), np.zeros(3))),),
        container,
    )

    def planning(*args, **kwargs):
        if planning_failure:
            raise GraspFailure(planning_failure, "no plan")
        return plan

    backends = PlaceBackends(
        Source(),
        SimpleNamespace(propose=lambda *a, **k: container),
        SimpleNamespace(plan=planning),
        SimpleNamespace(verify=lambda *a, **k: verdict or PlaceVerdict(True)),
        executor,
        effector,
        lambda _: setattr(effector, "gripped", True),
    )
    return held, backends


def run(held, backends, **kwargs):
    return PlacePipeline(backends).run(
        held, "the blue bowl on the left", deadline=float("inf"), cancelled=lambda: False, **kwargs
    )


def test_release_retreat_and_verification_are_all_required_for_success():
    held, backends = fixture()
    outcome = run(held, backends)
    assert outcome.success and outcome.release_status == "released" and outcome.custody_outcome == "released"
    assert outcome.verification == "success" and np.allclose(backends.executor.position, 0)
    assert backends.effector.log == [EffectorAction.RELEASE]
    assert "verify_place" in outcome.completed_phases


@pytest.mark.parametrize("reason", ["PLACE_UNSEEN", "PLACED_OUTSIDE"])
def test_visual_failure_does_not_claim_object_is_still_held(reason):
    held, backends = fixture(verdict=PlaceVerdict(False, reason))
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == reason
    assert outcome.custody_outcome == "released" and outcome.release_status == "released"
    assert "verify_place" not in outcome.completed_phases
    assert outcome.execution_detail.failed_phase == "verify_place"


def test_no_plan_returns_custody_without_releasing_or_moving():
    held, backends = fixture(planning_failure="PLACE_PLAN_FAILED")
    outcome = run(held, backends)
    assert not outcome.success and outcome.custody_outcome == "held"
    assert not backends.executor.log and not backends.effector.log


def test_unconfirmed_release_stops_before_retreat():
    held, backends = fixture(release_confirmed=False)
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == "RELEASE_UNCERTAIN"
    assert outcome.release_status == "unknown" and outcome.custody_outcome == "uncertain"
    assert [row[0] for row in backends.executor.log] == ["transfer"]


@pytest.mark.parametrize("option", [{"fail_on": "transfer"}, {"cancel_on": "transfer"}])
def test_controller_fault_or_cancel_never_releases(option):
    held, backends = fixture(**option)
    outcome = run(held, backends)
    assert not outcome.success and outcome.custody_outcome == "uncertain"
    assert backends.executor.stopped and not backends.effector.log
    assert outcome.execution_detail.recovery_state == "FAILED"  # Stopped, but object custody needs recovery.


def test_unknown_stop_preserves_the_original_failure_and_keeps_motion_quarantined():
    held, backends = fixture(fail_on="transfer")

    def stop_unconfirmed(_reason):
        raise RuntimeError("controller stop acknowledgement lost")

    backends.executor.stop = stop_unconfirmed
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == "MOTION_FAILED"
    assert outcome.custody_outcome == "uncertain" and not backends.effector.log
    assert outcome.execution_detail.recovery_state == "UNKNOWN"
    assert "stop acknowledgement lost" in outcome.message
    assert outcome.report["first_execution_failure"]["code"] == "MOTION_FAILED"
    assert "stop acknowledgement lost" not in outcome.report["first_execution_failure"]["message"]


def test_unknown_grip_during_transfer_stops_without_retrace():
    held, backends = fixture()
    holding = backends.effector.holding
    backends.effector.holding = lambda: None if backends.executor.position[0] > 0.4 else holding()
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == "GRASP_UNCERTAIN"
    assert outcome.custody_outcome == "uncertain" and not backends.effector.log
    assert [row[0] for row in backends.executor.log] == ["transfer"]


@pytest.mark.parametrize(
    "grip, reason", [(None, "GRASP_UNCERTAIN"), ("contact fingers stopped pressing", "GRIP_LOST_IN_TRANSFER")]
)
def test_grip_failure_after_capture_keeps_its_evidence_without_starting_transfer(grip, reason):
    held, backends = fixture()
    capture = backends.observation.capture

    def capture_then_lose_feedback(**kwargs):
        observation = capture(**kwargs)
        backends.effector.holding = lambda: grip
        return observation

    backends.observation.capture = capture_then_lose_feedback
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == reason
    assert outcome.custody_outcome == "uncertain"
    assert outcome.execution_detail.failed_phase == "observe_container"
    assert outcome.report["grip_failure"] == {"phase": "observe_container", "reason": grip}
    assert ("feedback unavailable" if grip is None else grip) in outcome.message
    assert not backends.executor.log and not backends.effector.log


def test_retry_success_clears_failure_detail():
    held, backends = fixture()
    plan = backends.planner.plan
    calls = []

    def retry(*args, **kwargs):
        calls.append(None)
        if len(calls) == 1:
            raise GraspFailure("PLACE_PLAN_FAILED", "try another capture", retryable=True)
        return plan(*args, **kwargs)

    backends.planner.plan = retry
    outcome = run(held, backends)
    assert outcome.success and len(calls) == 2
    assert outcome.execution_detail.retry_count == 1
    assert outcome.execution_detail.failed_phase == outcome.execution_detail.reason_code == ""


def test_cleanup_failure_preserves_confirmed_release():
    held, backends = fixture()

    def broken_finish():
        raise RuntimeError("stream cleanup failed")

    backends.executor.finish = broken_finish
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == "CLEANUP_FAILED"
    assert outcome.release_status == "released"
    assert outcome.execution_detail.suggested_action == "MANUAL_CHECK"


def test_budget_expiry_during_transfer_stops_before_release():
    held, backends = fixture()
    now = [0.0]
    execute = backends.executor.execute

    def expires(*args, **kwargs):
        now[0] = 2.0
        return execute(*args, **kwargs)

    backends.executor.execute = expires
    outcome = PlacePipeline(backends, execute_reserve_s=0, clock=lambda: now[0]).run(
        held, "the blue bowl", deadline=1.0, cancelled=lambda: False
    )
    assert not outcome.success and outcome.reason_code == "SKILL_TIMEOUT"
    assert backends.executor.stopped and not backends.effector.log


@pytest.mark.parametrize("fault", ["unknown_grip", "lost_grip", "moved", "cancelled"])
def test_retry_requires_confirmed_grip_unchanged_handover_and_no_cancel(fault):
    held, backends = fixture()
    calls, cancelled = [], [False]

    def plan(*args, **kwargs):
        calls.append(None)
        if fault == "unknown_grip":
            backends.effector.holding = lambda: None
        elif fault == "lost_grip":
            backends.effector.holding = lambda: "contact lost"
        elif fault == "moved":
            backends.executor.position[:] = 0.03
        else:
            cancelled[0] = True
        raise GraspFailure("PLACE_PLAN_FAILED", "no path in first snapshot", retryable=True)

    backends.planner.plan = plan
    outcome = PlacePipeline(backends).run(held, "bowl", deadline=float("inf"), cancelled=lambda: cancelled[0])
    assert not outcome.success and len(calls) == backends.observation.captures == 1
    assert (
        outcome.reason_code
        == {
            "unknown_grip": "GRASP_UNCERTAIN",
            "lost_grip": "GRIP_LOST_IN_TRANSFER",
            "moved": "HELD_OBJECT_MISMATCH",
            "cancelled": "CANCELED",
        }[fault]
    )
    assert outcome.report["first_failure"]["code"] == "PLACE_PLAN_FAILED"
    assert not backends.effector.log and not backends.executor.log


def test_recoverable_planning_failure_is_bounded_and_keeps_each_failure():
    held, backends = fixture()

    def plan(*args, **kwargs):
        raise GraspFailure("PLACE_PLAN_FAILED", f"snapshot {backends.observation.captures}", retryable=True)

    backends.planner.plan = plan
    outcome = run(held, backends)
    assert not outcome.success and backends.observation.captures == 2
    assert outcome.custody_outcome == "held" and outcome.release_status == "not_released"
    assert outcome.report["first_failure"]["message"] == "snapshot 1"
    assert [row["message"] for row in outcome.report["failures"]] == ["snapshot 1", "snapshot 2"]
    assert len(outcome.report["retries"]) == 1
    assert outcome.report["failures"][-1]["retry_decision"] == "attempt_limit"
    assert not backends.effector.log and not backends.executor.log


def test_retry_keeps_execution_reserve_and_requires_time_to_repeat_preparation():
    held, backends = fixture()
    now = [0.0]

    def plan(*args, deadline, **kwargs):
        assert deadline == 85.0
        now[0] = 45.0  # 40 s of planning time left cannot repeat a 45 s preparation.
        raise GraspFailure("PLACE_PLAN_FAILED", "search exhausted", retryable=True)

    backends.planner.plan = plan
    outcome = PlacePipeline(backends, clock=lambda: now[0]).run(held, "bowl", deadline=100.0, cancelled=lambda: False)
    assert not outcome.success and outcome.reason_code == "PLACE_PLAN_FAILED"
    assert backends.observation.captures == 1 and not outcome.report.get("retries")
    assert outcome.report["failures"][0]["retry_decision"] == "insufficient_budget"
    assert outcome.custody_outcome == "held" and not backends.effector.log and not backends.executor.log


@pytest.mark.parametrize("overrun", [False, True])
def test_no_motion_starts_when_execution_reserve_is_unavailable(overrun):
    held, backends = fixture()
    now = [0.0]
    original = backends.planner.plan

    def plan(*args, deadline, **kwargs):
        assert deadline == 85.0
        now[0] = 86.0
        return original(*args, deadline=deadline, **kwargs)

    backends.planner.plan = plan
    outcome = PlacePipeline(backends, clock=lambda: now[0]).run(
        held, "bowl", deadline=100.0 if overrun else 10.0, cancelled=lambda: False
    )
    assert outcome.reason_code == "EXECUTION_BUDGET_INSUFFICIENT"
    assert backends.observation.captures == int(overrun)
    assert not backends.executor.log and not backends.effector.log


def test_cleanup_failure_does_not_erase_first_planning_failure():
    held, backends = fixture(planning_failure="PLACE_PLAN_FAILED")

    def finish():
        raise RuntimeError("cleanup transport failed")

    backends.executor.finish = finish
    outcome = run(held, backends)
    assert outcome.reason_code == "CLEANUP_FAILED"
    assert outcome.report["first_failure"]["code"] == "PLACE_PLAN_FAILED"
    assert [row["code"] for row in outcome.report["failures"]] == ["PLACE_PLAN_FAILED", "CLEANUP_FAILED"]
    assert not backends.effector.log


@pytest.mark.parametrize("reserve", [-1, True, "15", float("inf"), float("nan")])
def test_invalid_execution_reserve_is_rejected(reserve):
    _, backends = fixture()
    with pytest.raises(ValueError, match="execute_reserve_s"):
        PlacePipeline(backends, execute_reserve_s=reserve)


def dropping(backends):
    """The grip slips once the transfer is past its middle."""
    holding = backends.effector.holding
    backends.effector.holding = lambda: "object slipped" if backends.executor.position[0] > 0.4 else holding()


def test_drop_in_transfer_opens_the_hand_and_returns_home_on_a_fresh_plan():
    held, backends = fixture()
    dropping(backends)
    home = np.full(3, -0.2)
    requests = []

    def plan_home(held_object, *, deadline, cancelled):
        requests.append(held_object)
        return Move(trajectory("home", backends.executor.position.copy(), home))

    backends.planner.plan_home = plan_home
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == "GRIP_LOST_IN_TRANSFER"
    assert outcome.custody_outcome == "lost" and backends.effector.log == [EffectorAction.RELEASE]
    assert requests == [held] and np.allclose(backends.executor.position, home)
    assert [row[0] for row in backends.executor.log][-1] == "home" and not backends.executor.stopped
    assert outcome.execution_detail.recovery_state == "COMPLETED"
    assert outcome.execution_detail.suggested_action == "WAIT_USER"


def test_partial_transfer_rollback_failure_keeps_the_original_drop():
    held, backends = fixture(fail_on="retreat_transfer_partial")
    dropping(backends)
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == "MOTION_FAILED"
    first = outcome.report["first_execution_failure"]
    assert first["phase"] == "transfer" and first["code"] == "GRIP_LOST_IN_TRANSFER"
    assert first["message"] == "object slipped"
    assert "tracking error" in outcome.message
    assert backends.executor.stopped and not backends.effector.log


def test_drop_with_no_way_home_stops_where_it_released_for_a_manual_check():
    held, backends = fixture()
    dropping(backends)

    def plan_home(held_object, *, deadline, cancelled):
        raise GraspFailure("HOME_PLAN_FAILED", "start state in collision")

    backends.planner.plan_home = plan_home
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == "GRIP_LOST_IN_TRANSFER"
    assert outcome.custody_outcome == "lost" and backends.effector.log == [EffectorAction.RELEASE]
    assert "home" not in [row[0] for row in backends.executor.log] and backends.executor.stopped
    assert "could not return home" in outcome.message
    assert outcome.execution_detail.recovery_state == "FAILED"
    assert outcome.execution_detail.suggested_action == "MANUAL_CHECK"


def test_confirmed_release_feedback_precedes_retreat_and_success_waits_for_cleanup():
    held, backends = fixture()
    events, finished = [], []
    backends.executor.finish = lambda: finished.append("executor")
    backends.effector.finish = lambda: finished.append("effector")

    def phase(name, event, data):
        detail = data["execution_detail"]
        events.append((name, event, detail))
        assert not detail["place_succeeded"] and not finished
        if name == "release_confirmed":
            assert backends.effector.log == [EffectorAction.RELEASE]
            assert [row[0] for row in backends.executor.log] == ["transfer"]
            assert detail["release_status"] == "RELEASED"

    outcome = run(held, backends, on_phase=phase)
    assert finished == ["executor", "effector"]
    assert any(name == "release_confirmed" for name, _, _ in events)
    assert outcome.execution_detail.place_succeeded
    assert outcome.execution_detail.verification_status == "SUCCESS"


@pytest.mark.parametrize("option", [{"fail_on": "retreat"}, {"cancel_on": "retreat"}])
def test_post_release_motion_failure_preserves_public_facts(option):
    held, backends = fixture(**option)
    outcome = run(held, backends)
    assert not outcome.success
    assert outcome.execution_detail.release_status == "RELEASED"
    assert outcome.execution_detail.verification_status == "NOT_RUN"
    assert not outcome.execution_detail.place_succeeded


def test_cleanup_failure_after_visual_success_retains_public_facts():
    held, backends = fixture()

    def fail():
        raise RuntimeError("cleanup failed")

    backends.executor.finish = fail
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == "CLEANUP_FAILED"
    assert outcome.execution_detail.release_status == "RELEASED"
    assert outcome.execution_detail.verification_status == "SUCCESS"
    assert not outcome.execution_detail.place_succeeded
