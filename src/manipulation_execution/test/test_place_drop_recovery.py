"""Preparation-stage drops close custody without hiding recovery failures."""

import numpy as np
import pytest
from test_grasp_pipeline import trajectory
from test_place_pipeline import fixture, run

from manipulation_execution.grasp_pipeline.contracts import EffectorAction, GraspFailure, Move


def lose_after_capture(backends, grip="contact fingers no longer press the object"):
    capture = backends.observation.capture

    def capture_then_lose(**kwargs):
        observation = capture(**kwargs)
        backends.effector.holding = lambda: grip
        return observation

    backends.observation.capture = capture_then_lose


def enable_home(backends):
    def plan_home(held, *, deadline, cancelled):
        return Move(trajectory("recovery_home", backends.executor.position.copy(), np.full(3, -0.2)))

    backends.planner.plan_home = plan_home


def test_preparation_drop_opens_empty_hand_and_returns_home_before_releasing_custody():
    held, backends = fixture()
    lose_after_capture(backends)
    enable_home(backends)
    outcome = run(held, backends)
    assert not outcome.success and outcome.reason_code == "GRIP_LOST_IN_TRANSFER"
    assert outcome.custody_outcome == "lost" and outcome.release_status == "released"
    assert outcome.verification == "not_run" and not outcome.execution_detail.place_succeeded
    assert outcome.execution_detail.failed_phase == "observe_container"
    assert outcome.execution_detail.recovery_state == "COMPLETED"
    assert outcome.execution_detail.suggested_action == "WAIT_USER"
    assert backends.effector.log == [EffectorAction.RELEASE]
    assert [row[0] for row in backends.executor.log] == ["recovery_home"]
    assert np.allclose(backends.executor.position, -0.2)
    assert not backends.executor.stopped


def test_unknown_preparation_feedback_never_opens_or_moves_even_with_a_home_planner():
    held, backends = fixture()
    lose_after_capture(backends, None)
    enable_home(backends)
    outcome = run(held, backends)
    assert outcome.reason_code == "GRASP_UNCERTAIN" and outcome.custody_outcome == "uncertain"
    assert not backends.executor.log and not backends.effector.log


def test_unconfirmed_recovery_release_keeps_custody_and_never_starts_home():
    held, backends = fixture(release_confirmed=False)
    lose_after_capture(backends)
    enable_home(backends)
    outcome = run(held, backends)
    assert outcome.reason_code == "GRIP_LOST_IN_TRANSFER"
    assert outcome.custody_outcome == "uncertain" and outcome.release_status == "unknown"
    assert outcome.execution_detail.recovery_state == "FAILED"
    assert not backends.executor.log and backends.executor.stopped


@pytest.mark.parametrize("stop_confirmed", [True, False])
def test_home_failure_preserves_known_empty_hand_but_never_hides_an_unknown_stop(stop_confirmed):
    held, backends = fixture()
    lose_after_capture(backends)

    def no_home(*args, **kwargs):
        raise GraspFailure("HOME_PLAN_FAILED", "no collision-free way home")

    def no_stop(_reason):
        raise RuntimeError("stop acknowledgement lost")

    backends.planner.plan_home = no_home
    if not stop_confirmed:
        backends.executor.stop = no_stop
    outcome = run(held, backends)
    assert outcome.reason_code == "GRIP_LOST_IN_TRANSFER" and "way home" in outcome.message
    assert outcome.release_status == "released" and not backends.executor.log
    assert outcome.custody_outcome == ("lost" if stop_confirmed else "uncertain")
    assert outcome.execution_detail.recovery_state == ("FAILED" if stop_confirmed else "UNKNOWN")


def test_replanning_then_preparation_drop_keeps_both_failures_and_recovers_once():
    held, backends = fixture()
    capture = backends.observation.capture
    captures = []

    def observe(**kwargs):
        observation = capture(**kwargs)
        captures.append(None)
        if len(captures) == 2:
            backends.effector.holding = lambda: "contact lost"
        return observation

    def replan(*args, **kwargs):
        raise GraspFailure("PLACE_PLAN_FAILED", "Robot moved beyond the planning snapshot", retryable=True)

    backends.observation.capture, backends.planner.plan = observe, replan
    enable_home(backends)
    outcome = run(held, backends)
    assert len(captures) == 2 and outcome.execution_detail.retry_count == 1
    assert outcome.report["first_failure"]["code"] == "PLACE_PLAN_FAILED"
    assert outcome.reason_code == "GRIP_LOST_IN_TRANSFER"
    assert outcome.custody_outcome == "lost" and outcome.execution_detail.recovery_state == "COMPLETED"
    assert backends.effector.log == [EffectorAction.RELEASE]
