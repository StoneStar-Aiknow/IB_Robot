"""The PickObject message tells an Agent the outcome, the robot's state, the retries and what to do next."""

from __future__ import annotations

import pytest

from manipulation_execution.grasp_pipeline.outcome_message import FAILURES, outcome_message
from manipulation_execution.grasp_pipeline.pipeline import PickOutcome


def executed(*states):
    return {"executions": [{"phases": [{"name": name, "state": state} for name, state in states]}]}


def test_a_success_after_retries_says_what_was_retried_and_why():
    outcome = PickOutcome(
        success=True,
        released_after_success=True,
        retries=[
            {"attempt": 1, "retry": "reobserve", "code": "OBSERVATION_UNAVAILABLE", "message": "372 ms"},
            {"attempt": 2, "retry": "regrasp", "code": "GRASP_VERIFICATION_FAILED", "message": "grip lost"},
        ],
        report=executed(("close", "succeeded")),
    )
    lines = outcome_message(outcome).splitlines()
    assert lines[0] == "PICK SUCCEEDED. The object was grasped, lifted and put back where it was."
    assert lines[1] == "State: the arm is back at home with the hand open; the object is on the table."
    assert lines[2] == (
        "Retries (2): 1. looked again after OBSERVATION_UNAVAILABLE: robot sensor data dropped out; "
        "2. grasped again after GRASP_VERIFICATION_FAILED: the hand did not keep hold of the object"
    )
    assert len(lines) == 3  # no advice or detail on success


def test_a_failure_before_motion_says_nothing_was_touched():
    outcome = PickOutcome(
        error_code="NO_EXECUTABLE_CANDIDATE",
        message="RuntimeError: No candidate produced a complete three-segment plan",
        retries=[{"attempt": 1, "retry": "reobserve", "code": "NO_EXECUTABLE_CANDIDATE", "message": "x"}],
    )
    lines = outcome_message(outcome).splitlines()
    assert lines[0] == "PICK FAILED [NO_EXECUTABLE_CANDIDATE] No safe grasp could be planned for the target."
    assert lines[1] == "State: the arm did not move; the object was not touched."
    assert lines[2].startswith("Retries (1): 1. looked again after NO_EXECUTABLE_CANDIDATE")
    assert lines[3].startswith("Next: ask the user to move the object closer")
    assert lines[4] == "Detail: RuntimeError: No candidate produced a complete three-segment plan"


@pytest.mark.parametrize(
    "phases, state",
    [
        ((("close", "succeeded"), ("hold_lifted", "grip_lost")), "the object slipped out after the lift"),
        ((("close", "succeeded"),), "the hand closed on nothing"),
    ],
)
def test_a_lost_grasp_says_where_the_object_may_be(phases, state):
    outcome = PickOutcome(error_code="GRASP_VERIFICATION_FAILED", message="grip lost", report=executed(*phases))
    message = outcome_message(outcome)
    assert f"State: {state}" in message and "the arm is back at home, hand open." in message
    assert "Next: look at the table again before retrying" in message


def test_an_arm_stopped_in_place_tells_the_agent_not_to_retry():
    outcome = PickOutcome(
        error_code="MOTION_FAILED", message="tracking error 0.31 rad", at_rest=False, held=True, report=executed()
    )
    message = outcome_message(outcome)
    assert "State: the arm was stopped where it was and may still hold the object" in message
    assert "Next: do not retry; the operator must check the robot first." in message


def test_an_unknown_code_and_a_long_detail_are_still_readable():
    outcome = PickOutcome(error_code="PLANNER_FAULT", message="x" * 1000)
    lines = outcome_message(outcome).splitlines()
    assert lines[0] == "PICK FAILED [PLANNER_FAULT] The pick failed."
    assert lines[-1].endswith("...") and len(lines[-1]) == len("Detail: ") + 300


def test_every_code_the_x2_pipeline_returns_has_an_explanation():
    returned = {
        "INVALID_TARGET",
        "TARGET_NOT_VISIBLE",
        "NO_EXECUTABLE_CANDIDATE",
        "REOBSERVE_REQUIRED",
        "OBSERVATION_UNAVAILABLE",
        "TARGET_MOVED",
        "GRASP_VERIFICATION_FAILED",
        "SKILL_TIMEOUT",
        "TASK_TIMEOUT",
        "CANCELED",
        "RPC_FAILED",
        "RPC_TIMEOUT",
        "NOT_READY",
        "MOTION_FAILED",
        "WATCH_FAILED",
        "EFFECTOR_FAILED",
        "INTERNAL_ERROR",
    }
    assert returned <= set(FAILURES)
