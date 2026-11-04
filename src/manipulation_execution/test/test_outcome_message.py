"""Bounded summaries complement structured facts; they never infer a home pose."""

import pytest

from embodied_common.execution_detail import ExecutionDetail
from manipulation_execution.grasp_pipeline.outcome_message import FAILURES, failure_action, outcome_message
from manipulation_execution.grasp_pipeline.pipeline import PickOutcome


@pytest.mark.parametrize("code", sorted(FAILURES))
def test_every_known_failure_has_a_bounded_summary_and_machine_advice(code):
    detail = ExecutionDetail(schema_version=1, reason_code=code)
    outcome = PickOutcome(error_code=code, message="底层异常" * 1000, execution_detail=detail)
    message = outcome_message(outcome)
    assert code in message and len(message) <= 300
    assert all(label not in message for label in ("State:", "Next:", "Retries:", "Detail:"))
    assert failure_action(code, code, "NOT_NEEDED") == FAILURES[code].suggested_action
    for state in ("UNKNOWN", "FAILED", "IN_PROGRESS"):
        assert failure_action(code, code, state) == "MANUAL_CHECK"


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("observe_only", "Observation completed."),
        ("plan_only", "Grasp planning completed. Motion was not executed."),
    ],
)
def test_diagnostic_success_does_not_claim_a_grasp(mode, expected):
    assert outcome_message(PickOutcome(success=True, mode=mode)) == expected


def test_success_requires_confirmed_grip_and_release_never_infers_home():
    unknown = outcome_message(PickOutcome(success=True))
    assert "not confirmed" in unknown
    success = outcome_message(
        PickOutcome(
            success=True, held=True, released_after_success=True, retries=[{"retry": "regrasp", "state": "planned"}]
        )
    )
    assert "lifted and verified" in success and "release completed" in success
    assert "home" not in success and "again" not in success


def test_specific_planning_reason_and_unknown_fallback():
    detail = ExecutionDetail(schema_version=1, reason_code="PLAN_TIMEOUT")
    message = outcome_message(PickOutcome(error_code="NO_EXECUTABLE_CANDIDATE", execution_detail=detail))
    assert "timed out" in message
    unknown = outcome_message(PickOutcome(error_code="NEW_ERROR", at_rest=True, held=True))
    assert "home" not in unknown and "stopped" not in unknown
    assert failure_action("NEW_ERROR", "", "NOT_NEEDED") == "MANUAL_CHECK"
