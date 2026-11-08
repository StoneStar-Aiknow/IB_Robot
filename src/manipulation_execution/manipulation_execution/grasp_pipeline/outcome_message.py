"""Short grasp summaries and structured interaction advice.

Execution facts belong to ExecutionDetail. Full diagnostics and retry history
stay in the pipeline report; neither Agent nor TTS parses this text for policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from embodied_common.execution_detail import short_message

if TYPE_CHECKING:
    from .pipeline import PickOutcome


@dataclass(frozen=True)
class FailureSpec:
    summary: str
    suggested_action: str


FAILURES = {
    "INVALID_TARGET": FailureSpec("The target name is empty or unsupported.", "WAIT_USER"),
    "TARGET_NOT_VISIBLE": FailureSpec("The target was not detected.", "WAIT_USER"),
    "NO_EXECUTABLE_CANDIDATE": FailureSpec("No complete grasp path was found.", "WAIT_USER"),
    "NO_COMPLETE_PATH": FailureSpec("No complete grasp path was found.", "WAIT_USER"),
    "OUT_OF_REACH": FailureSpec("No grasp candidate is within reach.", "WAIT_USER"),
    "PLAN_TIMEOUT": FailureSpec("Grasp planning timed out.", "WAIT_SERVICE"),
    "PLAN_REJECTED": FailureSpec("The planned motion was rejected.", "MANUAL_CHECK"),
    "PLANNER_FAULT": FailureSpec("The grasp planner failed.", "MANUAL_CHECK"),
    "REOBSERVE_REQUIRED": FailureSpec("The observation changed during planning.", "WAIT_USER"),
    "OBSERVATION_UNAVAILABLE": FailureSpec("Camera or robot feedback is unavailable.", "WAIT_SERVICE"),
    "TARGET_MOVED": FailureSpec("The target moved during the approach.", "WAIT_USER"),
    "GRASP_VERIFICATION_FAILED": FailureSpec("The grasp could not be verified.", "WAIT_USER"),
    "EMPTY_GRASP": FailureSpec("Nothing was held after closing.", "WAIT_USER"),
    "GRIP_LOST": FailureSpec("The object slipped out of the grasp.", "WAIT_USER"),
    "GRASP_UNCERTAIN": FailureSpec("The grip state could not be confirmed.", "MANUAL_CHECK"),
    "SKILL_TIMEOUT": FailureSpec("The pick timed out.", "WAIT_SERVICE"),
    "EXECUTION_BUDGET_INSUFFICIENT": FailureSpec("Too little time remains to start motion.", "WAIT_SERVICE"),
    "TASK_TIMEOUT": FailureSpec("The task budget expired.", "WAIT_SERVICE"),
    "CANCELED": FailureSpec("The pick was cancelled.", "NONE"),
    "RPC_FAILED": FailureSpec("A grasp service call failed.", "WAIT_SERVICE"),
    "RPC_TIMEOUT": FailureSpec("A grasp service call timed out.", "WAIT_SERVICE"),
    "NOT_READY": FailureSpec("The grasp services are not ready.", "WAIT_SERVICE"),
    "SERVICE_UNAVAILABLE": FailureSpec("A grasp service is unavailable.", "WAIT_SERVICE"),
    "MOTION_FAILED": FailureSpec("The arm did not finish its motion.", "MANUAL_CHECK"),
    "WATCH_FAILED": FailureSpec("Target monitoring failed.", "MANUAL_CHECK"),
    "EFFECTOR_FAILED": FailureSpec("The end effector did not finish its action.", "MANUAL_CHECK"),
    "INTERNAL_ERROR": FailureSpec("The grasp software failed unexpectedly.", "MANUAL_CHECK"),
}
_UNKNOWN = FailureSpec("The pick failed.", "MANUAL_CHECK")


def failure_spec(code: str, reason_code: str = "") -> FailureSpec:
    return FAILURES.get(reason_code) or FAILURES.get(code, _UNKNOWN)


def failure_action(code: str, reason_code: str, recovery_state: str) -> str:
    if recovery_state in {"UNKNOWN", "FAILED", "IN_PROGRESS"}:
        return "MANUAL_CHECK"
    return failure_spec(code, reason_code).suggested_action


def outcome_message(outcome: PickOutcome) -> str:
    if outcome.success:
        if outcome.mode == "observe_only":
            return "Observation completed."
        if outcome.mode == "plan_only":
            return "Grasp planning completed. Motion was not executed."
        if outcome.held is not True:
            return "The execution completed; the grasp is not confirmed."
        summary = "The object was grasped, lifted and verified."
        if outcome.released_after_success:
            summary += " The requested release completed."
        return summary
    reason = outcome.execution_detail.reason_code
    summary = f"PICK FAILED [{outcome.error_code or 'UNKNOWN'}] {failure_spec(outcome.error_code, reason).summary}"
    first = outcome.report.get("first_execution_failure")
    if first and first.get("reason_code") not in {reason, outcome.error_code}:
        code = first.get("reason_code") or first.get("error_code", "UNKNOWN")
        summary += f" Earlier execution [{code}]: {failure_spec(code).summary}"
    return short_message(summary)
