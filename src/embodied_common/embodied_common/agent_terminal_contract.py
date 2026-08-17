"""Shared Agent-plan execution error and terminal classification contract."""

from __future__ import annotations

from typing import Any

GOAL_SUCCEEDED = 4
GOAL_CANCELED = 5
GOAL_ABORTED = 6
TERMINAL_GOAL_STATUSES = {GOAL_SUCCEEDED, GOAL_CANCELED, GOAL_ABORTED}

STABLE_AGENT_EXECUTION_CODES = frozenset(
    {
        "CAPABILITY_NOT_READY",
        "CONTROL_MODE_MISMATCH",
        "GATEWAY_FINALIZATION_FAILED",
        "GOAL_NOT_FOUND",
        "MOTION_NOT_AUTHORIZED",
        "SKILL_AGENT_PLAN_EXPIRED",
        "SKILL_AGENT_PLAN_NOT_FOUND",
        "SKILL_BUSY",
        "SKILL_CANCELLED",
        "SKILL_CANCEL_TIMEOUT",
        "SKILL_DISPATCH_NOT_AUTHORIZED",
        "SKILL_EXECUTION_BUSY",
        "SKILL_EXECUTOR_IDENTITY_MISMATCH",
        "SKILL_LIMIT_VIOLATION",
        "SKILL_REGISTRY_EPOCH_MISMATCH",
        "SKILL_REGISTRY_NOT_READY",
        "SKILL_REGISTRY_VERSION_MISMATCH",
        "SKILL_REQUEST_ID_CONFLICT",
        "SKILL_SCHEMA_INVALID",
        "SKILL_SNAPSHOT_DIGEST_MISMATCH",
        "SKILL_SNAPSHOT_NOT_RETAINED",
        "SKILL_TASK_BUDGET_MISMATCH",
        "SKILL_TASK_DEADLINE_EXPIRED",
        "SKILL_WORKFLOW_DIGEST_MISMATCH",
        "SKILL_WORKFLOW_LEASE_MISMATCH",
        "SKILL_WORKFLOW_STEP_MISMATCH",
        "TIMEOUT_EXCEEDS_POLICY",
    }
)
UNCERTAIN_AGENT_MOTION_CODES = frozenset(
    {
        "GATEWAY_FINALIZATION_FAILED",
        "SKILL_CANCEL_TIMEOUT",
        "SKILL_EXECUTION_BUSY",
    }
)
DEFINITE_AGENT_FAILURE_CODES = STABLE_AGENT_EXECUTION_CODES - UNCERTAIN_AGENT_MOTION_CODES - {"SKILL_CANCELLED"}


def stable_agent_execution_error_code(error_code: str) -> str:
    """Map child errors into the stable public Agent execution taxonomy."""
    code = str(error_code).strip()
    if code in STABLE_AGENT_EXECUTION_CODES:
        return code
    if code == "CANCEL_CLEANUP_TIMEOUT":
        return "SKILL_CANCEL_TIMEOUT"
    if "TIMEOUT" in code:
        return "SKILL_TASK_DEADLINE_EXPIRED"
    if "EXECUTOR" in code and ("IDENTITY" in code or "VERSION" in code):
        return "SKILL_EXECUTOR_IDENTITY_MISMATCH"
    return "CAPABILITY_NOT_READY"


def classify_agent_terminal(status: Any, result: dict[str, Any], expectation: dict[str, Any]) -> str | None:
    """Classify exact terminals while preserving known uncertain-motion failures."""
    try:
        goal_status = int(status)
        actual_generation = int(result.get("actual_registry_generation", 0))
    except (TypeError, ValueError):
        return None
    exact_identity = (
        str(result.get("plan_id", "")) == expectation["plan_id"]
        and str(result.get("plan_digest", "")) == expectation["plan_digest"]
        and str(result.get("actual_registry_epoch", "")) == expectation["registry_epoch"]
        and actual_generation == expectation["registry_generation"]
        and str(result.get("actual_registry_digest", "")) == expectation["registry_digest"]
    )
    completed = result.get("completed_step_count")
    if not isinstance(completed, int) or isinstance(completed, bool) or not 0 <= completed <= expectation["step_count"]:
        return None
    success = bool(result.get("success"))
    error_code = str(result.get("error_code", ""))
    if exact_identity and goal_status == GOAL_SUCCEEDED and success and not error_code:
        return "succeeded" if completed == expectation["step_count"] else None
    if exact_identity and goal_status == GOAL_CANCELED and not success and error_code == "SKILL_CANCELLED":
        return "stopped"
    if goal_status != GOAL_ABORTED or success or error_code not in STABLE_AGENT_EXECUTION_CODES:
        return None
    if not exact_identity:
        return None
    if error_code in UNCERTAIN_AGENT_MOTION_CODES:
        return "unknown"
    if error_code in DEFINITE_AGENT_FAILURE_CODES:
        return "failed"
    return None
