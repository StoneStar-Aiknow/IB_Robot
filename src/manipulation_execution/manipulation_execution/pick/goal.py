"""Admission of a delegated ``PickObject`` goal, shared by every pick executor.

A goal is admitted only if it names this executor's identity, carries a
complete dispatch binding with registry identity and a live task budget, and
asks for a known mode. A supervised direct goal carries no dispatch nonce; a
Gateway goal must carry one. Which executor serves the action does not change
these rules.

Fields an executor does not serve are refused before any motion, never
ignored: a grasp spec its effector cannot honour (``GRASP_SPEC_UNSUPPORTED``,
also for an unknown spec version) or that is malformed
(``GRASP_SPEC_INVALID``), and custody fields, which only the custody put-back
(``MODE_RECOVER``, not served yet) uses.
"""

from __future__ import annotations

import math

from embodied_common.dispatch_binding import delegated_executor_identity_matches
from embodied_common.pick_contract import GraspCapability, check_grasp_spec
from ibrobot_msgs.action import PickObject

PICK_MODES = frozenset(
    {PickObject.Goal.MODE_EXECUTE, PickObject.Goal.MODE_PLAN_ONLY, PickObject.Goal.MODE_OBSERVE_ONLY}
)


def _seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec / 1_000_000_000


def pick_goal_rejection(goal, identity: dict[str, str], now_s: float, capability: GraspCapability | None = None) -> str:
    """Why ``goal`` must be rejected at ``now_s`` (ROS time, seconds), or "" to admit it.

    ``capability`` is what this executor's effector declares; the default
    honours no grasp spec field.
    """
    if not str(goal.target_query).strip():
        return "empty target_query"
    if not delegated_executor_identity_matches(goal.expected_executor, identity):
        return "expected_executor does not match this executor"
    dispatch_nonce = str(goal.dispatch_binding.dispatch_nonce).strip()
    if bool(goal.supervised_direct) == bool(dispatch_nonce):
        return "supervised_direct requires an empty dispatch_nonce; delegated dispatch requires one"
    binding = goal.dispatch_binding
    budget = binding.task_budget
    if (
        binding.schema_version != 1
        or not str(binding.task_id).strip()
        or not str(binding.root_task_id).strip()
        or not binding.expected_registry_epoch
        or binding.expected_registry_generation <= 0
        or not binding.expected_registry_digest
        or budget.schema_version != 1
    ):
        return "dispatch_binding is incomplete"
    started, deadline = _seconds(budget.started_at), _seconds(budget.deadline)
    timeout_sec = float(goal.timeout_sec)
    if (
        budget.started_at.sec < 0
        or budget.deadline.sec < 0
        or not 0 <= budget.started_at.nanosec < 1_000_000_000
        or not 0 <= budget.deadline.nanosec < 1_000_000_000
        or not math.isfinite(started)
        or not math.isfinite(deadline)
        or deadline <= started
        or deadline <= now_s
        or not math.isfinite(timeout_sec)
        or timeout_sec <= 0.0
    ):
        return "task_budget or timeout_sec is invalid or already expired"
    if int(goal.mode) not in PICK_MODES:
        return f"unknown mode {goal.mode}"
    _, spec_error = check_grasp_spec(goal.grasp_spec, capability or GraspCapability())
    if spec_error:
        return f"{spec_error}: this executor cannot honour the requested grasp spec"
    if any(str(getattr(goal, name)).strip() for name in ("custody_id", "custody_epoch", "request_id")):
        return "custody fields are only for MODE_RECOVER, which this executor does not serve"
    return ""


def remaining_budget_s(goal, now_s: float) -> float:
    """Seconds left for this goal: its own timeout, capped by the shared task budget."""
    return min(float(goal.timeout_sec), _seconds(goal.dispatch_binding.task_budget.deadline) - now_s)
