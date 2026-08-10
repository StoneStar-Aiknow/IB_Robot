"""Shared ROS action terminal handling for scheduler idempotency ledgers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from inference_service.scheduler.ledger import IdempotencyLedger, LedgerEntry, LedgerResolution

_ROS_STATUS_SUCCEEDED = 4
_ROS_STATUS_CANCELED = 5
_ROS_STATUS_ABORTED = 6


@dataclass(frozen=True)
class ResolutionErrorCodes:
    """Layer-specific error names for the shared post-resolution flow."""

    ledger_full: str
    ledger_error: str
    internal_error: str


def replay_terminal(goal_handle: Any, entry: LedgerEntry) -> Any:
    """Replay both the cached result and its original ROS goal status."""

    if entry.ros_status == _ROS_STATUS_SUCCEEDED:
        goal_handle.succeed()
    elif entry.ros_status == _ROS_STATUS_CANCELED:
        goal_handle.canceled()
    else:
        goal_handle.abort()
    return entry.result


def finish_terminal(
    ledger: IdempotencyLedger,
    *,
    key: tuple,
    goal_handle: Any,
    result: Any,
    not_started_outcome: int,
) -> Any:
    """Record a terminal result and release retry-safe NOT_STARTED entries."""

    ros_status = (
        _ROS_STATUS_CANCELED
        if goal_handle.is_cancel_requested and getattr(result.error, "code", "") == "request_canceled"
        else (_ROS_STATUS_SUCCEEDED if result.success else _ROS_STATUS_ABORTED)
    )
    ledger.set_terminal(
        key,
        outcome=int(result.outcome.value),
        success=bool(result.success),
        result=result,
        ros_status=ros_status,
    )
    if int(result.outcome.value) == not_started_outcome:
        ledger.discard_terminal(key)
    return result


def execute_resolved_action(
    ledger: IdempotencyLedger,
    resolution: LedgerResolution,
    *,
    key: tuple,
    goal_handle: Any,
    execute: Callable[[LedgerEntry], Any],
    failure: Callable[[str, str, int], Any],
    error_codes: ResolutionErrorCodes,
    not_started_outcome: int,
    unknown_outcome: int,
    log_internal_error: Callable[[Exception], None],
    prepare_entry: Callable[[LedgerEntry], None] | None = None,
) -> Any:
    """Run the mechanical ledger flow after layer-specific request resolution."""

    if resolution.is_full_reject:
        return failure(error_codes.ledger_full, "", not_started_outcome)
    if resolution.is_waiter_limit:
        return failure("duplicate_waiter_limit", "", not_started_outcome)
    entry = resolution.entry
    if entry is None:
        return failure(error_codes.ledger_error, "", unknown_outcome)
    if prepare_entry is not None:
        prepare_entry(entry)
    if resolution.is_cached_terminal:
        return replay_terminal(goal_handle, entry)
    if resolution.is_duplicate_payload:
        deadline_monotonic_ns = time.monotonic_ns() + max(0, entry.effective_deadline_utc_ns - time.time_ns())
        try:
            timeout = max(0, deadline_monotonic_ns - time.monotonic_ns()) / 1_000_000_000
            if not entry.terminal_event.wait(timeout):
                return failure("duplicate_wait_timeout", "", unknown_outcome)
            return replay_terminal(goal_handle, entry)
        finally:
            ledger.release_waiter(key)
    try:
        result = execute(entry)
    except Exception as exc:  # noqa: BLE001
        log_internal_error(exc)
        result = failure(error_codes.internal_error, str(exc), unknown_outcome)
    return finish_terminal(
        ledger,
        key=key,
        goal_handle=goal_handle,
        result=result,
        not_started_outcome=not_started_outcome,
    )


__all__ = [
    "ResolutionErrorCodes",
    "execute_resolved_action",
    "finish_terminal",
    "replay_terminal",
]
