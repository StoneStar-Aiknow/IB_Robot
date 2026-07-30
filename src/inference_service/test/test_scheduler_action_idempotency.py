"""Shared post-resolution action idempotency flow tests."""

from __future__ import annotations

import time
from types import SimpleNamespace

from inference_service.scheduler.action_idempotency import ResolutionErrorCodes, execute_resolved_action
from inference_service.scheduler.ledger import IdempotencyLedger, LedgerAction, LedgerResolution, ResolutionKind


class _GoalHandle:
    def __init__(self) -> None:
        self.is_cancel_requested = False
        self.aborted = False
        self.succeeded = False
        self.canceled_status = False

    def abort(self) -> None:
        self.aborted = True

    def succeed(self) -> None:
        self.succeeded = True

    def canceled(self) -> None:
        self.canceled_status = True


def _ledger() -> IdempotencyLedger:
    return IdempotencyLedger(
        max_session_records=2,
        max_duplicate_waiters_per_request=2,
        terminal_session_retention_ns=1_000_000_000,
        now_ns=time.monotonic_ns,
    )


def _result(*, outcome: int, success: bool = False, code: str = ""):
    return SimpleNamespace(
        outcome=SimpleNamespace(value=outcome),
        success=success,
        error=SimpleNamespace(code=code),
    )


def _run(ledger, resolution, *, key, execute, failures, prepared=None, logged=None):
    goal_handle = _GoalHandle()
    result = execute_resolved_action(
        ledger,
        resolution,
        key=key,
        goal_handle=goal_handle,
        execute=execute,
        failure=lambda code, message, outcome: (
            failures.append((code, message, outcome)) or _result(outcome=outcome, code=code)
        ),
        error_codes=ResolutionErrorCodes("layer_full", "layer_error", "layer_internal"),
        not_started_outcome=1,
        unknown_outcome=3,
        log_internal_error=(logged if logged is not None else lambda _exc: None),
        prepare_entry=(prepared.append if prepared is not None else None),
    )
    return goal_handle, result


def test_resolution_rejections_keep_layer_specific_error_codes_and_skip_entry_preparation():
    ledger = _ledger()
    failures = []
    prepared = []

    _run(
        ledger,
        LedgerResolution(ResolutionKind.FULL_REJECT, None),
        key=("full",),
        execute=lambda _entry: None,
        failures=failures,
        prepared=prepared,
    )
    _run(
        ledger,
        LedgerResolution(ResolutionKind.DUPLICATE_WAITER_LIMIT, SimpleNamespace()),
        key=("waiters",),
        execute=lambda _entry: None,
        failures=failures,
        prepared=prepared,
    )
    _run(
        ledger,
        LedgerResolution(ResolutionKind.NEW, None),
        key=("missing",),
        execute=lambda _entry: None,
        failures=failures,
        prepared=prepared,
    )

    assert failures == [
        ("layer_full", "", 1),
        ("duplicate_waiter_limit", "", 1),
        ("layer_error", "", 3),
    ]
    assert prepared == []


def test_cached_terminal_prepares_entry_and_replays_original_ros_status():
    ledger = _ledger()
    key = (LedgerAction.OPEN, "session")
    first = ledger.resolve(
        action=LedgerAction.OPEN,
        key=key,
        payload_fingerprint="a" * 64,
        effective_deadline_utc_ns=time.time_ns() + 1_000_000_000,
    )
    expected = _result(outcome=2, success=True)
    ledger.set_terminal(key, outcome=2, success=True, result=expected, ros_status=4)
    replay = ledger.resolve(
        action=LedgerAction.OPEN,
        key=key,
        payload_fingerprint="a" * 64,
        effective_deadline_utc_ns=first.entry.effective_deadline_utc_ns,
    )
    prepared = []

    goal_handle, actual = _run(
        ledger,
        replay,
        key=key,
        execute=lambda _entry: (_ for _ in ()).throw(AssertionError("must replay")),
        failures=[],
        prepared=prepared,
    )

    assert actual is expected
    assert prepared == [first.entry]
    assert goal_handle.succeeded


def test_duplicate_wait_timeout_releases_waiter_and_returns_unknown():
    ledger = _ledger()
    key = (LedgerAction.OPEN, "session")
    ledger.resolve(
        action=LedgerAction.OPEN,
        key=key,
        payload_fingerprint="a" * 64,
        effective_deadline_utc_ns=time.time_ns() - 1,
    )
    duplicate = ledger.resolve(
        action=LedgerAction.OPEN,
        key=key,
        payload_fingerprint="a" * 64,
        effective_deadline_utc_ns=time.time_ns() - 1,
    )
    failures = []

    _goal_handle, result = _run(
        ledger,
        duplicate,
        key=key,
        execute=lambda _entry: (_ for _ in ()).throw(AssertionError("must wait")),
        failures=failures,
    )

    assert result.outcome.value == 3
    assert failures == [("duplicate_wait_timeout", "", 3)]
    assert duplicate.entry.waiter_count == 1


def test_new_execution_records_terminal_and_discards_retry_safe_not_started():
    ledger = _ledger()
    key = (LedgerAction.OPEN, "session")
    resolution = ledger.resolve(
        action=LedgerAction.OPEN,
        key=key,
        payload_fingerprint="a" * 64,
        effective_deadline_utc_ns=time.time_ns() + 1_000_000_000,
    )
    expected = _result(outcome=1)

    _goal_handle, actual = _run(
        ledger,
        resolution,
        key=key,
        execute=lambda entry: expected if entry is resolution.entry else None,
        failures=[],
    )

    assert actual is expected
    assert not ledger.has(key)


def test_execution_exception_uses_layer_error_and_records_unknown_terminal():
    ledger = _ledger()
    key = (LedgerAction.OPEN, "session")
    resolution = ledger.resolve(
        action=LedgerAction.OPEN,
        key=key,
        payload_fingerprint="a" * 64,
        effective_deadline_utc_ns=time.time_ns() + 1_000_000_000,
    )
    failures = []
    logged = []

    _goal_handle, result = _run(
        ledger,
        resolution,
        key=key,
        execute=lambda _entry: (_ for _ in ()).throw(RuntimeError("boom")),
        failures=failures,
        logged=logged.append,
    )

    assert result.outcome.value == 3
    assert failures == [("layer_internal", "boom", 3)]
    assert len(logged) == 1 and str(logged[0]) == "boom"
    assert ledger.get_entry(key).result is result
