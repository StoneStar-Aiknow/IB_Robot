"""Unit tests for the bounded idempotent ledger.

Pure-Python with a fake monotonic clock. Verifies: duplicate same-payload
attachment, request_conflict on payload mismatch, cached-terminal replay,
waiter limit, max_session_records hard cap + Open FULL_REJECT, retention
eviction of terminal entries (but never active/orphan), stable generation-0
Close identity, and that the first terminal wins (replay fidelity).
"""

from __future__ import annotations

import threading

import pytest

from inference_service.scheduler.ledger import (
    IdempotencyLedger,
    LedgerAction,
    LedgerError,
    ResolutionKind,
    close_key,
    dispatch_key,
    open_key,
)


class _Clock:
    def __init__(self, start: int = 1_000_000_000) -> None:
        self.t = start
        self._lock = threading.Lock()

    def now_ns(self) -> int:
        with self._lock:
            return self.t

    def advance(self, ns: int) -> None:
        with self._lock:
            self.t += ns


def _make(**overrides):
    clock = _Clock()
    defaults = dict(
        max_session_records=4,
        max_duplicate_waiters_per_request=2,
        terminal_session_retention_ns=1_000_000_000,
        now_ns=clock.now_ns,
    )
    defaults.update(overrides)
    return IdempotencyLedger(**defaults), clock


# ---------------------------------------------------------------------------
# Duplicate same-payload requests attach; different payloads conflict.
# ---------------------------------------------------------------------------


def test_new_entry_then_duplicate_same_payload_attaches():
    ledger, _ = _make()
    key = dispatch_key("s1", 1, "r1")
    fp = "a" * 64
    r1 = ledger.resolve(action=LedgerAction.DISPATCH, key=key, payload_fingerprint=fp, effective_deadline_utc_ns=100)
    assert r1.is_new
    r2 = ledger.resolve(action=LedgerAction.DISPATCH, key=key, payload_fingerprint=fp, effective_deadline_utc_ns=100)
    assert r2.is_duplicate_payload
    assert r2.entry is r1.entry
    assert r2.entry.waiter_count == 2


def test_same_key_different_payload_raises_conflict():
    ledger, _ = _make()
    key = dispatch_key("s1", 1, "r1")
    ledger.resolve(action=LedgerAction.DISPATCH, key=key, payload_fingerprint="a" * 64, effective_deadline_utc_ns=100)
    with pytest.raises(LedgerError, match="request_conflict"):
        ledger.resolve(
            action=LedgerAction.DISPATCH, key=key, payload_fingerprint="b" * 64, effective_deadline_utc_ns=100
        )


def test_duplicate_waiter_limit_returns_not_started():
    ledger, _ = _make(max_duplicate_waiters_per_request=2)
    key = dispatch_key("s1", 1, "r1")
    fp = "a" * 64
    ledger.resolve(action=LedgerAction.DISPATCH, key=key, payload_fingerprint=fp, effective_deadline_utc_ns=100)
    ledger.resolve(
        action=LedgerAction.DISPATCH, key=key, payload_fingerprint=fp, effective_deadline_utc_ns=100
    )  # waiter 2
    r3 = ledger.resolve(action=LedgerAction.DISPATCH, key=key, payload_fingerprint=fp, effective_deadline_utc_ns=100)
    assert r3.is_waiter_limit
    assert r3.entry.waiter_count == 2  # not incremented


# ---------------------------------------------------------------------------
# Cached-terminal replay preserves the first terminal result.
# ---------------------------------------------------------------------------


def test_cached_terminal_returned_on_replay():
    ledger, _ = _make()
    key = dispatch_key("s1", 1, "r1")
    ledger.resolve(action=LedgerAction.DISPATCH, key=key, payload_fingerprint="a" * 64, effective_deadline_utc_ns=100)
    ledger.set_terminal(key, outcome=2, success=True, result={"action": [1, 2]}, ros_status=4)  # SUCCEEDED
    r = ledger.resolve(
        action=LedgerAction.DISPATCH, key=key, payload_fingerprint="a" * 64, effective_deadline_utc_ns=100
    )
    assert r.is_cached_terminal
    assert r.entry.success is True
    assert r.entry.result == {"action": [1, 2]}


def test_cached_terminal_replay_rejects_different_payload():
    ledger, _ = _make()
    key = dispatch_key("s1", 1, "r1")
    ledger.resolve(action=LedgerAction.DISPATCH, key=key, payload_fingerprint="a" * 64, effective_deadline_utc_ns=100)
    ledger.set_terminal(key, outcome=2, success=True, result={"action": [1, 2]}, ros_status=4)

    with pytest.raises(LedgerError, match="request_conflict"):
        ledger.resolve(
            action=LedgerAction.DISPATCH,
            key=key,
            payload_fingerprint="b" * 64,
            effective_deadline_utc_ns=100,
        )


def test_first_terminal_wins_replay_fidelity():
    ledger, _ = _make()
    key = open_key("s1")
    ledger.resolve(action=LedgerAction.OPEN, key=key, payload_fingerprint="a" * 64, effective_deadline_utc_ns=100)
    ledger.set_terminal(key, outcome=2, success=True, result="first", ros_status=4)
    # A late competing terminal must NOT overwrite.
    ledger.set_terminal(key, outcome=1, success=False, result="late", ros_status=2)
    entry = ledger.get_entry(key)
    assert entry.success is True
    assert entry.result == "first"
    assert entry.outcome == 2


def test_discard_terminal_allows_retry_after_not_started_result_was_delivered():
    ledger, _ = _make()
    key = open_key("s1")
    first = ledger.resolve(
        action=LedgerAction.OPEN,
        key=key,
        payload_fingerprint="a" * 64,
        effective_deadline_utc_ns=100,
    )
    ledger.set_terminal(key, outcome=1, success=False, result="not-started", ros_status=6)
    assert first.entry is not None and first.entry.terminal_event.is_set()

    ledger.discard_terminal(key)
    retry = ledger.resolve(
        action=LedgerAction.OPEN,
        key=key,
        payload_fingerprint="b" * 64,
        effective_deadline_utc_ns=200,
    )

    assert retry.is_new


# ---------------------------------------------------------------------------
# max_session_records caps Open records but not Close or Dispatch.
# ---------------------------------------------------------------------------


def test_open_full_reject_when_session_records_full():
    ledger, _ = _make(max_session_records=2)
    # fill with 2 active opens
    ledger.resolve(
        action=LedgerAction.OPEN, key=open_key("s1"), payload_fingerprint="a" * 64, effective_deadline_utc_ns=100
    )
    ledger.resolve(
        action=LedgerAction.OPEN, key=open_key("s2"), payload_fingerprint="a" * 64, effective_deadline_utc_ns=100
    )
    r = ledger.resolve(
        action=LedgerAction.OPEN, key=open_key("s3"), payload_fingerprint="a" * 64, effective_deadline_utc_ns=100
    )
    assert r.is_full_reject


def test_dispatch_not_capped_by_session_records():
    ledger, _ = _make(max_session_records=1)
    ledger.resolve(
        action=LedgerAction.OPEN, key=open_key("s1"), payload_fingerprint="a" * 64, effective_deadline_utc_ns=100
    )
    # dispatch with a new request id is NOT rejected even though records==cap
    r = ledger.resolve(
        action=LedgerAction.DISPATCH,
        key=dispatch_key("s1", 1, "r1"),
        payload_fingerprint="b" * 64,
        effective_deadline_utc_ns=100,
    )
    assert r.is_new


# ---------------------------------------------------------------------------
# Retention evicts terminal records only, never active or orphan records.
# ---------------------------------------------------------------------------


def test_terminal_evicted_after_retention():
    ledger, clock = _make(terminal_session_retention_ns=1_000_000_000)
    key = open_key("s1")
    ledger.resolve(action=LedgerAction.OPEN, key=key, payload_fingerprint="a" * 64, effective_deadline_utc_ns=100)
    ledger.set_terminal(key, outcome=2, success=False, result=None, ros_status=2)
    assert ledger.has(key)
    clock.advance(1_500_000_000)  # past retention
    # trigger eviction via next resolve
    ledger.resolve(
        action=LedgerAction.OPEN, key=open_key("s2"), payload_fingerprint="b" * 64, effective_deadline_utc_ns=100
    )
    assert not ledger.has(key)


def test_active_entry_never_evicted_by_retention():
    ledger, clock = _make(terminal_session_retention_ns=1_000_000_000, max_session_records=10)
    key = open_key("s1")
    ledger.resolve(action=LedgerAction.OPEN, key=key, payload_fingerprint="a" * 64, effective_deadline_utc_ns=100)
    clock.advance(10_000_000_000)  # way past retention
    ledger.resolve(
        action=LedgerAction.OPEN, key=open_key("s2"), payload_fingerprint="b" * 64, effective_deadline_utc_ns=100
    )
    assert ledger.has(key)  # still present — active entries are never evicted


def test_terminal_event_wakes_duplicate_waiters():
    ledger, _ = _make()
    key = dispatch_key("s1", 1, "r1")
    resolution = ledger.resolve(
        action=LedgerAction.DISPATCH,
        key=key,
        payload_fingerprint="a" * 64,
        effective_deadline_utc_ns=100,
    )
    assert resolution.entry is not None
    assert not resolution.entry.terminal_event.is_set()
    ledger.set_terminal(key, outcome=2, success=True, result="done", ros_status=4)
    assert resolution.entry.terminal_event.is_set()


def test_terminal_result_cache_is_bounded_per_session():
    ledger, _ = _make(max_terminal_entries_per_session=2)
    for index in range(3):
        key = dispatch_key("s1", 1, f"r{index}")
        ledger.resolve(
            action=LedgerAction.DISPATCH,
            key=key,
            payload_fingerprint=str(index) * 64,
            effective_deadline_utc_ns=100,
        )
        ledger.set_terminal(key, outcome=2, success=True, result=index, ros_status=4)
    assert not ledger.has(dispatch_key("s1", 1, "r0"))
    assert ledger.has(dispatch_key("s1", 1, "r1"))
    assert ledger.has(dispatch_key("s1", 1, "r2"))


# ---------------------------------------------------------------------------
# Generation-0 Close remains a stable cleanup identity.
# ---------------------------------------------------------------------------


def test_generation_zero_close_replays_without_aliasing_to_later_generation():
    ledger, _ = _make()
    zero_key = close_key("s1", 0)
    real_key = close_key("s1", 5)
    zero = ledger.resolve(
        action=LedgerAction.CLOSE,
        key=zero_key,
        payload_fingerprint="c" * 64,
        effective_deadline_utc_ns=100,
    )
    ledger.set_terminal(zero_key, outcome=2, success=True, result="closed", ros_status=4)

    replay = ledger.resolve(
        action=LedgerAction.CLOSE,
        key=zero_key,
        payload_fingerprint="c" * 64,
        effective_deadline_utc_ns=100,
    )
    later_generation = ledger.resolve(
        action=LedgerAction.CLOSE,
        key=real_key,
        payload_fingerprint="c" * 64,
        effective_deadline_utc_ns=100,
    )

    assert zero.entry is replay.entry
    assert replay.is_cached_terminal
    assert replay.entry.result == "closed"
    assert later_generation.is_new
    assert later_generation.entry is not zero.entry


def test_release_waiter_decrements_without_removing():
    ledger, _ = _make()
    key = dispatch_key("s1", 1, "r1")
    ledger.resolve(action=LedgerAction.DISPATCH, key=key, payload_fingerprint="a" * 64, effective_deadline_utc_ns=100)
    ledger.resolve(action=LedgerAction.DISPATCH, key=key, payload_fingerprint="a" * 64, effective_deadline_utc_ns=100)
    assert ledger.get_entry(key).waiter_count == 2
    ledger.release_waiter(key)
    assert ledger.get_entry(key).waiter_count == 1
    assert ledger.has(key)  # entry still present


# ---------------------------------------------------------------------------
# action-specific keys are distinct
# ---------------------------------------------------------------------------


def test_keys_for_different_actions_do_not_collide():
    assert open_key("s1") != close_key("s1", 1)
    assert close_key("s1", 1) != dispatch_key("s1", 1, "r1")
    assert close_key("s1", 0) != close_key("s1", 1)


# ---------------------------------------------------------------------------
# thread-safety smoke
# ---------------------------------------------------------------------------


def test_concurrent_resolves_are_thread_safe():
    ledger, _ = _make(max_session_records=100, max_duplicate_waiters_per_request=100)
    results: list = []
    lock = threading.Lock()

    def worker(i: int) -> None:
        key = dispatch_key("s1", 1, f"r{i}")
        r = ledger.resolve(
            action=LedgerAction.DISPATCH, key=key, payload_fingerprint="a" * 64, effective_deadline_utc_ns=100
        )
        with lock:
            results.append(r.kind)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    # all distinct keys -> all NEW
    assert all(kind == ResolutionKind.NEW for kind in results)
    assert len(ledger) == 20
