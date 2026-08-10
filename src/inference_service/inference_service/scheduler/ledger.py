"""Bounded idempotent ledgers for the Scheduler and pipeline.

Two separate ledgers exist: the Global Scheduler keeps a
session/request-level ledger, and each pipeline keeps its own. They use
different keys but the same shape. This module is the shared implementation.

Ledger rules:
  - Open:           key = session_id            ; Global Open has no route payload;
                    pipeline-local Open fingerprints its absolute deadline.
  - Close:          key = (session_id, requested gen); generation 0 remains a stable cleanup identity.
  - ScheduledDispatch: key = (session_id, gen, request_id); fingerprint covers
    observation, prompt, session identity, priority, and target. Priority 0 also
    covers fallback order and the caller deadline; non-zero, lower priorities ignore both.
  - duplicate key, same payload -> attach to existing future or return cached terminal (no new reservation, never reject even if capacity full)
  - duplicate key, different payload -> request_conflict
  - duplicate waiter count capped by max_duplicate_waiters_per_request
  - action wrappers discard terminal NOT_STARTED entries after waking current
    waiters, so a later attempt may run again without risking duplicate effects
  - active/orphan entries never evicted; terminal entries retained for terminal_session_retention then cleared
  - max_session_records hard cap: ledger-miss Open is rejected when full

This module is pure-Python (no ROS) so it can be unit-tested with fake futures.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LedgerAction(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    DISPATCH = "dispatch"


@dataclass
class LedgerEntry:
    """One idempotency entry. Holds the terminal result once known."""

    key: tuple
    action: LedgerAction
    payload_fingerprint: str
    effective_deadline_utc_ns: int
    terminal: bool = False
    outcome: int = 0  # InferenceOutcome value (0=UNSPECIFIED until terminal)
    success: bool = False
    result: Any = None  # opaque terminal result payload
    ros_status: int = 0  # ROS GoalStatus, for replay fidelity
    waiter_count: int = 0
    # monotonic_ns when the terminal result was recorded (for retention expiry)
    terminal_monotonic_ns: int = 0
    terminal_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)


class LedgerError(Exception):
    """Raised for conflict / capacity / duplicate-waiter-limit violations."""


class IdempotencyLedger:
    """A bounded idempotent ledger keyed by action-specific keys.

    Thread-safe. The caller supplies:
      - `terminal_result_class`/opaque: results are stored as-is; this ledger
        does not interpret them.
      - capacity: max_session_records (hard cap on total entries), and
        max_duplicate_waiters_per_request.
      - terminal_session_retention_ns: how long a terminal entry is kept
        before eviction (active/orphan entries are never evicted by retention).
      - now_ns(): a monotonic clock injection (so tests are deterministic).
    """

    def __init__(
        self,
        *,
        max_session_records: int,
        max_duplicate_waiters_per_request: int,
        terminal_session_retention_ns: int,
        now_ns: Callable[[], int],
        max_entries: int | None = None,
        max_terminal_entries_per_session: int | None = None,
    ) -> None:
        if max_session_records <= 0:
            raise ValueError("max_session_records must be positive")
        if max_duplicate_waiters_per_request <= 0:
            raise ValueError("max_duplicate_waiters_per_request must be positive")
        if terminal_session_retention_ns <= 0:
            raise ValueError("terminal_session_retention_ns must be positive")
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive when set")
        if max_terminal_entries_per_session is not None and max_terminal_entries_per_session <= 0:
            raise ValueError("max_terminal_entries_per_session must be positive when set")
        self._lock = threading.RLock()
        self._entries: dict[tuple, LedgerEntry] = {}
        self._max_session_records = max_session_records
        self._max_duplicate_waiters_per_request = max_duplicate_waiters_per_request
        self._retention_ns = terminal_session_retention_ns
        self._now_ns = now_ns
        self._max_entries = max_entries
        self._max_terminal_entries_per_session = max_terminal_entries_per_session

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        *,
        action: LedgerAction,
        key: tuple,
        payload_fingerprint: str,
        effective_deadline_utc_ns: int,
    ) -> LedgerResolution:
        """Resolve a request against the ledger.

        Returns a LedgerResolution describing the outcome:
          - DUPLICATE_PAYLOAD: attach to existing in-flight entry (or return cached terminal)
          - CONFLICT: same key, different payload -> request_conflict
          - DUPLICATE_WAITER_LIMIT: too many waiters on one in-flight entry
          - NEW: ledger miss; caller proceeds and must call set_terminal()
                 exactly once. For Open on a full ledger -> FULL_REJECT.
        """
        with self._lock:
            self._evict_expired_terminals_locked()
            existing = self._lookup_locked(key)
            if existing is not None:
                if existing.payload_fingerprint != payload_fingerprint:
                    raise LedgerError("request_conflict")
                if existing.terminal:
                    return LedgerResolution(ResolutionKind.CACHED_TERMINAL, entry=existing)
                if existing.waiter_count >= self._max_duplicate_waiters_per_request:
                    return LedgerResolution(ResolutionKind.DUPLICATE_WAITER_LIMIT, entry=existing)
                existing.waiter_count += 1
                return LedgerResolution(ResolutionKind.DUPLICATE_PAYLOAD, entry=existing)
            if action is not LedgerAction.OPEN and len(key) > 1:
                self._evict_session_terminal_overflow_locked(str(key[1]))
            # Ledger miss. Open is capped by session records; the optional total
            # bound lets the ROS ingress cap terminal request replay storage.
            open_entries = sum(entry.action is LedgerAction.OPEN for entry in self._entries.values())
            if action is LedgerAction.OPEN and open_entries >= self._max_session_records:
                return LedgerResolution(ResolutionKind.FULL_REJECT, entry=None)
            if self._max_entries is not None and len(self._entries) >= self._max_entries:
                return LedgerResolution(ResolutionKind.FULL_REJECT, entry=None)
            entry = LedgerEntry(
                key=key,
                action=action,
                payload_fingerprint=payload_fingerprint,
                effective_deadline_utc_ns=effective_deadline_utc_ns,
                waiter_count=1,
            )
            self._entries[key] = entry
            return LedgerResolution(ResolutionKind.NEW, entry=entry)

    def set_terminal(
        self,
        key: tuple,
        *,
        outcome: int,
        success: bool,
        result: Any,
        ros_status: int,
    ) -> None:
        """Record a terminal result for an entry. Idempotent: subsequent calls are
        ignored because the first terminal preserves replay fidelity."""
        with self._lock:
            entry = self._lookup_locked(key)
            if entry is None:
                # Late terminal for an already-evicted/unknown entry; ignore but
                # surface for diagnostics. This is not an error path.
                return
            if entry.terminal:
                return
            entry.terminal = True
            entry.outcome = outcome
            entry.success = success
            entry.result = result
            entry.ros_status = ros_status
            entry.terminal_monotonic_ns = self._now_ns()
            entry.terminal_event.set()

    def release_waiter(self, key: tuple) -> None:
        """Decrement a duplicate waiter (e.g. on waiter-local deadline) without
        affecting the underlying entry. Never removes the entry."""
        with self._lock:
            entry = self._lookup_locked(key)
            if entry is not None and entry.waiter_count > 0:
                entry.waiter_count -= 1

    def discard_terminal(self, key: tuple) -> None:
        """Forget a terminal entry when the caller proved no side effect started."""

        with self._lock:
            entry = self._lookup_locked(key)
            if entry is not None and entry.terminal:
                self._entries.pop(key, None)

    # ------------------------------------------------------------------
    # introspection (tests / diagnostics)
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def has(self, key: tuple) -> bool:
        with self._lock:
            return key in self._entries

    def get_entry(self, key: tuple) -> LedgerEntry | None:
        with self._lock:
            return self._lookup_locked(key)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _lookup_locked(self, key: tuple) -> LedgerEntry | None:
        return self._entries.get(key)

    def _evict_expired_terminals_locked(self) -> None:
        now = self._now_ns()
        expired = [
            key
            for key, entry in self._entries.items()
            if entry.terminal and (now - entry.terminal_monotonic_ns) > self._retention_ns
        ]
        for key in expired:
            del self._entries[key]

    def _evict_session_terminal_overflow_locked(self, session_id: str) -> None:
        limit = self._max_terminal_entries_per_session
        if limit is None:
            return
        terminals = sorted(
            (
                entry
                for entry in self._entries.values()
                if entry.action is not LedgerAction.OPEN
                and len(entry.key) > 1
                and entry.key[1] == session_id
                and entry.terminal
            ),
            key=lambda entry: entry.terminal_monotonic_ns,
        )
        while len(terminals) >= limit:
            oldest = terminals.pop(0)
            self._entries.pop(oldest.key, None)


class ResolutionKind:
    """Resolution outcome labels for LedgerResolution."""

    CACHED_TERMINAL = "cached_terminal"
    DUPLICATE_PAYLOAD = "duplicate_payload"
    DUPLICATE_WAITER_LIMIT = "duplicate_waiter_limit"
    NEW = "new"
    FULL_REJECT = "full_reject"


@dataclass
class LedgerResolution:
    kind: str
    entry: LedgerEntry | None

    @property
    def is_new(self) -> bool:
        return self.kind == ResolutionKind.NEW

    @property
    def is_duplicate_payload(self) -> bool:
        return self.kind == ResolutionKind.DUPLICATE_PAYLOAD

    @property
    def is_cached_terminal(self) -> bool:
        return self.kind == ResolutionKind.CACHED_TERMINAL

    @property
    def is_full_reject(self) -> bool:
        return self.kind == ResolutionKind.FULL_REJECT

    @property
    def is_waiter_limit(self) -> bool:
        return self.kind == ResolutionKind.DUPLICATE_WAITER_LIMIT


def open_key(session_id: str) -> tuple:
    return (LedgerAction.OPEN, session_id)


def close_key(session_id: str, generation: int) -> tuple:
    return (LedgerAction.CLOSE, session_id, generation)


def dispatch_key(session_id: str, generation: int, request_id: str) -> tuple:
    return (LedgerAction.DISPATCH, session_id, generation, request_id)


__all__ = [
    "IdempotencyLedger",
    "LedgerAction",
    "LedgerEntry",
    "LedgerError",
    "LedgerResolution",
    "ResolutionKind",
    "close_key",
    "dispatch_key",
    "open_key",
]
