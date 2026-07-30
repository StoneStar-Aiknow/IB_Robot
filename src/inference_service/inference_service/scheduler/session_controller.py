"""Product-session controller kernel (pure-Python, ROS-free).

Implements the pipeline-side product-session state machine, the two-phase
reset/close barrier semantics, generation fencing, lease arithmetic, and
work-class public-capacity admission.

This is the testable core that PipelinePolicyNode will wrap with ROS action
servers. Keeping it ROS-free lets the lifecycle / fencing / lease / capacity
rules be deterministically unit-tested with a fake monotonic clock.

Whole-graph concurrency is bounded by the configured work-class capacity and
the backend's declared per-instance capacity.

Key invariants:
  - exactly one ACTIVE session per controller.
  - serving state uses STARTING/IDLE/RESETTING/ACTIVE/FAILED/CLOSING.
  - generation fencing: a non-zero fence_generation is assigned when a reset/
    close barrier starts; only a successful drain publishes ACTIVE/IDLE with
    the higher generation. Stale completion callbacks that see an older
    generation fence drop their output.
  - lease updated only by product activity (Open/Dispatch). Global owns the
    production idle Close.
  - public_capacity admission bounds concurrent session control and action generation.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum

from inference_service.scheduler.work_classes import WorkClass

# InferenceServingStatus.state values.


class ServingState(IntEnum):
    STARTING = 1
    IDLE = 2
    RESETTING = 3
    ACTIVE = 4
    FAILED = 5
    CLOSING = 6


class SessionControllerError(Exception):
    """Raised for admission / session / generation violations."""


@dataclass
class WorkClassCapacity:
    work_class: WorkClass
    max_in_flight: int


@dataclass
class AdmissionDecision:
    """Result of a public-capacity admission attempt."""

    accepted: bool
    code: str
    work_class: WorkClass
    in_flight_after: int = 0


@dataclass
class SessionSnapshot:
    """Read-only view of the controller's session and serving state."""

    state: ServingState
    product_session_id: str
    product_session_generation: int
    fence_generation: int
    boot_id: str
    lease_expires_at_ns: int
    quarantine: bool


class ProductSessionController:
    """Pure-Python product-session controller kernel.

    Caller injects:
      - boot_id: UUID4 of this pipeline process boot.
      - capacities: per work-class public_capacity.
      - session_idle_timeout_ns / status_stale_timeout_ns.
      - now_ns(): monotonic clock for all time judgments (no wall-clock skew).
    """

    def __init__(
        self,
        *,
        boot_id: str,
        capacities: dict[WorkClass, WorkClassCapacity],
        session_idle_timeout_ns: int,
        now_ns: Callable[[], int],
    ) -> None:
        if WorkClass.SESSION_CONTROL not in capacities or WorkClass.ACTION_GENERATION not in capacities:
            raise SessionControllerError("session_control and action_generation capacities are required")
        self._lock = threading.RLock()
        self._boot_id = boot_id
        self._capacities = capacities
        self._idle_timeout_ns = session_idle_timeout_ns
        self._now_ns = now_ns

        self._state: ServingState = ServingState.STARTING
        self._product_session_id: str = ""
        self._product_generation: int = 0
        self._fence_generation: int = 0  # fencing identity; 0 until first barrier assigns one
        self._lease_expires_at_ns: int = 0
        self._quarantine: bool = False
        # Work-class live counters.
        self._in_flight: dict[WorkClass, int] = {wc: 0 for wc in capacities}
        self._last_product_activity_ns: int = 0

    # ------------------------------------------------------------------
    # lifecycle: Open -> reset barrier -> ACTIVE
    # ------------------------------------------------------------------

    def mark_ready(self) -> None:
        """Publish the initial IDLE state after model/backend startup completes."""
        with self._lock:
            if self._state != ServingState.STARTING:
                raise SessionControllerError(f"cannot mark ready from {self._state.name}")
            self._state = ServingState.IDLE

    def begin_open(self, session_id: str) -> OpenResult:
        """Atomically acquire the session slot.

        Returns NOT_STARTED if a session is already active (caller may then try
        fallback). On acquiring the slot the controller enters RESETTING
        and assigns a non-zero fence_generation; the caller then runs the
        two-phase drain/reset and calls finish_open().
        """
        with self._lock:
            if not self._try_acquire_capacity_locked(WorkClass.SESSION_CONTROL):
                return OpenResult(accepted=False, code="no_session_capacity", fence_generation=0)
            if self._state in (ServingState.ACTIVE, ServingState.RESETTING, ServingState.CLOSING):
                self._release_capacity_locked(WorkClass.SESSION_CONTROL)
                return OpenResult(accepted=False, code="session_busy", fence_generation=0)
            if self._state == ServingState.FAILED and self._quarantine:
                self._release_capacity_locked(WorkClass.SESSION_CONTROL)
                return OpenResult(accepted=False, code="session_failed", fence_generation=self._fence_generation)
            # acquire slot: mark RESETTING, assign fence generation.
            self._state = ServingState.RESETTING
            self._product_session_id = session_id
            self._fence_generation = self._next_generation_locked()
            self._last_product_activity_ns = self._now_ns()
            self._recompute_lease_locked()
            return OpenResult(accepted=True, code="", fence_generation=self._fence_generation)

    def finish_open(self, *, success: bool) -> None:
        """Publish ACTIVE or FAILED after the drain/reset barrier.

        On success the fence_generation becomes the ACTIVE product_generation.
        On definite failure we keep the fence (FAILED) so a precise Close can
        reconcile because a reset that started after acceptance does not fall back.
        """
        with self._lock:
            if self._state != ServingState.RESETTING:
                raise SessionControllerError(f"cannot finish Open from {self._state.name}")
            if success:
                self._product_generation = self._fence_generation
                self._state = ServingState.ACTIVE
                self._quarantine = False
            else:
                self._state = ServingState.FAILED
                self._quarantine = True
            self._release_capacity_locked(WorkClass.SESSION_CONTROL)

    # ------------------------------------------------------------------
    # Public-capacity admission.
    # ------------------------------------------------------------------

    def admit(self, work_class: WorkClass, *, generation: int, session_id: str | None = None) -> AdmissionDecision:
        """Apply atomic capacity admission to a ledger-miss request.

        Session-control admission is state-independent and bounds concurrent
        Open/Close callbacks. Action generation additionally requires the
        current active session and generation fence.
        """
        with self._lock:
            if work_class != WorkClass.ACTION_GENERATION:
                raise SessionControllerError("session-control capacity is owned by Open/Close lifecycle methods")
            cap = self._capacities[work_class]
            if self._state != ServingState.ACTIVE:
                return AdmissionDecision(False, "session_not_active", work_class)
            if session_id is not None and session_id != self._product_session_id:
                return AdmissionDecision(False, "session_mismatch", work_class)
            if generation != self._product_generation:
                return AdmissionDecision(False, "generation_mismatch", work_class)
            if self._in_flight[work_class] < cap.max_in_flight:
                self._in_flight[work_class] += 1
                return AdmissionDecision(True, "", work_class, self._in_flight[work_class])
            return AdmissionDecision(False, "no_session_capacity", work_class)

    def release_in_flight(self, work_class: WorkClass) -> bool:
        """Release an in-flight slot idempotently."""
        with self._lock:
            if self._in_flight[work_class] > 0:
                self._in_flight[work_class] -= 1
                return True
            return False

    # ------------------------------------------------------------------
    # Lease and idle expiry.
    # ------------------------------------------------------------------

    def record_product_activity(self) -> None:
        """Open and Dispatch renew the lease; internal maintenance does not."""
        with self._lock:
            self._last_product_activity_ns = self._now_ns()
            self._recompute_lease_locked()

    # ------------------------------------------------------------------
    # Close barrier.
    # ------------------------------------------------------------------

    def begin_close(self, *, generation: int) -> CloseResult:
        """Stop new admission and trigger the Close drain barrier.

        Accepts the requested generation; generation 0 is the explicit cleanup
        wildcard for an uncertain Open. Returns the closed (old) generation and
        the new fence (drain) generation.
        """
        with self._lock:
            if not self._try_acquire_capacity_locked(WorkClass.SESSION_CONTROL):
                return CloseResult(
                    accepted=False,
                    code="no_session_capacity",
                    closed_generation=self._product_generation,
                    drain_generation=0,
                )
            if self._state == ServingState.IDLE:
                self._release_capacity_locked(WorkClass.SESSION_CONTROL)
                return CloseResult(accepted=False, code="cleanup_not_needed", closed_generation=0, drain_generation=0)
            if self._state == ServingState.CLOSING:
                self._release_capacity_locked(WorkClass.SESSION_CONTROL)
                return CloseResult(
                    accepted=False,
                    code="close_in_progress",
                    closed_generation=self._product_generation,
                    drain_generation=self._fence_generation,
                )
            if generation != 0 and generation != self._product_generation:
                # mismatch: either stale Close or unknown-Open not yet resolved.
                self._release_capacity_locked(WorkClass.SESSION_CONTROL)
                return CloseResult(
                    accepted=False,
                    code="generation_mismatch",
                    closed_generation=self._product_generation,
                    drain_generation=0,
                )
            self._state = ServingState.CLOSING
            closed = self._product_generation or self._fence_generation
            self._fence_generation = self._next_generation_locked()
            return CloseResult(
                accepted=True, code="", closed_generation=closed, drain_generation=self._fence_generation
            )

    def finish_close(self, *, success: bool) -> None:
        """Publish IDLE after drain, or FAILED."""
        with self._lock:
            if self._state != ServingState.CLOSING:
                raise SessionControllerError(f"cannot finish Close from {self._state.name}")
            if success:
                self._product_generation = self._fence_generation
                self._product_session_id = ""
                self._state = ServingState.IDLE
                self._quarantine = False
            else:
                self._state = ServingState.FAILED
                self._quarantine = True
            self._release_capacity_locked(WorkClass.SESSION_CONTROL)

    def mark_failed_quarantine(self) -> None:
        """Move unknown outcomes or integrity errors to FAILED quarantine."""
        with self._lock:
            self._state = ServingState.FAILED
            self._quarantine = True

    # ------------------------------------------------------------------
    # Generation fencing for completion callbacks.
    # ------------------------------------------------------------------

    def is_stale_generation(self, generation: int) -> bool:
        """A completion callback whose generation predates the fence is stale."""
        with self._lock:
            return generation != 0 and generation < self._fence_generation

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def snapshot(self) -> SessionSnapshot:
        with self._lock:
            return SessionSnapshot(
                state=self._state,
                product_session_id=self._product_session_id,
                product_session_generation=self._product_generation,
                fence_generation=self._fence_generation,
                boot_id=self._boot_id,
                lease_expires_at_ns=self._lease_expires_at_ns,
                quarantine=self._quarantine,
            )

    def capacity_count(self, work_class: WorkClass) -> int:
        with self._lock:
            return self._in_flight[work_class]

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _recompute_lease_locked(self) -> None:
        if self._last_product_activity_ns == 0:
            return
        self._lease_expires_at_ns = self._last_product_activity_ns + self._idle_timeout_ns

    def _next_generation_locked(self) -> int:
        return max(self._fence_generation, self._product_generation) + 1

    def _try_acquire_capacity_locked(self, work_class: WorkClass) -> bool:
        capacity = self._capacities[work_class]
        if self._in_flight[work_class] >= capacity.max_in_flight:
            return False
        self._in_flight[work_class] += 1
        return True

    def _release_capacity_locked(self, work_class: WorkClass) -> None:
        if self._in_flight[work_class] > 0:
            self._in_flight[work_class] -= 1


@dataclass
class OpenResult:
    accepted: bool
    code: str
    fence_generation: int


@dataclass
class CloseResult:
    accepted: bool
    code: str
    closed_generation: int
    drain_generation: int


__all__ = [
    "AdmissionDecision",
    "CloseResult",
    "OpenResult",
    "ProductSessionController",
    "ServingState",
    "SessionControllerError",
    "SessionSnapshot",
    "WorkClass",
    "WorkClassCapacity",
]
