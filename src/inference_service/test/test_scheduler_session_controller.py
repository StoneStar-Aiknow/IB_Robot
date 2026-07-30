"""Unit tests for the product-session controller kernel.

Pure-Python with a fake monotonic clock. Covers: single ACTIVE session,
generation fencing + stale-completion drop, two-phase Open/Close barrier
transitions, work-class admission and release, capacity accounting, lease
renewal, and generation fencing.
"""

from __future__ import annotations

import threading

import pytest

from inference_service.scheduler.session_controller import (
    ProductSessionController,
    ServingState,
    SessionControllerError,
    WorkClass,
    WorkClassCapacity,
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


def _make_controller(*, idle_timeout_ns: int = 30_000_000_000):
    clock = _Clock()
    caps = {
        WorkClass.SESSION_CONTROL: WorkClassCapacity(WorkClass.SESSION_CONTROL, 1),
        WorkClass.ACTION_GENERATION: WorkClassCapacity(WorkClass.ACTION_GENERATION, 1),
    }
    return (
        ProductSessionController(
            boot_id="00112233-4455-6677-8899-aabbccddeeff",
            capacities=caps,
            session_idle_timeout_ns=idle_timeout_ns,
            now_ns=clock.now_ns,
        ),
        clock,
    )


# ---------------------------------------------------------------------------
# Exactly one ACTIVE session.
# ---------------------------------------------------------------------------


def test_open_acquire_then_second_open_rejected():
    ctl, _ = _make_controller()
    r1 = ctl.begin_open("s1")
    assert r1.accepted and r1.fence_generation == 1
    assert ctl.snapshot().state == ServingState.RESETTING
    r2 = ctl.begin_open("s2")
    assert not r2.accepted and r2.code == "no_session_capacity"


def test_finish_open_success_publishes_active_with_generation():
    ctl, _ = _make_controller()
    ctl.begin_open("s1")
    ctl.finish_open(success=True)
    snap = ctl.snapshot()
    assert snap.state == ServingState.ACTIVE
    assert snap.product_session_generation == 1
    assert snap.product_session_id == "s1"


def test_finish_open_failure_keeps_fence_failed():
    ctl, _ = _make_controller()
    r = ctl.begin_open("s1")
    ctl.finish_open(success=False)
    snap = ctl.snapshot()
    assert snap.state == ServingState.FAILED
    assert snap.quarantine
    # Retain the fence identity for precise Close reconciliation.
    assert snap.fence_generation == r.fence_generation


# ---------------------------------------------------------------------------
# Generation fencing drops stale completions.
# ---------------------------------------------------------------------------


def test_stale_generation_completion_is_detected():
    ctl, _ = _make_controller()
    ctl.begin_open("s1")
    ctl.finish_open(success=True)
    old_gen = ctl.snapshot().product_session_generation
    # a new reset barrier bumps the fence beyond old_gen
    ctl.begin_close(generation=old_gen)
    assert ctl.is_stale_generation(old_gen)  # old completion must be dropped
    assert not ctl.is_stale_generation(ctl.snapshot().fence_generation)


# ---------------------------------------------------------------------------
# Work-class admission and release.
# ---------------------------------------------------------------------------


def test_admit_requires_active_and_matching_generation():
    ctl, _ = _make_controller()
    # not active yet
    d = ctl.admit(WorkClass.ACTION_GENERATION, generation=1)
    assert not d.accepted and d.code == "session_not_active"
    ctl.begin_open("s1")
    ctl.finish_open(success=True)
    gen = ctl.snapshot().product_session_generation
    d = ctl.admit(WorkClass.ACTION_GENERATION, generation=gen)
    assert d.accepted
    # wrong generation -> reject
    d = ctl.admit(WorkClass.ACTION_GENERATION, generation=gen + 999)
    assert not d.accepted and d.code == "generation_mismatch"


def test_admit_rejects_when_in_flight_full_then_accepts_after_release():
    ctl, _ = _make_controller()
    ctl.begin_open("s1")
    ctl.finish_open(success=True)
    gen = ctl.snapshot().product_session_generation
    # capacity 1 in-flight, 0 waiters -> first accepted, second no_session_capacity
    d1 = ctl.admit(WorkClass.ACTION_GENERATION, generation=gen)
    assert d1.accepted
    d2 = ctl.admit(WorkClass.ACTION_GENERATION, generation=gen)
    assert not d2.accepted and d2.code == "no_session_capacity"
    ctl.release_in_flight(WorkClass.ACTION_GENERATION)
    d3 = ctl.admit(WorkClass.ACTION_GENERATION, generation=gen)
    assert d3.accepted


def test_release_in_flight_is_idempotent_terminalize():
    ctl, _ = _make_controller()
    ctl.begin_open("s1")
    ctl.finish_open(success=True)
    gen = ctl.snapshot().product_session_generation
    ctl.admit(WorkClass.ACTION_GENERATION, generation=gen)
    assert ctl.release_in_flight(WorkClass.ACTION_GENERATION)  # first release
    assert not ctl.release_in_flight(WorkClass.ACTION_GENERATION)  # second is a no-op


# ---------------------------------------------------------------------------
# Only product activity renews the lease; Global owns expiry decisions.
# ---------------------------------------------------------------------------


def test_lease_renews_on_product_activity_not_internal():
    ctl, clock = _make_controller(idle_timeout_ns=1_000_000_000)
    ctl.begin_open("s1")
    ctl.finish_open(success=True)
    lease1 = ctl.snapshot().lease_expires_at_ns
    clock.advance(500_000_000)
    ctl.record_product_activity()
    lease2 = ctl.snapshot().lease_expires_at_ns
    assert lease2 == lease1 + 500_000_000


def test_open_and_close_count_session_control_capacity():
    ctl, _ = _make_controller()
    assert ctl.capacity_count(WorkClass.SESSION_CONTROL) == 0

    ctl.begin_open("s1")
    assert ctl.capacity_count(WorkClass.SESSION_CONTROL) == 1
    ctl.finish_open(success=True)
    assert ctl.capacity_count(WorkClass.SESSION_CONTROL) == 0

    ctl.begin_close(generation=ctl.snapshot().product_session_generation)
    assert ctl.capacity_count(WorkClass.SESSION_CONTROL) == 1
    ctl.finish_close(success=True)
    assert ctl.capacity_count(WorkClass.SESSION_CONTROL) == 0


# ---------------------------------------------------------------------------
# Close barrier returns the closed and drain generations.
# ---------------------------------------------------------------------------


def test_close_barrier_returns_closed_and_drain_generations():
    ctl, _ = _make_controller()
    ctl.begin_open("s1")
    ctl.finish_open(success=True)
    active_gen = ctl.snapshot().product_session_generation
    r = ctl.begin_close(generation=active_gen)
    assert r.accepted
    assert r.closed_generation == active_gen
    assert r.drain_generation > active_gen
    ctl.finish_close(success=True)
    snap = ctl.snapshot()
    assert snap.state == ServingState.IDLE
    assert snap.product_session_generation == r.drain_generation
    assert snap.product_session_id == ""


def test_close_on_idle_returns_cleanup_not_needed():
    ctl, _ = _make_controller()
    ctl.begin_open("s1")
    ctl.finish_open(success=True)
    ctl.begin_close(generation=ctl.snapshot().product_session_generation)
    ctl.finish_close(success=True)
    # already IDLE
    r = ctl.begin_close(generation=ctl.snapshot().product_session_generation)
    assert not r.accepted and r.code == "cleanup_not_needed"


def test_close_with_wrong_generation_rejected():
    ctl, _ = _make_controller()
    ctl.begin_open("s1")
    ctl.finish_open(success=True)
    gen = ctl.snapshot().product_session_generation
    r = ctl.begin_close(generation=gen + 777)
    assert not r.accepted and r.code == "generation_mismatch"


def test_concurrent_close_does_not_start_a_second_barrier():
    ctl, _ = _make_controller()
    ctl.begin_open("s1")
    ctl.finish_open(success=True)
    gen = ctl.snapshot().product_session_generation
    first = ctl.begin_close(generation=gen)
    second = ctl.begin_close(generation=gen)
    assert first.accepted
    assert not second.accepted
    assert second.code == "no_session_capacity"


def test_generation_zero_close_waits_for_open_capacity():
    ctl, _ = _make_controller()
    ctl.begin_open("s1")
    closed = ctl.begin_close(generation=0)
    assert not closed.accepted
    assert closed.code == "no_session_capacity"

    ctl.finish_open(success=True)
    closed = ctl.begin_close(generation=0)
    assert closed.accepted
    assert closed.closed_generation == ctl.snapshot().product_session_generation
    assert closed.drain_generation > closed.closed_generation


def test_missing_capacity_constructor_rejected():
    clock = _Clock()
    with pytest.raises(SessionControllerError):
        ProductSessionController(
            boot_id="b",
            capacities={WorkClass.SESSION_CONTROL: WorkClassCapacity(WorkClass.SESSION_CONTROL, 1)},
            session_idle_timeout_ns=1,
            now_ns=clock.now_ns,
        )
