"""Request-level routing and lifecycle tests for GlobalSchedulerCore."""

from __future__ import annotations

import threading

import pytest

from inference_service.scheduler.global_scheduler_core import (
    GlobalSchedulerCore,
    GlobalSessionState,
    PipelineCandidate,
    SchedulerError,
)
from inference_service.scheduler.idempotency import IdempotencyError

SESSION_ID = "123e4567-e89b-42d3-a456-426614174000"
OTHER_SESSION_ID = "123e4567-e89b-42d3-a456-426614174001"


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


def _candidate(pid: str, *, group: str = "g1", hwr: str = "ascend:0") -> PipelineCandidate:
    return PipelineCandidate(
        pipeline_id=pid,
        compatibility_group=group,
        hardware_resource_id=hwr,
        hardware_profile_fingerprint="a" * 64,
        deployment_fingerprint="d" * 64,
        runtime_policy_fingerprint="r" * 64,
        endpoint_open=f"/inference/{pid}/session/open",
        endpoint_dispatch=f"/inference/{pid}/dispatch",
        endpoint_close=f"/inference/{pid}/session/close",
        endpoint_serving_status=f"/inference/{pid}/serving_status",
        profile_path="",
    )


def _make(candidates=None, **overrides):
    clock = _Clock()
    defaults = {
        "candidates": candidates or [_candidate("pi05")],
        "max_session_records": 4,
        "max_product_requests_per_session": 4,
        "terminal_session_retention_ns": 1_000_000_000,
        "session_idle_timeout_ns": 30_000_000_000,
        "max_fallback_pipelines": 16,
        "now_ns": clock.now_ns,
    }
    defaults.update(overrides)
    return GlobalSchedulerCore(**defaults), clock


def _open_active(scheduler: GlobalSchedulerCore) -> None:
    decision = scheduler.open_session(session_id=SESSION_ID)
    assert decision.session_generation == 1


def _bind(
    scheduler: GlobalSchedulerCore,
    pipeline_id: str,
    *,
    pipeline_generation: int,
    logical_generation: int = 1,
) -> None:
    decision = scheduler.prepare_dispatch_candidate(
        session_id=SESSION_ID,
        session_generation=logical_generation,
        pipeline_id=pipeline_id,
    )
    assert decision.needs_open
    scheduler.record_binding_open_success(
        session_id=SESSION_ID,
        pipeline_id=pipeline_id,
        pipeline_generation=pipeline_generation,
        hardware_resource_id="ascend:0",
    )


def test_open_creates_active_route_independent_session_without_binding():
    scheduler, clock = _make(candidates=[_candidate("pi05"), _candidate("backup")])

    decision = scheduler.open_session(session_id=SESSION_ID)

    record = scheduler.session_record(SESSION_ID)
    assert decision.session_generation == 1
    assert decision.lease_expires_at_ns == clock.now_ns() + 30_000_000_000
    assert decision.replay is False
    assert scheduler.session_state(SESSION_ID) == GlobalSessionState.ACTIVE
    assert record is not None and record.bindings == {}


def test_open_validates_uuid_capacity_and_replays_without_routing_payload():
    scheduler, _ = _make(max_session_records=1)
    with pytest.raises(IdempotencyError):
        scheduler.open_session(session_id="bad")

    first = scheduler.open_session(session_id=SESSION_ID)
    replay = scheduler.open_session(session_id=SESSION_ID)
    assert replay.replay
    assert replay.session_generation == first.session_generation

    with pytest.raises(SchedulerError, match="max_session_records"):
        scheduler.open_session(session_id=OTHER_SESSION_ID)


def test_every_dispatch_can_select_and_lazily_bind_a_different_pipeline():
    scheduler, _ = _make(candidates=[_candidate("pi05"), _candidate("backup")])
    _open_active(scheduler)

    first_plan = scheduler.resolve_dispatch_plan(
        session_id=SESSION_ID,
        session_generation=1,
        target_pipeline_id="backup",
        fallback_chain=["pi05"],
        priority=0,
    )
    assert first_plan.candidate_ids == ("backup", "pi05")
    _bind(scheduler, "backup", pipeline_generation=7)
    scheduler.record_request_terminal(SESSION_ID)

    second_plan = scheduler.resolve_dispatch_plan(
        session_id=SESSION_ID,
        session_generation=1,
        target_pipeline_id="pi05",
        fallback_chain=[],
        priority=0,
    )
    assert second_plan.candidate_ids == ("pi05",)
    _bind(scheduler, "pi05", pipeline_generation=11)

    record = scheduler.session_record(SESSION_ID)
    assert record is not None
    assert record.session_generation == 1
    assert record.bindings["backup"].pipeline_generation == 7
    assert record.bindings["pi05"].pipeline_generation == 11


def test_positive_priority_uses_only_target_and_ignores_fallback_payload():
    scheduler, _ = _make()
    _open_active(scheduler)

    plan = scheduler.resolve_dispatch_plan(
        session_id=SESSION_ID,
        session_generation=1,
        target_pipeline_id="pi05",
        fallback_chain=["missing", "pi05"],
        priority=3,
    )

    assert plan.candidate_ids == ("pi05",)


def test_priority_zero_validates_target_fallback_bound_and_compatibility():
    scheduler, _ = _make(
        candidates=[_candidate("pi05"), _candidate("backup"), _candidate("other", group="g2")],
        max_fallback_pipelines=1,
    )
    _open_active(scheduler)

    with pytest.raises(SchedulerError, match="unknown_target_pipeline"):
        scheduler.resolve_dispatch_plan(
            session_id=SESSION_ID,
            session_generation=1,
            target_pipeline_id="missing",
            fallback_chain=[],
            priority=0,
        )
    with pytest.raises(SchedulerError, match="max_fallback_exceeded"):
        scheduler.resolve_dispatch_plan(
            session_id=SESSION_ID,
            session_generation=1,
            target_pipeline_id="pi05",
            fallback_chain=["backup", "other"],
            priority=0,
        )
    with pytest.raises(SchedulerError, match="fallback_compatibility_group_mismatch"):
        scheduler.resolve_dispatch_plan(
            session_id=SESSION_ID,
            session_generation=1,
            target_pipeline_id="pi05",
            fallback_chain=["other"],
            priority=0,
        )


def test_dispatch_limit_generation_and_unknown_session_are_enforced():
    scheduler, _ = _make(max_product_requests_per_session=1)
    with pytest.raises(SchedulerError, match="session_not_found"):
        scheduler.resolve_dispatch_plan(
            session_id=SESSION_ID,
            session_generation=1,
            target_pipeline_id="pi05",
            fallback_chain=[],
            priority=0,
        )
    _open_active(scheduler)
    with pytest.raises(SchedulerError, match="generation_mismatch"):
        scheduler.resolve_dispatch_plan(
            session_id=SESSION_ID,
            session_generation=2,
            target_pipeline_id="pi05",
            fallback_chain=[],
            priority=0,
        )
    scheduler.resolve_dispatch_plan(
        session_id=SESSION_ID,
        session_generation=1,
        target_pipeline_id="pi05",
        fallback_chain=[],
        priority=0,
    )
    scheduler.record_request_terminal(SESSION_ID)
    with pytest.raises(SchedulerError, match="max_product_requests_per_session"):
        scheduler.resolve_dispatch_plan(
            session_id=SESSION_ID,
            session_generation=1,
            target_pipeline_id="pi05",
            fallback_chain=[],
            priority=0,
        )


def test_close_during_lazy_binding_open_cleans_late_success():
    scheduler, _ = _make()
    _open_active(scheduler)
    scheduler.resolve_dispatch_plan(
        session_id=SESSION_ID,
        session_generation=1,
        target_pipeline_id="pi05",
        fallback_chain=[],
        priority=0,
    )
    binding = scheduler.prepare_dispatch_candidate(
        session_id=SESSION_ID,
        session_generation=1,
        pipeline_id="pi05",
    )
    assert binding.needs_open

    scheduler.begin_close(session_id=SESSION_ID, session_generation=1)
    close_raced = scheduler.record_binding_open_success(
        session_id=SESSION_ID,
        pipeline_id="pi05",
        pipeline_generation=9,
        hardware_resource_id="ascend:0",
    )

    assert close_raced
    assert scheduler.session_state(SESSION_ID) == GlobalSessionState.CLOSING
    bindings = scheduler.close_bindings(SESSION_ID)
    assert [(item.pipeline_id, item.pipeline_generation) for item in bindings] == [("pi05", 9)]


def test_close_without_any_binding_succeeds_and_advances_logical_generation():
    scheduler, _ = _make()
    _open_active(scheduler)

    scheduler.begin_close(session_id=SESSION_ID, session_generation=1)
    assert scheduler.close_bindings(SESSION_ID) == []

    assert scheduler.record_close_complete(SESSION_ID, success=True) == 2
    assert scheduler.session_state(SESSION_ID) == GlobalSessionState.CLOSED


def test_close_drains_every_used_binding_and_advances_only_logical_generation():
    scheduler, _ = _make(candidates=[_candidate("pi05"), _candidate("backup")])
    _open_active(scheduler)
    _bind(scheduler, "pi05", pipeline_generation=3)
    _bind(scheduler, "backup", pipeline_generation=8)

    scheduler.begin_close(session_id=SESSION_ID, session_generation=1)
    assert {(item.pipeline_id, item.pipeline_generation) for item in scheduler.close_bindings(SESSION_ID)} == {
        ("pi05", 3),
        ("backup", 8),
    }
    scheduler.record_binding_close_success(SESSION_ID, "pi05")
    scheduler.record_binding_close_success(SESSION_ID, "backup")

    assert scheduler.record_close_complete(SESSION_ID, success=True) == 2
    assert scheduler.session_state(SESSION_ID) == GlobalSessionState.CLOSED


def test_not_started_close_restores_active_binding_for_retry():
    scheduler, _ = _make()
    _open_active(scheduler)
    _bind(scheduler, "pi05", pipeline_generation=6)
    scheduler.begin_close(session_id=SESSION_ID, session_generation=1)
    assert [binding.pipeline_id for binding in scheduler.close_bindings(SESSION_ID)] == ["pi05"]

    scheduler.record_close_not_started(SESSION_ID)

    assert scheduler.session_state(SESSION_ID) == GlobalSessionState.ACTIVE
    decision = scheduler.prepare_dispatch_candidate(
        session_id=SESSION_ID,
        session_generation=1,
        pipeline_id="pi05",
    )
    assert decision.pipeline_generation == 6


def test_unknown_cleanup_is_not_retention_purged():
    scheduler, clock = _make(max_session_records=1, terminal_session_retention_ns=10)
    _open_active(scheduler)
    _bind(scheduler, "pi05", pipeline_generation=3)
    scheduler.mark_session_failed(SESSION_ID, pipeline_id="pi05")
    clock.advance(100)

    with pytest.raises(SchedulerError, match="max_session_records"):
        scheduler.open_session(session_id=OTHER_SESSION_ID)
    assert scheduler.session_record(SESSION_ID).unresolved_cleanup


def test_lease_renews_per_request_and_waits_for_in_flight_terminal():
    scheduler, clock = _make(session_idle_timeout_ns=100)
    _open_active(scheduler)
    clock.advance(80)
    scheduler.resolve_dispatch_plan(
        session_id=SESSION_ID,
        session_generation=1,
        target_pipeline_id="pi05",
        fallback_chain=[],
        priority=0,
    )
    clock.advance(200)
    assert scheduler.expired_sessions() == []
    scheduler.record_request_terminal(SESSION_ID)
    assert scheduler.expired_sessions() == [SESSION_ID]
