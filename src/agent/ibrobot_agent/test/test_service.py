from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.async_presentation import AsyncPresentationPort, ModeRoutingPresentationPort
from ibrobot_agent.contracts import (
    ASYNC_PRESENTATION_MODE,
    CATALOG_SKILL_NAMES_ZH,
    AgentRequest,
    ExecutionResult,
    PlannerIdentity,
    PlannerOutcome,
    PlannerRepair,
)
from ibrobot_agent.conversation_store import SQLiteConversationStore
from ibrobot_agent.planner import RulePlanner
from ibrobot_agent.presentation import PresentationGate
from ibrobot_agent.request_store import SQLiteRequestStore
from ibrobot_agent.service import AgentService


class FakePlanner:
    def __init__(self, outcome):
        self.identity = PlannerIdentity("fake", "fake", "json", "1", "1", "fake-config")
        self.outcome = outcome
        self.calls = 0

    def plan(self, request, context, catalog, cancel_token, *, on_preview=None):
        self.calls += 1
        return self.outcome


class PreviewingPlanner(FakePlanner):
    """Hand the reply over mid-plan, as a streaming model planner does."""

    def plan(self, request, context, catalog, cancel_token, *, on_preview=None):
        self.calls += 1
        if on_preview is not None:
            on_preview(self.outcome.user_message)
        return self.outcome


class BlockingPlanner(FakePlanner):
    def __init__(self, outcome):
        super().__init__(outcome)
        self.started = threading.Event()
        self.release = threading.Event()

    def plan(self, request, context, catalog, cancel_token, *, on_preview=None):
        self.calls += 1
        self.started.set()
        self.release.wait(2)
        return self.outcome


class FakeCatalog:
    def get_status(self):
        return {
            "robot_name": "so101_single_arm",
            "active_control_mode": "moveit_planning",
            "control_plane_ready": True,
            "motion_authorized": False,
            "registry_epoch": "epoch",
            "registry_generation": 1,
            "registry_digest": "digest",
        }

    def get_catalog(self, status):
        return {
            "robot_name": "so101_single_arm",
            "skills": [{"name": "wave_hello", "planner_visible": True, "semantic_level": "skill"}],
        }


class UnavailableCatalog:
    def get_status(self):
        raise RuntimeError("offline")

    def get_catalog(self, status):
        raise RuntimeError("offline")


class Events:
    def __init__(self):
        self.items = []
        self.ready = threading.Event()

    def publish(self, event):
        self.items.append(event)
        self.ready.set()


class FailingEvents(Events):
    def publish(self, event):
        raise RuntimeError("event transport unavailable")


class RenderedEvents(Events):
    def __init__(self):
        super().__init__()
        self.gate = PresentationGate(0.5)

    def publish(self, event):
        super().publish(event)
        if event.event_type == "presentation":
            assert self.gate.acknowledge(
                event.request_key,
                digest=event.detail["presentation_digest"],
                token=event.detail["receipt_token"],
            )


class FailingFinishStore(SQLiteRequestStore):
    def finish(self, *args, **kwargs):
        raise RuntimeError("ledger unavailable")


class FakeExecution:
    def __init__(self, *, status="succeeded", execution_mode="immediate_after_presentation"):
        self.status = status
        self.execution_mode = execution_mode
        self.calls = 0
        self.stop_calls = 0
        self.submitted = 0

    def execute(
        self,
        proposal,
        *,
        expected_registry_identity,
        presentation_callback,
        submission_callback,
        stop_event,
        continues_from_execution_id="",
    ):
        from ibrobot_agent.contracts import ExecutionResult, Presentation, TaskRef

        self.calls += 1
        task_ref = TaskRef("task-1", "plan-1", "plan-digest", "epoch", 1, "digest", len(proposal.outcome.steps))
        presentation_callback(
            Presentation(
                task_ref=task_ref,
                plan_kind=1,
                steps=proposal.outcome.steps,
                execution_mode=self.execution_mode,
                proposed_task_budget_sec=90.0,
                summary=proposal.outcome.summary,
            )
        )
        submission_callback(task_ref, {"confirmed": True})
        self.submitted += 1
        return ExecutionResult(self.status, task_ref, "" if self.status == "succeeded" else "FAILED", "done", {})

    def request_stop(self, request_key):
        self.stop_calls += 1


def _request(request_id: str = "r1", *, text: str = "hello", reply_to_request_id: str | None = None) -> AgentRequest:
    return AgentRequest(
        schema_version=1,
        request_id=request_id,
        session_id="s1",
        channel_id="test",
        principal_id="p1",
        robot_scope="so101_single_arm",
        text=text,
        received_at=datetime.now(timezone.utc),
        reply_to_request_id=reply_to_request_id,
    )


def _service(tmp_path, outcome):
    events = Events()
    planner = FakePlanner(outcome)
    service = AgentService(
        planner=planner,
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=events,
    )
    return service, planner, events


def test_conversation_is_answered_without_execution(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    service, planner, events = _service(tmp_path, PlannerOutcome(kind="conversation", user_message="你好"))
    try:
        accepted = service.send_message(_request())
        assert accepted.accepted
        assert events.ready.wait(2)
        for _ in range(100):
            if service._store.get_request(_request().to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert planner.calls == 1
        for _ in range(100):
            if service._store.get_request(_request().to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert service._store.get_request(_request().to_key()).state == "ANSWERED"
    finally:
        service.close()


def _wait_for_event(events, event_type):
    # The ledger state flips before the event is published, so wait on the event itself.
    for _ in range(200):
        if any(event.event_type == event_type for event in events.items):
            return
        time.sleep(0.01)
    raise AssertionError(f"no {event_type} event was published")


def test_a_planner_preview_is_published_as_an_event_before_the_answer(tmp_path):
    """The Agent only publishes the preview; channels decide whether to say it."""
    events = Events()
    service = AgentService(
        planner=PreviewingPlanner(PlannerOutcome(kind="conversation", user_message="你好呀")),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=events,
    )
    try:
        assert service.send_message(_request()).accepted
        _wait_for_event(events, "conversation")
        published = [(event.event_type, event.state, event.user_message) for event in events.items]
        assert published[0] == ("reply_preview", "PLANNING", "你好呀")
        assert published[-1][0] == "conversation"
        # The ledger keeps what was offered before validation.
        assert service._store.latest_event_detail(_request().to_key(), "reply_preview")["message"] == "你好呀"
    finally:
        service.close()


def test_a_preview_is_not_published_for_a_stopped_request(tmp_path):
    events = Events()
    service = AgentService(
        planner=PreviewingPlanner(PlannerOutcome(kind="conversation", user_message="你好呀")),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=events,
    )
    cancelled = threading.Event()
    cancelled.set()
    try:
        service._emit_reply_preview(_request().to_key(), "你好呀", cancelled)
        assert events.items == []
    finally:
        service.close()


def _proposal_identity(service):
    for _ in range(200):
        snapshot = service._store.latest_event_detail(_request().to_key(), "proposal_snapshot")
        if snapshot is not None:
            return snapshot["planner_identity"]
        time.sleep(0.01)
    raise AssertionError("no proposal was recorded")


def test_a_direct_command_proposal_is_not_attributed_to_the_model(tmp_path):
    """The ledger must show that no model chose this plan."""
    planner = FakePlanner(PlannerOutcome(kind="conversation", user_message="unused"))
    service = AgentService(
        planner=planner,
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=Events(),
        direct_commands={"挥手": "wave_hello"},
    )
    try:
        assert service.send_message(_request(text="挥手")).accepted
        identity = _proposal_identity(service)
    finally:
        service.close()

    assert planner.calls == 0
    assert identity["route"] == "direct_command"
    assert identity["returned_model"] == "none"


def test_a_planned_proposal_keeps_the_planner_identity(tmp_path):
    outcome = PlannerOutcome(
        kind="workflow",
        user_message="准备挥手",
        summary="wave_hello",
        steps=(CanonicalWorkflowStep(1, "wave_hello"),),
    )
    service = AgentService(
        planner=FakePlanner(outcome),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=Events(),
        direct_commands={"点头": "nod_yes"},
    )
    try:
        assert service.send_message(_request(text="跟大家打个招呼")).accepted
        identity = _proposal_identity(service)
    finally:
        service.close()

    assert identity["route"] == "fake"


def test_conversation_remains_available_when_gateway_is_offline(tmp_path):
    service = AgentService(
        planner=RulePlanner(),
        catalog=UnavailableCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
    )
    try:
        request = _request(text="你好")
        service.send_message(request)
        for _ in range(100):
            if service.get_request(request.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert service.get_request(request.to_key()).state == "ANSWERED"
    finally:
        service.close()


def test_llm_conversation_can_answer_when_gateway_catalog_is_offline(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    service = AgentService(
        planner=FakePlanner(PlannerOutcome(kind="conversation", user_message="我是小智。")),
        catalog=UnavailableCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
    )
    try:
        request = _request(text="你的名字是什么？")
        service.send_message(request)
        for _ in range(100):
            if service.get_request(request.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        record = service.get_request(request.to_key())
        assert record.state == "ANSWERED"
        assert record.terminal.message == "我是小智。"
    finally:
        service.close()


def test_dry_run_keeps_proposal_ready_and_never_calls_execution(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    service, _, events = _service(
        tmp_path,
        PlannerOutcome(
            kind="workflow",
            user_message="准备挥手",
            summary="挥手",
            steps=(CanonicalWorkflowStep(1, "wave_hello"),),
        ),
    )
    try:
        service.send_message(_request())
        assert events.ready.wait(2)
        assert service._store.get_request(_request().to_key()).state == "ANSWERED"
        assert any(event.event_type == "proposal_ready" for event in events.items)
    finally:
        service.close()


def test_duplicate_request_is_idempotent(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    service, planner, events = _service(tmp_path, PlannerOutcome(kind="conversation", user_message="ok"))
    try:
        request = _request()
        service.send_message(request)
        assert events.ready.wait(2)
        service.send_message(request)
        time.sleep(0.05)
        assert planner.calls == 1
    finally:
        service.close()


def test_second_motion_request_is_rejected_while_planning(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    planner = BlockingPlanner(PlannerOutcome(kind="conversation", user_message="ok"))
    service = AgentService(
        planner=planner,
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
    )
    try:
        assert service.send_message(_request("r1")).accepted
        assert planner.started.wait(2)
        second = service.send_message(_request("r2"))
        assert not second.accepted
        assert second.reason_code == "BUSY"
        record = service.get_request(_request("r2").to_key())
        assert record.state == "FAILED"
        assert record.terminal.status == "failed"
        assert record.terminal.error_code == "BUSY"
        retry = service.send_message(_request("r2"))
        assert not retry.accepted
        assert retry.reason_code == "BUSY"
    finally:
        planner.release.set()
        service.close()


def test_busy_request_is_idempotent_and_requires_new_request_id(tmp_path):
    planner = BlockingPlanner(PlannerOutcome(kind="conversation", user_message="ok"))
    service = AgentService(
        planner=planner, catalog=FakeCatalog(), store=SQLiteRequestStore(tmp_path / "requests.sqlite3")
    )
    try:
        assert service.send_message(_request("r1")).accepted
        assert planner.started.wait(2)
        busy = service.send_message(_request("r2"))
        assert not busy.accepted and busy.reason_code == "BUSY"
        replay = service.send_message(_request("r2"))
        assert not replay.accepted and replay.reason_code == "BUSY"
    finally:
        planner.release.set()
        service.close()


def test_execution_requires_non_empty_allowlist(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    planner = FakePlanner(PlannerOutcome(kind="conversation", user_message="ok"))
    with pytest.raises(ValueError, match="non-empty incubation allowlist"):
        AgentService(
            planner=planner,
            catalog=FakeCatalog(),
            store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
            execution_enabled=True,
        )


def test_execution_requires_execution_port(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    with pytest.raises(ValueError, match="requires an execution port"):
        AgentService(
            planner=FakePlanner(PlannerOutcome(kind="conversation", user_message="ok")),
            catalog=FakeCatalog(),
            store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
            execution_enabled=True,
            allowed_skills={"wave_hello"},
        )


def test_read_only_status_returns_gateway_facts(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    service, _, events = _service(
        tmp_path,
        PlannerOutcome(kind="read_only", user_message="checking", query_kind="status"),
    )
    try:
        request = _request()
        service.send_message(request)
        for _ in range(100):
            if service.get_request(request.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        for _ in range(100):
            if any(event.event_type == "read_only" for event in events.items):
                break
            time.sleep(0.01)
        assert "so101_single_arm" in service.get_request(request.to_key()).terminal.message
        assert "运动未授权" in service.get_request(request.to_key()).terminal.message
        assert any(event.event_type == "read_only" for event in events.items)
    finally:
        service.close()


def test_skill_query_uses_deterministic_path_and_bounds_message(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    class LargeCatalog(FakeCatalog):
        def get_catalog(self, status):
            return {
                "skills": [
                    {"name": f"skill_{index:03d}", "planner_visible": True, "semantic_level": "skill"}
                    for index in range(80)
                ]
            }

    planner = FakePlanner(
        PlannerOutcome(
            kind="workflow",
            user_message="不应调用",
            summary="不应调用",
            steps=(CanonicalWorkflowStep(1, "wave_hello"),),
        )
    )
    events = Events()
    service = AgentService(
        planner=planner,
        catalog=LargeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=events,
    )
    try:
        request = _request(text="当前有哪些技能")
        service.send_message(request)
        for _ in range(100):
            record = service.get_request(request.to_key())
            if record.state == "ANSWERED":
                break
            time.sleep(0.01)
        assert record.state == "ANSWERED"
        assert len(record.terminal.message) <= 150
        assert "另有" in record.terminal.message
        assert planner.calls == 0
    finally:
        service.close()


def test_execution_persists_presentation_submission_and_terminal(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    events = RenderedEvents()
    execution = FakeExecution()
    service = AgentService(
        planner=FakePlanner(
            PlannerOutcome(
                kind="workflow",
                user_message="准备挥手",
                summary="挥手",
                steps=(CanonicalWorkflowStep(1, "wave_hello"),),
            )
        ),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        execution=execution,
        event_sink=events,
        presentation=events.gate,
        execution_enabled=True,
        allowed_skills={"wave_hello"},
    )
    try:
        request = _request()
        service.send_message(request)
        for _ in range(100):
            record = service.get_request(request.to_key())
            if record.state == "SUCCEEDED":
                break
            time.sleep(0.01)
        assert record.state == "SUCCEEDED"
        assert record.may_have_submitted
        assert record.task_ref is not None
        assert execution.calls == 1
    finally:
        service.close()


def test_empty_execution_message_still_delivers_terminal_event(tmp_path):
    class EmptyMessageExecution(FakeExecution):
        def execute(self, *args, **kwargs):
            return replace(super().execute(*args, **kwargs), message="")

    events = RenderedEvents()
    conversation = SQLiteConversationStore(tmp_path / "conversation.sqlite3")
    service = AgentService(
        planner=RulePlanner(),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "requests.sqlite3"),
        conversation=conversation,
        execution=EmptyMessageExecution(),
        event_sink=events,
        presentation=events.gate,
        execution_enabled=True,
        allowed_skills={"wave_hello"},
    )
    try:
        request = _request(text="请挥手")
        assert service.send_message(request).accepted
        deadline = time.monotonic() + 3.0
        while not any(event.event_type == "terminal" for event in events.items) and time.monotonic() < deadline:
            time.sleep(0.01)
        terminal = [event for event in events.items if event.event_type == "terminal"]
        assert len(terminal) == 1
        assert terminal[0].state == "SUCCEEDED"
        assert service.get_request(request.to_key()).state == "SUCCEEDED"
        assert conversation.context(request)[-1]["content"] == "执行完成。"
    finally:
        service.close()


def test_natural_language_wave_routes_to_skill_execution(tmp_path):
    execution = FakeExecution()
    events = RenderedEvents()
    service = AgentService(
        planner=RulePlanner(),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        execution=execution,
        event_sink=events,
        presentation=events.gate,
        execution_enabled=True,
        allowed_skills={"wave_hello"},
    )
    try:
        request = _request(text="请挥手")
        assert service.send_message(request).accepted
        for _ in range(200):
            record = service.get_request(request.to_key())
            if record.state == "SUCCEEDED":
                break
            time.sleep(0.01)
        assert record.state == "SUCCEEDED"
        assert execution.calls == 1
        assert record.task_ref is not None
        assert record.may_have_submitted
    finally:
        service.close()


def test_unknown_execution_quarantines_robot_scope(tmp_path):
    from ibrobot_agent.contracts import PlannerOutcome

    execution = FakeExecution(status="unknown")
    events = RenderedEvents()
    service = AgentService(
        planner=FakePlanner(
            PlannerOutcome(
                kind="workflow",
                user_message="准备挥手",
                summary="挥手",
                steps=(CanonicalWorkflowStep(1, "wave_hello"),),
            )
        ),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        execution=execution,
        event_sink=events,
        presentation=events.gate,
        execution_enabled=True,
        allowed_skills={"wave_hello"},
    )
    try:
        first = _request("r1")
        service.send_message(first)
        for _ in range(100):
            if service.get_request(first.to_key()).state == "UNKNOWN":
                break
            time.sleep(0.01)
        second = service.send_message(_request("r2"))
        assert not second.accepted
        assert second.reason_code == "ROBOT_QUARANTINED"
    finally:
        service.close()


def test_unknown_quarantine_survives_service_restart(tmp_path):
    path = tmp_path / "requests.sqlite3"
    store = SQLiteRequestStore(path)
    key = _request().to_key()
    store.admit(key, input_hash="hash", session_id="session-1")
    store.begin_planning(key, expected_generation=0)
    store.quarantine(key, expected_generation=1, reason="x" * 301)
    assert store.is_robot_quarantined(key.robot_scope)
    store.close()

    restarted = SQLiteRequestStore(path)
    assert restarted.is_robot_quarantined(key.robot_scope)
    record = restarted.get_request(key)
    assert record.state == "UNKNOWN"
    assert record.terminal.error_code == "ROBOT_QUARANTINED"
    assert len(record.terminal.message) == 300
    restarted.close()


def test_terminal_state_is_persisted_when_event_transport_fails(tmp_path):
    service = AgentService(
        planner=FakePlanner(PlannerOutcome(kind="conversation", user_message="ok")),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "requests.sqlite3"),
        event_sink=FailingEvents(),
    )
    try:
        request = _request()
        assert service.send_message(request).accepted
        for _ in range(100):
            if service.get_request(request.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert service.get_request(request.to_key()).state == "ANSWERED"
    finally:
        service.close()


def test_terminal_persistence_failure_quarantines_service(tmp_path):
    service = AgentService(
        planner=FakePlanner(PlannerOutcome(kind="conversation", user_message="ok")),
        catalog=FakeCatalog(),
        store=FailingFinishStore(tmp_path / "requests.sqlite3"),
    )
    try:
        assert service.send_message(_request()).accepted
        for _ in range(100):
            if not service.healthy:
                break
            time.sleep(0.01)
        assert not service.healthy
        rejected = service.send_message(_request("r2"))
        assert not rejected.accepted
        assert rejected.reason_code == "STORAGE_UNAVAILABLE"
    finally:
        service.close()


@pytest.mark.parametrize("failure_stage", ["before_write", "after_write", "storage_unavailable"])
def test_submission_persistence_failure_uses_ledger_state(tmp_path, monkeypatch, failure_stage):
    store = SQLiteRequestStore(tmp_path / "request.sqlite3")
    execution = FakeExecution()
    events = RenderedEvents()
    mark_submitted = store.mark_submitted
    get_request = store.get_request

    def unavailable(*args, **kwargs):
        raise RuntimeError("ledger unavailable")

    def failing_submission(*args, **kwargs):
        if failure_stage != "before_write":
            mark_submitted(*args, **kwargs)
        if failure_stage == "storage_unavailable":
            monkeypatch.setattr(store, "get_request", unavailable)
        raise RuntimeError("submission persistence interrupted")

    monkeypatch.setattr(store, "mark_submitted", failing_submission)
    service = AgentService(
        planner=RulePlanner(),
        catalog=FakeCatalog(),
        store=store,
        execution=execution,
        event_sink=events,
        presentation=events.gate,
        execution_enabled=True,
        allowed_skills={"wave_hello"},
    )
    request = _request(text="请挥手")
    try:
        assert service.send_message(request).accepted
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            record = get_request(request.to_key())
            if (failure_stage == "before_write" and record.state == "FAILED") or not service.healthy:
                break
            time.sleep(0.01)
        record = get_request(request.to_key())
        assert execution.calls == 1
        assert execution.submitted == 0
        if failure_stage == "before_write":
            assert record.state == "FAILED"
            assert not record.may_have_submitted
            assert service.healthy
        else:
            assert record.may_have_submitted
            assert not service.healthy
            if failure_stage == "after_write":
                assert record.state == "UNKNOWN"
                assert store.is_robot_quarantined(request.robot_scope)
    finally:
        service.close()


@pytest.mark.parametrize("broken_transport", [False, True])
def test_no_display_receipt_or_failed_transport_prevents_submission(tmp_path, broken_transport):
    execution = FakeExecution()
    events = FailingEvents() if broken_transport else Events()
    service = AgentService(
        planner=RulePlanner(),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "requests.sqlite3"),
        execution=execution,
        presentation=PresentationGate(0.05),
        event_sink=events,
        execution_enabled=True,
        allowed_skills={"wave_hello"},
    )
    req = _request(text="请挥手")
    try:
        assert service.send_message(req).accepted
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and service.get_request(req.to_key()).state != "FAILED":
            time.sleep(0.01)
        record = service.get_request(req.to_key())
        assert record.state == "FAILED"
        assert not record.may_have_submitted
        assert execution.submitted == 0
    finally:
        service.close()


def _async_service(tmp_path, events, *, store=None, execution=None):
    execution = execution or FakeExecution(execution_mode=ASYNC_PRESENTATION_MODE)
    service = AgentService(
        planner=RulePlanner(),
        catalog=FakeCatalog(),
        store=store or SQLiteRequestStore(tmp_path / "requests.sqlite3"),
        execution=execution,
        presentation=ModeRoutingPresentationPort(
            async_port=AsyncPresentationPort(), legacy_port=PresentationGate(0.05)
        ),
        event_sink=events,
        execution_enabled=True,
        allowed_skills={"wave_hello"},
    )
    return service, execution


def _wait_state(service, key, states, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and service.get_request(key).state not in states:
        time.sleep(0.01)
    return service.get_request(key)


def test_long_failure_is_persisted_and_does_not_quarantine_robot(tmp_path):
    message = "finger drift exceeded; preparation=" + "x" * 2208

    class LongFailure(FakeExecution):
        def execute(self, *args, **kwargs):
            return replace(super().execute(*args, **kwargs), message=message)

    events = Events()
    execution = LongFailure(status="failed", execution_mode=ASYNC_PRESENTATION_MODE)
    service, _ = _async_service(tmp_path, events, execution=execution)
    request = _request(text="请挥手")
    try:
        assert service.send_message(request).accepted
        record = _wait_state(service, request.to_key(), {"FAILED", "UNKNOWN"})
        assert record.state == "FAILED"
        assert record.terminal.detail["raw_message"] == message
        assert len(record.terminal.message) == 300
        assert not service._store.is_robot_quarantined(request.robot_scope)
        _wait_for_event(events, "terminal")
        assert not any(event.state == "UNKNOWN" for event in events.items)
        assert execution.stop_calls == 0
    finally:
        service.close()


@pytest.mark.parametrize("broken_transport", [False, True])
def test_async_program_executes_without_receipt_or_working_display(tmp_path, broken_transport):
    events = FailingEvents() if broken_transport else Events()
    service, execution = _async_service(tmp_path, events)
    req = _request(text="请挥手")
    try:
        assert service.send_message(req).accepted
        record = _wait_state(service, req.to_key(), {"SUCCEEDED", "FAILED", "UNKNOWN"})
        assert record.state == "SUCCEEDED"
        assert execution.submitted == 1
        if not broken_transport:
            types = [event.event_type for event in events.items]
            assert "presentation" in types
            assert "presentation_rendered" not in types
            presentation = next(event for event in events.items if event.event_type == "presentation")
            assert presentation.detail["async_display"] is True
    finally:
        service.close()


def test_async_presentation_is_published_only_after_the_task_binding_is_durable(tmp_path):
    store = SQLiteRequestStore(tmp_path / "requests.sqlite3")
    observed = []

    class BindingProbe(Events):
        def publish(self, event):
            if event.event_type == "presentation":
                record = store.get_request(event.request_key)
                observed.append((record.state, record.task_ref is not None, record.may_have_submitted))
            super().publish(event)

    service, execution = _async_service(tmp_path, BindingProbe(), store=store)
    req = _request(text="请挥手")
    try:
        assert service.send_message(req).accepted
        assert _wait_state(service, req.to_key(), {"SUCCEEDED"}).state == "SUCCEEDED"
        assert observed == [("MAY_EXECUTE", True, False)]
    finally:
        service.close()


def test_concurrent_emits_publish_in_sequence_order(tmp_path):
    recorded = threading.Event()

    class SlowRecordStore(SQLiteRequestStore):
        def record_event(self, *args, **kwargs):
            sequence = super().record_event(*args, **kwargs)
            if not recorded.is_set():
                recorded.set()
                time.sleep(0.2)  # widen the window between sequence allocation and publish
            return sequence

    store = SlowRecordStore(tmp_path / "request.sqlite3")
    key = _request().to_key()
    store.admit(key, input_hash="hash", session_id="session-1")
    events = Events()
    service = AgentService(planner=FakePlanner(None), catalog=FakeCatalog(), store=store, event_sink=events)
    try:
        terminal = threading.Thread(target=service._emit, args=(key, "terminal", "CANCELLED", "done"))
        terminal.start()
        assert recorded.wait(2)
        service._emit(key, "stop_requested", "STOPPING", "stop requested")
        terminal.join(2)
        assert [event.event_type for event in events.items] == ["terminal", "stop_requested"]
        sequences = [event.sequence for event in events.items]
        assert sequences == sorted(sequences)
    finally:
        service.close()


def test_async_task_binding_failure_never_submits(tmp_path):
    class FailingBindStore(SQLiteRequestStore):
        def bind_task(self, *args, **kwargs):
            raise RuntimeError("ledger unavailable")

    service, execution = _async_service(tmp_path, Events(), store=FailingBindStore(tmp_path / "requests.sqlite3"))
    req = _request(text="请挥手")
    try:
        assert service.send_message(req).accepted
        record = _wait_state(service, req.to_key(), {"FAILED", "UNKNOWN"})
        assert record.state == "FAILED"
        assert not record.may_have_submitted
        assert execution.submitted == 0
    finally:
        service.close()


@pytest.mark.parametrize("text", ["当前状态如何？", "list skills", "有哪些姿态"])
def test_factual_queries_bypass_model_planning(tmp_path, text):
    service, planner, events = _service(tmp_path, PlannerOutcome(kind="conversation", user_message="unused"))
    request = _request(text=text)
    try:
        assert service.send_message(request).accepted
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and service.get_request(request.to_key()).state != "ANSWERED":
            time.sleep(0.01)
        assert service.get_request(request.to_key()).state == "ANSWERED"
        assert planner.calls == 0
    finally:
        service.close()


def test_quarantine_rejects_terminal_request(tmp_path):
    store = SQLiteRequestStore(tmp_path / "request.sqlite3")
    key = _request().to_key()
    store.admit(key, input_hash="hash", session_id="session-1")
    store.finish(
        key,
        expected_generation=0,
        result=ExecutionResult(status="succeeded", task_ref=None, error_code="", message="done", detail={}),
    )
    with pytest.raises(Exception, match="terminal"):
        store.quarantine(key, expected_generation=0, reason="late unknown")
    store.close()


def test_clarification_reply_is_bound_and_consumed(tmp_path):
    service = AgentService(
        planner=RulePlanner(),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
    )
    try:
        clarification = _request("clarify-1", text="帮我做一个动作")
        service.send_message(clarification)
        for _ in range(100):
            if service.get_request(clarification.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        answer = _request("answer-1", text="挥手", reply_to_request_id="clarify-1")
        service.send_message(answer)
        for _ in range(100):
            if service.get_request(answer.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert service.get_request(answer.to_key()).terminal.error_code == "DRY_RUN_ONLY"

        duplicate = _request("answer-2", text="挥手", reply_to_request_id="clarify-1")
        service.send_message(duplicate)
        for _ in range(100):
            if service.get_request(duplicate.to_key()).state == "ANSWERED":
                break
            time.sleep(0.01)
        assert service.get_request(duplicate.to_key()).terminal.error_code == "CLARIFICATION_EXPIRED"
    finally:
        service.close()


def test_step_schema_version_comes_from_the_catalog_contract():
    # Observed on hardware: the model emitted schema_version 2 for an arm skill
    # and the Gateway rejected the Program with SKILL_SCHEMA_INVALID.
    from ibrobot_agent.contracts import PlannerOutcome

    catalog = {"skills": [{"name": "wave_hello"}, {"name": "nav_turn", "schema_version": 2}]}

    def bind(*steps):
        outcome = PlannerOutcome(kind="workflow", user_message="执行", summary="执行", steps=steps)
        return AgentService._bind_step_schema_versions(outcome, catalog).steps

    bound = bind(CanonicalWorkflowStep(2, "wave_hello"), CanonicalWorkflowStep(1, "nav_turn"))
    assert [(step.skill_name, step.schema_version) for step in bound] == [("wave_hello", 1), ("nav_turn", 2)]
    with pytest.raises(RuntimeError, match="cannot be bound"):
        bind(CanonicalWorkflowStep(2, "wave_hello", degree=30.0))

    # A malformed or unsupported catalog version is a planner failure, not a crash.
    for bad in (None, "1", True, 3):
        catalog = {"skills": [{"name": "wave_hello", "schema_version": bad}]}
        with pytest.raises(RuntimeError):
            bind(CanonicalWorkflowStep(1 if bad != 1 else 2, "wave_hello"))


class NamedCatalog(FakeCatalog):
    """A Gateway catalog carrying Chinese skill names, as the node's catalog adapter adds them."""

    def get_catalog(self, status):
        return {
            "robot_name": "aimdk_x2",
            "skills": [
                {"name": "wave_hand", "planner_visible": True, "semantic_level": "skill"},
                {"name": "handshake", "planner_visible": True, "semantic_level": "skill"},
            ],
            CATALOG_SKILL_NAMES_ZH: {"wave_hand": "挥手", "handshake": "握手"},
        }


class CatalogRecordingPlanner(FakePlanner):
    def plan(self, request, context, catalog, cancel_token, *, on_preview=None):
        self.catalog = catalog
        return super().plan(request, context, catalog, cancel_token, on_preview=on_preview)


def test_plan_is_presented_with_the_chinese_catalog_names_of_its_steps(tmp_path):
    # Observed on X2: the model summarized the plan as "wave_hand", the voice channel spoke
    # that summary, and the Chinese TTS frontend rejected it, so the gesture reply was silent.
    events = Events()
    planner = CatalogRecordingPlanner(
        PlannerOutcome(
            kind="workflow",
            user_message="准备执行：wave_hand、handshake。",
            summary="wave_hand、handshake",
            steps=(CanonicalWorkflowStep(1, "wave_hand"), CanonicalWorkflowStep(1, "handshake")),
        )
    )
    service = AgentService(
        planner=planner,
        catalog=NamedCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        execution=FakeExecution(execution_mode=ASYNC_PRESENTATION_MODE),
        presentation=ModeRoutingPresentationPort(
            async_port=AsyncPresentationPort(), legacy_port=PresentationGate(0.05)
        ),
        event_sink=events,
        execution_enabled=True,
        allowed_skills={"wave_hand", "handshake"},
    )
    try:
        request = _request(text="挥手然后握手")
        assert service.send_message(request).accepted
        terminal = _wait_state(service, request.to_key(), {"SUCCEEDED"})
        assert terminal.state == "SUCCEEDED", terminal
        presentation = next(event for event in events.items if event.event_type == "presentation")
        assert presentation.detail["presentation"]["summary"] == "挥手、握手"
        # The names only word replies: what the planner is given is unchanged.
        assert CATALOG_SKILL_NAMES_ZH not in planner.catalog
    finally:
        service.close()


def test_plan_summary_is_kept_when_a_step_has_no_chinese_catalog_name():
    """A partly named plan is never partly rewritten; a renderer that cannot use it fails visibly."""
    outcome = PlannerOutcome(
        kind="workflow",
        user_message="执行",
        summary="wave_hand、nav_turn",
        steps=(CanonicalWorkflowStep(1, "wave_hand"), CanonicalWorkflowStep(1, "nav_turn")),
    )

    for catalog in ({CATALOG_SKILL_NAMES_ZH: {"wave_hand": "挥手"}}, {}):
        assert AgentService._bind_summary_to_catalog_names(outcome, catalog).summary == "wave_hand、nav_turn"


def test_terminal_preserves_admission_error_for_the_voice_channel(tmp_path):
    class BlockedExecution(FakeExecution):
        def execute(self, *args, **kwargs):
            result = super().execute(*args, **kwargs)
            return replace(result, status="failed", error_code="HELD_OBJECT_RECOVERY_REQUIRED")

    events = Events()
    service = AgentService(
        planner=FakePlanner(
            PlannerOutcome(
                kind="workflow", user_message="挥手", summary="挥手", steps=(CanonicalWorkflowStep(1, "wave_hello"),)
            )
        ),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        execution=BlockedExecution(execution_mode=ASYNC_PRESENTATION_MODE),
        presentation=ModeRoutingPresentationPort(
            async_port=AsyncPresentationPort(), legacy_port=PresentationGate(0.05)
        ),
        event_sink=events,
        execution_enabled=True,
        allowed_skills={"wave_hello"},
    )
    try:
        request = _request(text="挥手")
        assert service.send_message(request).accepted
        _wait_state(service, request.to_key(), {"FAILED"})
        _wait_for_event(events, "terminal")
        terminal = next(event for event in events.items if event.event_type == "terminal")
        assert terminal.detail["error_code"] == "HELD_OBJECT_RECOVERY_REQUIRED"
        assert "需要先恢复" in terminal.user_message
    finally:
        service.close()


@pytest.mark.parametrize("catalog_name", ["柠檬", "抓取水果"])
@pytest.mark.parametrize(
    ("target", "expected"),
    [("a lemon", "抓取柠檬"), ("an apple", "抓取苹果"), ("an orange", "抓取橙子"), ("a red cup", "抓取物体")],
)
def test_pick_summary_names_the_requested_object_instead_of_the_first_fruit_alias(catalog_name, target, expected):
    outcome = PlannerOutcome(
        kind="workflow",
        user_message="准备抓取并放进容器。",
        summary="pick_object、place_in_container",
        steps=(
            CanonicalWorkflowStep(1, "pick_object", target_name=target),
            CanonicalWorkflowStep(1, "place_in_container", target_name=target, container_name="an orange bowl"),
        ),
    )
    catalog = {CATALOG_SKILL_NAMES_ZH: {"pick_object": catalog_name, "place_in_container": "放进容器"}}
    bound = AgentService._bind_summary_to_catalog_names(outcome, catalog)
    assert bound.summary == f"{expected}、放进容器"
    assert bound.steps == outcome.steps


@pytest.mark.parametrize("target", ["双手举过头顶", "arms_up"])
def test_pose_summary_cannot_name_the_opposite_first_seed_alias(target):
    outcome = PlannerOutcome(
        kind="workflow",
        user_message="举起双手再回准备位。",
        summary="举起双手、回准备位",
        steps=(
            CanonicalWorkflowStep(1, "free_pose", target_name=target),
            CanonicalWorkflowStep(1, "return_to_task_pose", target_name="grasp_ready"),
        ),
    )
    catalog = {CATALOG_SKILL_NAMES_ZH: {"free_pose": "双臂放下", "return_to_task_pose": "回准备位"}}
    bound = AgentService._bind_summary_to_catalog_names(outcome, catalog)
    assert bound.summary == ("双手举过头顶、回准备位" if target != "arms_up" else outcome.summary)
    assert bound.steps == outcome.steps


def test_skill_list_names_skills_the_way_plans_do(tmp_path):
    class PartlyNamedCatalog(NamedCatalog):
        def get_catalog(self, status):
            catalog = super().get_catalog(status)
            catalog["skills"].append({"name": "nav_turn", "planner_visible": True, "semantic_level": "skill"})
            return catalog

    service = AgentService(
        planner=FakePlanner(PlannerOutcome(kind="conversation", user_message="unused")),
        catalog=PartlyNamedCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=Events(),
    )
    try:
        request = _request(text="当前有哪些技能")
        assert service.send_message(request).accepted
        record = _wait_state(service, request.to_key(), {"ANSWERED"})
        # A skill without a Chinese name keeps its own name instead of being left out.
        assert record.terminal.message == "当前可规划技能：握手、nav_turn、挥手。"
    finally:
        service.close()


def test_a_repaired_planner_reply_is_answered_and_its_repairs_audited(tmp_path, caplog):
    outcome = PlannerOutcome(
        kind="needs_clarification",
        user_message="你想让我挥哪只手？",
        missing_fields=("request",),
        repairs=(PlannerRepair("missing_fields", "defaulted", None),),
    )
    service, _planner, _events = _service(tmp_path, outcome)
    try:
        request = _request(text="挥手")
        assert service.send_message(request).accepted
        record = _wait_state(service, request.to_key(), {"ANSWERED"})

        # The user hears the question rather than a failed execution, and the operator can see why.
        assert record.terminal.message == "你想让我挥哪只手？"
        detail = service._store.latest_event_detail(request.to_key(), "planner_outcome")
        assert detail["planner_repairs"] == [{"field": "missing_fields", "action": "defaulted", "original": None}]
        assert any("repaired" in entry.getMessage() for entry in caplog.records if entry.levelname == "WARNING")
    finally:
        service.close()


def test_catalog_target_alias_is_bound_before_execution():
    from embodied_common.workflow_contracts import CanonicalWorkflowStep

    outcome = PlannerOutcome(
        kind="workflow",
        user_message="好的",
        summary="回位",
        steps=(CanonicalWorkflowStep(1, "return_to_task_pose", target_name="抓取初始位置"),),
    )
    skill = {
        "name": "return_to_task_pose",
        "parameters": {"properties": {"target_name": {"enum": ["grasp_ready"]}}},
        "target_options": [{"target_name": "grasp_ready", "aliases": ["抓取初始位置"]}],
    }
    result = AgentService._bind_catalog_targets(outcome, {"skills": [skill]})
    assert result.steps[0].target_name == "grasp_ready"
    assert outcome.steps[0].target_name == "抓取初始位置"
    skill["target_options"] = []
    with pytest.raises(RuntimeError, match="invalid or ambiguous"):
        AgentService._bind_catalog_targets(outcome, {"skills": [skill]})


def test_catalog_target_rejects_ambiguous_alias():
    from embodied_common.workflow_contracts import CanonicalWorkflowStep

    outcome = PlannerOutcome(
        kind="workflow",
        user_message="好的",
        summary="姿态",
        steps=(CanonicalWorkflowStep(1, "free_pose", target_name="举手"),),
    )
    skill = {
        "name": "free_pose",
        "parameters": {"properties": {"target_name": {"enum": ["left_up", "right_up"]}}},
        "target_options": [{"target_name": name, "aliases": ["举手"]} for name in ("left_up", "right_up")],
    }
    with pytest.raises(RuntimeError, match="invalid or ambiguous"):
        AgentService._bind_catalog_targets(outcome, {"skills": [skill]})


@pytest.mark.parametrize("delivery", [None, {"container_name": "configured bowl", "container_label": "配置的碗"}])
def test_summit_service_refuses_unpaired_pick_before_proposal(tmp_path, delivery):
    outcome = PlannerOutcome(
        kind="workflow",
        user_message="好。",
        summary="水果",
        steps=(CanonicalWorkflowStep(1, "pick_object", target_name="an apple"),),
    )
    service = AgentService(
        planner=FakePlanner(outcome),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "fruit.db"),
        robot_scope="aimdk_x2_summit",
        fruit_delivery=delivery,
    )
    try:
        with pytest.raises(ValueError, match="fruit"):
            service._validate_workflow(
                outcome, {"skills": [{"name": "pick_object", "planner_visible": True, "semantic_level": "skill"}]}
            )
    finally:
        service.close()
