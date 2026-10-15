"""「继续」: the Agent resolves it against its own ledger and the Gateway ledger."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import pytest

from embodied_common.control_phrases import STOP_PHRASES
from ibrobot_agent.continuation import (
    CONTINUE_REFUSED,
    decide_continuation,
    is_continue_request,
    select_anchor,
)
from ibrobot_agent.contracts import (
    AgentRequest,
    ExecutionResult,
    Presentation,
    RequestKey,
    RequestRecord,
    TaskRef,
)
from ibrobot_agent.planner import RulePlanner
from ibrobot_agent.presentation import PresentationGate
from ibrobot_agent.request_store import SQLiteRequestStore
from ibrobot_agent.service import AgentService

THREE_STEPS = "挥手然后点头再摇头"
SKILLS = ("wave_hello", "nod_yes", "shake_no")
TASK_REF = TaskRef("task-1", "plan-1", "plan-digest", "epoch", 1, "digest", 3)


def _steps(*names):
    return [{"schema_version": 1, "skill_name": name} for name in names]


def _record(state, *, request_id="r1", text=THREE_STEPS, detail=None, error_code="", task_ref=TASK_REF):
    proposal = {"request": {"text": text}, "outcome": {"steps": _steps(*SKILLS)}} if text else None
    status = {
        "CANCELLED": "cancelled",
        "SUCCEEDED": "succeeded",
        "FAILED": "failed",
        "UNKNOWN": "unknown",
        "CANCELLED_BEFORE_EXECUTION": "cancelled",
        "ANSWERED": "succeeded",
    }.get(state)
    terminal = (
        None
        if status is None
        else ExecutionResult(status, task_ref, error_code, "", detail if detail is not None else {})
    )
    return RequestRecord(
        key=RequestKey("so101_single_arm", "voice_asr", "local_operator", request_id),
        session_id="s1",
        state=state,
        planning_generation=1,
        stop_requested=state == "CANCELLED",
        may_have_submitted=task_ref is not None,
        input_hash="hash",
        task_ref=task_ref,
        proposal_json=json.dumps(proposal) if proposal else "",
        terminal=terminal,
    )


def _canceled(completed, execution_id="exec-1"):
    return {
        "found": True,
        "execution_id": execution_id,
        "state": "CANCELED",
        "completed_step_count": completed,
        "canonical_steps": _steps(*SKILLS),
    }


@pytest.mark.parametrize("text", ["继续", "继续。", " 继续吧 ", "Go on", "continue"])
def test_continue_vocabulary_is_normalized(text):
    assert is_continue_request(text)


@pytest.mark.parametrize("text", ["继续挥手", "不要继续", "挥手"])
def test_commands_that_mention_continue_are_not_continuations(text):
    assert not is_continue_request(text)


def test_anchor_skips_answers_and_admission_rejections():
    answered = _record("ANSWERED", request_id="chat", task_ref=None)
    busy = _record("FAILED", request_id="busy", task_ref=None, error_code="BUSY")
    cancelled = _record("CANCELLED", request_id="task")

    assert select_anchor([answered, busy, cancelled]).key.request_id == "task"
    assert select_anchor([answered]) is None


def test_confirmed_cancellation_continues_with_completed_progress():
    decision = decide_continuation(
        _record("CANCELLED", detail={"submitted": True, "execution_id": "exec-1"}), gateway=_canceled(1)
    )

    assert decision.text == THREE_STEPS
    assert decision.error_code == ""
    assert "1/3" in decision.message
    # Spoken by the voice bridge while the continued request is interruptible: hearing a stop
    # word in its own reply would stop the robot.
    assert not any(phrase in decision.message for phrase in STOP_PHRASES)
    assert decision.hint["completed_step_count"] == 1
    assert decision.hint["completed_steps"] == [{"skill_name": "wave_hello"}]
    assert decision.hint["interrupted_step"] == {"skill_name": "nod_yes"}
    assert decision.hint["continues_from_execution_id"] == "exec-1"


def test_stop_before_submission_restarts_the_whole_task():
    decision = decide_continuation(_record("CANCELLED", detail={"submitted": False}), gateway={"found": False})

    assert decision.text == THREE_STEPS
    assert decision.hint["completed_step_count"] == 0
    assert decision.hint["interrupted_step"] == {"skill_name": "wave_hello"}


@pytest.mark.parametrize(
    ("anchor", "gateway", "gateway_error"),
    [
        # The Gateway cannot be asked: never guess that the robot stopped safely.
        (_record("CANCELLED", detail={"submitted": True}), None, "service timeout"),
        # Agent says submitted, Gateway has no such execution.
        (_record("CANCELLED", detail={"submitted": True}), {"found": False}, ""),
        # Gateway disagrees about the outcome.
        (_record("CANCELLED", detail={"submitted": True}), {**_canceled(1), "state": "UNKNOWN"}, ""),
        # Gateway execution id differs from the one the Agent recorded.
        (_record("CANCELLED", detail={"submitted": True, "execution_id": "exec-1"}), _canceled(1, "exec-2"), ""),
        # Progress beyond the plan is corrupt.
        (_record("CANCELLED", detail={"submitted": True}), _canceled(7), ""),
        (_record("FAILED"), None, ""),
        (_record("UNKNOWN"), None, ""),
        (_record("CANCELLED_BEFORE_EXECUTION", text="", task_ref=None), None, ""),
        # No bound proposal means no original text to re-plan.
        (_record("CANCELLED", text=""), _canceled(1), ""),
    ],
)
def test_unconfirmed_or_inconsistent_ledgers_refuse_without_motion(anchor, gateway, gateway_error):
    decision = decide_continuation(anchor, gateway=gateway, gateway_error=gateway_error)

    assert decision.text is None
    assert decision.error_code == CONTINUE_REFUSED


@pytest.mark.parametrize(
    ("anchor", "gateway"),
    [(None, None), (_record("SUCCEEDED"), None), (_record("CANCELLED", detail={"submitted": True}), _canceled(3))],
)
def test_nothing_left_to_continue_is_answered_without_motion(anchor, gateway):
    decision = decide_continuation(anchor, gateway=gateway)

    assert decision.text is None and decision.error_code == ""


def test_chained_continuation_keeps_steps_finished_before_the_first_stop():
    first = decide_continuation(_record("CANCELLED", detail={"submitted": True}), gateway=_canceled(1))
    second_gateway = {**_canceled(1), "canonical_steps": _steps("nod_yes", "shake_no")}

    second = decide_continuation(
        _record("CANCELLED", request_id="r2", detail={"submitted": True}),
        gateway=second_gateway,
        previous_hint=first.hint,
    )

    assert second.text == THREE_STEPS
    assert second.hint["completed_step_count"] == 2
    assert second.hint["step_count"] == 3
    assert second.hint["interrupted_step"] == {"skill_name": "shake_no"}


# ---------------------------------------------------------------------------
# AgentService end to end with the deterministic planner and a fake Gateway.


class _Catalog:
    def get_status(self):
        return {
            "robot_name": "so101_single_arm",
            "active_control_mode": "moveit_planning",
            "control_plane_ready": True,
            "motion_authorized": True,
            "registry_epoch": "epoch",
            "registry_generation": 1,
            "registry_digest": "digest",
        }

    def get_catalog(self, status):
        return {"skills": [{"name": name, "planner_visible": True, "semantic_level": "skill"} for name in SKILLS]}


class _Events:
    def __init__(self):
        self.items = []
        self.gate = PresentationGate(0.5)

    def publish(self, event):
        self.items.append(event)
        if event.event_type == "presentation":
            assert self.gate.acknowledge(
                event.request_key, digest=event.detail["presentation_digest"], token=event.detail["receipt_token"]
            )


class _Gateway:
    """Cancels each Program after ``stop_after`` steps and keeps a Gateway-style ledger."""

    def __init__(self, stop_after):
        self.stop_after = list(stop_after)
        self.programs: list[list[str]] = []
        self.ledger: dict[str, dict] = {}
        self.lookup_error: Exception | None = None

    def execute(self, proposal, *, expected_registry_identity, presentation_callback, submission_callback, stop_event):
        steps = proposal.outcome.steps
        task_ref = TaskRef(f"task-{len(self.programs)}", "plan", "digest", "epoch", 1, "digest", len(steps))
        presentation_callback(
            Presentation(task_ref, 1 if len(steps) == 1 else 2, steps, "immediate_after_presentation", 90.0, "plan")
        )
        submission_callback(task_ref, {})
        names = [step.skill_name for step in steps]
        self.programs.append(names)
        request_id = proposal.request.request_id
        execution_id = f"exec-{request_id}"
        completed = self.stop_after.pop(0) if self.stop_after else len(names)
        state = "CANCELED" if completed < len(names) else "SUCCEEDED"
        self.ledger[request_id] = {
            "found": True,
            "execution_id": execution_id,
            "state": state,
            "completed_step_count": completed,
            "canonical_steps": _steps(*names),
        }
        detail = {"submitted": True, "execution_id": execution_id, "state": state, "completed_step_count": completed}
        if state == "CANCELED":
            return ExecutionResult("cancelled", task_ref, "SKILL_CANCELLED", "stopped", detail)
        return ExecutionResult("succeeded", task_ref, "", "done", detail)

    def request_stop(self, request_key):
        return None

    def lookup_execution(self, request_id):
        if self.lookup_error is not None:
            raise self.lookup_error
        return self.ledger.get(request_id, {"found": False})


def _service(tmp_path, gateway):
    events = _Events()
    service = AgentService(
        planner=RulePlanner(),
        catalog=_Catalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        execution=gateway,
        event_sink=events,
        presentation=events.gate,
        execution_enabled=True,
        allowed_skills=set(SKILLS),
    )
    return service, events


def _say(service, request_id, text, *, session_id="s1"):
    request = AgentRequest(
        schema_version=1,
        request_id=request_id,
        session_id=session_id,
        channel_id="voice_asr",
        principal_id="local_operator",
        robot_scope="so101_single_arm",
        text=text,
        received_at=datetime.now(timezone.utc),
    )
    assert service.send_message(request).accepted
    terminal = {"SUCCEEDED", "FAILED", "CANCELLED", "CANCELLED_BEFORE_EXECUTION", "ANSWERED", "UNKNOWN"}
    for _ in range(300):
        record = service.get_request(request.to_key())
        if record.state in terminal:
            return record
        time.sleep(0.01)
    raise AssertionError(f"{request_id} did not finish: {record.state}")


def _events_of(events, request_id, event_type):
    return [
        item for item in events.items if item.request_key.request_id == request_id and item.event_type == event_type
    ]


def test_continue_replans_only_the_unfinished_steps(tmp_path):
    gateway = _Gateway(stop_after=[1])
    service, events = _service(tmp_path, gateway)
    try:
        assert _say(service, "r1", THREE_STEPS).state == "CANCELLED"
        record = _say(service, "r2", "继续")

        assert record.state == "SUCCEEDED"
        assert gateway.programs == [list(SKILLS), ["nod_yes", "shake_no"]]
        (continuation,) = _events_of(events, "r2", "continuation")
        assert "1/3" in continuation.user_message
        assert continuation.detail["continuation"]["continues_from_request_id"] == "r1"
        # The ledger keeps the original text, so a later "继续" can chain from this request.
        assert json.loads(record.proposal_json)["request"]["text"] == THREE_STEPS
    finally:
        service.close()


def test_chained_stop_and_continue_never_repeats_finished_steps(tmp_path):
    gateway = _Gateway(stop_after=[1, 1])
    service, _events = _service(tmp_path, gateway)
    try:
        _say(service, "r1", THREE_STEPS)
        assert _say(service, "r2", "继续").state == "CANCELLED"
        assert _say(service, "r3", "继续").state == "SUCCEEDED"

        assert gateway.programs == [list(SKILLS), ["nod_yes", "shake_no"], ["shake_no"]]
        # Nothing is left after a success.
        answered = _say(service, "r4", "继续")
        assert answered.state == "ANSWERED"
        assert len(gateway.programs) == 3
    finally:
        service.close()


def test_unreadable_gateway_ledger_refuses_without_motion(tmp_path):
    gateway = _Gateway(stop_after=[1])
    service, events = _service(tmp_path, gateway)
    try:
        _say(service, "r1", THREE_STEPS)
        gateway.lookup_error = RuntimeError("GetExecution timed out")

        record = _say(service, "r2", "继续")

        assert record.state == "ANSWERED"
        assert record.terminal.error_code == CONTINUE_REFUSED
        assert len(gateway.programs) == 1
        assert _events_of(events, "r2", "continuation_refused")
    finally:
        service.close()


def test_continue_only_sees_its_own_session(tmp_path):
    gateway = _Gateway(stop_after=[1])
    service, _events = _service(tmp_path, gateway)
    try:
        _say(service, "r1", THREE_STEPS, session_id="s1")

        record = _say(service, "r2", "继续", session_id="s2")

        assert record.state == "ANSWERED"
        assert len(gateway.programs) == 1
    finally:
        service.close()
