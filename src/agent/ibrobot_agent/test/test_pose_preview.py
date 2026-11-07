"""Arm poses described in words are composed before the plan is presented."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import PlannerOutcome
from ibrobot_agent.node import _RosPosePreview
from ibrobot_agent.request_store import SQLiteRequestStore
from ibrobot_agent.service import AgentService

from .test_service import Events, FakeCatalog, FakePlanner, _request


def test_disabling_lookup_keeps_writable_feedback_independent():
    port = object.__new__(_RosPosePreview)
    port._lookup, port._feedback = None, object()
    calls = []

    def call(client, request, _timeout):
        calls.append((client, request.verdict))
        return SimpleNamespace(success=True, pose_id="saved", message="confirmed")

    port._call = call
    assert port.has_pose_services
    assert port.lookup("双手举过头顶") == {"hit": False, "poses": []}
    assert not calls
    assert port.feedback("session", "good")["pose_id"] == "saved"
    assert calls == [(port._feedback, "good")]


class FreePoseCatalog(FakeCatalog):
    def get_catalog(self, status):
        catalog = super().get_catalog(status)
        catalog["skills"] = [
            *catalog["skills"],
            {"name": "free_pose", "planner_visible": True, "semantic_level": "skill"},
        ]
        return catalog


class FakePreview:
    def __init__(self, answer):
        self.answer = answer
        self.texts = []
        self.sessions = []

    def compose(self, text, *, session_id="", utterance=""):
        self.texts.append(text)
        self.sessions.append((session_id, utterance))
        return self.answer


def _pose_outcome(*skills):
    steps = tuple(
        CanonicalWorkflowStep(1, name, target_name="双手向前伸直" if name == "free_pose" else "") for name in skills
    )
    return PlannerOutcome(kind="workflow", user_message="好的", summary="pose", steps=steps)


def _run(tmp_path, outcome, preview):
    events = Events()
    service = AgentService(
        planner=FakePlanner(outcome),
        catalog=FreePoseCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=events,
        pose_preview=preview,
    )
    request = _request(text="双手向前伸直")
    try:
        assert service.send_message(request).accepted
        for _ in range(300):
            state = service.get_request(request.to_key()).state
            if state in {"ANSWERED", "SUCCEEDED", "FAILED", "PROPOSAL_READY"} or any(
                e.event_type in {"proposal_ready", "terminal"} for e in events.items
            ):
                break
            time.sleep(0.01)
        time.sleep(0.05)
        return service.get_request(request.to_key()).state, events.items
    finally:
        service.close()


COMPOSED = {
    "success": True, "error_code": "", "message": "composed pose", "kind": "pose", "user_message": "好的，双手向前伸直",
    "composition_id": "abc", "description_json": '{"kind": "pose"}', "duration_sec": 4.1, "approximate": False,
}  # fmt: skip


def test_the_pose_is_composed_and_shown_before_the_proposal(tmp_path):
    preview = FakePreview(COMPOSED)
    _, events = _run(tmp_path, _pose_outcome("free_pose"), preview)
    kinds = [e.event_type for e in events]
    assert preview.texts == ["双手向前伸直"]
    assert kinds.index("pose_preview") < kinds.index("proposal_ready")
    shown = next(e for e in events if e.event_type == "pose_preview")
    assert shown.user_message == "好的，双手向前伸直"
    assert shown.detail["composition_id"] == "abc" and shown.detail["description"] == {"kind": "pose"}


def test_a_pose_the_composer_refuses_is_answered_and_never_proposed(tmp_path):
    refused = dict(COMPOSED, kind="rejected", user_message="这个动作会打到自己，我不做")
    state, events = _run(tmp_path, _pose_outcome("free_pose"), FakePreview(refused))
    assert state == "ANSWERED"
    assert "proposal_ready" not in [e.event_type for e in events]
    assert any(e.event_type == "pose_refused" and "打到自己" in e.user_message for e in events)


def test_a_composition_failure_is_answered_with_its_reason(tmp_path):
    failed = {"success": False, "error_code": "POSE_NO_SAFE_PATH", "message": "no collision-free path"}
    state, events = _run(tmp_path, _pose_outcome("free_pose"), FakePreview(failed))
    assert state == "ANSWERED"
    refused = next(e for e in events if e.event_type == "pose_refused")
    assert "no collision-free path" in refused.user_message


def test_a_plan_that_does_not_start_with_a_pose_is_not_previewed(tmp_path):
    preview = FakePreview(COMPOSED)
    _, events = _run(tmp_path, _pose_outcome("wave_hello", "free_pose"), preview)
    assert preview.texts == []
    assert "proposal_ready" in [e.event_type for e in events]


# --- the pose memory -------------------------------------------------------


class FakeMemory:
    def __init__(self, hit=False, pending=False):
        self.hit = hit
        self.pending = pending
        self.lookups = []
        self.verdicts = []

    def lookup(self, text):
        self.lookups.append(text)
        return {"hit": self.hit, "pose_id": "p1", "state": "active", "user_message": f"好的，{text}"}

    def feedback(self, session_id, verdict):
        self.verdicts.append((session_id, verdict))
        return {"success": True, "pose_id": "p1" if self.pending else "", "message": "confirmed"}


def _run_memory(tmp_path, text, memory, planner_outcome=None):
    events = Events()
    planner = FakePlanner(planner_outcome or _pose_outcome("free_pose"))
    service = AgentService(
        planner=planner,
        catalog=FreePoseCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=events,
        pose_preview=FakePreview(COMPOSED),
        pose_memory=memory,
    )
    request = _request(text=text)
    try:
        assert service.send_message(request).accepted
        for _ in range(300):
            if any(e.event_type in {"proposal_ready", "terminal"} or e.state == "ANSWERED" for e in events.items):
                break
            time.sleep(0.01)
        time.sleep(0.05)
        return planner, events.items, service.get_request(request.to_key()).state
    finally:
        service.close()


def test_a_confirmed_remembered_pose_skips_the_planner(tmp_path):
    memory = FakeMemory(hit=True)
    planner, events, _ = _run_memory(tmp_path, "双手向前伸直", memory)
    assert planner.calls == 0
    proposal = next(e for e in events if e.event_type == "proposal_ready")
    assert "remembered" in proposal.user_message
    assert any(e.event_type == "pose_preview" for e in events)


@pytest.mark.parametrize("text", ["不要双手向前伸直", "如果双手向前伸直会怎样"])
def test_negated_or_hypothetical_words_never_recall_a_pose(tmp_path, text):
    memory = FakeMemory(hit=True)
    planner, _, _ = _run_memory(tmp_path, text, memory)
    assert memory.lookups == []
    assert planner.calls == 1


@pytest.mark.parametrize("decision,verdict", [("accept", "good"), ("decline", "next_task"), ("clarify", "status")])
def test_model_confirmation_is_applied_without_second_planning(tmp_path, decision, verdict):
    memory = FakeMemory(pending=True)
    outcome = PlannerOutcome(kind="conversation", user_message="模型回复", pose_feedback=decision)
    planner, events, state = _run_memory(tmp_path, "自然口语确认", memory, outcome)
    assert state == "ANSWERED" and planner.calls == 1
    assert memory.verdicts[-1][1] == verdict
    assert not any(e.event_type == "proposal_ready" for e in events)


@pytest.mark.parametrize("pending,decision", [(False, "accept"), (True, None)])
def test_missing_confirmation_never_saves(tmp_path, pending, decision):
    memory = FakeMemory(pending=pending)
    outcome = PlannerOutcome(kind="conversation", user_message="已经保存了", pose_feedback=decision)
    planner, events, state = _run_memory(tmp_path, "可以保存", memory, outcome)
    assert state == "ANSWERED" and planner.calls == 1
    assert all(v not in ("good", "next_task") for _, v in memory.verdicts)
    assert not any(e.user_message == "已经保存了" for e in events)


def test_new_task_is_explicitly_classified_before_cancelling_offer(tmp_path):
    memory = FakeMemory(pending=True)
    outcome = _pose_outcome("free_pose")
    from dataclasses import replace

    outcome = replace(outcome, pose_feedback="new_task")
    planner, events, _ = _run_memory(tmp_path, "双手向前伸直", memory, outcome)
    assert planner.calls == 1 and memory.lookups == []
    assert memory.verdicts[-1][1] == "next_task"
    assert any(e.event_type == "proposal_ready" for e in events)


@pytest.mark.parametrize("current_id", ["", "changed"])
def test_expired_or_replaced_pending_pose_is_not_saved(tmp_path, current_id):
    class ChangedMemory(FakeMemory):
        def feedback(self, session_id, verdict):
            result = super().feedback(session_id, verdict)
            if verdict == "status" and len(self.verdicts) > 1:
                result["pose_id"] = current_id
            return result

    memory = ChangedMemory(pending=True)
    outcome = PlannerOutcome(kind="conversation", user_message="已保存", pose_feedback="accept")
    planner, events, state = _run_memory(tmp_path, "可以保存", memory, outcome)
    assert planner.calls == 1 and state == "ANSWERED"
    assert all(v == "status" for _, v in memory.verdicts)
    assert any("已过期或动作已改变" in e.user_message for e in events)


def test_model_failure_preserves_the_pending_offer(tmp_path, monkeypatch):
    def fail(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("mock model unavailable")

    monkeypatch.setattr(FakePlanner, "plan", fail)
    memory = FakeMemory(pending=True)
    planner, events, state = _run_memory(tmp_path, "动作到位，可以保存", memory)
    assert state == "ANSWERED" and planner.calls == 1
    assert [v for _, v in memory.verdicts] == ["status"]
    assert any("暂时没能确认" in e.user_message for e in events)


def test_workflow_without_a_pending_save_decision_does_not_execute(tmp_path):
    memory = FakeMemory(hit=True, pending=True)
    planner, events, state = _run_memory(tmp_path, "可以保存", memory)
    assert state == "ANSWERED" and planner.calls == 1 and memory.lookups == []
    assert [v for _, v in memory.verdicts] == ["status"]
    assert not any(e.event_type == "proposal_ready" for e in events)


def test_save_and_new_motion_use_one_planning_result(tmp_path):
    from dataclasses import replace

    memory = FakeMemory(pending=True)
    outcome = replace(_pose_outcome("free_pose"), pose_feedback="accept")
    planner, events, _ = _run_memory(tmp_path, "保存然后伸手", memory, outcome)
    assert planner.calls == 1
    assert memory.verdicts[-1][1] == "good"
    assert any(e.event_type == "reply_preview" and "记住了" in e.user_message for e in events)
    assert any(e.event_type == "proposal_ready" for e in events)


def test_failed_storage_never_reports_success_or_executes_followup(tmp_path):
    from dataclasses import replace

    class RefusingMemory(FakeMemory):
        def feedback(self, session_id, verdict):
            result = super().feedback(session_id, verdict)
            if verdict == "good":
                result.update(success=False, message="pose changed since execution")
            return result

    memory = RefusingMemory(pending=True)
    outcome = replace(_pose_outcome("free_pose"), pose_feedback="accept")
    _, events, state = _run_memory(tmp_path, "保存然后伸手", memory, outcome)
    assert state == "ANSWERED"
    assert any("保存未完成" in e.user_message for e in events)
    assert not any(e.event_type == "proposal_ready" for e in events)


def test_the_preview_carries_the_conversation_and_the_users_words(tmp_path):
    preview = FakePreview(COMPOSED)
    _run(tmp_path, _pose_outcome("free_pose"), preview)
    session, utterance = preview.sessions[0]
    assert session == "test/p1/s1"
    assert utterance == "双手向前伸直"
