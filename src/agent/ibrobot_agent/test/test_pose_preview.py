"""Arm poses described in words are composed before the plan is presented."""

from __future__ import annotations

import time

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import PlannerOutcome
from ibrobot_agent.request_store import SQLiteRequestStore
from ibrobot_agent.service import AgentService

from .test_service import Events, FakeCatalog, FakePlanner, _request


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

    def compose(self, text):
        self.texts.append(text)
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
