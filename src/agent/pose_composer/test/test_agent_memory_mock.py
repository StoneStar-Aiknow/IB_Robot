"""One Agent -> Composer -> SQLite consent/recall smoke test; transport is mocked.

Voice routing and individual consent/error branches belong to ibrobot_agent;
memory lifecycle, solver and physical motion have their own focused suites.
"""

import json
from types import SimpleNamespace

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.async_presentation import AsyncPresentationPort
from ibrobot_agent.contracts import (
    ExecutionResult,
    PlannerIdentity,
    PlannerOutcome,
    Presentation,
    TaskRef,
)
from pose_composer.composer import Composer
from pose_composer.memory import PoseMemory
from pose_composer.model import load


def test_explicit_save_then_recall_skips_both_models(tmp_path, agent_session):
    text = "左手向前伸直"
    description = {"kind": "pose", "left": {"upper": [1, 0, 0], "fore": [1, 0, 0]}, "right": {"keep": True}}
    calls = dict(planner=0, generator=0, motion=0)
    memory = PoseMemory(tmp_path / "poses.sqlite3", "mock-x2")

    def generate(_messages):
        calls["generator"] += 1
        return json.dumps(description)

    composer = Composer(generate, memory=memory)
    start, prepared = dict(load().rest), None

    def compose(target, *, session_id, utterance):
        nonlocal prepared
        prepared = composer.compose(target, start, session_id=session_id, utterance=utterance)
        return dict(
            success=True,
            kind="pose",
            composition_id=prepared.composition_id,
            description_json=json.dumps(prepared.description),
            from_memory=prepared.from_memory,
        )

    def lookup(target):
        record = memory.lookup(target, confirmed_only=True)
        return dict(hit=record is not None, pose_id=record.pose_id if record else "", poses=memory.planning_catalog())

    def feedback(session_id, verdict):
        pose_id, message = composer.feedback(session_id, verdict)
        return dict(success=True, pose_id=pose_id, message=message)

    def plan(request, context, _catalog, _cancel, **_kwargs):
        calls["planner"] += 1
        if request.text == "保存":
            assert context["pending_pose_save"]
            return PlannerOutcome(kind="conversation", user_message="确认保存", pose_feedback="accept")
        return PlannerOutcome(
            kind="workflow",
            user_message="好的",
            summary=text,
            steps=(CanonicalWorkflowStep(1, "free_pose", target_name=text),),
        )

    def execute(proposal, *, presentation_callback, submission_callback, **_kwargs):
        nonlocal start
        calls["motion"] += 1
        ref = TaskRef(proposal.request.request_id, "plan", "digest", "epoch", 1, "catalog", 1)
        presentation_callback(
            Presentation(ref, 1, proposal.outcome.steps, "direct_execute_async_presentation", 60, text)
        )
        submission_callback(ref, {})
        value = composer.take(prepared.text, prepared.composition_id)
        start = dict(value.composition.frames[-1])
        composer.succeeded(value)
        return ExecutionResult("succeeded", ref, "", "done", {})

    service = agent_session.create(
        planner=SimpleNamespace(identity=PlannerIdentity("mock", "mock", "json", "1", "1", "mock"), plan=plan),
        execution=SimpleNamespace(execute=execute),
        execution_enabled=True,
        allowed_skills={"free_pose"},
        presentation=AsyncPresentationPort(),
        pose_preview=SimpleNamespace(compose=compose),
        pose_memory=SimpleNamespace(lookup=lookup, feedback=feedback),
        robot_scope="mock_x2",
    )

    try:
        agent_session.send(service, text, "SUCCEEDED")
        agent_session.send(service, text, "SUCCEEDED")
        assert not lookup(text)["hit"]
        motions = calls["motion"]
        agent_session.send(service, "保存", "ANSWERED")
        assert lookup(text)["hit"] and calls["motion"] == motions
        before = calls.copy()
        agent_session.send(service, text, "SUCCEEDED")
        assert calls == {**before, "motion": before["motion"] + 1}
        assert prepared.from_memory
    finally:
        service.close()
        memory.close()
