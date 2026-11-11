"""The Agent starts a pick's perception once its validated plan names the target, before the Gateway."""

from __future__ import annotations

import time

import pytest

from embodied_common.workflow_contracts import CanonicalWorkflowStep
from ibrobot_agent.contracts import PlannerOutcome
from ibrobot_agent.pick_prefetch import first_pick
from ibrobot_agent.request_store import SQLiteRequestStore
from ibrobot_agent.service import AgentService

from .test_service import Events, FakeCatalog, FakePlanner, _request


def step(skill, target="an apple", container=""):
    return CanonicalWorkflowStep(schema_version=1, skill_name=skill, target_name=target, container_name=container)


@pytest.mark.parametrize(
    "steps,expected",
    [
        ((step("pick_object"),), ("an apple", "")),
        ((step("pick_object"), step("place_in_container", container="a black bowl")), ("an apple", "a black bowl")),
        # The place is of another object: the pick hands nothing over.
        ((step("pick_object"), step("place_in_container", "an orange", "a black bowl")), ("an apple", "")),
        # Not the first step: the robot moves before it, so a capture now would be refused.
        ((step("return_to_task_pose", "grasp_ready"), step("pick_object")), None),
        ((step("pick_object", target="  "),), None),
        ((), None),
    ],
)
def test_only_a_workflow_that_starts_with_a_pick_is_prefetched(steps, expected):
    assert first_pick(steps) == expected


def _planned(tmp_path, outcome, hook, events=None):
    events = events or Events()
    service = AgentService(
        planner=FakePlanner(outcome),
        catalog=FakeCatalog(),
        store=SQLiteRequestStore(tmp_path / "request.sqlite3"),
        event_sink=events,
        workflow_planned=hook,
    )
    try:
        assert service.send_message(_request()).accepted
        for _ in range(300):
            if any(event.event_type == "proposal_ready" for event in events.items) or events.ready.is_set():
                break
            time.sleep(0.01)
        return service._store.get_request(_request().to_key()), events
    finally:
        service.close()


WAVE = PlannerOutcome(
    kind="workflow", user_message="ok", summary="wave", steps=(CanonicalWorkflowStep(1, "wave_hello"),)
)


def test_the_validated_steps_reach_the_hook_before_the_proposal(tmp_path):
    events, seen = Events(), []

    def hook(steps):
        seen.append((tuple(s.skill_name for s in steps), [event.event_type for event in events.items]))

    _planned(tmp_path, WAVE, hook, events)
    assert len(seen) == 1 and seen[0][0] == ("wave_hello",)
    assert "proposal_ready" not in seen[0][1]  # before the proposal, so before the Gateway
    assert any(event.event_type == "proposal_ready" for event in events.items)


def test_a_failing_hook_does_not_stop_the_plan(tmp_path):
    def hook(steps):
        raise RuntimeError("prefetch service gone")

    _, events = _planned(tmp_path, WAVE, hook)
    assert any(event.event_type == "proposal_ready" for event in events.items)


def test_a_conversation_does_not_reach_the_hook(tmp_path):
    seen = []
    _planned(tmp_path, PlannerOutcome(kind="conversation", user_message="你好"), seen.append)
    assert seen == []
