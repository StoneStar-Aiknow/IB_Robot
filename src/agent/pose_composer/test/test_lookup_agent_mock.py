"""Fixed-library recall reaches the real Agent without a writable memory or model call."""

from types import SimpleNamespace

import pytest

from ibrobot_agent.contracts import PlannerIdentity, PlannerOutcome
from ibrobot_agent.node import _RosPosePreview
from ibrobot_msgs.srv import LookupArmPose
from pose_composer.fixed_library import FixedComposer
from pose_composer.node import PoseComposerNode


@pytest.fixture(scope="session", autouse=True)
def configured():
    # This test uses only a read-only alias library, never the generated backend.
    return None


@pytest.mark.parametrize(
    "enabled,text,calls",
    [(True, "双手举过头顶", 0), (True, "双臂放下", 0), (False, "双手举过头顶", 1), (True, "未知动作", 1)],
)
def test_read_only_seed_lookup_controls_agent_model_shortcut(agent_session, enabled, text, calls):
    composer = FixedComposer(
        {
            "poses": {
                "arms_up": {"phrases": ["双手举过头顶"], "user_message": "举臂", "description": {}},
                "arms_down": {
                    "phrases": ["双臂放下"],
                    "user_message": "回到 HOME",
                    "description": {
                        "kind": "pose",
                        "left": {"task_pose": "grasp_ready"},
                        "right": {"task_pose": "grasp_ready"},
                    },
                },
            }
        }
    )
    server = object.__new__(PoseComposerNode)
    server._composer = composer
    server.get_parameter = lambda _name: SimpleNamespace(value=enabled)
    port = object.__new__(_RosPosePreview)
    port._lookup = object() if enabled else None
    port._feedback = None
    port._call = lambda _client, request, _timeout: server._on_lookup(request, LookupArmPose.Response())
    assert composer.memory is None
    assert port.has_pose_services is enabled
    assert not port.feedback("session", "good")["success"]

    class Planner:
        identity = PlannerIdentity("mock", "mock", "json", "1", "1", "mock")
        count = 0

        def plan(self, *_args, **_kwargs):
            self.count += 1
            return PlannerOutcome(kind="conversation", user_message="模型被调用")

    planner = Planner()
    service = agent_session.create(
        planner=planner,
        pose_memory=port if port.has_pose_services else None,
    )
    try:
        request = agent_session.send(service, text, "ANSWERED", "so101_single_arm")
        record = service.get_request(request.to_key())
        assert record.state == "ANSWERED"
        assert not record.may_have_submitted
        assert planner.count == calls
        timing = agent_session.events[-1].detail["planning_timing"]
        assert timing["agent_catalog"]["status"] == "executed"
        assert timing["agent_model"]["status"] == ("executed" if calls else "skipped")
        assert timing["agent_lookup"]["status"] == ("executed" if enabled else "skipped")
        if calls == 0:
            assert (
                agent_session.store.latest_event_detail(request.to_key(), "proposal_snapshot")["outcome"]["steps"][0][
                    "target_name"
                ]
                == text
            )
    finally:
        service.close()


def test_disabled_lookup_endpoint_returns_no_hit_or_catalog():
    server = object.__new__(PoseComposerNode)
    server.get_parameter = lambda _name: SimpleNamespace(value=False)
    # No composer is supplied: the disabled endpoint must not read either source.
    result = server._on_lookup(LookupArmPose.Request(text="双手举过头顶"), LookupArmPose.Response())
    assert not result.hit and result.catalog_json == "[]"
