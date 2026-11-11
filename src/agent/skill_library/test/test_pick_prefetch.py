"""The Gateway asks the pick executor to perceive during the preparation, with the place's container."""

from types import SimpleNamespace

import pytest

from skill_library.skill_executor_node import SkillExecutorNode


class _Client:
    def __init__(self, ready=True):
        self.ready, self.requests = ready, []

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        self.requests.append(request)


def _step(skill, target="an apple", container="a black bowl"):
    return SimpleNamespace(skill_name=skill, target_name=target, container_name=container)


def _gateway(client, steps, placement=True):
    node = object.__new__(SkillExecutorNode)
    node._pick_prefetch_client = client
    node._active_workflow = SimpleNamespace(workflow_steps=steps)
    node._placement_execution = {"backend": "x:y"} if placement else {}
    node.get_logger = lambda: SimpleNamespace(warning=lambda message: None)
    return node


def _goal(index=0, target="an apple"):
    from ibrobot_msgs.msg import DispatchBinding

    binding = DispatchBinding(schema_version=1, workflow_step_index=index, root_task_id="root")
    return SimpleNamespace(target_name=target, dispatch_binding=binding)


@pytest.mark.parametrize(
    "steps,placement,container",
    [
        ([_step("pick_object"), _step("place_in_container")], True, "a black bowl"),
        ([_step("pick_object")], True, ""),
        ([_step("pick_object"), _step("place_in_container")], False, ""),
    ],
)
def test_the_prefetch_names_the_target_and_only_a_following_places_container(steps, placement, container):
    client = _Client()
    _gateway(client, steps, placement)._request_pick_prefetch(_goal())
    (request,) = client.requests
    assert request.target_query == "an apple" and request.container_query == container
    assert request.dispatch_binding.root_task_id == "root"


@pytest.mark.parametrize("client", [None, _Client(ready=False)])
def test_without_a_ready_prefetch_service_nothing_is_asked(client):
    node = _gateway(client, [_step("pick_object")])
    node._request_pick_prefetch(_goal())
    assert client is None or client.requests == []


def test_a_failing_request_never_stops_the_pick():
    class Broken(_Client):
        def call_async(self, request):
            raise RuntimeError("middleware down")

    warnings = []
    node = _gateway(Broken(), [_step("pick_object")])
    node.get_logger = lambda: SimpleNamespace(warning=warnings.append)
    node._request_pick_prefetch(_goal())
    assert warnings and "middleware down" in warnings[0]


def test_the_prefetch_is_requested_before_the_preparation():
    order = []

    class Stop(Exception):
        pass

    node = _gateway(_Client(), [_step("pick_object")])
    node._task_poses = {
        "skill_entry_poses": {"pick_object": "grasp_ready"},
        "poses": {"grasp_ready": {"timeout_sec": 30}},
    }
    node._ensure_skill_capabilities = lambda *_: (True, "")
    node._request_pick_prefetch = lambda goal: order.append("prefetch")

    def prepare(*args, **kwargs):
        order.append("prepare")
        raise Stop

    node._execute_pose_composer_skill = prepare
    goal = SimpleNamespace(
        request=SimpleNamespace(skill_name="pick_object", timeout_sec=0.0),
        is_cancel_requested=False,
        publish_feedback=lambda feedback: None,
    )
    with pytest.raises(Stop):
        node._execute_prepared_pick_skill(goal, {})
    assert order == ["prefetch", "prepare"]
