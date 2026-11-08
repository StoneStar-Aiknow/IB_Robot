"""PlaceObject transfers custody over ROS and preserves the Agent's queries."""

import json
import threading
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from test_grasp_pipeline_node import goal, wait
from test_place_pipeline import fixture

from embodied_common.dispatch_binding import fill_delegated_executor_identity
from ibrobot_msgs.action import PlaceObject
from manipulation_execution.held_custody import HeldCustodyServer
from manipulation_execution.place_pipeline_node import PlacePipelineNode


def make_place_backends(**_kwargs):
    return fixture()[1]


def test_place_claims_and_returns_custody_through_the_action():
    rclpy.init()
    action = "/test/place_pipeline/execute_place"
    config = {"backend": f"{__name__}:make_place_backends", "pipeline": {"max_attempts": 1}}
    node = PlacePipelineNode(
        parameter_overrides=[
            Parameter("action_name", value=action),
            Parameter("placement_execution_json", value=json.dumps(config)),
        ]
    )
    owner = Node("place_test_custody_owner")
    custody = HeldCustodyServer(owner)
    held, _ = fixture()
    custody.custody.accept(held, recover=lambda: False, release=lambda: None, holding=lambda: "", root_task_id="task-1")
    client = ActionClient(owner, PlaceObject, action)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.add_node(owner)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    queries = []
    propose = node.backends.proposer.propose

    def record(observation, query, held, **kwargs):
        queries.append(query)
        return propose(observation, query, held, **kwargs)

    node.backends.proposer.propose = record
    try:
        assert client.wait_for_server(timeout_sec=5.0)
        deadline = time.monotonic() + 5.0
        while node.custody.latest is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert node.custody.latest is not None
        pick = goal(config, now_s=node.get_clock().now().nanoseconds * 1e-9)
        pick.dispatch_binding.dispatch_nonce = "place-nonce"
        request = PlaceObject.Goal(
            target_query=held.target_query,
            container_query="the stainless steel tray on the left",
            timeout_sec=30.0,
            dispatch_binding=pick.dispatch_binding,
        )
        fill_delegated_executor_identity(request.expected_executor, node.identity)
        handle = wait(client.send_goal_async(request))
        assert handle.accepted
        result = wait(handle.get_result_async()).result
        assert result.success and result.place_succeeded
        assert result.release_status == result.RELEASE_RELEASED
        assert result.verification_status == result.VERIFICATION_SUCCESS
        assert result.execution_detail.phase == "verify_place"
        assert result.execution_detail.phase_state == "SUCCEEDED"
        assert custody.custody.state == "released"
        assert queries == [request.container_query]
        assert not node.active

        # The prior release cannot be claimed for a second placement.
        repeated = wait(client.send_goal_async(request))
        assert repeated.accepted
        assert not wait(repeated.get_result_async()).result.success
    finally:
        executor.shutdown(timeout_sec=5.0)
        thread.join(5.0)
        assert not thread.is_alive()
        client.destroy()
        node.destroy_node()
        owner.destroy_node()
        rclpy.shutdown()


def test_custody_return_failure_preserves_typed_release_and_verification():
    from types import SimpleNamespace

    from rclpy.serialization import deserialize_message, serialize_message

    from embodied_common.dispatch_binding import delegated_executor_identity
    from embodied_common.execution_detail import ExecutionDetail
    from manipulation_execution.grasp_pipeline.contracts import GraspFailure
    from manipulation_execution.grasp_pipeline.place_pipeline import PlaceOutcome

    held, _ = fixture()

    def failed_return(*_args):
        raise GraspFailure("CUSTODY_UNAVAILABLE", "custody service unavailable")

    node = object.__new__(PlacePipelineNode)
    node.custody = SimpleNamespace(claim=lambda *_args, **_kwargs: held, returned=failed_return)
    node.pipeline = SimpleNamespace(
        run=lambda *_args, **_kwargs: PlaceOutcome(
            success=True,
            release_status="released",
            custody_outcome="released",
            verification="success",
            execution_detail=ExecutionDetail(
                2,
                "verify_place",
                "SUCCEEDED",
                "",
                "",
                "NOT_NEEDED",
                0,
                "NONE",
                "RELEASED",
                "SUCCESS",
                True,
            ),
        )
    )
    node.identity = delegated_executor_identity(
        name="placement_pipeline",
        endpoint_name="/test/place",
        configuration={},
    )
    node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1_000_000_000))
    node.get_logger = lambda: SimpleNamespace(info=lambda _message: None)
    request = PlaceObject.Goal(target_query="apple", container_query="bowl", timeout_sec=30.0)
    request.dispatch_binding.task_budget.deadline.sec = 31
    handle = SimpleNamespace(
        request=request,
        is_cancel_requested=False,
        publish_feedback=lambda _feedback: None,
        abort=lambda: None,
        succeed=lambda: None,
        canceled=lambda: None,
    )
    result = node._execute(handle)
    restored = deserialize_message(serialize_message(result), PlaceObject.Result)
    assert not restored.success and not restored.place_succeeded
    assert restored.release_status == PlaceObject.Result.RELEASE_RELEASED
    assert restored.verification_status == PlaceObject.Result.VERIFICATION_SUCCESS
    assert restored.execution_detail.release_status == "RELEASED"
    assert restored.execution_detail.verification_status == "SUCCESS"
    assert restored.execution_detail.failed_phase == "return_custody"
    assert not restored.execution_detail.place_succeeded
