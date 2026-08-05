from threading import Lock
from types import SimpleNamespace

import pytest

from embodied_common.dispatch_binding import delegated_executor_identity, fill_delegated_executor_identity, new_binding
from ibrobot_msgs.action import PickObject
from manipulation_execution.pick_executor_node import PickExecutorNode, PickFlowError


class _PrimitiveClient:
    def __init__(self):
        self.goals = []

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return object()


class _PrimitiveHarness:
    _run_primitive = PickExecutorNode._run_primitive

    def __init__(self, *, success: bool = False) -> None:
        self._primitive_client = _PrimitiveClient()
        self._dispatch_nonce = "delegated-nonce"
        self._dispatch_binding = new_binding(task_id="task")
        self._dispatch_binding.dispatch_nonce = self._dispatch_nonce
        self._dispatch_binding.expected_registry_epoch = "epoch-1"
        self._dispatch_binding.expected_registry_generation = 2
        self._dispatch_binding.expected_registry_digest = "digest-2"
        self._rpc_timeout = 1.0
        self._arm_joint_names = []
        self._primitive_handle = SimpleNamespace(accepted=True, get_result_async=lambda: object())
        self._action_result = SimpleNamespace(
            result=SimpleNamespace(success=success, error_code="MOTION_FAILED", message="motion failed")
        )
        self._wait_count = 0

    @staticmethod
    def _remaining(_deadline: float) -> float:
        return 1.0

    def _wait_future(self, _future, _goal_handle, _deadline, _timeout_sec, _label):
        self._wait_count += 1
        return self._primitive_handle if self._wait_count == 1 else self._action_result


@pytest.mark.parametrize("primitive_name", ["move_to_named_pose", "move_to_pose"])
def test_cancelable_pose_primitive_failure_remains_retryable(primitive_name: str) -> None:
    executor = _PrimitiveHarness()

    with pytest.raises(PickFlowError) as exc_info:
        executor._run_primitive(object(), 1.0, "task", primitive_name)

    assert exc_info.value.retryable is True


def test_move_to_configuration_failure_is_not_retryable() -> None:
    executor = _PrimitiveHarness()

    with pytest.raises(PickFlowError) as exc_info:
        executor._run_primitive(object(), 1.0, "task", "move_to_configuration")

    assert exc_info.value.code == "MOTION_FAILED"
    assert exc_info.value.retryable is False


def test_delegated_dispatch_nonce_is_forwarded_to_primitive() -> None:
    executor = _PrimitiveHarness(success=True)

    executor._run_primitive(object(), 1.0, "task", "move_to_named_pose")

    assert executor._primitive_client.goals[0].dispatch_binding.dispatch_nonce == "delegated-nonce"
    assert executor._primitive_client.goals[0].dispatch_binding.expected_registry_generation == 2


def test_pick_executor_rejects_identity_mismatch_and_requires_dispatch_nonce() -> None:
    executor = object.__new__(PickExecutorNode)
    executor._goal_lock = Lock()
    executor._goal_active = False
    executor._dispatch_nonce = ""
    executor._dispatch_binding = None
    executor._executor_identity = delegated_executor_identity(
        name="grasp_pipeline", endpoint_name="/manipulation/execute_pick"
    )
    goal = PickObject.Goal()
    goal.target_query = "banana"
    goal.dispatch_binding.dispatch_nonce = "nonce-1"

    assert executor._handle_goal(goal).name == "REJECT"
    fill_delegated_executor_identity(goal.expected_executor, executor._executor_identity)
    assert executor._handle_goal(goal).name == "ACCEPT"
    assert executor._dispatch_nonce == "nonce-1"
