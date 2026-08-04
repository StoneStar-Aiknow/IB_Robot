from types import SimpleNamespace

import pytest

from manipulation_execution.pick_executor_node import PickExecutorNode, PickFlowError


class _PrimitiveClient:
    def __init__(self) -> None:
        self.goal = None

    def send_goal_async(self, goal):
        self.goal = goal
        return object()


class _PrimitiveHarness:
    _run_primitive = PickExecutorNode._run_primitive
    _execution_token = staticmethod(PickExecutorNode._execution_token)

    def __init__(self, *, success: bool = False) -> None:
        self._primitive_client = _PrimitiveClient()
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


def test_pick_goal_uuid_is_forwarded_as_internal_execution_token() -> None:
    executor = _PrimitiveHarness(success=True)
    goal_handle = SimpleNamespace(goal_id=SimpleNamespace(uuid=[7] * 16))

    executor._run_primitive(goal_handle, 1.0, "task", "open_gripper")

    assert executor._primitive_client.goal.execution_token == bytes([7] * 16).hex()
