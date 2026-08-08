import threading
import time
from types import SimpleNamespace

from sensor_msgs.msg import JointState

from task_dispatch.task_executor_node import TaskExecutorNode


class _Logger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, message: str) -> None:
        self.messages.append(message)


class _UnexpectedActionClient:
    def server_is_ready(self):
        raise AssertionError("gripper action must not be queried for a confirmed no-op")


def _executor(position: float = 1.0, *, age_s: float = 0.0, joint_name: str = "6") -> TaskExecutorNode:
    executor = TaskExecutorNode.__new__(TaskExecutorNode)
    executor._skip_redundant_gripper_open = True
    executor._gripper_open_position = 1.0
    executor._gripper_position_tolerance = 0.05
    executor._joint_state_max_age_s = 0.25
    executor._gripper_joint = "6"
    executor._joint_state_lock = threading.Lock()
    executor._latest_joint_state = JointState(name=[joint_name], position=[position])
    executor._latest_joint_state_received_at = time.monotonic() - age_s
    executor._gripper_action_client = _UnexpectedActionClient()
    logger = _Logger()
    executor.get_logger = lambda: logger
    return executor


def test_fresh_open_feedback_skips_gripper_trajectory() -> None:
    executor = _executor(position=0.98)

    result = executor._exec_gripper(SimpleNamespace(gripper_position=1.0))

    assert result == (True, "already at open target")


def test_stale_feedback_does_not_skip_open() -> None:
    executor = _executor(position=1.0, age_s=0.5)

    assert executor._is_redundant_gripper_open(1.0) is False


def test_missing_gripper_joint_does_not_skip_open() -> None:
    executor = _executor(position=1.0, joint_name="5")

    assert executor._is_redundant_gripper_open(1.0) is False


def test_close_command_is_never_skipped() -> None:
    executor = _executor(position=0.0)

    assert executor._is_redundant_gripper_open(0.0) is False


def test_out_of_tolerance_open_feedback_does_not_skip() -> None:
    executor = _executor(position=0.90)

    assert executor._is_redundant_gripper_open(1.0) is False
