import threading
import time
from types import SimpleNamespace

import pytest
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
from task_dispatch.task_executor_node import TaskExecutorNode, gripper_trajectory_duration

# SO-101: open 1.0, closed 0.15 (resolved by the robot_config launch builder).
_STROKE = 0.85


class _Done:
    def __init__(self, value) -> None:
        self._value = value

    def done(self) -> bool:
        return True

    def result(self):
        return self._value


class _RecordingActionClient:
    def __init__(self) -> None:
        self.goals = []

    def server_is_ready(self) -> bool:
        return True

    def send_goal_async(self, goal, feedback_callback=None):
        self.goals.append(goal)
        result = SimpleNamespace(result=SimpleNamespace(error_code=FollowJointTrajectory.Result.SUCCESSFUL))
        return _Done(SimpleNamespace(accepted=True, get_result_async=lambda: _Done(result)))


def _executor(position: float, *, age_s: float = 0.0) -> TaskExecutorNode:
    executor = TaskExecutorNode.__new__(TaskExecutorNode)
    executor._skip_redundant_gripper_open = False
    executor._gripper_open_position = 1.0
    executor._gripper_position_tolerance = 0.05
    executor._joint_state_max_age_s = 0.25
    executor._gripper_joint = "6"
    executor._gripper_limits = {"min": 0.0, "max": 2.0}
    executor._gripper_stroke = _STROKE
    executor._cancel_requested = False
    executor._joint_state_lock = threading.Lock()
    executor._latest_joint_state = JointState(name=["6"], position=[position])
    executor._latest_joint_state_received_at = time.monotonic() - age_s
    executor._gripper_action_client = _RecordingActionClient()
    executor._gripper_feedback_cb = lambda _message: None
    executor.get_logger = lambda: SimpleNamespace(info=lambda _message: None)
    return executor


def _sent_duration(executor: TaskExecutorNode) -> float:
    (goal,) = executor._gripper_action_client.goals
    stamp = goal.trajectory.points[0].time_from_start
    return stamp.sec + stamp.nanosec / 1e9


def test_duration_follows_distance_and_never_beats_the_previous_open_close_speed() -> None:
    # A full open or close keeps the previous fixed 1 s.
    assert gripper_trajectory_duration(1.0, 0.15, _STROKE) == pytest.approx(1.0)
    assert gripper_trajectory_duration(0.15, 1.0, _STROKE) == pytest.approx(1.0)
    assert gripper_trajectory_duration(0.575, 0.15, _STROKE) == pytest.approx(0.5)
    # Longer than a configured stroke is capped, already-there uses the floor.
    assert gripper_trajectory_duration(2.0, 0.0, _STROKE) == pytest.approx(1.0)
    assert gripper_trajectory_duration(0.15, 0.1534, _STROKE) == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("current", "stroke"),
    [(None, _STROKE), (float("nan"), _STROKE), (0.5, 0.0), (0.5, float("nan"))],
    ids=["no-feedback", "nan-feedback", "stroke-unknown", "nan-stroke"],
)
def test_duration_keeps_the_fixed_full_stroke_time_without_usable_inputs(current, stroke) -> None:
    assert gripper_trajectory_duration(1.0, current, stroke) == pytest.approx(1.0)


def test_command_at_the_current_position_is_still_sent_with_the_shortest_duration() -> None:
    # SO-101 rests at 0.1534 for the 0.15 closed target: no skip decision, the
    # target is commanded again and only the trajectory is short.
    executor = _executor(position=0.1534)

    assert executor._exec_gripper(SimpleNamespace(gripper_position=0.15)) == (True, "ok")
    (goal,) = executor._gripper_action_client.goals
    assert list(goal.trajectory.points[0].positions) == [0.15]
    assert _sent_duration(executor) == pytest.approx(0.1)


def test_full_travel_keeps_the_previous_duration_and_stale_feedback_falls_back_to_it() -> None:
    moving = _executor(position=0.15)
    assert moving._exec_gripper(SimpleNamespace(gripper_position=1.0)) == (True, "ok")
    assert _sent_duration(moving) == pytest.approx(1.0)

    stale = _executor(position=0.1534, age_s=1.0)
    assert stale._exec_gripper(SimpleNamespace(gripper_position=0.15)) == (True, "ok")
    assert _sent_duration(stale) == pytest.approx(1.0)


def test_duration_just_below_a_whole_second_is_encoded_without_nanosecond_overflow() -> None:
    executor = _executor(position=0.15)
    executor._gripper_stroke = 0.85
    executor._exec_gripper(SimpleNamespace(gripper_position=0.15 + 0.85 * 0.9999999999))
    stamp = executor._gripper_action_client.goals[0].trajectory.points[0].time_from_start
    assert stamp.nanosec < 1_000_000_000
    assert stamp.sec + stamp.nanosec / 1e9 == pytest.approx(1.0)


def test_unknown_stroke_keeps_the_previous_fixed_duration_for_every_command() -> None:
    executor = _executor(position=0.1534)
    executor._gripper_stroke = 0.0
    assert executor._exec_gripper(SimpleNamespace(gripper_position=0.15)) == (True, "ok")
    assert _sent_duration(executor) == pytest.approx(1.0)
