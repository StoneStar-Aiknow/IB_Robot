from types import SimpleNamespace

import rclpy

from skill_library import skill_executor_node
from skill_library.resolver import PrimitiveSpec
from skill_library.skill_executor_node import SkillExecutorNode


class _Future:
    def __init__(self, *, done: bool, result=None) -> None:
        self._done = done
        self._result = result
        self._callbacks = []

    def done(self) -> bool:
        return self._done

    def result(self):
        return self._result

    def add_done_callback(self, callback) -> None:
        if self._done:
            callback(self)
            return
        self._callbacks.append(callback)

    def set_result(self, result) -> None:
        self._result = result
        self._done = True
        for callback in self._callbacks:
            callback(self)


class _ChildGoalHandle:
    accepted = True

    def __init__(self, events=None, *, complete_result_on_cancel: bool = True) -> None:
        self.events = events if events is not None else []
        self.complete_result_on_cancel = complete_result_on_cancel
        self.cancel_count = 0
        self.result_future = _Future(done=False)

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_count += 1
        self.events.append("child_cancel")
        return _Future(done=True)

    def complete_cancel_cleanup(self) -> None:
        if self.complete_result_on_cancel and self.cancel_count and not self.result_future.done():
            self.events.append("child_terminal")
            self.result_future.set_result(SimpleNamespace(result=SimpleNamespace(error_code="")))


class _ParentGoalHandle:
    def __init__(self, events=None) -> None:
        self.events = events if events is not None else []
        self.request = SimpleNamespace(
            task_id="task-1",
            skill_name="test_skill",
            target_name="",
            place_name="",
            motion_direction="",
            motion_distance=0.0,
            timeout_sec=0.01,
        )
        self.cancel_checks = 0
        self.canceled_count = 0
        self.abort_count = 0

    @property
    def is_cancel_requested(self) -> bool:
        self.cancel_checks += 1
        return self.cancel_checks >= 2

    def publish_feedback(self, _feedback) -> None:
        pass

    def canceled(self) -> None:
        self.canceled_count += 1
        self.events.append("parent_cancelled")

    def abort(self) -> None:
        self.abort_count += 1
        self.events.append("parent_aborted")


class _PrimitiveGoalHandle:
    def __init__(self) -> None:
        self.request = SimpleNamespace(
            primitive_name="move_to_joint_positions",
            pose_name="",
            relative_dx=0.0,
            relative_dy=0.0,
            relative_dz=0.0,
            gripper_position=0.0,
            joint_names=["joint_1"],
            joint_positions=[0.1],
            joint_waypoints=[],
            joint_waypoint_count=0,
            primitive_duration_sec=0.4,
            waypoint_duration_sec=0.0,
            timeout_sec=1.0,
            task_id="task-1",
        )
        self.canceled_count = 0
        self.abort_count = 0

    @property
    def is_cancel_requested(self) -> bool:
        return True

    def publish_feedback(self, _feedback) -> None:
        pass

    def canceled(self) -> None:
        self.canceled_count += 1

    def abort(self) -> None:
        self.abort_count += 1


def _make_skill_node(send_goal_future) -> SkillExecutorNode:
    node = object.__new__(SkillExecutorNode)
    node._validate_skill = lambda *_args, **_kwargs: (True, "")
    node._current_joint_positions = lambda: []
    node._named_targets = {}
    node._gripper_open = 1.0
    node._gripper_closed = 0.0
    node._skill_templates = {}
    node._relative_motion_direction_mapping = {}
    node._arm_joint_names = []
    node._rpc_timeout = 0.01
    node._debug = False
    node._primitive_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal: send_goal_future,
    )
    return node


def test_wait_for_future_runs_cancel_callback_once(monkeypatch):
    cancelled = []
    monkeypatch.setattr(rclpy, "ok", lambda: True)

    completed = SkillExecutorNode._wait_for_future(
        _Future(done=False),
        timeout_sec=1.0,
        cancel_requested=lambda: True,
        cancel_callback=lambda: cancelled.append(True),
    )

    assert completed is False
    assert cancelled == [True]


def test_execute_skill_waits_for_child_terminal_before_parent_cancel(monkeypatch):
    events = []
    child_goal_handle = _ChildGoalHandle(events)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(skill_executor_node.time, "sleep", lambda _seconds: child_goal_handle.complete_cancel_cleanup())
    node = _make_skill_node(_Future(done=True, result=child_goal_handle))
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _ParentGoalHandle(events)

    result = node._execute_skill(parent_goal_handle)

    assert result.success is False
    assert result.error_code == "SKILL_CANCELLED"
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done()
    assert parent_goal_handle.canceled_count == 1
    assert parent_goal_handle.abort_count == 0
    assert events.index("child_terminal") < events.index("parent_cancelled")


def test_execute_skill_drains_child_accepted_after_parent_cancel(monkeypatch):
    events = []
    child_goal_handle = _ChildGoalHandle(events)
    send_goal_future = _Future(done=False)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = _make_skill_node(send_goal_future)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    def advance_cancel_cleanup(_seconds) -> None:
        if not send_goal_future.done():
            send_goal_future.set_result(child_goal_handle)
        else:
            child_goal_handle.complete_cancel_cleanup()

    monkeypatch.setattr(skill_executor_node.time, "sleep", advance_cancel_cleanup)
    parent_goal_handle = _ParentGoalHandle(events)

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCELLED"
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done()
    assert parent_goal_handle.canceled_count == 1
    assert events.index("child_terminal") < events.index("parent_cancelled")


def test_execute_skill_aborts_when_cancel_cleanup_does_not_reach_terminal(monkeypatch):
    events = []
    child_goal_handle = _ChildGoalHandle(events, complete_result_on_cancel=False)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = _make_skill_node(_Future(done=True, result=child_goal_handle))
    node._rpc_timeout = 0.0
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _ParentGoalHandle(events)

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_CLEANUP_TIMEOUT"
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done() is False
    assert parent_goal_handle.abort_count == 1
    assert parent_goal_handle.canceled_count == 0
    assert events[-1] == "parent_aborted"


def test_exec_arm_joint_trajectory_waits_for_downstream_terminal_after_cancel(monkeypatch):
    events = []
    child_goal_handle = _ChildGoalHandle(events)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(skill_executor_node.time, "sleep", lambda _seconds: child_goal_handle.complete_cancel_cleanup())
    node = object.__new__(SkillExecutorNode)
    node._rpc_timeout = 0.01
    node._debug = False
    node._arm_trajectory_action_name = "/arm/follow_joint_trajectory"
    node._arm_trajectory_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal: _Future(done=True, result=child_goal_handle),
    )

    ok, message = node._exec_arm_joint_trajectory(_PrimitiveGoalHandle(), ["joint_1"], [0.1], "task-1", 1.0, 0.4)

    assert ok is False
    assert message.startswith("cancelled")
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done()
    assert events == ["child_cancel", "child_terminal"]


def test_exec_arm_joint_trajectory_reports_cancel_cleanup_timeout(monkeypatch):
    child_goal_handle = _ChildGoalHandle(complete_result_on_cancel=False)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = object.__new__(SkillExecutorNode)
    node._rpc_timeout = 0.0
    node._debug = False
    node._arm_trajectory_action_name = "/arm/follow_joint_trajectory"
    node._arm_trajectory_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal: _Future(done=True, result=child_goal_handle),
    )

    ok, message = node._exec_arm_joint_trajectory(_PrimitiveGoalHandle(), ["joint_1"], [0.1], "task-1", 1.0, 0.4)

    assert ok is False
    assert message.startswith("cancel cleanup timed out")
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done() is False


def test_execute_primitive_aborts_when_downstream_cancel_cleanup_times_out():
    node = object.__new__(SkillExecutorNode)
    node._validate_primitive = lambda *_args, **_kwargs: (True, "")
    node._exec_arm_joint_trajectory = lambda *_args, **_kwargs: (
        False,
        "cancel cleanup timed out during arm joint trajectory execution",
    )
    goal_handle = _PrimitiveGoalHandle()

    result = node._execute_primitive(goal_handle)

    assert result.success is False
    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert goal_handle.abort_count == 1
    assert goal_handle.canceled_count == 0
