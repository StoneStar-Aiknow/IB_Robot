import threading
from threading import RLock
from types import SimpleNamespace

import pytest
import rclpy

from skill_library import skill_executor_node
from skill_library.gateway_policy import (
    BoundedRequestLedger,
    ExecutionOwner,
    GatewayPolicy,
    GatewayRequest,
    RootExecutionLease,
    RuntimeSnapshot,
    SkillRequirements,
)
from skill_library.resolver import PrimitiveSpec
from skill_library.skill_executor_node import SkillExecutorNode


class _Future:
    def __init__(self, *, done: bool, result=None, exception: Exception | None = None) -> None:
        self._done = done
        self._result = result
        self._exception = exception
        self._callbacks = []

    def done(self) -> bool:
        return self._done

    def result(self):
        if self._exception is not None:
            raise self._exception
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

    def set_exception(self, exception: Exception) -> None:
        self._exception = exception
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


class _CancelResponseRaisesGoalHandle(_ChildGoalHandle):
    def cancel_goal_async(self):
        self.cancel_count += 1
        self.events.append("child_cancel")
        self.result_future.set_result(SimpleNamespace(result=SimpleNamespace(error_code="")))
        return _Future(done=True, exception=RuntimeError("cancel response unavailable"))


class _ParentGoalHandle:
    def __init__(self, events=None) -> None:
        self.events = events if events is not None else []
        self.feedback = []
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
        self.succeeded_count = 0

    @property
    def is_cancel_requested(self) -> bool:
        self.cancel_checks += 1
        return self.cancel_checks >= 2

    def publish_feedback(self, feedback) -> None:
        self.feedback.append(feedback)

    def canceled(self) -> None:
        self.canceled_count += 1
        self.events.append("parent_cancelled")

    def abort(self) -> None:
        self.abort_count += 1
        self.events.append("parent_aborted")

    def succeed(self) -> None:
        self.succeeded_count += 1
        self.events.append("parent_succeeded")


class _PrimitiveGoalHandle:
    def __init__(self) -> None:
        self.request = SimpleNamespace(
            primitive_name="move_to_joint_positions",
            pose_name="",
            relative_dx=0.0,
            relative_dy=0.0,
            relative_dz=0.0,
            gripper_position=0.0,
            velocity_scaling=0.0,
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
        self.succeeded_count = 0

    @property
    def is_cancel_requested(self) -> bool:
        return True

    def publish_feedback(self, _feedback) -> None:
        pass

    def canceled(self) -> None:
        self.canceled_count += 1

    def abort(self) -> None:
        self.abort_count += 1

    def succeed(self) -> None:
        self.succeeded_count += 1


class _NoCancelParentGoalHandle(_ParentGoalHandle):
    @property
    def is_cancel_requested(self) -> bool:
        return False


class _ManualCancelParentGoalHandle(_ParentGoalHandle):
    def __init__(self, events=None) -> None:
        super().__init__(events)
        self.cancel_requested = False

    @property
    def is_cancel_requested(self) -> bool:
        return self.cancel_requested


def _make_skill_node(send_goal_future) -> SkillExecutorNode:
    node = object.__new__(SkillExecutorNode)
    node._skill_goal_lock = threading.Lock()
    node._skill_goal_active = False
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
        send_goal_async=lambda _goal, **_kwargs: send_goal_future,
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


def test_best_effort_cancel_audits_propagation_once():
    events = []
    goal_handle = SimpleNamespace(cancel_goal_async=lambda: events.append("cancel"))
    node = object.__new__(SkillExecutorNode)
    node._audit_cancel_propagated = lambda: events.append("audit")

    node._best_effort_cancel_goal(goal_handle)

    assert events == ["audit", "cancel"]


def test_fresh_ee_pose_snapshot_is_independent_and_rejects_stale_pose(monkeypatch):
    monotonic_time = [100.0]
    monkeypatch.setattr(skill_executor_node.time, "monotonic", lambda: monotonic_time[0])
    node = object.__new__(SkillExecutorNode)
    node._state_lock = RLock()
    node._robot_state_freshness_sec = 0.1
    pose = skill_executor_node.PoseStamped()
    pose.pose.position.x = 0.2
    pose.pose.orientation.w = 1.0

    node._handle_ee_pose(pose)
    snapshot = node._fresh_ee_pose_snapshot()

    assert snapshot.pose is not pose
    assert snapshot.pose.pose.position.x == pytest.approx(0.2)
    assert snapshot.received_monotonic == pytest.approx(100.0)
    pose.pose.position.x = 0.4
    assert snapshot.pose.pose.position.x == pytest.approx(0.2)

    monotonic_time[0] += 0.09
    assert node._ee_pose_receipt_is_fresh(snapshot.received_monotonic) is True
    monotonic_time[0] += 0.02
    assert node._ee_pose_receipt_is_fresh(snapshot.received_monotonic) is False
    assert node._fresh_ee_pose_snapshot() is None


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

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done() is False
    assert parent_goal_handle.abort_count == 1
    assert parent_goal_handle.canceled_count == 0
    assert events[-1] == "parent_aborted"


def test_execute_skill_aborts_when_child_cancel_response_is_unknown(monkeypatch):
    events = []
    child_goal_handle = _CancelResponseRaisesGoalHandle(events)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = _make_skill_node(_Future(done=True, result=child_goal_handle))
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _ParentGoalHandle(events)

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert child_goal_handle.cancel_count == 1
    assert parent_goal_handle.abort_count == 1
    assert parent_goal_handle.canceled_count == 0


def test_execute_skill_send_timeout_fails_closed_when_late_goal_cleanup_is_unknown(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    node = _make_skill_node(_Future(done=False))
    node._rpc_timeout = 0.0
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _NoCancelParentGoalHandle()

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert parent_goal_handle.abort_count == 1


def test_execute_skill_feedback_uses_sanitized_step_progress(monkeypatch):
    child_goal_handle = _ChildGoalHandle()
    child_goal_handle.result_future.set_result(
        SimpleNamespace(result=SimpleNamespace(success=True, error_code="", message=""))
    )
    node = _make_skill_node(_Future(done=True, result=child_goal_handle))
    audit_events = []
    node._active_audit_context = {"task_id": "task-1", "payload_hash": "digest", "skill": "test_skill"}
    node._audit = lambda event, **fields: audit_events.append((event, fields))
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [
            PrimitiveSpec("move_to_named_pose", pose_name="private_pose"),
            PrimitiveSpec("move_to_joint_positions", joint_names=["joint_1"], joint_positions=[0.1]),
        ],
    )
    parent_goal_handle = _NoCancelParentGoalHandle()

    result = node._execute_skill(parent_goal_handle)

    assert result.success is True
    assert [(feedback.state, feedback.detail) for feedback in parent_goal_handle.feedback] == [
        ("executing", "step 1 of 2"),
        ("executing", "step 2 of 2"),
    ]
    assert all(
        value not in feedback.detail
        for feedback in parent_goal_handle.feedback
        for value in ("move_to_named_pose", "move_to_joint_positions", "private_pose", "joint_1")
    )
    assert [fields for event, fields in audit_events if event == "primitive_started"] == [
        {"task_id": "task-1", "payload_hash": "digest", "skill": "test_skill", "step_count": 1},
        {"task_id": "task-1", "payload_hash": "digest", "skill": "test_skill", "step_count": 2},
    ]


class _GatewayPolicyStub:
    def __init__(self, events, *, finalize_raises: bool = False) -> None:
        self.events = events
        self.finalize_raises = finalize_raises
        self.admission = SimpleNamespace(
            admitted=True,
            effective_timeout_sec=1.0,
            prepared_request=SimpleNamespace(identity=SimpleNamespace(task_id="task-1", payload_hash="digest")),
        )
        self.finalize_calls = 0

    def admit(self, *_args, **_kwargs):
        return self.admission

    def finalize(self, *_args, **_kwargs):
        self.finalize_calls += 1
        self.events.append("finalize")
        if self.finalize_raises:
            raise RuntimeError("injected finalization failure")


def _make_gateway_wrapper_node(events, *, finalize_raises: bool = False) -> SkillExecutorNode:
    node = object.__new__(SkillExecutorNode)
    node._skill_goal_lock = threading.Lock()
    node._skill_goal_active = False
    node._gateway_policy = _GatewayPolicyStub(events, finalize_raises=finalize_raises)
    node._runtime_snapshot = lambda: SimpleNamespace()
    node._validate_skill = lambda *_args, **_kwargs: (True, "")
    node._audit = lambda *_args, **_kwargs: None
    node.get_logger = lambda: SimpleNamespace(error=lambda _message: None)
    node._active_skill_admission = None
    node._active_skill_owner = None
    node._active_audit_context = None
    node._state_lock = RLock()
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}
    node._retained_admission_cleanup = {}
    return node


def _make_retained_gateway_node(send_goal_future) -> tuple[SkillExecutorNode, GatewayPolicy, RootExecutionLease]:
    lease = RootExecutionLease()
    node = _make_skill_node(send_goal_future)
    node._gateway_lease = lease
    node._gateway_ledger = BoundedRequestLedger(4)
    node._gateway_policy = GatewayPolicy(
        {"default_skill_timeout_sec": 1.0, "task_budget_sec": 2.0},
        {"test_skill": SkillRequirements()},
        parameter_schemas={
            "test_skill": {
                "type": "object",
                "additionalProperties": False,
                "properties": {},
                "required": [],
            }
        },
        ledger=node._gateway_ledger,
        lease=lease,
    )
    node._runtime_snapshot = lambda: RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="gateway_mode",
        required_control_mode="gateway_mode",
    )
    node._audit = lambda *_args, **_kwargs: None
    node.get_logger = lambda: SimpleNamespace(error=lambda _message: None)
    node._active_skill_admission = None
    node._active_skill_owner = None
    node._active_audit_context = None
    node._state_lock = RLock()
    node._internal_goal_lock = node._state_lock
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}
    node._retained_admission_cleanup = {}
    return node, node._gateway_policy, lease


class _TerminalOnCancelGoalHandle(_ChildGoalHandle):
    def cancel_goal_async(self):
        cancel_future = super().cancel_goal_async()
        self.result_future.set_result(SimpleNamespace(result=SimpleNamespace(error_code="")))
        return cancel_future


class _ResultFutureRaisesGoalHandle:
    accepted = True

    def __init__(self) -> None:
        self.cancel_count = 0

    def get_result_async(self):
        raise RuntimeError("result future unavailable")

    def cancel_goal_async(self):
        self.cancel_count += 1
        return _Future(done=True)


class _ExternalPolicyStub:
    def __init__(self, release_result) -> None:
        self.release_result = release_result
        self.token = object()

    def borrow_internal(self, *_args, **_kwargs):
        return None

    def admit_external_primitive(self, *_args, **_kwargs):
        return "", self.token

    def release_external_primitive(self, _token):
        if isinstance(self.release_result, Exception):
            raise self.release_result
        return self.release_result


class _PrimitiveActionGoalHandle:
    def __init__(self, goal_id, task_id: str = "task-1") -> None:
        self.goal_id = goal_id
        self.request = SimpleNamespace(
            task_id=task_id,
            primitive_name="open_gripper",
            pose_name="",
        )
        self.abort_count = 0
        self.canceled_count = 0
        self.succeeded_count = 0

    def abort(self) -> None:
        self.abort_count += 1

    def canceled(self) -> None:
        self.canceled_count += 1

    def succeed(self) -> None:
        self.succeeded_count += 1

    @property
    def is_cancel_requested(self) -> bool:
        return False

    def publish_feedback(self, _feedback) -> None:
        pass


class _CountingReleasePolicy(_ExternalPolicyStub):
    def __init__(self) -> None:
        super().__init__(True)
        self.release_count = 0

    def release_external_primitive(self, token):
        self.release_count += 1
        return super().release_external_primitive(token)


def test_deferred_terminal_records_child_intent_without_terminalizing_parent():
    parent_goal_handle = _ParentGoalHandle()
    deferred_goal_handle = skill_executor_node._DeferredTerminalGoalHandle(parent_goal_handle)

    deferred_goal_handle.canceled()

    assert deferred_goal_handle.terminal_intent == "canceled"
    assert parent_goal_handle.canceled_count == 0
    assert parent_goal_handle.abort_count == 0
    assert parent_goal_handle.succeeded_count == 0


def test_deferred_terminal_commits_once_using_recorded_intent():
    parent_goal_handle = _ParentGoalHandle()
    deferred_goal_handle = skill_executor_node._DeferredTerminalGoalHandle(parent_goal_handle)

    deferred_goal_handle.succeed()

    assert deferred_goal_handle.commit() is True
    assert deferred_goal_handle.commit() is False
    assert parent_goal_handle.succeeded_count == 1
    assert parent_goal_handle.abort_count == 0


def test_gateway_defers_parent_success_until_policy_finalization():
    events = []
    node = _make_gateway_wrapper_node(events)
    parent_goal_handle = _ParentGoalHandle(events)

    def child(goal_handle, **_kwargs):
        events.append("child_terminal")
        goal_handle.succeed()
        return SimpleNamespace(success=True, error_code="", message="", executed_primitives=[])

    node._execute_skill_child = child

    result = node._execute_skill(parent_goal_handle)

    assert result.success is True
    assert events == ["child_terminal", "finalize", "parent_succeeded"]


def test_gateway_aborts_instead_of_committing_success_when_finalization_fails():
    events = []
    node = _make_gateway_wrapper_node(events, finalize_raises=True)
    parent_goal_handle = _ParentGoalHandle(events)

    def child(goal_handle, **_kwargs):
        events.append("child_terminal")
        goal_handle.succeed()
        return SimpleNamespace(success=True, error_code="", message="", executed_primitives=[])

    node._execute_skill_child = child

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "GATEWAY_FINALIZATION_FAILED"
    assert events == ["child_terminal", "finalize", "parent_aborted"]


def test_gateway_cleanup_unknown_aborts_without_finalizing_or_releasing_lease():
    events = []
    node = _make_gateway_wrapper_node(events)
    parent_goal_handle = _ParentGoalHandle(events)

    def child(goal_handle, **_kwargs):
        events.append("child_terminal")
        goal_handle.abort()
        return SimpleNamespace(
            success=False,
            error_code="SKILL_CANCEL_TIMEOUT",
            message="cleanup unknown",
            executed_primitives=[],
        )

    node._execute_skill_child = child

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert node._gateway_policy.finalize_calls == 0
    assert events == ["child_terminal", "parent_aborted"]


def test_external_primitive_keeps_lease_when_downstream_cleanup_is_unknown():
    lease = RootExecutionLease()
    policy = GatewayPolicy(
        {"default_skill_timeout_sec": 1.0, "task_budget_sec": 2.0},
        {},
        parameter_schemas={},
        ledger=BoundedRequestLedger(1),
        lease=lease,
    )
    node = object.__new__(SkillExecutorNode)
    node._gateway_policy = policy
    node._gateway_lease = lease
    node._active_skill_admission = None
    node._active_skill_owner = None
    node._runtime_snapshot = lambda: RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="gateway_mode",
        required_control_mode="gateway_mode",
    )
    node._execute_primitive_unchecked = lambda _goal_handle: SimpleNamespace(
        success=False,
        error_code="CANCEL_CLEANUP_TIMEOUT",
        message="cleanup unknown",
    )
    goal_handle = _PrimitiveGoalHandle()

    result = node._execute_primitive(goal_handle)

    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert lease.owner == ExecutionOwner.external_primitive("task-1")


@pytest.mark.parametrize("release_result", [False, RuntimeError("release failed")])
def test_external_primitive_aborts_when_lease_release_cannot_be_confirmed(release_result):
    node = object.__new__(SkillExecutorNode)
    node._gateway_policy = _ExternalPolicyStub(release_result)
    node._runtime_snapshot = lambda: RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="gateway_mode",
        required_control_mode="gateway_mode",
    )
    node._state_lock = RLock()
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.succeed() or SimpleNamespace(success=True, error_code="", message="")
    )
    goal_handle = _PrimitiveGoalHandle()

    result = node._execute_primitive(goal_handle)

    assert result.error_code == "GATEWAY_FINALIZATION_FAILED"
    assert goal_handle.abort_count == 1
    assert goal_handle.succeeded_count == 0


def test_gateway_send_future_exception_retains_root_and_revokes_internal_goal(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    send_future = _Future(done=True, exception=RuntimeError("send future failed"))
    node, policy, lease = _make_retained_gateway_node(send_future)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _NoCancelParentGoalHandle()

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert parent_goal_handle.abort_count == 1
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert policy._ledger.query("task-1").state == "active"
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}


def test_revoked_internal_uuid_cannot_borrow_after_send_future_exception(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    sent_goals = {}
    node, _policy, _lease = _make_retained_gateway_node(_Future(done=True, exception=RuntimeError("send failed")))
    node._primitive_client.send_goal_async = lambda goal, **kwargs: (
        sent_goals.update(goal=goal, goal_id=kwargs["goal_uuid"]) or node._primitive_client_future
    )
    node._primitive_client_future = _Future(done=True, exception=RuntimeError("send failed"))
    downstream_calls = []
    node._execute_primitive_unchecked = lambda _goal_handle: downstream_calls.append(True)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    result = node._execute_skill(_NoCancelParentGoalHandle())
    late_goal_handle = SimpleNamespace(
        goal_id=sent_goals["goal_id"],
        request=SimpleNamespace(
            task_id="task-1",
            primitive_name="open_gripper",
            pose_name="",
        ),
        abort=lambda: None,
    )
    late_result = node._execute_primitive(late_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert late_result.error_code == "SKILL_BUSY"
    assert downstream_calls == []


def test_gateway_result_future_exception_retains_root_and_revokes_internal_goal(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    primitive_handle = _ResultFutureRaisesGoalHandle()
    node, policy, lease = _make_retained_gateway_node(_Future(done=True, result=primitive_handle))
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    result = node._execute_skill(_NoCancelParentGoalHandle())

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert policy._ledger.query("task-1").state == "active"
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}
    assert primitive_handle.cancel_count == 1


def test_late_rejected_internal_goal_converges_retained_root_admission(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    send_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_future)
    node._rpc_timeout = 0.0
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    result = node._execute_skill(_NoCancelParentGoalHandle())
    send_future.set_result(SimpleNamespace(accepted=False))

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert lease.owner is None
    assert policy._ledger.query("task-1").state == "terminal"
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}


def test_late_accepted_terminal_goal_converges_retained_root_admission(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    send_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_future)
    node._rpc_timeout = 0.0
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    result = node._execute_skill(_NoCancelParentGoalHandle())
    send_future.set_result(_TerminalOnCancelGoalHandle())

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert lease.owner is None
    assert policy._ledger.query("task-1").state == "terminal"
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}


def test_late_accepted_goal_with_unknown_terminal_state_keeps_root_busy(monkeypatch):
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    send_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_future)
    node._rpc_timeout = 0.0
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )

    result = node._execute_skill(_NoCancelParentGoalHandle())
    send_future.set_result(_ChildGoalHandle(complete_result_on_cancel=False))

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert policy._ledger.query("task-1").state == "active"


def test_gateway_parent_cancel_drains_pending_internal_goal_before_terminalizing(monkeypatch):
    events = []
    child_goal_handle = _ChildGoalHandle(events)
    send_goal_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_goal_future)
    finalized_error_codes = []
    original_finalize = policy.finalize
    policy.finalize = lambda admission, **kwargs: (
        finalized_error_codes.append(kwargs["error_code"]) or original_finalize(admission, **kwargs)
    )
    audit_events = []
    node._audit = lambda event, **_fields: audit_events.append(event)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _ManualCancelParentGoalHandle(events)

    def advance_pending_goal(_seconds) -> None:
        if not parent_goal_handle.cancel_requested:
            parent_goal_handle.cancel_requested = True
        elif not send_goal_future.done():
            send_goal_future.set_result(child_goal_handle)
        else:
            child_goal_handle.complete_cancel_cleanup()

    monkeypatch.setattr(skill_executor_node.time, "sleep", advance_pending_goal)

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCELLED"
    assert child_goal_handle.cancel_count == 1
    assert child_goal_handle.result_future.done()
    assert parent_goal_handle.canceled_count == 1
    assert parent_goal_handle.abort_count == 0
    assert events.index("child_terminal") < events.index("parent_cancelled")
    assert finalized_error_codes == ["SKILL_CANCELLED"]
    assert policy._ledger.query("task-1").state == "terminal"
    assert lease.owner is None
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}
    assert audit_events.count("cancel_propagated") == 1


def test_gateway_parent_cancel_retains_late_accepted_goal_without_terminal(monkeypatch):
    child_goal_handle = _ChildGoalHandle(complete_result_on_cancel=False)
    send_goal_future = _Future(done=False)
    node, policy, lease = _make_retained_gateway_node(send_goal_future)
    monkeypatch.setattr(rclpy, "ok", lambda: True)
    monkeypatch.setattr(
        skill_executor_node,
        "resolve_skill_primitives",
        lambda *_args, **_kwargs: [PrimitiveSpec("open_gripper", gripper_position=1.0)],
    )
    parent_goal_handle = _ManualCancelParentGoalHandle()

    def advance_pending_goal(_seconds) -> None:
        if not parent_goal_handle.cancel_requested:
            parent_goal_handle.cancel_requested = True
        elif not send_goal_future.done():
            send_goal_future.set_result(child_goal_handle)

    monkeypatch.setattr(skill_executor_node.time, "sleep", advance_pending_goal)

    result = node._execute_skill(parent_goal_handle)

    assert result.error_code == "SKILL_CANCEL_TIMEOUT"
    assert child_goal_handle.cancel_count == 1
    assert parent_goal_handle.abort_count == 1
    assert parent_goal_handle.canceled_count == 0
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert policy._ledger.query("task-1").state == "active"
    assert len(node._pending_internal_primitive_goals) == 1
    assert node._active_internal_primitive_goals == {}


def _admit_internal_primitive(node, policy):
    owner = ExecutionOwner.skill_command("task-1")
    admission = policy.admit(
        GatewayRequest(task_id="task-1", skill_name="test_skill"),
        node._runtime_snapshot(),
        owner,
    )
    assert admission.admitted
    node._active_skill_admission = admission
    node._active_skill_owner = owner
    node._active_audit_context = {"task_id": "task-1"}
    goal_id = skill_executor_node.UUID(uuid=[1] * 16)
    goal_key = node._register_internal_primitive_goal(goal_id, admission, "task-1")
    return admission, goal_id, goal_key


def test_clean_internal_primitive_terminal_converges_later_retained_root():
    node, policy, lease = _make_retained_gateway_node(_Future(done=False))
    admission, goal_id, goal_key = _admit_internal_primitive(node, policy)
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.succeed() or SimpleNamespace(success=True, error_code="", message="")
    )
    child_goal_handle = _PrimitiveActionGoalHandle(goal_id)

    result = node._execute_primitive(child_goal_handle)

    assert result.success is True
    assert child_goal_handle.succeeded_count == 1
    cleanup = node._retained_admission_cleanup[id(admission)]
    assert getattr(cleanup, "pending_goal_key", None) == goal_key
    assert getattr(cleanup, "confirmed_goal_key", None) == goal_key
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}
    assert lease.owner == ExecutionOwner.skill_command("task-1")

    node._retain_admission_cleanup(admission, {"task_id": "task-1"}, 0.1, 1)

    assert policy._ledger.query("task-1").state == "terminal"
    assert lease.owner is None
    assert node._retained_admission_cleanup == {}
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}
    assert goal_key not in node._pending_internal_primitive_goals


def test_cleanup_unknown_internal_primitive_does_not_confirm_retained_root():
    node, policy, lease = _make_retained_gateway_node(_Future(done=False))
    admission, goal_id, _goal_key = _admit_internal_primitive(node, policy)
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.abort()
        or SimpleNamespace(success=False, error_code="CANCEL_CLEANUP_TIMEOUT", message="cleanup unknown")
    )

    result = node._execute_primitive(_PrimitiveActionGoalHandle(goal_id))

    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    cleanup = node._retained_admission_cleanup[id(admission)]
    assert cleanup.pending_goal_key is not None
    assert cleanup.confirmed_goal_key is None

    node._retain_admission_cleanup(admission, {"task_id": "task-1"}, 0.1, 1)

    assert policy._ledger.query("task-1").state == "active"
    assert lease.owner == ExecutionOwner.skill_command("task-1")
    assert node._retained_admission_cleanup[id(admission)].confirmed_goal_key is None


def test_only_current_internal_goal_cleanup_can_converge_retained_root():
    node, policy, lease = _make_retained_gateway_node(_Future(done=False))
    admission, first_goal_id, first_goal_key = _admit_internal_primitive(node, policy)
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.succeed() or SimpleNamespace(success=True, error_code="", message="")
    )

    first_result = node._execute_primitive(_PrimitiveActionGoalHandle(first_goal_id))

    assert first_result.success is True
    second_goal_id = skill_executor_node.UUID(uuid=[2] * 16)
    second_goal_key = node._register_internal_primitive_goal(second_goal_id, admission, "task-1")
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.abort()
        or SimpleNamespace(success=False, error_code="CANCEL_CLEANUP_TIMEOUT", message="cleanup unknown")
    )

    second_result = node._execute_primitive(_PrimitiveActionGoalHandle(second_goal_id))
    node._retain_admission_cleanup(admission, {"task_id": "task-1"}, 0.1, 2)

    assert second_result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert policy._ledger.query("task-1").state == "active"
    assert lease.owner == ExecutionOwner.skill_command("task-1")

    node._confirm_late_internal_cleanup(admission, first_goal_key)

    assert policy._ledger.query("task-1").state == "active"
    assert lease.owner == ExecutionOwner.skill_command("task-1")

    node._confirm_late_internal_cleanup(admission, second_goal_key)

    assert policy._ledger.query("task-1").state == "terminal"
    assert lease.owner is None
    assert node._retained_admission_cleanup == {}
    assert node._pending_internal_primitive_goals == {}
    assert node._active_internal_primitive_goals == {}


def test_external_primitive_commits_child_terminal_intent_after_successful_release():
    node = object.__new__(SkillExecutorNode)
    node._gateway_policy = _ExternalPolicyStub(True)
    node._runtime_snapshot = lambda: RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="gateway_mode",
        required_control_mode="gateway_mode",
    )
    node._state_lock = RLock()
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}
    node._execute_primitive_unchecked = lambda goal_handle: (
        goal_handle.succeed() or SimpleNamespace(success=False, error_code="CHILD_ERROR", message="child error")
    )
    goal_handle = _PrimitiveGoalHandle()

    result = node._execute_primitive(goal_handle)

    assert result.error_code == "CHILD_ERROR"
    assert goal_handle.succeeded_count == 1
    assert goal_handle.abort_count == 0


def test_external_primitive_late_terminal_releases_original_token_once():
    downstream_terminal = _Future(done=False)
    node = object.__new__(SkillExecutorNode)
    node._gateway_policy = _CountingReleasePolicy()
    node._runtime_snapshot = lambda: RuntimeSnapshot(
        motion_authorized=True,
        active_control_mode="gateway_mode",
        required_control_mode="gateway_mode",
    )
    node._state_lock = RLock()
    node._pending_internal_primitive_goals = {}
    node._active_internal_primitive_goals = {}

    def unchecked(goal_handle):
        goal_handle.late_cleanup_confirmation.watch_result_future(downstream_terminal)
        goal_handle.abort()
        return SimpleNamespace(success=False, error_code="CANCEL_CLEANUP_TIMEOUT", message="cleanup unknown")

    node._execute_primitive_unchecked = unchecked

    result = node._execute_primitive(_PrimitiveGoalHandle())

    assert result.error_code == "CANCEL_CLEANUP_TIMEOUT"
    assert node._gateway_policy.release_count == 0

    downstream_terminal.set_result(SimpleNamespace(result=SimpleNamespace(error_code="")))
    downstream_terminal.set_result(SimpleNamespace(result=SimpleNamespace(error_code="")))

    assert node._gateway_policy.release_count == 1


def test_gateway_finalization_cannot_clear_a_newer_active_admission():
    node, policy, _lease = _make_retained_gateway_node(_Future(done=False))
    released = threading.Event()
    allow_first_wrapper_to_return = threading.Event()
    original_finalize = policy.finalize

    def finalize(admission, **kwargs):
        terminal = original_finalize(admission, **kwargs)
        if admission.prepared_request.identity.task_id == "task-a":
            released.set()
            assert allow_first_wrapper_to_return.wait(timeout=1.0)
        return terminal

    policy.finalize = finalize

    def child(goal_handle, **_kwargs):
        if goal_handle.request.task_id == "task-a":
            goal_handle.succeed()
            return SimpleNamespace(success=True, error_code="", message="", executed_primitives=[])
        goal_handle.abort()
        return SimpleNamespace(
            success=False,
            error_code="SKILL_CANCEL_TIMEOUT",
            message="cleanup unknown",
            executed_primitives=[],
        )

    node._execute_skill_child = child
    first_parent = _NoCancelParentGoalHandle()
    first_parent.request.task_id = "task-a"
    second_parent = _NoCancelParentGoalHandle()
    second_parent.request.task_id = "task-b"
    first_result = []
    second_result = []
    first_thread = threading.Thread(target=lambda: first_result.append(node._execute_skill(first_parent)))
    second_thread = threading.Thread(target=lambda: second_result.append(node._execute_skill(second_parent)))

    first_thread.start()
    assert released.wait(timeout=1.0)
    second_thread.start()
    allow_first_wrapper_to_return.set()
    first_thread.join(timeout=1.0)
    second_thread.join(timeout=1.0)

    assert first_result[0].success is True
    assert second_result[0].error_code == "SKILL_CANCEL_TIMEOUT"
    assert node._active_skill_admission.prepared_request.identity.task_id == "task-b"


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


def test_exec_arm_joint_trajectory_cancels_when_result_future_creation_fails():
    child_goal_handle = _ResultFutureRaisesGoalHandle()
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


def test_exec_arm_via_task_dispatch_cancels_when_result_future_creation_fails():
    child_goal_handle = _ResultFutureRaisesGoalHandle()
    node = object.__new__(SkillExecutorNode)
    node._rpc_timeout = 0.0
    node._debug = False
    node._task_executor_action_name = "/task_dispatch/execute_task_plan"
    node._task_executor_client = SimpleNamespace(
        wait_for_server=lambda **_kwargs: True,
        send_goal_async=lambda _goal: _Future(done=True, result=child_goal_handle),
    )

    ok, message = node._exec_arm_via_task_dispatch(
        _PrimitiveGoalHandle(),
        "move_to_named_pose",
        skill_executor_node.Pose(),
        "task-1",
        1.0,
    )

    assert ok is False
    assert message.startswith("cancel cleanup timed out")
    assert child_goal_handle.cancel_count == 1


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
