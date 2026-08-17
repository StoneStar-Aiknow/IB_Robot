"""Unit tests for task executor child action IDs without a ROS graph."""

import threading
from types import SimpleNamespace

import pytest
from action_msgs.msg import GoalStatus

from embodied_agent.task_executor_node import TaskExecutorNode, derive_skill_task_id
from embodied_common.dispatch_binding import new_binding, workflow_step
from ibrobot_msgs.msg import TaskCommand


class _DoneFuture:
    def __init__(self, value):
        self._value = value

    def done(self) -> bool:
        return True

    def result(self):
        return self._value


class _AcceptedGoalHandle:
    accepted = True

    def get_result_async(self):
        result = SimpleNamespace(success=True, error_code="", message="completed")
        return _DoneFuture(SimpleNamespace(result=result, status=GoalStatus.STATUS_SUCCEEDED))

    def cancel_goal_async(self):
        return _DoneFuture(SimpleNamespace(goals_canceling=[object()]))


class _ImmediateActionClient:
    def __init__(self):
        self.goals = []

    def wait_for_server(self, *, timeout_sec: float) -> bool:
        return True

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return _DoneFuture(_AcceptedGoalHandle())


class _ImmediateServiceClient:
    def __init__(self, callback):
        self._callback = callback
        self.requests = []

    def wait_for_service(self, *, timeout_sec: float) -> bool:
        return True

    def call_async(self, request):
        self.requests.append(request)
        return _DoneFuture(self._callback(request))


def _make_message(task_id: str, skill_sequence: list[str]):
    message = TaskCommand()
    message.dispatch_binding = new_binding(task_id=task_id)
    message.dispatch_binding.expected_registry_epoch = "epoch-1"
    message.dispatch_binding.expected_registry_generation = 1
    message.dispatch_binding.expected_registry_digest = "registry-digest"
    message.context_json = "{}"
    message.timeout_sec = 30.0
    message.dispatch_binding.task_budget.schema_version = 1
    message.dispatch_binding.task_budget.started_at.sec = 1
    message.dispatch_binding.task_budget.deadline.sec = 2_000_000_000
    message.workflow_steps = [
        workflow_step(
            skill_name=skill,
            target_name="target",
            place_name="place",
            motion_direction="forward",
            motion_distance=0.1,
            timeout_sec=30.0,
        )
        for skill in skill_sequence
    ]
    return message


def _make_executor():
    node = object.__new__(TaskExecutorNode)
    node._default_timeout = 180.0
    node._rpc_timeout = 5.0
    node._debug = False
    node._active_task_id = "parent"
    node._active_task_lock = threading.Lock()
    node._active_task_lock.acquire()
    node._skill_client = _ImmediateActionClient()
    node._gateway_status_service = "/status"
    node._begin_workflow_service = "/begin"
    node._finalize_workflow_service = "/finalize"
    node._gateway_status_client = _ImmediateServiceClient(
        lambda _request: SimpleNamespace(
            registry_epoch="epoch-1",
            registry_generation=1,
            registry_digest="registry-digest",
        )
    )
    node._begin_workflow_client = _ImmediateServiceClient(
        lambda request: SimpleNamespace(
            success=True,
            root_lease_nonce="root-nonce",
            workflow_digest=request.dispatch_binding.workflow_digest,
            error_code="",
            message="",
        )
    )
    node._finalize_workflow_client = _ImmediateServiceClient(
        lambda _request: SimpleNamespace(success=True, error_code="", message="")
    )
    statuses = []
    node._publish_status = lambda **kwargs: statuses.append(kwargs)
    return node, statuses


def test_derive_skill_task_id_trims_parent_and_is_deterministic():
    assert derive_skill_task_id(" task-1 ", 0) == "task-1/skill/0001"
    assert derive_skill_task_id("task-1", 1) == "task-1/skill/0002"
    assert derive_skill_task_id("task-1", 0) == derive_skill_task_id("task-1", 0)


@pytest.mark.parametrize("parent_task_id", ["", "   "])
def test_derive_skill_task_id_rejects_empty_parent(parent_task_id):
    with pytest.raises(ValueError):
        derive_skill_task_id(parent_task_id, 0)


@pytest.mark.parametrize("skill_index", [-1, True, 1.0, "0"])
def test_derive_skill_task_id_rejects_invalid_index(skill_index):
    with pytest.raises((TypeError, ValueError)):
        derive_skill_task_id("task-1", skill_index)


def test_execute_task_dispatches_distinct_child_ids_and_parent_statuses():
    node, statuses = _make_executor()
    message = _make_message("parent", ["first_skill", "second_skill"])

    try:
        node._execute_task(message)  # noqa: SLF001
    finally:
        assert not node._active_task_lock.locked()  # noqa: SLF001

    assert [goal.dispatch_binding.task_id for goal in node._skill_client.goals] == [  # noqa: SLF001
        "parent/skill/0001",
        "parent/skill/0002",
    ]
    assert [goal.schema_version for goal in node._skill_client.goals] == [1, 1]  # noqa: SLF001
    assert {goal.dispatch_binding.root_lease_nonce for goal in node._skill_client.goals} == {"root-nonce"}
    assert len({goal.dispatch_binding.workflow_digest for goal in node._skill_client.goals}) == 1
    assert len(node._finalize_workflow_client.requests) == 1  # noqa: SLF001
    assert {status["task_id"] for status in statuses} == {"parent"}
    assert statuses[-1]["state"] == "completed"
    assert statuses[-1]["completed_skills"] == ["first_skill", "second_skill"]


def test_execute_task_derives_first_child_id_for_single_skill():
    node, statuses = _make_executor()
    message = _make_message("parent", ["only_skill"])

    try:
        node._execute_task(message)  # noqa: SLF001
    finally:
        assert not node._active_task_lock.locked()  # noqa: SLF001

    assert [goal.dispatch_binding.task_id for goal in node._skill_client.goals] == ["parent/skill/0001"]  # noqa: SLF001
    assert {status["task_id"] for status in statuses} == {"parent"}


def test_execute_task_preserves_navigation_step_fields():
    node, _statuses = _make_executor()
    message = _make_message("parent", ["nav_abs_coordinate"])
    message.workflow_steps = [
        workflow_step(
            schema_version=2,
            skill_name="nav_abs_coordinate",
            x=0.0,
            y=-3.0,
            yaw=0.0,
            timeout_sec=30.0,
        )
    ]

    try:
        node._execute_task(message)  # noqa: SLF001
    finally:
        assert not node._active_task_lock.locked()  # noqa: SLF001

    goal = node._skill_client.goals[0]  # noqa: SLF001
    assert goal.schema_version == 2
    assert goal.has_x is True and goal.x == pytest.approx(0.0)
    assert goal.has_y is True and goal.y == pytest.approx(-3.0)
    assert goal.has_yaw is True and goal.yaw == pytest.approx(0.0)


def test_execute_task_preserves_zero_step_timeout_for_gateway_identity():
    node, _statuses = _make_executor()
    message = _make_message("parent", ["only_skill"])
    message.workflow_steps = [workflow_step(skill_name="only_skill", timeout_sec=0.0)]

    try:
        node._execute_task(message)  # noqa: SLF001
    finally:
        assert not node._active_task_lock.locked()  # noqa: SLF001

    assert node._skill_client.goals[0].timeout_sec == pytest.approx(0.0)  # noqa: SLF001


def test_execute_task_rejects_invalid_parent_without_dispatch():
    node, statuses = _make_executor()
    message = _make_message("   ", ["only_skill"])

    try:
        node._execute_task(message)  # noqa: SLF001
    finally:
        assert not node._active_task_lock.locked()  # noqa: SLF001

    assert node._skill_client.goals == []  # noqa: SLF001
    assert len(statuses) == 1
    assert statuses[0]["task_id"] == ""
    assert statuses[0]["error_code"] == "INVALID_TASK_ID"


class _PendingFuture:
    def done(self) -> bool:
        return False


class _UnknownBeginClient(_ImmediateServiceClient):
    def call_async(self, request):
        self.requests.append(request)
        return _PendingFuture()


class _UnknownAcceptanceClient(_ImmediateActionClient):
    def send_goal_async(self, goal):
        self.goals.append(goal)
        return _PendingFuture()


class _TimedOutGoalHandle:
    accepted = True

    def __init__(self, cancel_response):
        self._result_future = _PendingFuture()
        self._cancel_response = cancel_response

    def get_result_async(self):
        return self._result_future

    def cancel_goal_async(self):
        return _DoneFuture(self._cancel_response)


class _TimedOutActionClient(_ImmediateActionClient):
    def __init__(self, cancel_response):
        super().__init__()
        self._handle = _TimedOutGoalHandle(cancel_response)

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return _DoneFuture(self._handle)


def test_unknown_goal_acceptance_does_not_finalize_or_report_terminal_state():
    node, statuses = _make_executor()
    node._skill_client = _UnknownAcceptanceClient()  # noqa: SLF001
    node._wait_for_future = lambda future, timeout_sec: future.done()  # noqa: SLF001

    node._execute_task(_make_message("parent", ["only_skill"]))  # noqa: SLF001

    assert node._finalize_workflow_client.requests == []  # noqa: SLF001
    assert statuses[-1]["state"] == "unknown"
    assert statuses[-1]["error_code"] == "CHILD_STATE_UNKNOWN"


def test_cancel_rejection_does_not_finalize_or_report_timeout_as_terminal():
    node, statuses = _make_executor()
    node._skill_client = _TimedOutActionClient(SimpleNamespace(goals_canceling=[]))  # noqa: SLF001
    node._wait_for_future = lambda future, timeout_sec: future.done()  # noqa: SLF001

    node._execute_task(_make_message("parent", ["only_skill"]))  # noqa: SLF001

    assert node._finalize_workflow_client.requests == []  # noqa: SLF001
    assert statuses[-1]["state"] == "unknown"
    assert statuses[-1]["error_code"] == "CHILD_STATE_UNKNOWN"


def test_unknown_workflow_begin_does_not_dispatch_or_finalize():
    node, statuses = _make_executor()
    node._begin_workflow_client = _UnknownBeginClient(lambda _request: None)  # noqa: SLF001
    node._wait_for_future = lambda future, timeout_sec: future.done()  # noqa: SLF001

    node._execute_task(_make_message("parent", ["only_skill"]))  # noqa: SLF001

    assert node._skill_client.goals == []  # noqa: SLF001
    assert node._finalize_workflow_client.requests == []  # noqa: SLF001
    assert statuses[-1]["state"] == "unknown"
    assert statuses[-1]["error_code"] == "CHILD_STATE_UNKNOWN"
