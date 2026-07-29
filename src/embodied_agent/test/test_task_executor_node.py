"""Unit tests for task executor child action IDs without a ROS graph."""

import json
import threading
from types import SimpleNamespace

import pytest

from embodied_agent.task_executor_node import TaskExecutorNode, derive_skill_task_id


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
        return _DoneFuture(SimpleNamespace(result=result))

    def cancel_goal_async(self):
        return _DoneFuture(None)


class _ImmediateActionClient:
    def __init__(self):
        self.goals = []

    def wait_for_server(self, *, timeout_sec: float) -> bool:
        return True

    def send_goal_async(self, goal):
        self.goals.append(goal)
        return _DoneFuture(_AcceptedGoalHandle())


def _make_message(task_id: str, skill_sequence: list[str]):
    return SimpleNamespace(
        task_id=task_id,
        context_json=json.dumps({"skill_sequence": skill_sequence}),
        timeout_sec=30.0,
        target_name="target",
        place_name="place",
        motion_direction="forward",
        motion_distance=0.1,
    )


def _make_executor():
    node = object.__new__(TaskExecutorNode)
    node._default_timeout = 180.0
    node._rpc_timeout = 5.0
    node._debug = False
    node._active_task_id = "parent"
    node._active_task_lock = threading.Lock()
    node._active_task_lock.acquire()
    node._skill_client = _ImmediateActionClient()
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

    assert [goal.task_id for goal in node._skill_client.goals] == [  # noqa: SLF001
        "parent/skill/0001",
        "parent/skill/0002",
    ]
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

    assert [goal.task_id for goal in node._skill_client.goals] == ["parent/skill/0001"]  # noqa: SLF001
    assert {status["task_id"] for status in statuses} == {"parent"}


def test_execute_task_rejects_invalid_parent_without_dispatch():
    node, statuses = _make_executor()
    message = _make_message("   ", ["only_skill"])

    try:
        node._execute_task(message)  # noqa: SLF001
    finally:
        assert not node._active_task_lock.locked()  # noqa: SLF001

    assert node._skill_client.goals == []  # noqa: SLF001
    assert len(statuses) == 1
    assert statuses[0]["task_id"] == "   "
    assert statuses[0]["error_code"] == "INVALID_TASK_ID"
