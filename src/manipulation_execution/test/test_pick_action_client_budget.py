"""The supervised client's dispatch budgets on the ROS clock (lekiwi-nav-grasp R3).

The pick executor and the Gateway read ``task_budget.started_at``/``deadline``
against their own ROS clock — simulation time under ``use_sim_time`` — so the
client must stamp the budget from ITS node clock, never ``time.time()``.
These tests drive the client's own builders (no copy of the dispatch-binding
code) and verify the produced goal against the executor's own admission
check (``pick_goal_rejection``): admitted on the sim clock, rejected as
expired once the deadline passes. The result watchdog stays WALL time and
cancels the goal when it expires.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from rclpy.clock import ClockType
from rclpy.time import Time

from manipulation_execution.pick.action_client import (
    PickActionClient,
    build_pick_goal,
    fill_dispatch_budget,
)
from manipulation_execution.pick.goal import pick_goal_rejection

_IDENTITY = {
    "name": "grasp_pipeline",
    "contract_version": "1",
    "endpoint_kind": "action",
    "endpoint_name": "/manipulation/execute_pick",
    "configuration_digest": "digest-1",
    "model_deployment_name": "deployment",
    "model_fingerprint": "fingerprint",
    "model_bundle_digest": "bundle",
}

_REGISTRY = {"epoch": "7", "generation": 3, "digest": "registry-digest"}


def _sim_now(sec: float) -> Time:
    return Time(seconds=int(sec), nanoseconds=int((sec % 1.0) * 1_000_000_000), clock_type=ClockType.ROS_TIME)


def test_the_dispatch_budget_is_stamped_from_the_ros_clock_reading():
    from ibrobot_msgs.action import PickObject

    binding = PickObject.Goal().dispatch_binding
    # a SIM-time reading far from any wall epoch: the budget must carry it
    # verbatim (the old code stamped time.time() here — an epoch-sized
    # offset the executor reads as an expired budget on sim time)
    fill_dispatch_budget(binding, _sim_now(100.5), 30.0)
    assert binding.task_budget.schema_version == 1
    assert binding.task_budget.started_at.sec == 100
    assert binding.task_budget.started_at.nanosec == 500_000_000
    assert binding.task_budget.deadline.sec == 130
    assert binding.task_budget.deadline.nanosec == 500_000_000


def test_the_built_goal_admits_on_the_executor_clock():
    goal = build_pick_goal(
        task_id="supervised-pick-a",
        registry=_REGISTRY,
        identity=_IDENTITY,
        timeout_sec=25.0,
        task_budget_sec=30.0,
        prompt="banana",
        release_after_success=False,
        now=_sim_now(100.5),
    )
    assert goal.supervised_direct is True
    assert goal.target_query == "banana"
    # admitted on the executor's sim clock right after the stamp
    assert pick_goal_rejection(goal, _IDENTITY, now_s=101.0) == ""


def test_the_built_goal_expires_on_the_executor_clock():
    goal = build_pick_goal(
        task_id="supervised-pick-a",
        registry=_REGISTRY,
        identity=_IDENTITY,
        timeout_sec=25.0,
        task_budget_sec=30.0,
        prompt="banana",
        release_after_success=False,
        now=_sim_now(100.5),
    )
    # the deadline is 130.5 on the ROS clock: an executor reading past it
    # rejects the goal as expired (a wall-time stamp would never expire
    # inside a simulation, or expire instantly — the defect R3 fixes)
    reason = pick_goal_rejection(goal, _IDENTITY, now_s=131.0)
    assert "expired" in reason


class _NeverDoneFuture:
    def done(self) -> bool:
        return False


class _DoneFuture:
    def __init__(self, result):
        self._result = result

    def done(self) -> bool:
        return True

    def result(self):
        return self._result


def test_the_result_watchdog_cancels_the_goal_when_it_expires():
    handle = SimpleNamespace(
        get_result_async=lambda: _NeverDoneFuture(),
        cancelled=False,
    )

    def cancel():
        handle.cancelled = True

    handle.cancel_goal_async = cancel
    spun = []
    fake_self = SimpleNamespace(_spin_until=lambda future, timeout_sec: spun.append(timeout_sec))
    with pytest.raises(RuntimeError, match="result timed out after 25.0s"):
        PickActionClient._await_result(fake_self, handle, 25.0)
    assert handle.cancelled, "an expired result watchdog must cancel the goal"
    # the watchdog stays WALL time (timeout + the 10 s delivery margin)
    assert spun == [35.0]


def test_the_result_watchdog_returns_the_wrapped_result():
    expected = SimpleNamespace(success=True)
    handle = SimpleNamespace(
        get_result_async=lambda: _DoneFuture(SimpleNamespace(result=expected)),
        cancel_goal_async=lambda: pytest.fail("a completing goal is never cancelled"),
    )
    fake_self = SimpleNamespace(_spin_until=lambda future, timeout_sec: None)
    assert PickActionClient._await_result(fake_self, handle, 25.0) is expected


def test_the_client_node_accepts_programmatic_sim_time():
    import rclpy

    rclpy.init(args=[])
    try:
        args = SimpleNamespace(
            action_name="/manipulation/execute_pick",
            status_service="/embodied/get_skill_gateway_status",
            snapshot_service="/embodied/get_skill_snapshot",
        )
        node = PickActionClient(args, use_sim_time=True)
        try:
            # the explicit in-process entry switches the node clock to
            # simulation time (R3); without /clock it reads time zero
            assert node.get_clock().clock_type == ClockType.ROS_TIME
            assert node.get_clock().now().nanoseconds == 0
        finally:
            node.destroy_node()
    finally:
        rclpy.shutdown()
