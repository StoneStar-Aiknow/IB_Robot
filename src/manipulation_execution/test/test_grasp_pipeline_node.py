"""The node serves PickObject over ROS for a robot whose backends are named in its configuration."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from test_grasp_pipeline import Effector, Executor, Proposer, Source, StreamingPlanner

from embodied_common.dispatch_binding import delegated_executor_identity, fill_delegated_executor_identity
from ibrobot_msgs.action import PickObject
from manipulation_execution.grasp_pipeline.contracts import EffectorAction, GraspBackends, MotionOutcome, MotionState
from manipulation_execution.grasp_pipeline.pipeline import PickOutcome
from manipulation_execution.grasp_pipeline_node import (
    GraspPipelineNode,
    pipeline_config,
    terminal_record,
    verification_status,
)
from manipulation_execution.pick_goal import pick_goal_rejection, remaining_budget_s

ACTION = "/test/grasp_pipeline/execute_pick"
FACTORY_CALLS = []


class SlowExecutor(Executor):
    """Waits on each frame until cancelled, when ``slow`` is set."""

    slow = False

    def execute(self, segment, *, carrying, halted, cancelled):
        if self.slow:
            ended = time.monotonic() + 10.0
            while time.monotonic() < ended:
                if cancelled():
                    return MotionOutcome(MotionState.CANCELLED, "cancelled", self.position.copy())
                time.sleep(0.01)
        return super().execute(segment, carrying=carrying, halted=halted, cancelled=cancelled)


def make_backends(*, node, grasp_execution, description):
    executor = SlowExecutor()
    executor.slow = bool(grasp_execution.get("test_slow"))
    backends = GraspBackends(Source(), Proposer(), StreamingPlanner(), executor, Effector())
    FACTORY_CALLS.append((node, grasp_execution, description, backends))
    return backends


WARMUP = {"ready": threading.Event(), "attempts": 0}


def make_warming_backends(*, node, grasp_execution, description):
    """Backends whose warmup fails until the test lets it succeed."""

    def prepare():
        WARMUP["attempts"] += 1
        if not WARMUP["ready"].is_set():
            raise RuntimeError("models not loaded yet")

    backends = make_backends(node=node, grasp_execution=grasp_execution, description=description)
    return replace(backends, prepare=prepare)


def configuration(**extra):
    return {"backend": f"{__name__}:make_backends", "pipeline": {"max_attempts": 1}, **extra}


def identity(config):
    return delegated_executor_identity(name="grasp_pipeline", endpoint_name=ACTION, configuration=config)


def goal(config, *, now_s, mode=PickObject.Goal.MODE_EXECUTE, query="red bottle"):
    message = PickObject.Goal()
    message.target_query = query
    message.mode = mode
    message.supervised_direct = True
    message.timeout_sec = 30.0
    fill_delegated_executor_identity(message.expected_executor, identity(config))
    binding = message.dispatch_binding
    binding.schema_version, binding.task_id, binding.root_task_id = 1, "task-1", "task-1"
    binding.expected_registry_epoch, binding.expected_registry_generation = "epoch-1", 1
    binding.expected_registry_digest = "digest-1"
    binding.task_budget.schema_version = 1
    binding.task_budget.started_at.sec = int(now_s) - 1
    binding.task_budget.deadline.sec = int(now_s) + 60
    return message


def test_admission_names_the_first_broken_rule():
    config = configuration()
    now = 1000.0
    assert pick_goal_rejection(goal(config, now_s=now), identity(config), now) == ""
    assert remaining_budget_s(goal(config, now_s=now), now) == pytest.approx(30.0)
    cases = {
        "empty target_query": lambda g: setattr(g, "target_query", " "),
        "expected_executor": lambda g: setattr(g.expected_executor, "configuration_digest", "0" * 64),
        "dispatch_nonce": lambda g: setattr(g.dispatch_binding, "dispatch_nonce", "nonce"),
        "dispatch_binding is incomplete": lambda g: setattr(g.dispatch_binding, "expected_registry_digest", ""),
        "already expired": lambda g: setattr(g.dispatch_binding.task_budget.deadline, "sec", int(now) - 1),
        "unknown mode": lambda g: setattr(g, "mode", 9),
    }
    for reason, breaks in cases.items():
        message = goal(config, now_s=now)
        breaks(message)
        assert reason in pick_goal_rejection(message, identity(config), now)
    delegated = goal(config, now_s=now)
    delegated.supervised_direct, delegated.dispatch_binding.dispatch_nonce = False, "nonce-1"
    assert pick_goal_rejection(delegated, identity(config), now) == ""


def test_pipeline_options_are_checked_and_verification_follows_the_close():
    assert pipeline_config({"pipeline": {"max_moved_replans": 0}}).max_moved_replans == 0
    with pytest.raises(ValueError, match="unknown keys"):
        pipeline_config({"pipeline": {"retries": 3}})
    assert verification_status(PickOutcome(completed_phases=["plan"])) == PickObject.Result.VERIFICATION_NOT_RUN
    for held, status in ((True, "SUCCESS"), (False, "FAILED"), (None, "UNCERTAIN")):
        outcome = PickOutcome(completed_phases=["close"], held=held)
        assert verification_status(outcome) == getattr(PickObject.Result, f"VERIFICATION_{status}")


@pytest.fixture
def ros():
    rclpy.init()
    yield
    rclpy.shutdown()


def serve(config):
    node = GraspPipelineNode(
        parameter_overrides=[
            Parameter("action_name", value=ACTION),
            Parameter("grasp_execution_json", value=json.dumps(config)),
            Parameter("interface_description_json", value=json.dumps({"robot": {"id": "x2"}})),
        ]
    )
    client_node = Node("grasp_pipeline_test_client")
    client = ActionClient(client_node, PickObject, ACTION)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.add_node(client_node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    assert client.wait_for_server(timeout_sec=10.0)
    return node, client_node, client, executor, thread


def stop(node, client_node, executor, thread):
    executor.shutdown()
    thread.join(5.0)
    node.destroy_node()
    client_node.destroy_node()


def wait(future, seconds=20.0):
    ended = time.monotonic() + seconds
    while not future.done() and time.monotonic() < ended:
        time.sleep(0.01)
    assert future.done()
    return future.result()


def test_the_node_runs_a_goal_through_the_configured_backends(ros):
    FACTORY_CALLS.clear()
    config = configuration()
    node, client_node, client, executor, thread = serve(config)
    try:
        assert FACTORY_CALLS[0][0] is node and FACTORY_CALLS[0][1] == config
        assert FACTORY_CALLS[0][2] == {"robot": {"id": "x2"}}
        feedback = []
        now = node.get_clock().now().nanoseconds / 1e9
        handle = wait(client.send_goal_async(goal(config, now_s=now), feedback_callback=feedback.append))
        assert handle.accepted
        result = wait(handle.get_result_async()).result
        assert result.success and result.error_code == ""
        assert result.verification_status == PickObject.Result.VERIFICATION_SUCCESS
        assert result.completed_phases[-4:] == ["grasp", "close", "lift", "hold_lifted"]
        assert result.candidate_index == 1 and result.attempts == 1
        assert result.actual_executor.configuration_digest == identity(config)["configuration_digest"]
        assert json.loads(result.pipeline_timings_json)["plan"] == {"planner": "streaming"}
        phases = [message.feedback.phase for message in feedback]
        assert phases[:3] == ["observe", "perceive", "plan"] and "approach" in phases

        plan_only = goal(config, now_s=now, mode=PickObject.Goal.MODE_PLAN_ONLY)
        result = wait(wait(client.send_goal_async(plan_only)).get_result_async()).result
        assert result.success and result.verification_status == PickObject.Result.VERIFICATION_NOT_RUN

        foreign = goal(configuration(other=1), now_s=now)
        assert not wait(client.send_goal_async(foreign)).accepted
    finally:
        stop(node, client_node, executor, thread)


def test_a_cancelled_goal_stops_the_arm_and_ends_cancelled(ros):
    config = configuration(test_slow=True)
    node, client_node, client, executor, thread = serve(config)
    try:
        started = threading.Event()

        def on_feedback(message):
            if message.feedback.phase == "approach":
                started.set()

        now = node.get_clock().now().nanoseconds / 1e9
        handle = wait(client.send_goal_async(goal(config, now_s=now), feedback_callback=on_feedback))
        assert started.wait(10.0)
        wait(handle.cancel_goal_async())
        response = wait(handle.get_result_async())
        assert response.status == 5  # GoalStatus.STATUS_CANCELED
        assert response.result.error_code == "CANCELED" and not response.result.success
        backends = FACTORY_CALLS[-1]
        assert backends[1]["test_slow"]
        # One goal at a time: the node takes the next goal once this one ended.
        quick = wait(client.send_goal_async(goal(config, now_s=now, mode=PickObject.Goal.MODE_OBSERVE_ONLY)))
        assert quick.accepted and wait(quick.get_result_async()).result.success
    finally:
        stop(node, client_node, executor, thread)


def test_a_robot_that_always_puts_back_does_so_unasked(ros):
    config = configuration(pipeline={"max_attempts": 1, "release_after_success": True})
    node, client_node, client, executor, thread = serve(config)
    try:
        now = node.get_clock().now().nanoseconds / 1e9
        for mode in (PickObject.Goal.MODE_OBSERVE_ONLY, PickObject.Goal.MODE_PLAN_ONLY):
            dry_run = goal(config, now_s=now, mode=mode)
            result = wait(wait(client.send_goal_async(dry_run)).get_result_async()).result
            assert result.success and not result.released_after_success
        message = goal(config, now_s=now)
        assert not message.release_after_success
        result = wait(wait(client.send_goal_async(message)).get_result_async()).result
        assert result.success and result.released_after_success
        moves = [name for name, _, _ in FACTORY_CALLS[-1][3].executor.log]
        # Lifted, held, then put back where it came from and the hand opened.
        assert moves[-3:] == ["retreat_lift", "retreat_grasp", "retreat_approach"]
        assert FACTORY_CALLS[-1][3].effector.log[-1] is EffectorAction.RELEASE
    finally:
        stop(node, client_node, executor, thread)


def test_goals_are_refused_until_the_backends_are_warm(ros, monkeypatch):
    monkeypatch.setattr("manipulation_execution.grasp_pipeline_node.PREPARE_RETRY_S", 0.05)
    WARMUP["ready"].clear()
    WARMUP["attempts"] = 0
    config = configuration(backend=f"{__name__}:make_warming_backends")
    node, client_node, client, executor, thread = serve(config)
    try:
        ended = time.monotonic() + 5.0
        while WARMUP["attempts"] < 2 and time.monotonic() < ended:
            time.sleep(0.01)
        now = node.get_clock().now().nanoseconds / 1e9
        assert WARMUP["attempts"] >= 2
        assert not wait(client.send_goal_async(goal(config, now_s=now))).accepted
        WARMUP["ready"].set()
        ended = time.monotonic() + 5.0
        while not node._ready.is_set() and time.monotonic() < ended:
            time.sleep(0.01)
        handle = wait(client.send_goal_async(goal(config, now_s=now)))
        assert handle.accepted and wait(handle.get_result_async()).result.success
    finally:
        stop(node, client_node, executor, thread)


def test_a_robot_without_a_backend_keeps_the_primitive_executor(ros):
    with pytest.raises(ValueError, match="pick_executor_node"):
        GraspPipelineNode(parameter_overrides=[Parameter("grasp_execution_json", value="{}")])


def test_terminal_record_keeps_each_attempts_watch_summary_and_grip():
    outcome = PickOutcome(error_code="GRASP_VERIFICATION_FAILED")
    outcome.report.update(
        plan={"arm_side": "right"},
        watch=[
            {
                "mode": "stop",
                "imu": False,
                "frames": 27,
                "votes": {"still": 0, "moved": 0, "unknown": 27},
                "skipped": 1,
                "config": {"large": True},
                "crop": [0, 1, 2, 3],
                "host": {"mode": "stop", "fault": None},
            }
        ],
        executions=[
            {
                "phases": [
                    {"name": "approach", "state": "succeeded", "reason": ""},
                    {"name": "close", "state": "succeeded", "gripped": True, "contacts": {"thumb": {"by": "lag"}}},
                    {"name": "lift", "state": "grip_lost", "reason": "the contact fingers no longer press"},
                ],
                "retreated": True,
            }
        ],
    )
    record = json.loads(json.dumps(terminal_record(outcome)))
    assert record["event"] == "grasp_pipeline_terminal" and record["arm_side"] == "right"
    assert record["watch"] == [
        {
            "mode": "stop",
            "imu": False,
            "frames": 27,
            "votes": {"still": 0, "moved": 0, "unknown": 27},
            "skipped": 1,
            "host": {"mode": "stop", "fault": None},
        }
    ]
    assert record["grip"] == [
        [
            {"name": "close", "state": "succeeded", "gripped": True, "contacts": {"thumb": {"by": "lag"}}},
            {"name": "lift", "state": "grip_lost", "reason": "the contact fingers no longer press"},
        ]
    ]


def test_terminal_record_without_watch_or_execution_is_empty_lists():
    record = terminal_record(PickOutcome(error_code="NO_EXECUTABLE_CANDIDATE"))
    assert record["watch"] == [] and record["grip"] == [] and record["arm_side"] == ""
