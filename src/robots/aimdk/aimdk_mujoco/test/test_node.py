"""Real ROS node behavior with a fake simulator; no physics assets or viewer required."""

import math
import os
import time
from concurrent.futures import Future
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import rclpy
from aimdk_mujoco import node as wrapper
from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import Trigger

from ibrobot_msgs.action import ExecuteSimulationPolicy

SUCCEEDED, CANCELED, ABORTED = 4, 5, 6


@pytest.fixture
def node(monkeypatch):
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID"), "Root conftest.py ROS domain isolation must be loaded"
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(wrapper.time, "monotonic", lambda: clock.now)

    def submit(function, *args):
        future = Future()
        try:
            future.set_result(function(*args))
        except Exception as exc:
            future.set_exception(exc)
        return future

    monkeypatch.setattr(
        wrapper,
        "PolicyLoader",
        Mock(return_value=Mock(submit=Mock(side_effect=submit), busy=False, close=Mock(return_value=True))),
    )
    truth = {
        "time": 0.52,
        "position": [1.0, 2.0, 0.8],
        "quaternion_wxyz": [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
        "linear_velocity_world": [1.0, 0.0, 0.0],
        "angular_velocity_body": [0.1, 0.2, 0.3],
    }
    sim = SimpleNamespace(
        controller="mc_rl_velocity",
        session=Mock(),
        policy_steps=0,
        execution_running=False,
        motion_running=False,
        selection_finished=False,
        is_standing=Mock(return_value=True),
        motion=SimpleNamespace(frame=7, end=9),
        ground_truth=lambda: truth,
        pelvis_id=0,
        joint_names=["joint_b", "joint_a"],
        qpos_indices=[1, 0],
        qvel_indices=[0, 1],
        data=SimpleNamespace(
            time=0.5,
            xmat=[[0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
            qpos=[0.1, 0.2],
            qvel=[0.3, 0.4],
        ),
    )

    def pause():
        sim.execution_running = False
        sim.motion_running = False

    def step(_command):
        if sim.execution_running and sim.session is not None and (sim.controller != "bmimic" or sim.motion_running):
            sim.policy_steps += 1
            sim.data.time += 0.02
            if sim.controller == "bmimic":
                sim.motion.frame += 1
                sim.motion_running = sim.motion.frame <= sim.motion.end

    def select(config, motion, reset):
        assert reset is False, "Reset selections must use prepare_policy/activate_policy"
        sim.controller = config.stem
        sim.execution_running = True
        sim.motion_running = sim.controller == "bmimic"

    def prepare(config, motion):
        return SimpleNamespace(controller=config.stem, motion=motion, session=object())

    def activate(candidate, *, transition=False):
        sim.controller = candidate.controller
        sim.session = candidate.session
        sim.policy_steps = 0
        sim.motion.frame = 7
        sim.selection_finished = False
        sim.execution_running = True
        sim.motion_running = sim.controller == "bmimic"

    sim.pause_execution = Mock(side_effect=pause)
    sim.step = Mock(side_effect=step)
    sim.select_policy = Mock(side_effect=select)
    sim.prepare_policy = Mock(side_effect=prepare)
    sim.activate_policy = Mock(side_effect=activate)
    rclpy.init(args=[])
    try:
        instance = wrapper.X2Node(sim, config_dir="/unused/controllers")
        try:
            instance.wall_clock = clock
            monkeypatch.setattr(instance, "close", Mock(wraps=instance.close))
            monkeypatch.setattr(instance, "get_logger", Mock(return_value=Mock()))
            for publisher in ("_odom_pub", "_joints_pub", "_imu_pub", "_clock_pub", "_diagnostics_pub"):
                monkeypatch.setattr(instance, publisher, Mock())
            yield instance
        finally:
            if not instance.close.called:
                if instance._goal is None:
                    instance._reserved = False
                instance.close()
            instance.destroy_node()
    finally:
        rclpy.shutdown()


@pytest.fixture
def action_api(node):
    executor = SingleThreadedExecutor()
    client_node = rclpy.create_node("simulation_policy_test_client")
    executor.add_node(node)
    executor.add_node(client_node)
    client = ActionClient(client_node, ExecuteSimulationPolicy, "/simulation/execute_policy")
    services = {name: client_node.create_client(Trigger, f"/simulation/{name}") for name in ("stand", "stop", "start")}

    def wait(predicate):
        deadline = time.perf_counter() + 5.0
        while not predicate():
            assert time.perf_counter() < deadline, "ROS callback did not complete within 5 seconds"
            executor.spin_once(timeout_sec=0.01)

    def result(future):
        wait(future.done)
        return future.result()

    def send(controller="mc_rl_velocity", motion="", duration=0.06, reset=True, feedback=None, transition=False):
        return result(
            client.send_goal_async(
                ExecuteSimulationPolicy.Goal(
                    controller=controller,
                    motion=motion,
                    duration_s=duration,
                    reset_before_start=reset,
                    transition_from_current=transition,
                ),
                feedback_callback=None if feedback is None else lambda msg: feedback.append(msg.feedback),
            )
        )

    def service(name):
        return result(services[name].call_async(Trigger.Request()))

    def tick_until_prepared():
        for _ in range(2):
            node.tick()
            if node._prepared:
                return
        pytest.fail("Policy did not activate within two deterministic ticks")

    try:
        assert client.wait_for_server(timeout_sec=5.0)
        for service_client in services.values():
            assert service_client.wait_for_service(timeout_sec=5.0)
        yield SimpleNamespace(
            client=client,
            send=send,
            result=result,
            wait=wait,
            service=service,
            tick_until_prepared=tick_until_prepared,
        )
    finally:
        if not node.close.called:
            node._finish(False, "Test teardown")
            wait(lambda: not node._reserved)
        client.destroy()
        client_node.destroy_node()
        executor.shutdown()


def test_action_success_reserves_until_result_and_pauses(node, action_api):
    feedback = []
    goal = action_api.send(duration=0.041, reset=False, feedback=feedback)
    assert goal.accepted
    assert node._reserved and not node._prepared
    assert node._goal is not None
    node.sim.select_policy.assert_not_called()
    result = goal.get_result_async()
    assert not result.done()
    assert not action_api.send().accepted
    node.sim.policy_steps = 20
    for _ in range(3):
        node.tick()
    assert node._reserved and node._done.done()
    response = node._on_start(Trigger.Request(), Trigger.Response())
    assert not response.success and "No active goal" in response.message
    reply = action_api.result(result)
    assert reply.status == SUCCEEDED
    assert reply.result.success
    assert reply.result.completed_steps == 3
    assert reply.result.message == "Completed; physics paused, not active balance"
    assert node.sim.selection_finished
    assert not node.sim.execution_running and not node.sim.motion_running
    node.sim.pause_execution.assert_called_once()
    assert node._goal is None and node._done is None and not node._reserved
    node.sim.select_policy.assert_called_once_with(node.config_dir / "mc_rl_velocity.yaml", None, False)
    action_api.wait(lambda: bool(feedback))
    assert feedback[0].state == "running"
    assert feedback[0].completed_steps == 0
    assert feedback[0].total_steps == 3


def test_action_prepared_cancel_never_integrates_again(node, action_api):
    goal = action_api.send()
    assert goal.accepted
    result = goal.get_result_async()
    action_api.tick_until_prepared()
    assert node._prepared and node.sim.policy_steps == 1
    canceled = action_api.result(goal.cancel_goal_async())
    assert len(canceled.goals_canceling) == 1
    assert node._goal.is_cancel_requested
    before = (node.sim.policy_steps, node.sim.data.time)
    node.tick()
    reply = action_api.result(result)
    assert reply.status == CANCELED
    assert not reply.result.success
    assert reply.result.completed_steps == 1
    assert reply.result.message == "Canceled; physics paused, not active balance"
    assert (node.sim.policy_steps, node.sim.data.time) == before
    assert not node.sim.execution_running and not node.sim.motion_running
    assert node.sim.prepare_policy.call_count == 1
    assert node.sim.activate_policy.call_count == 1
    node.sim.select_policy.assert_not_called()
    assert not node.sim.selection_finished
    node.tick()
    assert (node.sim.policy_steps, node.sim.data.time) == before
    assert not node._reserved and node._goal is None


def test_action_runtime_fault_aborts_and_latches(node, action_api):
    goal = action_api.send()
    assert goal.accepted
    result = goal.get_result_async()
    action_api.tick_until_prepared()
    node.sim.step.side_effect = RuntimeError("fallen")
    node.tick()
    reply = action_api.result(result)
    assert reply.status == ABORTED
    assert not reply.result.success and reply.result.completed_steps == 1
    assert reply.result.message == node._fault == "Physics fault (latched): fallen"
    assert not node.sim.execution_running and not node.sim.motion_running
    assert not node.sim.selection_finished
    assert not node._reserved and node._goal is None
    node.get_logger().error.assert_called_once()
    msg = Twist()
    msg.linear.x = 0.2
    node._on_command(msg)
    assert node._current_command() == (0.0, 0.0, 0.0)
    assert not action_api.send().accepted
    for name in ("start", "stop", "stand"):
        response = action_api.service(name)
        assert not response.success and response.message == node._fault
    before = node.sim.step.call_count
    node.wall_clock.now += 2.0
    node.tick()
    assert node.sim.step.call_count == before
    status = node._diagnostics_pub.publish.call_args.args[0].status[0]
    assert status.level == DiagnosticStatus.ERROR and status.message == node._fault


def test_explicit_shutdown_aborts_active_action_and_pauses(node, action_api):
    goal = action_api.send()
    assert goal.accepted
    result = goal.get_result_async()
    action_api.tick_until_prepared()
    before = (node.sim.policy_steps, node.sim.data.time)
    node.close()
    assert not node.sim.execution_running and not node.sim.motion_running
    assert not node.sim.selection_finished
    assert (node.sim.policy_steps, node.sim.data.time) == before
    assert not node._reserved and node._goal is None
    node._loader.close.assert_called_once_with()
    reply = action_api.result(result)
    assert reply.status == ABORTED
    assert not reply.result.success and reply.result.message == "Simulation shutting down"
    assert reply.result.completed_steps == 1
