"""Real ROS node behavior with a fake simulator; no physics assets or viewer required."""

import math
import os
import threading
import time
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import rclpy
from aimdk_mujoco import node as wrapper
from aimdk_mujoco.policy_loader import PolicyLoader
from diagnostic_msgs.msg import DiagnosticStatus
from geometry_msgs.msg import Twist
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor
from std_srvs.srv import Trigger

from ibrobot_msgs.action import ExecuteSimulationPolicy

# action_msgs is a transitive dependency of rclpy, not a declared test dependency.
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
                # Velocity-only fixtures reserve admission without a real action.
                if instance._goal is None:
                    instance._reserved = False
                instance.close()
            instance.destroy_node()
    finally:
        rclpy.shutdown()


@pytest.fixture
def velocity_node(node):
    # Admit velocity without inventing a goal handle that teardown would finish.
    node._reserved = node._prepared = True
    node.sim.execution_running = True
    return node


@pytest.fixture
def action_api(node, request):
    executor = rclpy.get_global_executor() if getattr(request, "param", None) == "global" else SingleThreadedExecutor()
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
        # Release an unfinished goal while its execute coroutine still has an executor.
        if not node.close.called:
            node._finish(False, "Test teardown")
            wait(lambda: not node._reserved)
        client.destroy()
        client_node.destroy_node()
        executor.shutdown()


def test_initial_stand_and_wall_timeout(velocity_node):
    node = velocity_node
    assert node._current_command() == (0.0, 0.0, 0.0)
    msg = Twist()
    msg.linear.x = 0.2
    node._on_command(msg)
    node.wall_clock.now = 10.499
    assert node._current_command() == (0.2, 0.0, 0.0)
    node.wall_clock.now = 10.5
    assert node._current_command() == (0.0, 0.0, 0.0)
    assert node.sim.data.time == 0.5


def test_preview_rejects_motion_without_latching_and_reports_no_active_balance(velocity_node, tmp_path):
    node = velocity_node
    node.sim.session = None
    node.sim.policy_path = tmp_path / "policy.onnx"
    node.sim.preview_message = "Model preview paused; place authorized weights for active balance"
    msg = Twist()
    msg.linear.x = 0.2
    node._on_command(msg)
    assert node._current_command() == (0.0, 0.0, 0.0)
    assert node._fault is None
    node.get_logger().warning.assert_called_once_with(f"{node.sim.preview_message}; command rejected")
    node.tick()
    status = node._diagnostics_pub.publish.call_args.args[0].status[0]
    assert status.level == DiagnosticStatus.WARN
    assert status.message == "Physics paused; submit execute_policy or resume active goal"
    reply = node._on_stand(Trigger.Request(), Trigger.Response())
    assert reply.success and "Physics paused, not active balance" in reply.message
    node.sim.policy_path = Path(__file__)  # Existing file; no policy assets are needed.
    node.sim.execution_running = True
    node._on_command(msg)
    assert node._current_command() == (0.2, 0.0, 0.0)


@pytest.mark.parametrize("command", [(-0.3, -0.3, -0.2), (0.5, 0.3, 0.2)])
def test_inclusive_bounds_and_refresh(velocity_node, command):
    node = velocity_node
    msg = Twist()
    msg.linear.x, msg.linear.y, msg.angular.z = command
    node._on_command(msg)
    node.wall_clock.now += 0.4
    node._on_command(msg)
    node.wall_clock.now += 0.4
    assert node._current_command() == command


@pytest.mark.parametrize(
    "component,value",
    [
        ("linear.x", -0.30001),
        ("linear.x", 0.50001),
        ("linear.y", -0.30001),
        ("linear.y", 0.30001),
        ("angular.z", -0.20001),
        ("angular.z", 0.20001),
        ("linear.z", 0.1),
        ("angular.x", 0.1),
        ("angular.y", 0.1),
    ]
    + [
        (component, value)
        for component in ("linear.x", "linear.y", "linear.z", "angular.x", "angular.y", "angular.z")
        for value in (math.nan, math.inf, -math.inf)
    ],
)
def test_invalid_commands_reject_and_clear(velocity_node, component, value):
    node = velocity_node
    valid = Twist()
    valid.linear.x = 0.2
    node._on_command(valid)
    assert node._current_command() == (0.2, 0.0, 0.0)
    msg = Twist()
    field, axis = component.split(".")
    setattr(getattr(msg, field), axis, value)
    node._on_command(msg)
    assert node._current_command() == (0.0, 0.0, 0.0)
    assert node._command_time is None
    node.get_logger().warning.assert_called_once()


@pytest.mark.parametrize("controller", ["mc_rl_velocity", "bmimic"])
def test_stand_request_pauses_all_physics_without_active_balance(velocity_node, controller):
    node = velocity_node
    msg = Twist()
    msg.linear.x = 0.2
    node._on_command(msg)
    node.sim.controller = controller
    node.sim.motion_running = controller == "bmimic"
    before = (node.sim.policy_steps, node.sim.data.time)
    response = node._on_stand(Trigger.Request(), Trigger.Response())
    assert response.success and "Physics paused, not active balance" in response.message
    node.sim.pause_execution.assert_called_once()
    assert not node.sim.execution_running and not node.sim.motion_running
    node.tick()
    assert (node.sim.policy_steps, node.sim.data.time) == before
    node.sim.step.assert_called_once_with((0.0, 0.0, 0.0))
    node._clock_pub.publish.assert_called_once()


@pytest.mark.parametrize("simulation_time", [0.5, math.nan])
def test_fault_latches_and_services_fail(node, simulation_time):
    node.sim.data.time = simulation_time
    node.sim.step.side_effect = RuntimeError("fallen")
    node.tick()
    assert "fallen" in node._fault
    node.get_logger().error.assert_called_once()
    node._clock_pub.publish.assert_not_called()
    assert node._diagnostics_pub.publish.call_args.args[0].status[0].level == DiagnosticStatus.ERROR
    msg = Twist()
    msg.linear.x = 0.2
    node._on_command(msg)
    response = node._on_stand(Trigger.Request(), Trigger.Response())
    assert not response.success
    assert "fallen" in response.message
    node.wall_clock.now += 2.0
    node.tick()
    assert node._current_command() == (0.0, 0.0, 0.0)
    node.sim.step.assert_called_once()
    assert node._diagnostics_pub.publish.call_count == 2


def test_ground_truth_rotates_twist_and_marks_acceleration_unavailable(node):
    node._publish_state()
    odom = node._odom_pub.publish.call_args.args[0]
    assert (odom.header.frame_id, odom.child_frame_id) == ("world", "pelvis")
    assert odom.pose.pose.position.z == 0.8
    assert odom.pose.pose.orientation.w == pytest.approx(math.sqrt(0.5))
    assert odom.twist.twist.linear.x == pytest.approx(0.0)
    assert odom.twist.twist.linear.y == pytest.approx(-1.0)
    assert odom.twist.twist.angular.z == 0.3
    joints = node._joints_pub.publish.call_args.args[0]
    assert joints.name == ["joint_b", "joint_a"]
    assert list(joints.position) == [0.2, 0.1]
    assert list(joints.velocity) == [0.3, 0.4]
    assert not joints.effort
    imu = node._imu_pub.publish.call_args.args[0]
    assert imu.header.frame_id == "pelvis"
    assert imu.orientation == odom.pose.pose.orientation
    assert imu.angular_velocity == odom.twist.twist.angular
    assert imu.linear_acceleration_covariance[0] == -1.0
    clock = node._clock_pub.publish.call_args.args[0]
    assert clock.clock == odom.header.stamp == joints.header.stamp == imu.header.stamp
    assert clock.clock.sec == 0
    assert clock.clock.nanosec == 520_000_000


def test_dance_rejects_velocity_and_start_stop_are_explicit(node):
    node.sim.controller = "bmimic"
    node.sim.motion_running = False
    msg = Twist()
    msg.linear.x = 0.2
    node._on_command(msg)
    assert node._command_time is None
    assert node._current_command() == (0, 0, 0)
    node.tick()
    assert node._diagnostics_pub.publish.call_args.args[0].status[0].level == DiagnosticStatus.WARN
    response = node._on_start(Trigger.Request(), Trigger.Response())
    assert not response.success and "No active goal" in response.message
    assert not node.sim.execution_running and not node.sim.motion_running
    response = node._on_stand(Trigger.Request(), Trigger.Response())
    assert response.success and "not active balance" in response.message
    node.sim.pause_execution.assert_called_once()
    node._fault = "latched"
    response = node._on_start(Trigger.Request(), Trigger.Response())
    assert not response.success and response.message == "latched"


@pytest.mark.parametrize(
    "controller,reserved,prepared,running",
    [
        ("mc_rl_velocity", False, False, True),
        ("mc_rl_velocity", True, False, True),
        ("mc_rl_velocity", True, True, False),
        ("bmimic", True, True, True),
    ],
)
def test_velocity_requires_active_prepared_running_goal(node, controller, reserved, prepared, running):
    node.sim.controller = controller
    node._reserved, node._prepared = reserved, prepared
    node.sim.execution_running = running
    msg = Twist()
    msg.linear.x = 0.2
    node._on_command(msg)
    assert node._current_command() == (0.0, 0.0, 0.0)
    assert node._command_time is None
    node.get_logger().warning.assert_called_once()


@pytest.mark.parametrize(
    "controller,motion,duration,steps",
    [
        ("mc_rl_velocity", "", 0.041, 3),
        ("bmimic", "fan_dance", 0.0, 3),
    ],
)
def test_action_success_reserves_until_result_and_pauses(node, action_api, controller, motion, duration, steps):
    feedback = []
    goal = action_api.send(controller, motion, duration, reset=False, feedback=feedback)
    assert goal.accepted
    assert node._reserved and not node._prepared
    assert node._goal is not None
    node.sim.select_policy.assert_not_called()
    result = goal.get_result_async()
    assert not result.done()
    # The reservation rejects a second goal even before deferred preparation.
    assert not action_api.send().accepted
    node.sim.policy_steps = 20  # Preparation may reset/reuse a preexisting simulation counter.
    for _ in range(steps):
        node.tick()
    # _finish has paused physics, but _execute has not yet consumed its Future.
    assert node._reserved and node._done.done()
    response = node._on_start(Trigger.Request(), Trigger.Response())
    assert not response.success and "No active goal" in response.message
    reply = action_api.result(result)
    assert reply.status == SUCCEEDED
    assert reply.result.success
    assert reply.result.completed_steps == steps
    assert reply.result.message == "Completed; physics paused, not active balance"
    assert node.sim.selection_finished
    assert not node.sim.execution_running and not node.sim.motion_running
    node.sim.pause_execution.assert_called_once()
    assert node._goal is None and node._done is None and not node._reserved
    node.sim.select_policy.assert_called_once_with(node.config_dir / f"{controller}.yaml", motion or None, False)
    action_api.wait(lambda: bool(feedback))
    assert feedback[0].state == "running"
    assert feedback[0].completed_steps == 0
    assert feedback[0].total_steps == steps
    before = (node.sim.policy_steps, node.sim.data.time)
    node.tick()
    assert (node.sim.policy_steps, node.sim.data.time) == before
    assert not action_api.service("start").success
    next_goal = action_api.send(duration=0.02)
    assert next_goal.accepted
    next_result = next_goal.get_result_async()
    action_api.tick_until_prepared()
    assert action_api.result(next_result).result.completed_steps == 1


def test_action_simultaneous_requests_reserve_exactly_one_goal(node, action_api):
    pending = [
        action_api.client.send_goal_async(
            ExecuteSimulationPolicy.Goal(controller="mc_rl_velocity", duration_s=0.02, reset_before_start=True)
        )
        for _ in range(2)
    ]
    goals = [action_api.result(future) for future in pending]
    accepted = [goal for goal in goals if goal.accepted]
    assert len(accepted) == 1
    assert node._reserved and not node._prepared
    node.sim.select_policy.assert_not_called()
    result = accepted[0].get_result_async()
    action_api.tick_until_prepared()
    assert action_api.result(result).status == SUCCEEDED
    node.sim.prepare_policy.assert_called_once()
    node.sim.activate_policy.assert_called_once()
    node.sim.select_policy.assert_not_called()
    assert not node._reserved


@pytest.mark.parametrize("controller,motion,duration", [("mc_rl_velocity", "", 0.06), ("bmimic", "fan_dance", 0.0)])
@pytest.mark.parametrize("prepared", [False, True])
def test_action_cancel_before_or_after_prepare_never_integrates_again(
    node, action_api, prepared, controller, motion, duration
):
    goal = action_api.send(controller, motion, duration)
    assert goal.accepted
    result = goal.get_result_async()
    if prepared:
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
    assert reply.result.completed_steps == int(prepared)
    assert reply.result.message == "Canceled; physics paused, not active balance"
    assert (node.sim.policy_steps, node.sim.data.time) == before
    assert not node.sim.execution_running and not node.sim.motion_running
    assert node.sim.prepare_policy.call_count == int(prepared)
    assert node.sim.activate_policy.call_count == int(prepared)
    node.sim.select_policy.assert_not_called()
    assert not node.sim.selection_finished
    node.tick()
    assert (node.sim.policy_steps, node.sim.data.time) == before
    assert not node._reserved and node._goal is None
    assert not action_api.service("start").success


@pytest.mark.parametrize(
    "controller,motion,duration",
    [
        ("mc_rl_velocity", "", 0.06),
        ("bmimic", "fan_dance", 0.0),
    ],
)
@pytest.mark.parametrize("pause_service", ["stand", "stop"])
def test_action_services_pause_and_resume_only_prepared_goal(
    node, action_api, controller, motion, duration, pause_service
):
    assert not action_api.service("start").success
    feedback = []
    goal = action_api.send(controller, motion, duration, feedback=feedback)
    assert goal.accepted
    result = goal.get_result_async()
    response = action_api.service("start")
    assert not response.success and "No active goal" in response.message
    action_api.tick_until_prepared()
    assert node._prepared and node.sim.policy_steps == 1
    if controller == "mc_rl_velocity":
        msg = Twist()
        msg.linear.x = 0.2
        node._on_command(msg)
        assert node._current_command() == (0.2, 0.0, 0.0)
    response = action_api.service(pause_service)
    assert response.success and "Physics paused, not active balance" in response.message
    assert node._current_command() == (0.0, 0.0, 0.0)
    before = (node.sim.policy_steps, node.sim.data.time)
    node.wall_clock.now += 0.21
    node.tick()
    action_api.wait(lambda: any(item.state == "paused" for item in feedback))
    paused = next(item for item in feedback if item.state == "paused")
    assert (paused.completed_steps, paused.total_steps) == (1, 3)
    assert (node.sim.policy_steps, node.sim.data.time) == before
    assert not result.done()
    assert not node.sim.execution_running and not node.sim.motion_running
    assert not action_api.send().accepted
    response = action_api.service("start")
    assert response.success and response.message == "Current execution resumed"
    assert node.sim.execution_running
    assert node.sim.motion_running == (controller == "bmimic")
    node.tick()
    node.tick()
    assert action_api.result(result).status == SUCCEEDED
    node.sim.prepare_policy.assert_called_once_with(node.config_dir / f"{controller}.yaml", motion or None)
    node.sim.activate_policy.assert_called_once()
    node.sim.select_policy.assert_not_called()
    assert not action_api.service("start").success


@pytest.mark.parametrize(
    "controller,motion,duration",
    [
        ("unknown", "", 1.0),
        ("", "", 1.0),
        ("mc_rl_velocity", "fan_dance", 1.0),
        ("mc_rl_velocity", "", 0.0),
        ("mc_rl_velocity", "", -0.01),
        ("mc_rl_velocity", "", 3600.01),
        ("mc_rl_velocity", "", math.nan),
        ("mc_rl_velocity", "", math.inf),
        ("mc_rl_velocity", "", -math.inf),
        ("bmimic", "", 0.0),
        ("bmimic", "fan_dance", 0.02),
        ("bmimic", "fan_dance", -0.02),
        ("bmimic", "fan_dance", math.nan),
        ("bmimic", "fan_dance", math.inf),
    ],
)
def test_action_invalid_goal_rejects_without_reserving(node, action_api, controller, motion, duration):
    assert not action_api.send(controller, motion, duration).accepted
    assert not node._reserved and node._goal is None and node._fault is None
    node.sim.select_policy.assert_not_called()
    assert action_api.send().accepted


@pytest.mark.parametrize("duration", [0.001, 3600.0])
def test_action_velocity_duration_boundaries_accept(action_api, duration):
    assert action_api.send(duration=duration).accepted


@pytest.mark.parametrize("error", [ValueError("unknown motion"), RuntimeError("policy unavailable")])
@pytest.mark.parametrize("stage", ["prepare_policy", "activate_policy", "select_policy"])
def test_action_preparation_failure_aborts_without_fault_and_next_goal_accepts(node, action_api, error, stage):
    reset = stage != "select_policy"
    failing = getattr(node.sim, stage)
    original = failing.side_effect
    failing.side_effect = error
    goal = action_api.send(reset=reset)
    assert goal.accepted
    result = goal.get_result_async()
    before = (node.sim.policy_steps, node.sim.data.time)
    node.tick()
    if reset:
        assert not node._prepared and not node._done.done()
        node.tick()
    reply = action_api.result(result)
    assert reply.status == ABORTED
    assert not reply.result.success and reply.result.completed_steps == 0
    assert reply.result.message == f"Policy preparation failed: {error}"
    assert node._fault is None and not node._reserved
    assert (node.sim.policy_steps, node.sim.data.time) == before
    assert not node.sim.execution_running and not node.sim.motion_running
    node.sim.pause_execution.assert_called_once()
    node.get_logger().error.assert_not_called()
    assert node.sim.activate_policy.call_count == int(stage == "activate_policy")
    assert not node.sim.selection_finished
    failing.side_effect = original
    next_goal = action_api.send(duration=0.02)
    assert next_goal.accepted
    next_result = next_goal.get_result_async()
    action_api.tick_until_prepared()
    assert action_api.result(next_result).status == SUCCEEDED


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


def test_action_prepares_without_live_writes_then_activates_on_next_tick(node, action_api):
    feedback = []
    node.sim.policy_steps = 17
    node.sim.selection_finished = True
    session = node.sim.session
    goal = action_api.send("bmimic", "fan_dance", 0.0, feedback=feedback)
    assert goal.accepted
    result = goal.get_result_async()
    before = (node.sim.policy_steps, node.sim.data.time, node.sim.motion.frame)
    node.tick()
    node.sim.prepare_policy.assert_called_once_with(node.config_dir / "bmimic.yaml", "fan_dance")
    node.sim.activate_policy.assert_not_called()
    node.sim.select_policy.assert_not_called()
    assert not node._prepared
    assert node.sim.selection_finished
    assert node.sim.controller == "mc_rl_velocity" and node.sim.session is session
    assert not node.sim.execution_running and not node.sim.motion_running
    assert (node.sim.policy_steps, node.sim.data.time, node.sim.motion.frame) == before
    action_api.wait(lambda: bool(feedback))
    assert feedback[0].state == "preparing"
    assert not action_api.service("start").success
    candidate = node._loading.result()
    node.tick()
    node.sim.activate_policy.assert_called_once_with(candidate)
    assert node.sim.session is candidate.session
    assert node._prepared and node.sim.policy_steps == 1
    assert not node.sim.selection_finished
    assert node.sim.controller == "bmimic" and node.sim.motion_running
    node.tick()
    node.tick()
    assert action_api.result(result).status == SUCCEEDED
    assert node.sim.selection_finished


@pytest.mark.parametrize("load_fails", [False, True])
def test_cancel_during_prepare_blocks_admission_until_done_and_discards_stale_candidate(node, action_api, load_fails):
    node._loader = PolicyLoader()
    started, release = threading.Event(), threading.Event()
    prepare = node.sim.prepare_policy.side_effect
    candidates = []

    def blocked_prepare(*args):
        candidates.append(prepare(*args))
        started.set()
        assert release.wait(5), "Test did not release policy preparation"
        if load_fails:
            raise ValueError("canceled policy failed to load")
        return candidates[0]

    node.sim.prepare_policy.side_effect = blocked_prepare
    try:
        feedback = []
        goal = action_api.send("bmimic", "fan_dance", 0.0, feedback=feedback)
        assert goal.accepted
        result = goal.get_result_async()
        before = (node.sim.policy_steps, node.sim.data.time)
        node.tick()
        assert started.wait(5)
        pending = node._loading
        assert not pending.done() and not node._prepared
        action_api.wait(lambda: bool(feedback))
        assert feedback[0].state == "preparing"
        node.tick()
        assert not action_api.service("start").success
        assert action_api.service("stop").success
        assert not action_api.send().accepted
        cancel = action_api.result(goal.cancel_goal_async())
        assert len(cancel.goals_canceling) == 1
        node.tick()
        reply = action_api.result(result)
        assert reply.status == CANCELED and reply.result.completed_steps == 0
        assert not reply.result.success and not node._reserved
        assert node._loading is pending and not pending.done()
        assert not pending.cancel() and node._loader.busy
        assert not action_api.send().accepted
        node.tick()
        assert (node.sim.policy_steps, node.sim.data.time) == before
        node.sim.activate_policy.assert_not_called()
        (stale,) = candidates
        node.sim.prepare_policy.assert_called_once_with(node.config_dir / "bmimic.yaml", "fan_dance")
        release.set()
        action_api.wait(lambda: not node._loader.busy)
        assert pending.done()
        if load_fails:
            assert isinstance(pending.exception(), ValueError)
        else:
            assert pending.result() is stale
        node.tick()
        node.sim.activate_policy.assert_not_called()
        assert node._fault is None
        node.sim.prepare_policy.side_effect = prepare
        next_goal = action_api.send(duration=0.02)
        assert next_goal.accepted and node._loading is None
        next_result = next_goal.get_result_async()
        node.tick()
        fresh = node._loading.result(timeout=5)
        assert fresh is not stale and fresh.controller == "mc_rl_velocity"
        node.sim.activate_policy.assert_not_called()
        node.tick()
        node.sim.activate_policy.assert_called_once_with(fresh)
        assert node.sim.session is fresh.session
        assert action_api.result(next_result).status == SUCCEEDED
        assert node.sim.selection_finished
    finally:
        release.set()
        assert node._loader.close(timeout=5)


@pytest.mark.parametrize("pause_service", ["stand", "stop"])
@pytest.mark.parametrize("reset", [False, True])
def test_stop_before_preparation_stays_paused_until_explicit_resume(node, action_api, pause_service, reset):
    goal = action_api.send(duration=0.06, reset=reset)
    result = goal.get_result_async()
    if reset:
        node.tick()
        assert not node._prepared
    assert action_api.service(pause_service).success
    before = (node.sim.policy_steps, node.sim.data.time)
    node.tick()
    assert node._prepared
    assert not node.sim.execution_running and not node.sim.motion_running
    assert (node.sim.policy_steps, node.sim.data.time) == before
    assert not result.done()
    assert action_api.service("start").success
    for _ in range(3):
        node.tick()
    assert action_api.result(result).status == SUCCEEDED


@pytest.mark.parametrize("action_api", ["global", "explicit"], indirect=True)
def test_close_delivers_active_action_result_and_pauses(node, action_api):
    goal = action_api.send()
    assert goal.accepted
    result = goal.get_result_async()
    action_api.tick_until_prepared()
    # Ensure the result request and the suspended execute coroutine are registered.
    barrier = action_api.service("stop")
    assert barrier.success
    assert action_api.service("start").success
    # A registered response callback rules out a result-request/discovery race.
    server_result = node._action._result_futures[bytes(goal.goal_id.uuid)]
    action_api.wait(lambda: bool(server_result._callbacks))
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


def test_close_rejects_new_goal_while_draining_result(node, action_api):
    goal = action_api.send()
    result = goal.get_result_async()
    action_api.tick_until_prepared()
    requests = []

    def submit_during_close(_future):
        requests.append(
            action_api.client.send_goal_async(
                ExecuteSimulationPolicy.Goal(controller="mc_rl_velocity", duration_s=0.02, reset_before_start=True)
            )
        )

    node._done.add_done_callback(submit_during_close)
    node.close()
    assert action_api.result(result).status == ABORTED
    assert len(requests) == 1
    assert not action_api.result(requests[0]).accepted


@pytest.mark.parametrize("load_fails", [False, True])
def test_close_bounds_pending_loader_and_never_activates_late_result(node, action_api, load_fails):
    node._loader = PolicyLoader()
    started, release = threading.Event(), threading.Event()
    prepare = node.sim.prepare_policy.side_effect

    def blocked_prepare(*args):
        started.set()
        assert release.wait(5), "Test did not release policy preparation"
        if load_fails:
            raise ValueError("late preparation failed")
        return prepare(*args)

    node.sim.prepare_policy.side_effect = blocked_prepare
    try:
        goal = action_api.send()
        result = goal.get_result_async()
        node.tick()
        assert started.wait(5)
        pending = node._loading
        before = (node.sim.policy_steps, node.sim.data.time, node.sim.session)
        start = time.perf_counter()
        node.close()
        assert time.perf_counter() - start < 1.5
        assert node._closing and node._action is None and node._loading is None
        assert not pending.done() and node._loader.busy
        node.get_logger().warning.assert_called_once_with(
            "Policy loader unfinished after 0.5s; abandoning daemon preparation without activation"
        )
        reply = action_api.result(result)
        assert reply.status == ABORTED and reply.result.completed_steps == 0
        assert reply.result.message == "Simulation shutting down"
        logger_calls = list(node.get_logger().mock_calls)
        step_calls = node.sim.step.call_count
        release.set()
        assert node._loader.close(timeout=5)
        assert pending.done()
        if load_fails:
            assert isinstance(pending.exception(), ValueError)
        else:
            assert pending.result() is not None
        node._tick_goal()
        node.tick()
        node.close()
        node.sim.activate_policy.assert_not_called()
        assert (node.sim.policy_steps, node.sim.data.time, node.sim.session) == before
        assert node.sim.step.call_count == step_calls
        assert node.get_logger().mock_calls == logger_calls
        assert not node.sim.execution_running
    finally:
        release.set()
        assert node._loader.close(timeout=5)


def test_transition_activates_without_reset(node, action_api):
    goal = action_api.send(reset=False, transition=True)
    result = goal.get_result_async()
    action_api.tick_until_prepared()
    assert node.sim.activate_policy.call_count == 1
    assert node.sim.activate_policy.call_args.kwargs == {"transition": True}
    node.sim.select_policy.assert_not_called()
    node.tick()
    node.tick()
    assert action_api.result(result).status == SUCCEEDED


@pytest.mark.parametrize("duration", [0.9, 30.1])
def test_recovery_rejects_invalid_duration(action_api, duration):
    assert not action_api.send("recover_stand", duration=duration, reset=False, transition=True).accepted


def test_recovery_rejects_reset_and_requires_transition(action_api):
    assert not action_api.send("recover_stand", duration=5.0).accepted
    assert not action_api.send("recover_stand", duration=5.0, reset=False).accepted
    assert not action_api.send(reset=True, transition=True).accepted


def test_recovery_requires_continuous_stability_and_blocks_velocity(node, action_api):
    goal = action_api.send("recover_stand", duration=5.0, reset=False, transition=True)
    result = goal.get_result_async()
    action_api.tick_until_prepared()
    msg = Twist()
    msg.linear.x = 0.4
    node._on_command(msg)
    assert node._current_command() == (0.0, 0.0, 0.0)
    for _ in range(30):
        node.tick()
    node.sim.is_standing.return_value = False
    node.tick()
    node.sim.is_standing.return_value = True
    for _ in range(49):
        node.tick()
    assert not node._done.done()
    node.tick()
    response = action_api.result(result)
    assert response.status == SUCCEEDED and response.result.success
    assert "settled" in response.result.message
    assert not node.sim.execution_running
    node.sim.prepare_policy.assert_called_once_with(node.config_dir / "mc_rl_velocity.yaml", None)


def test_recovery_times_out_without_fault(node, action_api):
    node.sim.is_standing.return_value = False
    goal = action_api.send("recover_stand", duration=1.0, reset=False, transition=True)
    result = goal.get_result_async()
    action_api.tick_until_prepared()
    for _ in range(49):
        node.tick()
    response = action_api.result(result)
    assert response.status == ABORTED and not response.result.success
    assert "timed out" in response.result.message
    assert node._fault is None and not node.sim.execution_running
