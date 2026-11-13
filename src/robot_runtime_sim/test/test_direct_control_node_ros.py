"""ROS-level behaviour of the direct-control node on the CPU toy platform: gate, cancel, stop, clock."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

rclpy = pytest.importorskip("rclpy")

from control_msgs.action import FollowJointTrajectory  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from trajectory_msgs.msg import JointTrajectoryPoint  # noqa: E402

from ibrobot_msgs.srv import MoveToConfiguration, SetRuntimeMode, StopRuntime  # noqa: E402
from robot_runtime_sim.direct_control.node import SimulationRuntime  # noqa: E402
from robot_runtime_sim.testing.direct_control_fakes import TOY_PROFILE as PROFILE  # noqa: E402
from robot_runtime_sim.testing.direct_control_fakes import ToyAdapter, wait_until  # noqa: E402

EXTENSION = "robot_runtime_sim.testing.direct_control_fakes:attach_toy_gate"


@pytest.fixture
def rig(request):
    rclpy.init()
    adapter = ToyAdapter()
    from rclpy.parameter import Parameter

    profile = str(PROFILE)
    mode = getattr(request, "param", None)
    if mode == "gated_by_profile":
        # How a caller selects a layer: its own copy of the profile lists it.
        document = yaml.safe_load(PROFILE.read_text())
        document["simulation"]["toy"]["extensions"] = [EXTENSION]
        profile = str(request.getfixturevalue("tmp_path") / "toy_with_extension.yaml")
        Path(profile).write_text(yaml.safe_dump(document))
    overrides = [Parameter("profile", value=profile), Parameter("platform", value="toy")]
    if mode == "gated":
        overrides.append(Parameter("extensions", value=[EXTENSION]))
    runtime = SimulationRuntime(adapter=adapter, node_name="toy_runtime", parameter_overrides=overrides)
    client = Node("toy_client")
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(runtime)
    executor.add_node(client)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    yield runtime, client, adapter
    executor.shutdown()
    spin.join(timeout=5.0)
    runtime.destroy_node()
    client.destroy_node()
    rclpy.shutdown()


def _call(node, srv, name, request, timeout=10.0):
    cli = node.create_client(srv, name)
    assert cli.wait_for_service(timeout_sec=10.0)
    future = cli.call_async(request)
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    return future.result()


def _gripper_goal(opening):
    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = ["a_grip"]
    goal.trajectory.points = [JointTrajectoryPoint(positions=[opening])]
    return goal


def _wait(future, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not future.done() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert future.done()
    return future.result()


@pytest.mark.parametrize("rig", ["gated", "gated_by_profile"], indirect=True)
def test_an_extension_gate_holds_motion_and_a_cancelled_gripper_goal_ends_cancelled(rig):
    runtime, client, adapter = rig
    core = runtime.core
    assert _call(client, SetRuntimeMode, "/runtime/set_mode", SetRuntimeMode.Request(mode="trajectory")).success
    request = MoveToConfiguration.Request()
    request.target_joint_state.name, request.target_joint_state.position = ["a_grip"], [0.0]
    refused = _call(client, MoveToConfiguration, "/motion/move_to_joint", request)
    assert refused.outcome == "REJECTED" and "toy gate" in refused.message
    runtime.toy_gate_open.add(core.generation)  # what an evaluation layer's admission would do

    action = ActionClient(client, FollowJointTrajectory, "/a_grip_trajectory_controller/follow_joint_trajectory")
    assert action.wait_for_server(timeout_sec=10.0)
    handle = _wait(action.send_goal_async(_gripper_goal(0.0)))
    assert handle.accepted
    deadline = time.monotonic() + 5.0
    while adapter.env.tick < 3 and time.monotonic() < deadline:
        time.sleep(0.01)
    _wait(handle.cancel_goal_async())
    result = _wait(handle.get_result_async())
    from action_msgs.msg import GoalStatus

    assert result.status == GoalStatus.STATUS_CANCELED
    at_cancel = adapter.env.tick
    time.sleep(0.5)
    assert adapter.env.tick <= at_cancel + 1  # gripper_ticks is 100: it was really withdrawn
    assert adapter.env.grip["a"] > 0.5  # far from the 0.0 target


def test_finish_then_stop_stops_physical_ticks(rig):
    _runtime, client, adapter = rig
    assert _call(client, SetRuntimeMode, "/runtime/set_mode", SetRuntimeMode.Request(mode="trajectory")).success
    from std_srvs.srv import Trigger

    assert _call(client, Trigger, "/simulation/finish_session", Trigger.Request()).success
    time.sleep(0.3)
    assert adapter.env.tick > 3  # finishing: hold ticks run out the platform clock
    stop = _call(client, StopRuntime, "/runtime/stop", StopRuntime.Request(policy="HOLD"))
    assert stop.success
    time.sleep(0.1)
    at_stop = adapter.env.tick
    time.sleep(0.5)
    assert adapter.env.tick == at_stop
    again = _call(client, Trigger, "/simulation/finish_session", Trigger.Request())
    assert not again.success  # stop latched: the operator must clear it first


def test_state_and_images_carry_the_simulated_source_stamp_and_a_new_session_is_republished(rig):
    """Header stamps are the observation's simulated time; a reset at the same tick is a new frame."""
    from geometry_msgs.msg import PoseStamped
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import Image

    runtime, client, adapter = rig
    core = runtime.core
    images, poses, clocks = [], [], []
    client.create_subscription(Image, "/toy/cam/image_raw", images.append, 10)
    client.create_subscription(PoseStamped, "/robot_status/a/ee_pose", poses.append, 10)
    client.create_subscription(Clock, "/clock", clocks.append, 10)

    def stamp(msg):
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def clock_s(msg):
        return msg.clock.sec + msg.clock.nanosec * 1e-9

    assert _call(client, SetRuntimeMode, "/runtime/set_mode", SetRuntimeMode.Request(mode="trajectory")).success
    request = MoveToConfiguration.Request()
    request.target_joint_state.name, request.target_joint_state.position = ["a_grip"], [0.5]
    assert _call(client, MoveToConfiguration, "/motion/move_to_joint", request).success  # ticks: time moves
    deadline = time.monotonic() + 5.0
    while (not images or not poses or not clocks) and time.monotonic() < deadline:
        time.sleep(0.02)
    observed = core.observation_source()[3]
    assert observed > 0.0 and observed == pytest.approx(core.sim_time_s())
    time.sleep(0.3)
    assert stamp(poses[-1]) == pytest.approx(observed) and stamp(images[-1]) == pytest.approx(observed)
    assert clock_s(clocks[-1]) == pytest.approx(observed)
    images.clear()
    clocks.clear()
    time.sleep(1.2)  # idle: no tick; the frame is re-sent and /clock repeats, both at the same time
    assert len(images) >= 2 and all(stamp(m) == pytest.approx(observed) for m in images)
    assert clocks and all(clock_s(m) == pytest.approx(observed) for m in clocks)
    # new session at the same tick 0 (the toy platform restarts its loop)
    adapter.platform.join()
    adapter.env.reset()
    adapter.env.session_mark = 7
    platform = adapter.platform.__class__(core, adapter.platform.codec, adapter.env, step_delay_s=0.02).start()
    adapter.platform = platform
    images.clear()
    deadline = time.monotonic() + 5.0
    while not images and time.monotonic() < deadline:
        time.sleep(0.02)
    assert images and bytes(images[0].data)[0] == 7
    assert stamp(images[0]) >= observed  # simulated time never goes back at a reset


def test_the_platform_result_goes_to_listeners_only(rig):
    runtime, client, adapter = rig
    heard = []
    runtime.add_platform_result_listener(heard.append)
    topics = {name for name, _types in client.get_topic_names_and_types()}
    assert "/evaluation/native_result" not in topics
    from robot_runtime_sim.direct_control.ports import PlatformResult

    core = runtime.core
    adapter.platform.join()  # the toy platform ends its session itself here
    core.session_end(
        {"session_id": core.session_id, "generation": core.generation}, PlatformResult(native_success=True)
    )
    assert heard and heard[-1]["native_success"] is True and heard[-1]["session_id"] == core.session_id


# ---------------------------------------------------------------- review round 2 (F1, F2, F3)


def _pose_request(command_id: str = "", x: float = 2.0):
    from ibrobot_msgs.srv import MoveToPose

    request = MoveToPose.Request()
    request.target_pose.position.x, request.target_pose.position.y, request.target_pose.position.z = x, 0.1, 0.1
    request.target_pose.orientation.w = 1.0
    request.velocity_scaling = 1.0  # 1.9 rad at 0.06 rad/tick: 32 ticks of 20 ms wall each
    request.target_group = "a"
    request.command_id = command_id
    return request


def test_a_gripper_goal_cancelled_before_its_submit_never_moves(rig):
    """F2 on the real watcher path: the watcher's cancel lands first; submit must not queue the command."""
    from action_msgs.msg import GoalStatus

    runtime, client, adapter = rig
    core = runtime.core
    assert _call(client, SetRuntimeMode, "/runtime/set_mode", SetRuntimeMode.Request(mode="trajectory")).success
    real_submit, seen = core.submit, {}

    def submit_after_the_watcher(cmd, timeout_s):  # barrier: hold the submit until the watcher cancelled
        seen["cmd"] = cmd
        assert wait_until(cmd.done.is_set)
        return real_submit(cmd, timeout_s)

    core.submit = submit_after_the_watcher
    action = ActionClient(client, FollowJointTrajectory, "/a_grip_trajectory_controller/follow_joint_trajectory")
    assert action.wait_for_server(timeout_sec=10.0)
    handle = _wait(action.send_goal_async(_gripper_goal(0.0)))
    assert handle.accepted and wait_until(lambda: "cmd" in seen)
    _wait(handle.cancel_goal_async())
    assert _wait(handle.get_result_async()).status == GoalStatus.STATUS_CANCELED
    time.sleep(0.3)
    assert seen["cmd"] not in core._queue and core._active is None
    assert adapter.env.tick == 0 and adapter.env.grip["a"] == 1.0  # no tick, the gripper never moved


def test_a_clock_value_read_before_a_newer_one_was_published_is_dropped(rig):
    """F3: state timer reads 0.40, the observation thread publishes 0.44, the timer then publishes 0.40."""
    runtime, _client, _adapter = rig
    sent = []
    runtime._ros_clock_pub = SimpleNamespace(publish=lambda msg: sent.append(msg.clock.sec + msg.clock.nanosec * 1e-9))
    runtime._publish_ros_clock(0.44)  # the observation thread: the newer tick went out first
    runtime._core.sim_time_s = lambda: 0.40  # what the state timer had read before that
    runtime._publish_clock()  # ...and now publishes
    runtime._publish_ros_clock(0.48)
    assert sent == sorted(sent) and pytest.approx(0.48) == sent[-1]
    assert all(round(value, 2) != 0.40 for value in sent)


def test_a_pose_move_is_withdrawn_by_its_callers_command_id(rig):
    """F1 (runtime side): the MoveToPose command_id is the id /simulation/cancel_command withdraws."""
    from ibrobot_msgs.srv import CancelSimulationCommand, MoveToPose

    runtime, client, adapter = rig
    assert _call(client, SetRuntimeMode, "/runtime/set_mode", SetRuntimeMode.Request(mode="trajectory")).success
    move_client = client.create_client(MoveToPose, "/motion/move_to_pose")
    assert move_client.wait_for_service(timeout_sec=10.0)
    move = move_client.call_async(_pose_request("task-abc"))
    assert wait_until(lambda: adapter.env.tick >= 3)
    cancel = _call(
        client,
        CancelSimulationCommand,
        "/simulation/cancel_command",
        CancelSimulationCommand.Request(command_id="task-abc"),
    )
    at_ack = adapter.env.tick
    assert cancel.success and cancel.outcome == "CANCELLED" and cancel.ticks_executed >= 2
    assert _wait(move).outcome == "CANCELLED"
    time.sleep(0.3)
    assert adapter.env.tick <= at_ack + 1  # at most the tick already handed to the platform
