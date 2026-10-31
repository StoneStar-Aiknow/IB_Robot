"""The ROS binding of the streaming executor, against a fake runtime over real topics."""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ibrobot_msgs.msg import JointCommandState, RuntimeStatus
from manipulation_execution.pick.pipeline.contracts import ArmSegment, MotionState, SegmentKind
from manipulation_execution.pick.pipeline.ros_stream import RosStreamIO, StreamBindingError, bind_stream
from manipulation_execution.pick.pipeline.streaming import StreamConfig, StreamExecutor

ARM = ("left_a", "left_b", "right_a")
QOS = {"reliability": "reliable", "durability": "volatile", "history": "keep_last", "depth": 10}
ROBOT = {"id": "test", "type": "test_arm", "runtime_name": "fake", "runtime_version": "1"}
SAG = 0.04


def description(**changes):
    interfaces = {
        "joint.arm_stream": {
            "kind": "topic",
            "direction": "subscribe",
            "message_type": "std_msgs/msg/Float64MultiArray",
            "endpoint": "/test/arm/commands",
            "joint_names": list(ARM),
            "modes": ["policy_stream"],
            "qos": QOS,
        },
        "joint.state": {
            "kind": "topic",
            "direction": "publish",
            "message_type": "sensor_msgs/msg/JointState",
            "endpoint": "/test/joint_states",
            "joint_names": list(ARM),
            "qos": QOS,
        },
        "joint.head_state": {
            "kind": "topic",
            "direction": "publish",
            "message_type": "sensor_msgs/msg/JointState",
            "endpoint": "/test/head_states",
            "joint_names": ["head"],
            "qos": QOS,
        },
        "joint.arm_command_state": {
            "kind": "topic",
            "direction": "publish",
            "message_type": "ibrobot_msgs/msg/JointCommandState",
            "endpoint": "/test/arm/command_state",
            "joint_names": list(ARM),
            "qos": QOS,
        },
        "runtime.status": {
            "kind": "topic",
            "direction": "publish",
            "message_type": "ibrobot_msgs/msg/RuntimeStatus",
            "endpoint": "/test/runtime_status",
            "qos": QOS,
        },
    }
    for key, value in changes.items():
        interfaces[key] = value
    return {"robot": ROBOT, "interfaces": interfaces}


def bind(desc=None):
    return bind_stream(
        desc or description(),
        stream="joint.arm_stream",
        feedback=["joint.state", "joint.head_state"],
        command="joint.arm_command_state",
    )


def test_binding_names_the_first_interface_that_does_not_fit():
    binding = bind()
    assert binding.stream.joint_names == ARM and [view.interface_id for view in binding.feedback] == [
        "joint.state",
        "joint.head_state",
    ]
    assert binding.command.endpoint == "/test/arm/command_state" and binding.mode == "policy_stream"
    bad = {
        "joint.arm_stream": dict(description()["interfaces"]["joint.arm_stream"], modes=["stream"]),
        "joint.state": dict(description()["interfaces"]["joint.state"], joint_names=["left_a"]),
        "joint.arm_command_state": dict(
            description()["interfaces"]["joint.arm_command_state"], message_type="sensor_msgs/msg/JointState"
        ),
    }
    for key, value in bad.items():
        with pytest.raises(StreamBindingError, match=key if key != "joint.state" else "do not report"):
            bind(description(**{key: value}))
    renamed = description(
        **{"joint.state": dict(description()["interfaces"]["joint.state"], joint_names=["L_a", "L_b", "R_a"])}
    )
    with pytest.raises(StreamBindingError, match="do not report"):
        bind(renamed)
    aliases = {"left_a": "L_a", "left_b": "L_b", "right_a": "R_a"}
    aliased = bind_stream(renamed, stream="joint.arm_stream", feedback=["joint.state"], feedback_names=aliases)
    assert aliased.stream.joint_names == ARM
    assert aliased.feedback_aliases == {"L_a": "left_a", "L_b": "left_b", "R_a": "right_a"}
    with pytest.raises(StreamBindingError, match="one feedback name"):
        bind_stream(
            renamed,
            stream="joint.arm_stream",
            feedback=["joint.state"],
            feedback_names={"left_a": "L_a", "left_b": "L_a"},
        )
    with pytest.raises(StreamBindingError, match="live runtime description"):
        bind_stream({}, stream="joint.arm_stream", feedback=[])


class FakeRuntime(Node):
    """Holds a target; the arm settles SAG below it. Publishes views, the echo and a status.

    ``state_names`` renames the joints in the arm state view, as the X2 names
    its hand joints differently from the stream that commands them.
    """

    def __init__(self, state_names=ARM):
        super().__init__("fake_stream_runtime")
        self.state_names = tuple(state_names)
        self.target = np.array([0.1, 0.2, 0.3]) + SAG
        self.measured = self.target - SAG
        self.received = []
        self.mode, self.latched = "policy_stream", False
        self.publishing = True
        self.echo = True
        self._joints = self.create_publisher(JointState, "/test/joint_states", 10)
        self._head = self.create_publisher(JointState, "/test/head_states", 10)
        self._echo = self.create_publisher(JointCommandState, "/test/arm/command_state", 10)
        self._status = self.create_publisher(RuntimeStatus, "/test/runtime_status", 10)
        self.create_subscription(Float64MultiArray, "/test/arm/commands", self._on_command, 10)
        self.create_timer(0.01, self._tick)

    def _on_command(self, message):
        self.received.append((time.monotonic(), list(message.data)))
        self.target = np.array(message.data)

    def _tick(self):
        if not self.publishing:
            return
        self.measured += (self.target - SAG - self.measured) * 0.3
        stamp = self.get_clock().now().to_msg()
        self._joints.publish(JointState(name=list(self.state_names), position=self.measured.tolist()))
        self._head.publish(JointState(name=["head"], position=[0.0]))
        if self.echo:
            self._echo.publish(
                JointCommandState(
                    name=list(ARM), position=self.target.tolist(), stiffness=[40.0] * 3, damping=[2.0] * 3
                )
            )
        self._status.publish(
            RuntimeStatus(
                runtime_name="fake",
                runtime_version="1",
                lifecycle="ACTIVE",
                active_mode=self.mode,
                stop_latched=self.latched,
                interface_description_json=json.dumps({"robot": ROBOT}),
                stamp=stamp,
            )
        )


@pytest.fixture
def ros():
    rclpy.init()
    runtime = FakeRuntime()
    client = Node("stream_client")
    io = RosStreamIO(client, bind())
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(runtime)
    executor.add_node(client)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    ended = time.monotonic() + 5.0
    while (not io.command().positions or io.refusal() is not None or not io.feedback().healthy) and (
        time.monotonic() < ended
    ):
        time.sleep(0.02)
    yield runtime, io
    io.close()
    executor.shutdown()
    runtime.destroy_node()
    client.destroy_node()
    rclpy.shutdown()


class RecordingPublisher:
    """Stands in for the DDS publisher: every command that left, in order."""

    def __init__(self):
        self.messages = []

    def publish(self, message):
        self.messages.append(list(message.data))


class HeldPublisher(RecordingPublisher):
    """Hold one actual publisher call so another thread can race the send."""

    def __init__(self, *, hold_call):
        super().__init__()
        self.calls = 0
        self.hold_call = hold_call
        self.held = threading.Event()
        self.go = threading.Event()

    def publish(self, message):
        self.calls += 1
        if self.calls == self.hold_call:
            self.held.set()
            assert self.go.wait(timeout=5.0)
        super().publish(message)


class HeldArray:
    """Stands in for the message constructor and holds one call until released.

    Holding the keepalive's construction is holding it between copying the
    command and sending it, the window a release or a newer command must not
    be able to slip through.
    """

    def __init__(self, make):
        self.make = make
        self.calls = 0
        self.hold_call = None
        self.held = threading.Event()
        self.go = threading.Event()

    def __call__(self, *, data):
        self.calls += 1
        if self.calls == self.hold_call:
            self.held.set()
            assert self.go.wait(timeout=5.0)
        return self.make(data=data)


def hold_the_io(io, *, hold_call):
    """Replace the io's publisher and message constructor with controllable ones."""
    held = HeldArray(io._array)
    held.hold_call = hold_call
    recorder = RecordingPublisher()
    io._publisher = recorder
    io._array = held
    return recorder, held


def test_the_io_merges_views_reads_the_echo_and_keeps_the_last_command_alive(ros):
    runtime, io = ros
    sample = io.feedback()
    assert sample.healthy and set(sample.positions) == {*ARM, "head"}
    assert time.monotonic() - sample.received_at < 0.1
    np.testing.assert_allclose([io.command().positions[name] for name in ARM], runtime.target)
    assert io.refusal() is None
    io.publish([0.0, 0.0, 0.0])
    time.sleep(0.45)
    io.release()
    count = len(runtime.received)
    assert count >= 4  # re-sent about every 0.1 s while held
    time.sleep(0.3)
    assert len(runtime.received) == count  # released: nothing more
    runtime.latched = True
    time.sleep(0.1)
    assert "stop is latched" in io.refusal()
    runtime.latched, runtime.mode = False, "stream"
    time.sleep(0.1)
    assert "owned by another producer" in io.refusal()


def test_release_holds_the_stream_until_an_in_flight_keepalive_is_out_and_sends_nothing_after_it(ros):
    """A keepalive caught mid-send finishes first: release() returns only once no old command is left."""
    _, io = ros
    recorder, held = hold_the_io(io, hold_call=2)  # 1: the executor's own publish, 2: its keepalive
    io.publish([1.0, 1.0, 1.0])
    assert held.held.wait(timeout=5.0)  # the keeper has copied the command and is sending it
    released = threading.Event()
    threading.Thread(target=lambda: (io.release(), released.set()), daemon=True).start()
    time.sleep(0.1)  # the release has had every chance to return wrongly early
    held.go.set()  # the held send completes
    assert released.wait(timeout=5.0)  # only then may release() return
    assert len(recorder.messages) >= 2  # the original command and the keepalive that was in flight
    sent = len(recorder.messages)
    time.sleep(0.3)  # a live keepalive would re-send within this window
    assert len(recorder.messages) == sent  # released: nothing more


def test_release_waits_for_an_in_flight_publish_before_it_stops_the_stream(ros):
    """release() cannot return while the ordinary command send is still in flight."""
    _, io = ros
    io._keepalive_s = 10.0
    publisher = HeldPublisher(hold_call=1)
    io._publisher = publisher
    published = threading.Event()
    sender = threading.Thread(target=lambda: (io.publish([1.0, 1.0, 1.0]), published.set()), daemon=True)
    sender.start()
    assert publisher.held.wait(timeout=5.0)
    released = threading.Event()
    stopper = threading.Thread(target=lambda: (io.release(), released.set()), daemon=True)
    stopper.start()
    returned_while_sending = released.wait(timeout=0.3)
    publisher.go.set()
    assert published.wait(timeout=5.0)
    assert released.wait(timeout=5.0)
    sender.join(timeout=5.0)
    stopper.join(timeout=5.0)
    assert not returned_while_sending
    assert publisher.messages == [[1.0, 1.0, 1.0]]
    time.sleep(0.3)
    assert publisher.messages == [[1.0, 1.0, 1.0]]


def test_a_keepalive_in_flight_cannot_publish_after_a_newer_command(ros):
    """A newer command lands after the keepalive it raced, never before an older one."""
    _, io = ros
    recorder, held = hold_the_io(io, hold_call=2)
    io.publish([1.0, 1.0, 1.0])
    assert held.held.wait(timeout=5.0)  # the keeper is sending the old command
    newer = threading.Thread(target=lambda: io.publish([2.0, 2.0, 2.0]), daemon=True)
    newer.start()
    newer.join(timeout=0.3)  # an unserialized send runs through while the keeper holds no lock
    held.go.set()  # the keepalive completes; the newer command follows it through the same boundary
    newer.join(timeout=5.0)
    first_new = recorder.messages.index([2.0, 2.0, 2.0])
    assert all(message == [2.0, 2.0, 2.0] for message in recorder.messages[first_new:])
    assert recorder.messages[-1] == [2.0, 2.0, 2.0]
    io.release()


def test_feedback_read_after_the_runtime_falls_silent_is_as_old_as_its_last_sample(ros):
    runtime, io = ros
    # The client's executor holds none of it: samples are taken when read.
    assert not list(io._node.subscriptions)
    runtime.publishing = False
    time.sleep(0.1)  # what was in flight has arrived
    first = io.feedback().received_at
    time.sleep(0.3)
    sample = io.feedback()
    assert sample.received_at == first
    assert time.monotonic() - sample.received_at >= 0.3
    assert time.monotonic() - io.command().received_at >= 0.3


def test_a_bound_report_that_never_arrived_stops_the_executor_instead_of_being_skipped(ros):
    """Without the board the executor would skip its gain and board checks and take over the sagging arm."""
    runtime, _ = ros
    runtime.echo = False
    client = Node("late_client")
    io = RosStreamIO(client, bind())
    executor = MultiThreadedExecutor(num_threads=1)
    executor.add_node(client)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        ended = time.monotonic() + 5.0
        while not io.feedback().healthy and time.monotonic() < ended:
            time.sleep(0.02)
        report = io.command()
        assert report is not None and report.received_at == float("-inf") and not report.positions
        sent = len(runtime.received)
        stream = StreamExecutor(io, StreamConfig(expected_stiffness=40.0, expected_damping=2.0))
        here = np.array([io.feedback().positions[name] for name in ARM[:2]])
        segment = ArmSegment("approach", SegmentKind.TARGET, ARM[:2], np.stack([here, here + 0.05]))
        outcome = stream.execute(segment, carrying=False, halted=lambda: False, cancelled=lambda: False)
        assert outcome.state is MotionState.FAILED and "report is stale" in outcome.reason
        time.sleep(0.2)
        assert len(runtime.received) == sent  # nothing was sent
    finally:
        io.close()
        executor.shutdown()
        thread.join(timeout=5.0)
        client.destroy_node()
        runtime.echo = True


def test_the_executor_streams_a_segment_through_ros(ros):
    runtime, io = ros
    executor = StreamExecutor(
        io,
        StreamConfig(
            tracking_error_rad=0.3,
            arrival_error_rad=0.2,
            expected_stiffness=40.0,
            expected_damping=2.0,
            locked_joints=("head", "right_a"),
        ),
    )
    start = io.feedback()
    here = np.array([start.positions[name] for name in ARM[:2]])
    takeover = runtime.target.copy()
    segment = ArmSegment(
        "approach",
        SegmentKind.TRAJECTORY,
        ARM[:2],
        np.linspace(here, here + [0.1, -0.05], 11),
        np.linspace(0.0, 0.5, 11),
    )
    outcome = executor.execute(segment, carrying=False, halted=lambda: False, cancelled=lambda: False)
    assert outcome.state is MotionState.SUCCEEDED, outcome.reason
    np.testing.assert_allclose(runtime.received[0][1], takeover, atol=1e-9)
    np.testing.assert_allclose(runtime.target[:2], here + [0.1, -0.05] + SAG, atol=1e-6)
    executor.finish()


def test_a_view_that_names_its_joints_differently_still_serves_the_executor():
    """feedback_names reaches the executor: it reads the stream's joint names, not the view's."""
    rclpy.init()
    runtime = FakeRuntime(state_names=("L_a", "L_b", "R_a"))
    client = Node("stream_client_renamed")
    renamed = description(
        **{"joint.state": dict(description()["interfaces"]["joint.state"], joint_names=["L_a", "L_b", "R_a"])}
    )
    io = RosStreamIO(
        client,
        bind_stream(
            renamed,
            stream="joint.arm_stream",
            feedback=["joint.state"],
            command="joint.arm_command_state",
            feedback_names={"left_a": "L_a", "left_b": "L_b", "right_a": "R_a"},
        ),
    )
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(runtime)
    executor.add_node(client)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        ended = time.monotonic() + 5.0
        while (not io.feedback().healthy or io.refusal() is not None or not io.command().positions) and (
            time.monotonic() < ended
        ):
            time.sleep(0.02)
        sample = io.feedback()
        assert set(sample.positions) == set(ARM)  # renamed to the stream's joints
        np.testing.assert_allclose([sample.positions[name] for name in ARM], runtime.measured, atol=1e-9)
        stream = StreamExecutor(
            io,
            StreamConfig(
                tracking_error_rad=0.3,
                arrival_error_rad=0.2,
                expected_stiffness=40.0,
                expected_damping=2.0,
                locked_joints=("right_a",),
            ),
        )
        here = np.array([sample.positions[name] for name in ARM[:2]])
        segment = ArmSegment(
            "approach",
            SegmentKind.TRAJECTORY,
            ARM[:2],
            np.linspace(here, here + [0.1, -0.05], 11),
            np.linspace(0.0, 0.5, 11),
        )
        outcome = stream.execute(segment, carrying=False, halted=lambda: False, cancelled=lambda: False)
        assert outcome.state is MotionState.SUCCEEDED, outcome.reason
        np.testing.assert_allclose(runtime.target[:2], here + [0.1, -0.05] + SAG, atol=1e-6)
        stream.finish()
    finally:
        io.close()
        executor.shutdown()
        thread.join(timeout=5.0)
        runtime.destroy_node()
        client.destroy_node()
        rclpy.shutdown()


def test_a_sample_the_middleware_did_not_time_is_too_old_to_drive_the_arm(ros):
    """CycloneDDS leaves received_timestamp 0; such a sample waited an unknown time in the queue."""
    _, io = ros
    io.feedback()  # a timed sample first, so the refusal is the untimed one's age
    for key in bind().feedback:
        io._on_view(key.interface_id, JointState(name=["head"], position=[0.0]), None)
    io._on_command(JointCommandState(name=list(ARM)), None)
    assert io.feedback().received_at == float("-inf")
    assert io.command().received_at == float("-inf")
    stream = StreamExecutor(io, StreamConfig(expected_stiffness=40.0, expected_damping=2.0))
    here = np.zeros(2)
    segment = ArmSegment("approach", SegmentKind.TARGET, ARM[:2], np.stack([here, here + 0.05]))
    outcome = stream.execute(segment, carrying=False, halted=lambda: False, cancelled=lambda: False)
    assert outcome.state is MotionState.FAILED and "stale" in outcome.reason
