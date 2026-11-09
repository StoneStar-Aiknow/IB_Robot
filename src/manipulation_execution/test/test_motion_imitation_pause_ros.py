"""Verify the pause service actually removes and recreates its DDS camera reader."""

import os
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

from embodied_common.motion_imitation_contract import PAUSE_SERVICE, PAUSED_STATE_TOPIC
from ibrobot_msgs.msg import ActorState
from ibrobot_msgs.srv import SetPerceptionPaused
from manipulation_execution.motion_imitation_stream_node import MotionImitationStreamNode
from manipulation_execution.perception_pause_follower import PAUSED_STATE_QOS, PausableSubscription


def _spin_until(executor, predicate):
    deadline = time.monotonic() + 5.0
    while not predicate():
        assert time.monotonic() < deadline, "the local ROS graph did not reach the expected state"
        executor.spin_once(timeout_sec=0.02)


def test_pause_ownership_releases_only_the_camera_reader_until_the_last_owner_finishes():
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID")
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    rclpy.init()
    stream = observer = None
    executor = SingleThreadedExecutor()
    try:
        topic = "/test_manipulation_pause/rgb/compressed"
        stream = MotionImitationStreamNode(parameter_overrides=[Parameter("rgb_compressed_topic", value=topic)])
        observer = Node("test_manipulation_pause_observer")
        camera = observer.create_publisher(CompressedImage, topic, qos_profile_sensor_data)
        actors = []
        observer.create_subscription(ActorState, "/motion_imitation_stream/actor_state", actors.append, 10)
        states = []
        observer.create_subscription(Bool, PAUSED_STATE_TOPIC, states.append, PAUSED_STATE_QOS)
        follower_frames = []
        follower = PausableSubscription(
            observer, CompressedImage, topic, follower_frames.append, qos_profile_sensor_data
        )
        client = observer.create_client(SetPerceptionPaused, PAUSE_SERVICE)
        executor.add_node(stream)
        executor.add_node(observer)
        assert client.wait_for_service(timeout_sec=5.0)
        _spin_until(executor, lambda: camera.get_subscription_count() == 2)
        _spin_until(executor, lambda: bool(states))
        assert states[-1].data is False  # Latched: a reader that starts now learns the state.

        def request(owner, paused):
            future = client.call_async(SetPerceptionPaused.Request(owner_id=owner, paused=paused))
            _spin_until(executor, future.done)
            return future.result()

        assert request("pick-1", True).success
        # One acknowledged pause releases the stream's reader and every follower's.
        _spin_until(executor, lambda: camera.get_subscription_count() == 0)
        assert follower.paused and not follower.subscribed
        _spin_until(executor, lambda: bool(actors))
        assert actors[-1].reason == "perception_paused" and not actors[-1].actor_present
        assert observer.count_publishers(topic) == 1  # The platform camera stays active.
        assert request("pick-1", True).success
        assert request("place-2", True).success
        assert request("pick-1", False).paused
        assert camera.get_subscription_count() == 0
        assert request("pick-1", False).paused
        assert not request("place-2", False).paused
        _spin_until(executor, lambda: camera.get_subscription_count() == 2)
        assert not follower.paused and follower.subscribed
        camera.publish(CompressedImage())
        _spin_until(executor, lambda: bool(follower_frames))  # The recreated reader really receives.
        assert not request("pick-1", True).success  # A delayed acquisition cannot undo its release.
        assert not request("place-2", False).paused
        assert camera.get_subscription_count() == 2
    finally:
        executor.shutdown(timeout_sec=5.0)
        if stream is not None:
            stream.destroy_node()
        if observer is not None:
            observer.destroy_node()
        rclpy.shutdown()


def test_a_reader_started_during_a_pause_never_subscribes_until_the_pause_ends():
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    rclpy.init()
    publisher_node = reader_node = None
    executor = SingleThreadedExecutor()
    try:
        topic = "/test_manipulation_pause/late/compressed"
        state_topic = "/test_manipulation_pause/late/paused"
        publisher_node = Node("test_pause_state_source")
        state = publisher_node.create_publisher(Bool, state_topic, PAUSED_STATE_QOS)
        camera = publisher_node.create_publisher(CompressedImage, topic, qos_profile_sensor_data)
        state.publish(Bool(data=True))
        reader_node = Node("test_pause_late_reader")
        changes = []
        reader = PausableSubscription(
            reader_node,
            CompressedImage,
            topic,
            lambda _message: None,
            qos_profile_sensor_data,
            on_change=changes.append,
            state_topic=state_topic,
        )
        executor.add_node(publisher_node)
        executor.add_node(reader_node)
        _spin_until(executor, lambda: reader.paused)
        _spin_until(executor, lambda: camera.get_subscription_count() == 0)
        state.publish(Bool(data=False))
        _spin_until(executor, lambda: not reader.paused)
        _spin_until(executor, lambda: camera.get_subscription_count() == 1)
        assert changes == [True, False]
    finally:
        executor.shutdown(timeout_sec=5.0)
        for node in (publisher_node, reader_node):
            if node is not None:
                node.destroy_node()
        rclpy.shutdown()


def test_repeated_pause_under_a_fast_publisher_does_not_end_the_executor():
    """Destroying the reader under a MultiThreadedExecutor raised InvalidHandle on the robot."""
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    rclpy.init()
    node = source = None
    executor = MultiThreadedExecutor(num_threads=3)
    stop = threading.Event()
    failures = []

    def spin():
        try:
            executor.spin()
        except Exception as exc:  # noqa: BLE001
            failures.append(repr(exc))

    try:
        topic = "/test_manipulation_pause/stress/compressed"
        node = Node("test_pause_stress_reader")
        source = Node("test_pause_stress_source")
        camera = source.create_publisher(CompressedImage, topic, qos_profile_sensor_data)
        frames = []
        reader = PausableSubscription(
            node, CompressedImage, topic, frames.append, qos_profile_sensor_data, state_topic=None
        )
        executor.add_node(node)
        executor.add_node(source)
        spinner = threading.Thread(target=spin, daemon=True)
        spinner.start()

        def publish():
            while not stop.is_set():
                camera.publish(CompressedImage())
                time.sleep(0.001)

        threading.Thread(target=publish, daemon=True).start()
        deadline = time.monotonic() + 5.0
        while not frames:
            assert time.monotonic() < deadline, "no frame reached the reader"
            time.sleep(0.01)
        for _ in range(40):
            reader.set_paused(True)
            assert not reader.subscribed
            reader.set_paused(False)
            assert reader.subscribed
        assert spinner.is_alive() and not failures
        frames.clear()
        deadline = time.monotonic() + 5.0
        while not frames:
            assert time.monotonic() < deadline, "the reader did not receive after the last resume"
            time.sleep(0.01)
    finally:
        stop.set()
        time.sleep(0.05)
        executor.shutdown(timeout_sec=5.0)
        for item in (node, source):
            if item is not None:
                item.destroy_node()
        rclpy.shutdown()
