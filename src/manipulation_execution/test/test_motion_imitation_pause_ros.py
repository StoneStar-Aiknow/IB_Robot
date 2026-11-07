"""Verify the pause service actually removes and recreates its DDS camera reader."""

import os
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage

from embodied_common.motion_imitation_contract import PAUSE_SERVICE
from ibrobot_msgs.msg import ActorState
from ibrobot_msgs.srv import SetPerceptionPaused
from manipulation_execution.motion_imitation_stream_node import MotionImitationStreamNode


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
        client = observer.create_client(SetPerceptionPaused, PAUSE_SERVICE)
        executor.add_node(stream)
        executor.add_node(observer)
        assert client.wait_for_service(timeout_sec=5.0)
        _spin_until(executor, lambda: camera.get_subscription_count() == 1)

        def request(owner, paused):
            future = client.call_async(SetPerceptionPaused.Request(owner_id=owner, paused=paused))
            _spin_until(executor, future.done)
            return future.result()

        assert request("pick-1", True).success
        _spin_until(executor, lambda: camera.get_subscription_count() == 0)
        _spin_until(executor, lambda: bool(actors))
        assert actors[-1].reason == "perception_paused" and not actors[-1].actor_present
        assert observer.count_publishers(topic) == 1  # The platform camera stays active.
        assert request("pick-1", True).success
        assert request("place-2", True).success
        assert request("pick-1", False).paused
        assert camera.get_subscription_count() == 0
        assert request("pick-1", False).paused
        assert not request("place-2", False).paused
        _spin_until(executor, lambda: camera.get_subscription_count() == 1)
        assert not request("pick-1", True).success  # A delayed acquisition cannot undo its release.
        assert not request("place-2", False).paused
        assert camera.get_subscription_count() == 1
    finally:
        executor.shutdown(timeout_sec=5.0)
        if stream is not None:
            stream.destroy_node()
        if observer is not None:
            observer.destroy_node()
        rclpy.shutdown()
