"""TF that waits in its reader queue and is parsed only when a lookup needs it."""

from __future__ import annotations

import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from aimdk_grasp.lazy_transforms import LazyTransforms  # noqa: E402
from geometry_msgs.msg import TransformStamped  # noqa: E402
from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.time import Time  # noqa: E402
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster, TransformException  # noqa: E402

from robot_runtime.on_demand import OnDemandSubscriptions  # noqa: E402


def transform(x: float, stamp_s: int, parent="torso_link", child="camera") -> TransformStamped:
    message = TransformStamped()
    message.header.frame_id, message.child_frame_id = parent, child
    message.header.stamp.sec = stamp_s
    message.transform.translation.x = x
    message.transform.rotation.w = 1.0
    return message


@pytest.fixture
def graph():
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    publisher = rclpy.create_node("lazy_transforms_publisher")
    executor = SingleThreadedExecutor()
    executor.add_node(publisher)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    subscriptions = OnDemandSubscriptions(publisher.context, "lazy_transforms_reader")
    yield publisher, subscriptions
    subscriptions.close()
    executor.shutdown()
    publisher.destroy_node()
    if owned and rclpy.ok():
        rclpy.shutdown()


def _send(publisher, broadcaster, transforms):
    ended = time.monotonic() + 5.0
    while publisher.count_subscribers("/tf") == 0:
        assert time.monotonic() < ended, "/tf reader never matched"
        time.sleep(0.02)
    for message in transforms:
        broadcaster.sendTransform(message)
    time.sleep(0.2)


def test_tf_is_parsed_by_the_lookup_not_on_arrival(graph):
    publisher, subscriptions = graph
    transforms = LazyTransforms(subscriptions)
    _send(publisher, TransformBroadcaster(publisher), [transform(0.1, 10), transform(0.3, 11)])
    assert transforms.buffer.all_frames_as_string() == ""
    found = transforms.lookup_transform("torso_link", "camera", Time(seconds=10, nanoseconds=500_000_000))
    assert found.transform.translation.x == pytest.approx(0.2)


def test_tf_beyond_the_queue_is_dropped_unparsed(graph):
    publisher, subscriptions = graph
    transforms = LazyTransforms(subscriptions, queue=2)
    _send(publisher, TransformBroadcaster(publisher), [transform(0.1, 10), transform(0.2, 11), transform(0.3, 12)])
    assert transforms.lookup_transform("torso_link", "camera", Time(seconds=12)).transform.translation.x == 0.3
    with pytest.raises(TransformException):
        transforms.lookup_transform("torso_link", "camera", Time(seconds=10))


def test_static_tf_published_before_the_reader_is_found(graph):
    publisher, subscriptions = graph
    StaticTransformBroadcaster(publisher).sendTransform(transform(0.05, 0, parent="head_link", child="camera_optical"))
    transforms = LazyTransforms(subscriptions)
    ended = time.monotonic() + 5.0
    while True:
        try:
            found = transforms.lookup_transform("head_link", "camera_optical", Time(seconds=42))
            break
        except TransformException:
            assert time.monotonic() < ended, "static TF never arrived"
            time.sleep(0.05)
    assert found.transform.translation.x == pytest.approx(0.05)
