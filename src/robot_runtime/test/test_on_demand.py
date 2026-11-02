"""Subscriptions taken on demand: nothing runs until a reader drains, and samples keep when they arrived."""

import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")

from rclpy.executors import SingleThreadedExecutor  # noqa: E402
from rclpy.qos import QoSProfile  # noqa: E402
from std_msgs.msg import Int32, String  # noqa: E402

from robot_runtime.on_demand import OnDemandSubscriptions, received_monotonic  # noqa: E402


@pytest.fixture
def graph():
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    publisher = rclpy.create_node("on_demand_publisher")
    # Spinning: an executor nearby must not run the on-demand callbacks.
    executor = SingleThreadedExecutor()
    executor.add_node(publisher)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    subscriptions = OnDemandSubscriptions(publisher.context, "on_demand_reader")
    yield publisher, subscriptions
    subscriptions.close()
    executor.shutdown()
    publisher.destroy_node()
    if owned and rclpy.ok():
        rclpy.shutdown()


def _connected(publisher_handle, timeout_s=5.0):
    ended = time.monotonic() + timeout_s
    while publisher_handle.get_subscription_count() == 0:
        assert time.monotonic() < ended, "reader never matched"
        time.sleep(0.02)


def test_samples_wait_for_a_drain_and_keep_when_they_were_received(graph):
    publisher, subscriptions = graph
    seen = []
    subscriptions.subscribe(String, "/on_demand/a", QoSProfile(depth=10), 10, lambda m, t: seen.append((m.data, t)))
    output = publisher.create_publisher(String, "/on_demand/a", 10)
    _connected(output)
    sent = time.monotonic()
    output.publish(String(data="first"))
    time.sleep(0.3)
    assert seen == []  # nothing ran on arrival
    assert subscriptions.drain() == 1
    ((data, received),) = seen
    assert data == "first" and sent - 0.05 <= received <= sent + 0.1  # not the drain time, 0.3 s later
    assert subscriptions.drain() == 0


class _NeverEmpty:
    """A reader that always has another sample, the way a live one does while publishers run."""

    def __init__(self, available: int) -> None:
        self.msg_type, self.raw, self.taken, self._available = Int32, False, 0, available
        self.handle = self

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def take_message(self, _msg_type, _raw):
        if self.taken >= self._available:
            return None
        self.taken += 1
        return Int32(data=self.taken), {"received_timestamp": 0}


def test_one_drain_takes_at_most_the_depth_from_a_reader_that_is_never_empty(graph):
    """The bound, asserted on counts rather than on timing.

    What made an unbounded take loop unbounded was samples arriving *during* the loop:
    KEEP_LAST already caps what is stored, so a quiet reader drains the same either way,
    and a flooding publisher only reproduces it on a host that cannot convert samples as
    fast as they are produced -- which is why this asserts the count against a reader that
    never reports empty instead of timing one against a real publisher. Measured on the
    X2's 310P, one unbounded drain of vendor joint state ran 1.9 s and the 50 Hz timer
    calling it fired at 0.5 Hz, while the runtime still believed its feedback was fresh.
    """
    _publisher, subscriptions = graph
    seen = []
    reader = _NeverEmpty(available=10_000)
    subscriptions._entries.append((reader, lambda m, _t: seen.append(m.data), 6))

    assert subscriptions.drain() == 6
    assert reader.taken == 6 and len(seen) == 6
    assert subscriptions.drain() == 6  # the next call takes its own bounded batch


def test_a_drain_passes_samples_of_all_topics_in_arrival_order_and_keeps_only_the_newest(graph):
    publisher, subscriptions = graph
    seen = []
    subscriptions.subscribe(Int32, "/on_demand/b", QoSProfile(depth=10), 3, lambda m, _t: seen.append(("b", m.data)))
    subscriptions.subscribe(Int32, "/on_demand/c", QoSProfile(depth=10), 3, lambda m, _t: seen.append(("c", m.data)))
    b = publisher.create_publisher(Int32, "/on_demand/b", 10)
    c = publisher.create_publisher(Int32, "/on_demand/c", 10)
    _connected(b)
    _connected(c)
    for value in range(5):
        b.publish(Int32(data=value))
        time.sleep(0.01)
        c.publish(Int32(data=value))
        time.sleep(0.01)
    time.sleep(0.2)
    subscriptions.drain()
    # Three kept per topic, interleaved as they arrived.
    assert seen == [("b", 2), ("c", 2), ("b", 3), ("c", 3), ("b", 4), ("c", 4)]


def test_a_sample_without_a_receive_time_stays_unknown():
    """A backlogged sample the middleware did not time cannot be proven fresh."""
    assert received_monotonic({"received_timestamp": 0}, now_monotonic=7.0, now_ns=10**9) is None
    assert received_monotonic({}, now_monotonic=7.0, now_ns=10**9) is None
    assert received_monotonic({"received_timestamp": 9 * 10**8}, now_monotonic=7.0, now_ns=10**9) == pytest.approx(6.9)


def test_a_stale_backlog_without_a_receive_time_is_not_reported_as_just_received(graph):
    """The reader paused, the publisher stopped, the middleware wrote no receive time (CycloneDDS).

    The one queued sample is delivered, but with None: its age is unknown, so a freshness
    gate must refuse it rather than take the drain time for its arrival.
    """
    _publisher, subscriptions = graph
    seen = []
    reader = _NeverEmpty(available=1)
    subscriptions._entries.append((reader, lambda m, received: seen.append((m.data, received)), 5))
    time.sleep(0.2)

    assert subscriptions.drain() == 1
    assert seen == [(1, None)]
    assert subscriptions.drain() == 0


def test_a_launch_node_name_remap_does_not_rename_the_reader_node():
    """launch_ros passes -r __node:=<owner> to the process; the reader keeps its own name, not a duplicate."""
    from rclpy.context import Context

    context = Context()
    rclpy.init(args=["--ros-args", "-r", "__node:=owner"], context=context)
    try:
        owner = rclpy.create_node("anything", context=context)
        readers = OnDemandSubscriptions(context, "owner_readers")
        assert owner.get_name() == "owner"  # the remap is in force for ordinary nodes
        assert readers._node.get_name() == "owner_readers"
        readers.close()
        owner.destroy_node()
    finally:
        rclpy.try_shutdown(context=context)


def test_the_reader_node_keeps_the_launch_namespace_and_topic_remaps():
    """Only the node name is pinned: a namespaced, remapped launch still feeds the readers."""
    from rclpy.context import Context

    context = Context()
    rclpy.init(
        args=[
            "--ros-args",
            "-r",
            "__node:=owner",
            "-r",
            "__ns:=/robot_a",
            "-r",
            "/camera/image:=/robot_a/camera/image",
        ],
        context=context,
    )
    publisher = rclpy.create_node("publisher", context=context, use_global_arguments=False)
    readers = OnDemandSubscriptions(context, "owner_readers")
    try:
        assert readers._node.get_name() == "owner_readers"
        assert readers._node.get_namespace() == "/robot_a"
        seen = []
        readers.subscribe(Int32, "/camera/image", None, 5, lambda m, _t: seen.append(("absolute", m.data)))
        readers.subscribe(Int32, "status", None, 5, lambda m, _t: seen.append(("relative", m.data)))
        image = publisher.create_publisher(Int32, "/robot_a/camera/image", 10)
        status = publisher.create_publisher(Int32, "/robot_a/status", 10)
        _connected(image)
        _connected(status)
        image.publish(Int32(data=1))
        status.publish(Int32(data=2))
        ended = time.monotonic() + 5.0
        while len(seen) < 2 and time.monotonic() < ended:
            readers.drain()
            time.sleep(0.02)
        assert sorted(seen) == [("absolute", 1), ("relative", 2)]
    finally:
        readers.close()
        publisher.destroy_node()
        rclpy.try_shutdown(context=context)
