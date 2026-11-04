"""The feedback history fills from the X2 runtime's public topics, bound by description.

Runs the real runtime against its vendor mock; skips without the AimDK overlay.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("aimdk_msgs", reason="AimDK overlay not sourced; source the vendor workspace to run these")

from aimdk_grasp.feedback import FeedbackHistory  # noqa: E402
from aimdk_grasp.feedback_ros import subscribe_feedback  # noqa: E402
from aimdk_robot import dataplane  # noqa: E402
from aimdk_robot.runtime_node import AimdkRuntimeNode  # noqa: E402
from aimdk_robot.vendor_mock_node import AimdkVendorMock  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402

from robot_runtime.interface_description import build_description  # noqa: E402
from robot_runtime.on_demand import OnDemandSubscriptions  # noqa: E402
from robot_runtime.profile import load_profile  # noqa: E402

PROFILE = Path(__file__).resolve().parents[2] / "aimdk_robot" / "profiles" / "x2_ultra_omnihand.yaml"


@pytest.fixture
def graph():
    owned = not rclpy.ok()
    if owned:
        rclpy.init()
    mock = AimdkVendorMock(parameter_overrides=[rclpy.parameter.Parameter("hand_type", value="dexterous")])
    # The views are published by the runtime's data plane, its own process.
    data_plane = dataplane.start_process(load_profile(PROFILE))
    runtime = AimdkRuntimeNode(
        parameter_overrides=[
            rclpy.parameter.Parameter("profile", value=os.fspath(PROFILE)),
            rclpy.parameter.Parameter("use_sim", value=True),
            rclpy.parameter.Parameter("startup_timeout_s", value=8.0),
        ]
    )
    consumer = rclpy.create_node("aimdk_grasp_feedback_test")
    executor = MultiThreadedExecutor(num_threads=4)
    for node in (mock, runtime, consumer):
        executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        yield SimpleNamespace(context=consumer.context, consumer=consumer, runtime=runtime)
    finally:
        # The runtime's startup checks publish status from their own thread;
        # let them finish before the nodes are destroyed under them.
        runtime._startup_thread.join(timeout=10.0)
        data_plane.terminate()
        data_plane.wait(timeout=5.0)
        executor.shutdown()
        for node in (consumer, runtime, mock):
            node.destroy_node()
        if owned and rclpy.ok():
            rclpy.shutdown()


def _wait(predicate, what: str, timeout_s: float = 15.0):
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        try:
            last = predicate()
            if last:
                return last
        except ValueError as exc:
            last = exc
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what} (last: {last!r})")


def test_feedback_assembles_from_the_bound_runtime_views(graph):
    description = build_description(load_profile(os.fspath(PROFILE)), use_sim=True)
    history = FeedbackHistory()
    subscriptions = OnDemandSubscriptions(graph.context, "aimdk_grasp_feedback_views")
    subscribe_feedback(subscriptions, description, history)

    def healthy():
        subscriptions.drain()
        latest = history.latest()
        return latest if latest.healthy else None

    feedback = _wait(healthy, "healthy feedback")

    assert feedback.left_hand.shape == (10,)
    assert {"left_shoulder_pitch_joint", "right_wrist_roll_joint", "head_yaw_joint"} <= set(feedback.positions)
    image_stamp = int(feedback.received_at * 1e9) + 5_000_000
    assert history.at(image_stamp).left_hand.shape == (10,)
    subscriptions.close()


def test_a_description_without_the_hand_view_is_refused(graph):
    description = build_description(load_profile(os.fspath(PROFILE)), use_sim=True)
    del description["interfaces"]["joint.hand_state"]

    subscriptions = OnDemandSubscriptions(graph.context, "aimdk_grasp_feedback_refused")
    with pytest.raises(ValueError, match="joint.hand_state"):
        subscribe_feedback(subscriptions, description, FeedbackHistory())
    subscriptions.close()


def test_a_runtime_whose_status_stopped_is_reported_unhealthy(graph):
    """The data plane keeps the views flowing after the runtime's process is gone.

    Stopping the runtime's timers leaves its status unpublished while the
    views continue, which is exactly the split the status age gate exists
    for: the assembled feedback keeps the joints, and reports the runtime
    unhealthy once the last verdict ages out.
    """
    description = build_description(load_profile(os.fspath(PROFILE)), use_sim=True)
    history = FeedbackHistory(max_status_age_s=2.0)
    subscriptions = OnDemandSubscriptions(graph.context, "aimdk_grasp_status_age")

    def healthy():
        subscriptions.drain()
        latest = history.latest()
        return latest if latest.healthy else None

    def stale():
        subscriptions.drain()
        latest = history.latest()
        return None if latest.healthy else latest

    try:
        subscribe_feedback(subscriptions, description, history)
        _wait(healthy, "healthy feedback")

        for timer in graph.runtime.timers:
            timer.cancel()

        feedback = _wait(stale, "feedback that reports the runtime stale")
        assert {"left_shoulder_pitch_joint", "head_yaw_joint"} <= set(feedback.positions)
    finally:
        subscriptions.close()
