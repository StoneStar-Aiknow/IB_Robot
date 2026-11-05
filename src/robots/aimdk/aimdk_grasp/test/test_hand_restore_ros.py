"""Public ROS ports against local Mock topics/services in the isolated test domain."""

import json
import os
import threading
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest
import rclpy
from aimdk_grasp.hand_restore import RestoreContext, RestoreError
from aimdk_grasp.hand_restore_ros import RuntimeRestorePorts
from aimdk_grasp.joints import HAND_PARTS, HEAD, LEFT_ARM, LEFT_HAND, RIGHT_ARM, RIGHT_HAND
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from ibrobot_msgs.msg import JointCommandState, RuntimeStatus
from ibrobot_msgs.srv import GetSkillGatewayStatus


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    pytest.fail("Mock ROS condition did not arrive")


def test_public_feedback_ownership_stream_order_and_stop(monkeypatch):
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID")
    assert os.environ["ROS_LOCALHOST_ONLY"] == "1"
    rclpy.init()
    publisher = rclpy.create_node("mock_restore_runtime")
    consumer = rclpy.create_node("restore_consumer")
    descriptions = (
        ("joint.state", "sensor_msgs/msg/JointState", LEFT_ARM + RIGHT_ARM),
        ("joint.head_state", "sensor_msgs/msg/JointState", HEAD),
        ("joint.hand_state", "sensor_msgs/msg/JointState", LEFT_HAND + RIGHT_HAND),
        ("joint.arm_command_state", "ibrobot_msgs/msg/JointCommandState", LEFT_ARM + RIGHT_ARM),
        ("runtime.status", "ibrobot_msgs/msg/RuntimeStatus", ()),
        (
            "joint.hand_stream",
            "std_msgs/msg/Float64MultiArray",
            tuple(f"{s}_{p}" for s in ("left", "right") for p in HAND_PARTS),
        ),
    )
    identity = {"runtime_name": "mock_restore", "runtime_version": "1.0"}
    description = {
        "robot": identity,
        "interfaces": {
            name: {
                "kind": "topic",
                "direction": "subscribe" if name == "joint.hand_stream" else "publish",
                "message_type": kind,
                "endpoint": f"/restore_test/{name.replace('.', '_')}",
                "joint_names": list(names),
                "modes": ["policy_stream"],
                "qos": {"depth": 10, "reliability": "reliable"},
            }
            for name, kind, names in descriptions
        },
    }
    types = {
        "sensor_msgs/msg/JointState": JointState,
        "ibrobot_msgs/msg/JointCommandState": JointCommandState,
        "ibrobot_msgs/msg/RuntimeStatus": RuntimeStatus,
    }
    publishers = {
        name: publisher.create_publisher(types[kind], description["interfaces"][name]["endpoint"], 10)
        for name, kind, _ in descriptions
        if kind in types
    }
    owner, stopped, received = ["task"], [False], []
    stamp_lag_sec = [0]

    def status_request(_request, response):
        response.schema_version = 1
        response.control_plane_ready = response.motion_authorized = response.busy = True
        response.active_task_id = owner[0]
        return response

    publisher.create_service(GetSkillGatewayStatus, "/restore_test/gateway_status", status_request)
    publisher.create_subscription(
        Float64MultiArray,
        description["interfaces"]["joint.hand_stream"]["endpoint"],
        lambda message: received.append(list(message.data)),
        10,
    )

    def publish():
        stamp = publisher.get_clock().now().to_msg()
        stamp.sec -= stamp_lag_sec[0]
        for name, _, names in descriptions[:3]:
            message = JointState()
            message.header.stamp, message.name = stamp, list(names)
            message.position = [0.1] * len(names)
            publishers[name].publish(message)
        publishers["joint.arm_command_state"].publish(
            JointCommandState(
                name=list(LEFT_ARM + RIGHT_ARM), position=[0.0] * 14, stiffness=[40.0] * 14, damping=[2.0] * 14
            )
        )
        publishers["runtime.status"].publish(
            RuntimeStatus(
                **identity,
                lifecycle="ACTIVE",
                active_mode="policy_stream",
                stop_latched=stopped[0],
                stamp=stamp,
                interface_description_json=json.dumps(description),
            )
        )

    publisher.create_timer(0.02, publish)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(publisher)
    executor.add_node(consumer)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    ports = None
    try:
        ports = RuntimeRestorePorts(consumer, description, gateway_service="/restore_test/gateway_status")
        ports.prepare(RestoreContext("task", time.monotonic() + 5, lambda: False))
        observation = ports.observe()
        assert observation.gateway.busy and observation.gateway.active_task_id == "task"
        assert observation.feedback.right_hand.tolist() == [0.1] * 10
        # A separate request after an idle interval must not reuse the earlier
        # task's ownership, including a result already completed by the executor.
        owner[0] = "next-task"
        ports._gateway_at -= 13.46
        ports._requested_at -= 13.46
        ports.prepare(RestoreContext("next-task", time.monotonic() + 5, lambda: False))
        observation = ports.observe()
        assert observation.gateway.active_task_id == "next-task"
        assert 0 <= observation.gateway.age_s <= 1.0
        # Model a geometry computation that stops draining while DDS continues
        # receiving. Per-sample decoding has a cost: replaying the full history
        # would make the latest measurement stale before observe() returns.
        record = ports.history.record

        def decode(*args, **kwargs):
            time.sleep(0.004)
            return record(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(ports.history, "record", decode)
            time.sleep(0.5)
            assert 0 <= ports.observe().feedback_age_s <= 0.1
        # Receiving an old source timestamp must still fail freshness, even
        # though its middleware receipt is recent.
        stamp_lag_sec[0] = 1
        time.sleep(0.2)
        assert ports.observe().feedback_age_s > 0.1
        stamp_lag_sec[0] = 0
        wait_for(lambda: 0 <= ports.observe().feedback_age_s <= 0.1)
        wait_for(lambda: ports.refusal() is None)
        command = [0.2] * 10 + [0.4] * 10
        ports.publish(np.array(command[:10]), np.array(command[10:]))
        wait_for(lambda: received and received[-1] == command)
        ports.release()
        # Let already queued DDS messages drain, then verify the keepalive has stopped.
        time.sleep(0.15)
        count = len(received)
        time.sleep(0.2)
        assert len(received) == count
        owner[0] = "another-task"
        wait_for(lambda: ports.observe().gateway.active_task_id == "another-task")
        stopped[0] = True
        wait_for(lambda: ports.refusal() and "latched" in ports.refusal())
        assert not ports.observe().runtime.ok
    finally:
        if ports is not None:
            ports.close()
        executor.shutdown(timeout_sec=5)
        thread.join(timeout=5)
        consumer.destroy_node()
        publisher.destroy_node()
        rclpy.shutdown()
    assert not thread.is_alive()


def test_delayed_gateway_response_keeps_request_age(monkeypatch):
    from aimdk_grasp import hand_restore_ros

    ports = object.__new__(RuntimeRestorePorts)
    ports._requested_at, ports._gateway_at = 10.0, float("-inf")
    ports._pending = Future()
    reply = SimpleNamespace(active_task_id="old-owner")
    ports._pending.set_result(reply)
    ports._client = SimpleNamespace(service_is_ready=lambda: False)
    ports._gateway = None
    monkeypatch.setattr(hand_restore_ros.time, "monotonic", lambda: 12.0)
    ports._poll_gateway()
    assert ports._gateway is reply and ports._gateway_at == 10.0


@pytest.mark.parametrize("old_pending", ["none", "completed", "late"])
@pytest.mark.parametrize("respond", [True, False])
def test_prepare_requires_new_ownership_rpc_after_idle(monkeypatch, old_pending, respond):
    from aimdk_grasp import hand_restore_ros

    clock = [100.0]
    monkeypatch.setattr(hand_restore_ros.time, "monotonic", lambda: clock[0])
    ports = object.__new__(RuntimeRestorePorts)
    old = Future()
    previous = SimpleNamespace(active_task_id="previous-task")
    current = SimpleNamespace(active_task_id="current-task")
    if old_pending == "completed":
        old.set_result(previous)
    ports._pending = None if old_pending == "none" else old
    ports._requested_at = ports._gateway_at = 86.54  # Reported 13.46 s idle-cache failure.
    ports._gateway = previous
    ports._status = SimpleNamespace(lifecycle="ACTIVE", faults=[], stop_latched=False, active_mode="policy_stream")
    ports._board = object()
    ports._hands = SimpleNamespace(refusal=lambda: None)
    ports._subscriptions = SimpleNamespace(drain=lambda: None)
    ports.history = SimpleNamespace(latest=lambda: object())
    ports._request = lambda **kwargs: kwargs
    removed, issued = [], []

    def request(_message):
        future = Future()
        issued.append((clock[0], future))
        return future

    ports._client = SimpleNamespace(
        service_is_ready=lambda: True, call_async=request, remove_pending_request=removed.append
    )

    def wait(seconds):
        clock[0] += seconds
        if not old.done():
            old.set_result(previous)  # Late detached answer must not be accepted.
        if respond and clock[0] >= 100.04 and not issued[-1][1].done():
            issued[-1][1].set_result(current)

    ports.wait = wait
    ctx = RestoreContext("current-task", 110.0, lambda: False)
    if respond:
        ports.prepare(ctx)
        assert ports._gateway is current
        assert ports._gateway_at == issued[0][0] >= 100.0
        assert 100.04 <= clock[0] < 100.1
    else:
        with pytest.raises(RestoreError, match="status unavailable"):
            ports.prepare(ctx)
        assert ports._gateway is None
    if old_pending != "none":
        assert removed[0] is old
