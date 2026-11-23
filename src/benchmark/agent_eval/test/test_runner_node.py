"""ROS wiring: the runtime's latched topics and robot state reach the EventBook; requests go out."""

from __future__ import annotations

import json
import os
import time

import rclpy
from agent_eval_testkit import fixture_body


def wait_for(predicate, timeout_s=5.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_runner_node_hears_the_runtime_and_publishes_requests():
    assert os.environ.get("IBROBOT_TEST_ROS_DOMAIN_ID"), "needs the per-process ROS domain (root conftest)"
    from agent_eval.runner_node import RosRunner, Topics
    from geometry_msgs.msg import PoseStamped
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    from std_msgs.msg import String

    rclpy.init()
    try:
        peer = rclpy.create_node("agent_eval_test_peer")
        latched = QoSProfile(
            depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
        )
        session_pub = peer.create_publisher(String, Topics.session, latched)
        start = {"event": "session_start", "session_id": "s0001", "generation": 1, "wall_time": time.time()}
        session_pub.publish(String(data=json.dumps(start)))  # before the runner exists: latched delivery
        ros = RosRunner(fixture_body(), node_name="agent_eval_test_runner")
        try:
            pose_pub = peer.create_publisher(PoseStamped, "/robot_status/left_arm/ee_pose", 10)
            requests = []
            peer.create_subscription(String, Topics.agent_request, lambda m: requests.append(json.loads(m.data)), 10)
            got = []

            def spin_peer_until(predicate):
                return wait_for(lambda: (rclpy.spin_once(peer, timeout_sec=0.02), predicate())[1])

            assert spin_peer_until(lambda: got.append(ros.book.next_episode(0.0)) or got[-1] is not None)
            assert got[-1]["session_id"] == "s0001"
            pose = PoseStamped()
            pose.pose.position.z = 0.9
            pose.pose.orientation.w = 1.0
            assert spin_peer_until(
                lambda: (
                    pose_pub.publish(pose)
                    or ros.book.observation(("s0001", 1), "x", 10).arms["left"].tcp_xyz == (0.0, 0.0, 0.9)
                )
            )
            payload = {"schema_version": 1, "request_id": "r1", "session_id": "s1", "text": "hi"}
            assert spin_peer_until(lambda: ros.publish_request(payload) or bool(requests))
            assert requests[0] == payload
        finally:
            ros.close()
            peer.destroy_node()
    finally:
        rclpy.try_shutdown()
