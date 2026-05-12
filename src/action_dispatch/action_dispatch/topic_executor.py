#!/usr/bin/env python3
"""
Minimal Topic Executor.
Publishes actions to ros2_control topics based on contract specifications.
"""

from typing import Any

import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from robot_config.tracing_utils import create_trace_logger

_trace = create_trace_logger("ib_trace.execute")


class TopicExecutor:
    """
    Topic-based action executor for high-frequency position control.
    Uses action_specs from contract to route actions to correct topics.
    """

    def __init__(self, node: Node, config: dict[str, Any]):
        self.node = node
        self.action_specs = config.get("action_specs", [])
        self._publishers: dict[str, Any] = {}

        # Use Reliable delivery so ros2_control command subscribers accept live action topics.
        self._qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=1)

    def initialize(self) -> bool:
        """Initialize publishers based on contract."""
        for spec in self.action_specs:
            topic = spec.topic
            if not topic:
                continue

            if "Float64MultiArray" in spec.ros_type:
                pub = self.node.create_publisher(Float64MultiArray, topic, self._qos)
                self._publishers[topic] = {"pub": pub, "type": "float", "spec": spec}
            elif "JointTrajectory" in spec.ros_type:
                pub = self.node.create_publisher(JointTrajectory, topic, self._qos)
                self._publishers[topic] = {"pub": pub, "type": "trajectory", "spec": spec}

            self.node.get_logger().info(f"Created publisher for {topic}")
        return True

    def execute(self, action: np.ndarray, metadata: dict[str, Any] | None = None) -> bool:
        """Route action to publishers."""
        metadata = metadata or {}
        request_id = str(metadata.get("request_id", ""))
        execute_index = int(metadata.get("execute_index", -1))
        queue_size = int(metadata.get("queue_size", -1))

        # Flat tracking of index in the action vector
        current_idx = 0

        for topic, info in self._publishers.items():
            spec = info["spec"]

            # Determine how many joints this topic expects
            num_joints = len(spec.names) if spec.names else 0

            # 1. Slice action based on expected joint count
            if num_joints > 0:
                data = action[current_idx : current_idx + num_joints]
                current_idx += num_joints
            else:
                data = action

            # 2. Convert to list of pure Python floats
            data_list = [float(x) for x in data.ravel()]

            # 3. Publish
            if info["type"] == "float":
                msg = Float64MultiArray(data=data_list)
                info["pub"].publish(msg)
            elif info["type"] == "trajectory":
                traj = JointTrajectory()
                point = JointTrajectoryPoint(positions=data_list)
                point.time_from_start.nanosec = 10000000  # 10ms
                traj.points.append(point)
                info["pub"].publish(traj)
            _trace.info(
                "[action_topic_publish] request_id=%s index=%d topic=%s values=%d queue_size=%d",
                request_id,
                execute_index,
                topic,
                len(data_list),
                queue_size,
            )
        return True
