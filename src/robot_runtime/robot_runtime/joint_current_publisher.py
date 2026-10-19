#!/usr/bin/env python3

"""Simulated joint-current publisher for physics transports (D6b).

A runtime that advertises a ``joint.current`` interface cannot read motor
currents from a physics simulator, so on the Gazebo transport this
robot-agnostic node republishes the simulated joint effort from
``/joint_states`` as ``ibrobot_msgs/msg/JointCurrent`` on the advertised
endpoint, scaled per joint by the profile (default 1, matching the hardware
adapter's effort-as-current semantics). The transport identity in the public
description already marks the stream simulated.
"""

from __future__ import annotations

import sys
from typing import Any

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from ibrobot_msgs.msg import JointCurrent


def joint_currents_from_efforts(joint_names: list[str], message: JointState, scale: list[float]) -> list[float] | None:
    """Per-joint effort × scale in the profile's joint order; None without effort data."""
    if not message.effort:
        return None
    effort_by_name = dict(zip(message.name, message.effort, strict=False))
    currents = []
    for name, factor in zip(joint_names, scale, strict=True):
        effort = effort_by_name.get(name)
        if effort is None:
            return None
        currents.append(float(effort) * float(factor))
    return currents


class JointCurrentPublisher(Node):
    def __init__(self) -> None:
        super().__init__("simulated_joint_current")
        self.declare_parameter("joint_names", [""])
        self.declare_parameter("endpoint", "/joint_currents")
        self.declare_parameter("scale", [1.0])
        self.declare_parameter("joint_states_topic", "/joint_states")
        self._joint_names = [str(name) for name in self.get_parameter("joint_names").value if str(name)]
        self._scale = [float(value) for value in self.get_parameter("scale").value]
        if not self._joint_names:
            raise ValueError("joint_current_publisher requires the 'joint_names' parameter")
        if len(self._scale) not in (0, 1, len(self._joint_names)):
            raise ValueError(
                "joint_current_publisher 'scale' must have one entry per joint "
                f"({len(self._joint_names)}), got {len(self._scale)}"
            )
        if len(self._scale) == 1:
            self._scale = self._scale * len(self._joint_names)
        elif not self._scale:
            self._scale = [1.0] * len(self._joint_names)
        endpoint = str(self.get_parameter("endpoint").value)
        self._pub = self.create_publisher(JointCurrent, endpoint, 10)
        self.create_subscription(
            JointState, str(self.get_parameter("joint_states_topic").value), self._on_joint_state, 10
        )

    def _on_joint_state(self, message: JointState) -> None:
        currents = joint_currents_from_efforts(self._joint_names, message, self._scale)
        if currents is None:
            return
        out = JointCurrent()
        out.header = message.header
        if out.header.stamp.sec == 0 and out.header.stamp.nanosec == 0:
            out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(self._joint_names)
        out.current = currents
        self._pub.publish(out)


def main(argv: Any = None) -> int:
    rclpy.init(args=argv)
    node = JointCurrentPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
