"""Relay /cmd_vel to /base_controller/cmd_vel_unstamped for Gazebo E2E tests."""

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class CmdVelRelay(Node):
    def __init__(self):
        super().__init__("cmd_vel_relay")
        self.pub = self.create_publisher(Twist, "/base_controller/cmd_vel_unstamped", 10)
        self.sub = self.create_subscription(Twist, "/cmd_vel", self._cb, 10)

    def _cb(self, msg: Twist):
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = CmdVelRelay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
