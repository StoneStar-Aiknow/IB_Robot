"""Ground-truth odometry relay for robot_config navigation simulation tests."""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class GtOdomNode(Node):
    def __init__(self):
        super().__init__("gt_odom_node")
        self.declare_parameter("spawn_x", -1.5)
        self.declare_parameter("spawn_y", -1.5)
        self.spawn_x = self.get_parameter("spawn_x").value
        self.spawn_y = self.get_parameter("spawn_y").value

        self.br = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, "/odom", 10)
        self.sub = self.create_subscription(Odometry, "/model/lekiwi/odometry", self._cb, 10)

    def _cb(self, msg: Odometry):
        odom = Odometry()
        odom.header.stamp = msg.header.stamp
        odom.header.frame_id = "odom"
        odom.child_frame_id = "base_footprint"
        odom.pose = msg.pose
        odom.pose.pose.position.x -= self.spawn_x
        odom.pose.pose.position.y -= self.spawn_y
        odom.pose.pose.position.z = 0.0
        odom.twist = msg.twist
        self.pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = msg.header.stamp
        transform.header.frame_id = "odom"
        transform.child_frame_id = "base_footprint"
        transform.transform.translation.x = odom.pose.pose.position.x
        transform.transform.translation.y = odom.pose.pose.position.y
        transform.transform.translation.z = 0.0
        transform.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(transform)


def main(args=None):
    rclpy.init(args=args)
    node = GtOdomNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
