"""Ground truth odometry relay node for Gazebo simulation.

Subscribes to the Gazebo OdometryPublisher plugin output (world-frame pose),
subtracts the spawn offset, and republishes as standard ROS /odom topic
and odom→base_footprint TF.

This avoids the 3x Y-axis drift from wheel odometry in DART physics,
which does not simulate omni wheel lateral slip accurately.

Usage:
    ros2 run robot_navigation gt_odom_node --ros-args
        -p spawn_x:=-1.5 -p spawn_y:=-1.5
"""

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

        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = "odom"
        t.child_frame_id = "base_footprint"
        t.transform.translation.x = odom.pose.pose.position.x
        t.transform.translation.y = odom.pose.pose.position.y
        t.transform.translation.z = 0.0
        t.transform.rotation = msg.pose.pose.orientation
        self.br.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    node = GtOdomNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
