"""Temporary ROS interfaces for testing the SLAM/Nav2 integration contract."""

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster


class MockSlamNavInterfaces(Node):
    """Publish deterministic localization data and readiness responses."""

    def __init__(self) -> None:
        super().__init__("mock_slam_nav_interfaces")
        self.declare_parameter("slam_ready", True)
        self.declare_parameter("navigation_ready", True)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("odom_topic", "/odometry/filtered")
        self.declare_parameter("publish_rate_hz", 20.0)
        self._slam_ready = bool(self.get_parameter("slam_ready").value)
        self._navigation_ready = bool(self.get_parameter("navigation_ready").value)
        self._map_frame = str(self.get_parameter("map_frame").value)
        self._odom_frame = str(self.get_parameter("odom_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._camera_frame = str(self.get_parameter("camera_frame").value)
        self._tf = TransformBroadcaster(self)
        self._static_tf = StaticTransformBroadcaster(self)
        self._odom_pub = self.create_publisher(Odometry, str(self.get_parameter("odom_topic").value), 10)
        self.create_service(Trigger, "/slam/readiness", self._slam_readiness)
        self.create_service(Trigger, "/navigation/readiness", self._navigation_readiness)
        self._publish_static_tf()
        rate = max(float(self.get_parameter("publish_rate_hz").value), 1.0)
        self.create_timer(1.0 / rate, self._publish_localization)

    def _response(self, ready: bool, label: str, response: Trigger.Response) -> Trigger.Response:
        response.success = ready
        response.message = f"{label} mock ready" if ready else f"{label} mock not ready"
        return response

    def _slam_readiness(self, _request, response):
        return self._response(self._slam_ready, "SLAM", response)

    def _navigation_readiness(self, _request, response):
        return self._response(self._navigation_ready, "navigation", response)

    def _publish_static_tf(self) -> None:
        transform = TransformStamped()
        transform.header.frame_id = self._base_frame
        transform.child_frame_id = self._camera_frame
        transform.transform.rotation.w = 1.0
        self._static_tf.sendTransform(transform)

    def _publish_localization(self) -> None:
        now = self.get_clock().now().to_msg()
        map_to_odom = TransformStamped()
        map_to_odom.header.stamp = now
        map_to_odom.header.frame_id = self._map_frame
        map_to_odom.child_frame_id = self._odom_frame
        map_to_odom.transform.rotation.w = 1.0
        self._tf.sendTransform(map_to_odom)

        odom_to_base = TransformStamped()
        odom_to_base.header.stamp = now
        odom_to_base.header.frame_id = self._odom_frame
        odom_to_base.child_frame_id = self._base_frame
        odom_to_base.transform.rotation.w = 1.0
        self._tf.sendTransform(odom_to_base)

        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.orientation.w = 1.0
        self._odom_pub.publish(odom)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockSlamNavInterfaces()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
