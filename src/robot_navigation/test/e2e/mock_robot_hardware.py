"""MockRobotHardware for Layer 2 Nav2 E2E tests.

Lightweight robot simulator that:
  - Subscribes /cmd_vel (Twist)
  - Integrates pose (x, y, theta) at 20Hz
  - Publishes /odom (Odometry) + TF (odom -> base_link)
  - Publishes /scan (LaserScan, all max-range for open space)
  - Publishes /joint_states (via FK inverse of cmd_vel)

This replaces Gazebo for testing the full Nav2 stack without GPU.
"""

import math

from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState, LaserScan
from tf2_ros import TransformBroadcaster

from robot_navigation.cmd_vel_bridge_node import _body_to_wheel_radps

# Default parameters matching LeKiwi robot
WHEEL_RADIUS = 0.05
BASE_RADIUS = 0.125


class MockRobotHardware(Node):
    """Lightweight robot simulator for Nav2 E2E testing.

    Subscribes /cmd_vel, integrates pose, publishes /odom + /scan + /joint_states + TF.
    """

    def __init__(self, **kwargs):
        super().__init__("mock_robot_hardware", **kwargs)

        # Parameters
        self.declare_parameter("wheel_radius", WHEEL_RADIUS)
        self.declare_parameter("base_radius", BASE_RADIUS)
        self.declare_parameter("scan_range_max", 10.0)
        self.declare_parameter("scan_angle_min", -math.pi)
        self.declare_parameter("scan_angle_max", math.pi)
        self.declare_parameter("scan_angle_increment", math.pi / 180.0)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("publish_rate", 20.0)

        self.wheel_radius = self.get_parameter("wheel_radius").value
        self.base_radius = self.get_parameter("base_radius").value
        self.scan_range_max = self.get_parameter("scan_range_max").value
        self.scan_angle_min = self.get_parameter("scan_angle_min").value
        self.scan_angle_max = self.get_parameter("scan_angle_max").value
        self.scan_angle_increment = self.get_parameter("scan_angle_increment").value
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        publish_rate = self.get_parameter("publish_rate").value

        # State
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.pose_theta = 0.0
        self.current_vx = 0.0
        self.current_vy = 0.0
        self.current_vtheta = 0.0
        self.last_update_time = self.get_clock().now().nanoseconds / 1e9

        # QoS profiles
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        qos_best_effort = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # Subscriber
        self.cmd_vel_sub = self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_callback, qos_reliable)

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, "/odom", qos_reliable)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", qos_best_effort)
        self.joint_states_pub = self.create_publisher(JointState, "/joint_states", qos_best_effort)

        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Timer
        timer_period = 1.0 / publish_rate
        self._timer = self.create_timer(timer_period, self._update)

        self.get_logger().info(
            f"MockRobotHardware started (rate={publish_rate}Hz, scan_range_max={self.scan_range_max}m)"
        )

    def _cmd_vel_callback(self, msg: Twist):
        """Cache latest velocity command."""
        self.current_vx = msg.linear.x
        self.current_vy = msg.linear.y
        self.current_vtheta = msg.angular.z

    def _update(self):
        """Timer callback: integrate pose, publish all topics."""
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.last_update_time

        if dt <= 0 or dt > 1.0:
            self.last_update_time = now
            return

        # Integrate pose in world frame
        cos_theta = math.cos(self.pose_theta)
        sin_theta = math.sin(self.pose_theta)

        self.pose_x += (self.current_vx * cos_theta - self.current_vy * sin_theta) * dt
        self.pose_y += (self.current_vx * sin_theta + self.current_vy * cos_theta) * dt
        self.pose_theta += self.current_vtheta * dt
        self.pose_theta = math.atan2(math.sin(self.pose_theta), math.cos(self.pose_theta))

        self.last_update_time = now

        ros_now = self.get_clock().now().to_msg()

        self._publish_odom(ros_now)
        self._publish_tf(ros_now)
        self._publish_scan(ros_now)
        self._publish_joint_states(ros_now)

    def _publish_odom(self, stamp):
        """Publish Odometry message."""
        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame

        odom.pose.pose.position.x = self.pose_x
        odom.pose.pose.position.y = self.pose_y
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation.z = math.sin(self.pose_theta / 2.0)
        odom.pose.pose.orientation.w = math.cos(self.pose_theta / 2.0)

        odom.twist.twist.linear.x = self.current_vx
        odom.twist.twist.linear.y = self.current_vy
        odom.twist.twist.angular.z = self.current_vtheta

        self.odom_pub.publish(odom)

    def _publish_tf(self, stamp):
        """Publish odom -> base_link TF."""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = self.pose_x
        t.transform.translation.y = self.pose_y
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(self.pose_theta / 2.0)
        t.transform.rotation.w = math.cos(self.pose_theta / 2.0)
        self.tf_broadcaster.sendTransform(t)

    def _publish_scan(self, stamp):
        """Publish fake LaserScan (all max-range for open space)."""
        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = self.base_frame
        scan.angle_min = self.scan_angle_min
        scan.angle_max = self.scan_angle_max
        scan.angle_increment = self.scan_angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 0.1
        scan.range_min = 0.1
        scan.range_max = self.scan_range_max

        num_readings = int((scan.angle_max - scan.angle_min) / scan.angle_increment) + 1
        scan.ranges = [self.scan_range_max] * num_readings

        self.scan_pub.publish(scan)

    def _publish_joint_states(self, stamp):
        """Publish joint_states from current velocity via IK."""
        wheel_speeds = _body_to_wheel_radps(
            self.current_vx,
            self.current_vy,
            self.current_vtheta,
            self.wheel_radius,
            self.base_radius,
            100.0,  # no max scaling limit for mock
        )

        js = JointState()
        js.header.stamp = stamp
        js.name = ["7", "8", "9"]
        js.velocity = wheel_speeds

        self.joint_states_pub.publish(js)
