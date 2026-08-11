import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from robot_navigation.global_localization_gate import GlobalLocalizationGate


class AmclGlobalLocalization(Node):
    def __init__(self) -> None:
        super().__init__("amcl_global_localization")
        self.declare_parameter("startup_delay", 3.0)
        self.declare_parameter("required_scans", 10)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("service_name", "/reinitialize_global_localization")
        startup_delay = float(self.get_parameter("startup_delay").value)
        required_scans = int(self.get_parameter("required_scans").value)
        scan_topic = str(self.get_parameter("scan_topic").value)
        service_name = str(self.get_parameter("service_name").value)
        self._gate = GlobalLocalizationGate(startup_delay, required_scans)
        self._start_time = self.get_clock().now()
        self._request_pending = False
        self._client = self.create_client(Empty, service_name)
        self._scan_subscription = self.create_subscription(
            LaserScan, scan_topic, self._on_scan, qos_profile_sensor_data
        )
        self._timer = self.create_timer(0.5, self._try_global_localization)

    def _on_scan(self, _message: LaserScan) -> None:
        self._gate.record_scan()

    def _try_global_localization(self) -> None:
        elapsed = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        if self._request_pending or not self._gate.should_trigger(elapsed) or not self._client.service_is_ready():
            return
        self._request_pending = True
        future = self._client.call_async(Empty.Request())
        future.add_done_callback(self._on_request_done)

    def _on_request_done(self, future) -> None:
        self._request_pending = False
        try:
            future.result()
        except Exception as exc:
            self.get_logger().error(f"AMCL global localization request failed: {exc}")
            return
        self._gate.mark_triggered()
        self._timer.cancel()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AmclGlobalLocalization()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
