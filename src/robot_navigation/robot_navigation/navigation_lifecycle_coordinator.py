"""Start the Nav2 lifecycle manager after the existing warm-up delay."""

import time

import rclpy
from nav2_msgs.srv import ManageLifecycleNodes
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node


def join_service_name(namespace: str, service_name: str) -> str:
    """Join an optional namespace to an absolute ROS service name."""
    namespace = namespace.strip("/")
    service_name = "/" + service_name.lstrip("/")
    return f"/{namespace}{service_name}" if namespace else service_name


class NavigationLifecycleCoordinator(Node):
    def __init__(self) -> None:
        super().__init__("navigation_lifecycle_coordinator")
        self.declare_parameter("startup_delay_sec", 10.0)
        self.declare_parameter("service_name", "/lifecycle_manager_navigation/manage_nodes")
        self.declare_parameter("namespace", "")
        self.declare_parameter("service_wait_timeout_sec", 2.0)
        self.declare_parameter("request_timeout_sec", 30.0)
        self.declare_parameter("retry_count", 3)
        self.declare_parameter("retry_interval_sec", 1.0)

        namespace = str(self.get_parameter("namespace").value)
        service_name = str(self.get_parameter("service_name").value)
        self._service_name = join_service_name(namespace, service_name)
        self._startup_delay = float(self.get_parameter("startup_delay_sec").value)
        self._service_wait_timeout = float(self.get_parameter("service_wait_timeout_sec").value)
        self._request_timeout = float(self.get_parameter("request_timeout_sec").value)
        self._retry_count = int(self.get_parameter("retry_count").value)
        self._retry_interval = float(self.get_parameter("retry_interval_sec").value)
        self._started_at = time.monotonic()
        self._attempt = 0
        self._attempt_started_at = self._started_at + self._startup_delay
        self._next_attempt_at = self._started_at + self._startup_delay
        self._request_future = None
        self._request_deadline = 0.0
        self._client = self.create_client(ManageLifecycleNodes, self._service_name)
        self._timer = self.create_timer(0.1, self._tick)

    def _tick(self) -> None:
        now = time.monotonic()
        if self._request_future is not None:
            if self._request_future.done():
                self._handle_response()
            elif now >= self._request_deadline:
                self.get_logger().warning("Lifecycle startup is still in progress; waiting without retry")
                self._request_deadline = now + self._request_timeout
            return

        if now < self._next_attempt_at:
            return
        if self._attempt > self._retry_count:
            self.get_logger().error("Lifecycle startup retries exhausted")
            self._timer.cancel()
            return
        if not self._client.service_is_ready():
            if now - self._attempt_started_at >= self._service_wait_timeout:
                self.get_logger().error("Lifecycle manager service is unavailable")
                self._schedule_retry(now)
            return

        request = ManageLifecycleNodes.Request()
        request.command = ManageLifecycleNodes.Request.STARTUP
        self._request_future = self._client.call_async(request)
        self._request_deadline = now + self._request_timeout

    def _handle_response(self) -> None:
        future = self._request_future
        self._request_future = None
        try:
            response = future.result()
        except Exception as exc:
            self.get_logger().error(f"Lifecycle startup failed: {exc}")
            self._schedule_retry(time.monotonic())
            return
        if not response.success:
            self.get_logger().error("Lifecycle manager rejected startup request")
            self._schedule_retry(time.monotonic())
            return

        self.get_logger().info("Navigation lifecycle startup completed")
        self._timer.cancel()
        rclpy.shutdown()

    def _schedule_retry(self, now: float) -> None:
        self._attempt += 1
        self._attempt_started_at = now
        self._next_attempt_at = now + self._retry_interval


def main(args=None) -> None:
    rclpy.init(args=args)
    node = NavigationLifecycleCoordinator()
    try:
        rclpy.spin(node)
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
