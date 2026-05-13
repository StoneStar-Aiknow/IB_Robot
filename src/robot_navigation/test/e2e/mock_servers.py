"""Mock servers for E2E tests.

Provides real rclpy action/service servers that auto-succeed, so
the pipeline under test interacts with real ROS2 middleware.

Usage:
    mock_nav = MockNavigateToPoseServer(executor)
    mock_eval = MockTriggerServer("/action_dispatcher/start_evaluate", executor)
    mock_stop = MockTriggerServer("/action_dispatcher/stop_evaluate", executor)
    mock_hotwords = MockSetHotwordsServer(executor)
    executor.spin_once()  # let callbacks fire
"""

import threading

from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

try:
    from ibrobot_msgs.srv import SetHotwords
except ImportError:
    SetHotwords = None


class MockNavigateToPoseServer:
    """Real NavigateToPose action server that auto-succeeds.

    Attributes:
        received_goals: list of NavigateToPose.Goal received in order.
        succeeded_goals: list of goals that were auto-succeeded.
    """

    def __init__(
        self,
        executor: MultiThreadedExecutor,
        action_name: str = "navigate_to_pose",
        delay_sec: float = 0.0,
    ):
        self._node = Node("mock_nav_server")
        self._cb_group = ReentrantCallbackGroup()
        self._delay_sec = delay_sec
        self.received_goals: list = []
        self.succeeded_goals: list = []
        self._lock = threading.Lock()

        self._server = ActionServer(
            self._node,
            NavigateToPose,
            action_name,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._cb_group,
        )
        executor.add_node(self._node)

    def _goal_callback(self, goal_request):
        with self._lock:
            self.received_goals.append(goal_request)
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        import time

        try:
            if self._delay_sec > 0:
                # Busy-wait in small increments so cancel is detected promptly
                deadline = time.monotonic() + self._delay_sec
                while time.monotonic() < deadline:
                    if goal_handle.is_cancel_requested:
                        goal_handle.canceled()
                        result = NavigateToPose.Result()
                        return result
                    time.sleep(0.05)

            goal_handle.succeed()
            with self._lock:
                self.succeeded_goals.append(goal_handle)
        except Exception:
            import logging

            logging.getLogger("mock_servers").exception("execute_callback failed")
        result = NavigateToPose.Result()
        return result

    def destroy(self):
        self._server.destroy()
        self._node.destroy_node()


class MockTriggerServer:
    """Real Trigger service server that records calls and auto-succeeds.

    Attributes:
        calls: list of Trigger.Request objects received.
    """

    def __init__(
        self,
        service_name: str,
        executor: MultiThreadedExecutor,
    ):
        self._node = Node(f"mock_trigger_{service_name.replace('/', '_')}")
        self._cb_group = ReentrantCallbackGroup()
        self.calls: list = []
        self._lock = threading.Lock()

        self._srv = self._node.create_service(
            Trigger,
            service_name,
            self._callback,
            callback_group=self._cb_group,
        )
        executor.add_node(self._node)

    def _callback(self, request, response):
        with self._lock:
            self.calls.append(request)
        response.success = True
        response.message = "mock success"
        return response

    def destroy(self):
        self._node.destroy_service(self._srv)
        self._node.destroy_node()


class MockSetHotwordsServer:
    """Real SetHotwords service server that records calls and auto-succeeds.

    Attributes:
        calls: list of (hotwords, boost_scores) tuples received.
    """

    def __init__(self, executor: MultiThreadedExecutor):
        if SetHotwords is None:
            self._node = None
            self.calls = []
            return

        self._node = Node("mock_set_hotwords")
        self._cb_group = ReentrantCallbackGroup()
        self.calls: list = []
        self._lock = threading.Lock()

        self._srv = self._node.create_service(
            SetHotwords,
            "/voice_asr_node/set_hotwords",
            self._callback,
            callback_group=self._cb_group,
        )
        executor.add_node(self._node)

    def _callback(self, request, response):
        with self._lock:
            self.calls.append((list(request.hotwords), list(request.boost_scores)))
        response.success = True
        response.error_message = ""
        return response

    def destroy(self):
        if self._node is not None:
            self._node.destroy_service(self._srv)
            self._node.destroy_node()
