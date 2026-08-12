import math
import threading
import time
from enum import Enum

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient, ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from ibrobot_msgs.action import ExecuteNavigation
from robot_navigation.navigation_command_core import (
    CommandType,
    GoalValidationError,
    StopVelocityGate,
    quaternion_to_yaw,
    resolve_navigation_target,
)


class NavigationState(str, Enum):
    IDLE = "idle"
    RESOLVING = "resolving"
    SENDING = "sending"
    RUNNING = "running"
    CANCELING = "canceling"
    STOPPING = "stopping"
    FAULT = "fault"


class NavigationCommandServer(Node):
    def __init__(self) -> None:
        super().__init__("navigation_command_server")
        defaults = {
            "action_name": "/navigation/execute",
            "cancel_service_name": "/navigation/cancel_current",
            "nav2_action_name": "/navigate_to_pose",
            "stop_velocity_topic": "/cmd_vel_safe",
            "global_frame": "map",
            "base_frame": "base_link",
            "nav2_server_timeout": 5.0,
            "tf_timeout": 1.0,
            "cancel_timeout": 10.0,
            "linear_stop_threshold": 0.01,
            "angular_stop_threshold": 0.05,
            "stop_stable_duration": 0.5,
            "stop_confirmation_timeout": 3.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
            setattr(self, name, self.get_parameter(name).value)

        self._callback_group = ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self.state = NavigationState.IDLE
        self._generation = 0
        self._nav_goal_handle = None
        self._cancel_requested = threading.Event()
        self._cancel_complete = threading.Event()
        self._stop_confirmed = threading.Event()
        self._stop_gate = StopVelocityGate(
            linear_threshold=float(self.linear_stop_threshold),
            angular_threshold=float(self.angular_stop_threshold),
            stable_duration=float(self.stop_stable_duration),
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._nav_client = ActionClient(
            self, NavigateToPose, self.nav2_action_name, callback_group=self._callback_group
        )
        self._action_server = ActionServer(
            self,
            ExecuteNavigation,
            self.action_name,
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self._cancel_service = self.create_service(
            Trigger,
            self.cancel_service_name,
            self._handle_cancel_current,
            callback_group=self._callback_group,
        )
        self._stop_velocity_sub = self.create_subscription(
            Twist,
            self.stop_velocity_topic,
            self._stop_velocity_callback,
            10,
            callback_group=self._callback_group,
        )

    def destroy_node(self):
        self._action_server.destroy()
        return super().destroy_node()

    def _goal_callback(self, _goal_request) -> GoalResponse:
        return GoalResponse.ACCEPT

    def _cancel_callback(self, _goal_handle) -> CancelResponse:
        self._request_cancel()
        return CancelResponse.ACCEPT

    def _handle_cancel_current(self, _request, response):
        with self._lock:
            if self.state == NavigationState.IDLE:
                response.success = True
                response.message = "Navigation is already idle"
                return response

        self._request_cancel()
        timeout = float(self.cancel_timeout) + float(self.stop_confirmation_timeout)
        response.success = self._cancel_complete.wait(timeout=timeout)
        response.message = (
            "Navigation canceled and velocity command is stable at zero"
            if response.success
            else f"Navigation cancellation did not complete within {timeout:.1f} seconds"
        )
        return response

    def _request_cancel(self) -> None:
        self._cancel_requested.set()
        with self._lock:
            nav_goal_handle = self._nav_goal_handle
            if self.state not in (NavigationState.IDLE, NavigationState.FAULT):
                self.state = NavigationState.CANCELING
        if nav_goal_handle is not None:
            nav_goal_handle.cancel_goal_async()

    def _stop_velocity_callback(self, msg: Twist) -> None:
        with self._lock:
            if self.state != NavigationState.STOPPING:
                return
        if self._stop_gate.observe(
            vx=msg.linear.x,
            vy=msg.linear.y,
            wz=msg.angular.z,
            now=time.monotonic(),
        ):
            self._stop_confirmed.set()

    def _start_goal(self) -> int | None:
        with self._lock:
            if self.state != NavigationState.IDLE:
                return None
            self._generation += 1
            generation = self._generation
            self.state = NavigationState.RESOLVING
            self._nav_goal_handle = None
        self._cancel_requested.clear()
        self._cancel_complete.clear()
        self._stop_confirmed.clear()
        self._stop_gate.reset()
        return generation

    def _finish(self, generation: int, state: NavigationState = NavigationState.IDLE) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._nav_goal_handle = None
            self.state = state
        self._cancel_requested.clear()

    def _resolve_target(self, request) -> PoseStamped:
        command_type = int(request.command_type)
        pose = request.target_pose.pose
        if command_type == CommandType.ABSOLUTE_POSE:
            yaw = quaternion_to_yaw(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
            base_x = base_y = base_yaw = 0.0
        else:
            try:
                transform = self._tf_buffer.lookup_transform(
                    self.global_frame,
                    self.base_frame,
                    rclpy.time.Time(),
                    timeout=Duration(seconds=float(self.tf_timeout)),
                )
            except TransformException as exc:
                raise GoalValidationError(f"Transform map -> base_link is unavailable: {exc}") from exc
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            base_x = translation.x
            base_y = translation.y
            base_yaw = quaternion_to_yaw(rotation.x, rotation.y, rotation.z, rotation.w)
            yaw = 0.0

        x, y, yaw = resolve_navigation_target(
            command_type=command_type,
            value=float(request.value),
            target_frame=request.target_pose.header.frame_id,
            target_x=pose.position.x,
            target_y=pose.position.y,
            target_yaw=yaw,
            base_x=base_x,
            base_y=base_y,
            base_yaw=base_yaw,
        )
        target = PoseStamped()
        target.header.frame_id = self.global_frame
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.orientation.z = math.sin(yaw / 2.0)
        target.pose.orientation.w = math.cos(yaw / 2.0)
        return target

    def _execute_callback(self, goal_handle):
        result = ExecuteNavigation.Result()
        generation = self._start_goal()
        if generation is None:
            result.error_code = ExecuteNavigation.Result.BUSY
            result.message = "Another navigation command is active"
            goal_handle.abort()
            return result

        try:
            target = self._resolve_target(goal_handle.request)
            result.resolved_target_pose = target
        except GoalValidationError as exc:
            result.error_code = (
                ExecuteNavigation.Result.TF_UNAVAILABLE
                if str(exc).startswith("Transform")
                else ExecuteNavigation.Result.INVALID_GOAL
            )
            result.message = str(exc)
            goal_handle.abort()
            self._finish(generation)
            return result

        if self._cancel_requested.is_set() or goal_handle.is_cancel_requested:
            return self._complete_without_nav_goal(goal_handle, result, generation)
        if not self._wait_for_nav2_server():
            if self._cancel_requested.is_set() or goal_handle.is_cancel_requested:
                return self._complete_without_nav_goal(goal_handle, result, generation)
            result.error_code = ExecuteNavigation.Result.NAV2_UNAVAILABLE
            result.message = "Nav2 NavigateToPose action server is unavailable"
            goal_handle.abort()
            self._finish(generation)
            return result

        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = target
        self.state = NavigationState.SENDING
        send_future = self._nav_client.send_goal_async(
            nav_goal,
            feedback_callback=lambda feedback: self._forward_feedback(goal_handle, feedback, generation),
        )
        nav_goal_handle = self._wait_future(send_future, float(self.nav2_server_timeout))
        if nav_goal_handle is None or not nav_goal_handle.accepted:
            result.error_code = ExecuteNavigation.Result.GOAL_REJECTED
            result.message = "Nav2 rejected the navigation goal"
            goal_handle.abort()
            self._finish(generation)
            return result

        with self._lock:
            if generation == self._generation:
                self._nav_goal_handle = nav_goal_handle
                self.state = NavigationState.RUNNING
        if self._cancel_requested.is_set() or goal_handle.is_cancel_requested:
            self.state = NavigationState.CANCELING
            nav_goal_handle.cancel_goal_async()

        wrapped_result = self._wait_future(nav_goal_handle.get_result_async(), None)
        if wrapped_result is None:
            result.error_code = ExecuteNavigation.Result.INTERNAL_ERROR
            result.message = "Nav2 result wait failed"
            goal_handle.abort()
            self._finish(generation, NavigationState.FAULT)
            return result
        if wrapped_result.status == GoalStatus.STATUS_SUCCEEDED:
            result.success = True
            result.error_code = ExecuteNavigation.Result.NONE
            result.message = "Navigation succeeded"
            goal_handle.succeed()
            self._finish(generation)
            return result
        if wrapped_result.status == GoalStatus.STATUS_CANCELED:
            return self._complete_cancellation(goal_handle, result, generation)

        result.error_code = ExecuteNavigation.Result.NAVIGATION_ABORTED
        result.message = f"Nav2 navigation ended with status {wrapped_result.status}"
        goal_handle.abort()
        self._finish(generation)
        return result

    def _complete_without_nav_goal(self, goal_handle, result, generation: int):
        self.state = NavigationState.STOPPING
        self._stop_gate.reset()
        if self._stop_confirmed.wait(timeout=float(self.stop_confirmation_timeout)):
            result.error_code = ExecuteNavigation.Result.NAVIGATION_CANCELED
            result.message = "Navigation stopped before a Nav2 goal was sent"
            goal_handle.abort()
            self._finish(generation)
            self._cancel_complete.set()
        else:
            result.error_code = ExecuteNavigation.Result.STOP_TIMEOUT
            result.message = "Velocity command did not become stable at zero"
            goal_handle.abort()
            self._finish(generation, NavigationState.FAULT)
        return result

    def _complete_cancellation(self, goal_handle, result, generation: int):
        self.state = NavigationState.STOPPING
        self._stop_gate.reset()
        self._stop_confirmed.clear()
        if self._stop_confirmed.wait(timeout=float(self.stop_confirmation_timeout)):
            result.error_code = ExecuteNavigation.Result.NAVIGATION_CANCELED
            result.message = "Navigation stopped and velocity command is stable at zero"
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
            else:
                goal_handle.abort()
            self._finish(generation)
            self._cancel_complete.set()
        else:
            result.error_code = ExecuteNavigation.Result.STOP_TIMEOUT
            result.message = "Nav2 canceled, but velocity command did not become stable at zero"
            goal_handle.abort()
            self._finish(generation, NavigationState.FAULT)
        return result

    def _forward_feedback(self, goal_handle, feedback_message, generation: int) -> None:
        with self._lock:
            if generation != self._generation:
                return
            state = self.state.value
        nav_feedback = feedback_message.feedback
        feedback = ExecuteNavigation.Feedback()
        feedback.state = state
        feedback.distance_remaining = nav_feedback.distance_remaining
        feedback.estimated_time_remaining = nav_feedback.estimated_time_remaining
        feedback.number_of_recoveries = nav_feedback.number_of_recoveries
        goal_handle.publish_feedback(feedback)

    @staticmethod
    def _wait_future(future, timeout: float | None):
        completed = threading.Event()
        future.add_done_callback(lambda _future: completed.set())
        if not completed.wait(timeout=timeout):
            return None
        try:
            return future.result()
        except Exception:
            return None

    def _wait_for_nav2_server(self) -> bool:
        deadline = time.monotonic() + float(self.nav2_server_timeout)
        while not self._cancel_requested.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            if self._nav_client.wait_for_server(timeout_sec=min(0.1, remaining)):
                return True
        return False


def main(args=None):
    rclpy.init(args=args)
    node = NavigationCommandServer()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
