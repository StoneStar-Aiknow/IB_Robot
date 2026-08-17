"""Collision-aware dynamic following through Humble Nav2 actions."""

import math
import time

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode
from rclpy.parameter import Parameter
from std_srvs.srv import Trigger
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformException, TransformListener

from ibrobot_msgs.msg import TrackState

from .following_core import FollowGoal, PathReplacementGate, should_replan, stand_off_goal


class DynamicTargetFollowerNode(LifecycleNode):
    """Convert actionable target states into serialized Nav2 path updates."""

    def __init__(self, *, parameter_overrides: list[Parameter] | None = None):
        super().__init__("dynamic_target_follower", parameter_overrides=parameter_overrides)
        self.declare_parameter("enabled", False)
        self.declare_parameter("production_ready", False)
        self.declare_parameter("track_state_topic", "/object_tracker/track_state")
        self.declare_parameter("global_frame", "map")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("target_frame", "odom")
        self.declare_parameter("stand_off_distance_m", 0.8)
        self.declare_parameter("replan_displacement_m", 0.2)
        self.declare_parameter("replan_heading_rad", math.radians(15.0))
        self.declare_parameter("minimum_replan_interval_s", 1.0)
        self.declare_parameter("target_timeout_s", 1.0)
        self.declare_parameter("max_covariance_m2", 0.25)
        self.declare_parameter("action_timeout_s", 10.0)
        self.declare_parameter("compute_path_action", "/compute_path_to_pose")
        self.declare_parameter("follow_path_action", "/follow_path")
        self.declare_parameter("planner_id", "")
        self.declare_parameter("controller_id", "FollowPath")
        self.declare_parameter("goal_checker_id", "")
        self.declare_parameter("require_slam_readiness", False)
        self.declare_parameter("require_navigation_readiness", False)
        self.declare_parameter("slam_readiness_service", "/slam/readiness")
        self.declare_parameter("navigation_readiness_service", "/navigation/readiness")
        self.declare_parameter("readiness_poll_interval_s", 0.5)

        self._enabled = bool(self.get_parameter("enabled").value)
        self._global_frame = str(self.get_parameter("global_frame").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._target_frame = str(self.get_parameter("target_frame").value)
        self._stand_off = float(self.get_parameter("stand_off_distance_m").value)
        self._replan_distance = float(self.get_parameter("replan_displacement_m").value)
        self._replan_heading = float(self.get_parameter("replan_heading_rad").value)
        self._replan_interval = float(self.get_parameter("minimum_replan_interval_s").value)
        self._target_timeout = float(self.get_parameter("target_timeout_s").value)
        self._max_covariance = float(self.get_parameter("max_covariance_m2").value)
        self._action_timeout = float(self.get_parameter("action_timeout_s").value)
        self._compute_path_action = str(self.get_parameter("compute_path_action").value)
        self._follow_path_action = str(self.get_parameter("follow_path_action").value)
        self._planner_id = str(self.get_parameter("planner_id").value)
        self._controller_id = str(self.get_parameter("controller_id").value)
        self._goal_checker_id = str(self.get_parameter("goal_checker_id").value)
        self._require_slam_readiness = bool(self.get_parameter("require_slam_readiness").value)
        self._require_navigation_readiness = bool(self.get_parameter("require_navigation_readiness").value)
        self._readiness_poll_interval = float(self.get_parameter("readiness_poll_interval_s").value)

        self._group = ReentrantCallbackGroup()
        self._tf = Buffer()
        self._tf_listener = TransformListener(self._tf, self)
        self._compute_client = ActionClient(
            self, ComputePathToPose, self._compute_path_action, callback_group=self._group
        )
        self._follow_client = ActionClient(self, FollowPath, self._follow_path_action, callback_group=self._group)
        self._track_sub = self.create_subscription(
            TrackState,
            str(self.get_parameter("track_state_topic").value),
            self._on_track_state,
            10,
            callback_group=self._group,
        )
        self._timer = self.create_timer(0.2, self._tick, callback_group=self._group)
        self._slam_readiness_client = self.create_client(
            Trigger, str(self.get_parameter("slam_readiness_service").value), callback_group=self._group
        )
        self._navigation_readiness_client = self.create_client(
            Trigger, str(self.get_parameter("navigation_readiness_service").value), callback_group=self._group
        )

        self._latest_state = None
        self._latest_goal: FollowGoal | None = None
        self._last_plan_time = 0.0
        self._planning = False
        self._following = False
        self._follow_goal_handle = None
        self._follow_goal_pending = False
        self._cancel_deadline = None
        self._planning_deadline = None
        self._faulted = False
        self._paths = PathReplacementGate()
        self._last_reason = "disabled"
        self._slam_ready = not self._require_slam_readiness
        self._navigation_ready = not self._require_navigation_readiness
        self._readiness_pending = False
        self._last_readiness_poll = 0.0

    def _on_track_state(self, message: TrackState) -> None:
        self._latest_state = message

    def _tick(self) -> None:
        if not self._enabled:
            return
        if self._cancel_deadline is not None and time.monotonic() > self._cancel_deadline:
            self._paths.fail_closed()
            self._faulted = True
            self._cancel_deadline = None
            self._last_reason = "FollowPath cancellation timed out"
        if self._planning_deadline is not None and time.monotonic() > self._planning_deadline:
            self._planning = False
            self._planning_deadline = None
            self._last_reason = "ComputePathToPose timed out"
        if self._faulted:
            return
        self._poll_readiness()
        if not self._slam_ready or not self._navigation_ready:
            self._stop_following("SLAM or navigation readiness is unavailable")
            return
        state = self._latest_state
        if state is None:
            return
        if state.lifecycle_state != TrackState.TRACKING or not state.actionable:
            self._stop_following("target is not actionable")
            return
        if not state.measured and state.prediction_only:
            return
        if state.header.frame_id != self._target_frame:
            self._last_reason = f"target frame must be {self._target_frame}, got {state.header.frame_id}"
            return
        age = (self.get_clock().now() - rclpy.time.Time.from_msg(state.header.stamp)).nanoseconds / 1e9
        if age > self._target_timeout:
            self._stop_following("target state is stale")
            return
        covariance = max(float(state.pose.covariance[0]), float(state.pose.covariance[7]))
        if covariance > self._max_covariance:
            self._stop_following("target covariance exceeds admission limit")
            return
        if self._planning or self._follow_goal_pending:
            return
        try:
            target = self._target_in_map(state)
            robot = self._robot_position()
            candidate = stand_off_goal(target, robot, self._stand_off)
        except (TransformException, ValueError) as error:
            self._last_reason = str(error)
            return
        if not should_replan(
            self._latest_goal,
            candidate,
            displacement_m=self._replan_distance,
            heading_delta_rad=self._replan_heading,
            elapsed_s=time.monotonic() - self._last_plan_time,
            minimum_interval_s=self._replan_interval,
        ):
            return
        self._plan(candidate)

    def _poll_readiness(self) -> None:
        if self._readiness_pending or time.monotonic() - self._last_readiness_poll < self._readiness_poll_interval:
            return
        clients = []
        if self._require_slam_readiness:
            clients.append((self._slam_readiness_client, "SLAM", "_on_slam_readiness"))
        if self._require_navigation_readiness:
            clients.append((self._navigation_readiness_client, "navigation", "_on_navigation_readiness"))
        if not clients:
            return
        self._readiness_pending = True
        self._last_readiness_poll = time.monotonic()
        pending = len(clients)

        def complete(_future) -> None:
            nonlocal pending
            pending -= 1
            if pending == 0:
                self._readiness_pending = False

        for client, label, callback_name in clients:
            if not client.service_is_ready():
                setattr(self, f"_{'slam' if label == 'SLAM' else 'navigation'}_ready", False)
                self._last_reason = f"{label} readiness service is unavailable"
                pending -= 1
                continue
            future = client.call_async(Trigger.Request())
            future.add_done_callback(getattr(self, callback_name))
            future.add_done_callback(complete)
        if pending == 0:
            self._readiness_pending = False

    def _on_slam_readiness(self, future) -> None:
        self._slam_ready, self._last_reason = self._readiness_result(future, "SLAM")

    def _on_navigation_readiness(self, future) -> None:
        self._navigation_ready, self._last_reason = self._readiness_result(future, "navigation")

    @staticmethod
    def _readiness_result(future, label: str) -> tuple[bool, str]:
        try:
            response = future.result()
            return bool(response.success), response.message or f"{label} readiness rejected"
        except Exception as error:
            return False, f"{label} readiness request failed: {error}"

    def _target_in_map(self, state: TrackState):
        pose = PoseStamped()
        pose.header = state.header
        pose.pose = state.pose.pose
        stamp = rclpy.time.Time.from_msg(state.header.stamp)
        transform = self._tf.lookup_transform(self._global_frame, state.header.frame_id, stamp)
        transformed = do_transform_pose(pose.pose, transform)
        return transformed.position.x, transformed.position.y

    def _robot_position(self):
        transform = self._tf.lookup_transform(self._global_frame, self._base_frame, rclpy.time.Time())
        return transform.transform.translation.x, transform.transform.translation.y

    def _plan(self, candidate: FollowGoal) -> None:
        if not self._compute_client.wait_for_server(timeout_sec=0.0):
            self._last_reason = "compute_path_to_pose is unavailable"
            return
        self._planning = True
        self._planning_deadline = time.monotonic() + self._action_timeout
        goal = ComputePathToPose.Goal()
        goal.goal = self._pose(candidate)
        goal.planner_id = self._planner_id
        future = self._compute_client.send_goal_async(goal)
        future.add_done_callback(lambda result: self._on_plan_goal(result, candidate))

    def _on_plan_goal(self, future, candidate: FollowGoal) -> None:
        try:
            handle = future.result()
            if handle is None or not handle.accepted:
                self._last_reason = "planner rejected target path"
                self._planning = False
                self._planning_deadline = None
                return
            result_future = handle.get_result_async()
            result_future.add_done_callback(lambda result: self._on_plan_result(result, candidate))
        except Exception as error:
            self._last_reason = f"planner request failed: {error}"
            self._planning = False
            self._planning_deadline = None

    def _on_plan_result(self, future, candidate: FollowGoal) -> None:
        self._planning = False
        self._planning_deadline = None
        try:
            wrapped = future.result()
            if wrapped.status != GoalStatus.STATUS_SUCCEEDED:
                self._last_reason = f"planner ended with status {wrapped.status}"
                return
            result = wrapped.result
            path = result.path
            if not path.poses:
                self._last_reason = "planner returned an empty path"
                return
            self._latest_goal = candidate
            self._last_plan_time = time.monotonic()
            if self._following:
                should_cancel = self._paths.request_replacement(path)
                if should_cancel and self._follow_goal_handle is not None:
                    self._follow_goal_handle.cancel_goal_async()
                    self._cancel_deadline = time.monotonic() + self._action_timeout
                return
            self._send_path(path)
        except Exception as error:
            self._last_reason = f"planner result failed: {error}"

    def _send_path(self, path) -> None:
        if not self._follow_client.wait_for_server(timeout_sec=0.0):
            self._last_reason = "follow_path is unavailable"
            return
        goal = FollowPath.Goal()
        goal.path = path
        goal.controller_id = self._controller_id
        goal.goal_checker_id = self._goal_checker_id
        future = self._follow_client.send_goal_async(goal)
        self._follow_goal_pending = True
        future.add_done_callback(self._on_follow_goal)

    def _on_follow_goal(self, future) -> None:
        self._follow_goal_pending = False
        try:
            handle = future.result()
            if handle is None or not handle.accepted:
                self._last_reason = "controller rejected follow path"
                self._following = False
                return
            self._follow_goal_handle = handle
            self._following = True
            self._paths.activate(handle)
            handle.get_result_async().add_done_callback(self._on_follow_result)
        except RuntimeError:
            self._paths.fail_closed()
            self._following = False
        except Exception as error:
            self._last_reason = f"follow path request failed: {error}"
            self._following = False

    def _on_follow_result(self, future) -> None:
        status = future.result().status
        self._following = False
        self._follow_goal_handle = None
        self._cancel_deadline = None
        if status == GoalStatus.STATUS_CANCELED:
            replacement = self._paths.on_active_terminal()
            if replacement is not None:
                self._send_path(replacement)
        elif status == GoalStatus.STATUS_SUCCEEDED:
            self._paths.on_active_terminal()
        elif status != GoalStatus.STATUS_SUCCEEDED:
            self._paths.fail_closed()

    def _stop_following(self, reason: str) -> None:
        self._last_reason = reason
        self._paths.pending_path = None
        if self._follow_goal_handle is not None and self._cancel_deadline is None:
            self._follow_goal_handle.cancel_goal_async()
            self._cancel_deadline = time.monotonic() + self._action_timeout
        self._latest_goal = None

    def _pose(self, candidate: FollowGoal) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self._global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(candidate.position[0])
        pose.pose.position.y = float(candidate.position[1])
        pose.pose.orientation.z = math.sin(candidate.yaw / 2.0)
        pose.pose.orientation.w = math.cos(candidate.yaw / 2.0)
        return pose


def main(args=None):
    rclpy.init(args=args)
    node = DynamicTargetFollowerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
