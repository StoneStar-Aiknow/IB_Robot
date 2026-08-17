"""Lifecycle node for the single-target RGB-D tracker.

Resolves targets from the semantic-map database, initializes a template tracker
by projecting the mapped 3D position into the camera, and publishes timestamped
``TrackState`` in the odom frame. Robot ego motion comes from the FAST-LIO
bridged odometry (``/odometry/filtered``) and transforms come from the live SLAM
TF tree; the ``mock_slam_nav_interfaces`` node is only for bench tests without a
localization stack.
"""

import json
import sqlite3

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleNode, TransitionCallbackReturn
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformException, TransformListener

from ibrobot_msgs.msg import TrackState
from ibrobot_msgs.srv import GetTrackState, StartTracking, StopTracking

from .motion import EgoCompensatedMotionClassifier, MotionEstimate, MotionState
from .session import SessionState, SingleTargetSession
from .template_tracker import TemplateTracker
from .tracker_pipeline import DepthParams, Intrinsics, TrackerPipeline

_STATE_VALUES = {
    SessionState.ACQUIRING: TrackState.ACQUIRING,
    SessionState.TRACKING: TrackState.TRACKING,
    SessionState.SEARCHING: TrackState.SEARCHING,
    SessionState.LOST: TrackState.LOST,
    SessionState.STOPPED: TrackState.STOPPED,
}


def _rotation_matrix(quaternion) -> np.ndarray:
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    norm = float(np.sqrt(x * x + y * y + z * z + w * w))
    if norm == 0.0:
        return np.eye(3)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _transform_to_matrix(transform) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation_matrix(transform.transform.rotation)
    matrix[:3, 3] = [
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z,
    ]
    return matrix


class TargetTrackerNode(LifecycleNode):
    """Resolve, track, and publish the state of one semantic target."""

    def __init__(self):
        super().__init__("target_tracker")
        self._declare_parameters()
        self._sessions = SingleTargetSession()
        self._pipeline: TrackerPipeline | None = None
        self._bridge = CvBridge()
        self._camera_info: CameraInfo | None = None
        self._robot_linear_speed = 0.0
        self._robot_angular_speed = 0.0
        self._pending_target: dict | None = None
        self._acquisition_attempts = 0
        self._last_bbox: tuple[float, float, float, float] | None = None
        self._last_depth_m: float | None = None
        self._service_group = ReentrantCallbackGroup()
        self._sensor_group = MutuallyExclusiveCallbackGroup()
        self._track_state_pub = self.create_publisher(
            TrackState, str(self.get_parameter("track_state_topic").value), 10
        )
        self._create_services()
        self._tf_buffer = Buffer(cache_time=Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._subs: list = []
        self._synchronizer = None

    def _declare_parameters(self) -> None:
        self.declare_parameter("queue_size", 10)
        self.declare_parameter("motion_window_samples", 3)
        self.declare_parameter("min_object_displacement_m", 0.08)
        self.declare_parameter("min_object_speed_mps", 0.05)
        self.declare_parameter("motion_sigma_multiplier", 3.0)
        self.declare_parameter("moving_confirmation_windows", 2)
        self.declare_parameter("stationary_confirmation_windows", 2)
        self.declare_parameter("max_motion_position_variance_m2", 0.25)
        self.declare_parameter("max_motion_sample_gap_s", 3.0)
        self.declare_parameter("robot_linear_motion_threshold_mps", 0.02)
        self.declare_parameter("robot_angular_motion_threshold_rps", 0.05)
        self.declare_parameter("track_state_topic", "/object_tracker/track_state")
        self.declare_parameter("rgb_topic", "/camera/realsense/color/image_raw")
        self.declare_parameter("aligned_depth_topic", "/camera/realsense/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/realsense/color/camera_info")
        self.declare_parameter("odometry_topic", "/odometry/filtered")
        self.declare_parameter("camera_optical_frame", "camera_color_optical_frame")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("sync_slop_sec", 0.033)
        self.declare_parameter("depth_scale", 0.001)
        self.declare_parameter("depth_min_m", 0.15)
        self.declare_parameter("depth_max_m", 8.0)
        self.declare_parameter("depth_min_valid_ratio", 0.2)
        self.declare_parameter("depth_central_fraction", 0.6)
        self.declare_parameter("innovation_gate", 9.21)
        self.declare_parameter("max_prediction_s", 0.5)
        self.declare_parameter("max_visual_failures", 5)
        self.declare_parameter("max_reacquisition_attempts", 3)
        self.declare_parameter("match_threshold", 0.35)
        self.declare_parameter("search_radius_px", 60.0)
        self.declare_parameter("search_radius_reacquire_px", 120.0)
        self.declare_parameter("semantic_database_path", "")
        self.declare_parameter("acquisition_frame_timeout", 60)
        self.declare_parameter("init_box_padding_px", 12)

    def _create_services(self) -> None:
        self.create_service(StartTracking, "~/start_tracking", self._start_tracking, callback_group=self._service_group)
        self.create_service(StopTracking, "~/stop_tracking", self._stop_tracking, callback_group=self._service_group)
        self.create_service(
            GetTrackState, "~/get_track_state", self._get_track_state, callback_group=self._service_group
        )

    def on_configure(self, state):
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state):
        rgb_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter("rgb_topic").value), qos_profile=qos_profile_sensor_data
        )
        depth_sub = message_filters.Subscriber(
            self, Image, str(self.get_parameter("aligned_depth_topic").value), qos_profile=qos_profile_sensor_data
        )
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub],
            queue_size=int(self.get_parameter("queue_size").value),
            slop=float(self.get_parameter("sync_slop_sec").value),
        )
        self._synchronizer.registerCallback(self._synced_frame)
        info_sub = self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._camera_info_callback,
            10,
            qos_profile=qos_profile_sensor_data,
            callback_group=self._sensor_group,
        )
        odom_sub = self.create_subscription(
            Odometry,
            str(self.get_parameter("odometry_topic").value),
            self._odometry_callback,
            10,
            callback_group=self._sensor_group,
        )
        self._subs = [rgb_sub, depth_sub, info_sub, odom_sub]
        return super().on_activate(state)

    def on_deactivate(self, state):
        self._subs = []
        self._synchronizer = None
        return super().on_deactivate(state)

    def on_cleanup(self, state):
        return TransitionCallbackReturn.SUCCESS

    def _camera_info_callback(self, message: CameraInfo) -> None:
        self._camera_info = message

    def _odometry_callback(self, message: Odometry) -> None:
        twist = message.twist.twist
        self._robot_linear_speed = float(np.hypot(twist.linear.x, twist.linear.y))
        self._robot_angular_speed = float(abs(twist.angular.z))

    def update_target_motion(
        self,
        *,
        stamp_s: float,
        position_odom: tuple[float, float],
        position_covariance,
        robot_linear_speed_mps: float = 0.0,
        robot_angular_speed_rps: float = 0.0,
    ) -> MotionEstimate:
        """Classify target motion after timestamped TF conversion into odom."""
        classifier = self._pipeline.motion if self._pipeline is not None else self._make_motion_classifier()
        return classifier.update(
            stamp_s=stamp_s,
            position_odom=position_odom,
            position_covariance=position_covariance,
            robot_linear_speed_mps=robot_linear_speed_mps,
            robot_angular_speed_rps=robot_angular_speed_rps,
        )

    @staticmethod
    def apply_motion_estimate(state: TrackState, estimate: MotionEstimate) -> TrackState:
        """Attach motion evidence to a state before publishing it."""
        state.motion_state = int(estimate.state)
        state.ego_compensated_speed = float(estimate.speed_mps)
        if estimate.state is MotionState.UNKNOWN:
            state.actionable = False
            state.state_reason = estimate.reason
        return state

    def _make_motion_classifier(self) -> EgoCompensatedMotionClassifier:
        return EgoCompensatedMotionClassifier(
            window_samples=int(self.get_parameter("motion_window_samples").value),
            min_displacement_m=float(self.get_parameter("min_object_displacement_m").value),
            min_speed_mps=float(self.get_parameter("min_object_speed_mps").value),
            sigma_multiplier=float(self.get_parameter("motion_sigma_multiplier").value),
            moving_confirmation_windows=int(self.get_parameter("moving_confirmation_windows").value),
            stationary_confirmation_windows=int(self.get_parameter("stationary_confirmation_windows").value),
            max_position_variance_m2=float(self.get_parameter("max_motion_position_variance_m2").value),
            max_sample_gap_s=float(self.get_parameter("max_motion_sample_gap_s").value),
            robot_linear_motion_threshold_mps=float(self.get_parameter("robot_linear_motion_threshold_mps").value),
            robot_angular_motion_threshold_rps=float(self.get_parameter("robot_angular_motion_threshold_rps").value),
        )

    def _start_tracking(self, request, response):
        response.success = False
        response.initial_state = TrackState.STOPPED
        existing = self._sessions.session
        if existing is not None and existing.state not in {SessionState.LOST, SessionState.STOPPED}:
            response.message = "an active tracking session already exists"
            return response
        if not request.object_id and not request.query_text:
            response.message = "object_id or query_text is required"
            return response

        target = self._resolve_semantic_target(request.object_id, request.query_text)
        if target is None:
            response.message = "semantic map does not contain the requested target"
            return response

        try:
            session = self._sessions.start(
                target["object_id"],
                navigation_ready=True,
                map_ready=True,
            )
        except (RuntimeError, ValueError) as error:
            response.message = str(error)
            return response

        self._pipeline = TrackerPipeline(
            session=self._sessions,
            template=TemplateTracker(match_threshold=float(self.get_parameter("match_threshold").value)),
            motion_classifier=self._make_motion_classifier(),
            innovation_gate=float(self.get_parameter("innovation_gate").value),
            max_prediction_s=float(self.get_parameter("max_prediction_s").value),
            max_visual_failures=int(self.get_parameter("max_visual_failures").value),
            search_radius_px=float(self.get_parameter("search_radius_px").value),
            search_radius_reacquire_px=float(self.get_parameter("search_radius_reacquire_px").value),
        )
        self._pending_target = target
        self._acquisition_attempts = 0
        self._last_bbox = None
        self._last_depth_m = None

        response.success = True
        response.session_id = session.session_id
        response.resolved_object_id = target["object_id"]
        response.initial_state = TrackState.ACQUIRING
        response.message = (
            f"tracking {target['label']} from semantic map; awaiting local confirmation at {target['position_map']}"
        )
        return response

    def _stop_tracking(self, request, response):
        try:
            session = self._sessions.stop(request.session_id, request.reason or "caller requested stop")
        except KeyError as error:
            response.success = False
            response.final_state = TrackState.STOPPED
            response.message = str(error)
            return response
        response.success = True
        response.final_state = _STATE_VALUES[session.state]
        response.message = session.reason
        return response

    def _get_track_state(self, request, response):
        session = self._sessions.session
        if session is None or request.session_id != session.session_id:
            response.found = False
            response.message = "unknown tracking session identifier"
            return response
        response.found = True
        response.state.session_id = session.session_id
        response.state.object_id = session.object_id
        response.state.lifecycle_state = _STATE_VALUES[session.state]
        response.state.actionable = session.state == SessionState.TRACKING
        response.state.state_reason = session.reason
        response.message = "tracking state available"
        return response

    def _resolve_semantic_target(self, object_id: str, query_text: str) -> dict | None:
        database_path = str(self.get_parameter("semantic_database_path").value)
        if not database_path:
            self.get_logger().error("semantic_database_path is not configured; cannot resolve targets")
            return None
        try:
            connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
        except sqlite3.Error as error:
            self.get_logger().error(f"cannot open semantic map database: {error}")
            return None
        try:
            if object_id:
                row = connection.execute(
                    "SELECT object_id, label, position_json, size_json, state FROM semantic_objects "
                    "WHERE object_id = ?",
                    (object_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT object_id, label, position_json, size_json, state FROM semantic_objects "
                    "WHERE label LIKE ? AND state != 'removed' ORDER BY observation_count DESC LIMIT 1",
                    (f"%{query_text}%",),
                ).fetchone()
        except sqlite3.Error as error:
            self.get_logger().error(f"semantic target lookup failed: {error}")
            return None
        finally:
            connection.close()
        if row is None:
            return None
        position = np.asarray(json.loads(row[2]), dtype=np.float64)
        size = np.asarray(json.loads(row[3]), dtype=np.float64)
        if position.size != 3 or not np.all(np.isfinite(position)):
            return None
        return {
            "object_id": row[0],
            "label": row[1],
            "position_map": position,
            "size": size,
        }

    def _synced_frame(self, rgb_msg: Image, depth_msg: Image) -> None:
        pipeline = self._pipeline
        if pipeline is None:
            return
        session = self._sessions.session
        if session is None or session.state in {SessionState.LOST, SessionState.STOPPED}:
            return

        stamp = Time.from_msg(rgb_msg.header.stamp)
        if self._camera_info is None:
            return
        intrinsics = Intrinsics(
            fx=float(self._camera_info.k[0]),
            fy=float(self._camera_info.k[4]),
            cx=float(self._camera_info.k[2]),
            cy=float(self._camera_info.k[5]),
        )
        optical_frame = str(self.get_parameter("camera_optical_frame").value) or rgb_msg.header.frame_id
        odom_frame = str(self.get_parameter("odom_frame").value)
        try:
            camera_to_odom = _transform_to_matrix(
                self._tf_buffer.lookup_transform(odom_frame, optical_frame, stamp, timeout=Duration(seconds=0.05))
            )
        except TransformException as error:
            self.get_logger().warn(f"camera-to-odom transform unavailable: {error}", throttle_duration_sec=5.0)
            snapshot = pipeline.predict_only(stamp.nanoseconds / 1e9)
            if snapshot is not None:
                self._publish_snapshot(snapshot, session)
            return

        try:
            gray = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="mono8")
            depth = self._bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        except Exception as error:  # noqa: BLE001 - cv_bridge raises assorted types
            self.get_logger().warn(f"image conversion failed: {error}", throttle_duration_sec=5.0)
            return

        if self._pending_target is not None:
            if not self._try_initialize(pipeline, gray, intrinsics, stamp, optical_frame, camera_to_odom):
                return
            self._pending_target = None

        snapshot = pipeline.process_observation(
            stamp_s=stamp.nanoseconds / 1e9,
            gray=gray,
            depth_image=depth,
            intrinsics=intrinsics,
            camera_to_odom=camera_to_odom,
            depth_params=DepthParams(
                scale=float(self.get_parameter("depth_scale").value),
                min_m=float(self.get_parameter("depth_min_m").value),
                max_m=float(self.get_parameter("depth_max_m").value),
                min_valid_ratio=float(self.get_parameter("depth_min_valid_ratio").value),
                central_fraction=float(self.get_parameter("depth_central_fraction").value),
            ),
            robot_linear_speed_mps=self._robot_linear_speed,
            robot_angular_speed_rps=self._robot_angular_speed,
        )
        if snapshot is None:
            return
        if snapshot.bbox is not None:
            self._last_bbox = snapshot.bbox
        if snapshot.depth_m is not None:
            self._last_depth_m = snapshot.depth_m
        self._publish_snapshot(snapshot, self._sessions.session)

    def _try_initialize(
        self,
        pipeline: TrackerPipeline,
        gray: np.ndarray,
        intrinsics: Intrinsics,
        stamp,
        optical_frame: str,
        camera_to_odom: np.ndarray,
    ) -> bool:
        assert self._pending_target is not None
        self._acquisition_attempts += 1
        if self._acquisition_attempts > int(self.get_parameter("acquisition_frame_timeout").value):
            session = self._sessions.session
            if session is not None:
                self._sessions.stop(session.session_id, "local confirmation window expired")
            self._pending_target = None
            return False

        target = self._pending_target
        map_frame = str(self.get_parameter("map_frame").value)
        try:
            camera_to_map = self._tf_buffer.lookup_transform(
                optical_frame, map_frame, stamp, timeout=Duration(seconds=0.05)
            )
        except TransformException as error:
            self.get_logger().warn(
                f"map-to-camera transform unavailable during acquisition: {error}", throttle_duration_sec=5.0
            )
            return False
        matrix = _transform_to_matrix(camera_to_map)
        homogeneous = matrix @ np.asarray([*target["position_map"], 1.0])
        if homogeneous[3] == 0.0:
            return False
        point = homogeneous[:3] / homogeneous[3]
        if point[2] <= float(self.get_parameter("depth_min_m").value):
            return False
        u = intrinsics.fx * point[0] / point[2] + intrinsics.cx
        v = intrinsics.fy * point[1] / point[2] + intrinsics.cy
        height, width = gray.shape[:2]
        if not (0.0 <= u < width and 0.0 <= v < height):
            return False

        size = (
            target["size"]
            if np.all(np.isfinite(target["size"])) and np.all(target["size"] > 0)
            else np.array([0.15, 0.15, 0.15])
        )
        half_u = intrinsics.fx * float(size[0]) / (2.0 * point[2])
        half_v = intrinsics.fy * float(size[1]) / (2.0 * point[2])
        padding = float(self.get_parameter("init_box_padding_px").value)
        bbox = (
            u - half_u - padding,
            v - half_v - padding,
            u + half_u + padding,
            v + half_v + padding,
        )
        if not pipeline.template.initialize(gray, bbox):
            return False
        odom_homogeneous = camera_to_odom @ np.asarray([*point, 1.0])
        pipeline.initialize_filter((float(odom_homogeneous[0]), float(odom_homogeneous[1])))
        self.get_logger().info(
            f"local confirmation initialized for {target['label']} at pixel ({u:.0f},{v:.0f}) depth {point[2]:.2f} m"
        )
        return True

    def _publish_snapshot(self, snapshot, session) -> None:
        if session is None:
            return
        state = TrackState()
        state.header.frame_id = str(self.get_parameter("odom_frame").value)
        state.header.stamp.sec = int(snapshot.stamp_s)
        state.header.stamp.nanosec = int((snapshot.stamp_s - int(snapshot.stamp_s)) * 1e9)
        state.session_id = session.session_id
        state.object_id = session.object_id
        state.lifecycle_state = _STATE_VALUES[session.state]
        state.pose.pose.position.x = snapshot.position_odom[0]
        state.pose.pose.position.y = snapshot.position_odom[1]
        state.pose.pose.position.z = snapshot.depth_m if snapshot.depth_m is not None else 0.0
        state.pose.covariance[0] = snapshot.position_variance_xy[0]
        state.pose.covariance[7] = snapshot.position_variance_xy[1]
        state.twist.twist.linear.x = snapshot.velocity_odom[0]
        state.twist.twist.linear.y = snapshot.velocity_odom[1]
        state.twist.covariance[0] = snapshot.velocity_variance_xy[0]
        state.twist.covariance[7] = snapshot.velocity_variance_xy[1]
        if self._last_bbox is not None:
            state.bbox = [float(v) for v in self._last_bbox]
        state.measured = snapshot.measured
        state.prediction_only = not snapshot.measured
        state.actionable = snapshot.measured and session.state == SessionState.TRACKING
        state.confidence = snapshot.confidence
        if snapshot.motion is not None:
            state = self.apply_motion_estimate(state, snapshot.motion)
        else:
            state.motion_state = TrackState.MOTION_UNKNOWN
        if snapshot.measured:
            state.last_seen.sec = int(snapshot.stamp_s)
            state.last_seen.nanosec = int((snapshot.stamp_s - int(snapshot.stamp_s)) * 1e9)
        state.state_reason = snapshot.reason
        self._track_state_pub.publish(state)


def main(args=None):
    rclpy.init(args=args)
    node = TargetTrackerNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()
