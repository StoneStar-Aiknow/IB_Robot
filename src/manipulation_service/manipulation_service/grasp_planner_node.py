"""GraspPlanner ROS 2 node integrating GraspGen with the detection pipeline."""

import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time as TimeMsg
from cv_bridge import CvBridge
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from ibrobot_msgs.msg import DetectionArray, GraspCandidate, GraspCandidateArray
from ibrobot_msgs.srv import DetectSegment, GroundingDetect, PlanGrasp, SegmentDetections
from inference_service.runtime_composition import (
    build_model_service_runtime_dependencies,
    require_runtime_dependencies,
)
from inference_service.unified_runtime import RegistrySet, RuntimeProviders

from .debug_viz import (
    DEBUG_OUTPUT_DEFAULT,
    DEBUG_OUTPUT_DIAGNOSTIC,
    DEBUG_OUTPUT_FULL,
    DEBUG_OUTPUT_NONE,
    DebugPreviewCandidate,
    debug_output_enabled,
    debug_points_or_fallback,
    failure_result_payload,
    normalize_debug_output_mode,
    start_full_debug_artifacts_thread,
    write_planner_result_json,
    write_result_json,
)
from .detection_geometry import compute_detection_geometry
from .diagnostics import diagnostic_to_lines, execution_table_diagnostic_details
from .grasp_types import GraspDiagnostic
from .graspgen_wrapper import (
    DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP,
    GraspGenWrapper,
)
from .params import parse_float_csv, parse_vector3_parameter, safe_name
from .table_plane import join_execution_table_fit


@dataclass
class DepthFrame:
    stamp_ns: int
    frame_id: str
    data: np.ndarray
    depth_scale: float


@dataclass
class CameraInfoFrame:
    stamp_ns: int
    msg: CameraInfo


@dataclass
class RGBFrame:
    stamp_ns: int
    msg: Image


def _stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _stamp_from_ns(stamp_ns: int) -> TimeMsg:
    stamp_ns = max(0, int(stamp_ns))
    return TimeMsg(sec=stamp_ns // 1_000_000_000, nanosec=stamp_ns % 1_000_000_000)


def _depth_scale_for_msg(msg: Image, fallback_scale: float) -> float:
    if msg.encoding in ("32FC1", "64FC1"):
        return 1.0
    return fallback_scale


_DEPTH_ENCODING_DTYPES = {
    "16UC1": np.uint16,
    "mono16": np.uint16,
    "32FC1": np.float32,
    "64FC1": np.float64,
}


def _decode_depth_image(msg: Image) -> np.ndarray:
    # cv_bridge's imgmsg_to_cv2 has been observed to silently mis-decode
    # 16UC1 depth messages as uint8 in this node's process (encoding-table
    # mismatch), corrupting all downstream depth math. Decode manually from
    # the message's declared encoding/step instead of relying on cv_bridge.
    base_dtype = _DEPTH_ENCODING_DTYPES.get(msg.encoding)
    if base_dtype is None:
        raise ValueError(f"unsupported depth encoding: {msg.encoding!r}")
    dtype = np.dtype(base_dtype).newbyteorder(">" if msg.is_bigendian else "<")
    itemsize = np.dtype(dtype).itemsize
    rows = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.step // itemsize)
    return rows[:, : msg.width].astype(base_dtype, copy=True)


class GraspPlannerNode(Node):
    def __init__(
        self,
        *,
        registry_set: RegistrySet | None = None,
        providers: RuntimeProviders | None = None,
    ):
        super().__init__("grasp_planner")
        registry_set, providers = require_runtime_dependencies(
            registry_set,
            providers,
            owner=type(self).__name__,
        )
        self._registry_set = registry_set
        self._providers = providers

        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self.declare_parameter("device", "cuda")
        self.declare_parameter("inference_backend", "local_cuda")
        self.declare_parameter("gripper_config", "graspgen_robotiq_2f_140.yml")
        self.declare_parameter("model_dir", "")
        self.declare_parameter("local_manifest_path", "")
        self.declare_parameter("local_deployment_name", "")
        self.declare_parameter("ascend_local_manifest_path", "")
        self.declare_parameter("ascend_local_deployment_name", "ascend_310p")
        self.declare_parameter("ascend_local_device_id", 0)
        self.declare_parameter("ascend_local_random_seed", -1)
        self.declare_parameter("startup_warmup", False)
        self.declare_parameter("rgb_topic", "/camera/wrist/image_raw")
        self.declare_parameter("depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/wrist/aligned_depth_to_color/camera_info")
        self.declare_parameter("detect_service", "/perception/grounding_detect")
        self.declare_parameter("segment_service", "")
        self.declare_parameter("legacy_detect_service", "~/detect_and_segment")
        self.declare_parameter("grasp_threshold", 0.5)
        self.declare_parameter("num_grasps", 2000)
        self.declare_parameter("topk_num_grasps", 300)
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("enable_collision_filter", True)
        self.declare_parameter("collision_threshold", 0.005)
        self.declare_parameter("collision_gripper", "")
        self.declare_parameter("enable_tabletop_filter", True)
        self.declare_parameter(
            "enable_source_gripper_tabletop_sweep",
            DEFAULT_ENABLE_SOURCE_GRIPPER_TABLETOP_SWEEP,
        )
        self.declare_parameter("require_tabletop_filter", True)
        self.declare_parameter("tabletop_clearance", 0.003)
        self.declare_parameter("tabletop_pregrasp_distance", 0.08)
        self.declare_parameter("tabletop_pregrasp_steps", 5)
        self.declare_parameter("tabletop_ransac_threshold", 0.006)
        self.declare_parameter("tabletop_min_inlier_ratio", 0.15)
        self.declare_parameter("tabletop_filter_mode", "strict")
        self.declare_parameter("adaptive_tabletop_low_profile_height", 0.035)
        self.declare_parameter("adaptive_tabletop_clearance_min", 0.001)
        self.declare_parameter("adaptive_tabletop_clearance_max", 0.002)
        self.declare_parameter("adaptive_tabletop_clearance_height_ratio", 0.12)
        self.declare_parameter("adaptive_tabletop_pregrasp_min", 0.02)
        self.declare_parameter("adaptive_tabletop_pregrasp_height_ratio", 2.0)
        self.declare_parameter("adaptive_tabletop_hard_floor", -0.002)
        self.declare_parameter("adaptive_tabletop_auto_tune", True)
        self.declare_parameter("adaptive_tabletop_retry_clearances", "0.002,0.001")
        self.declare_parameter("adaptive_tabletop_retry_pregrasp_height_ratio", 1.5)
        self.declare_parameter("adaptive_tabletop_retry_hard_floor", -0.003)
        self.declare_parameter("target_width_axis_local", "1.0,0.0,0.0")
        self.declare_parameter("target_width_percentile_low", 5.0)
        self.declare_parameter("target_width_percentile_high", 95.0)
        self.declare_parameter("target_width_min_m", 0.005)
        self.declare_parameter("target_width_max_m", 0.14)
        self.declare_parameter("enable_object_cloud_completion", True)
        self.declare_parameter("object_cloud_completion_mode", "mask_depth_inpaint")
        self.declare_parameter("object_cloud_completion_max_points", 5000)
        self.declare_parameter("object_cloud_completion_kernel_size", 5)
        self.declare_parameter("object_cloud_completion_min_neighbors", 6)
        self.declare_parameter("enable_object_cloud_prismatic_extrude", True)
        self.declare_parameter("object_cloud_prismatic_extrude_max_points", 8000)
        self.declare_parameter("object_cloud_prismatic_extrude_layers", 8)
        self.declare_parameter("enable_scene_cloud_table_holes", True)
        self.declare_parameter("scene_cloud_table_holes_max_points", 8000)
        self.declare_parameter("input_buffer_size", 30)
        self.declare_parameter("sync_max_age_sec", 0.20)
        self.declare_parameter("save_debug_outputs", False)
        self.declare_parameter("debug_output_dir", "outputs/grasp_pipeline")
        self.declare_parameter("debug_max_save_grasps", 10)
        self.declare_parameter("debug_render_preview", True)
        self.declare_parameter("debug_render_show_scene", False)
        self.declare_parameter("debug_render_labels", True)
        self.declare_parameter("debug_render_max_points", 60000)
        self.declare_parameter("debug_render_width", 1280)
        self.declare_parameter("debug_render_height", 720)
        self.declare_parameter("debug_render_interactive", True)
        self.declare_parameter("debug_interactive_show_scene", True)
        self.declare_parameter("debug_interactive_max_points", 120000)

        buffer_size = int(self.get_parameter("input_buffer_size").get_parameter_value().integer_value)
        self._rgb_buffer: deque[RGBFrame] = deque(maxlen=max(1, buffer_size))
        self._depth_buffer: deque[DepthFrame] = deque(maxlen=max(1, buffer_size))
        self._info_buffer: deque[CameraInfoFrame] = deque(maxlen=max(1, buffer_size))

        device = self.get_parameter("device").get_parameter_value().string_value
        inference_backend = self.get_parameter("inference_backend").get_parameter_value().string_value
        gripper_config = self.get_parameter("gripper_config").get_parameter_value().string_value
        model_dir = self.get_parameter("model_dir").get_parameter_value().string_value or None
        collision_gripper = self.get_parameter("collision_gripper").get_parameter_value().string_value or None

        self.get_logger().info(f"Loading GraspGen backend (backend={inference_backend}, gripper={gripper_config}) ...")
        self._wrapper = GraspGenWrapper(
            gripper_config=gripper_config,
            model_dir=model_dir,
            device=device,
            collision_gripper=collision_gripper,
            inference_backend=inference_backend,
            local_manifest_path=self.get_parameter("local_manifest_path").get_parameter_value().string_value,
            local_deployment_name=self.get_parameter("local_deployment_name").get_parameter_value().string_value,
            ascend_local_manifest_path=self.get_parameter("ascend_local_manifest_path")
            .get_parameter_value()
            .string_value,
            ascend_local_deployment_name=self.get_parameter("ascend_local_deployment_name")
            .get_parameter_value()
            .string_value,
            ascend_local_device_id=int(
                self.get_parameter("ascend_local_device_id").get_parameter_value().integer_value
            ),
            ascend_local_random_seed=(
                None
                if self.get_parameter("ascend_local_random_seed").get_parameter_value().integer_value < 0
                else self.get_parameter("ascend_local_random_seed").get_parameter_value().integer_value
            ),
            registry_set=self._registry_set,
            providers=self._providers,
        )
        self.get_logger().info(
            f"GraspGen backend loaded (backend={inference_backend}, "
            f"collision_gripper={self._wrapper.collision_gripper_name})."
        )
        if self.get_parameter("startup_warmup").get_parameter_value().bool_value:
            started = time.perf_counter()
            try:
                warmed = self._wrapper.warmup()
            except Exception as exc:
                self.get_logger().warn(f"GraspGen startup warmup failed; continuing: {exc}")
            else:
                self.get_logger().info(
                    f"GraspGen startup warmup completed enabled={warmed} duration_s={time.perf_counter() - started:.3f}"
                )

        cb_group = ReentrantCallbackGroup()
        detect_cb_group = ReentrantCallbackGroup()
        srv_cb_group = MutuallyExclusiveCallbackGroup()

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        detect_service = self.get_parameter("detect_service").get_parameter_value().string_value
        segment_service = self.get_parameter("segment_service").get_parameter_value().string_value

        self._rgb_sub = self.create_subscription(
            Image, rgb_topic, self._rgb_cb, qos_profile_sensor_data, callback_group=cb_group
        )
        self._depth_sub = self.create_subscription(
            Image, depth_topic, self._depth_cb, qos_profile_sensor_data, callback_group=cb_group
        )
        self._info_sub = self.create_subscription(
            CameraInfo, info_topic, self._info_cb, qos_profile_sensor_data, callback_group=cb_group
        )

        self._detect_client = self.create_client(GroundingDetect, detect_service, callback_group=detect_cb_group)
        self._segment_client = (
            self.create_client(SegmentDetections, segment_service, callback_group=detect_cb_group)
            if segment_service
            else None
        )

        self._srv = self.create_service(PlanGrasp, "~/plan_grasp", self._srv_cb, callback_group=srv_cb_group)
        self._legacy_detect_srv = self.create_service(
            DetectSegment,
            self.get_parameter("legacy_detect_service").get_parameter_value().string_value,
            self._legacy_detect_cb,
            callback_group=detect_cb_group,
        )

        self._grasp_pub = self.create_publisher(GraspCandidateArray, "~/grasps", 10)

        self.get_logger().info(
            f"GraspPlannerNode ready — rgb: {rgb_topic}, depth: {depth_topic}, info: {info_topic}, "
            f"detect: {detect_service}, segment: {segment_service or 'inline'}"
        )

    def _legacy_detect_cb(self, request: DetectSegment.Request, response: DetectSegment.Response):
        """Expose the PR259 DetectSegment contract over the typed perception pipeline."""

        started = time.perf_counter()
        segmentation_result, failure = self._get_segmentation_mask(
            request.text_prompt,
            float(request.confidence_threshold),
        )
        if failure is not None or segmentation_result is None:
            response.success = False
            response.message = f"Detection failed: {failure or 'detect_service_no_result'}"
            response.inference_time_ms = (time.perf_counter() - started) * 1000.0
            return response

        mask, mask_header, _confidence, detection = segmentation_result
        synchronized = self._get_synchronized_inputs(_stamp_to_ns(mask_header.stamp))
        if synchronized is None:
            response.success = False
            response.message = "No synchronized depth/CameraInfo for detection frame"
            response.inference_time_ms = (time.perf_counter() - started) * 1000.0
            return response

        depth_frame, camera_info = synchronized
        intrinsics = np.asarray(camera_info.k, dtype=np.float64).reshape(3, 3)
        surface_centroid, volume_centroid, volume_m3, point_count = compute_detection_geometry(
            mask,
            depth_frame.data,
            intrinsics,
            depth_scale=depth_frame.depth_scale,
        )
        detection.centroid_xyz.x = float(surface_centroid[0])
        detection.centroid_xyz.y = float(surface_centroid[1])
        detection.centroid_xyz.z = float(surface_centroid[2])
        detection.volume_centroid_xyz.x = float(volume_centroid[0])
        detection.volume_centroid_xyz.y = float(volume_centroid[1])
        detection.volume_centroid_xyz.z = float(volume_centroid[2])
        detection.volume_m3 = float(volume_m3)
        detection.point_count = int(point_count)
        detection.header = mask_header
        detection.mask.header = mask_header

        response.detections = DetectionArray(header=mask_header, detections=[detection])
        response.success = True
        response.inference_time_ms = (time.perf_counter() - started) * 1000.0
        response.message = f"Detected 1 object with {point_count} depth points"
        return response

    def _rgb_cb(self, msg: Image):
        with self._lock:
            self._rgb_buffer.append(RGBFrame(stamp_ns=_stamp_to_ns(msg.header.stamp), msg=msg))

    def _depth_cb(self, msg: Image):
        try:
            depth = _decode_depth_image(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to convert depth image: {exc}")
            return
        fallback_scale = self.get_parameter("depth_scale").get_parameter_value().double_value
        frame = DepthFrame(
            stamp_ns=_stamp_to_ns(msg.header.stamp),
            frame_id=msg.header.frame_id,
            data=depth,
            depth_scale=_depth_scale_for_msg(msg, fallback_scale),
        )
        with self._lock:
            self._depth_buffer.append(frame)

    def _info_cb(self, msg: CameraInfo):
        frame = CameraInfoFrame(stamp_ns=_stamp_to_ns(msg.header.stamp), msg=msg)
        with self._lock:
            self._info_buffer.append(frame)

    @staticmethod
    def _nearest_by_stamp(frames, target_stamp_ns: int):
        if not frames:
            return None, None
        if target_stamp_ns == 0 or all(frame.stamp_ns == 0 for frame in frames):
            return frames[-1], 0
        frame = min(frames, key=lambda f: abs(f.stamp_ns - target_stamp_ns))
        return frame, abs(frame.stamp_ns - target_stamp_ns)

    def _get_synchronized_inputs(self, target_stamp_ns: int):
        max_age_ns = int(self.get_parameter("sync_max_age_sec").get_parameter_value().double_value * 1_000_000_000)
        with self._lock:
            depth_frames = list(self._depth_buffer)
            info_frames = list(self._info_buffer)

        depth_frame, depth_age = self._nearest_by_stamp(depth_frames, target_stamp_ns)
        info_frame, info_age = self._nearest_by_stamp(info_frames, target_stamp_ns)
        if depth_frame is None or info_frame is None:
            return None
        if depth_age is not None and depth_age > max_age_ns:
            self.get_logger().warn(f"No synchronized depth frame: age {depth_age / 1e9:.3f}s > {max_age_ns / 1e9:.3f}s")
            return None
        if info_age is not None and info_age > max_age_ns:
            self.get_logger().warn(f"No synchronized CameraInfo: age {info_age / 1e9:.3f}s > {max_age_ns / 1e9:.3f}s")
            return None
        return depth_frame, info_frame.msg

    def _get_segmentation_mask(self, text_prompt: str, confidence_threshold: float):
        if not self._detect_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("GroundingDetect service not available")
            return None, "detect_service_unavailable"

        with self._lock:
            rgb_frame = self._rgb_buffer[-1] if self._rgb_buffer else None
        if rgb_frame is None:
            return None, "rgb_frame_not_ready"

        request = GroundingDetect.Request()
        request.image = rgb_frame.msg
        request.text_prompt = text_prompt
        request.box_threshold = confidence_threshold
        request.text_threshold = min(0.99, max(0.01, confidence_threshold))

        future = self._detect_client.call_async(request)
        done_event = threading.Event()
        future.add_done_callback(lambda _: done_event.set())
        if not done_event.wait(timeout=10.0):
            self.get_logger().error("Timed out waiting for GroundingDetect response")
            return None, "detect_service_timeout"

        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"GroundingDetect service call failed: {exc}")
            return None, f"detect_service_error: {exc}"
        if result is None:
            return None, "detect_service_no_result"
        if not result.success:
            return None, f"detect_service_failed: {result.message}"
        if len(result.detections.detections) == 0:
            return (
                None,
                f"no_detections: target '{text_prompt}' not found at confidence_threshold={confidence_threshold:.2f}",
            )
        best_det = max(result.detections.detections, key=lambda d: d.confidence)
        if best_det.confidence < confidence_threshold:
            return (
                None,
                f"low_confidence: best detection confidence={best_det.confidence:.4f} < threshold={confidence_threshold:.2f}",
            )
        if self._segment_client is not None:
            if not self._segment_client.wait_for_service(timeout_sec=5.0):
                return None, "segment_service_unavailable"
            segment_request = SegmentDetections.Request()
            segment_request.image = rgb_frame.msg
            segment_request.detections = DetectionArray(
                header=result.detections.header,
                detections=[best_det],
            )
            segment_future = self._segment_client.call_async(segment_request)
            segment_done = threading.Event()
            segment_future.add_done_callback(lambda _: segment_done.set())
            if not segment_done.wait(timeout=10.0):
                return None, "segment_service_timeout"
            try:
                segmented = segment_future.result()
            except Exception as exc:
                return None, f"segment_service_error: {exc}"
            if segmented is None or not segmented.success or not segmented.detections.detections:
                return None, "segment_service_no_result"
            candidates = sorted(segmented.detections.detections, key=lambda d: d.confidence, reverse=True)
            best_det = candidates[0]
        if not best_det.mask.data:
            return None, "detection_mask_missing"
        mask = self._bridge.imgmsg_to_cv2(best_det.mask, desired_encoding="mono8")
        return (mask, best_det.mask.header, best_det.confidence, best_det), None

    def _resolve_debug_output_mode(self, request_mode: str) -> str:
        mode = normalize_debug_output_mode(request_mode)
        if mode != DEBUG_OUTPUT_DEFAULT:
            return mode
        if self.get_parameter("save_debug_outputs").get_parameter_value().bool_value:
            return DEBUG_OUTPUT_FULL
        return DEBUG_OUTPUT_NONE

    def _new_debug_output_dir(self, text_prompt: str) -> Path:
        output_root = Path(self.get_parameter("debug_output_dir").get_parameter_value().string_value)
        if not output_root.is_absolute():
            output_root = Path.cwd() / output_root

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = output_root / f"{stamp}_{safe_name(text_prompt)}_{time.time_ns()}"
        out_dir.mkdir(parents=True, exist_ok=False)
        return out_dir

    def _save_failure_diagnostic_json(
        self,
        *,
        text_prompt: str,
        confidence_threshold: float,
        grasp_threshold: float,
        output_mode: str,
        message: str,
        diagnostic_details: list[str],
        elapsed_ms: float = 0.0,
    ) -> Path:
        out_dir = self._new_debug_output_dir(text_prompt)
        write_result_json(
            out_dir,
            failure_result_payload(
                text_prompt=text_prompt,
                confidence_threshold=confidence_threshold,
                grasp_threshold=grasp_threshold,
                output_mode=output_mode,
                message=message,
                diagnostic_details=diagnostic_details,
                elapsed_ms=elapsed_ms,
            ),
        )
        return out_dir

    def _populate_failure_response(
        self,
        response: PlanGrasp.Response,
        *,
        text_prompt: str,
        confidence_threshold: float,
        grasp_threshold: float,
        output_mode: str,
        message: str,
        diagnostic_details: list[str],
        elapsed_ms: float = 0.0,
    ):
        response.success = False
        response.message = message
        response.inference_time_ms = elapsed_ms
        response.diagnostic_details = [*diagnostic_details, f"debug_output_mode: {output_mode}"]
        if debug_output_enabled(output_mode):
            try:
                out_dir = self._save_failure_diagnostic_json(
                    text_prompt=text_prompt,
                    confidence_threshold=confidence_threshold,
                    grasp_threshold=grasp_threshold,
                    output_mode=output_mode,
                    message=message,
                    diagnostic_details=response.diagnostic_details,
                    elapsed_ms=elapsed_ms,
                )
                response.debug_output_dir = str(out_dir)
                response.diagnostic_details.append(f"debug_output_dir: {out_dir}")
            except Exception as exc:
                self.get_logger().warn(f"Failed to save failure diagnostic JSON: {exc}")
                response.diagnostic_details.append(f"debug_output_error: {exc}")
        return response

    def _save_debug_outputs(
        self,
        *,
        text_prompt: str,
        confidence_threshold: float,
        grasp_threshold: float,
        depth_frame: DepthFrame,
        camera_info: CameraInfo,
        binary_mask: np.ndarray,
        camera_intrinsics: np.ndarray,
        candidates,
        diagnostic: GraspDiagnostic,
        detection_confidence: float,
        detection_stamp_ns: int,
        elapsed_ms: float,
        output_mode: str,
    ):
        out_dir = self._new_debug_output_dir(text_prompt)

        max_grasps = int(self.get_parameter("debug_max_save_grasps").get_parameter_value().integer_value)
        vis_gripper = getattr(self._wrapper, "collision_gripper_name", self._wrapper.gripper_name)

        render_enabled = self.get_parameter("debug_render_preview").get_parameter_value().bool_value
        write_planner_result_json(
            out_dir=out_dir,
            text_prompt=text_prompt,
            confidence_threshold=confidence_threshold,
            grasp_threshold=grasp_threshold,
            elapsed_ms=elapsed_ms,
            output_mode=output_mode,
            diagnostic=diagnostic,
            detection_confidence=detection_confidence,
            detection_stamp_ns=detection_stamp_ns,
            depth_stamp_ns=depth_frame.stamp_ns,
            depth_frame_id=depth_frame.frame_id,
            camera_frame_id=camera_info.header.frame_id,
            camera_info_stamp_ns=_stamp_to_ns(camera_info.header.stamp),
            depth_scale=depth_frame.depth_scale,
            camera_info_k=list(camera_info.k),
            camera_info_width=camera_info.width,
            camera_info_height=camera_info.height,
            gripper_name=self._wrapper.gripper_name,
            visualization_gripper=vis_gripper,
            candidates=candidates,
            render_enabled=render_enabled,
        )
        if output_mode == DEBUG_OUTPUT_DIAGNOSTIC:
            return out_dir

        object_raw_pts = debug_points_or_fallback(diagnostic.object_pc_raw, np.zeros((0, 3), dtype=np.float32))
        object_pts = debug_points_or_fallback(diagnostic.object_pc_after_completion, object_raw_pts)
        object_graspgen_input_pts = debug_points_or_fallback(diagnostic.object_pc_inference_input, object_pts)
        scene_pts = debug_points_or_fallback(diagnostic.scene_pc_after_completion, np.zeros((0, 3), dtype=np.float32))

        show_scene = self.get_parameter("debug_render_show_scene").get_parameter_value().bool_value
        render_labels = self.get_parameter("debug_render_labels").get_parameter_value().bool_value
        render_interactive = self.get_parameter("debug_render_interactive").get_parameter_value().bool_value
        interactive_show_scene = self.get_parameter("debug_interactive_show_scene").get_parameter_value().bool_value
        width = int(self.get_parameter("debug_render_width").get_parameter_value().integer_value)
        height = int(self.get_parameter("debug_render_height").get_parameter_value().integer_value)
        interactive_max_points = int(
            self.get_parameter("debug_interactive_max_points").get_parameter_value().integer_value
        )
        render_max_points = int(self.get_parameter("debug_render_max_points").get_parameter_value().integer_value)
        debug_candidates = [
            DebugPreviewCandidate(
                pose_4x4=np.array(g.pose_4x4, dtype=np.float32, copy=True),
                confidence=float(g.confidence),
                collision_free=bool(g.collision_free),
            )
            for g in candidates[:max_grasps]
        ]
        start_full_debug_artifacts_thread(
            diagnostic=diagnostic,
            object_pts=object_pts,
            object_raw_pts=object_raw_pts,
            object_graspgen_input_pts=object_graspgen_input_pts,
            scene_pts=scene_pts,
            candidates=debug_candidates,
            vis_gripper=vis_gripper,
            out_dir=out_dir,
            max_grasps=max_grasps,
            render_enabled=render_enabled,
            render_max_points=render_max_points,
            width=width,
            height=height,
            show_scene=show_scene,
            render_labels=render_labels,
            render_interactive=render_interactive,
            interactive_max_points=interactive_max_points,
            interactive_show_scene=interactive_show_scene,
            logger=self.get_logger(),
        )

        return out_dir

    def _srv_cb(self, request: PlanGrasp.Request, response: PlanGrasp.Response):
        try:
            debug_output_mode = self._resolve_debug_output_mode(request.debug_output_mode)
        except ValueError as exc:
            response.success = False
            response.message = str(exc)
            response.diagnostic_details = ["failure_stage: input", f"failure_reason: {exc}"]
            response.inference_time_ms = 0.0
            return response

        confidence_threshold = request.confidence_threshold if request.confidence_threshold > 0 else 0.25
        grasp_threshold = (
            request.grasp_threshold
            if request.grasp_threshold > 0
            else self.get_parameter("grasp_threshold").get_parameter_value().double_value
        )

        if not request.text_prompt:
            return self._populate_failure_response(
                response,
                text_prompt=request.text_prompt,
                confidence_threshold=confidence_threshold,
                grasp_threshold=grasp_threshold,
                output_mode=debug_output_mode,
                message="Empty text_prompt",
                diagnostic_details=["failure_stage: input", "failure_reason: text_prompt is empty"],
            )

        self.get_logger().info(f"Getting segmentation for '{request.text_prompt}' ...")
        segmentation_result, detect_failure = self._get_segmentation_mask(request.text_prompt, confidence_threshold)
        if detect_failure is not None or segmentation_result is None:
            reason = detect_failure or "detect_service_no_result"
            return self._populate_failure_response(
                response,
                text_prompt=request.text_prompt,
                confidence_threshold=confidence_threshold,
                grasp_threshold=grasp_threshold,
                output_mode=debug_output_mode,
                message=f"Detection failed: {reason}",
                diagnostic_details=[
                    "failure_stage: detection",
                    f"failure_reason: {reason}",
                    f"text_prompt: {request.text_prompt}",
                    f"confidence_threshold: {confidence_threshold:.2f}",
                ],
            )
        mask, mask_header, best_confidence, _best_detection = segmentation_result

        detection_stamp_ns = _stamp_to_ns(mask_header.stamp)
        synced_inputs = self._get_synchronized_inputs(detection_stamp_ns)
        if synced_inputs is None:
            return self._populate_failure_response(
                response,
                text_prompt=request.text_prompt,
                confidence_threshold=confidence_threshold,
                grasp_threshold=grasp_threshold,
                output_mode=debug_output_mode,
                message="No synchronized depth/CameraInfo for detection frame",
                diagnostic_details=[
                    "failure_stage: sync",
                    "failure_reason: depth or CameraInfo not available within sync tolerance",
                ],
            )
        depth_frame, info = synced_inputs

        binary_mask = (mask > 0).astype(np.int32)

        K = np.array(info.k).reshape(3, 3)
        surface_centroid, volume_centroid, volume_m3, object_point_count = compute_detection_geometry(
            mask,
            depth_frame.data,
            K,
            depth_scale=depth_frame.depth_scale,
        )

        t0 = time.time()
        try:
            candidates, diag = self._wrapper.plan_grasps(
                depth_image=depth_frame.data,
                segmentation_mask=binary_mask,
                camera_intrinsics=K,
                depth_scale=depth_frame.depth_scale,
                grasp_threshold=grasp_threshold,
                num_grasps=int(self.get_parameter("num_grasps").get_parameter_value().integer_value),
                topk_num_grasps=int(self.get_parameter("topk_num_grasps").get_parameter_value().integer_value),
                enable_collision_filter=self.get_parameter("enable_collision_filter").get_parameter_value().bool_value,
                collision_threshold=self.get_parameter("collision_threshold").get_parameter_value().double_value,
                enable_tabletop_filter=self.get_parameter("enable_tabletop_filter").get_parameter_value().bool_value,
                enable_source_gripper_tabletop_sweep=self.get_parameter("enable_source_gripper_tabletop_sweep")
                .get_parameter_value()
                .bool_value,
                require_tabletop_filter=self.get_parameter("require_tabletop_filter").get_parameter_value().bool_value,
                tabletop_clearance=self.get_parameter("tabletop_clearance").get_parameter_value().double_value,
                tabletop_pregrasp_distance=self.get_parameter("tabletop_pregrasp_distance")
                .get_parameter_value()
                .double_value,
                tabletop_pregrasp_steps=int(
                    self.get_parameter("tabletop_pregrasp_steps").get_parameter_value().integer_value
                ),
                tabletop_ransac_threshold=self.get_parameter("tabletop_ransac_threshold")
                .get_parameter_value()
                .double_value,
                tabletop_min_inlier_ratio=self.get_parameter("tabletop_min_inlier_ratio")
                .get_parameter_value()
                .double_value,
                tabletop_filter_mode=self.get_parameter("tabletop_filter_mode").get_parameter_value().string_value,
                adaptive_tabletop_low_profile_height=self.get_parameter("adaptive_tabletop_low_profile_height")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_clearance_min=self.get_parameter("adaptive_tabletop_clearance_min")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_clearance_max=self.get_parameter("adaptive_tabletop_clearance_max")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_clearance_height_ratio=self.get_parameter("adaptive_tabletop_clearance_height_ratio")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_pregrasp_min=self.get_parameter("adaptive_tabletop_pregrasp_min")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_pregrasp_height_ratio=self.get_parameter("adaptive_tabletop_pregrasp_height_ratio")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_hard_floor=self.get_parameter("adaptive_tabletop_hard_floor")
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_auto_tune=self.get_parameter("adaptive_tabletop_auto_tune")
                .get_parameter_value()
                .bool_value,
                adaptive_tabletop_retry_clearances=parse_float_csv(
                    self.get_parameter("adaptive_tabletop_retry_clearances").get_parameter_value().string_value
                ),
                adaptive_tabletop_retry_pregrasp_height_ratio=self.get_parameter(
                    "adaptive_tabletop_retry_pregrasp_height_ratio"
                )
                .get_parameter_value()
                .double_value,
                adaptive_tabletop_retry_hard_floor=self.get_parameter("adaptive_tabletop_retry_hard_floor")
                .get_parameter_value()
                .double_value,
                target_width_axis_local=parse_vector3_parameter(
                    self.get_parameter("target_width_axis_local").get_parameter_value().string_value,
                    "target_width_axis_local",
                ),
                target_width_percentile_low=self.get_parameter("target_width_percentile_low")
                .get_parameter_value()
                .double_value,
                target_width_percentile_high=self.get_parameter("target_width_percentile_high")
                .get_parameter_value()
                .double_value,
                target_width_min_m=self.get_parameter("target_width_min_m").get_parameter_value().double_value,
                target_width_max_m=self.get_parameter("target_width_max_m").get_parameter_value().double_value,
                enable_object_cloud_completion=self.get_parameter("enable_object_cloud_completion")
                .get_parameter_value()
                .bool_value,
                object_cloud_completion_mode=self.get_parameter("object_cloud_completion_mode")
                .get_parameter_value()
                .string_value,
                object_cloud_completion_max_points=int(
                    self.get_parameter("object_cloud_completion_max_points").get_parameter_value().integer_value
                ),
                object_cloud_completion_kernel_size=int(
                    self.get_parameter("object_cloud_completion_kernel_size").get_parameter_value().integer_value
                ),
                object_cloud_completion_min_neighbors=int(
                    self.get_parameter("object_cloud_completion_min_neighbors").get_parameter_value().integer_value
                ),
                enable_object_cloud_prismatic_extrude=self.get_parameter("enable_object_cloud_prismatic_extrude")
                .get_parameter_value()
                .bool_value,
                object_cloud_prismatic_extrude_max_points=int(
                    self.get_parameter("object_cloud_prismatic_extrude_max_points").get_parameter_value().integer_value
                ),
                object_cloud_prismatic_extrude_layers=int(
                    self.get_parameter("object_cloud_prismatic_extrude_layers").get_parameter_value().integer_value
                ),
                enable_scene_cloud_table_holes=self.get_parameter("enable_scene_cloud_table_holes")
                .get_parameter_value()
                .bool_value,
                scene_cloud_table_holes_max_points=int(
                    self.get_parameter("scene_cloud_table_holes_max_points").get_parameter_value().integer_value
                ),
            )
        except Exception as exc:
            self.get_logger().error(f"GraspGen error: {exc}")
            return self._populate_failure_response(
                response,
                text_prompt=request.text_prompt,
                confidence_threshold=confidence_threshold,
                grasp_threshold=grasp_threshold,
                output_mode=debug_output_mode,
                message=f"GraspGen inference failed: {exc}",
                diagnostic_details=[
                    "failure_stage: graspgen_inference",
                    f"failure_reason: {exc}",
                ],
            )

        execution_table_parallel = getattr(diag, "_execution_table_fit_thread", None) is not None
        execution_table_started = time.perf_counter()
        execution_table_fit = join_execution_table_fit(diag)
        execution_table_wait_s = time.perf_counter() - execution_table_started
        execution_table_holder = getattr(diag, "_execution_table_fit_holder", None)
        execution_table_duration_s = execution_table_wait_s
        if isinstance(execution_table_holder, dict):
            execution_table_duration_s = float(execution_table_holder.get("duration_s", execution_table_duration_s))
        elapsed_ms = (time.time() - t0) * 1000.0

        diagnostic_lines = diagnostic_to_lines(diag, detection_confidence=best_confidence)
        diagnostic_lines.extend(
            execution_table_diagnostic_details(
                diag,
                execution_table_fit=execution_table_fit,
                execution_table_parallel=execution_table_parallel,
                execution_table_wait_s=execution_table_wait_s,
                execution_table_duration_s=execution_table_duration_s,
                final_grasp_count=len(candidates),
                debug_output_mode=debug_output_mode,
                detection_stamp_ns=detection_stamp_ns,
                depth_stamp_ns=depth_frame.stamp_ns,
                logger=self.get_logger(),
            )
        )

        if debug_output_enabled(debug_output_mode):
            try:
                out_dir = self._save_debug_outputs(
                    text_prompt=request.text_prompt,
                    confidence_threshold=confidence_threshold,
                    grasp_threshold=grasp_threshold,
                    depth_frame=depth_frame,
                    camera_info=info,
                    binary_mask=binary_mask,
                    camera_intrinsics=K,
                    candidates=candidates,
                    diagnostic=diag,
                    detection_confidence=best_confidence,
                    detection_stamp_ns=detection_stamp_ns,
                    elapsed_ms=elapsed_ms,
                    output_mode=debug_output_mode,
                )
                response.debug_output_dir = str(out_dir)
                diagnostic_lines.append(f"debug_output_dir: {out_dir}")
                self.get_logger().info(f"Saved grasp debug outputs to {out_dir}")
            except Exception as exc:
                self.get_logger().warn(f"Failed to save grasp debug outputs: {exc}")
                diagnostic_lines.append(f"debug_output_error: {exc}")

        geometry_stamp_ns = depth_frame.stamp_ns or detection_stamp_ns
        header = Header(stamp=_stamp_from_ns(geometry_stamp_ns), frame_id=info.header.frame_id)

        grasp_msgs = []
        for cand in candidates:
            gm = GraspCandidate()
            gm.header = header
            gm.pose_matrix = cand.pose_4x4.flatten().tolist()
            gm.confidence = cand.confidence
            gm.collision_free = cand.collision_free
            gm.target_width_m = float(cand.target_width_m)
            gm.target_width_quality = float(cand.target_width_quality)
            gm.width_axis_camera = [float(v) for v in cand.width_axis_camera]
            gm.target_width_min_offset_m = float(cand.target_width_min_offset_m)
            gm.target_width_max_offset_m = float(cand.target_width_max_offset_m)
            grasp_msgs.append(gm)

        grasp_array = GraspCandidateArray(header=header, grasps=grasp_msgs)
        self._grasp_pub.publish(grasp_array)

        response.grasps = grasp_array
        response.object_centroid_xyz.x = float(surface_centroid[0])
        response.object_centroid_xyz.y = float(surface_centroid[1])
        response.object_centroid_xyz.z = float(surface_centroid[2])
        response.object_volume_centroid_xyz.x = float(volume_centroid[0])
        response.object_volume_centroid_xyz.y = float(volume_centroid[1])
        response.object_volume_centroid_xyz.z = float(volume_centroid[2])
        response.object_volume_m3 = float(volume_m3)
        response.object_point_count = object_point_count
        response.execution_table_plane_found = execution_table_fit.plane is not None
        if execution_table_fit.plane is not None:
            execution_table = execution_table_fit.plane
            response.execution_table_plane_normal.x = float(execution_table.normal[0])
            response.execution_table_plane_normal.y = float(execution_table.normal[1])
            response.execution_table_plane_normal.z = float(execution_table.normal[2])
            response.execution_table_plane_offset = float(execution_table.d)
            response.execution_table_plane_inlier_ratio = float(execution_table.inlier_ratio)
        response.table_plane_found = bool(diag.tabletop_plane_found and diag.tabletop_plane_normal is not None)
        if response.table_plane_found:
            table_normal = diag.tabletop_plane_normal
            assert table_normal is not None
            response.table_plane_normal.x = float(table_normal[0])
            response.table_plane_normal.y = float(table_normal[1])
            response.table_plane_normal.z = float(table_normal[2])
            response.table_plane_offset = float(diag.tabletop_plane_d)
            response.table_plane_inlier_ratio = float(diag.tabletop_inlier_ratio)
        if diag.tabletop_object_top_xyz is not None:
            response.object_top_xyz.x = float(diag.tabletop_object_top_xyz[0])
            response.object_top_xyz.y = float(diag.tabletop_object_top_xyz[1])
            response.object_top_xyz.z = float(diag.tabletop_object_top_xyz[2])
        response.inference_time_ms = elapsed_ms
        response.diagnostic_details = diagnostic_lines

        if len(candidates) == 0:
            response.success = False
            response.message = f"No grasps generated: {diag.failure_reason}"
        else:
            response.success = True
            response.message = f"Generated {len(grasp_msgs)} grasps in {elapsed_ms:.1f}ms"

        self.get_logger().info(
            f"PlanGrasp('{request.text_prompt}'): {len(grasp_msgs)} grasps, {elapsed_ms:.1f}ms, diag={diag.failure_stage or 'ok'}"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    dependencies = build_model_service_runtime_dependencies()
    node = GraspPlannerNode(
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        dependencies.providers.close()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
