"""Grounded-SAM-2 ROS 2 node for open-vocabulary detection and segmentation."""

import threading
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from ibrobot_msgs.msg import Detection2D, DetectionArray
from ibrobot_msgs.srv import DetectSegment

from .grounded_sam2_wrapper import GroundedSAM2Wrapper


def _depth_scale_for_msg(msg: Image) -> float:
    if msg.encoding in ("32FC1", "64FC1"):
        return 1.0
    return 1000.0


class GroundedSAM2Node(Node):
    def __init__(self):
        super().__init__("grounded_sam2")

        self._bridge = CvBridge()
        self._lock = threading.Lock()

        self._latest_rgb: np.ndarray | None = None
        self._latest_depth: np.ndarray | None = None
        self._latest_camera_info: CameraInfo | None = None
        self._latest_rgb_header: Header | None = None
        self._latest_depth_scale: float = 1000.0

        self.declare_parameter("device", "cuda")
        self.declare_parameter("sam_checkpoint", "sam2.1_hiera_tiny.pt")
        self.declare_parameter("sam_config", "configs/sam2.1/sam2.1_hiera_t.yaml")
        self.declare_parameter("gdino_config", "GroundingDINO_SwinT_OGC.py")
        self.declare_parameter("gdino_checkpoint", "groundingdino_swint_ogc.pth")
        self.declare_parameter("gdino_text_encoder", "bert-base-uncased")
        self.declare_parameter("model_dir", "")
        self.declare_parameter("rgb_topic", "/camera/wrist/image_raw")
        self.declare_parameter("depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/wrist/aligned_depth_to_color/camera_info")
        self.declare_parameter("box_threshold", 0.35)
        self.declare_parameter("text_threshold", 0.25)

        device = self.get_parameter("device").get_parameter_value().string_value
        sam_ckpt = self.get_parameter("sam_checkpoint").get_parameter_value().string_value
        sam_cfg = self.get_parameter("sam_config").get_parameter_value().string_value
        gdino_cfg = self.get_parameter("gdino_config").get_parameter_value().string_value
        gdino_ckpt = self.get_parameter("gdino_checkpoint").get_parameter_value().string_value
        gdino_text_encoder = self.get_parameter("gdino_text_encoder").get_parameter_value().string_value
        model_dir = self.get_parameter("model_dir").get_parameter_value().string_value or None

        self.get_logger().info("Loading Grounded-SAM-2 models...")
        self._wrapper = GroundedSAM2Wrapper(
            device=device,
            sam_checkpoint=sam_ckpt,
            sam_config=sam_cfg,
            gdino_config=gdino_cfg,
            gdino_checkpoint=gdino_ckpt,
            gdino_text_encoder=gdino_text_encoder,
            model_dir=model_dir,
        )
        self.get_logger().info("Models loaded.")

        cb_group = ReentrantCallbackGroup()
        srv_cb_group = MutuallyExclusiveCallbackGroup()

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        info_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value

        self._rgb_sub = self.create_subscription(
            Image, rgb_topic, self._rgb_cb, qos_profile_sensor_data, callback_group=cb_group
        )
        self._depth_sub = self.create_subscription(
            Image, depth_topic, self._depth_cb, qos_profile_sensor_data, callback_group=cb_group
        )
        self._info_sub = self.create_subscription(
            CameraInfo, info_topic, self._info_cb, qos_profile_sensor_data, callback_group=cb_group
        )

        self._srv = self.create_service(
            DetectSegment, "~/detect_and_segment", self._srv_cb, callback_group=srv_cb_group
        )

        self._det_pub = self.create_publisher(DetectionArray, "~/detections", 10)

        self.get_logger().info(f"GroundedSAM2Node ready — subscribing: {rgb_topic}, {depth_topic}, {info_topic}")

    def _rgb_cb(self, msg: Image):
        try:
            rgb = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            self.get_logger().warn("Failed to convert RGB image")
            return
        with self._lock:
            self._latest_rgb = rgb
            self._latest_rgb_header = msg.header

    def _depth_cb(self, msg: Image):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception:
            self.get_logger().warn("Failed to convert depth image")
            return
        with self._lock:
            self._latest_depth = depth
            self._latest_depth_scale = _depth_scale_for_msg(msg)

    def _info_cb(self, msg: CameraInfo):
        with self._lock:
            self._latest_camera_info = msg

    def _srv_cb(self, request: DetectSegment.Request, response: DetectSegment.Response):
        with self._lock:
            rgb = self._latest_rgb
            depth = self._latest_depth
            info = self._latest_camera_info
            rgb_header = self._latest_rgb_header
            depth_scale = self._latest_depth_scale

        if rgb is None:
            response.success = False
            response.message = "No RGB image received yet"
            response.inference_time_ms = 0.0
            return response

        box_thresh = self.get_parameter("box_threshold").get_parameter_value().double_value
        if request.confidence_threshold > 0.0:
            box_thresh = float(request.confidence_threshold)
        text_thresh = self.get_parameter("text_threshold").get_parameter_value().double_value

        t0 = time.time()
        detections = self._wrapper.detect_and_segment(
            image_bgr=rgb,
            text_prompt=request.text_prompt,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
        )

        has_depth = depth is not None and info is not None
        if has_depth:
            K = np.array(info.k).reshape(3, 3)
            detections = GroundedSAM2Wrapper.compute_3d_centroids(
                detections,
                depth,
                K,
                depth_scale=depth_scale,
            )

        elapsed_ms = (time.time() - t0) * 1000.0

        h, w = rgb.shape[:2]
        if rgb_header is not None:
            header = Header(stamp=rgb_header.stamp, frame_id=rgb_header.frame_id)
        else:
            header = Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id=info.header.frame_id if info else "camera_front_optical_frame",
            )

        det_msgs = []
        for det in detections:
            dm = Detection2D()
            dm.header = header
            dm.label = det.label
            dm.confidence = det.confidence
            dm.bbox = [
                float(det.bbox_xyxy[0]),
                float(det.bbox_xyxy[1]),
                float(det.bbox_xyxy[2]),
                float(det.bbox_xyxy[3]),
            ]

            mask_msg = self._bridge.cv2_to_imgmsg(det.mask, encoding="mono8")
            mask_msg.header = header
            dm.mask = mask_msg

            dm.centroid_xyz = Point(
                x=float(det.centroid_xyz[0]),
                y=float(det.centroid_xyz[1]),
                z=float(det.centroid_xyz[2]),
            )
            dm.volume_centroid_xyz = Point(
                x=float(det.volume_centroid_xyz[0]),
                y=float(det.volume_centroid_xyz[1]),
                z=float(det.volume_centroid_xyz[2]),
            )
            dm.volume_m3 = float(det.volume_m3)
            dm.point_count = det.point_count
            det_msgs.append(dm)

        det_array = DetectionArray(header=header, detections=det_msgs)
        self._det_pub.publish(det_array)

        response.detections = det_array
        response.inference_time_ms = elapsed_ms
        response.success = True
        response.message = f"Detected {len(det_msgs)} objects in {elapsed_ms:.1f}ms"

        self.get_logger().info(f"DetectSegment('{request.text_prompt}'): {len(det_msgs)} objects, {elapsed_ms:.1f}ms")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GroundedSAM2Node()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
