"""Publish and display a read-only live overlay for a candidate artifact."""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2
from sensor_msgs_py import point_cloud2


def _matrix(value):
    transform = value["transform"]
    x, y, z, w = transform["rotation_xyzw"]
    matrix = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    result = np.eye(4)
    result[:3, :3] = matrix
    result[:3, 3] = transform["translation"]
    return result


def _overlay(image, points, camera_from_body, camera_info):
    frame = np.asarray(image).copy()
    camera = points @ camera_from_body[:3, :3].T + camera_from_body[:3, 3]
    camera = camera[np.isfinite(camera).all(axis=1) & (camera[:, 2] > 0)]
    if not len(camera):
        return frame
    matrix = np.asarray(camera_info.k, dtype=float).reshape(3, 3)
    projected = camera @ matrix.T
    pixels = np.rint(projected[:, :2] / projected[:, 2:3]).astype(int)
    inside = (
        (pixels[:, 0] >= 0) & (pixels[:, 0] < frame.shape[1]) & (pixels[:, 1] >= 0) & (pixels[:, 1] < frame.shape[0])
    )
    pixels = pixels[inside]
    depths = camera[inside, 2]
    if not len(pixels):
        return frame
    near, far = np.percentile(depths, [5, 95])
    normalized = np.clip((depths - near) / max(far - near, 1e-9), 0.0, 1.0)
    colors = cv2.applyColorMap(np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
    stride = max(1, len(pixels) // 12000)
    for (x, y), color in zip(pixels[::stride], colors[::stride], strict=True):
        cv2.circle(frame, (int(x), int(y)), 2, tuple(int(value) for value in color), -1, cv2.LINE_AA)
    return frame


class LiveOverlay(Node):
    def __init__(
        self,
        artifact: Path,
        mount: Path,
        output_topic: str,
        max_fps: float,
        jpeg_quality: int,
        display: bool = True,
    ):
        super().__init__("robot_calibration_live_overlay")
        self._artifact = yaml.safe_load(artifact.read_bytes())
        self._mount = yaml.safe_load(mount.read_bytes())
        self._bridge = CvBridge()
        self._info = None
        self._cloud = None
        self._cloud_history = deque(maxlen=3)
        self._minimum_interval = 1.0 / max_fps
        self._last_publish_time = 0.0
        self._jpeg_quality = jpeg_quality
        self._publisher = self.create_publisher(CompressedImage, output_topic, qos_profile_sensor_data)
        self._display = display
        self.create_subscription(CameraInfo, "/camera/front/camera_info", self._info_callback, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/cloud_registered_body", self._cloud_callback, qos_profile_sensor_data)
        self.create_subscription(Image, "/camera/front/image_raw", self._image_callback, qos_profile_sensor_data)

    def _info_callback(self, message):
        self._info = message

    def _cloud_callback(self, message):
        cloud = point_cloud2.read_points_numpy(message, field_names=["x", "y", "z"])
        self._cloud_history.append(cloud)
        self._cloud = np.concatenate(self._cloud_history, axis=0)

    def _image_callback(self, message):
        if self._info is None or self._cloud is None:
            return
        now = time.monotonic()
        if now - self._last_publish_time < self._minimum_interval:
            return
        base_from_camera = _matrix(self._artifact)
        base_from_body = _matrix(self._mount)
        camera_from_body = np.linalg.inv(base_from_camera) @ base_from_body
        image = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        result = _overlay(image, self._cloud, camera_from_body, self._info)
        encoded, buffer = cv2.imencode(".jpg", result, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not encoded:
            self.get_logger().error("failed to encode calibration overlay as JPEG")
            return
        output = CompressedImage()
        output.header = message.header
        output.format = "jpeg"
        output.data = buffer.tobytes()
        self._publisher.publish(output)
        self._last_publish_time = now
        if self._display:
            cv2.imshow("calib overlay", result)
            cv2.waitKey(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--mount", type=Path, required=True)
    parser.add_argument("--output-topic", default="/calib/overlay/compressed")
    parser.add_argument("--max-fps", type=float, default=5.0)
    parser.add_argument("--jpeg-quality", type=int, default=75)
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args(argv)
    rclpy.init()
    if args.max_fps <= 0 or not 1 <= args.jpeg_quality <= 100:
        raise ValueError("max-fps must be positive; jpeg-quality must be within 1-100")
    node = LiveOverlay(
        args.artifact,
        args.mount,
        args.output_topic,
        args.max_fps,
        args.jpeg_quality,
        display=not args.no_display,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.no_display:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
    return 0
