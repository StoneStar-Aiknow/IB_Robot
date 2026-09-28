"""Publish bounded-bandwidth image and cloud previews for remote capture viewing."""

import argparse
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from sensor_msgs_py import point_cloud2


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--image-topic", default="/camera/front/image_raw")
    result.add_argument("--cloud-topic", default="/cloud_registered_body")
    result.add_argument("--output-image-topic", default="/calib/preview/image/compressed")
    result.add_argument("--output-cloud-topic", default="/calib/preview/cloud")
    result.add_argument("--max-fps", type=float, default=8.0)
    result.add_argument("--jpeg-quality", type=int, default=70)
    result.add_argument("--max-points", type=int, default=6000)
    return result


def select_preview_points(points: np.ndarray, max_points: int) -> np.ndarray:
    """Keep an evenly spaced, bounded sample without changing XYZ coordinates."""
    points = np.asarray(points)
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
    return points[indices]


def create_preview_cloud(message: PointCloud2, max_points: int) -> PointCloud2:
    fields = {field.name: field for field in message.fields}
    field_names = ["x", "y", "z"]
    if "intensity" in fields:
        field_names.append("intensity")
    points = point_cloud2.read_points_numpy(message, field_names=field_names)
    finite = points[np.isfinite(points).all(axis=1)]
    preview = select_preview_points(finite, max_points)
    return point_cloud2.create_cloud(
        message.header,
        [
            PointField(name=name, offset=index * 4, datatype=PointField.FLOAT32, count=1)
            for index, name in enumerate(field_names)
        ],
        preview.tolist(),
    )


class CapturePreview(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("robot_calibration_capture_preview")
        self._bridge = CvBridge()
        self._minimum_interval = 1.0 / args.max_fps
        self._last_image_time = 0.0
        self._last_cloud_time = 0.0
        self._jpeg_quality = args.jpeg_quality
        self._max_points = args.max_points
        self._image_publisher = self.create_publisher(CompressedImage, args.output_image_topic, qos_profile_sensor_data)
        self._cloud_publisher = self.create_publisher(PointCloud2, args.output_cloud_topic, qos_profile_sensor_data)
        self.create_subscription(Image, args.image_topic, self._image_callback, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, args.cloud_topic, self._cloud_callback, qos_profile_sensor_data)

    def _ready(self, last_time: float) -> tuple[bool, float]:
        now = time.monotonic()
        return now - last_time >= self._minimum_interval, now

    def _image_callback(self, message: Image) -> None:
        ready, now = self._ready(self._last_image_time)
        if not ready:
            return
        frame = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])
        if not encoded:
            self.get_logger().error("failed to encode capture preview as JPEG")
            return
        output = CompressedImage()
        output.header = message.header
        output.format = "jpeg"
        output.data = buffer.tobytes()
        self._image_publisher.publish(output)
        self._last_image_time = now

    def _cloud_callback(self, message: PointCloud2) -> None:
        ready, now = self._ready(self._last_cloud_time)
        if not ready:
            return
        self._cloud_publisher.publish(create_preview_cloud(message, self._max_points))
        self._last_cloud_time = now


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.max_fps <= 0 or not 1 <= args.jpeg_quality <= 100 or args.max_points <= 0:
        raise ValueError("max-fps and max-points must be positive; jpeg-quality must be within 1-100")
    rclpy.init()
    node = CapturePreview(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
