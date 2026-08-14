"""Decode the low-bandwidth calibration JPEG preview with sensor-data QoS."""

import argparse

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input-topic", default="/calib/preview/image/compressed")
    result.add_argument("--output-topic", default="/calib/preview/image")
    result.add_argument("--node-name", default="robot_calibration_preview_decoder")
    return result


class PreviewDecoder(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args.node_name)
        self._bridge = CvBridge()
        self._publisher = self.create_publisher(Image, args.output_topic, qos_profile_sensor_data)
        self.create_subscription(CompressedImage, args.input_topic, self._callback, qos_profile_sensor_data)

    def _callback(self, message: CompressedImage) -> None:
        encoded = np.frombuffer(message.data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warning("failed to decode calibration JPEG preview")
            return
        output = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        output.header = message.header
        self._publisher.publish(output)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    rclpy.init()
    node = PreviewDecoder(args)
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
