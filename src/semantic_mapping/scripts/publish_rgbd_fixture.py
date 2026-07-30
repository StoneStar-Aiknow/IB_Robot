#!/usr/bin/env python3
"""Publish an aligned RGB-D fixture and verify the semantic-map query service."""

import argparse
import json
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster

from ibrobot_msgs.srv import GetSemanticObjects


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--timeout-sec", type=float, default=120.0)
    parser.add_argument("--global-frame", default="camera_init")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    return parser.parse_args()


class FixturePublisher(Node):
    def __init__(self, fixture: Path, global_frame: str, camera_frame: str):
        super().__init__("semantic_mapping_fixture_publisher")
        image = cv2.imread(str(fixture / "color.png"))
        depth = cv2.imread(str(fixture / "depth.png"), cv2.IMREAD_UNCHANGED)
        if image is None or depth is None:
            raise FileNotFoundError(f"RGB-D fixture is incomplete: {fixture}")
        if image.shape[:2] != depth.shape[:2]:
            raise ValueError("fixture depth must be aligned to the color image")

        camera_info_data = json.loads((fixture / "camera_info.json").read_text())
        self._bridge = CvBridge()
        self._color = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        self._depth = self._bridge.cv2_to_imgmsg(depth, encoding="16UC1")
        self._camera_info = CameraInfo(
            height=int(camera_info_data["height"]),
            width=int(camera_info_data["width"]),
            distortion_model=camera_info_data["distortion_model"],
            d=camera_info_data["d"],
            k=camera_info_data["k"],
            r=camera_info_data["r"],
            p=camera_info_data["p"],
        )
        self._camera_frame = camera_frame
        self._color_pub = self.create_publisher(Image, "/camera/front/image_raw", qos_profile_sensor_data)
        self._depth_pub = self.create_publisher(
            Image, "/camera/front/aligned_depth_to_color/image_raw", qos_profile_sensor_data
        )
        self._info_pub = self.create_publisher(CameraInfo, "/camera/front/camera_info", qos_profile_sensor_data)
        self._client = self.create_client(GetSemanticObjects, "/semantic_mapping/get_objects")
        self.create_timer(0.2, self._publish)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = global_frame
        transform.child_frame_id = camera_frame
        transform.transform.translation.x = 1.0
        transform.transform.translation.y = 2.0
        transform.transform.rotation.w = 1.0
        self._tf_broadcaster = StaticTransformBroadcaster(self)
        self._tf_broadcaster.sendTransform(transform)

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        for message in (self._color, self._depth, self._camera_info):
            message.header.stamp = stamp
            message.header.frame_id = self._camera_frame
        self._color_pub.publish(self._color)
        self._depth_pub.publish(self._depth)
        self._info_pub.publish(self._camera_info)

    def query(self):
        request = GetSemanticObjects.Request()
        request.include_inactive = True
        return self._client.call_async(request)


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = FixturePublisher(Path(args.fixture), args.global_frame, args.camera_frame)
    deadline = time.monotonic() + args.timeout_sec
    try:
        if not node._client.wait_for_service(timeout_sec=min(30.0, args.timeout_sec)):
            raise RuntimeError("semantic-map query service did not become available")
        while time.monotonic() < deadline:
            future = node.query()
            rclpy.spin_until_future_complete(node, future, timeout_sec=2.0)
            if future.done() and future.result() is not None:
                objects = future.result().semantic_map.objects
                if objects:
                    for semantic_object in objects:
                        position = semantic_object.pose.pose.pose.position
                        print(
                            f"label={semantic_object.label!r} id={semantic_object.object_id} "
                            f"position={[round(position.x, 4), round(position.y, 4), round(position.z, 4)]}"
                        )
                    print(f"ROS_SEMANTIC_MAPPING_NODE=PASS objects={len(objects)}")
                    return
            rclpy.spin_once(node, timeout_sec=0.2)
        raise TimeoutError("semantic mapping did not produce an object before the timeout")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
