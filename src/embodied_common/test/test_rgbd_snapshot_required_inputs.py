"""Unit tests for build_snapshot's generic required_inputs gating.

A request that declares required_inputs only blocks on the named inputs, so a
pure-vision request (e.g. the Sorting Hat game, which needs primary_image only)
succeeds with EE pose / joint state offline. Omitting required_inputs keeps the
historical strict default (primary image + EE pose + joint state all required).
"""

import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node

from embodied_common.rgbd_snapshot import SceneSnapshotBuffer

_SNAPSHOT_KWARGS = dict(max_scene_age_sec=0.5, max_image_width=320, jpeg_quality=70)


def _buffer_with_image_only(node: Node) -> SceneSnapshotBuffer:
    """A buffer whose primary image is fresh but EE pose / joint state are absent."""
    buffer = SceneSnapshotBuffer(
        node=node,
        primary_camera_topic="/camera/front/image_raw",
        wrist_camera_topic="",
        ee_pose_topic="/robot_status/ee_pose",
        joint_state_topic="/joint_states",
    )
    image_msg = CvBridge().cv2_to_imgmsg(np.zeros((48, 64, 3), dtype=np.uint8), encoding="bgr8")
    primary = buffer._views["primary"]  # noqa: SLF001
    primary.image_msg = image_msg
    primary.image_time = time.monotonic()
    # ee_pose / joint_state intentionally left as None (offline).
    return buffer


def test_required_inputs_primary_image_only_succeeds_without_pose_or_joints():
    rclpy.init()
    try:
        node = Node("test_required_inputs_image_only")
        node.declare_parameter("debug_tracing", False)
        buffer = _buffer_with_image_only(node)

        snapshot = buffer.build_snapshot(required_inputs={"primary_image"}, **_SNAPSHOT_KWARGS)

        assert snapshot["errors"] == []
        assert snapshot["ee_pose"] is None
        assert snapshot["joint_state"] is None
        assert snapshot["image_data_url"]
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_default_required_inputs_still_requires_pose_and_joints():
    rclpy.init()
    try:
        node = Node("test_required_inputs_strict_default")
        node.declare_parameter("debug_tracing", False)
        buffer = _buffer_with_image_only(node)

        # Omitting required_inputs preserves the historical strict behavior.
        snapshot = buffer.build_snapshot(**_SNAPSHOT_KWARGS)

        assert any("ee pose unavailable" in err for err in snapshot["errors"])
        assert any("joint state unavailable" in err for err in snapshot["errors"])
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_required_inputs_without_primary_image_does_not_block_on_image():
    rclpy.init()
    try:
        node = Node("test_required_inputs_no_image")
        node.declare_parameter("debug_tracing", False)
        # No primary image injected; only ee_pose required this time.
        buffer = SceneSnapshotBuffer(
            node=node,
            primary_camera_topic="/camera/front/image_raw",
            wrist_camera_topic="",
            ee_pose_topic="/robot_status/ee_pose",
            joint_state_topic="/joint_states",
        )

        snapshot = buffer.build_snapshot(required_inputs={"ee_pose"}, **_SNAPSHOT_KWARGS)

        # Primary image is missing but not required, so it must not appear as an error.
        assert not any("camera image" in err for err in snapshot["errors"])
        # ee_pose is required and offline, so it must be reported.
        assert any("ee pose unavailable" in err for err in snapshot["errors"])
    finally:
        node.destroy_node()
        rclpy.shutdown()
