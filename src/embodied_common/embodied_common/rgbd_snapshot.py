"""Shared RGB-D scene snapshot collection for perception and VLM planning."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2

from embodied_common.depth_context import (
    summarize_camera_info,
    summarize_depth_frame,
    summarize_pointcloud_metadata,
)

# Generic snapshot input keys a request may declare via ``required_inputs``.
# A request that omits ``required_inputs`` keeps the historical strict behavior
# (primary image + EE pose + joint state all required); a request that names a
# subset only blocks on those inputs, letting pure-vision requests succeed with
# EE pose / joint state offline. Unknown keys are ignored by callers.
PRIMARY_IMAGE_INPUT = "primary_image"
EE_POSE_INPUT = "ee_pose"
JOINT_STATE_INPUT = "joint_state"
KNOWN_REQUIRED_INPUTS = frozenset({PRIMARY_IMAGE_INPUT, EE_POSE_INPUT, JOINT_STATE_INPUT})
_DEFAULT_REQUIRED_INPUTS = KNOWN_REQUIRED_INPUTS


@dataclass
class _CameraViewState:
    name: str
    role: str
    image_topic: str = ""
    camera_info_topic: str = ""
    aligned_depth_topic: str = ""
    pointcloud_topic: str = ""
    image_msg: Image | None = None
    image_time: float = 0.0
    camera_info_msg: CameraInfo | None = None
    camera_info_time: float = 0.0
    depth_msg: Image | None = None
    depth_time: float = 0.0
    pointcloud_msg: PointCloud2 | None = None
    pointcloud_time: float = 0.0


class SceneSnapshotBuffer:
    """Maintain the latest multi-view RGB-D scene state for analysis and planning."""

    def __init__(
        self,
        node: Node,
        primary_camera_topic: str,
        wrist_camera_topic: str,
        ee_pose_topic: str,
        joint_state_topic: str,
        primary_camera_info_topic: str = "",
        primary_aligned_depth_topic: str = "",
        primary_pointcloud_topic: str = "",
        wrist_camera_info_topic: str = "",
        wrist_aligned_depth_topic: str = "",
        wrist_pointcloud_topic: str = "",
        debug: bool = False,
    ) -> None:
        self._node = node
        self._debug = debug
        self._bridge = CvBridge()
        self._views: dict[str, _CameraViewState] = {
            "primary": _CameraViewState(
                name="primary",
                role="main_scene",
                image_topic=primary_camera_topic,
                camera_info_topic=primary_camera_info_topic,
                aligned_depth_topic=primary_aligned_depth_topic,
                pointcloud_topic=primary_pointcloud_topic,
            ),
            "wrist": _CameraViewState(
                name="wrist",
                role="end_effector_view",
                image_topic=wrist_camera_topic,
                camera_info_topic=wrist_camera_info_topic,
                aligned_depth_topic=wrist_aligned_depth_topic,
                pointcloud_topic=wrist_pointcloud_topic,
            ),
        }

        self._latest_ee_pose: PoseStamped | None = None
        self._latest_ee_pose_time = 0.0
        self._latest_joint_state: JointState | None = None
        self._latest_joint_state_time = 0.0

        for view_name, view in self._views.items():
            if view.image_topic:
                node.create_subscription(
                    Image,
                    view.image_topic,
                    self._make_image_handler(view_name),
                    qos_profile_sensor_data,
                )
            if view.camera_info_topic:
                node.create_subscription(
                    CameraInfo,
                    view.camera_info_topic,
                    self._make_camera_info_handler(view_name),
                    qos_profile_sensor_data,
                )
            if view.aligned_depth_topic:
                node.create_subscription(
                    Image,
                    view.aligned_depth_topic,
                    self._make_depth_handler(view_name),
                    qos_profile_sensor_data,
                )
            if view.pointcloud_topic:
                node.create_subscription(
                    PointCloud2,
                    view.pointcloud_topic,
                    self._make_pointcloud_handler(view_name),
                    qos_profile_sensor_data,
                )

        node.create_subscription(PoseStamped, ee_pose_topic, self._handle_ee_pose, 10)
        node.create_subscription(JointState, joint_state_topic, self._handle_joint_state, 10)

    @classmethod
    def from_node(cls, node: Node) -> SceneSnapshotBuffer:
        """Construct from a ROS node using the standard camera/pose parameters."""

        def _str(name: str) -> str:
            return node.get_parameter(name).get_parameter_value().string_value

        return cls(
            node,
            primary_camera_topic=_str("primary_camera_topic"),
            wrist_camera_topic=_str("wrist_camera_topic"),
            primary_camera_info_topic=_str("primary_camera_info_topic"),
            primary_aligned_depth_topic=_str("primary_aligned_depth_topic"),
            primary_pointcloud_topic=_str("primary_pointcloud_topic"),
            wrist_camera_info_topic=_str("wrist_camera_info_topic"),
            wrist_aligned_depth_topic=_str("wrist_aligned_depth_topic"),
            wrist_pointcloud_topic=_str("wrist_pointcloud_topic"),
            ee_pose_topic=_str("ee_pose_topic"),
            joint_state_topic=_str("joint_state_topic"),
            debug=node.get_parameter("debug_tracing").get_parameter_value().bool_value,
        )

    def _make_image_handler(self, view_name: str):
        def _handler(msg: Image) -> None:
            view = self._views[view_name]
            view.image_msg = msg
            view.image_time = time.monotonic()
            if self._debug and view_name == "wrist":
                self._node.get_logger().debug(f"[embodied-debug] received wrist image on {view.image_topic}")

        return _handler

    def _make_camera_info_handler(self, view_name: str):
        def _handler(msg: CameraInfo) -> None:
            view = self._views[view_name]
            view.camera_info_msg = msg
            view.camera_info_time = time.monotonic()

        return _handler

    def _make_depth_handler(self, view_name: str):
        def _handler(msg: Image) -> None:
            view = self._views[view_name]
            view.depth_msg = msg
            view.depth_time = time.monotonic()

        return _handler

    def _make_pointcloud_handler(self, view_name: str):
        def _handler(msg: PointCloud2) -> None:
            view = self._views[view_name]
            view.pointcloud_msg = msg
            view.pointcloud_time = time.monotonic()

        return _handler

    def _handle_ee_pose(self, msg: PoseStamped) -> None:
        self._latest_ee_pose = msg
        self._latest_ee_pose_time = time.monotonic()

    def _handle_joint_state(self, msg: JointState) -> None:
        self._latest_joint_state = msg
        self._latest_joint_state_time = time.monotonic()

    def _image_to_data_url(self, msg: Image, max_width: int, jpeg_quality: int) -> str:
        cv_image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if max_width > 0 and cv_image.shape[1] > max_width:
            scale = float(max_width) / float(cv_image.shape[1])
            target_size = (max_width, max(1, int(cv_image.shape[0] * scale)))
            cv_image = cv2.resize(cv_image, target_size, interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(
            ".jpg",
            cv_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
        )
        if not ok:
            raise RuntimeError("failed to encode camera image as JPEG")
        image_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{image_b64}"

    def _depth_msg_to_meters(self, msg: Image) -> np.ndarray:
        depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_array = np.asarray(depth)
        if depth_array.dtype == np.uint16:
            return depth_array.astype(np.float32) / 1000.0
        return depth_array.astype(np.float32)

    def _pose_to_dict(self) -> dict[str, Any] | None:
        if self._latest_ee_pose is None:
            return None
        pose = self._latest_ee_pose.pose
        return {
            "frame_id": self._latest_ee_pose.header.frame_id,
            "position": {
                "x": float(pose.position.x),
                "y": float(pose.position.y),
                "z": float(pose.position.z),
            },
            "orientation": {
                "x": float(pose.orientation.x),
                "y": float(pose.orientation.y),
                "z": float(pose.orientation.z),
                "w": float(pose.orientation.w),
            },
        }

    def _joint_state_to_dict(self) -> dict[str, Any] | None:
        if self._latest_joint_state is None:
            return None
        return {
            "name": list(self._latest_joint_state.name),
            "position": [float(value) for value in self._latest_joint_state.position],
        }

    def _build_view_snapshot(
        self,
        view: _CameraViewState,
        now: float,
        max_scene_age_sec: float,
        max_image_width: int,
        jpeg_quality: int,
        required_image: bool,
    ) -> tuple[dict[str, Any], list[str]]:
        issues = []
        image_data_url = ""

        if not view.image_topic:
            if required_image:
                issues.append(f"{view.name} camera image topic is not configured")
        elif view.image_msg is None:
            if required_image:
                issues.append(f"{view.name} camera image unavailable")
        elif now - view.image_time > max_scene_age_sec:
            if required_image:
                issues.append(f"{view.name} camera image is stale: {now - view.image_time:.3f}s")
        else:
            image_data_url = self._image_to_data_url(
                view.image_msg,
                max_width=max_image_width,
                jpeg_quality=jpeg_quality,
            )

        camera_info_summary = summarize_camera_info(view.camera_info_msg)
        if view.camera_info_topic and view.camera_info_msg and now - view.camera_info_time > max_scene_age_sec:
            camera_info_summary = dict(camera_info_summary)
            camera_info_summary["available"] = False
            camera_info_summary["stale"] = True
            camera_info_summary["message"] = f"{view.name} camera_info is stale: {now - view.camera_info_time:.3f}s"
        elif view.camera_info_topic and view.camera_info_msg is None:
            camera_info_summary = {"available": False, "message": f"{view.name} camera_info unavailable"}

        depth_summary: dict[str, Any]
        if not view.aligned_depth_topic:
            depth_summary = {"available": False}
        elif view.depth_msg is None:
            depth_summary = {"available": False, "message": f"{view.name} depth image unavailable"}
        elif now - view.depth_time > max_scene_age_sec:
            depth_summary = {
                "available": False,
                "stale": True,
                "message": f"{view.name} depth image is stale: {now - view.depth_time:.3f}s",
            }
        else:
            depth_summary = summarize_depth_frame(
                self._depth_msg_to_meters(view.depth_msg),
                camera_info=view.camera_info_msg,
            )
            depth_summary["topic"] = view.aligned_depth_topic

        pointcloud_summary = summarize_pointcloud_metadata(view.pointcloud_msg)
        if view.pointcloud_topic and view.pointcloud_msg and now - view.pointcloud_time > max_scene_age_sec:
            pointcloud_summary = dict(pointcloud_summary)
            pointcloud_summary["available"] = False
            pointcloud_summary["stale"] = True
            pointcloud_summary["message"] = f"{view.name} pointcloud is stale: {now - view.pointcloud_time:.3f}s"
        elif view.pointcloud_topic and view.pointcloud_msg is None:
            pointcloud_summary = {"available": False, "message": f"{view.name} pointcloud unavailable"}
        if view.pointcloud_topic and pointcloud_summary:
            pointcloud_summary["topic"] = view.pointcloud_topic

        return (
            {
                "view_name": view.name,
                "role": view.role,
                "camera_topic": view.image_topic,
                "image_data_url": image_data_url,
                "camera_info_topic": view.camera_info_topic,
                "aligned_depth_topic": view.aligned_depth_topic,
                "pointcloud_topic": view.pointcloud_topic,
                "camera_info": camera_info_summary,
                "depth_summary": depth_summary,
                "pointcloud_summary": pointcloud_summary,
                "issues": issues,
            },
            issues,
        )

    def build_snapshot(
        self,
        max_scene_age_sec: float,
        max_image_width: int,
        jpeg_quality: int,
        require_depth: bool = False,
        require_pointcloud: bool = False,
        required_inputs: set[str] | None = None,
    ) -> dict[str, Any]:
        now = time.monotonic()
        errors = []

        # None => historical strict behavior (all inputs required); an explicit
        # set only blocks on the named inputs, so a pure-vision request can
        # succeed with EE pose / joint state offline.
        effective_inputs = _DEFAULT_REQUIRED_INPUTS if required_inputs is None else required_inputs

        primary_view, primary_issues = self._build_view_snapshot(
            self._views["primary"],
            now=now,
            max_scene_age_sec=max_scene_age_sec,
            max_image_width=max_image_width,
            jpeg_quality=jpeg_quality,
            required_image=PRIMARY_IMAGE_INPUT in effective_inputs,
        )
        wrist_view, wrist_issues = self._build_view_snapshot(
            self._views["wrist"],
            now=now,
            max_scene_age_sec=max_scene_age_sec,
            max_image_width=max_image_width,
            jpeg_quality=jpeg_quality,
            required_image=False,
        )
        errors.extend(primary_issues)

        has_any_depth = any(view["depth_summary"].get("available", False) for view in (primary_view, wrist_view))
        has_any_pointcloud = any(
            view["pointcloud_summary"].get("available", False) for view in (primary_view, wrist_view)
        )

        if require_depth and not has_any_depth:
            message = primary_view["depth_summary"].get("message", "depth data unavailable")
            errors.append(message)
        if require_pointcloud and not has_any_pointcloud:
            message = primary_view["pointcloud_summary"].get("message", "pointcloud unavailable")
            errors.append(message)

        if EE_POSE_INPUT in effective_inputs:
            if self._latest_ee_pose is None:
                errors.append("ee pose unavailable")
            elif now - self._latest_ee_pose_time > max_scene_age_sec:
                errors.append(f"ee pose is stale: {now - self._latest_ee_pose_time:.3f}s")

        if JOINT_STATE_INPUT in effective_inputs:
            if self._latest_joint_state is None:
                errors.append("joint state unavailable")
            elif now - self._latest_joint_state_time > max_scene_age_sec:
                errors.append(f"joint state is stale: {now - self._latest_joint_state_time:.3f}s")

        images = []
        for view in (primary_view, wrist_view):
            if view["image_data_url"]:
                images.append(
                    {
                        "view_name": view["view_name"],
                        "role": view["role"],
                        "camera_topic": view["camera_topic"],
                        "image_data_url": view["image_data_url"],
                    }
                )

        rgbd_context = {
            "multi_view_enabled": len(images) > 1,
            "available_views": [item["view_name"] for item in images],
            "depth_available_views": [
                view["view_name"]
                for view in (primary_view, wrist_view)
                if view["depth_summary"].get("available", False)
            ],
            "pointcloud_available_views": [
                view["view_name"]
                for view in (primary_view, wrist_view)
                if view["pointcloud_summary"].get("available", False)
            ],
            "views": {
                "primary": {
                    "role": primary_view["role"],
                    "camera_topic": primary_view["camera_topic"],
                    "camera_info": primary_view["camera_info"],
                    "depth_summary": primary_view["depth_summary"],
                    "pointcloud_summary": primary_view["pointcloud_summary"],
                    "issues": primary_view["issues"],
                },
                "wrist": {
                    "role": wrist_view["role"],
                    "camera_topic": wrist_view["camera_topic"],
                    "camera_info": wrist_view["camera_info"],
                    "depth_summary": wrist_view["depth_summary"],
                    "pointcloud_summary": wrist_view["pointcloud_summary"],
                    "issues": wrist_issues,
                },
            },
        }

        return {
            "camera_topic": primary_view["camera_topic"],
            "image_data_url": primary_view["image_data_url"],
            "camera_info": primary_view["camera_info"],
            "depth_summary": primary_view["depth_summary"],
            "pointcloud_summary": primary_view["pointcloud_summary"],
            "wrist_camera_topic": wrist_view["camera_topic"],
            "wrist_image_data_url": wrist_view["image_data_url"],
            "images": images,
            "camera_views": {
                "primary": primary_view,
                "wrist": wrist_view,
            },
            "rgbd_context": rgbd_context,
            "ee_pose": self._pose_to_dict(),
            "joint_state": self._joint_state_to_dict(),
            "errors": errors,
        }
