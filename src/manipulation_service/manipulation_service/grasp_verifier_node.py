"""ROS 2 node for post-grasp success verification."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState

from ibrobot_msgs.msg import JointCurrent
from ibrobot_msgs.srv import VerifyGrasp

from .grasp_verification import (
    DepthVisibilityStats,
    GraspVerificationInput,
    evaluate_grasp,
)


@dataclass
class TimedSample:
    received_ns: int
    value: object


def _image_to_depth_m(msg: Image, depth_scale: float) -> np.ndarray:
    encoding = msg.encoding.upper()
    if encoding in {"16UC1", "MONO16"}:
        dtype = np.uint16
        scale = depth_scale
    elif encoding == "32FC1":
        dtype = np.float32
        scale = 1.0
    elif encoding == "64FC1":
        dtype = np.float64
        scale = 1.0
    else:
        raise ValueError(f"unsupported depth encoding: {msg.encoding}")

    item_size = np.dtype(dtype).itemsize
    row_items = max(1, int(msg.step) // item_size)
    raw = np.frombuffer(msg.data, dtype=dtype)
    if raw.size < row_items * msg.height:
        raise ValueError(
            f"depth image data too short: have {raw.size} items, expected at least {row_items * msg.height}"
        )
    depth = raw[: row_items * msg.height].reshape((msg.height, row_items))[:, : msg.width]
    depth = depth.astype(np.float32, copy=False)
    if scale != 1.0:
        depth = depth / float(scale)
    return depth


def _depth_visibility_stats(
    msg: Image,
    *,
    depth_scale: float,
    min_valid_fraction: float,
    near_depth_m: float,
    max_near_fraction: float,
) -> DepthVisibilityStats:
    depth = _image_to_depth_m(msg, depth_scale)
    valid = np.isfinite(depth) & (depth > 0.0)
    valid_count = int(valid.sum())
    total = int(depth.size)
    valid_fraction = float(valid_count / total) if total else 0.0
    if valid_count == 0:
        return DepthVisibilityStats(
            valid_fraction=0.0,
            near_fraction=0.0,
            median_depth_m=None,
            occluded=True,
        )

    valid_depth = depth[valid]
    near_fraction = float(np.count_nonzero(valid_depth <= near_depth_m) / valid_count)
    median_depth_m = float(np.median(valid_depth))
    occluded = valid_fraction < min_valid_fraction or near_fraction > max_near_fraction
    return DepthVisibilityStats(
        valid_fraction=valid_fraction,
        near_fraction=near_fraction,
        median_depth_m=median_depth_m,
        occluded=occluded,
    )


class GraspVerifierNode(Node):
    """Samples post-grasp sensors and exposes ``~/verify_grasp``."""

    def __init__(self):
        super().__init__("grasp_verifier")

        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("joint_current_topic", "/so101_follower/joint_currents")
        self.declare_parameter("wrist_depth_topic", "/camera/wrist/aligned_depth_to_color/image_raw")
        self.declare_parameter(
            "gripper_joint",
            "",
            descriptor=ParameterDescriptor(dynamic_typing=True),
        )
        self.declare_parameter("max_sample_age_s", 2.0)
        self.declare_parameter("default_post_grasp_wait_s", 0.0)
        self.declare_parameter("gripper_closed_position", 0.0)
        self.declare_parameter("gripper_contact_min_opening", 0.08)
        self.declare_parameter("gripper_no_contact_max_opening", 0.03)
        self.declare_parameter("current_contact_threshold_a", 0.08)
        self.declare_parameter("depth_scale", 1000.0)
        self.declare_parameter("wrist_min_valid_depth_fraction", 0.25)
        self.declare_parameter("wrist_near_depth_m", 0.12)
        self.declare_parameter("wrist_max_near_fraction", 0.70)

        self._lock = threading.Lock()
        self._latest_joint_state: TimedSample | None = None
        self._latest_joint_current: TimedSample | None = None
        self._latest_wrist_depth: TimedSample | None = None

        cb_group = ReentrantCallbackGroup()
        self.create_subscription(
            JointState,
            self.get_parameter("joint_state_topic").value,
            self._joint_state_cb,
            10,
            callback_group=cb_group,
        )
        self.create_subscription(
            JointCurrent,
            self.get_parameter("joint_current_topic").value,
            self._joint_current_cb,
            10,
            callback_group=cb_group,
        )
        self.create_subscription(
            Image,
            self.get_parameter("wrist_depth_topic").value,
            self._wrist_depth_cb,
            qos_profile_sensor_data,
            callback_group=cb_group,
        )
        self.create_service(VerifyGrasp, "~/verify_grasp", self._verify_cb, callback_group=cb_group)

        self.get_logger().info(
            "GraspVerifier ready: service=~/verify_grasp "
            f"joint_state={self.get_parameter('joint_state_topic').value} "
            f"current={self.get_parameter('joint_current_topic').value} "
            f"wrist_depth={self.get_parameter('wrist_depth_topic').value}"
        )

    def _now_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _joint_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._latest_joint_state = TimedSample(received_ns=self._now_ns(), value=msg)

    def _joint_current_cb(self, msg: JointCurrent) -> None:
        with self._lock:
            self._latest_joint_current = TimedSample(received_ns=self._now_ns(), value=msg)

    def _wrist_depth_cb(self, msg: Image) -> None:
        try:
            stats = _depth_visibility_stats(
                msg,
                depth_scale=float(self.get_parameter("depth_scale").value),
                min_valid_fraction=float(self.get_parameter("wrist_min_valid_depth_fraction").value),
                near_depth_m=float(self.get_parameter("wrist_near_depth_m").value),
                max_near_fraction=float(self.get_parameter("wrist_max_near_fraction").value),
            )
        except Exception as exc:
            self.get_logger().debug(f"Skipping wrist depth frame: {exc}")
            return
        with self._lock:
            self._latest_wrist_depth = TimedSample(received_ns=self._now_ns(), value=stats)

    def _verify_cb(self, request: VerifyGrasp.Request, response: VerifyGrasp.Response):
        wait_s = float(request.post_grasp_wait_s)
        if wait_s <= 0.0:
            wait_s = float(self.get_parameter("default_post_grasp_wait_s").value)
        if wait_s > 0.0:
            time.sleep(wait_s)

        input_data = self._build_input(expected_target_width_m=float(request.expected_target_width_m))
        result = evaluate_grasp(input_data)

        response.success = bool(result.success)
        response.status = int(result.status)
        response.confidence = float(result.confidence)
        response.message = result.message
        response.evidence = result.evidence
        self.get_logger().info(
            f"VerifyGrasp task_id={request.task_id!r} prompt={request.text_prompt!r}: "
            f"status={result.status} confidence={result.confidence:.2f} message={result.message}"
        )
        return response

    def _build_input(self, *, expected_target_width_m: float) -> GraspVerificationInput:
        now_ns = self._now_ns()
        max_age_ns = int(float(self.get_parameter("max_sample_age_s").value) * 1_000_000_000)
        with self._lock:
            joint_state_sample = self._latest_joint_state
            current_sample = self._latest_joint_current
            wrist_depth_sample = self._latest_wrist_depth

        joint_state = self._fresh_value(joint_state_sample, now_ns, max_age_ns)
        current = self._fresh_value(current_sample, now_ns, max_age_ns)
        wrist_depth = self._fresh_value(wrist_depth_sample, now_ns, max_age_ns)

        gripper_joint = str(self.get_parameter("gripper_joint").value).strip()
        if not gripper_joint:
            gripper_joint = self._infer_gripper_joint(joint_state, current)

        return GraspVerificationInput(
            gripper_position=self._extract_named_value(joint_state, gripper_joint, "position"),
            gripper_closed_position=float(self.get_parameter("gripper_closed_position").value),
            gripper_contact_min_opening=float(self.get_parameter("gripper_contact_min_opening").value),
            gripper_no_contact_max_opening=float(self.get_parameter("gripper_no_contact_max_opening").value),
            gripper_joint=gripper_joint,
            gripper_current_abs_a=self._extract_abs_current(current, gripper_joint),
            current_contact_threshold_a=float(self.get_parameter("current_contact_threshold_a").value),
            wrist_depth=wrist_depth,
            expected_target_width_m=expected_target_width_m,
        )

    @staticmethod
    def _fresh_value(sample: TimedSample | None, now_ns: int, max_age_ns: int):
        if sample is None or now_ns - sample.received_ns > max_age_ns:
            return None
        return sample.value

    @staticmethod
    def _infer_gripper_joint(joint_state: JointState | None, current: JointCurrent | None) -> str:
        candidate_names: list[str] = []
        if joint_state is not None:
            candidate_names.extend(joint_state.name)
        if current is not None:
            candidate_names.extend(current.name)
        if "6" in candidate_names:
            return "6"
        for name in candidate_names:
            if "gripper" in name.lower():
                return name
        return candidate_names[-1] if candidate_names else ""

    @staticmethod
    def _extract_named_value(msg: JointState | None, joint_name: str, field_name: str) -> float | None:
        if msg is None or not joint_name or joint_name not in msg.name:
            return None
        values = getattr(msg, field_name)
        idx = msg.name.index(joint_name)
        if idx >= len(values):
            return None
        return float(values[idx])

    @staticmethod
    def _extract_abs_current(msg: JointCurrent | None, joint_name: str) -> float | None:
        if msg is None or not joint_name or joint_name not in msg.name:
            return None
        idx = msg.name.index(joint_name)
        if idx >= len(msg.current):
            return None
        return abs(float(msg.current[idx]))


def main(args=None):
    rclpy.init(args=args)
    node = GraspVerifierNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
