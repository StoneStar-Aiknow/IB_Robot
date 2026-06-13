#!/usr/bin/env python3
"""Validate embodied motion effects using joint states and camera image changes."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String


@dataclass(frozen=True)
class MotionEffectMetrics:
    max_joint_delta_rad: float
    mean_image_delta: float | None
    image_samples_before: int
    image_samples_after: int
    joint_names_moved: list[str]


def joint_position_map(msg: JointState) -> dict[str, float]:
    return {
        name: float(position)
        for name, position in zip(msg.name, msg.position, strict=False)
        if name and math.isfinite(float(position))
    }


def joint_deltas(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {name: abs(after[name] - before[name]) for name in before.keys() & after.keys()}


def max_joint_delta(before: dict[str, float], after: dict[str, float]) -> tuple[float, list[str]]:
    deltas = joint_deltas(before, after)
    if not deltas:
        return 0.0, []
    max_delta = max(deltas.values())
    moved_names = sorted(name for name, delta in deltas.items() if delta > 0.0)
    return max_delta, moved_names


def max_joint_delta_across_samples(
    before: dict[str, float], samples: Iterable[dict[str, float]]
) -> tuple[float, list[str]]:
    max_delta = 0.0
    moved_names: set[str] = set()
    for sample in samples:
        deltas = joint_deltas(before, sample)
        if not deltas:
            continue
        sample_max = max(deltas.values())
        max_delta = max(max_delta, sample_max)
        moved_names.update(name for name, delta in deltas.items() if delta > 0.0)
    return max_delta, sorted(moved_names)


def image_to_gray(msg: Image) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in {"rgb8", "bgr8"}:
        return data.reshape(msg.height, msg.width, 3).astype(np.float32).mean(axis=2)
    if msg.encoding in {"rgba8", "bgra8"}:
        return data.reshape(msg.height, msg.width, 4).astype(np.float32)[:, :, :3].mean(axis=2)
    if msg.encoding in {"mono8", "8UC1"}:
        return data.reshape(msg.height, msg.width).astype(np.float32)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def mean_image_delta(before: Image | None, after: Image | None) -> float | None:
    if before is None or after is None:
        return None
    before_gray = image_to_gray(before)
    after_gray = image_to_gray(after)
    if before_gray.shape != after_gray.shape:
        raise ValueError(f"image shape changed from {before_gray.shape} to {after_gray.shape}")
    return float(np.mean(np.abs(after_gray - before_gray)))


def motion_effect_passed(
    metrics: MotionEffectMetrics,
    *,
    min_joint_delta_rad: float,
    min_image_delta: float,
    require_image_change: bool,
) -> bool:
    joint_ok = metrics.max_joint_delta_rad >= min_joint_delta_rad
    image_ok = metrics.mean_image_delta is not None and metrics.mean_image_delta >= min_image_delta
    return joint_ok and (image_ok or not require_image_change)


class MotionEffectProbe(Node):
    def __init__(self, *, joint_topic: str, image_topic: str, command_topic: str) -> None:
        super().__init__("embodied_motion_effect_probe")
        self.latest_joint_state: JointState | None = None
        self.latest_image: Image | None = None
        self.image_count = 0
        self.create_subscription(JointState, joint_topic, self._joint_cb, 10)
        self.create_subscription(Image, image_topic, self._image_cb, 10)
        self.command_pub = self.create_publisher(String, command_topic, 10)

    def _joint_cb(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def _image_cb(self, msg: Image) -> None:
        self.latest_image = msg
        self.image_count += 1

    def publish_command(self, command: str) -> None:
        msg = String()
        msg.data = command
        self.command_pub.publish(msg)


def spin_until(node: MotionEffectProbe, predicate, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
        if predicate():
            return True
    return False


def collect_motion_effect(args: argparse.Namespace) -> MotionEffectMetrics:
    rclpy.init()
    node = MotionEffectProbe(
        joint_topic=args.joint_topic,
        image_topic=args.image_topic,
        command_topic=args.command_topic,
    )
    try:
        if not spin_until(node, lambda: node.latest_joint_state is not None, args.initial_timeout_sec):
            raise RuntimeError(f"no JointState received on {args.joint_topic}")

        if args.require_image and not spin_until(node, lambda: node.latest_image is not None, args.initial_timeout_sec):
            raise RuntimeError(f"no Image received on {args.image_topic}")

        before_joint = joint_position_map(node.latest_joint_state)
        before_image = node.latest_image
        before_image_count = node.image_count

        node.publish_command(args.command)
        action_start = time.monotonic()
        joint_samples = []
        while time.monotonic() - action_start < args.settle_sec:
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.latest_joint_state is not None:
                joint_samples.append(joint_position_map(node.latest_joint_state))

        after_joint = joint_position_map(node.latest_joint_state)
        after_image = node.latest_image
        after_image_count = node.image_count
        max_delta, moved_names = max_joint_delta_across_samples(before_joint, [*joint_samples, after_joint])
        image_delta = mean_image_delta(before_image, after_image)
        return MotionEffectMetrics(
            max_joint_delta_rad=max_delta,
            mean_image_delta=image_delta,
            image_samples_before=before_image_count,
            image_samples_after=after_image_count,
            joint_names_moved=moved_names,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", default="末端往上移动一点", help="Voice command sent to embodied agent")
    parser.add_argument("--command-topic", default="/voice_command")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--image-topic", default="/camera/front_camera/color/image_raw")
    parser.add_argument("--initial-timeout-sec", type=float, default=10.0)
    parser.add_argument("--settle-sec", type=float, default=8.0)
    parser.add_argument("--min-joint-delta-rad", type=float, default=0.005)
    parser.add_argument("--min-image-delta", type=float, default=2.0)
    parser.add_argument("--require-image", action="store_true", default=False)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    metrics = collect_motion_effect(args)
    passed = motion_effect_passed(
        metrics,
        min_joint_delta_rad=args.min_joint_delta_rad,
        min_image_delta=args.min_image_delta,
        require_image_change=args.require_image,
    )
    print(json.dumps({"passed": passed, "metrics": metrics.__dict__}, ensure_ascii=False, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
