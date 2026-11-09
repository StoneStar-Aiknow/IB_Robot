"""Hand perception drops its camera reader, and the parked frame, while the stream is paused."""

import json
import os
import time
from types import SimpleNamespace

import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Bool

from embodied_common.motion_imitation_contract import PAUSED_STATE_TOPIC
from manipulation_execution import hand_perception_node as module
from manipulation_execution.perception_pause_follower import PAUSED_STATE_QOS

TOPIC = "/test_hand_pause/rgb/compressed"


def _spin_until(executor, predicate):
    deadline = time.monotonic() + 5.0
    while not predicate():
        assert time.monotonic() < deadline, "the local ROS graph did not reach the expected state"
        executor.spin_once(timeout_sec=0.02)


@pytest.fixture
def stubbed_npu(monkeypatch, tmp_path):
    """The node's own logic with the NPU pipeline and its model files replaced."""
    calib = tmp_path / "calib.json"
    calib.write_text(json.dumps({"endpoints": {side: {"thumb": [0.0, 1.0]} for side in ("left", "right")}}))
    backend = SimpleNamespace(description="stub", close=lambda: None, reset=lambda: None)
    monkeypatch.setattr(
        module,
        "npu_hands",
        SimpleNamespace(
            load_extern=lambda *_args: SimpleNamespace(CURL_FINGERS=("thumb",)),
            model_paths=lambda _model_dir: {},
            extern_path=lambda _extern_dir, name: tmp_path / name,
            file_sha256=lambda _path: "stub",
            EXTERN_FILES=(),
            CALIB_MEDIAPIPE_JSON="calib.json",
            KEYPOINTS_PY="keypoints.py",
            NpuHands=lambda *_args: backend,
        ),
    )
    return calib


def test_pause_releases_the_camera_and_forgets_the_parked_frame(stubbed_npu):
    assert os.environ.get("ROS_LOCALHOST_ONLY") == "1"
    rclpy.init()
    hand = source = None
    executor = SingleThreadedExecutor()
    try:
        hand = module.HandPerceptionNode(
            parameter_overrides=[
                Parameter("camera_topic", value=TOPIC),
                Parameter("calib_path", value=str(stubbed_npu)),
            ]
        )
        source = Node("test_hand_pause_source")
        camera = source.create_publisher(CompressedImage, TOPIC, qos_profile_sensor_data)
        state = source.create_publisher(Bool, PAUSED_STATE_TOPIC, PAUSED_STATE_QOS)
        executor.add_node(hand)
        executor.add_node(source)
        _spin_until(executor, lambda: camera.get_subscription_count() == 1)

        hand._stopping.set()  # Keep the worker from taking the parked frame.
        hand._slot = (CompressedImage(), 0.0)
        state.publish(Bool(data=True))
        _spin_until(executor, lambda: camera.get_subscription_count() == 0)
        assert hand.camera.paused and hand._slot is None

        state.publish(Bool(data=False))
        _spin_until(executor, lambda: camera.get_subscription_count() == 1)
        assert not hand.camera.paused
    finally:
        executor.shutdown(timeout_sec=5.0)
        for node in (hand, source):
            if node is not None:
                node.destroy_node()
        rclpy.shutdown()
