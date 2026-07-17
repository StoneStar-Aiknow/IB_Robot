from __future__ import annotations

import atexit
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticStatus
from rclpy.action import ActionClient

from ibrobot_msgs.action import DispatchInfer
from inference_manifest import BundleFile, canonical_bundle_digest, sha256_file
from tensormsg.converter import TensorMsgConverter


@dataclass(frozen=True)
class SmokeRuntime:
    bundle_path: Path
    robot_config_path: Path
    process_environment: dict[str, str]


def prepare_smoke_runtime() -> SmokeRuntime:
    temp_dir = tempfile.TemporaryDirectory(prefix="ibrobot-inference-smoke-")
    atexit.register(temp_dir.cleanup)
    root = Path(temp_dir.name)
    bundle = root / "policy_bundle"
    bundle.mkdir()

    config = {
        "type": "act",
        "n_obs_steps": 1,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [6]},
            "observation.images.top": {"type": "VISUAL", "shape": [3, 16, 24]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        "device": "cuda",
        "chunk_size": 2,
        "n_action_steps": 2,
    }
    preprocessor = {
        "name": "policy_preprocessor",
        "steps": [
            {"registry_name": "to_batch_processor", "config": {}},
            {"registry_name": "device_processor", "config": {"device": "cuda"}},
            {
                "registry_name": "normalizer_processor",
                "config": {
                    "features": {**config["input_features"], **config["output_features"]},
                },
                "state_file": "policy_preprocessor_step_2_normalizer_processor.safetensors",
            },
        ],
    }
    postprocessor = {
        "name": "policy_postprocessor",
        "steps": [
            {
                "registry_name": "unnormalizer_processor",
                "config": {"features": config["output_features"]},
                "state_file": "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            },
            {"registry_name": "device_processor", "config": {"device": "cpu"}},
        ],
    }
    _write_json(bundle / "config.json", config)
    _write_json(bundle / "policy_preprocessor.json", preprocessor)
    _write_json(bundle / "policy_postprocessor.json", postprocessor)
    (bundle / "model.safetensors").write_bytes(b"ci-smoke-policy")
    (bundle / "policy_preprocessor_step_2_normalizer_processor.safetensors").write_bytes(b"pre-state")
    (bundle / "policy_postprocessor_step_0_unnormalizer_processor.safetensors").write_bytes(b"post-state")

    bundle_paths = (
        "config.json",
        "model.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_2_normalizer_processor.safetensors",
    )
    entries = [BundleFile(path=path, sha256=sha256_file(bundle / path)) for path in bundle_paths]
    _write_json(
        bundle / "inference_manifest.json",
        {
            "schema_version": 1,
            "bundle": {
                "name": "ci-smoke-act",
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {"algorithm": "sha256", "value": canonical_bundle_digest(entries)},
            },
            "deployments": {"cpu": {"backend": "torch", "device": "cpu"}},
        },
    )

    robot_config = {
        "robot": {
            "name": "ci_smoke_robot",
            "type": "test",
            "robot_type": "test",
            "contract": {
                "rate_hz": 10.0,
                "observations": [
                    {
                        "key": "observation.state",
                        "topic": "/ci_smoke/joint_states",
                        "type": "sensor_msgs/msg/JointState",
                        "selector": {"names": [f"position.joint_{index}" for index in range(6)]},
                    },
                    {
                        "key": "observation.images.top",
                        "topic": "/ci_smoke/camera/top",
                        "type": "sensor_msgs/msg/Image",
                        "image": {"resize": [16, 24], "encoding": "rgb8"},
                    },
                ],
                "actions": [],
            },
        }
    }
    robot_config_path = root / "robot.yaml"
    _write_json(robot_config_path, robot_config)

    fake_lerobot = Path(__file__).parent / "fake_lerobot"
    process_environment = os.environ.copy()
    current_pythonpath = process_environment.get("PYTHONPATH", "")
    process_environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(fake_lerobot), current_pythonpath) if path
    )
    process_environment["PYTHONUNBUFFERED"] = "1"
    return SmokeRuntime(bundle, robot_config_path, process_environment)


def send_inference(node, action_name: str, request_id: str, *, timeout: float = 20.0):
    client = ActionClient(node, DispatchInfer, action_name)
    try:
        deadline = time.monotonic() + timeout
        while not client.wait_for_server(timeout_sec=0.2):
            if time.monotonic() >= deadline:
                raise AssertionError(f"action server {action_name!r} did not become available")

        goal = DispatchInfer.Goal()
        goal.inference_id = request_id
        goal_future = client.send_goal_async(goal)
        _spin_until_complete(node, goal_future, deadline, f"goal {request_id!r}")
        goal_handle = goal_future.result()
        if goal_handle is None or not goal_handle.accepted:
            raise AssertionError(f"goal {request_id!r} was rejected")

        result_future = goal_handle.get_result_async()
        _spin_until_complete(node, result_future, deadline, f"result {request_id!r}")
        wrapped_result = result_future.result()
        if wrapped_result is None:
            raise AssertionError(f"goal {request_id!r} returned no result")
        return wrapped_result.result
    finally:
        client.destroy()


def assert_successful_action(test_case, result, pipeline_id: str, request_id: str) -> None:
    test_case.assertTrue(result.success, result.message)
    test_case.assertEqual(pipeline_id, result.pipeline_id)
    test_case.assertEqual(request_id, result.request_id)
    test_case.assertEqual(2, result.chunk_size)
    test_case.assertGreaterEqual(result.backend_latency_ms, 0.0)
    decoded = TensorMsgConverter.from_variant(result.action_chunk)["action"]
    test_case.assertEqual((2, 6), tuple(decoded.shape))
    np.testing.assert_allclose(decoded.detach().cpu().numpy(), np.arange(12, dtype=np.float32).reshape(2, 6))


def wait_for_distributed_ready(node, health_topic: str, *, timeout: float = 20.0) -> dict[str, str]:
    ready_values: dict[str, str] = {}

    def callback(message: DiagnosticStatus) -> None:
        nonlocal ready_values
        values = {item.key: item.value for item in message.values}
        if (
            message.level == DiagnosticStatus.OK
            and values.get("state") == "ready"
            and values.get("remote_state") == "ready"
            and values.get("session_id")
            and int(values.get("session_generation", "0")) >= 1
        ):
            ready_values = values

    subscription = node.create_subscription(DiagnosticStatus, health_topic, callback, 10)
    try:
        deadline = time.monotonic() + timeout
        while not ready_values and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not ready_values:
            raise AssertionError(f"distributed pipeline on {health_topic!r} did not become ready")
        return ready_values
    finally:
        node.destroy_subscription(subscription)


def _spin_until_complete(node, future, deadline: float, description: str) -> None:
    while not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not future.done():
        raise AssertionError(f"timed out waiting for {description}")
    exception = future.exception()
    if exception is not None:
        raise exception


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
