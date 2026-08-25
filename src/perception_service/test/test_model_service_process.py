import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import pytest
import rclpy

from ibrobot_msgs.srv import EchoModel
from inference_manifest import (
    BundleFile,
    canonical_bundle_digest,
    canonical_semantic_identity_json,
    load_inference_manifest,
    semantic_identity_fingerprint,
)


def _bundle(root: Path) -> Path:
    root.mkdir()
    marker = root / "identity.txt"
    marker.write_text("dummy-echo", encoding="utf-8")
    entry = BundleFile(path=marker.name)
    manifest = {
        "schema_version": 3,
        "bundle": {
            "uuid": "123e4567-e89b-42d3-a456-426614174000",
            "revision": 1,
            "name": "dummy-echo",
            "files": [entry.model_dump(mode="json")],
            "digest": {
                "algorithm": "sha256",
                "scope": "structure",
                "value": canonical_bundle_digest("123e4567-e89b-42d3-a456-426614174000", 1, "dummy-echo", [entry]),
            },
        },
        "model": {
            "interface": "tensor_model",
            "model_type": "dummy_echo",
            "operation": "echo",
            "inputs": [{"semantic": "echo.input", "dtype": "float32", "shape": [2]}],
            "outputs": [{"semantic": "echo.output", "dtype": "float32", "shape": [2]}],
            "semantic_identity": {
                "logical_model_revision": "dummy-echo-v1",
                "preprocessing_contract": "identity-float32-v1",
                "output_semantics": "identity-float32-v1",
            },
        },
        "deployments": {
            "cpu": {
                "uuid": "123e4567-e89b-42d3-a456-426614174001",
                "revision": 1,
                "execution_contract": {
                    "state_scope": "request",
                    "execution_structure": "direct",
                    "cancellation_granularity": "request_boundary",
                },
                "runtime_profile": {
                    "backend": "torch",
                    "target": {"runtime": "torch"},
                    "profile": {"device": "cpu"},
                },
            }
        },
    }
    (root / "inference_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _stop_process(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    assert process.stdout is not None
    return process.stdout.read()


def test_generic_model_service_process_serves_typed_echo_request(tmp_path: Path):
    bundle_path = _bundle(tmp_path / "echo")
    validated = load_inference_manifest(bundle_path, "cpu")
    endpoint = f"/test/model_echo_{uuid4().hex}"
    instance_id = f"echo-{uuid4().hex}"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "inference_service.model_service_node",
            "--ros-args",
            "-p",
            f"instance_id:={instance_id}",
            "-p",
            f"bundle_path:={bundle_path}",
            "-p",
            "deployment:=cpu",
            "-p",
            "adapter_class:=perception_service.echo_adapter:EchoServicePlugin",
            "-p",
            "service_type:=ibrobot_msgs/srv/EchoModel",
            "-p",
            f"service_endpoint:={endpoint}",
            "-p",
            "required:=true",
            "-p",
            "require_semantic_identity:=true",
            "-p",
            "configuration_generation:=42",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    rclpy.init()
    client_node = rclpy.create_node(f"model_service_test_client_{uuid4().hex}")
    logs = ""
    try:
        client = client_node.create_client(EchoModel, endpoint)
        deadline = time.monotonic() + 15.0
        while not client.wait_for_service(timeout_sec=0.2):
            if process.poll() is not None:
                logs = _stop_process(process)
                pytest.fail(f"model service exited before discovery (code={process.returncode}):\n{logs}")
            if time.monotonic() >= deadline:
                pytest.fail("timed out waiting for the generic model service")

        request = EchoModel.Request()
        request.request_id = "process-e2e"
        request.value = [1.25, -2.5]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(client_node, future, timeout_sec=10.0)
        response = future.result()

        assert response is not None
        assert response.success
        assert response.message == "echoed 2 values"
        assert response.value == pytest.approx([1.25, -2.5])
        assert response.inference_time_ms >= 0.0
        assert response.model.instance_id == instance_id
        assert response.model.model_name == "dummy-echo"
        assert response.model.model_version == "1"
        assert response.model.manifest_fingerprint == validated.manifest.bundle.digest.value
        assert response.model.deployment_name == "cpu"
        assert response.model.deployment_fingerprint == validated.fingerprint
        assert response.model.backend == "torch"
        assert response.model.runtime_version == "dummy-echo-v1"
        assert response.model.runtime_state == "ready"
        assert response.model.ready
        assert response.model.configuration_generation == 42
        identity = validated.manifest.model.semantic_identity
        assert identity is not None
        assert response.model.semantic_identity_json == canonical_semantic_identity_json(identity)
        assert response.model.semantic_identity_fingerprint == semantic_identity_fingerprint(identity)
    finally:
        client_node.destroy_node()
        rclpy.shutdown()
        if process.stdout is not None and not logs:
            logs = _stop_process(process)

    assert process.returncode == 0, logs
