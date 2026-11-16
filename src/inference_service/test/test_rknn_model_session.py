"""RKNN model-session execute() regression tests against a fake RKNNLite SDK."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_service.backends import RuntimeContext
from inference_service.backends.errors import BackendInferenceError
from inference_service.model_sessions import RKNNModelSession
from inference_service.unified_runtime import ExecutionContext, ModelRequest
from test.manifest_fixtures import TEST_BUNDLE_UUID, TEST_DEPLOYMENT_UUID


class _FakeRKNNLite:
    """Minimal RKNNLite stand-in: records calls and echoes a fixed output."""

    __version__ = "fake-rknnlite-1.0"
    NPU_CORE_ALL = 7
    instances: list[_FakeRKNNLite] = []

    def __init__(self) -> None:
        self.loaded: str | None = None
        self.init_kwargs: dict[str, object] = {}
        self.calls: list[list[np.ndarray]] = []
        self.released = False
        _FakeRKNNLite.instances.append(self)

    def load_rknn(self, path: str) -> int:
        self.loaded = path
        return 0

    def init_runtime(self, target=None, core_mask=None) -> int:
        self.init_kwargs = {"target": target, "core_mask": core_mask}
        return 0

    def inference(self, inputs, data_format=None):
        self.calls.append([np.asarray(value) for value in inputs])
        return [np.asarray([float(np.asarray(inputs[0]).sum())], dtype=np.float32)]

    def release(self) -> None:
        self.released = True


def _rknn_manifest() -> dict[str, Any]:
    return {
        "schema_version": 3,
        "bundle": {
            "uuid": TEST_BUNDLE_UUID,
            "revision": 1,
            "name": "test-rknn-silero",
            "files": [],
            "digest": {"algorithm": "sha256", "scope": "structure", "value": "0" * 64},
        },
        "model": {
            "interface": "tensor_model",
            "model_type": "silero_vad",
            "operation": "vad",
            "inputs": [{"semantic": "host.silero.audio", "dtype": "float32", "shape": [4]}],
            "outputs": [{"semantic": "host.silero.prob", "dtype": "float32", "shape": [1]}],
        },
        "deployments": {
            "rk3588": {
                "uuid": TEST_DEPLOYMENT_UUID,
                "revision": 1,
                "execution_contract": {
                    "state_scope": "request",
                    "execution_structure": "direct",
                    "cancellation_granularity": "request_boundary",
                },
                "runtime_profile": {
                    "backend": "rknn",
                    "target": {"soc": "rk3588", "runtime": "rknn-lite"},
                    "profile": {"device_id": 0, "target_name": "rk3588", "core_mask": 7},
                },
                "artifacts": {"model": {"path": "artifacts/model.rknn", "format": "rknn"}},
                "execution": ["model"],
                "bindings": {
                    "model": {
                        "inputs": [
                            {
                                "semantic": "host.silero.audio",
                                "runtime_name": "input",
                                "index": 0,
                                "dtype": "float32",
                                "shape": [4],
                            }
                        ],
                        "outputs": [
                            {
                                "semantic": "host.silero.prob",
                                "runtime_name": "prob",
                                "index": 0,
                                "dtype": "float32",
                                "shape": [1],
                            }
                        ],
                    }
                },
            }
        },
    }


def _write_rknn_bundle(root: Path) -> None:
    manifest = _rknn_manifest()
    artifact = root / "artifacts" / "model.rknn"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"fake-rknn-model")
    asset = root / "assets" / "adapter.json"
    asset.parent.mkdir(parents=True, exist_ok=True)
    asset.write_text(json.dumps({"backend": "rknn"}), encoding="utf-8")
    entries = [BundleFile(path="assets/adapter.json")]
    manifest["bundle"]["files"] = [entry.model_dump(mode="json") for entry in entries]
    manifest["bundle"]["digest"]["value"] = canonical_bundle_digest(
        TEST_BUNDLE_UUID, 1, manifest["bundle"]["name"], entries
    )
    (root / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


@pytest.fixture
def loaded_session(tmp_path: Path) -> RKNNModelSession:
    _FakeRKNNLite.instances.clear()
    _write_rknn_bundle(tmp_path)
    context = RuntimeContext(load_inference_manifest(str(tmp_path), "rk3588"))
    session = RKNNModelSession(rknn_loader=lambda: _FakeRKNNLite)
    session.load(context)
    return session


def test_rknn_session_execute_runs_manifest_role(loaded_session: RKNNModelSession) -> None:
    audio = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    outputs = loaded_session.execute(
        ModelRequest(inputs={"host.silero.audio": audio}),
        ExecutionContext("rknn-execute-1"),
    )

    assert set(outputs) == {"host.silero.prob"}
    np.testing.assert_allclose(outputs["host.silero.prob"], np.asarray([10.0], dtype=np.float32))
    (runtime,) = _FakeRKNNLite.instances
    assert runtime.loaded is not None and runtime.loaded.endswith("artifacts/model.rknn")
    assert runtime.init_kwargs == {"target": "rk3588", "core_mask": 7}
    assert len(runtime.calls) == 1
    loaded_session.close()
    assert runtime.released


def test_rknn_session_execute_role_matches_execute(loaded_session: RKNNModelSession) -> None:
    audio = np.asarray([0.5, 0.5, 1.0, 2.0], dtype=np.float32)
    request = ModelRequest(inputs={"host.silero.audio": audio})

    via_role = loaded_session.execute_role("model", request.inputs, request, ExecutionContext("rknn-role-1"))
    via_execute = loaded_session.execute(request, ExecutionContext("rknn-execute-2"))

    np.testing.assert_allclose(via_role["host.silero.prob"], via_execute["host.silero.prob"])
    loaded_session.close()


def test_rknn_session_execute_rejects_unknown_role(loaded_session: RKNNModelSession) -> None:
    request = ModelRequest(inputs={"host.silero.audio": np.zeros(4, dtype=np.float32)})

    with pytest.raises(BackendInferenceError) as excinfo:
        loaded_session.execute_role("missing", request.inputs, request, ExecutionContext("rknn-role-2"))

    assert excinfo.value.code == "unknown_execution_role"
    loaded_session.close()
