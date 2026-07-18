from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch

from inference_service.backends import (
    BackendCapabilities,
    BackendDescriptor,
    BackendRegistry,
    BackendResult,
    InferenceRequest,
    LifecycleBackend,
    PartialLoadRollback,
    ResourceDomainAdmissions,
    RuntimeContext,
)
from inference_service.core.pure_inference_engine import PureInferenceEngine
from inference_service.pipeline_policy_node import PipelinePolicyNode
from tests.manifest_fixtures import create_policy_bundle, make_manifest, write_manifest


class _FacadeBackend(LifecycleBackend):
    def __init__(self) -> None:
        super().__init__("torch", BackendCapabilities(), domains=ResourceDomainAdmissions())
        self.requests: list[InferenceRequest] = []
        self.close_calls = 0

    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        self.context = context

    def _infer(self, request: InferenceRequest) -> BackendResult:
        self.requests.append(request)
        return BackendResult(
            action=np.full((2, 6), 3.0, dtype=np.float32),
            actual_chunk_size=2,
            backend_latency_ms=0.1,
        )

    def _close(self) -> None:
        self.close_calls += 1


def test_rad_to_lerobot_preserves_float32_without_joint_conversion():
    node = SimpleNamespace(_joint_rad_limits=[])

    converted = PipelinePolicyNode._rad_to_lerobot(node, np.array([1.0, 2.0], dtype=np.float64))

    assert converted.dtype == np.float32
    assert converted.flags.c_contiguous


def test_rad_to_lerobot_returns_float32_after_joint_conversion():
    node = SimpleNamespace(_joint_rad_limits=[(-1.0, 1.0, 200.0, -100.0)])

    converted = PipelinePolicyNode._rad_to_lerobot(node, np.array([0.0], dtype=np.float32))

    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, np.array([0.0], dtype=np.float32))


def test_to_policy_inputs_converts_numpy_observations_to_contiguous_tensors():
    image = np.zeros((2, 3, 4), dtype=np.float32).transpose(1, 0, 2)

    converted = PipelinePolicyNode._to_policy_inputs(
        {"observation.images.top": image, "observation.state": np.arange(6, dtype=np.float32)}
    )

    assert isinstance(converted["observation.images.top"], torch.Tensor)
    assert converted["observation.images.top"].shape == (3, 2, 4)
    assert converted["observation.images.top"].is_contiguous()
    assert converted["observation.state"].dtype == torch.float32


def _bundle(root: Path) -> Path:
    root.mkdir()
    paths = create_policy_bundle(root)
    config_path = root / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"chunk_size": 2, "max_action_dim": 8})
    config_path.write_text(json.dumps(config), encoding="utf-8")
    write_manifest(root, make_manifest(root, paths))
    return root


def _registry(monkeypatch, created: list[_FacadeBackend]) -> BackendRegistry:
    module = ModuleType("tests.pure_facade_backend")

    def create_backend(_context: RuntimeContext) -> _FacadeBackend:
        backend = _FacadeBackend()
        created.append(backend)
        return backend

    module.create_backend = create_backend
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return BackendRegistry(
        {
            "torch": BackendDescriptor(
                name="torch",
                factory="tests.pure_facade_backend:create_backend",
                supported_policy_families=frozenset({"act"}),
                target_validator=lambda deployment: None,
            )
        }
    )


def test_pure_engine_uses_validated_manifest_registry_pipeline_and_clean_shutdown(monkeypatch, tmp_path):
    created: list[_FacadeBackend] = []
    engine = PureInferenceEngine(
        _bundle(tmp_path / "bundle"),
        "cpu",
        pipeline_id="smoke",
        runtime_options={"trace": True},
        registry=_registry(monkeypatch, created),
    )

    noise = np.ones((1, 2, 8), dtype=np.float32)
    result = engine(
        {"observation.state": np.zeros((1, 6), dtype=np.float32)},
        prompt="pick banana",
        control_inputs={"noise": noise},
        capture_raw_action=True,
    )

    assert result.shape == (2, 6)
    assert result.chunk_size == 2
    assert engine.chunk_size == 2
    assert result.policy_type == "act"
    assert result.backend_type == "torch"
    np.testing.assert_array_equal(result.raw_action, result.action)
    assert created[0].requests[0].prompt == "pick banana"
    np.testing.assert_array_equal(created[0].requests[0].inputs["noise"], noise)
    assert created[0].requests[0].metadata["pipeline_id"] == "smoke"
    assert created[0].context.runtime_options == {"trace": True}
    assert engine.capabilities == created[0].capabilities
    assert engine.policy_metadata.policy_type == "act"
    assert engine.nominal_chunk_size == 2
    assert engine.max_action_dimension == 8
    engine.close()
    engine.close()
    assert created[0].close_calls == 1


def test_pure_engine_contains_no_backend_identity_dispatch():
    source = inspect.getsource(PureInferenceEngine)

    for backend_name in ("ascend", "hisilicon", "rknn", "hmm"):
        assert backend_name not in source
