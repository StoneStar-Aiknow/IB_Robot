from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from inference_manifest import BundleFile, canonical_bundle_digest, load_inference_manifest
from inference_service.backends import InferenceRequest
from inference_service.pipeline import create_pipeline_manager
from inference_service.runtime_composition import build_policy_runtime_dependencies
from robot_config.inference_config import parse_inference_config
from tests.manifest_fixtures import TEST_BUNDLE_UUID, TEST_DEPLOYMENT_UUID

MODEL_NAME = "ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515"


def _workspace() -> Path:
    return Path(__file__).resolve().parents[3]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _create_tracked_equivalent_bundle(root: Path) -> Path:
    fixture = Path(__file__).parent / "assets" / MODEL_NAME
    root.mkdir()
    paths = (
        "config.json",
        "model.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "policy_preprocessor.json",
        "policy_preprocessor_step_3_normalizer_processor.safetensors",
    )
    for path in paths:
        (root / path).write_bytes((fixture / path).read_bytes())
    entries = [BundleFile(path=path) for path in paths]
    _write_json(
        root / "inference_manifest.json",
        {
            "schema_version": 3,
            "bundle": {
                "uuid": TEST_BUNDLE_UUID,
                "revision": 1,
                "name": MODEL_NAME,
                "files": [entry.model_dump(mode="json") for entry in entries],
                "digest": {
                    "algorithm": "sha256",
                    "scope": "structure",
                    "value": canonical_bundle_digest(TEST_BUNDLE_UUID, 1, MODEL_NAME, entries),
                },
            },
            "model": {
                "interface": "policy",
                "model_type": "act",
                "operation": "predict",
                "inputs": [
                    {"semantic": "observation.state", "dtype": "float32", "shape": [6]},
                    {"semantic": "observation.images.top", "dtype": "float32", "shape": [3, 16, 24]},
                ],
                "outputs": [{"semantic": "action", "dtype": "float32", "shape": [6]}],
            },
            "deployments": {
                "cpu": {
                    "uuid": TEST_DEPLOYMENT_UUID,
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
        },
    )
    return root


def _install_fake_lerobot(monkeypatch, torch_module, calls):
    lerobot_module = ModuleType("lerobot")
    configs_module = ModuleType("lerobot.configs")
    config_module = ModuleType("lerobot.configs.policies")
    policies_module = ModuleType("lerobot.policies")
    factory_module = ModuleType("lerobot.policies.factory")

    class FakeConfig:
        def __init__(self, policy_type: str, device: str) -> None:
            self.type = policy_type
            self.device = device

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            raw = json.loads((Path(path) / "config.json").read_text(encoding="utf-8"))
            calls["metadata_path"] = Path(path)
            return cls(raw["type"], raw["device"])

    class FakePolicy:
        supports_attention = False

        def __init__(self, config) -> None:
            self.config = config
            self.model = SimpleNamespace(supports_attention=False)
            self.closed = False

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["policy_path"] = Path(path)
            calls["policy_kwargs"] = kwargs
            return cls(kwargs["config"])

        def to(self, device):
            calls["device"] = str(device)
            return self

        def eval(self):
            return self

        def predict_action_chunk(self, batch):
            calls["batch"] = dict(batch)
            return torch_module.zeros((1, 4, 6), dtype=torch_module.float32)

        def reset(self):
            calls["reset"] = calls.get("reset", 0) + 1

    def get_policy_class(policy_type):
        assert policy_type == "act"
        return FakePolicy

    def make_pre_post_processors(**kwargs):
        calls["processor_path"] = Path(kwargs["pretrained_path"])
        return lambda batch: batch, lambda action: action

    config_module.PreTrainedConfig = FakeConfig
    factory_module.get_policy_class = get_policy_class
    factory_module.make_pre_post_processors = make_pre_post_processors
    monkeypatch.setitem(sys.modules, "lerobot", lerobot_module)
    monkeypatch.setitem(sys.modules, "lerobot.configs", configs_module)
    monkeypatch.setitem(sys.modules, "lerobot.configs.policies", config_module)
    monkeypatch.setitem(sys.modules, "lerobot.policies", policies_module)
    monkeypatch.setitem(sys.modules, "lerobot.policies.factory", factory_module)


def test_named_ignored_bundle_manifest_validates() -> None:
    bundle = _workspace() / "models" / MODEL_NAME
    if not bundle.is_dir():
        pytest.skip(f"local smoke bundle is unavailable: {bundle}")

    validated = load_inference_manifest(bundle, "cpu")

    assert validated.manifest.bundle.name == MODEL_NAME
    assert validated.policy.policy_type == "act"
    assert validated.deployment.backend == "torch"


def test_tracked_equivalent_cpu_bundle_runs_unified_registry_pipeline_end_to_end(monkeypatch, tmp_path):
    torch = pytest.importorskip("torch")
    bundle = _create_tracked_equivalent_bundle(tmp_path / MODEL_NAME)
    calls = {}
    _install_fake_lerobot(monkeypatch, torch, calls)
    inference = parse_inference_config(
        {
            "control_modes": {
                "model_inference": {
                    "inference": {
                        "enabled": True,
                        "pipelines": {
                            "cpu_smoke": {
                                "model_path": str(bundle),
                                "deployment": "cpu",
                                "execution_mode": "monolithic",
                                "request_timeout": 2.0,
                            }
                        },
                    }
                }
            }
        },
        "model_inference",
    )
    pipeline = inference.pipelines["cpu_smoke"]
    validated = pipeline.validated_manifest
    dependencies = build_policy_runtime_dependencies()
    manager = create_pipeline_manager(
        "cpu_smoke",
        validated,
        request_timeout=2.0,
        registry_set=dependencies.registry_set,
        providers=dependencies.providers,
    )

    result = manager.infer(
        "cpu_smoke",
        InferenceRequest(
            request_id="smoke-1",
            inputs={
                "observation.state": np.zeros(6, dtype=np.float32),
                "observation.images.top": np.zeros((480, 640, 3), dtype=np.uint8),
                "observation.images.wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            },
        ),
    )

    assert result.pipeline_id == "cpu_smoke"
    assert result.bundle == MODEL_NAME
    assert result.backend == "torch"
    assert result.actual_chunk_size == 4
    assert tuple(result.action.shape) == (4, 6)
    assert calls["metadata_path"] == bundle
    assert calls["policy_path"] == bundle
    assert calls["processor_path"] == bundle
    assert calls["device"] == "cpu"
    manager.reset("cpu_smoke")
    manager.close()
    manager.close()
    dependencies.providers.close()
