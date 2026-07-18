from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from inference_manifest import load_inference_manifest
from inference_service.backends import RuntimeContext
from inference_service.pipeline.processors import create_lerobot_processor_views
from tests.manifest_fixtures import create_policy_bundle, make_manifest, write_manifest


def test_local_tokenizer_reference_is_resolved_against_bundle(monkeypatch, tmp_path: Path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    paths = create_policy_bundle(bundle, "smolvla", local_tokenizer=True, include_weights=False)
    write_manifest(bundle, make_manifest(bundle, paths, deployment_name="rk3588", compiled=True))
    context = RuntimeContext(load_inference_manifest(bundle, "rk3588"))
    captured: dict[str, object] = {}

    class FakeConfig:
        device = "cuda"

        @classmethod
        def from_pretrained(cls, path, local_files_only):
            captured["config_path"] = path
            captured["local_files_only"] = local_files_only
            return cls()

    policies = ModuleType("lerobot.configs.policies")
    policies.PreTrainedConfig = FakeConfig
    factory = ModuleType("lerobot.policies.factory")

    def resolve_policy_config(policy_type):
        captured["resolved_policy_type"] = policy_type

    def make_processors(**kwargs):
        captured.update(kwargs)
        return (lambda inputs: inputs), (lambda action: action)

    factory.get_policy_config_class = resolve_policy_config
    factory.make_pre_post_processors = make_processors
    monkeypatch.setitem(sys.modules, policies.__name__, policies)
    monkeypatch.setitem(sys.modules, factory.__name__, factory)

    preprocessor, postprocessor = create_lerobot_processor_views()
    preprocessor.load(context)

    assert captured["config_path"] == str(bundle.resolve())
    assert captured["local_files_only"] is True
    assert captured["resolved_policy_type"] == "smolvla"
    assert captured["preprocessor_overrides"] == {
        "device_processor": {"device": "cpu"},
        "tokenizer_processor": {"tokenizer_name": str((bundle / "tokenizer").resolve())},
    }
    assert captured["postprocessor_overrides"] == {"device_processor": {"device": "cpu"}}
    assert preprocessor({"observation.state": 1}) == {"observation.state": 1}
    action = postprocessor(np.ones((1, 2), dtype=np.float32))
    assert isinstance(action, torch.Tensor)
    assert torch.equal(action, torch.ones((1, 2)))
