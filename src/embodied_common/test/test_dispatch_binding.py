from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from embodied_common.dispatch_binding import delegated_executor_identity, load_delegated_model_identity


def test_delegated_executor_identity_requires_complete_model_identity():
    with pytest.raises(ValueError, match="all present"):
        delegated_executor_identity(
            name="grasp_pipeline",
            endpoint_name="/pick",
            model_deployment_name="torch_cuda",
        )


def test_load_delegated_model_identity_uses_strict_manifest(monkeypatch):
    validated = SimpleNamespace(
        deployment_name="torch_cuda",
        fingerprint="a" * 64,
        manifest=SimpleNamespace(bundle=SimpleNamespace(digest=SimpleNamespace(value="b" * 64))),
    )
    monkeypatch.setattr("inference_manifest.load_inference_manifest", lambda *_args: validated)

    identity = load_delegated_model_identity({"model_bundle_path": "models/grasp", "model_deployment": "torch_cuda"})

    assert identity == {
        "model_deployment_name": "torch_cuda",
        "model_fingerprint": "a" * 64,
        "model_bundle_digest": "b" * 64,
    }


def test_checked_in_grasp_manifest_is_valid(monkeypatch):
    from inference_manifest import load_inference_manifest

    validated = load_inference_manifest("models/grasp", "torch_cuda")

    assert validated.manifest.model.family == "graspgen"
    assert len(validated.fingerprint) == 64
    assert json.loads((validated.bundle_root / "model-source.json").read_text(encoding="utf-8"))["repository"]

    monkeypatch.setenv("WORKSPACE", str(validated.bundle_root.parents[1]))
    identity = load_delegated_model_identity(
        {"model_bundle_path": "$(env WORKSPACE)/models/grasp", "model_deployment": "torch_cuda"}
    )
    assert identity["model_fingerprint"] == validated.fingerprint
