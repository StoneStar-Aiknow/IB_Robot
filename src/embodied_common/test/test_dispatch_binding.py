from __future__ import annotations

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


def test_manifest_identity_loader_uses_user_supplied_bundle(monkeypatch, tmp_path):
    validated = SimpleNamespace(
        deployment_name="torch_cuda",
        fingerprint="a" * 64,
        manifest=SimpleNamespace(bundle=SimpleNamespace(digest=SimpleNamespace(value="b" * 64))),
    )
    monkeypatch.setattr("inference_manifest.load_inference_manifest", lambda *_args: validated)
    bundle = tmp_path / "grasp"
    bundle.mkdir()
    monkeypatch.setenv("WORKSPACE", str(tmp_path))

    identity = load_delegated_model_identity(
        {"model_bundle_path": "$(env WORKSPACE)/grasp", "model_deployment": "torch_cuda"}
    )

    assert identity["model_fingerprint"] == validated.fingerprint
