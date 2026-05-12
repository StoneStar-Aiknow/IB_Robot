#!/usr/bin/env python3
"""Tests for Ascend OM inference_service integration."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from inference_service.core import (
    AscendOM3403PolicyWrapper,
    AscendOMPolicyWrapper,
    PureInferenceEngine,
    create_ascend_om_policy_wrapper,
    resolve_3403_worker_path,
    resolve_device,
    resolve_om_model_path,
)


def test_resolve_device_accepts_ascend_om_aliases():
    assert resolve_device("ascend_om").type == "cpu"
    assert resolve_device("ascend_om_3403").type == "cpu"
    assert resolve_device("ascend-om").type == "cpu"


def test_create_ascend_wrapper_by_device_name():
    assert isinstance(create_ascend_om_policy_wrapper("ascend_om"), AscendOMPolicyWrapper)
    assert isinstance(create_ascend_om_policy_wrapper("ascend-om-3403"), AscendOM3403PolicyWrapper)


def test_resolve_om_model_path_from_policy_config(tmp_path):
    model_path = tmp_path / "model" / "legacy.om"
    model_path.parent.mkdir()
    model_path.write_bytes(b"om")
    (tmp_path / "config.json").write_text(
        json.dumps({"om_model_path": "model/legacy.om"}),
        encoding="utf-8",
    )

    assert resolve_om_model_path(str(tmp_path)) == model_path.resolve()


def test_resolve_om_model_path_from_env(tmp_path, monkeypatch):
    model_path = tmp_path / "env.om"
    model_path.write_bytes(b"om")
    monkeypatch.setenv("ASCEND_OM_MODEL_PATH", str(model_path))

    assert resolve_om_model_path(str(tmp_path)) == model_path.resolve()


def test_resolve_3403_worker_path_from_env(tmp_path, monkeypatch):
    model_path = tmp_path / "model.om"
    model_path.write_bytes(b"om")
    worker = tmp_path / "main"
    worker.write_text("#!/bin/sh\n", encoding="utf-8")
    worker.chmod(0o755)
    monkeypatch.setenv("SVP_WORKER_EXECUTABLE", str(worker))

    assert resolve_3403_worker_path(str(tmp_path), model_path, {}) == worker.resolve()


def test_pure_engine_selects_ascend_wrapper(monkeypatch, tmp_path):
    model_path = tmp_path / "model.om"
    model_path.write_bytes(b"om")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "om_model_path": str(model_path),
                "chunk_size": 2,
                "input_features": {"observation.state": {"shape": [3]}},
            }
        ),
        encoding="utf-8",
    )

    class FakeACTWrapper:
        def __init__(self, path, config):
            self.path = path
            self.config = config

        def predict(self, batch):
            assert batch
            assert self.config.chunk_size == 2
            return (torch.arange(12, dtype=torch.float32).reshape(1, 2, 6),)

    monkeypatch.setattr(
        "inference_service.core.ascend_om.policy_wrapper.ACTWrapper",
        FakeACTWrapper,
    )

    engine = PureInferenceEngine(policy_path=str(tmp_path), device="ascend_om")
    result = engine({"observation.state": torch.ones(1, 3)})

    assert result.policy_type == "ascend_om"
    assert result.action.shape == (2, 6)


def test_pure_engine_selects_ascend_3403_wrapper(monkeypatch, tmp_path):
    model_path = tmp_path / "model.om"
    model_path.write_bytes(b"om")
    worker_path = tmp_path / "main"
    worker_path.write_text("#!/bin/sh\n", encoding="utf-8")
    worker_path.chmod(0o755)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "om_model_path": str(model_path),
                "cpp_executable": str(worker_path),
                "chunk_size": 2,
            }
        ),
        encoding="utf-8",
    )

    class FakeACT3403Policy:
        def __init__(self, worker, model):
            self.worker = worker
            self.model = model

        def predict(self, batch):
            assert batch
            return (torch.arange(12, dtype=torch.float32).reshape(1, 2, 6),)

        def close(self):
            pass

    monkeypatch.setattr(
        "inference_service.core.ascend_om.policy_wrapper.ACT3403Policy",
        FakeACT3403Policy,
    )

    engine = PureInferenceEngine(policy_path=str(tmp_path), device="ascend_om_3403")
    result = engine({"observation.state": torch.ones(1, 3)})

    assert result.policy_type == "ascend_om_3403"
    assert result.action.shape == (2, 6)
