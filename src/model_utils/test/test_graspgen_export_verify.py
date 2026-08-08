# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""The ONNX numerical comparison is an export gate, not a printed statistic."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from model_utils.graspgen_export.export_onnx import (
    VERIFY_COSINE_DEFICIT,
    VERIFY_MAX_RELATIVE,
    VERIFY_MEAN_RELATIVE,
    ExportArtifact,
    _export_artifact,
    _output_metrics,
    _tolerance_violations,
    _verify_onnx,
)


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _artifact() -> ExportArtifact:
    torch.manual_seed(20260807)
    return ExportArtifact("tiny", _Linear().eval(), (torch.randn(3, 8),), ["x"], ["y"])


def _export(tmp_path: Path, artifact: ExportArtifact) -> Path:
    _export_artifact(artifact, tmp_path, opset=14, verify=False, constant_folding=True, simplify=False)
    return tmp_path / f"{artifact.name}.onnx"


def _break(artifact: ExportArtifact, delta: float) -> None:
    """Move the PyTorch weights after the graph was exported, so the two disagree."""
    with torch.no_grad():
        artifact.model.linear.weight.add_(delta)


def test_output_metrics_report_both_absolute_and_relative_error():
    reference = np.array([[2.0, -4.0]], dtype=np.float32)
    candidate = np.array([[2.0, -4.4]], dtype=np.float32)

    metrics = _output_metrics(reference, candidate)

    assert metrics["max_abs"] == pytest.approx(0.4)
    assert metrics["mean_abs"] == pytest.approx(0.2)
    assert metrics["reference_scale"] == pytest.approx(4.0)
    assert metrics["max_relative"] == pytest.approx(0.1)
    assert metrics["mean_relative"] == pytest.approx(0.05)
    assert metrics["cosine"] < 1.0


def test_output_metrics_keep_an_all_zero_reference_comparable():
    """A zero reference has no scale; the gate must still see the error, not a NaN."""
    metrics = _output_metrics(np.zeros((2, 2), dtype=np.float32), np.full((2, 2), 0.5, dtype=np.float32))

    assert metrics["reference_scale"] == pytest.approx(1.0)
    assert metrics["max_relative"] == pytest.approx(0.5)
    assert _tolerance_violations("y", metrics, 1.0)


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({"max_relative": VERIFY_MAX_RELATIVE * 10}, "max_relative"),
        ({"mean_relative": VERIFY_MEAN_RELATIVE * 10}, "mean_relative"),
        ({"cosine": 1.0 - VERIFY_COSINE_DEFICIT * 10}, "cosine"),
    ],
)
def test_tolerance_violations_flag_each_limit_independently(metrics, expected):
    baseline = {"max_relative": 0.0, "mean_relative": 0.0, "cosine": 1.0}

    violations = _tolerance_violations("y", {**baseline, **metrics}, 1.0)

    assert len(violations) == 1
    assert violations[0].startswith(f"y {expected}=")


def test_tolerance_violations_accept_a_faithful_export():
    assert _tolerance_violations("y", {"max_relative": 0.0, "mean_relative": 0.0, "cosine": 1.0}, 1.0) == []


def test_verify_onnx_returns_metrics_for_a_faithful_export(tmp_path):
    artifact = _artifact()
    path = _export(tmp_path, artifact)

    metrics = _verify_onnx(path, artifact)

    assert set(metrics) == {"y"}
    assert metrics["y"]["cosine"] >= 1.0 - VERIFY_COSINE_DEFICIT
    assert metrics["y"]["max_relative"] <= VERIFY_MAX_RELATIVE


def test_verify_onnx_raises_when_the_graph_no_longer_matches_pytorch(tmp_path):
    artifact = _artifact()
    path = _export(tmp_path, artifact)
    _break(artifact, 0.05)

    with pytest.raises(RuntimeError, match="ONNX verification failed for tiny"):
        _verify_onnx(path, artifact)


def test_verify_onnx_tolerance_scale_can_widen_the_envelope(tmp_path):
    """The scale is the documented re-baselining knob, so it has to actually apply."""
    artifact = _artifact()
    path = _export(tmp_path, artifact)
    _break(artifact, 0.05)

    with pytest.raises(RuntimeError):
        _verify_onnx(path, artifact, tolerance_scale=1.0)
    assert _verify_onnx(path, artifact, tolerance_scale=1e9)["y"]["max_abs"] > 0.0


def test_verify_onnx_refuses_to_silently_skip_without_onnxruntime(tmp_path, monkeypatch):
    """A missing verifier must fail the export instead of publishing an unchecked graph."""
    artifact = _artifact()
    path = _export(tmp_path, artifact)

    def _import_module(name: str):
        raise ImportError(name)

    monkeypatch.setattr(
        "model_utils.graspgen_export.export_onnx.importlib",
        types.SimpleNamespace(import_module=_import_module),
    )

    with pytest.raises(RuntimeError, match="onnxruntime is required"):
        _verify_onnx(path, artifact)


def test_export_artifact_publishes_the_measured_metrics(tmp_path):
    record = _export_artifact(_artifact(), tmp_path, opset=14, verify=True, constant_folding=True, simplify=False)

    assert record["onnx"] == "tiny.onnx"
    assert set(record["verification"]) == {"y"}
    assert record["verification"]["y"]["cosine"] >= 1.0 - VERIFY_COSINE_DEFICIT


def test_export_artifact_records_nothing_when_verification_is_skipped(tmp_path):
    record = _export_artifact(_artifact(), tmp_path, opset=14, verify=False, constant_folding=True, simplify=False)

    assert "verification" not in record


def test_export_artifact_deletes_a_graph_that_fails_the_gate(tmp_path, monkeypatch):
    """A rejected graph must not stay where onnx2om could compile it into an OM."""

    def _reject(path, artifact, tolerance_scale=1.0):
        # The graph exists on disk at this point; the gate rejecting it is what the
        # caller has to react to, whatever made the numbers drift.
        assert path.is_file()
        raise RuntimeError(f"ONNX verification failed for {artifact.name}: y cosine=0.5 below 1.0")

    monkeypatch.setattr("model_utils.graspgen_export.export_onnx._verify_onnx", _reject)

    with pytest.raises(RuntimeError, match="ONNX verification failed"):
        _export_artifact(_artifact(), tmp_path, opset=14, verify=True, constant_folding=True, simplify=False)

    assert not (tmp_path / "tiny.onnx").exists()
