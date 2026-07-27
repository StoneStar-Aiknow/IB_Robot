from __future__ import annotations

import torch

from model_utils import pi05_dist_metrics
from model_utils.pi05_dist_metrics import evaluate_pi05


def test_evaluate_pi05_returns_structured_metrics_and_keeps_console_report(capsys):
    values = [torch.tensor([[1.0, 0.0], [0.5, 0.5]])]

    metrics = evaluate_pi05(values, values, raw_preds=values, raw_targets=values)
    output = capsys.readouterr().out

    assert metrics["normalized"]["first_frame"]["mean_cos"] == 1.0
    assert metrics["unnormalized"]["first_frame"]["mean_cos"] == 1.0
    assert "PI05 distributional evaluation" in output
    assert "Method C: first-frame cosine (normalized)" in output


def test_wasserstein_metrics_have_numpy_fallback_without_scipy(monkeypatch):
    monkeypatch.setattr(pi05_dist_metrics, "_HAS_SCIPY", False)
    predictions = [torch.tensor([[1.0], [3.0]])]
    targets = [torch.tensor([[0.0], [2.0]])]

    metrics = pi05_dist_metrics.wasserstein_per_dim(predictions, targets)

    assert metrics["mean_w1"] == 1.0
    assert metrics["mean_ratio"] == 1.0
