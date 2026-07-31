"""Tests for stage-specific backend conformance evaluation."""

import numpy as np
import pytest

from perception_service.conformance import (
    ConformanceThresholds,
    embedding_cosine,
    evaluate_backend_conformance,
    mask_iou,
)


def _inputs() -> dict:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    return {
        "reference_masks": [mask],
        "candidate_masks": [mask.copy()],
        "reference_embedding": np.asarray([0.6, 0.8], dtype=np.float32),
        "candidate_embedding": np.asarray([0.6, 0.8], dtype=np.float32),
        "reference_label": "cup",
        "candidate_label": "cup",
        "reference_label_score": 0.83,
        "candidate_label_score": 0.82,
        "reference_centroid": np.asarray([1.0, 2.0, 3.0]),
        "candidate_centroid": np.asarray([1.001, 2.0, 3.0]),
        "reference_extent": np.asarray([0.1, 0.2, 0.3]),
        "candidate_extent": np.asarray([0.1, 0.201, 0.3]),
        "reference_failure": (False, ""),
        "candidate_failure": (False, ""),
    }


def test_identical_observable_stages_pass_conformance() -> None:
    report = evaluate_backend_conformance(**_inputs())

    assert report.passed
    assert report.metrics["minimum_mask_iou"] == 1.0
    assert report.metrics["embedding_cosine"] == pytest.approx(1.0)
    assert report.metrics["label_top1_match"] is True
    assert report.failures == ()


def test_stage_failures_are_reported_separately() -> None:
    inputs = _inputs()
    inputs["candidate_masks"] = [np.ones((8, 8), dtype=np.uint8)]
    inputs["candidate_embedding"] = np.asarray([1.0, 0.0], dtype=np.float32)
    inputs["candidate_label"] = "table"
    inputs["candidate_label_score"] = 0.5
    inputs["candidate_centroid"] = np.asarray([1.1, 2.0, 3.0])
    inputs["candidate_extent"] = np.asarray([0.2, 0.2, 0.3])
    inputs["candidate_failure"] = (True, "not ready")

    report = evaluate_backend_conformance(**inputs)

    assert not report.passed
    assert set(report.failures) == {
        "mask IoU is below threshold",
        "embedding cosine is below threshold",
        "label top-1 differs",
        "label score delta exceeds threshold",
        "centroid error exceeds threshold",
        "extent error exceeds threshold",
        "failure semantics differ",
    }


def test_thresholds_are_explicit_and_mask_count_is_bounded() -> None:
    inputs = _inputs()
    inputs["candidate_masks"] = []

    report = evaluate_backend_conformance(
        **inputs,
        thresholds=ConformanceThresholds(maximum_mask_count_delta=0),
    )

    assert not report.passed
    assert report.metrics["mask_count_delta"] == 1
    assert "mask count delta exceeds threshold" in report.failures


def test_metric_helpers_reject_incompatible_tensors() -> None:
    with pytest.raises(ValueError, match="identical dimensions"):
        mask_iou(np.ones((2, 2)), np.ones((3, 3)))
    with pytest.raises(ValueError, match="identical dimensions"):
        embedding_cosine(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="non-zero norm"):
        embedding_cosine(np.zeros(2), np.ones(2))
