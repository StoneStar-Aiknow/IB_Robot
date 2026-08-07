"""Tests for stage-specific backend conformance evaluation."""

import numpy as np
import pytest

from perception_service.conformance import (
    ConformanceThresholds,
    embedding_cosine,
    evaluate_backend_conformance,
    evaluate_grasp_conformance,
    grasp_pose_error,
    mask_iou,
)


def _pose(translation, rotation=None) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    if rotation is not None:
        pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def _rotation_z(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    return np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _grasps() -> dict:
    poses = [_pose([0.1, 0.2, 0.3]), _pose([0.4, 0.5, 0.6], _rotation_z(30.0))]
    return {
        "reference_poses": poses,
        "candidate_poses": [pose.copy() for pose in poses],
        "reference_confidences": [0.91, 0.72],
        "candidate_confidences": [0.90, 0.73],
    }


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


def test_two_backends_that_agree_on_every_grasp_pass_conformance() -> None:
    report = evaluate_grasp_conformance(**_grasps())

    assert report.passed
    assert report.metrics["grasp_translation_error_m"] == pytest.approx(0.0)
    assert report.metrics["grasp_rotation_error_deg"] == pytest.approx(0.0)
    assert report.metrics["grasp_confidence_delta"] == pytest.approx(0.01)
    assert report.failures == ()


def test_grasps_are_compared_rank_by_rank_rather_than_as_a_set() -> None:
    """The executor only ever tries the first few, so a reordered list is not equivalent."""
    inputs = _grasps()
    inputs["candidate_poses"] = list(reversed(inputs["candidate_poses"]))
    inputs["candidate_confidences"] = list(reversed(inputs["candidate_confidences"]))

    report = evaluate_grasp_conformance(**inputs)

    assert not report.passed
    assert set(report.failures) == {
        "grasp translation error exceeds threshold",
        "grasp rotation error exceeds threshold",
        "grasp confidence delta exceeds threshold",
    }


def test_each_grasp_dimension_fails_on_its_own_threshold() -> None:
    inputs = _grasps()
    inputs["candidate_poses"] = [_pose([0.1, 0.2, 0.3], _rotation_z(20.0)), _pose([0.4, 0.5, 0.6], _rotation_z(30.0))]

    report = evaluate_grasp_conformance(**inputs)

    assert report.metrics["grasp_translation_error_m"] == pytest.approx(0.0)
    assert report.metrics["grasp_rotation_error_deg"] == pytest.approx(20.0)
    assert report.failures == ("grasp rotation error exceeds threshold",)


def test_a_backend_that_returns_fewer_grasps_is_not_conformant() -> None:
    inputs = _grasps()
    inputs["candidate_poses"] = inputs["candidate_poses"][:1]
    inputs["candidate_confidences"] = inputs["candidate_confidences"][:1]

    report = evaluate_grasp_conformance(**inputs)

    assert not report.passed
    assert report.metrics["grasp_count_delta"] == 1
    assert report.failures == ("grasp count delta exceeds threshold",)


def test_grasp_rotation_error_is_geodesic_so_it_survives_the_axis_angle_round_trip() -> None:
    """``matrix_to_rt``/``rt_to_matrix`` change the representation, not the rotation."""
    translation, rotation = grasp_pose_error(_pose([0.0, 0.0, 0.0]), _pose([0.003, 0.0, 0.0], _rotation_z(-180.0)))

    assert translation == pytest.approx(0.003)
    assert rotation == pytest.approx(180.0)


def test_comparing_two_empty_grasp_sets_is_conformant_rather_than_an_error() -> None:
    """A cloud both backends refuse is agreement; it must not divide by an empty maximum."""
    report = evaluate_grasp_conformance(
        reference_poses=[], candidate_poses=[], reference_confidences=[], candidate_confidences=[]
    )

    assert report.passed
    assert report.metrics == {
        "grasp_count_delta": 0,
        "grasp_translation_error_m": 0.0,
        "grasp_rotation_error_deg": 0.0,
        "grasp_confidence_delta": 0.0,
    }


def test_grasp_thresholds_are_explicit_and_can_be_tightened() -> None:
    inputs = _grasps()

    report = evaluate_grasp_conformance(**inputs, thresholds=ConformanceThresholds(maximum_grasp_confidence_delta=0.0))

    assert not report.passed
    assert report.failures == ("grasp confidence delta exceeds threshold",)
