# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""GraspGen's preprocessing and decoding are an adapter, like every other perception model.

They used to be a policy backend's private helpers, which meant the constants they depend
on came out of a fabricated LeRobot ``config.json`` and nothing could construct them
without an ACL device. Here they are read from ``assets/adapter.json`` and exercised as
plain NumPy, exactly the way ``RamPlusAdapter`` is.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import GRASPGEN_DEPLOYMENT

from inference_service.generic_runtime import DeploymentIdentity, NamedTensorResult, RuntimeLatency
from perception_service.graspgen_adapter import (
    GRASPGEN_POSTPROCESSING,
    GRASPGEN_PREPROCESSING,
    GraspGenAdapter,
    GraspGenConfig,
)

_ASSETS = {
    "family": "graspgen",
    "preprocessing": GRASPGEN_PREPROCESSING,
    "postprocessing": GRASPGEN_POSTPROCESSING,
    "kappa": 2.0,
    "diffusion_steps": 4,
    "grasp_batch_size": 8,
    "point_count": 64,
    "geometry": {"npoints": [256, 64], "radii": [0.02, 0.04], "nsamples": [64, 128]},
}


def _adapter(**overrides) -> GraspGenAdapter:
    assets = {**_ASSETS, **overrides}
    return GraspGenAdapter(GraspGenConfig.from_assets(assets))


def _named_result(outputs: dict[str, np.ndarray]) -> NamedTensorResult:
    return NamedTensorResult(
        outputs=outputs,
        deployment=DeploymentIdentity("bundle", "uuid", 1, "ascend_310p", "uuid", 1, "fingerprint", "ascend"),
        latency=RuntimeLatency(1.0, 1.0),
    )


def _result(poses: np.ndarray, confidence: np.ndarray) -> NamedTensorResult:
    return _named_result({"grasp.poses": poses, "grasp.confidence": confidence})


def test_the_identity_names_the_contracts_the_packager_stamps_into_the_bundle():
    identity = GraspGenAdapter.identity

    assert identity.family == "graspgen"
    assert identity.preprocessing == GRASPGEN_PREPROCESSING
    assert identity.postprocessing == GRASPGEN_POSTPROCESSING
    assert identity.supported_deployments == frozenset({"torch_cuda", "ascend_310p", "ascend_310b"})
    assert GraspGenAdapter.compiled_abi_finalized is True


def test_from_bundle_reads_the_packaged_adapter_asset(graspgen_bundle):
    adapter = GraspGenAdapter.from_bundle(graspgen_bundle)

    assert isinstance(adapter.config, GraspGenConfig)
    assert adapter.config.grasp_batch_size == 1000
    assert GRASPGEN_DEPLOYMENT in GraspGenAdapter.identity.supported_deployments


def test_from_bundle_refuses_a_bundle_packaged_for_another_model(graspgen_bundle):
    """The identity is what stops a SigLIP2 bundle from being driven as GraspGen."""
    asset_path = graspgen_bundle / "assets" / "adapter.json"
    assets = json.loads(asset_path.read_text(encoding="utf-8"))
    assets["preprocessing"] = "siglip2-image-224-v1"
    asset_path.write_text(json.dumps(assets), encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        GraspGenAdapter.from_bundle(graspgen_bundle)


@pytest.mark.parametrize(
    ("assets", "expected"),
    [
        ({"kappa": 0.0}, "kappa must be positive"),
        ({"kappa": float("inf")}, "kappa must be finite"),
        ({"kappa": "2.0"}, "kappa must be a real number"),
        ({"diffusion_steps": -1}, "diffusion_steps must be positive"),
        ({"diffusion_steps": 1.5}, "diffusion_steps must be an int"),
        ({"point_count": 0}, "point_count must be positive"),
        ({"geometry": [[256, 64], [0.02, 0.04]]}, "geometry must be an object"),
        ({"geometry": {"npoints": [256]}}, "two sampled stages"),
        ({"geometry": {"radii": [0.02, -0.04]}}, r"geometry.radii\[1\] must be positive"),
    ],
)
def test_the_adapter_asset_is_validated_rather_than_trusted(assets, expected):
    """A wrong constant here is a silent shape failure on the device, so it fails here."""
    with pytest.raises(ValueError, match=expected):
        _adapter(**assets)


def test_geometry_defaults_to_the_shared_contract_when_the_asset_omits_it():
    config = GraspGenConfig.from_assets({key: value for key, value in _ASSETS.items() if key != "geometry"})

    assert (config.npoints, config.radii, config.nsamples) == ((256, 64), (0.02, 0.04), (64, 128))


def test_prepare_centres_and_scales_the_cloud_and_hands_back_the_centre():
    adapter = _adapter()
    points = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 5.0]], dtype=np.float32)

    tensors, center = adapter.prepare(points)

    np.testing.assert_allclose(center, [1.0, 2.0, 4.0])
    scaled = tensors["observation.object_points"]
    np.testing.assert_allclose(scaled, (points - center) * 2.0)
    assert scaled.dtype == np.float32
    assert scaled.flags.c_contiguous


def test_prepare_drops_non_finite_points_before_anything_else():
    adapter = _adapter()
    points = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float32)

    tensors, _ = adapter.prepare(points)

    assert len(tensors["observation.object_points"]) == 2


def test_prepare_caps_the_cloud_at_the_packaged_point_count():
    adapter = _adapter(point_count=16)
    points = np.random.default_rng(3).normal(size=(500, 3)).astype(np.float32) * np.float32(0.02)

    tensors, _ = adapter.prepare(points)

    assert len(tensors["observation.object_points"]) == 16


def test_prepare_is_deterministic_so_one_cloud_maps_to_one_grasp_set():
    adapter = _adapter(point_count=16)
    points = np.random.default_rng(4).normal(size=(500, 3)).astype(np.float32) * np.float32(0.02)

    first, _ = adapter.prepare(points)
    second, _ = adapter.prepare(points)

    np.testing.assert_array_equal(first["observation.object_points"], second["observation.object_points"])


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        (np.zeros((4, 2), dtype=np.float32), r"shape \[N, 3\]"),
        (np.zeros((4, 3, 3), dtype=np.float32), r"shape \[N, 3\]"),
        (np.full((4, 3), np.nan, dtype=np.float32), "no finite points"),
    ],
)
def test_prepare_rejects_a_cloud_that_is_not_an_object_point_cloud(points, expected):
    with pytest.raises(ValueError, match=expected):
        _adapter().prepare(points)


def test_prepare_keeps_a_cloud_that_the_outlier_filter_would_empty():
    """Two points are all statistical outlier removal needs to reject everything."""
    adapter = _adapter()
    points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32)

    tensors, _ = adapter.prepare(points)

    assert len(tensors["observation.object_points"]) == 2


def test_postprocess_sorts_by_confidence_and_restores_the_camera_frame():
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(3)])
    poses[:, :3, 3] = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    confidence = np.array([0.1, 0.9, 0.5], dtype=np.float32)

    candidates = _adapter().postprocess(
        _result(poses, confidence), object_center=np.array([0.0, 0.0, 1.0], dtype=np.float32)
    )

    assert [candidate.confidence for candidate in candidates] == pytest.approx([0.9, 0.5, 0.1])
    np.testing.assert_allclose(candidates[0].pose_matrix[:3, 3], [1.0, 0.0, 1.0])
    assert candidates[0].pose_matrix.dtype == np.float32


def test_postprocess_applies_the_confidence_floor_before_the_count_limit():
    poses = np.stack([np.eye(4, dtype=np.float32) for _ in range(4)])
    confidence = np.array([0.9, 0.4, 0.8, 0.2], dtype=np.float32)

    candidates = _adapter().postprocess(_result(poses, confidence), max_grasps=3, min_confidence=0.5)

    assert [candidate.confidence for candidate in candidates] == pytest.approx([0.9, 0.8])


def test_postprocess_leaves_the_poses_alone_when_no_centre_is_given():
    poses = np.stack([np.eye(4, dtype=np.float32)])
    poses[0, :3, 3] = [1.0, 2.0, 3.0]

    candidates = _adapter().postprocess(_result(poses, np.array([0.5], dtype=np.float32)))

    np.testing.assert_allclose(candidates[0].pose_matrix[:3, 3], [1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    ("outputs", "expected"),
    [
        ({"grasp.confidence": np.zeros(2, dtype=np.float32)}, "missing 'grasp.poses'"),
        ({"grasp.poses": np.zeros((2, 4, 4), dtype=np.float32)}, "missing 'grasp.confidence'"),
    ],
)
def test_postprocess_names_the_semantic_the_runtime_failed_to_return(outputs, expected):
    with pytest.raises(RuntimeError, match=expected):
        _adapter().postprocess(_named_result(outputs))


@pytest.mark.parametrize(
    ("poses", "confidence", "expected"),
    [
        (np.zeros((2, 3, 4), dtype=np.float32), np.zeros(2, dtype=np.float32), r"shape \[N, 4, 4\]"),
        (np.zeros((2, 4, 4), dtype=np.float32), np.zeros(3, dtype=np.float32), "2 poses but 3 confidences"),
        (np.full((2, 4, 4), np.nan, dtype=np.float32), np.zeros(2, dtype=np.float32), "non-finite"),
    ],
)
def test_postprocess_refuses_a_grasp_set_the_caller_could_not_act_on(poses, confidence, expected):
    with pytest.raises(RuntimeError, match=expected):
        _adapter().postprocess(_result(poses, confidence))
