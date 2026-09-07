from __future__ import annotations

import inspect

import numpy as np

from model_utils.graspgen_contract import GRASPGEN_NPOINTS, GRASPGEN_NSAMPLES, GRASPGEN_RADII
from perception_service.graspgen_geometry import (
    ball_query,
    build_pointnet_geometry,
    furthest_point_sample,
    group_features,
    group_xyz,
)


def _reference_furthest_point_sample(xyz: np.ndarray, npoint: int) -> np.ndarray:
    points = np.ascontiguousarray(xyz, dtype=np.float32)
    indices = np.zeros(npoint, dtype=np.int64)
    distances = np.full(len(points), np.float32(1e10), dtype=np.float32)
    old = 0
    valid = np.sum(points * points, axis=1, dtype=np.float32) > np.float32(1e-3)
    for output_index in range(1, npoint):
        delta = points - points[old]
        distance = np.sum(delta * delta, axis=1, dtype=np.float32)
        distances[valid] = np.minimum(distances[valid], distance[valid])
        if np.any(valid):
            old = int(np.argmax(np.where(valid, distances, np.float32(-1.0))))
        else:
            old = 0
        indices[output_index] = old
    return indices


def _reference_ball_query(xyz: np.ndarray, centers: np.ndarray, radius: float, nsample: int) -> np.ndarray:
    points = np.ascontiguousarray(xyz, dtype=np.float32)
    query = np.ascontiguousarray(centers, dtype=np.float32)
    delta = query[:, None, :] - points[None, :, :]
    distance2 = np.sum(delta * delta, axis=-1, dtype=np.float32)
    radius2 = np.float32(radius) * np.float32(radius)
    indices = np.zeros((len(query), nsample), dtype=np.int64)
    for query_index, row in enumerate(distance2):
        matches = np.flatnonzero(row < radius2)[:nsample]
        if len(matches):
            indices[query_index].fill(int(matches[0]))
            indices[query_index, : len(matches)] = matches
    return indices


def _reference_group_xyz(xyz: np.ndarray, centers: np.ndarray, indices: np.ndarray) -> np.ndarray:
    grouped = xyz[indices] - centers[:, None, :]
    return np.ascontiguousarray(grouped.transpose(2, 0, 1)[None], dtype=np.float32)


def _reference_group_features(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    grouped = features[0].T[indices]
    return np.ascontiguousarray(grouped.transpose(2, 0, 1)[None], dtype=np.float32)


def test_furthest_point_sample_matches_reference_bit_for_bit():
    rng = np.random.default_rng(20260730)
    points = rng.normal(size=(512, 3)).astype(np.float32) * np.float32(0.05)
    points[:8] = np.float32(0.0)

    expected = _reference_furthest_point_sample(points, 96)
    actual = furthest_point_sample(points, 96)

    np.testing.assert_array_equal(actual, expected)


def test_furthest_point_sample_preserves_all_invalid_behavior():
    points = np.zeros((32, 3), dtype=np.float32)

    np.testing.assert_array_equal(furthest_point_sample(points, 16), np.zeros(16, dtype=np.int64))


def test_ball_query_matches_reference_bit_for_bit():
    rng = np.random.default_rng(42)
    points = rng.normal(size=(512, 3)).astype(np.float32) * np.float32(0.04)
    centers = points[rng.choice(len(points), 64, replace=False)]

    expected = _reference_ball_query(points, centers, 0.035, 48)
    actual = ball_query(points, centers, 0.035, 48)

    np.testing.assert_array_equal(actual, expected)


def test_grouping_helpers_match_reference_layout_and_values():
    rng = np.random.default_rng(17)
    points = rng.normal(size=(128, 3)).astype(np.float32)
    centers = points[[2, 7, 11, 19]]
    indices = rng.integers(0, len(points), size=(len(centers), 12), dtype=np.int64)
    features = rng.normal(size=(1, 64, len(points))).astype(np.float32)

    grouped_xyz = group_xyz(points, centers, indices)
    grouped_features = group_features(features, indices)

    np.testing.assert_array_equal(grouped_xyz, _reference_group_xyz(points, centers, indices))
    np.testing.assert_array_equal(grouped_features, _reference_group_features(features, indices))
    assert grouped_xyz.flags.c_contiguous
    assert grouped_features.flags.c_contiguous


def test_pointnet_geometry_matches_reference_composition():
    rng = np.random.default_rng(99)
    points = rng.normal(size=(256, 3)).astype(np.float32) * np.float32(0.04)
    npoints = (48, 12)
    radii = (0.03, 0.06)
    nsamples = (16, 24)

    stage1_fps = _reference_furthest_point_sample(points, npoints[0])
    stage1_xyz = points[stage1_fps]
    stage1_indices = _reference_ball_query(points, stage1_xyz, radii[0], nsamples[0])
    stage2_fps = _reference_furthest_point_sample(stage1_xyz, npoints[1])
    stage2_xyz = stage1_xyz[stage2_fps]
    stage2_indices = _reference_ball_query(stage1_xyz, stage2_xyz, radii[1], nsamples[1])
    actual = build_pointnet_geometry(points, npoints=npoints, radii=radii, nsamples=nsamples)

    np.testing.assert_array_equal(
        actual.stage1_input,
        _reference_group_xyz(points, stage1_xyz, stage1_indices),
    )
    np.testing.assert_array_equal(
        actual.stage2_relative_xyz,
        _reference_group_xyz(stage1_xyz, stage2_xyz, stage2_indices),
    )
    np.testing.assert_array_equal(actual.stage2_indices, stage2_indices)
    np.testing.assert_array_equal(actual.stage2_xyz, stage2_xyz)


def test_pointnet_geometry_defaults_come_from_the_shared_contract():
    """A caller that omits the counts must group points the way the OMs were traced.

    The exporter bakes these counts into the static input shapes of the eight subgraphs,
    so a default that drifted from ``inference_manifest.graspgen`` would not be caught by
    any manifest check - the session would simply hand an OM a wrongly sized group.
    """
    defaults = inspect.signature(build_pointnet_geometry).parameters

    assert defaults["npoints"].default == GRASPGEN_NPOINTS
    assert defaults["radii"].default == GRASPGEN_RADII
    assert defaults["nsamples"].default == GRASPGEN_NSAMPLES
