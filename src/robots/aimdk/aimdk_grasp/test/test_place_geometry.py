"""Visible cavity, clearance, and Agent-provided container descriptions."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.container_perception import opening_from_points
from aimdk_grasp.container_proposer import ContainerProposer, detect_masks
from aimdk_grasp.held_object import held_center, without_fringe, without_outliers
from aimdk_grasp.proposer import PerceptionSettings

from manipulation_execution.grasp_pipeline.contracts import GraspFailure, Observation
from manipulation_execution.grasp_pipeline.perception import Detection


def cloud(*, lid=False, size=0.2):
    edge = np.linspace(-size, size, 60)
    rim = np.concatenate(
        [np.column_stack([edge, np.full_like(edge, sign * size), np.full_like(edge, 0.1)]) for sign in (-1, 1)]
        + [np.column_stack([np.full_like(edge, sign * size), edge, np.full_like(edge, 0.1)]) for sign in (-1, 1)]
    )
    x, y = np.meshgrid(np.linspace(-size * 0.8, size * 0.8, 20), np.linspace(-size * 0.8, size * 0.8, 20))
    bottom = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, 0.1 if lid else 0)])
    return np.vstack([rim, bottom])


def test_opening_requires_observed_cavity_and_object_clearance():
    container = opening_from_points(cloud(), [0, 0, 1], radius_m=0.05)
    assert container.rim_height_m == pytest.approx(0.1)
    assert container.report["cavity_points"] == 400
    for points, radius, reason in [
        (cloud(lid=True), 0.05, "CONTAINER_NOT_OPEN"),
        (cloud(size=0.04), 0.05, "CONTAINER_TOO_SMALL"),
        (cloud()[:240], 0.05, "CONTAINER_NOT_OPEN"),
    ]:
        with pytest.raises(GraspFailure) as error:
            opening_from_points(points, [0, 0, 1], radius_m=radius)
        assert error.value.code == reason
    # The rejection names both sides of the comparison for the operator.
    with pytest.raises(GraspFailure) as error:
        opening_from_points(cloud(size=0.04), [0, 0, 1], radius_m=0.05)
    assert "inscribed radius 0.040 m" in str(error.value) and "held radius 0.050 m" in str(error.value)


def visible_half_sphere(radius=0.04, step=0.002):
    """Points on the camera-facing half of a sphere, about as dense as head-camera pixels."""
    count = int(2 * np.pi * radius**2 / step**2)
    index = np.arange(count) + 0.5
    polar, azimuth = np.arccos(1 - index / count), np.pi * (1 + 5**0.5) * index
    return radius * np.column_stack([np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), -np.cos(polar)])


def held_radius(points):
    return float(np.max(np.linalg.norm(points - points.mean(axis=0), axis=1)))


def test_rim_pixels_behind_the_object_do_not_inflate_the_held_radius():
    apple = visible_half_sphere()
    # cap3: a few silhouette pixels took the table's depth, ~(-8, -2, +4.7) cm off the centre.
    fringe = np.array([-0.08, -0.02, 0.047]) + np.random.default_rng(0).normal(0, 0.003, (7, 3))
    kept = without_fringe(np.vstack([apple, fringe]))
    assert len(kept) == len(apple)
    assert held_radius(kept) == pytest.approx(0.04, abs=0.005)
    assert held_radius(np.vstack([apple, fringe])) > 0.07  # what the maximum alone would have said


def test_large_parts_of_the_object_are_kept_even_when_a_depth_hole_separates_them():
    apple = visible_half_sphere()
    split = np.vstack([apple[apple[:, 0] < -0.0125], apple[apple[:, 0] > 0.0125]])
    assert len(without_fringe(split)) == len(split)


def test_a_sparse_or_fragmented_cloud_is_not_cut_apart():
    grid = np.stack(np.meshgrid(*[np.arange(0.0, 0.2, 0.025)] * 2), -1).reshape(-1, 2)
    sparse = np.column_stack([grid, np.zeros(len(grid))])  # spacing coarser than the 1.5 cm link
    assert len(without_fringe(sparse)) == len(sparse)
    scattered = np.random.default_rng(1).uniform(-1, 1, (30, 3))  # every piece below the share
    np.testing.assert_array_equal(without_fringe(scattered), scattered)


@pytest.mark.parametrize(
    "query", ["a basket", "the red bowl on the left", "a stainless steel tray", "右边的白色收纳盒"]
)
def test_container_description_reaches_detector_without_replacing_attributes(query):
    prompts = []

    def detect(rgb, frame, stamp, prompt, **kwargs):
        prompts.append(prompt)
        return [Detection("container", 0.9, np.array([0, 0, 10, 10]), np.ones((10, 10), bool))]

    services = SimpleNamespace(detect=detect)
    observation = Observation(
        np.zeros((10, 10, 3), np.uint8), np.ones((10, 10)), np.eye(3), "camera", 1, "base", np.eye(4)
    )
    found = ContainerProposer(services).propose(
        observation, query, None, deadline=float("inf"), cancelled=lambda: False
    )
    assert found.query == query and found.mask.all()
    assert prompts == [f"{query} ."]


def test_container_threshold_filters_low_score_candidates_without_changing_object_detection():
    thresholds = []

    def detect(rgb, frame, stamp, prompt, *, box_threshold, text_threshold, **kwargs):
        thresholds.append((box_threshold, text_threshold))
        score = 0.90 if prompt == "a basket ." else 0.45
        if score < box_threshold:
            return []
        return [Detection(prompt, score, np.array([0, 0, 10, 10]), np.ones((10, 10), bool))]

    services = SimpleNamespace(detect=detect)
    observation = Observation(
        np.zeros((10, 10, 3), np.uint8), np.ones((10, 10)), np.eye(3), "camera", 1, "base", np.eye(4)
    )
    proposer = ContainerProposer(services, settings=PerceptionSettings(box_threshold=0.50))
    with pytest.raises(GraspFailure, match="no container mask"):
        proposer.propose(observation, "a box", None, deadline=float("inf"), cancelled=lambda: False)
    assert proposer.propose(observation, "a basket", None, deadline=float("inf"), cancelled=lambda: False).mask.all()
    # Place verification detects the carried object through the shared helper.
    assert list(detect_masks(services, observation, "an orange", cancelled=lambda: False))[0].all()
    assert thresholds == [(0.50, 0.25), (0.50, 0.25), (0.25, 0.25)]


def test_held_center_encloses_a_one_sided_cloud_like_the_whole_object():
    # The camera sees the front half of a 4 cm ball: its mean sits 2 cm towards
    # the camera, so the mean-centred radius would be ~0.045 m instead of 0.04 m.
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(600, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    shell = 0.04 * directions[directions[:, 0] > 0] + [0.1, 0.2, 0.3]
    center = held_center(shell)
    reach = np.max(np.linalg.norm(shell - center, axis=1))
    mean_reach = np.max(np.linalg.norm(shell - shell.mean(axis=0), axis=1))
    assert reach == pytest.approx(0.04, abs=0.002)
    assert mean_reach > 0.044
    assert np.linalg.norm(center - [0.1, 0.2, 0.3]) < 0.004


def test_held_center_is_never_worse_than_the_mean():
    points = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.0, 0.02, 0.0], [0.01, 0.01, 0.03]])
    center = held_center(points)
    assert np.max(np.linalg.norm(points - center, axis=1)) <= np.max(
        np.linalg.norm(points - points.mean(axis=0), axis=1)
    )
    assert held_center(np.array([[1.0, 2.0, 3.0]])) == pytest.approx([1.0, 2.0, 3.0])


def enclosing(points):
    return float(np.max(np.linalg.norm(points - held_center(points), axis=1)))


def test_isolated_depth_noise_does_not_set_the_held_radius():
    rng = np.random.default_rng(2)
    apple = visible_half_sphere()
    apple = apple + rng.normal(0, 0.002, apple.shape)
    # A few points a centimetre off the surface (live 2026-11-06: 8-18 of 518 per pick).
    direction = apple[rng.choice(len(apple), 12, replace=False)]
    noise = direction * (1 + 0.012 / np.linalg.norm(direction, axis=1, keepdims=True))
    noisy = np.vstack([apple, noise])
    assert enclosing(noisy) > 0.05
    kept = without_outliers(noisy)
    assert len(kept) >= 0.95 * len(apple)
    # As if the noise points were never there (the surface's own 2 mm noise stays).
    assert enclosing(kept) <= enclosing(apple) + 0.001
    # Clean, small or uniform clouds lose (almost) nothing and are never emptied.
    assert len(without_outliers(apple)) >= 0.95 * len(apple)
    np.testing.assert_array_equal(without_outliers(apple[:10]), apple[:10])
