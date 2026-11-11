"""Visible cavity, clearance, and Agent-provided container descriptions."""

from dataclasses import replace
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


def point_linked_fringe(local):
    """Reference: without_fringe linking every point pair, as before voxel linking."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    tree = cKDTree(local)
    spacing = float(np.median(tree.query(local, k=2)[0][:, 1]))
    pairs = tree.query_pairs(max(0.015, 4.0 * spacing), output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=bool), (pairs[:, 0], pairs[:, 1])), shape=(len(local),) * 2)
    _, labels = connected_components(graph, directed=False)
    keep = np.bincount(labels)[labels] >= 0.05 * len(local)
    return local[keep] if keep.any() else local


@pytest.mark.parametrize("seed", range(4))
def test_voxel_linking_keeps_exactly_what_point_linking_kept(seed):
    rng = np.random.default_rng(seed)
    apple = visible_half_sphere(step=0.0009) + rng.normal(0, 0.0005, (1, 3))  # ~12k points, like a head-camera pick
    pieces = [
        apple.mean(0) + direction / np.linalg.norm(direction) * distance + rng.normal(0, 0.003, (count, 3))
        for direction, distance, count in zip(
            rng.normal(size=(5, 3)), rng.uniform(0.07, 0.12, 5), rng.integers(5, 900, 5), strict=True
        )
    ]
    local = np.vstack([apple, *pieces])
    np.testing.assert_array_equal(without_fringe(local), point_linked_fringe(local))


def test_a_piece_within_the_link_still_joins_across_voxels():
    apple = visible_half_sphere(step=0.0009)
    near = apple[np.argmax(apple[:, 0])] + np.array([0.014, 0.0, 0.0]) + np.zeros((40, 3))
    assert len(without_fringe(np.vstack([apple, near]))) == len(apple) + len(near)


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


def lopsided_bowl(radius=0.09, centre=(0.4, -0.05), near=150, far=12):
    """A round rim seen from one side: many points on the near half, few on the far half."""
    angles = np.concatenate([np.linspace(-np.pi / 2, np.pi / 2, near), np.linspace(np.pi / 2, 3 * np.pi / 2, far)])
    rim = np.column_stack(
        [centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles), np.full(len(angles), 0.1)]
    )
    x, y = np.meshgrid(np.linspace(-0.6, 0.6, 15) * radius, np.linspace(-0.6, 0.6, 15) * radius)
    bottom = np.column_stack([centre[0] + x.ravel(), centre[1] + y.ravel(), np.zeros(x.size)])
    return np.vstack([rim, bottom])


def test_opening_centre_is_the_largest_inscribed_circle_not_the_vertex_mean():
    from aimdk_grasp.place_targets import opening_centre
    from scipy.spatial import ConvexHull

    points = lopsided_bowl()[:, :2]
    hull = ConvexHull(points)
    centre, radius = opening_centre(hull.equations)
    np.testing.assert_allclose(centre, (0.4, -0.05), atol=0.004)
    assert radius == pytest.approx(0.09, abs=0.006)
    # The vertex mean leans to the densely seen near rim and understates the opening.
    mean = points[hull.vertices].mean(axis=0)
    understated = float(np.min(-(mean @ hull.equations[:, :2].T + hull.equations[:, 2])))
    assert understated < radius - 0.02


def test_a_one_sided_rim_view_is_not_refused_as_too_small():
    # Needs 0.068 + 0.015 = 0.083 m of the 0.09 m opening: the vertex mean read about 0.06 m here.
    container = opening_from_points(lopsided_bowl(), [0, 0, 1], radius_m=0.068)
    assert container.report["clearance_m"] == pytest.approx(0.09, abs=0.006)
    with pytest.raises(GraspFailure) as error:
        opening_from_points(lopsided_bowl(radius=0.07), [0, 0, 1], radius_m=0.068)
    assert error.value.code == "CONTAINER_TOO_SMALL"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cover_reach_keeps_exactly_what_the_full_loop_kept(seed):
    from aimdk_grasp.container_perception import cover_reach
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
    base_from_camera[:3, 3] = rng.normal(size=3)
    points = rng.uniform(-1.0, 1.0, size=(20000, 3))
    base = points @ base_from_camera[:3, :3].T + base_from_camera[:3, 3]
    # A palm-sized cluster of spheres sitting among the points.
    centers = base[rng.integers(len(base))] + rng.normal(scale=0.04, size=(13, 3))
    radii = rng.uniform(0.01, 0.05, size=13)
    full = np.full(len(base), np.inf)
    for center, radius in zip(centers, radii, strict=True):
        full = np.minimum(full, np.linalg.norm(base - center, axis=1) - radius)
    reach = cover_reach(points, base_from_camera, centers, radii, 0.02)
    for margin in (0.005, 0.01, 0.02):
        np.testing.assert_array_equal(reach > margin, full > margin)
    near = np.isfinite(reach)
    assert 0 < near.sum() < len(points)  # the box did skip far points
    np.testing.assert_array_equal(reach[near], full[near])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_within_spheres_answers_what_cover_reach_answers(seed):
    from aimdk_grasp.container_perception import cover_reach, within_spheres
    from scipy.spatial.transform import Rotation

    rng = np.random.default_rng(seed)
    base_from_camera = np.eye(4)
    base_from_camera[:3, :3] = Rotation.from_rotvec(rng.normal(size=3)).as_matrix()
    base_from_camera[:3, 3] = rng.normal(size=3)
    points = rng.uniform(-1.0, 1.0, size=(20000, 3))
    base = points @ base_from_camera[:3, :3].T + base_from_camera[:3, 3]
    # A hand-sized cluster of many small spheres.
    centers = base[rng.integers(len(base))] + rng.normal(scale=0.05, size=(300, 3))
    radii = rng.uniform(0.002, 0.06, size=300)
    for margin in (0.01, 0.052):
        within = within_spheres(points, base_from_camera, centers, radii, margin)
        np.testing.assert_array_equal(within, cover_reach(points, base_from_camera, centers, radii, margin) <= margin)
        assert 0 < within.sum() < len(points)


def test_the_holding_hand_leaves_neither_scene_within_the_coarse_margin(monkeypatch, capsys):
    """CR6 (11-08) / L02 (11-09): a 1-2 cm patch of the real holding hand, beyond the fine set's 1 cm self
    filter but inside the coarse set's 5.2 cm, rejected every planned way home at the handover pose."""
    import json
    from types import SimpleNamespace

    from aimdk_grasp import container_perception
    from aimdk_grasp.scene_estimation import PreparedScene, SceneEstimationConfig

    from manipulation_execution.grasp_pipeline.contracts import ContainerEstimate

    # Torso == camera. The palm at (0.35, -0.24, 0.22) with one 4 cm sphere; a wrist sphere behind it;
    # a left-arm sphere and a torso sphere that must not count as the holding hand.
    palm = np.eye(4)
    palm[:3, 3] = [0.35, -0.24, 0.22]
    spheres = np.array(
        [
            [0.35, -0.24, 0.22, 0.04],  # R_palm
            [0.30, -0.24, 0.24, 0.03],  # right_wrist_roll_link
            [0.35, 0.24, 0.22, 0.04],  # left_wrist_roll_link (the other arm)
            [0.00, 0.00, 0.00, 0.10],  # torso_link
        ]
    )
    model = SimpleNamespace(
        side="right",
        arm=("right_arm",),
        sphere_links=["R_palm", "right_wrist_roll_link", "left_wrist_roll_link", "torso_link"],
        link_indices={"R_palm": 0},
        last_link_frames=palm[:3][None, None],
        spheres=lambda feedback, q: (spheres[None], np.array([True])),
    )
    points = np.array(
        [
            [0.35, -0.24 - 0.04 - 0.03, 0.22],  # 3 cm off the palm sphere: the hand patch (dropped)
            [0.35, -0.24 - 0.04 - 0.08, 0.22],  # 8 cm off: a real obstacle (kept)
            [0.35, 0.24 - 0.04 - 0.03, 0.22],  # 3 cm off the OTHER arm (kept: not the holding hand)
            [0.45, 0.0, 0.0],  # the container (kept)
            [0.46, 0.0, 0.0],
        ]
    )
    n = len(points)
    prepared = PreparedScene(
        points, np.arange(n), np.ones(n, bool), np.ones(n, bool), np.ones(n, bool), np.ones(n, bool),
        np.empty((0, 3)), np.empty(0, int), (1, n), np.empty((0, 4)), np.eye(4), 1, "camera", 1.0, True,
        SceneEstimationConfig(), n, 0.0, 0.0,
    )  # fmt: skip
    seen = {}

    def complete(prepared, mask, **kwargs):
        seen["prepared"] = prepared
        return SimpleNamespace(support=SimpleNamespace(normal=np.array([0.0, 0, 1])), diagnostics={})

    monkeypatch.setattr(container_perception, "placement_cover", lambda *args: np.array([[0.0, 0, 0.06, 0.03]]))
    monkeypatch.setattr(container_perception, "complete_scene", complete)
    monkeypatch.setattr(
        container_perception,
        "opening_from_points",
        lambda points, normal, radius_m: ContainerEstimate(np.eye(3), np.array([0.0, 0, 1]), 0.0),
    )
    feedback = SimpleNamespace(vector=lambda names: np.zeros(1))
    observation = SimpleNamespace(
        robot_state=SimpleNamespace(live=SimpleNamespace(observation=SimpleNamespace(image_feedback=feedback))),
        base_from_camera=np.eye(4),
    )
    settings = SimpleNamespace(scene=SimpleNamespace(validation_self_filter_margin_m=0.01, self_filter_margin_m=0.052))
    held = SimpleNamespace(radius_m=0.04)
    container_perception.estimate_container(observation, np.ones(n, bool), held, model, settings, prepared=prepared)
    kept = seen["prepared"]
    np.testing.assert_array_equal(kept.keep, [False, True, True, True, True])
    np.testing.assert_array_equal(kept.planning_all, [False, True, True, True, True])
    # The container is that scene's target; its fine mesh keeps the environment's pitch, not the pick target's.
    assert kept.config.validation_target_mesh_pitch_m == kept.config.validation_mesh_pitch_m == 0.005
    assert replace(kept.config, validation_target_mesh_pitch_m=0.0025) == prepared.config
    line = json.loads(
        next(text for text in capsys.readouterr().err.splitlines() if text.startswith('{"estimate_container"'))
    )
    assert line["estimate_container"]["holding_hand_dropped"] == 1
