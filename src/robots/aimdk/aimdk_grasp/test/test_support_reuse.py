"""A place may reuse the pick's table plane only when its own frame still shows that table."""

import numpy as np
import pytest
from aimdk_grasp.scene_estimation import SceneEstimationConfig, SupportPlane, _reuse_support
from scipy.spatial.transform import Rotation

CONFIG = SceneEstimationConfig()


def _torso_from_camera(seed=0):
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(np.random.default_rng(seed).normal(size=3)).as_matrix()
    transform[:3, 3] = (0.1, -0.05, 0.6)
    return transform


def _to_camera(points, torso_from_camera):
    rotation, translation = torso_from_camera[:3, :3], torso_from_camera[:3, 3]
    return (np.asarray(points, float) - translation) @ rotation


def _table(height=0.0, half=0.4, step=0.01, centre=(0.4, 0.0)):
    x, y = np.meshgrid(
        np.arange(-half, half, step) + centre[0], np.arange(-half, half, step) + centre[1], indexing="ij"
    )
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, height)])


def _bowl(centre=(0.4, 0.05), radius=0.09, height=0.08):
    angles = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    z = np.linspace(0.005, height, 10)
    a, h = np.meshgrid(angles, z)
    return np.column_stack([centre[0] + radius * np.cos(a.ravel()), centre[1] + radius * np.sin(a.ravel()), h.ravel()])


def _hint():
    patch = _table(step=0.02)
    return SupportPlane(np.array([0.4, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), patch, {})


def _reuse(table, bowl, *, seed=0):
    torso_from_camera = _torso_from_camera(seed)
    return _reuse_support(
        _hint(), torso_from_camera, _to_camera(table, torso_from_camera), _to_camera(bowl, torso_from_camera), CONFIG
    ), torso_from_camera


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_same_table_is_reused_in_the_current_camera_frame(seed):
    rng = np.random.default_rng(seed)
    table = _table() + rng.normal(scale=0.002, size=(len(_table()), 3)) * [0, 0, 1]
    (plane, report), torso_from_camera = _reuse(table, _bowl(), seed=seed)
    assert report["accepted"] is True and plane is not None
    rotation, translation = torso_from_camera[:3, :3], torso_from_camera[:3, 3]
    # Back in the torso frame it is the hint itself, with the target's heights measured now.
    np.testing.assert_allclose(rotation @ plane.normal, [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(rotation @ plane.point + translation, [0.4, 0.0, 0.0], atol=1e-9)
    assert plane.target_heights_m["p05"] == pytest.approx(0.0125, abs=0.01)
    assert abs(report["offset_m"]) < 0.002 and report["inlier_share"] > 0.95


@pytest.mark.parametrize(
    "table,bowl,reason",
    [
        (_table(height=0.012), _bowl(), "the table moved or is not flat here"),
        (_table(), _bowl(centre=(1.2, 0.0)), "target centre outside the patch"),
        (_table(half=0.4)[::400], _bowl(), "too few table points near the target"),
        (_table(), _bowl() - [0, 0, 0.05], "target below the plane"),
    ],
)
def test_any_doubt_falls_back_to_the_search(table, bowl, reason):
    (plane, report), _ = _reuse(table, bowl)
    assert plane is None and report["accepted"] is False and report["reason"] == reason


class _Filtered(Exception):
    pass


def _filtered_scene(monkeypatch, near_hand, target_pixels):
    """What complete_scene passes on after its filters: (near points, target points)."""
    from aimdk_grasp import scene_estimation

    points = np.arange(18, dtype=np.float32).reshape(6, 3)
    n = len(points)
    prepared = scene_estimation.PreparedScene(
        points, np.arange(n), np.ones(n, bool), np.ones(n, bool), np.ones(n, bool), np.ones(n, bool),
        np.empty((0, 3)), np.empty(0, int), (1, n), np.empty((0, 4)), np.eye(4), 1, "camera", 1.0, True,
        CONFIG, n, 0.0, 0.0, moving_hand_near=near_hand,
    )  # fmt: skip
    seen = {}

    def capture(near, target, up, config):
        seen.update(near=np.asarray(near).copy(), target=np.asarray(target).copy())
        raise _Filtered

    monkeypatch.setattr(scene_estimation, "select_support_plane", capture)
    mask = np.zeros((1, n), bool)
    mask[0, target_pixels] = True
    with pytest.raises(_Filtered):
        scene_estimation.complete_scene(prepared, mask)
    return points, seen


def test_the_moving_hands_points_leave_the_scene_but_the_targets_stay(monkeypatch):
    """BK3 (2026-11-09): a patch of the real hand at grasp_ready refused all 22 approach paths."""
    near_hand = np.array([True, True, False, False, False, True])
    points, seen = _filtered_scene(monkeypatch, near_hand, [0, 3, 4])
    # Point 0 is near the hand but on the target: it stays. Points 1 and 5 are the hand's own.
    np.testing.assert_array_equal(seen["target"], points[[0, 3, 4]])
    np.testing.assert_array_equal(seen["near"], points[[2]])


def test_without_the_hand_mask_the_scene_is_unchanged(monkeypatch):
    points, seen = _filtered_scene(monkeypatch, None, [0, 3, 4])
    np.testing.assert_array_equal(seen["near"], points[[1, 2, 5]])
