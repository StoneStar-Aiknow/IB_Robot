"""Planner-independent candidate ordering and filtering."""

from __future__ import annotations

import numpy as np
import pytest

from manipulation_execution.grasp_pipeline import candidates
from manipulation_execution.grasp_pipeline.candidates import approach_priority, enclosed_points, sweep_boxes

#: The deployed OmniHand gripper description's closing sweep (models/graspgenx_omnihand_left_x2,
#: assets/x_grippers/omnihand_left_x2/config.json), copied so the test needs no model bundle.
OMNIHAND = {
    "sweep_volume": {
        "extents": [0.0737, 0.071, 0.04],
        "offset": [0.0, 0.0, 0.119],
        "extents2": [0.0368, 0.071, 0.03],
        "offset2": [0.0, 0.0, 0.099],
    }
}


def _grasps_with_approach(*approaches):
    grasps = np.repeat(np.eye(4)[None], len(approaches), axis=0)
    for grasp, approach in zip(grasps, approaches, strict=True):
        z = np.asarray(approach, dtype=float) / np.linalg.norm(approach)
        x = np.cross([0.0, 1.0, 0.0] if abs(z[1]) < 0.9 else [1.0, 0.0, 0.0], z)
        x /= np.linalg.norm(x)
        grasp[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    return grasps


def test_approach_priority_by_target_height():
    # side, top-down, oblique (31 deg down: neither class)
    grasps = _grasps_with_approach([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, -0.6])
    up = [0.0, 0.0, 1.0]
    priority, report = approach_priority(grasps, np.eye(4), up, 0.25)
    assert priority.tolist() == [0, 1, 1] and report["mode"] == "side_first"
    assert report["side"] == [0] and report["top_down"] == [1]
    # A tall object never gets a candidate coming down below the side band.
    assert report["excluded"] == [1, 2]
    priority, report = approach_priority(grasps, np.eye(4), up, 0.03)
    assert priority.tolist() == [1, 0, 1] and report["mode"] == "top_first"
    assert report["excluded"] == []
    for height in (0.08, None, float("nan")):
        priority, report = approach_priority(grasps, np.eye(4), up, height)
        assert priority.tolist() == [0, 0, 0] and report["mode"] == "none" and report["excluded"] == []
    # Up is the support normal expressed through base_from_camera, not camera z.
    flip = np.diag([1.0, 1.0, -1.0, 1.0])
    flip[1, 1] = -1.0
    priority, _ = approach_priority(grasps, flip, up, 0.03)
    assert priority.tolist() == [1, 1, 1]


def test_side_first_band_edge_and_switch(monkeypatch):
    # cos(approach, up) = -0.39 (kept), -0.41 (excluded), +0.6 (from below: kept, later)
    grasps = _grasps_with_approach([0.92, 0.0, -0.39], [0.91, 0.0, -0.41], [0.8, 0.0, 0.6])
    _, report = approach_priority(grasps, np.eye(4), [0.0, 0.0, 1.0], 0.2)
    assert report["excluded"] == [1]
    monkeypatch.setattr(candidates, "SIDE_FIRST_EXCLUDES_FROM_ABOVE", False)
    _, report = approach_priority(grasps, np.eye(4), [0.0, 0.0, 1.0], 0.2)
    assert report["excluded"] == []


def test_enclosed_points_counts_only_the_closing_sweep():
    # Target cloud: a 2 cm cube of points centred 12 cm ahead of the first
    # grasp (inside its sweep); the second grasp faces away from it.
    axis = np.linspace(-0.01, 0.01, 5)
    cube = np.stack(np.meshgrid(axis, axis, axis), axis=-1).reshape(-1, 3) + [0.0, 0.0, 0.12]
    toward = np.eye(4)
    away = np.eye(4)
    away[:3, :3] = np.diag([1.0, -1.0, -1.0])
    beside = np.eye(4)
    beside[0, 3] = 0.2
    boxes = sweep_boxes(OMNIHAND)
    counts = enclosed_points(np.stack([toward, away, beside]), cube, boxes)
    assert counts.tolist() == [len(cube), 0, 0]
    assert enclosed_points(np.stack([toward, away]), cube, boxes, candidates=[1]).tolist() == [-1, 0]


def test_sweep_boxes_are_half_extents_around_their_offsets():
    (offset, half), (offset2, half2) = sweep_boxes(OMNIHAND)
    np.testing.assert_allclose(offset, [0.0, 0.0, 0.119])
    np.testing.assert_allclose(half, [0.03685, 0.0355, 0.02])
    np.testing.assert_allclose(offset2, [0.0, 0.0, 0.099])
    np.testing.assert_allclose(half2, [0.0184, 0.0355, 0.015])
    with pytest.raises(KeyError):
        sweep_boxes({})
