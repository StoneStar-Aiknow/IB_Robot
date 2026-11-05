"""Solver, safety and path behaviour on the official model."""

import numpy as np
import pytest

from pose_composer import kin, planner, safety, solver, trajectory
from pose_composer.model import ARM_JOINTS, load

CROSSED = {"left_shoulder_pitch_joint": -1.2, "right_shoulder_pitch_joint": -1.2, "left_shoulder_yaw_joint": -1.4,
           "right_shoulder_yaw_joint": 1.4, "left_elbow_joint": -2.2, "right_elbow_joint": -2.2}  # fmt: skip


@pytest.mark.parametrize(
    ("side", "upper", "fore"),
    [
        ("left", [1, 0, 0], [1, 0, 0]),
        ("right", [0, -1, 0], [0, -1, 0]),
        ("left", [0, 0, 1], [0, 0, 1]),
        ("right", [0, -1, 0], [0, 0, 1]),
    ],
)
def test_direction_solve_reaches_reachable_targets(side, upper, fore):
    sol = solver.solve_dirs(side, upper, fore, load().rest)
    pose = kin.fk(sol.q)
    assert not sol.approx
    assert kin.angle_deg(pose.upper(side), upper) < 5
    assert kin.angle_deg(pose.fore(side), fore) < 5


def test_an_unreachable_direction_is_reported_not_faked():
    sol = solver.solve_dirs("left", [0, -1, 0], [0, -1, 0], load().rest)  # across the body
    assert sol.approx
    assert safety.limit_violations({**load().rest, **sol.q}) == []


def test_a_hand_target_is_reached_and_stays_clear_of_the_head():
    points = kin.fk({}).points
    target = points["head_top"] + np.array([0.0, 0.0, 0.08]) - solver.shoulder_points()["right"]
    sol = solver.solve_hand("right", target, [0, -1, 0], load().rest)
    q = {**load().rest, **sol.q}
    assert sol.residual_m["hand"] < 0.06
    assert safety.check_pose(q).ok


def test_safety_rejects_known_bad_poses_and_accepts_the_rest_pose():
    assert not safety.check_pose(CROSSED).ok
    assert not safety.check_pose({"right_elbow_joint": 0.3}).ok
    assert safety.check_pose({}).ok


def test_projection_makes_a_colliding_pose_safe():
    q, delta = solver.project_safe(CROSSED)
    assert safety.check_pose(q).ok
    assert delta > 0


def test_paths_are_speed_limited_and_end_on_target():
    rest = load().rest
    target = dict(rest, left_shoulder_pitch_joint=-2.9, right_shoulder_pitch_joint=-2.9)
    traj = trajectory.plan(rest, [target])
    speed = np.max(np.abs(np.diff(traj.positions, axis=0)) / np.diff(traj.times)[:, None])
    assert speed <= trajectory.DEFAULT_SPEED_RADPS * 1.01
    assert np.allclose(traj.positions[-1], [target[n] for n in ARM_JOINTS])


def test_a_path_that_would_sweep_through_the_body_goes_round_it():
    """Into a folded-arms pose from the rest pose: the straight path brushes the hip."""
    rest = load().rest
    folded = dict(rest)
    for side in ("left", "right"):
        chest = kin.fk({}).points["chest"] + np.array([0.12, -0.05 if side == "left" else 0.05, -0.02])
        folded.update(solver.solve_hand(side, chest - solver.shoulder_points()[side], None, folded).q)
    folded, _ = solver.project_safe(folded)
    plan = planner.plan_safe(rest, [folded], [0.5])
    assert plan.ok
    assert safety.check_path(plan.traj.times, plan.traj.positions, ARM_JOINTS).ok
