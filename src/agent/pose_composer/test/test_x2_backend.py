"""X2 trajectories share result contracts and fail-closed execution validation."""

import numpy as np
import pytest
from aimdk_robot.pose_backend import model, planner, safety, trajectory

from robot_runtime.pose_backend import Plan, Report


@pytest.mark.parametrize("kind", ["nan", "duplicate_time", "wrong_shape", "wrong_joint", "empty"])
def test_malformed_paths_fail_closed(kind):
    q = model.load().rest
    path = trajectory.plan(q, [q], [0.0])
    times, positions = path.times.copy(), path.positions.copy()
    joints = list(model.ARM_JOINTS)
    if kind == "nan":
        positions[1, 0] = np.nan
    elif kind == "duplicate_time":
        times[1] = times[0]
    elif kind == "wrong_shape":
        positions = positions[:, :-1]
    elif kind == "wrong_joint":
        joints[0] = "unknown_joint"
    else:
        times, positions = times[:0], positions[:0]
    assert not safety.check_path(times, positions, joints).ok


def test_execution_checks_clipped_trajectory_speed():
    q = model.load().rest
    positions = np.array([[q[n] for n in model.ARM_JOINTS]] * 3)
    # Even clipping must not turn a large measured-start discrepancy into a jump.
    positions[0, 0] = model.load().hard_limits[model.ARM_JOINTS[0]][0] - 1.0
    raw = Plan(trajectory.Trajectory(np.array([0.0, 0.01, 0.02]), positions), True, [], Report(True))
    result = planner.prepare_execution(raw)
    assert not result.ok
    assert any("peak speed" in note for note in result.report.notes)


def test_joint_limits_only_narrow_and_reject_missing_ranges():
    from robot_runtime.pose_backend import intersect_joint_limits

    assert intersect_joint_limits(["a"], {"a": [-2, 2]}, {"a": [-1, 3]}) == {"a": (-1.0, 2.0)}
    for bounds in ({}, {"a": [float("nan"), 1]}, {"a": [3, 4]}, {"a": [True, 2]}):
        with pytest.raises(ValueError):
            intersect_joint_limits(["a"], {"a": [-2, 2]}, bounds)
