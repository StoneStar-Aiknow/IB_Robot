"""Trajectory files for replaying named motions: validation and sampling, no ROS."""

from __future__ import annotations

import json

import pytest
from aimdk_robot.projection import VENDOR_ARM_ORDER
from aimdk_robot.trajectory import TrajectoryError, load_trajectory

JOINTS = list(VENDOR_ARM_ORDER)


def _write(tmp_path, **overrides):
    data = {
        "target": "right",
        "joints": JOINTS,
        "time_from_start_s": [0.0, 1.0],
        "positions": [[0.0] * len(JOINTS), [0.5] + [0.0] * (len(JOINTS) - 1)],
    }
    data.update(overrides)
    path = tmp_path / "wave.json"
    path.write_text(json.dumps(data))
    return path


def test_samples_are_mapped_to_vendor_order_and_interpolated(tmp_path):
    reversed_joints = list(reversed(JOINTS))
    row = [0.0] * len(JOINTS)
    row[reversed_joints.index("right_elbow_joint")] = 0.4
    trajectory = load_trajectory(
        _write(tmp_path, joints=reversed_joints, positions=[[0.0] * len(JOINTS), row]), max_speed_radps=3.0
    )
    elbow = JOINTS.index("right_elbow_joint")
    assert trajectory.sample(0.25)[elbow] == pytest.approx(0.1)
    assert trajectory.sample(5.0)[elbow] == 0.4


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"joints": JOINTS[:-1]}, "joints"),
        ({"time_from_start_s": [0.0, 0.0]}, "strictly increase"),
        ({"time_from_start_s": [0.0, 0.01]}, "rad/s"),
        # A recording without its side would replay that side for any requested one.
        ({"target": ""}, "target"),
        ({"target": "sideways"}, "target"),
    ],
)
def test_a_file_that_would_execute_an_unrecorded_motion_is_refused(tmp_path, overrides, reason):
    with pytest.raises(TrajectoryError, match=reason):
        load_trajectory(_write(tmp_path, **overrides), max_speed_radps=3.0)


# --- planned (caller-computed) trajectories: joint.trajectory goals ----------

LIMITS = {joint: (-2.0, 2.0) for joint in JOINTS}
START = [0.1] * len(JOINTS)


def _plan(names, times, points, **kw):
    from aimdk_robot.trajectory import planned_trajectory

    options = {"limits": LIMITS, "max_speed_radps": 6.0, "rate_hz": 50.0}
    options.update(kw)
    return planned_trajectory("trajectory", START, names, times, points, **options)


def test_a_partial_goal_moves_its_joints_and_holds_the_rest_from_the_measured_start():
    plan = _plan(["right_elbow_joint"], [1.0], [[-1.0]])
    elbow = JOINTS.index("right_elbow_joint")
    assert plan.sample(0.0) == pytest.approx(START)
    end = plan.sample(plan.duration)
    assert end[elbow] == pytest.approx(-1.0)
    assert [v for k, v in enumerate(end) if k != elbow] == pytest.approx([0.1] * (len(JOINTS) - 1))
    assert plan.duration == pytest.approx(1.0)


def test_waypoints_are_eased_so_the_motion_starts_and_ends_at_rest():
    plan = _plan(["left_elbow_joint"], [1.0, 2.0], [[-0.5], [-1.0]])
    k = JOINTS.index("left_elbow_joint")
    early = plan.sample(0.02)[k] - START[k]
    middle = plan.sample(0.52)[k] - plan.sample(0.5)[k]
    assert abs(early) < abs(middle)
    assert plan.sample(1.0)[k] == pytest.approx(-0.5)


@pytest.mark.parametrize(
    ("names", "times", "points", "reason"),
    [
        (["no_such_joint"], [1.0], [[0.0]], "unknown"),
        (["left_elbow_joint", "left_elbow_joint"], [1.0], [[0.0, 0.0]], "distinct"),
        (["left_elbow_joint"], [0.0], [[0.0]], "positive"),
        (["left_elbow_joint"], [1.0, 1.0], [[0.0], [0.1]], "increasing"),
        (["left_elbow_joint"], [1.0], [[float("nan")]], "finite"),
        (["left_elbow_joint"], [1.0], [[2.5]], "outside"),
        (["left_elbow_joint"], [0.1], [[-1.9]], "rad/s"),
    ],
)
def test_a_goal_that_would_execute_an_unplanned_motion_is_refused(names, times, points, reason):
    with pytest.raises(TrajectoryError, match=reason):
        _plan(names, times, points)
