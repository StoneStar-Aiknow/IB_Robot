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


def test_a_single_waypoint_is_eased_so_the_motion_starts_and_ends_at_rest():
    plan = _plan(["left_elbow_joint"], [1.0], [[-0.9]])
    k = JOINTS.index("left_elbow_joint")
    early = plan.sample(0.02)[k] - START[k]
    middle = plan.sample(0.52)[k] - plan.sample(0.5)[k]
    late = plan.sample(1.0)[k] - plan.sample(0.98)[k]
    assert abs(early) < abs(middle) and abs(late) < abs(middle)
    assert plan.sample(1.0)[k] == pytest.approx(-0.9)


def test_a_planned_path_keeps_its_timing_instead_of_stopping_at_every_waypoint():
    """A constant 2.5 rad/s plan with 0.1 s waypoints used to be commanded 1.19 -> 3.86 -> 1.19 rad/s."""
    k = JOINTS.index("left_elbow_joint")
    times = [0.1 * i for i in range(1, 9)]
    plan = _plan(["left_elbow_joint"], times, [[START[k] - 0.25 * i] for i in range(1, 9)])
    speeds = [
        abs(plan.positions[i][k] - plan.positions[i - 1][k]) / (plan.times[i] - plan.times[i - 1])
        for i in range(1, len(plan.times))
    ]
    assert speeds == pytest.approx([2.5] * len(speeds))
    # A plan at 4 rad/s, under the 6 rad/s cap, is not refused for an eased peak of about 6.3 rad/s.
    fast = _plan(["left_elbow_joint"], times[:5], [[START[k] - 0.4 * i] for i in range(1, 6)], max_speed_radps=6.0)
    assert fast.positions[-1][k] == pytest.approx(START[k] - 2.0)


@pytest.mark.parametrize("reference", ["start", "measured"])
def test_a_first_point_at_time_zero_where_the_arm_is_is_accepted_and_dropped(reference):
    """MoveIt and most planners begin with the current state at time 0."""
    measured = [0.25] * len(JOINTS)  # the arm settled below the target the command starts from
    k = JOINTS.index("left_elbow_joint")
    first = START[k] if reference == "start" else measured[k]
    plan = _plan(["left_elbow_joint"], [0.0, 1.0], [[first + 0.01], [-0.5]], measured=measured)
    assert plan.sample(0.0) == pytest.approx(START)
    assert plan.sample(plan.duration)[k] == pytest.approx(-0.5)
    assert plan.duration == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("names", "times", "points", "reason"),
    [
        (["no_such_joint"], [1.0], [[0.0]], "unknown"),
        (["left_elbow_joint", "left_elbow_joint"], [1.0], [[0.0, 0.0]], "distinct"),
        (["left_elbow_joint"], [0.0], [[0.1]], "after time 0"),
        (["left_elbow_joint"], [0.0, 1.0], [[0.3], [0.0]], "from where the arm starts"),
        (["left_elbow_joint"], [-0.1, 1.0], [[0.1], [0.0]], "negative"),
        (["left_elbow_joint"], [float("inf")], [[0.0]], "finite"),
        (["left_elbow_joint"], [2.0e9], [[0.0]], "limit"),
        (["left_elbow_joint"], [1.0, 1.0], [[0.0], [0.1]], "increasing"),
        (["left_elbow_joint"], [1.0], [[float("nan")]], "finite"),
        (["left_elbow_joint"], [1.0], [[2.5]], "outside"),
        (["left_elbow_joint"], [0.1], [[-1.9]], "rad/s"),
    ],
)
def test_a_goal_that_would_execute_an_unplanned_motion_is_refused(names, times, points, reason):
    with pytest.raises(TrajectoryError, match=reason):
        _plan(names, times, points)


def _tolerances(requested, final_offset=1.0):
    from aimdk_robot.trajectory import goal_tolerances

    final = list(START)
    final[JOINTS.index("left_elbow_joint")] += final_offset
    names = ["left_elbow_joint", "left_shoulder_pitch_joint"]
    return goal_tolerances(names, requested, START, final, default_rad=0.2, default_ratio=0.5)


def test_an_unset_goal_tolerance_only_catches_a_joint_that_stopped_well_short():
    """Coarse on purpose: a pose held against gravity (0.1-0.16 rad) must not fail the goal."""
    tolerances = _tolerances({})
    assert tolerances["left_elbow_joint"] == pytest.approx(0.5)  # half of its 1 rad travel
    assert tolerances["left_shoulder_pitch_joint"] == pytest.approx(0.2)  # does not move: the floor


def test_a_requested_goal_tolerance_is_kept_and_a_negative_one_is_not_checked():
    tolerances = _tolerances({"left_elbow_joint": 0.03, "left_shoulder_pitch_joint": -1.0})
    assert tolerances == {"left_elbow_joint": pytest.approx(0.03)}


def test_a_goal_tolerance_for_a_joint_that_is_not_an_arm_joint_is_refused():
    with pytest.raises(TrajectoryError, match="not arm joints"):
        _tolerances({"left_knee_joint": 0.1})
