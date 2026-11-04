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
