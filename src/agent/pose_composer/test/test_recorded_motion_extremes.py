"""Recorded arm ranges must agree for offline compilation and fixed execution."""

import json

import pytest
from aimdk_grasp.fixed_pose_backend import named_motion_extremes
from aimdk_robot.pose_backend.model import _recorded_extremes


@pytest.mark.parametrize("delivery", [False, True])
def test_recorded_arm_ranges_support_legacy_and_delivery_formats(tmp_path, delivery):
    joints = ["left_elbow_joint", "right_elbow_joint"]
    rows = [[-1.2, -1.1], [-0.8, -1.4], [-1.0, -1.3]]
    recorded = {"joints": joints, "positions": rows}
    if delivery:
        recorded = {
            "format": "ibrobot.fixed_motion.delivery",
            "arms": {"joints": joints, "positions_rad": rows},
            "hands": {"joints": ["left_thumb_roll"], "positions": [[100.0], [200.0], [300.0]]},
        }
    (tmp_path / "motion.json").write_text(json.dumps(recorded))
    profile = {"vendor": {"named_motion": {"trajectories": {"wave": "motion.json"}}}}
    expected = {"left_elbow_joint": [-1.2, -0.8], "right_elbow_joint": [-1.4, -1.1]}
    assert named_motion_extremes(profile, tmp_path) == expected
    assert _recorded_extremes(profile, tmp_path) == {name: tuple(bounds) for name, bounds in expected.items()}
