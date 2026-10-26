"""Planning output and scoped collision-profile regression checks."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.hand import HandMapping, hand_links
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.planner import grasp_to_palm_transform, palm_goal_from_camera_grasp, robot_config
from aimdk_grasp.robot_model import planning_model
from aimdk_grasp.trajectory import TimedTrajectory
from test_hand import PRESETS

#: The deployed OmniHand gripper description's palm mount and closing sweep (models/graspgenx_omnihand_left_x2,
#: assets/x_grippers/omnihand_left_x2/config.json), copied so the tests need no model bundle.
GRIPPER = {
    "base_rotation": [
        [0.0, 0.9002797228291237, -0.43531186597968585, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.43531186597968585, 0.9002797228291238, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "sweep_volume": {
        "extents": [0.0737, 0.071, 0.04],
        "offset": [0.0, 0.0, 0.119],
        "extents2": [0.0368, 0.071, 0.03],
        "offset2": [0.0, 0.0, 0.099],
    },
}


class Position:
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def result(names, q):
    plan = SimpleNamespace(joint_names=names, position=Position(q))
    return SimpleNamespace(success=np.array([True]), get_interpolated_plan=lambda: plan)


def test_augmented_trajectory_preserves_time_and_joint_order():
    names = RIGHT_ARM[:1] + tuple(reversed(LEFT_ARM))
    q = np.zeros((1, 1, 5, 8))
    q[..., 0] = 0.2
    q[..., 1:] = np.arange(7)
    trajectory = TimedTrajectory.from_curobo(result(names, q), 0.02, locked_positions={RIGHT_ARM[0]: 0.2})
    np.testing.assert_allclose(trajectory.times, [0, 0.02, 0.04, 0.06, 0.08])
    np.testing.assert_allclose(trajectory.positions[0], np.arange(7)[::-1])


@pytest.mark.parametrize("locked,value", [({}, 0.2), ({RIGHT_ARM[0]: 0.2}, 0.3)])
def test_augmented_trajectory_rejects_unknown_or_moving_extra_joint(locked, value):
    q = np.zeros((3, 8))
    q[:, 7] = 0.2
    q[-1, 7] = value
    with pytest.raises(ValueError, match="locked joint"):
        TimedTrajectory.from_curobo(result(LEFT_ARM + RIGHT_ARM[:1], q), 0.02, locked_positions=locked)


def test_default_reference_exclusions_are_scoped_to_hand_mount():
    feedback = Feedback(
        0.0, dict.fromkeys(LEFT_ARM + RIGHT_ARM + HEAD, 0.0), HandMapping(planning_model(), PRESETS).preset("open")
    )
    strict = robot_config(feedback, reference_hand_collisions=False)["robot_cfg"]["kinematics"]
    reference = robot_config(feedback, reference_hand_collisions=True)["robot_cfg"]["kinematics"]
    assert reference == robot_config(feedback)["robot_cfg"]["kinematics"]
    assert strict["collision_spheres"] == reference["collision_spheres"]
    assert strict["lock_joints"] == reference["lock_joints"]
    hand = set(hand_links(planning_model()))
    mount = {"left_wrist_yaw_link", "left_wrist_pitch_link", "left_wrist_roll_link"}
    additions = []
    for a, others in reference["self_collision_ignore"].items():
        for b in set(others) - set(strict["self_collision_ignore"].get(a, [])):
            assert (a in hand and b in hand | mount) or (b in hand and a in mount)
            additions.append((a, b))
    assert ("R_palm", "left_wrist_yaw_link") in additions
    assert "left_wrist_roll_link" not in reference["self_collision_ignore"]["right_wrist_roll_link"]
    assert "R_palm" not in reference["self_collision_ignore"]["torso_link"]
    assert "left_elbow_link" not in reference["self_collision_ignore"]["torso_link"]
    assert "right_elbow_link" not in reference["self_collision_ignore"]["torso_link"]
    assert "left_shoulder_yaw_link" in reference["self_collision_ignore"]["torso_link"]
    assert "right_shoulder_yaw_link" in reference["self_collision_ignore"]["torso_link"]
    assert reference["cspace"]["position_limit_clip"] == 0.0


def test_graspgen_base_is_converted_to_r_palm_from_the_gripper_description():
    base = grasp_to_palm_transform(GRIPPER)
    np.testing.assert_allclose(base[3], [0, 0, 0, 1])
    np.testing.assert_allclose(base[:3, :3].T @ base[:3, :3], np.eye(3), atol=1e-10)
    np.testing.assert_allclose(palm_goal_from_camera_grasp(np.eye(4), np.eye(4), base), base)


def test_graspgen_transform_rejects_nonrigid_inputs():
    with pytest.raises(ValueError, match="proper"):
        palm_goal_from_camera_grasp(np.diag([2, 1, 1, 1]), np.eye(4), np.eye(4))
