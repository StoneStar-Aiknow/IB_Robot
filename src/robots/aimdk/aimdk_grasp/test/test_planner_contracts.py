"""Planning output and scoped collision-profile regression checks."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.hand import HandMapping, hand_links
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.planner import grasp_to_palm_transform, palm_goal_from_camera_grasp, robot_config, scene_capacity
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


def test_the_right_arm_plans_on_its_own_model_with_the_left_arm_and_head_locked():
    from aimdk_grasp.planner import screen_robot_config
    from aimdk_grasp.robot_model import config_directory

    right_model = planning_model(config_directory("right"))
    right_hand = HandMapping(right_model, PRESETS, side="right")
    positions = dict.fromkeys(RIGHT_ARM + HEAD, 0.0)
    positions.update(zip(LEFT_ARM, np.linspace(-0.3, 0.3, 7), strict=True))
    feedback = Feedback(0.0, positions, np.zeros(10), right_hand=right_hand.preset("open"))
    kin = robot_config(feedback, side="right")["robot_cfg"]["kinematics"]
    assert kin["cspace"]["default_joint_position"] == [0.0] * 7
    for name, value in zip(LEFT_ARM, np.linspace(-0.3, 0.3, 7), strict=True):
        assert kin["lock_joints"][name] == pytest.approx(value)
    assert not set(RIGHT_ARM) & set(kin["lock_joints"])
    # The right hand's own (native) open pose is locked as the R_* joints.
    for name, value in right_hand.kinematic_alias(right_hand.preset("open")).items():
        assert kin["lock_joints"][name] == pytest.approx(value)
    screen = screen_robot_config(robot_config(feedback, side="right"), set(hand_links(right_model)), "right")
    screen = screen["robot_cfg"]["kinematics"]
    assert screen["collision_link_names"] and all(link.startswith("right_") for link in screen["collision_link_names"])
    # Without the right hand in the feedback the right arm cannot be planned.
    with pytest.raises(ValueError, match="lacks right hand"):
        robot_config(Feedback(0.0, positions, np.zeros(10)), side="right")


def test_a_right_arm_plan_keeps_the_right_joints_and_the_locked_left_arm():
    """cuRobo returns the planned right arm beside the locked left one (mock E2E 2026-11-03 refused it)."""
    names = LEFT_ARM[:1] + tuple(reversed(RIGHT_ARM))
    q = np.zeros((4, 8))
    q[:, 0] = -0.1
    q[:, 1:] = np.arange(7) * 0.1
    locked = {LEFT_ARM[0]: -0.1}
    trajectory = TimedTrajectory.from_curobo(result(names, q), 0.02, locked_positions=locked, arm=RIGHT_ARM)
    assert trajectory.joint_names == RIGHT_ARM
    np.testing.assert_allclose(trajectory.positions[0], (np.arange(7) * 0.1)[::-1])
    assert trajectory.reversed().joint_names == RIGHT_ARM and trajectory.trimmed().joint_names == RIGHT_ARM
    with pytest.raises(ValueError, match="Unexpected cuRobo planned joints"):
        TimedTrajectory.from_curobo(result(names, q), 0.02, locked_positions=locked)  # planned as the left arm
    with pytest.raises(ValueError, match="seven joints of one arm"):
        TimedTrajectory(np.array([0.0, 0.1]), np.zeros((2, 7)), LEFT_ARM[:6] + RIGHT_ARM[:1])


@pytest.mark.parametrize("meshes", [0, 1, 5])
def test_warmup_reserves_the_target_and_fine_leg_without_materializing_depth(meshes):
    from aimdk_grasp.scene_estimation import LazyScene

    def unexpected_materialization():
        pytest.fail("Sizing the collision cache must not build the fine mesh")

    scene = {"mesh": dict.fromkeys(range(meshes)), "cuboid": {"table": {}, "leg": {}}}
    lazy = LazyScene(unexpected_materialization)
    capacity = scene_capacity(scene, lazy)
    assert capacity["mesh"] >= max(3, meshes)
    assert capacity["cuboid"] >= len(scene["cuboid"])
    fine = {"mesh": dict.fromkeys(range(6)), "cuboid": dict.fromkeys(range(4))}
    capacity = scene_capacity(scene, fine)
    assert capacity["mesh"] >= 6 and capacity["cuboid"] >= 4
