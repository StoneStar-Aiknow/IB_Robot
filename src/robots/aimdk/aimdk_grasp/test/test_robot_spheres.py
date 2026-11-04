"""Collision spheres placed by the batched forward kinematics."""

from __future__ import annotations

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.robot_model import collision_robot_config, planning_model
from aimdk_grasp.robot_spheres import RobotSphereModel
from scipy.spatial.transform import Rotation

RELEASE = np.array([-0.5, 1.0, -0.75, 0.0, 1.57, 1.57, 0.0, 1.57, 0.0, 1.57])


@pytest.fixture(scope="module")
def model() -> RobotSphereModel:
    return RobotSphereModel()


def _origin(element) -> np.ndarray:
    matrix = np.eye(4)
    origin = element.find("origin")
    matrix[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]).as_matrix()
    matrix[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    return matrix


def _link_poses(positions: dict[str, float]) -> dict[str, np.ndarray]:
    """Joint-by-joint URDF forward kinematics from torso_link, mimic joints included."""
    urdf = planning_model()
    poses, pending = {"torso_link": np.eye(4)}, list(urdf.findall("joint"))
    while pending:
        joint = next((joint for joint in pending if joint.find("parent").get("link") in poses), None)
        if joint is None:
            return poses
        pending.remove(joint)
        pose = poses[joint.find("parent").get("link")] @ _origin(joint)
        if joint.get("type") == "revolute":
            mimic = joint.find("mimic")
            angle = (
                float(mimic.get("multiplier")) * positions[mimic.get("joint")]
                if mimic is not None
                else positions[joint.get("name")]
            )
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(
                np.array(joint.find("axis").get("xyz").split(), float) * angle
            ).as_matrix()
            pose = pose @ turn
        poses[joint.find("child").get("link")] = pose
    return poses


def test_spheres_follow_the_planning_model_for_every_state(model):
    rng = np.random.default_rng(5)
    limits = np.array([joint["bounds"] for name in LEFT_ARM for joint in model.joints if joint["name"] == name])
    q = rng.uniform(limits[:, 0], limits[:, 1], (6, 7))
    hands = rng.uniform(model.mapping.lower, model.mapping.upper, (6, 10))
    body = dict(zip(RIGHT_ARM + HEAD, rng.uniform(-0.3, 0.3, 9), strict=True))
    feedback = Feedback(0.0, {**dict.fromkeys(LEFT_ARM, 0.0), **body}, RELEASE)
    spheres, in_bounds = model.spheres(feedback, q, hand=hands)

    cover = collision_robot_config()["robot_cfg"]["kinematics"]["collision_spheres"]
    assert in_bounds.all()
    for state in range(len(q)):
        poses = _link_poses(
            {**body, **dict(zip(LEFT_ARM, q[state], strict=True)), **model.mapping.kinematic_alias(hands[state])}
        )
        expected = np.array(
            [[*(poses[link] @ [*s["center"], 1.0])[:3], s["radius"]] for link in model.links for s in cover[link]]
        )
        np.testing.assert_allclose(spheres[state], expected, atol=1e-12)
        single, _ = model.spheres(feedback, q[state : state + 1], hand=hands[state])
        np.testing.assert_allclose(single[0], spheres[state], atol=1e-12)


def test_a_state_beyond_a_joint_limit_is_flagged(model):
    feedback = Feedback(0.0, dict.fromkeys(LEFT_ARM + RIGHT_ARM + HEAD, 0.0), RELEASE)
    q = np.zeros((2, 7))
    q[1, LEFT_ARM.index("left_wrist_roll_joint")] = 1.0  # beyond the motion-control unit's 0.724
    _, in_bounds = model.spheres(feedback, q)
    assert in_bounds.tolist() == [True, False]
    with pytest.raises(ValueError, match="Nx7"):
        model.spheres(feedback, np.zeros(7))


def test_the_right_model_moves_the_right_arm_as_the_mirror_of_the_left(model):
    """At mirrored body and hand states the right model's spheres are the left model's mirrored about y = 0."""
    from aimdk_grasp.joints import ARM_MIRROR_SIGNS, HAND_SIGNS, swap_side_name
    from aimdk_grasp.robot_model import config_directory
    from scipy.spatial import cKDTree

    right = RobotSphereModel(directory=config_directory("right"))
    assert right.side == "right" and right.mapping.side == "right" and right.arm == RIGHT_ARM
    rng = np.random.default_rng(11)
    limits = np.array([joint["bounds"] for name in LEFT_ARM for joint in model.joints if joint["name"] == name])
    q = rng.uniform(limits[:, 0], limits[:, 1], (4, 7))
    other = rng.uniform(-0.3, 0.3, 7)
    head = np.array([0.2, 0.3])
    left_body = dict(zip(LEFT_ARM + RIGHT_ARM + HEAD, [0.0] * 14 + list(head), strict=True))
    left_body.update(zip(RIGHT_ARM, other, strict=True))
    right_body = dict(left_body, **dict(zip(LEFT_ARM, other * ARM_MIRROR_SIGNS, strict=True)))
    right_body.update(zip(HEAD, head * [-1.0, 1.0], strict=True))
    left_spheres, left_ok = model.spheres(Feedback(0.0, left_body, RELEASE), q)
    right_spheres, right_ok = right.spheres(
        Feedback(0.0, right_body, np.zeros(10), right_hand=RELEASE * HAND_SIGNS), q * ARM_MIRROR_SIGNS
    )
    np.testing.assert_array_equal(left_ok, right_ok)
    assert left_spheres.shape == right_spheres.shape
    # The right configuration is generated link by link from the left one (a
    # link's spheres may be listed in another order). Up to the description's
    # own left/right differences (test_mirror_side: < 8 mm).
    assert sorted(right.links) == sorted(swap_side_name(link) for link in model.links)
    # The torso cover is copied, not mirrored (mirror_side): it is checked
    # against the torso mesh itself, which is not exactly symmetric.
    np.testing.assert_array_equal(
        right.local[right.link_slices["torso_link"]], model.local[model.link_slices["torso_link"]]
    )
    for link in model.links:
        mine, theirs = model.link_slices[link], right.link_slices[swap_side_name(link)]
        assert len(mine) == len(theirs), link
        if link == "torso_link":
            continue
        for state in range(len(q)):
            mirrored = left_spheres[state, mine] * [1.0, -1.0, 1.0, 1.0]
            distance, _ = cKDTree(mirrored).query(right_spheres[state, theirs])
            assert distance.max() < 0.008, link
    left_center, left_radius = model.arm_reach()
    right_center, right_radius = right.arm_reach()
    np.testing.assert_allclose(right_center, left_center * [1.0, -1.0, 1.0], atol=1e-5)
    assert right_radius == pytest.approx(left_radius, abs=0.008)
    assert right.palm_reach()[1] == pytest.approx(model.palm_reach()[1], abs=0.008)
    with pytest.raises(ValueError, match="left_arm_reach of a right"):
        right.left_arm_reach()
    with pytest.raises(ValueError, match="cannot drive"):
        RobotSphereModel(mapping=model.mapping, directory=config_directory("right"))
