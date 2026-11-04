"""The right arm's planning model and configuration mirror the left arm's."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import yaml
from aimdk_grasp.joints import ARM_MIRROR_SIGNS, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.mirror_side import PLANNING_FILES, generate
from aimdk_grasp.planning_model import build_planning_model
from aimdk_grasp.robot_model import collision_robot_config, config_directory, hand_mount, planning_model_options
from scipy.spatial.transform import Rotation

#: Torso-sagittal mirror and the link-frame reflections the right hand is built with.
MIRROR = np.diag([1.0, -1.0, 1.0, 1.0])
FLIP_X = np.diag([-1.0, 1.0, 1.0, 1.0])
FLIP_Y = np.diag([1.0, -1.0, 1.0, 1.0])


def _reflection(link: str) -> np.ndarray:
    return FLIP_X if link == "R_palm" else FLIP_Y


def _origin(element: ET.Element) -> np.ndarray:
    matrix = np.eye(4)
    origin = element.find("origin")
    if origin is not None:
        matrix[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in origin.get("rpy", "0 0 0").split()]).as_matrix()
        matrix[:3, 3] = [float(v) for v in origin.get("xyz", "0 0 0").split()]
    return matrix


def _poses(robot: ET.Element, positions: dict[str, float]) -> dict[str, np.ndarray]:
    """Torso-frame pose of every link below ``torso_link``; mimic joints follow their leader."""
    poses = {"torso_link": np.eye(4)}
    pending = list(robot.findall("joint"))
    while pending:
        joint = next((joint for joint in pending if joint.find("parent").get("link") in poses), None)
        if joint is None:
            break
        pending.remove(joint)
        pose = poses[joint.find("parent").get("link")] @ _origin(joint)
        if joint.get("type") == "revolute":
            mimic = joint.find("mimic")
            if mimic is None:
                angle = positions.get(joint.get("name"), 0.0)
            else:
                leader = positions.get(mimic.get("joint"), 0.0)
                angle = float(mimic.get("multiplier", 1)) * leader + float(mimic.get("offset", 0))
            axis = np.array([float(value) for value in joint.find("axis").get("xyz").split()])
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            pose = pose @ turn
        poses[joint.find("child").get("link")] = pose
    return poses


def _limits(robot: ET.Element, name: str) -> tuple[float, float]:
    limit = robot.find(f"joint[@name='{name}']/limit")
    return float(limit.get("lower")), float(limit.get("upper"))


@pytest.fixture(scope="module")
def options() -> dict:
    config = yaml.safe_load((config_directory() / "robot.yml").read_text(encoding="utf-8"))
    return planning_model_options(config["planning_model"])


@pytest.fixture(scope="module")
def models(options) -> tuple[ET.Element, ET.Element]:
    return (
        build_planning_model(**options).getroot(),
        build_planning_model(**{**options, "side": "right"}).getroot(),
    )


def _hand_joints(model: ET.Element) -> list[str]:
    return [
        joint.get("name")
        for joint in model.findall("joint")
        if joint.get("name").startswith("R_") and joint.get("type") == "revolute" and joint.find("mimic") is None
    ]


def test_the_right_hand_is_the_left_hand_mirrored_in_its_wrist(models):
    left, right = models
    hand = _hand_joints(left)
    assert hand == _hand_joints(right)
    assert all(_limits(left, name) == _limits(right, name) for name in hand)
    rng = np.random.default_rng(0)
    for _ in range(20):
        positions = {name: rng.uniform(*_limits(left, name)) for name in hand}
        on_left, on_right = _poses(left, positions), _poses(right, positions)
        left_wrist, right_wrist = on_left["left_wrist_roll_link"], on_right["right_wrist_roll_link"]
        for link in (link.get("name") for link in right.findall("link") if link.get("name").startswith("R_")):
            expected = FLIP_Y @ np.linalg.inv(left_wrist) @ on_left[link] @ _reflection(link)
            np.testing.assert_allclose(np.linalg.inv(right_wrist) @ on_right[link], expected, atol=1e-9)


def test_at_mirrored_arm_positions_the_right_hand_is_the_left_hand_mirrored(models):
    """Up to the description's own left/right differences (about 0.3 deg on three arm joints)."""
    left, right = models
    hand = _hand_joints(left)
    rng = np.random.default_rng(1)
    errors = []
    for _ in range(50):
        arm = {name: rng.uniform(*_limits(left, name)) for name in LEFT_ARM}
        fingers = {name: rng.uniform(*_limits(left, name)) for name in hand}
        mirrored = {
            name: sign * arm[left_name]
            for left_name, name, sign in zip(LEFT_ARM, RIGHT_ARM, ARM_MIRROR_SIGNS, strict=True)
        }
        on_left, on_right = _poses(left, {**arm, **fingers}), _poses(right, {**mirrored, **fingers})
        for link in ("R_palm", "R_index_dip", "R_thumb_dip"):
            error = np.linalg.inv(MIRROR @ on_left[link] @ _reflection(link)) @ on_right[link]
            errors.append(np.linalg.norm(error[:3, 3]))
            assert np.linalg.det(on_right[link][:3, :3]) == pytest.approx(1.0)
    assert max(errors) < 0.008


def test_the_mirrored_palm_keeps_the_grasp_frame_a_sweep_volume_symmetry():
    """``G = P inv(B)``: mirroring the palm by ``x`` mirrors G by its ``y``, for the deployed ``B``.

    ``B`` is the gripper description's base_rotation (graspgenx_omnihand_left_x2,
    revision 2026-10-13); its form, a rotation about the palm's ``y``, is what matters.
    """
    angle = np.arctan2(0.435312, 0.90028)
    base = np.array([[0.0, np.cos(angle), -np.sin(angle)], [-1.0, 0.0, 0.0], [0.0, np.sin(angle), np.cos(angle)]])
    np.testing.assert_allclose(base @ FLIP_X[:3, :3] @ base.T, np.diag([1.0, -1.0, 1.0]), atol=1e-12)
    assert not np.allclose(base @ FLIP_Y[:3, :3] @ base.T, np.diag(np.diag(base @ FLIP_Y[:3, :3] @ base.T)))


def test_the_committed_right_configuration_is_the_generated_one():
    generated = generate(config_directory())
    for name in PLANNING_FILES:
        text = (config_directory("right") / name).read_text(encoding="utf-8")
        committed = yaml.safe_load(text) if name.endswith(".yml") else json.loads(text)
        assert committed == generated[name], name


def test_the_committed_right_gravity_and_leg_are_the_generated_ones():
    from aimdk_grasp.mirror_side import GRAVITY_FILE, LEG_FILE

    pytest.importorskip("trimesh")
    generated = generate(config_directory())
    for name in (GRAVITY_FILE, LEG_FILE):
        committed = json.loads((config_directory("right") / name).read_text(encoding="utf-8"))
        assert committed == json.loads(json.dumps(generated[name])), name


def test_the_right_configuration_plans_the_right_arm_and_loads_its_covers():
    left = collision_robot_config()["robot_cfg"]["kinematics"]
    right = collision_robot_config(config_directory("right"))["robot_cfg"]["kinematics"]
    assert right["cspace"]["joint_names"] == list(RIGHT_ARM)
    assert set(LEFT_ARM) <= set(right["lock_joints"]) and not set(RIGHT_ARM) & set(right["lock_joints"])
    assert right["collision_link_names"] == left["collision_link_names"]
    assert {link: len(spheres) for link, spheres in right["collision_spheres"].items()} == {
        link: len(spheres) for link, spheres in left["collision_spheres"].items()
    }
    pairs = {tuple(sorted((a, b))) for a, others in right["self_collision_ignore"].items() for b in others}
    wrists = {
        name for pair in pairs if any(n.startswith("R_") for n in pair) for name in pair if not name.startswith("R_")
    }
    assert wrists == set(hand_mount("right"))


def test_hand_spheres_are_reflected_like_their_links():
    left = collision_robot_config()["robot_cfg"]["kinematics"]["collision_spheres"]
    right = collision_robot_config(config_directory("right"))["robot_cfg"]["kinematics"]["collision_spheres"]
    for link, spheres in left.items():
        reflection = _reflection(link)[:3, :3] if link.startswith("R_") else np.eye(3)
        for a, b in zip(spheres, right[link], strict=True):
            np.testing.assert_allclose(reflection @ np.asarray(a["center"]), b["center"], atol=0)
            assert a["radius"] == b["radius"]
