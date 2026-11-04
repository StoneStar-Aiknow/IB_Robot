"""The X2 planning model generated from the official description."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml
from aimdk_grasp.joints import HAND_ALIAS_SIGNS, LEFT_ARM
from aimdk_grasp.planning_model import (
    HAND_URDF,
    PALM_TOOL_RPY,
    build_planning_model,
    description_share,
)
from aimdk_grasp.robot_model import config_directory, left_arm_limits, planning_model_options
from scipy.spatial.transform import Rotation

WRIST = "left_wrist_roll_link"


def _origin(element: ET.Element) -> np.ndarray:
    matrix = np.eye(4)
    origin = element.find("origin")
    if origin is not None:
        rpy = [float(value) for value in origin.get("rpy", "0 0 0").split()]
        matrix[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        matrix[:3, 3] = [float(value) for value in origin.get("xyz", "0 0 0").split()]
    return matrix


def _poses(robot: ET.Element, positions: dict[str, float], base: str = WRIST) -> dict[str, np.ndarray]:
    """Every link below ``base`` in the frame of ``base``."""
    poses = {base: np.eye(4)}
    pending = list(robot.findall("joint"))
    while pending:
        joint = next((joint for joint in pending if joint.find("parent").get("link") in poses), None)
        if joint is None:
            break
        pending.remove(joint)
        pose = poses[joint.find("parent").get("link")] @ _origin(joint)
        if joint.get("type") == "revolute":
            axis = np.array([float(value) for value in joint.find("axis").get("xyz").split()])
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec(axis * positions.get(joint.get("name"), 0.0)).as_matrix()
            pose = pose @ turn
        poses[joint.find("child").get("link")] = pose
    return poses


@pytest.fixture(scope="module")
def official_hand() -> ET.Element:
    return ET.parse(description_share() / HAND_URDF).getroot()


@pytest.fixture(scope="module")
def options() -> dict:
    config = yaml.safe_load((config_directory() / "robot.yml").read_text(encoding="utf-8"))
    return planning_model_options(config["planning_model"])


@pytest.fixture(scope="module")
def model(options) -> ET.Element:
    return build_planning_model(**options).getroot()


def test_the_model_is_one_tree_of_upper_body_and_aliased_hand_links(model):
    links = [link.get("name") for link in model.findall("link")]
    children = [joint.find("child").get("link") for joint in model.findall("joint")]
    parents = {joint.find("parent").get("link") for joint in model.findall("joint")}

    assert len(set(links)) == len(links)
    assert len(set(children)) == len(children)
    assert parents | set(children) <= set(links)
    assert len(set(links) - set(children)) == 1
    assert not [name for name in links if name.startswith("L_") or "knee" in name or "camera" in name]
    assert {"torso_link", "head_yaw_link", "right_wrist_roll_link", "R_palm", "R_thumb_dip"} <= set(links)
    mimics = {joint.get("name"): joint.find("mimic").attrib for joint in model.findall("joint[mimic]")}
    assert mimics == {"R_thumb_pip_joint": {"joint": "R_thumb_mcp_joint", "multiplier": "1.33", "offset": "0.0"}}
    meshes = [Path(mesh.get("filename")) for mesh in model.iter("mesh")]
    assert meshes and all(mesh.is_absolute() and mesh.is_file() for mesh in meshes)


@pytest.fixture(scope="module")
def official_mounts(options) -> ET.Element:
    return build_planning_model(**{**options, "palm_mount": "official", "thumb_mount": "official"}).getroot()


@pytest.mark.model_bundle("graspgenx_omnihand_left_x2")
def test_the_hand_on_the_palm_is_the_hand_the_grasp_model_proposes_for(model):
    """Every hand link sits on R_palm where the GraspGen-X gripper description has it (the AimDK SDK hand).

    Grasp poses are R_palm goals, and the hand collision spheres were fitted to
    that hand: a link elsewhere is a hand the grasps were not made for.
    """
    gripper = ET.parse(
        Path(__file__).resolve().parents[5]
        / "models/graspgenx_omnihand_left_x2/assets/x_grippers/omnihand_left_x2/gripper.urdf"
    ).getroot()
    generated, proposed = _poses(model, {}, base="R_palm"), _poses(gripper, {}, base="R_palm")
    shared = set(generated) & set(proposed) - {"R_palm"}
    assert len(shared) >= 16
    for link in sorted(shared):
        np.testing.assert_allclose(generated[link], proposed[link], atol=1e-6, err_msg=link)


def test_on_the_sdk_mount_the_palm_moves_with_its_fingers(model, official_mounts):
    """The SDK palm sits 3 mm closer to the wrist; what is on the palm is not re-placed one by one."""
    sdk, official = _poses(model, {}), _poses(official_mounts, {})
    np.testing.assert_allclose(sdk["R_palm"][:3, :3], official["R_palm"][:3, :3], atol=1e-12)
    assert np.linalg.norm(sdk["R_palm"][:3, 3] - official["R_palm"][:3, 3]) == pytest.approx(0.0032, abs=1e-4)
    on_palm = (
        np.linalg.inv(sdk["R_palm"]) @ sdk["R_middle_abad"],
        np.linalg.inv(official["R_palm"]) @ official["R_middle_abad"],
    )
    assert not np.allclose(*on_palm)  # the fingers take the SDK mounts, 0.53 degrees about the wrist axis
    with pytest.raises(ValueError, match="palm_mount"):
        build_planning_model(arm_limits={}, palm_mount="sideways")


def test_the_palm_keeps_the_pipeline_orientation_without_moving_the_hand(official_mounts, official_hand):
    model = official_mounts
    generated = _poses(model, {})
    official = _poses(official_hand, {})

    np.testing.assert_allclose(
        generated["R_palm"][:3, :3], Rotation.from_euler("xyz", PALM_TOOL_RPY).as_matrix(), atol=1e-9
    )
    np.testing.assert_allclose(generated["R_palm"][:3, 3], official["L_palm"][:3, 3], atol=1e-9)
    for name in ("visual", "collision"):
        palm_geometry = generated["R_palm"] @ _origin(model.find(f"link[@name='R_palm']/{name}"))
        official_geometry = official["L_palm"] @ _origin(official_hand.find(f"link[@name='L_palm']/{name}"))
        np.testing.assert_allclose(palm_geometry, official_geometry, atol=1e-9)
    np.testing.assert_allclose(generated["R_index_pip"], official["L_index_pip"], atol=1e-9)


def test_an_alias_position_is_the_signed_native_position(official_hand, official_mounts):
    model = official_mounts
    rng = np.random.default_rng(7)
    native = {f"L_{part}_joint": rng.uniform(0.0, 0.6) for part in HAND_ALIAS_SIGNS}
    native["L_thumb_pip_joint"] = 1.33 * native["L_thumb_mcp_joint"]
    alias = {f"R_{part}_joint": sign * native[f"L_{part}_joint"] for part, sign in HAND_ALIAS_SIGNS.items()}

    generated = _poses(model, alias)
    official = _poses(official_hand, native)

    for part in ("thumb_dip", "index_dip", "middle_dip", "ring_dip", "pinky_dip", "pinky_abad"):
        np.testing.assert_allclose(generated[f"R_{part}"], official[f"L_{part}"], atol=1e-9)
    for part, sign in HAND_ALIAS_SIGNS.items():
        native_limit = official_hand.find(f"joint[@name='L_{part}_joint']/limit")
        alias_limit = model.find(f"joint[@name='R_{part}_joint']/limit")
        bounds = sorted(sign * float(native_limit.get(end)) for end in ("lower", "upper"))
        assert [float(alias_limit.get("lower")), float(alias_limit.get("upper"))] == pytest.approx(bounds)


def test_arm_limits_only_narrow_the_urdf_range(model, options):
    official = ET.parse(description_share() / "urdf/x2_ultra.urdf").getroot()
    assert set(options["arm_limits"]) == set(LEFT_ARM)
    for name, (lower, upper) in options["arm_limits"].items():
        source = official.find(f"joint[@name='{name}']/limit")
        limit = model.find(f"joint[@name='{name}']/limit")
        assert float(limit.get("lower")) == pytest.approx(max(lower, float(source.get("lower"))))
        assert float(limit.get("upper")) == pytest.approx(min(upper, float(source.get("upper"))))

    with pytest.raises(ValueError, match="do not overlap"):
        build_planning_model(arm_limits={"left_elbow_joint": (1.0, 2.0)})
    with pytest.raises(ValueError, match="unknown joint"):
        build_planning_model(arm_limits={"left_knee": (0.0, 1.0)})
    with pytest.raises(ValueError, match="thumb_mount"):
        build_planning_model(arm_limits={}, thumb_mount="mirrored")


def test_left_arm_limits_are_read_and_checked_from_the_robot_configuration():
    full = {name: [-1.0, 1.0] for name in LEFT_ARM}
    assert left_arm_limits({"arm_limits": {**full, "right_elbow_joint": [-2.0, 0.0]}}) == {
        name: (-1.0, 1.0) for name in LEFT_ARM
    }
    with pytest.raises(ValueError, match="no arm_limits"):
        left_arm_limits({})
    with pytest.raises(ValueError, match="left_elbow_joint"):
        left_arm_limits({"arm_limits": {name: value for name, value in full.items() if name != "left_elbow_joint"}})
    with pytest.raises(ValueError, match="left_wrist_roll_joint"):
        left_arm_limits({"arm_limits": {**full, "left_wrist_roll_joint": [0.7, -1.5]}})
    with pytest.raises(ValueError, match="left_wrist_pitch_joint"):
        left_arm_limits({"arm_limits": {**full, "left_wrist_pitch_joint": [float("nan"), 0.5]}})
