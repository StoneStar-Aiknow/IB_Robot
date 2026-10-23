"""Production-model kinematic agreement: compiled MJCF vs production URDF.

Review finding 10: the shipped SO-101 profile's MuJoCo embodiment must agree
with the production URDF it serves alongside — the base→gripper transform at
several joint configurations under a nonzero spawn yaw, and the advertised
camera optical TF vs the MJCF camera frame the plugin actually renders from.
Everything here uses the production artifacts (shipped profile, MJCF template,
URDF xacro), not fixtures.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

_PACKAGE = Path(__file__).resolve().parents[1]
_PROFILE = _PACKAGE / "profiles" / "so101_single_arm.yaml"

_SPAWN = (0.12, -0.07, 0.03, 0.7)  # nonzero position AND yaw
_ARM_JOINTS = ["1", "2", "3", "4", "5"]
_CONFIGS = [
    ("rest", [0.0, -1.5854, 1.5708, 0.7854, 0.0]),
    ("mild", [0.2, 0.2, 0.2, 0.2, 0.2]),
    ("offset", [0.5, -1.1, 1.5708, 0.7854, 0.0]),
]


def _rpy_to_matrix(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k)


def _transform(xyz, rotation):
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = xyz
    return transform


def _urdf_chain_transform(urdf_xml: str, root: str, tip: str, joint_values: dict) -> np.ndarray:
    """root→tip transform of a URDF chain (fixed + revolute joints)."""
    tree = ET.fromstring(urdf_xml)
    # key by the CHILD link: every link has exactly one parent joint in a URDF
    # tree, while several joints (e.g. camera mounts) can share a parent link
    joint_of_child = {joint.find("child").get("link"): joint for joint in tree.findall("joint")}
    chain = []
    link = tip
    while link != root:
        joint = joint_of_child.get(link)
        assert joint is not None, f"URDF: no joint chain from {root} to {tip} (stuck at {link})"
        chain.append(joint)
        link = joint.find("parent").get("link")
    transform = np.eye(4)
    for joint in reversed(chain):  # root → tip order
        origin = joint.find("origin")
        xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0").split() if origin is not None else [0, 0, 0])])
        rpy = [float(v) for v in (origin.get("rpy", "0 0 0").split() if origin is not None else [0, 0, 0])]
        transform = transform @ _transform(xyz, _rpy_to_matrix(*rpy))
        if joint.get("type") == "revolute":
            axis_el = joint.find("axis")
            axis = [float(v) for v in (axis_el.get("xyz", "0 0 1").split() if axis_el is not None else [0, 0, 1])]
            angle = float(joint_values.get(joint.get("name"), 0.0))
            transform = transform @ _transform(np.zeros(3), _axis_angle(axis, angle))
    return transform


def _urdf_joint_transforms(tf_xml: str, camera: str, urdf: str = "", joint_values: dict | None = None):
    """((xyz, R) mount, (xyz, R) optical, mount parent link) from camera_tf_urdf.

    When ``urdf`` is given, a gripper-parented mount is composed through the
    arm chain at ``joint_values`` and returned as a world-anchored transform.
    """
    tree = ET.fromstring(f"<robot>{tf_xml}</robot>")
    out = {}
    parent_link = None
    for joint in tree.findall("joint"):
        name = joint.get("name", "")
        if name in (f"{camera}_camera_joint", f"{camera}_optical_joint"):
            origin = joint.find("origin")
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
            out[name] = (xyz, _rpy_to_matrix(*rpy))
            if name.endswith("_camera_joint"):
                parent_link = joint.find("parent").get("link")
    assert parent_link is not None, f"camera_tf_urdf carries no mount joint for {camera}"
    return out[f"{camera}_camera_joint"], out[f"{camera}_optical_joint"], parent_link


def _production_models(tmp_path):
    from robot_runtime.launch_support import render_mujoco_description
    from robot_runtime.mujoco import camera_tf_urdf, render_robot_mjcf
    from robot_runtime.profile import load_profile

    profile = load_profile(_PROFILE)
    peripherals = profile.get("peripherals") or []
    mjcf_path = render_robot_mjcf(profile, spawn_pose=_SPAWN, peripherals=peripherals, out_dir=tmp_path)
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    urdf = render_mujoco_description(profile, _PROFILE, peripherals, spawn_pose=_SPAWN, scene="")
    return profile, peripherals, model, urdf, camera_tf_urdf(peripherals, spawn_pose=_SPAWN)


def _mjcf_relative_transform(model, data, base_body: str, tip_body: str):
    mujoco.mj_forward(model, data)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body)
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tip_body)
    assert base >= 0 and tip >= 0, "production MJCF must carry the base and gripper bodies"
    base_rotation = data.xmat[base].reshape(3, 3)
    tip_rotation = data.xmat[tip].reshape(3, 3)
    position = base_rotation.T @ (data.xpos[tip] - data.xpos[base])
    return position, base_rotation.T @ tip_rotation


@pytest.mark.parametrize(("label", "configuration"), _CONFIGS, ids=[c[0] for c in _CONFIGS])
def test_production_mjcf_matches_the_production_urdf_base_to_gripper(tmp_path, label, configuration):
    profile, _peripherals, model, urdf, _tf = _production_models(tmp_path)
    data = mujoco.MjData(model)
    for joint, value in zip(_ARM_JOINTS, configuration, strict=True):
        address = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
        data.qpos[address] = value

    mjcf_position, mjcf_rotation = _mjcf_relative_transform(model, data, "base", "gripper")
    urdf_transform = _urdf_chain_transform(urdf, "base", "gripper", dict(zip(_ARM_JOINTS, configuration, strict=True)))
    urdf_position = urdf_transform[:3, 3]
    urdf_rotation = urdf_transform[:3, :3]

    assert mjcf_position == pytest.approx(urdf_position, abs=2e-3), f"{label}: base→gripper position"
    relative = urdf_rotation.T @ mjcf_rotation
    angle = math.acos(np.clip((np.trace(relative) - 1) / 2, -1, 1))
    assert angle < 2e-2, f"{label}: base→gripper rotation differs by {math.degrees(angle):.3f}°"


@pytest.mark.parametrize("camera", ["top", "wrist", "front"])
def test_production_camera_optical_tf_matches_the_mjcf_camera_frame(tmp_path, camera):
    """The TF optical frame coincides with the MJCF camera the plugin renders from.

    The URDF camera tree is rooted at the robot base (the spawn transform maps
    it into the true world): world/base-parented mounts compose directly under
    the spawn, gripper-parented mounts ride the arm chain (evaluated at the
    rest configuration, the same qpos the MJCF is set to).
    """
    _profile, _peripherals, model, urdf, tf_xml = _production_models(tmp_path)
    data = mujoco.MjData(model)
    rest = dict(zip(_ARM_JOINTS + ["6"], [0.0, -1.5854, 1.5708, 0.7854, 0.0, 0.0], strict=True))
    for joint, value in rest.items():
        address = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]
        data.qpos[address] = value
    mujoco.mj_forward(model, data)

    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"{camera}_camera")
    assert cid >= 0, f"production MJCF must carry the {camera} camera"
    cam_pos = np.array(data.cam_xpos[cid])
    cam_rot = data.cam_xmat[cid].reshape(3, 3)
    (mount_xyz, mount_rot), (opt_xyz, opt_rot), mount_parent = _urdf_joint_transforms(tf_xml, camera, urdf, rest)

    spawn_transform = _transform(np.array(_SPAWN[:3]), _rpy_to_matrix(0.0, 0.0, _SPAWN[3]))
    parent_transform = spawn_transform
    if mount_parent not in ("world", "base"):
        parent_transform = parent_transform @ _urdf_chain_transform(urdf, "base", mount_parent, rest)
    optical = parent_transform @ _transform(mount_xyz, mount_rot) @ _transform(opt_xyz, opt_rot)
    optical_pos = optical[:3, 3]
    optical_rot = optical[:3, :3]

    assert optical_pos == pytest.approx(cam_pos, abs=2e-3), f"{camera}: optical position"
    # optical +X == camera +X; optical +Y (image down) == camera -Y; optical +Z (view) == camera -Z
    assert optical_rot @ np.array([1, 0, 0]) == pytest.approx(cam_rot @ np.array([1, 0, 0]), abs=1e-6)
    assert optical_rot @ np.array([0, 1, 0]) == pytest.approx(-(cam_rot @ np.array([0, 1, 0])), abs=1e-6)
    assert optical_rot @ np.array([0, 0, 1]) == pytest.approx(-(cam_rot @ np.array([0, 0, 1])), abs=1e-6)
