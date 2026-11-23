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
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

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
    from robot_runtime.profile import load_profile
    from robot_runtime_sim.mujoco import camera_tf_urdf, render_robot_mjcf
    from robot_runtime_sim.mujoco_embodiment import render_mujoco_description

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


def test_gripper_follow_viewer_camera_tracks_the_gripper(tmp_path):
    """The viewer-only chase camera keeps a constant world offset from the gripper.

    It must follow the gripper's position, keep its orientation when the
    wrist rolls (mode="track"), sit behind and above the gripper relative to
    the robot heading. Whether it is published is checked against the
    launched plugin parameters below.
    """
    _profile, _peripherals, model, _urdf, _tf = _production_models(tmp_path)
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "gripper_follow")
    gripper = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    assert cid >= 0 and model.cam_bodyid[cid] == gripper

    data = mujoco.MjData(model)
    offsets, rotations = [], []
    for _label, configuration in _CONFIGS:
        for joint, value in zip(_ARM_JOINTS, configuration, strict=True):
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = value
        mujoco.mj_forward(model, data)
        offsets.append(np.array(data.cam_xpos[cid]) - data.xpos[gripper])
        rotations.append(data.cam_xmat[cid].reshape(3, 3).copy())
    for offset, rotation in zip(offsets[1:], rotations[1:], strict=True):
        assert offset == pytest.approx(offsets[0], abs=1e-9)
        assert rotation == pytest.approx(rotations[0], abs=1e-9)

    # in the robot heading frame: behind-right (+Y is behind, the robot reaches
    # along -Y) and above, so the forearm stays out of the line of sight
    heading = _rpy_to_matrix(0.0, 0.0, _SPAWN[3])
    local_offset = heading.T @ offsets[0]
    assert local_offset == pytest.approx([0.2628, 0.2117, 0.3705], abs=2e-3)
    view = heading.T @ (rotations[0] @ np.array([0.0, 0.0, -1.0]))
    assert view[1] < 0 and view[2] < 0, "looks forward and down at the gripper"


def test_launched_camera_plugin_publishes_only_the_declared_cameras(tmp_path):
    """The pick_banana launch publishes the profile's cameras and no viewer camera.

    mujoco_ros2_control 0.1.2's CameraPlugin walks every camera of the model
    it loads and streams any camera whose ``policy`` parameter is unset, so
    this applies that rule to the composed model the launch names and the
    params file the launch writes.
    """
    from robot_runtime.profile import load_profile
    from robot_runtime_sim.mujoco import CAMERA_PLUGIN_INSTANCE
    from robot_runtime_sim.mujoco_embodiment import (
        MUJOCO_SIMULATION_NODE,
        mujoco_embodiment_actions,
        render_mujoco_description,
    )
    from sim_models.scene_compiler import mujoco_model_camera_names

    profile = load_profile(_PROFILE)
    peripherals = profile.get("peripherals") or []
    description = render_mujoco_description(profile, _PROFILE, peripherals, spawn_pose=_SPAWN, scene="pick_banana")
    actions = mujoco_embodiment_actions(
        profile, _PROFILE, robot_description=description, mujoco_scene="pick_banana", spawn_pose=_SPAWN
    )
    # launch_ros objects are read by attribute: this package does not depend on launch_ros
    simulator = next(
        action for action in actions if getattr(action, "_Node__node_executable", None) == "ros2_control_node"
    )
    trees = [
        yaml.safe_load(Path("".join(getattr(part, "text", None) or str(part) for part in entry.param_file)).read_text())
        for entry in simulator._Node__parameters
        if hasattr(entry, "param_file")
    ]
    plugin = next(tree for tree in trees if MUJOCO_SIMULATION_NODE in tree)[MUJOCO_SIMULATION_NODE]["ros__parameters"]
    instance = plugin["mujoco_plugins"][CAMERA_PLUGIN_INSTANCE]

    model_path = re.search(r'<param name="mujoco_model">\s*([^<]*?)\s*</param>', description).group(1)
    cameras = mujoco_model_camera_names(model_path)
    assert {"side_view", "gripper_follow"} <= set(cameras), "the composed model carries the viewer cameras"
    policy = {name: (instance.get(name) or {}).get("policy", "streaming") for name in cameras}
    declared = {f"{p['name']}_camera" for p in peripherals if p.get("type") == "camera" and p.get("enabled", True)}
    assert {name for name, value in policy.items() if value == "streaming"} == declared
    assert policy["side_view"] == policy["gripper_follow"] == "disabled"


# Renders run in a child process: mujoco picks its GL backend once, at import,
# from MUJOCO_GL, and this process has already imported it. Exit 77 = no GL
# context (probed apart from the Renderer, whose own errors fail the test).
_RENDER_COMPARE = """
import sys
import mujoco
import numpy as np

def load(path):
    return mujoco.MjModel.from_binary_path(path) if path.endswith(".mjb") else mujoco.MjModel.from_xml_path(path)


with_path, without_path, joints, poses = sys.argv[1], sys.argv[2], sys.argv[3].split(","), sys.argv[4:]
models = [load(path) for path in (with_path, without_path)]
# Only a missing offscreen GL context may skip (exit 77); it is probed on its
# own, so any Renderer construction error below fails with its traceback.
try:
    probe = mujoco.GLContext(1, 1)
    probe.make_current()
    probe.free()
except Exception as exc:
    print(f"{type(exc).__name__}: {exc}")
    sys.exit(77)
# the recorded cameras' own 640x480 (also MuJoCo's default offscreen buffer)
renderers = [mujoco.Renderer(model, 480, 640) for model in models]
for pose in poses:
    frames = {}
    for model, renderer in zip(models, renderers):
        data = mujoco.MjData(model)
        for joint, value in zip(joints, map(float, pose.split(","))):
            data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)]] = value
        mujoco.mj_forward(model, data)
        for camera in ("top_camera", "wrist_camera"):
            renderer.update_scene(data, camera=camera)
            frames.setdefault(camera, []).append(renderer.render().copy())
    for camera, (a, b) in frames.items():
        if not np.array_equal(a, b):
            print(f"{camera} differs at pose {pose}")
            sys.exit(1)
"""


_COMPARE_POSES = [
    "0,0,0,0,0,0",  # qpos0, the simulator's start pose
    "0,-1.5854,1.5708,0.7854,0,0",  # the layout rest pose
    "0.05,0.5,-0.2,0.9,0,0.6",  # gripper over the banana area
    "-0.616,0.615,-0.371,0.664,0,0.3",  # gripper over the plate
]


def _render_compare(with_path: Path, without_path: Path) -> None:
    import os
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _RENDER_COMPARE,
            str(with_path),
            str(without_path),
            ",".join(_ARM_JOINTS + ["6"]),
            *_COMPARE_POSES,
        ],
        env={**os.environ, "MUJOCO_GL": "egl"},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if result.returncode == 77:
        pytest.skip(f"no offscreen GL context: {result.stdout.strip()}")
    assert result.returncode == 0, result.stdout + result.stderr


def test_viewer_camera_leaves_the_recorded_camera_images_pixel_identical(tmp_path):
    """Removing the viewer-only gripper_follow element changes no recorded pixel."""
    import re

    from robot_runtime.profile import load_profile
    from robot_runtime_sim.mujoco import render_robot_mjcf

    profile = load_profile(_PROFILE)
    mjcf_path = Path(
        render_robot_mjcf(profile, spawn_pose=_SPAWN, peripherals=profile.get("peripherals") or [], out_dir=tmp_path)
    )
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    stripped, count = re.subn(r"<camera\s+name=\"gripper_follow\"[^>]*/>", "", mjcf_path.read_text(encoding="utf-8"))
    assert count == 1
    stripped_path = mjcf_path.with_name("without_viewer_camera.xml")
    stripped_path.write_text(stripped, encoding="utf-8")
    without = mujoco.MjModel.from_xml_path(str(stripped_path))
    assert without.ncam == model.ncam - 1
    assert model.stat.extent == without.stat.extent and model.vis.map.znear == without.vis.map.znear

    _render_compare(mjcf_path, stripped_path)


def test_pick_banana_viewer_setup_leaves_extent_and_recorded_images_unchanged(tmp_path, monkeypatch):
    """Full production pick_banana composition, before vs after the viewer setup.

    The baseline is the same composition (production profile and robot, the
    scene's spawn, its objects, SDF meshes and render settings) with this
    change's viewer additions removed: no layout ``mujoco_viewer`` start
    view, no ``side_view`` scene camera, no ``gripper_follow`` robot camera,
    and the scene's previous ``<global azimuth="160" elevation="-20"/>``
    restored. The extent and znear/zfar (they scale every camera's clipping)
    must be equal, and the recorded top/wrist images — cameras looked up by
    name, since the viewer cameras shift camera ids — pixel-identical.
    """
    import re
    import tempfile

    import sim_models.scene_compiler as scene_compiler
    from robot_runtime.profile import load_profile
    from robot_runtime_sim.mujoco import render_robot_mjcf

    profile = load_profile(_PROFILE)
    peripherals = profile.get("peripherals") or []
    sdf_meshes = profile["simulation"]["mujoco"].get("sdf_meshes") or []
    spawn = scene_compiler.resolve_robot_spawn({}, "pick_banana")
    layout = scene_compiler.get_scene_layout("pick_banana")
    template = scene_compiler.get_scene_file("pick_banana", "mujoco")
    assert "mujoco_viewer" in layout

    def compose(name: str, robot_xml: Path, scene_file: Path, scene_layout: dict) -> Path:
        # each composition lands in its own temp dir (composed_model_path is
        # keyed by tempdir/domain) and never touches a running simulation's model
        out_dir = tmp_path / name
        out_dir.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(out_dir))
        monkeypatch.setattr(scene_compiler, "get_scene_file", lambda _scene, _platform: scene_file)
        monkeypatch.setattr(scene_compiler, "get_scene_layout", lambda _scene: scene_layout)
        return scene_compiler.compose_mujoco_model(str(robot_xml), "pick_banana", sdf_meshes)

    robot_dir = tmp_path / "robot"
    robot_xml = Path(render_robot_mjcf(profile, spawn_pose=spawn, peripherals=peripherals, out_dir=robot_dir))
    after_path = compose("after", robot_xml, template, layout)

    base_robot, robot_hits = re.subn(r"<camera\s+name=\"gripper_follow\"[^>]*/>", "", robot_xml.read_text())
    base_robot_xml = robot_xml.with_name("robot_baseline.xml")
    base_robot_xml.write_text(base_robot, encoding="utf-8")
    base_scene, scene_hits = re.subn(r"<camera\s+name=\"side_view\"[^>]*/>", "", template.read_text())
    base_scene, visual_hits = re.subn(
        r"(<visual>)", r'\1\n    <global azimuth="160" elevation="-20"/>', base_scene, count=1
    )
    assert (robot_hits, scene_hits, visual_hits) == (1, 1, 1)
    base_dir = tmp_path / "scene_baseline"
    base_dir.mkdir()
    (base_dir / "meshes").symlink_to(template.parent / "meshes")  # same {{MESHES_DIR}} assets
    base_template = base_dir / template.name
    base_template.write_text(base_scene, encoding="utf-8")
    base_layout = {key: value for key, value in layout.items() if key != "mujoco_viewer"}
    before_path = compose("before", base_robot_xml, base_template, base_layout)

    assert after_path.suffix == before_path.suffix == ".mjb"
    after = mujoco.MjModel.from_binary_path(str(after_path))
    before = mujoco.MjModel.from_binary_path(str(before_path))
    names = {m: {m.camera(i).name for i in range(m.ncam)} for m in (after, before)}
    assert names[after] - names[before] == {"side_view", "gripper_follow"}
    # the baseline really lacks the start view, the composition really has it
    assert before.vis.global_.fovy == pytest.approx(45.0) and after.vis.global_.fovy < 45.0
    assert tuple(after.stat.center) == pytest.approx(tuple(layout["mujoco_viewer"]["lookat"]))
    assert after.stat.extent == before.stat.extent
    assert after.vis.map.znear == before.vis.map.znear
    assert after.vis.map.zfar == before.vis.map.zfar
    _render_compare(after_path, before_path)
