# Copyright 2026 IB_Robot Contributors
# SPDX-License-Identifier: Apache-2.0

import re
from pathlib import Path
from xml.etree import ElementTree

import pytest
import xacro


def test_simulated_transport_model_is_single_arm_and_has_no_startup_home():
    package = Path(__file__).resolve().parents[1]
    document = xacro.process_file(
        str(package / "urdf/lerobot/so101/so101.urdf.xacro"),
        mappings={"use_sim": "false", "simulated": "true"},
    )
    robot = ElementTree.fromstring(document.toxml())
    controls = robot.findall("ros2_control")
    assert len(controls) == 1
    hardware = controls[0].find("hardware")
    assert hardware.findtext("plugin") == "so101_hardware/SO101SystemHardware"
    params = {node.get("name"): node.text or "" for node in hardware.findall("param")}
    assert params["simulated"] == "true"
    assert not params["reset_positions"]
    assert "home_positions" not in params
    joints = controls[0].findall("joint")
    assert [joint.get("name") for joint in joints] == [str(i) for i in range(1, 7)]
    for joint in joints:
        name = joint.get("name")
        assert joint.findtext("param[@name='id']") == name
        limit = robot.find(f"joint[@name='{name}']/limit")
        command = joint.find("command_interface[@name='position']")
        assert float(command.findtext("param[@name='min']")) == float(limit.get("lower"))
        assert float(command.findtext("param[@name='max']")) == float(limit.get("upper"))
    for mesh in robot.findall(".//mesh"):
        prefix = "package://so101_description/"
        assert mesh.get("filename").startswith(prefix)
        assert (package / mesh.get("filename").removeprefix(prefix)).is_file()


def _render(mappings):
    package = Path(__file__).resolve().parents[1]
    document = xacro.process_file(str(package / "urdf/lerobot/so101/so101.urdf.xacro"), mappings=mappings)
    return ElementTree.fromstring(document.toxml())


def test_ros2_control_xacro_renders_for_every_platform():
    """hardware, sdk, gazebo and mujoco argument sets all render (task 3.2)."""
    robot = _render({"use_sim": "false"})
    assert robot.findtext(".//hardware/plugin") == "so101_hardware/SO101SystemHardware"
    assert robot.find(".//state_interface[@name='effort']") is None

    robot = _render({"use_sim": "true", "sim_plugin": "gz_ros2_control/GazeboSimSystem", "sim_effort_state": "true"})
    assert robot.findtext(".//hardware/plugin") == "gz_ros2_control/GazeboSimSystem"
    assert robot.find(".//state_interface[@name='effort']") is not None

    # the SDK virtual transport: the hardware plugin with simulated=true
    robot = _render({"use_sim": "false", "simulated": "true"})
    assert robot.findtext(".//hardware/plugin") == "so101_hardware/SO101SystemHardware"
    params = {node.get("name"): node.text or "" for node in robot.find(".//ros2_control/hardware").findall("param")}
    assert params["simulated"] == "true"

    robot = _render(
        {
            "use_sim": "true",
            "sim_plugin": "mujoco_ros2_control/MujocoSystemInterface",
            "simulated": "false",
            "sim_effort_state": "true",
            "mujoco_model": "/tmp/composed.mjb",
            "mujoco_headless": "true",
        }
    )
    hardware = robot.find(".//ros2_control/hardware")
    assert hardware.findtext("plugin") == "mujoco_ros2_control/MujocoSystemInterface"
    params = {node.get("name"): node.text or "" for node in hardware.findall("param")}
    assert params["mujoco_model"] == "/tmp/composed.mjb"
    # cameras run through the simulator's CameraPlugin parameters; the
    # deprecated <sensor> auto-registration is off and the rate param is gone
    assert params["auto_register_cameras"] == "false"
    assert "camera_publish_rate" not in params
    assert params["headless"] == "true"
    assert robot.find(".//state_interface[@name='effort']") is not None
    assert robot.find(".//ros2_control/sensor") is None

    # headless stays off by default (the viewer path); auto-register stays off
    # whenever a mujoco model is configured
    robot = _render(
        {
            "use_sim": "true",
            "sim_plugin": "mujoco_ros2_control/MujocoSystemInterface",
            "mujoco_model": "/tmp/composed.mjb",
        }
    )
    hardware_params = {node.get("name") for node in robot.find(".//ros2_control/hardware").findall("param")}
    assert "headless" not in hardware_params
    assert "auto_register_cameras" in hardware_params

    # the Gazebo plugin never carries the MuJoCo-only params
    robot = _render({"use_sim": "true", "sim_plugin": "gz_ros2_control/GazeboSimSystem"})
    hardware_params = {node.get("name") for node in robot.find(".//ros2_control/hardware").findall("param")}
    assert "auto_register_cameras" not in hardware_params
    assert "camera_publish_rate" not in hardware_params


def test_servo_tuning_args_reach_the_hardware_plugin_only_when_set():
    """Unset tuning args leave the driver defaults (no param); set ones pass through verbatim."""

    def hardware_params(mappings):
        robot = _render(mappings)
        return {node.get("name"): node.text or "" for node in robot.find(".//ros2_control/hardware").findall("param")}

    tuning = ("profile_speed", "profile_acc", "position_kp")
    unset = hardware_params({"use_sim": "false"})
    assert not any(name in unset for name in tuning)

    params = hardware_params({"use_sim": "false", "profile_speed": "0", "profile_acc": "254", "position_kp": "24"})
    assert (params["profile_speed"], params["profile_acc"], params["position_kp"]) == ("0", "254", "24")

    only_acc = hardware_params({"use_sim": "false", "profile_acc": "100"})
    assert only_acc["profile_acc"] == "100"
    assert "profile_speed" not in only_acc and "position_kp" not in only_acc

    # simulation plugins never receive the servo tuning
    sim = hardware_params({"use_sim": "true", "profile_acc": "254"})
    assert "profile_acc" not in sim


def _render_text(mappings):
    package = Path(__file__).resolve().parents[1]
    document = xacro.process_file(str(package / "urdf/lerobot/so101/so101.urdf.xacro"), mappings=mappings)
    # strip the xacro autogeneration header and fold resolved `$(find pkg)`
    # paths back: both embed absolute install paths, which differ per checkout
    text = document.toxml()
    return re.sub(r"[^<>\"\s]*/share/([A-Za-z0-9_]+)/", r"$(find \1)/", text[text.index("<robot") :])


def test_root_link_default_keeps_every_render_byte_identical():
    """D8 (lekiwi-nav-grasp-sim-platforms): the `root_link` arg is transparent.

    The default (`world`) must keep every SO-101 render byte-identical to the
    pre-change output (checked-in goldens, header-normalized), an explicit
    `root_link:=world` must match the default exactly, and the rename only
    takes effect when an includer overrides the arg.
    """
    golden = Path(__file__).resolve().parent / "golden"
    for name, mappings in (
        ("so101_default.urdf.xml", {"use_sim": "false"}),
        ("so101_simulated.urdf.xml", {"use_sim": "false", "simulated": "true"}),
    ):
        rendered = _render_text(mappings)
        assert rendered == (golden / name).read_text(), f"SO-101 render changed for {mappings}"
        assert 'name="world"' in rendered and "arm_root" not in rendered
    assert _render_text({"use_sim": "false", "root_link": "world"}) == _render_text({"use_sim": "false"})
    renamed = _render_text({"use_sim": "false", "root_link": "arm_root"})
    assert 'name="arm_root"' in renamed and 'name="world"' not in renamed


def _jaw_collisions(robot, link_name):
    """The collision elements of one jaw link: [(mesh basename, xyz, rpy), …]."""
    link = robot.find(f"link[@name='{link_name}']")
    assert link is not None
    out = []
    for collision in link.findall("collision"):
        mesh = collision.find("geometry/mesh")
        origin = collision.find("origin")
        xyz = tuple(float(v) for v in (origin.get("xyz", "0 0 0")).split())
        rpy = tuple(float(v) for v in (origin.get("rpy", "0 0 0")).split())
        out.append((mesh.get("filename").rsplit("/", 1)[-1].removesuffix(".stl"), xyz, rpy))
    return out


def _mjcf_gripper_col_geoms():
    """The gripper_col geoms of so101.xml.template: {body: [(mesh, pos, euler)…]}."""
    package = Path(__file__).resolve().parents[1]
    text = (package / "mujoco/so101.xml.template").read_text()
    robot = ElementTree.fromstring(text)
    out = {}
    for body in robot.iter("body"):
        geoms = [g for g in body.findall("geom") if g.get("class") == "gripper_col"]
        if geoms:
            out[body.get("name")] = [
                (
                    g.get("mesh"),
                    tuple(float(v) for v in g.get("pos", "0 0 0").split()),
                    tuple(float(v) for v in g.get("euler", "0 0 0").split()),
                )
                for g in geoms
            ]
    return out


def test_jaw_collision_default_is_the_raw_mesh():
    """D7 (lekiwi-nav-grasp): the default keeps exactly one raw-STL collision
    per jaw link — every hardware/SDK/MoveIt render is byte-identical."""
    robot = _render({"use_sim": "false"})
    default_gripper = _jaw_collisions(robot, "gripper")
    assert default_gripper == [
        # the servo housing collision (untouched by jaw_collision) ...
        ("sts3215_03a_v1", (0.0077, 0.0001, -0.0234), (-1.5708, -5.55112e-17, -1.38213e-14)),
        # ... and the raw fixed-finger STL
        (
            "wrist_roll_follower_so101_v1",
            (5.55112e-17, -0.000218214, 0.000949706),
            (-3.14159, -5.55112e-17, -9.17912e-24),
        ),
    ]
    assert _jaw_collisions(robot, "jaw") == [
        ("moving_jaw_so101_v1", (-5.55112e-17, -1.94746e-17, 0.0189), (9.53145e-17, -4.66093e-24, 0.0))
    ]
    # 2.0b: an unknown value FAILS the render (an undefined macro inside a
    # taken xacro:if) instead of falling back — a typo can neither silently
    # enable the hulls nor silently keep the raw mesh
    with pytest.raises(Exception, match="invalid_jaw_collision_value"):
        _render_text({"use_sim": "false", "jaw_collision": "bogus"})


def test_jaw_collision_hulls_match_the_mjcf_template():
    """D7 (lekiwi-nav-grasp, maintainer decision 2026-10-29): `hulls` swaps
    both jaw links' collision to the CoACD hull decomposition with the poses
    exactly as so101.xml.template's gripper_col geoms (its eulerseq XYZ
    equals URDF rpy), and changes nothing else in the jaw links."""
    robot = _render({"use_sim": "false", "jaw_collision": "hulls"})
    mjcf = _mjcf_gripper_col_geoms()
    gripper_collisions = _jaw_collisions(robot, "gripper")
    moving = _jaw_collisions(robot, "jaw")
    # the servo housing collision is untouched; the finger STL becomes the
    # 8 CoACD hulls with the MJCF template's poses
    assert gripper_collisions[0] == ("sts3215_03a_v1", (0.0077, 0.0001, -0.0234), (-1.5708, -5.55112e-17, -1.38213e-14))
    fixed = gripper_collisions[1:]
    assert len(fixed) == 8 and len(moving) == 8
    assert fixed == [(mesh, pos, rpy) for mesh, pos, rpy in mjcf["gripper"]]
    assert moving == [(mesh, pos, rpy) for mesh, pos, rpy in mjcf["jaw"]]
    # the visuals are untouched: both raw STLs still render on the jaw links
    meshes = [
        mesh.get("filename").rsplit("/", 1)[-1].removesuffix(".stl")
        for mesh in robot.findall("link[@name='gripper']/visual/geometry/mesh")
    ]
    assert "wrist_roll_follower_so101_v1" in meshes
    assert "moving_jaw_so101_v1" in [
        mesh.get("filename").rsplit("/", 1)[-1].removesuffix(".stl")
        for mesh in robot.findall("link[@name='jaw']/visual/geometry/mesh")
    ]
    # the hull meshes exist in the package
    package = Path(__file__).resolve().parents[1]
    for mesh, _, _ in fixed + moving:
        assert (package / f"meshes/lerobot/so101/{mesh}.stl").is_file()


def test_jaw_collision_hulls_change_only_the_jaw_collisions():
    """The hulls render is byte-identical to the default render outside the
    two jaw links' collision blocks (pinned goldens, header-normalized)."""
    golden = Path(__file__).resolve().parent / "golden"
    rendered = _render_text({"use_sim": "false", "jaw_collision": "hulls"})
    assert rendered == (golden / "so101_jaw_hulls.urdf.xml").read_text()

    def strip_jaw_collisions(text):
        robot = ElementTree.fromstring(text)
        for name in ("gripper", "jaw"):
            link = robot.find(f"link[@name='{name}']")
            for collision in link.findall("collision"):
                link.remove(collision)
        return ElementTree.tostring(robot, encoding="unicode")

    default = (golden / "so101_default.urdf.xml").read_text()
    assert strip_jaw_collisions(rendered) == strip_jaw_collisions(default)


def _joint_six_limit(robot):
    return robot.find("joint[@name='6']/limit")


def test_gripper_effort_default_is_the_shipped_ten():
    """D7 (lekiwi-nav-grasp, "Gazebo gripper effort", task 2.0b): the default
    keeps the shipped joint-6 effort limit 10 — every hardware/SDK/MoveIt
    render is byte-identical (pinned goldens)."""
    robot = _render({"use_sim": "false"})
    assert _joint_six_limit(robot).get("effort") == "10"


def test_gripper_effort_arg_reaches_the_joint_limit():
    """Only the Gazebo render path sets the arg; when it is set the joint-6
    effort limit follows it exactly."""
    robot = _render({"use_sim": "false", "gripper_effort": "2.0"})
    assert _joint_six_limit(robot).get("effort") == "2.0"
    robot = _render({"use_sim": "true", "sim_plugin": "gz_ros2_control/GazeboSimSystem", "gripper_effort": "1.0"})
    assert _joint_six_limit(robot).get("effort") == "1.0"
