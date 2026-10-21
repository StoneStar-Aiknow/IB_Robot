# Copyright 2026 IB_Robot Contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from xml.etree import ElementTree

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
            "camera_publish_rate": "5.0",
            "mujoco_headless": "true",
        }
    )
    hardware = robot.find(".//ros2_control/hardware")
    assert hardware.findtext("plugin") == "mujoco_ros2_control/MujocoSystemInterface"
    params = {node.get("name"): node.text or "" for node in hardware.findall("param")}
    assert params["mujoco_model"] == "/tmp/composed.mjb"
    assert params["camera_publish_rate"] == "5.0"
    assert params["headless"] == "true"
    assert robot.find(".//state_interface[@name='effort']") is not None

    # headless stays off by default (the viewer path)
    robot = _render(
        {
            "use_sim": "true",
            "sim_plugin": "mujoco_ros2_control/MujocoSystemInterface",
            "mujoco_model": "/tmp/composed.mjb",
        }
    )
    assert "headless" not in {node.get("name") for node in robot.find(".//ros2_control/hardware").findall("param")}
