"""Device-free SO-101 input and runtime composition regressions."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from ament_index_python.packages import get_package_share_directory

from so101_hardware.leader_input import LeaderInput


def test_sdk_source_only_uses_readonly_lifecycle(tmp_path):
    events = []

    class Leader:
        def __init__(self, config):
            assert config.calibration_version == 1

        def connect(self):
            events.append("connect")
            return True

        def read(self):
            events.append("read")
            return {"1": {"position": 0.3}, "6": {"position": 0.7}}

        def disconnect(self):
            events.append("disconnect")

    path = tmp_path / "calibration.json"
    path.write_text("{}")
    sdk = SimpleNamespace(LeaderConfig=SimpleNamespace, LeaderArm=Leader)
    source = LeaderInput(
        {"calibration_version": 1, "calibration_file": str(path), "joint_order": ["1", "6"], "gripper_joint": "6"}, sdk
    )
    source.connect()
    assert source.read() == {"1": 0.3, "6": 0.7}
    source.disconnect()
    assert events == ["connect", "read", "disconnect"]


@pytest.mark.parametrize("version", [None, 0, 2, True])
def test_sdk_provisioning_ack_is_required(version):
    with pytest.raises(ValueError, match="explicit provisioned"):
        LeaderInput({"calibration_version": version})


def test_real_runtime_projection_composes_executor_and_all_inputs(monkeypatch):
    import yaml

    from robot_runtime.interface_description import build_description
    from robot_runtime.launch_support import render_robot_description
    from robot_teleop.launch_builders import teleop
    from so101_robot import teleoperation

    config_share = Path(get_package_share_directory("robot_config"))
    path = Path(get_package_share_directory("so101_robot")) / "profiles/so101_single_arm.yaml"
    profile = yaml.safe_load(path.read_text())
    xml = render_robot_description(profile, path, True)
    descriptor = build_description(profile, use_sim=True, robot_description=xml)
    item = yaml.safe_load((config_share / "config/robots/so101_single_arm.yaml").read_text())["robot"]
    item["robot_model"] = descriptor["model"]
    item["runtime"]["interface_description"] = descriptor
    item["teleoperation"]["input_config"] = str(config_share / "config/teleop/so101_leader_inputs.yaml")
    monkeypatch.setattr(teleoperation, "Node", lambda **kwargs: kwargs)
    executor = teleoperation.generate_teleoperation_nodes(profile, descriptor, {"robot_description": xml})[0]
    params = executor["parameters"][0]
    assert params["managed_teleop"] is True
    assert params["robot_description"] == xml
    assert params["joint_intent_topic"] == descriptor["interfaces"]["motion.arm.joints"]["endpoint"]
    monkeypatch.setattr(teleop, "Node", lambda **kwargs: kwargs)
    for device in ("so101_leader", "xbox_controller", "phone_teleop", "vr_teleop"):
        item["teleoperation"]["active_device"] = device
        nodes = teleop.generate_teleop_nodes(item)
        assert any(node["package"] == "robot_teleop" for node in nodes)
