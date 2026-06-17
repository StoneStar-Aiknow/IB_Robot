import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

base_module = types.ModuleType("robot_teleop")
base_teleop_module = types.ModuleType("robot_teleop.base_teleop")


class BaseTeleopDevice:
    def __init__(self, config, node=None):
        self._config = config
        self._node = node
        self._is_connected = False


base_teleop_module.BaseTeleopDevice = BaseTeleopDevice
sys.modules["robot_teleop"] = base_module
sys.modules["robot_teleop.base_teleop"] = base_teleop_module
sys.modules["robot_teleop.devices"] = types.ModuleType("robot_teleop.devices")

leader_arm_path = Path(__file__).resolve().parents[1] / "robot_teleop" / "devices" / "leader_arm.py"
spec = importlib.util.spec_from_file_location("robot_teleop.devices.leader_arm", leader_arm_path)
assert spec is not None
assert spec.loader is not None
leader_arm_module = importlib.util.module_from_spec(spec)
sys.modules["robot_teleop.devices.leader_arm"] = leader_arm_module
spec.loader.exec_module(leader_arm_module)
LeaderArmDevice = leader_arm_module.LeaderArmDevice


class FakeMotorsBus:
    def __init__(self, positions):
        self._positions = positions

    def sync_read(self, _register, normalize=False):
        assert normalize is False
        return self._positions


def _connected_device(config, positions, calibration=None):
    device = LeaderArmDevice(config)
    device._is_connected = True
    device.motors_bus = FakeMotorsBus(positions)
    device.calibration = calibration
    return device


def test_leader_arm_uses_direct_gripper_joint_names_only():
    device = LeaderArmDevice({"target": {"gripper_joint_names": ["joint6_left"]}})

    assert device.gripper_joints == {"6"}


def test_gripper_normalization_failure_skips_target_without_radians(caplog):
    device = _connected_device(
        {"joint_mapping": {"6": "joint6_left"}, "gripper_joint_names": ["joint6_left"]},
        {"6": 4095},
        calibration={},
    )

    targets = device.get_joint_targets()

    assert "joint6_left" not in targets
    assert "skipping publish" in caplog.text


def test_gripper_target_normalizes_with_drive_mode():
    device = _connected_device(
        {"joint_mapping": {"6": "joint6_left"}, "gripper_joint_names": ["joint6_left"]},
        {"6": 25},
        calibration={"6": SimpleNamespace(range_min=0, range_max=100, drive_mode=1)},
    )

    assert device.get_joint_targets()["joint6_left"] == 0.75


def test_arm_joint_still_uses_radians_path():
    device = _connected_device({}, {"1": 2049})

    assert device.get_joint_targets()["1"] == device.rad_per_step
