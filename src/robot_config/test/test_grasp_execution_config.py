from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config_dict

CONFIG = Path(__file__).parents[1] / "config" / "robots" / "so101_handeye_realsense_grasp.yaml"


def _write_config(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutate(data["robot"]["grasp_execution"])
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_grasp_execution_config_accepts_repository_profile() -> None:
    config = load_robot_config_dict(CONFIG)

    assert config["grasp_execution"]["ik"]["worker_count"] == 4


def test_grasp_execution_config_rejects_unknown_nested_key(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        lambda config: config["target_gripper"]["fixed_finger_base_side"].update({"min_aligment_cos": 0.0}),
    )

    with pytest.raises(ValueError, match="min_aligment_cos"):
        load_robot_config_dict(path)


def test_grasp_execution_config_rejects_negative_margin_gain(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        lambda config: config["target_gripper"].update({"fixed_finger_margin_width_gain": -0.1}),
    )

    with pytest.raises(ValueError, match="fixed_finger_margin_width_gain"):
        load_robot_config_dict(path)


def test_grasp_execution_config_rejects_excessive_worker_count(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda config: config["ik"].update({"worker_count": 9}))

    with pytest.raises(ValueError, match="worker_count"):
        load_robot_config_dict(path)
