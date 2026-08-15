from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]


def _lidar_config() -> dict:
    path = ROOT / "src/robot_config/config/robots/lekiwi_lidar.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["robot"]


def test_lidar_stages_declare_their_bridge_input_topics():
    config = _lidar_config()

    mapping = config["nav_stages"]["mapping"]["navigation"]
    navigation = config["nav_stages"]["navigation"]["navigation"]

    assert mapping["cmd_vel_bridge"]["cmd_vel_topic"] == "/cmd_vel"
    assert navigation["cmd_vel_bridge"]["cmd_vel_topic"] == "/cmd_vel_safe"
    assert navigation["command_server"]["stop_velocity_topic"] == "/cmd_vel_safe"


def test_dynamic_navigation_rejects_a_bridge_topic_that_bypasses_collision_monitor():
    from robot_config.launch_builders.navigation import _validate_navigation_velocity_contract

    config = _lidar_config()
    navigation = config["nav_stages"]["navigation"]["navigation"]
    navigation["cmd_vel_bridge"]["cmd_vel_topic"] = "/cmd_vel"

    with pytest.raises(ValueError, match="cmd_vel_safe"):
        _validate_navigation_velocity_contract(config, navigation)


def test_cmd_vel_builder_preserves_a_custom_non_dynamic_topic(monkeypatch):
    from robot_config.launch_builders import cmd_vel

    captured = {}

    class _Node:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(cmd_vel, "Node", _Node)

    nodes = cmd_vel.generate_cmd_vel_nodes(
        {
            "cmd_vel_bridge": {"enabled": True, "cmd_vel_topic": "/custom/cmd_vel"},
            "nav2_bringup": {"dyn_avoid_enabled": False},
        }
    )

    assert len(nodes) == 1
    assert captured["parameters"][0]["cmd_vel_topic"] == "/custom/cmd_vel"
