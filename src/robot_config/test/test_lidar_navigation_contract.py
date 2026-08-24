from pathlib import Path

import yaml

from robot_config import loader
from robot_config.loader import robot_context_schema_version

ROOT = Path(__file__).resolve().parents[3]


def _config() -> dict:
    path = ROOT / "src/robot_config/config/robots/lekiwi_lidar.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["robot"]


def test_lidar_profiles_keep_configured_tf_authorities_and_isolation():
    config = _config()
    mapping = config["nav_stages"]["mapping"]["navigation"]
    navigation = config["nav_stages"]["navigation"]["navigation"]

    assert mapping["slam_toolbox"]["enabled"] is True
    assert mapping["nav2_bringup"]["enabled"] is False
    assert navigation["slam_toolbox"]["enabled"] is False
    assert navigation["nav2_bringup"]["use_amcl"] is True
    assert mapping["fast_lio"].get("publish_tf", True) is True
    assert navigation["fast_lio"].get("publish_tf", True) is True
    assert mapping["cmd_vel_bridge"]["publish_tf"] is False
    assert navigation["cmd_vel_bridge"]["publish_tf"] is False


def test_fast_lio_native_tf_topics_are_isolated_from_navigation_tf():
    builder = ROOT / "src/robot_config/robot_config/launch_builders/fast_lio.py"
    content = builder.read_text(encoding="utf-8")

    assert '"/tf", config.get("isolated_tf_topic", "/fast_lio/tf_raw")' in content
    assert '"/tf_static", config.get("isolated_tf_static_topic", "/fast_lio/tf_static_raw")' in content


def test_navigation_stage_exposes_navigation_skills_but_mapping_stage_does_not(monkeypatch):
    config_path = ROOT / "src/robot_config/config/robots/lekiwi_lidar.yaml"
    mount_path = ROOT / "src/robot_config/config/hardware/lekiwi_mid360_mount.yaml"
    original_resolver = loader.resolve_ros_path

    def resolve_path(value):
        if value == "$(find robot_config)/config/hardware/lekiwi_mid360_mount.yaml":
            return str(mount_path)
        return original_resolver(value)

    monkeypatch.setattr(loader, "resolve_ros_path", resolve_path)

    mapping = loader.load_robot_config_dict(config_path, nav_stage="mapping")
    navigation = loader.load_robot_config_dict(config_path, nav_stage="navigation")

    assert not mapping.get("embodied", {}).get("skill_catalog_profile")
    assert mapping["default_control_mode"] == "teleop"
    assert robot_context_schema_version(mapping) == 1
    assert navigation["skill_required_control_mode"] == "base_navigation"
    assert navigation["default_control_mode"] == "base_navigation"
    assert (
        navigation["control_modes"]["base_navigation"]["controllers"]
        == mapping["control_modes"]["teleop"]["controllers"]
    )
    assert robot_context_schema_version(navigation) == 2
    assert "base_navigation" in navigation["control_modes"]
