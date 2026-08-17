from pathlib import Path

import yaml

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
