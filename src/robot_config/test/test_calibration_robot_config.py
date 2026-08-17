from pathlib import Path

import pytest
import yaml

from robot_config.launch_builders.perception import generate_tf_nodes
from robot_config.loader import load_robot_config_dict

CONFIG_PATH = Path(__file__).parents[1] / "config" / "robots" / "lekiwi_sensor_calib.yaml"


def test_calibration_config_starts_mid360_fast_lio_and_default_realsense_driver():
    config = load_robot_config_dict(CONFIG_PATH)

    peripherals = {item["name"]: item for item in config["peripherals"]}
    lidar = peripherals["mid360"]
    camera = peripherals["front"]
    assert lidar["driver"] == "livox_mid360"
    assert lidar["pointcloud_topic"] == "/livox/lidar"
    assert camera["driver"] == "realsense"
    assert camera["driver_camera_name"] == "front"
    assert camera["streams"] == ["color", "depth"]
    assert (camera["width"], camera["height"], camera["fps"]) == (640, 480, 30)
    assert camera["align_depth"] is False
    assert camera["enable_sync"] is False
    assert camera["initial_reset"] is False
    assert camera["transform"]["parent_frame"] == "base_link"

    navigation = config["navigation"]
    assert navigation["enabled"] is True
    assert navigation["fast_lio"]["enabled"] is True
    assert navigation["slam_toolbox"]["enabled"] is False
    assert navigation["nav2_bringup"]["enabled"] is False
    assert navigation["cmd_vel_bridge"]["enabled"] is True
    assert navigation["cmd_vel_bridge"]["publish_tf"] is False
    assert navigation["cmd_vel_bridge"]["publish_odom"] is False

    assert config["control_modes"]["teleop"]["controllers"] == [
        "joint_state_broadcaster",
        "base_velocity_controller",
    ]


def test_calibration_config_consumes_approved_camera_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / ".ros" / "ibrobot" / "calib" / "current" / "base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "status": "approved",
                "device": {"name": "front_camera", "serial": "camera-1"},
                "transform": {
                    "parent_frame": "base_link",
                    "child_frame": "camera_front_optical_frame",
                    "translation": [0.12, -0.34, 0.56],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_robot_config_dict(CONFIG_PATH)
    transform = next(item for item in config["peripherals"] if item["name"] == "front")["transform"]

    assert (transform["x"], transform["y"], transform["z"]) == (0.12, -0.34, 0.56)
    assert [transform[key] for key in ("qx", "qy", "qz", "qw")] == pytest.approx([0.5, -0.5, 0.5, 0.5])
    main_tf = next(
        node for node in generate_tf_nodes(config) if vars(node).get("_Node__node_name") == "static_tf_front"
    )
    arguments = [str(value) for value in main_tf._Node__arguments]
    assert "--roll" not in arguments
    assert [float(arguments[arguments.index(key) + 1]) for key in ("--qx", "--qy", "--qz", "--qw")] == pytest.approx(
        [0.5, -0.5, 0.5, 0.5]
    )


def test_calibration_config_keeps_placeholder_when_artifact_is_not_installed(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))

    config = load_robot_config_dict(CONFIG_PATH)
    transform = next(item for item in config["peripherals"] if item["name"] == "front")["transform"]

    assert (transform["x"], transform["y"], transform["z"]) == (0.0, 0.0, 0.0)
    assert (transform["roll"], transform["pitch"], transform["yaw"]) == (0.0, 0.0, 0.0)


def test_calibration_config_rejects_installed_non_approved_artifact(monkeypatch, tmp_path):
    artifact = tmp_path / ".ros" / "ibrobot" / "calib" / "current" / "base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("status: candidate\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(tmp_path))

    with pytest.raises(ValueError, match="approved camera calibration"):
        load_robot_config_dict(CONFIG_PATH)


def test_calibration_launch_includes_only_base_teleop_control_components():
    launch_path = Path(__file__).parents[1] / "launch/sensor_calibration.launch.py"
    content = launch_path.read_text(encoding="utf-8")

    assert "generate_ros2_control_nodes" in content
    assert "controller_startup_timeout=300.0" in content
    assert "RegisterEventHandler" not in content
    assert "OnProcessExit" not in content
    assert "generate_cmd_vel_nodes" in content
    assert "generate_static_tf_nodes" in content
    assert "generate_navigation_nodes" not in content
    assert "generate_nav2_nodes" not in content


def test_calibration_config_contains_no_realsense_overlay_or_solver_runtime():
    content = CONFIG_PATH.read_text(encoding="utf-8")
    document = yaml.safe_load(content)

    assert "overlay" not in content.lower()
    assert "src/realsense" not in content.lower()
    assert "fast_calib" not in document["robot"]
    assert document["robot"].get("recording", {}).get("enabled", False) is False
