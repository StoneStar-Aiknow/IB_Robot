"""Bind the capture overlay to the managed base without opening hardware."""

from pathlib import Path

import pytest
import yaml
from launch import LaunchContext
from launch.utilities import perform_substitutions

from robot_config.interface_binding import required_interface_ids
from robot_config.launch_builders.runtime import _write_peripherals_fragment
from robot_config.launch_registry.builtin_registry import build_launch_registry
from robot_config.launch_registry.preparation import prepare_launch
from robot_config.loader import load_robot_config_dict
from robot_config.testing.provider_binding import bind_offline, fixture_urdf, offline_description
from robot_runtime.peripherals import _peripheral_tf_nodes, merge_peripherals
from robot_runtime.profile import load_profile

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "src/robot_config/config/robots/lekiwi_semantic_capture.yaml"
PROFILE = ROOT / "src/robots/lekiwi/lekiwi_robot/profiles/lekiwi_lidar.yaml"


def _load():
    return load_robot_config_dict(CONFIG, defer_interface_binding=True)


@pytest.mark.parametrize("use_sim", ["false", "true"])
def test_capture_preparation_requests_live_base_binding(monkeypatch, tmp_path, use_sim):
    monkeypatch.setenv("HOME", str(tmp_path))
    context = LaunchContext()
    context.launch_configurations.update(robot_config="lekiwi_semantic_capture", use_sim=use_sim)
    prepared = prepare_launch(context, build_launch_registry())
    assert prepared.live_binding_required is True
    assert required_interface_ids(prepared.robot_config) == ["base.cmd_vel"]
    assert "require_model" not in prepared.robot_config["runtime"]
    assert "interface_description" not in prepared.robot_config["runtime"]
    assert prepared.control_mode == "teleop"
    assert not prepared.with_inference
    assert not next(p for p in prepared.robot_config["peripherals"] if p["name"] == "front")["transform"]
    if use_sim == "true":
        assert prepared.sim_platform == "mujoco"
        assert prepared.node_use_sim_time is True


def test_lidar_preparation_requests_live_binding_for_active_gamepad(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_robot_config_dict(CONFIG.with_name("lekiwi_lidar.yaml"))
    config["teleoperation"]["active_device"] = "lekiwi_gamepad"
    context = LaunchContext()
    context.launch_configurations["use_sim"] = "false"
    prepared = prepare_launch(context, build_launch_registry(), loaded_config=config)
    assert prepared.live_binding_required is True
    assert required_interface_ids(prepared.robot_config) == ["base.cmd_vel"]


@pytest.mark.parametrize("inactive", ["disabled", "unselected", "other_device", "providerless"])
def test_only_an_active_provider_gamepad_requires_base_binding(monkeypatch, tmp_path, inactive):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _load()
    if inactive == "disabled":
        config["teleoperation"]["enabled"] = False
    elif inactive == "unselected":
        config["teleoperation"]["active_device"] = ""
    elif inactive == "other_device":
        config["teleoperation"]["devices"][0]["type"] = "leader_arm"
    else:
        config.pop("runtime")
    assert required_interface_ids(config) == []


def test_capture_loads_offline_without_a_description(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = load_robot_config_dict(CONFIG)
    assert config["recording"]["semantic_dataset"] is True
    assert "interface_description" not in config["runtime"]


def test_capture_binds_camera_lidar_and_managed_base(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = bind_offline(_load(), PROFILE)
    interfaces = config["runtime"]["interface_description"]["interfaces"]
    recording = config["recording"]
    for name in (
        "camera.front.color",
        "camera.front.depth",
        "camera.front.aligned_depth",
        "camera.front.color_info",
        "camera.front.depth_info",
        "camera.front.aligned_depth_info",
        "lidar.mid360.points",
        "lidar.mid360.imu",
        "lidar.mid360.scan",
        "base.cmd_vel",
        "base.odom",
        "localization.cloud",
        "localization.odometry",
    ):
        assert interfaces[name]["endpoint"] in recording["topics"], name
    assert interfaces["base.cmd_vel"]["modes"] == ["base_navigation"]
    assert config["control_modes"]["teleop"]["runtime_mode"] == "base_navigation"
    assert config["teleoperation"]["active_device"] == "lekiwi_gamepad"
    assert recording["semantic_dataset"] is True
    assert recording["storage"] == "mcap"
    assert recording["camera_info_topic"] == interfaces["camera.front.color_info"]["endpoint"]
    assert "/wheel/odom" not in recording["topics"]


def test_capture_fragment_enables_mapping_and_aligned_rgbd(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    config = _load()
    path = Path(_write_peripherals_fragment(config))
    try:
        fragment = yaml.safe_load(path.read_text())
    finally:
        path.unlink()
    effective = merge_peripherals(load_profile(str(PROFILE))["peripherals"], fragment["peripherals"])
    camera = next(item for item in effective if item["name"] == "front")
    lidar = next(item for item in effective if item["name"] == "mid360")
    assert camera["enabled"] and camera["align_depth"] and camera["enable_sync"]
    assert camera["streams"] == ["color", "depth"]
    assert camera["driver"] == "realsense"
    assert lidar["scan_converter"]["min_height"] == -0.20
    assert fragment["fast_lio"]["enabled"] is True
    assert fragment["fast_lio"]["planar_output"] is False
    assert config["navigation"]["slam_toolbox"]["enabled"] is True
    assert config["navigation"]["nav2_bringup"]["enabled"] is False
    assert config["navigation"]["open_gate_on_startup"] is True
    # A missing calibration must not resurrect the profile's zero-pose mount.
    assert not camera["transform"]
    assert _peripheral_tf_nodes(camera) == []


def test_capture_passes_approved_camera_mount_to_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    artifact = tmp_path / ".ros/ibrobot/calib/current/base_to_front_camera.yaml"
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        yaml.safe_dump(
            {
                "status": "approved",
                "transform": {
                    "parent_frame": "base_link",
                    "child_frame": "camera_front_optical_frame",
                    "translation": [0.12, -0.34, 0.56],
                    "rotation_xyzw": [0, 0, 0, 1],
                },
            }
        )
    )
    config = _load()
    camera = next(item for item in config["peripherals"] if item["name"] == "front")
    assert [camera["transform"][key] for key in ("x", "y", "z")] == [0.12, -0.34, 0.56]
    assert [camera["transform"][key] for key in ("qx", "qy", "qz", "qw")] == pytest.approx([0.5, -0.5, 0.5, 0.5])
    assert _peripheral_tf_nodes(camera)
    assert (
        offline_description(config, PROFILE)["interfaces"]["camera.front.color"]["frame_id"]
        == "camera_front_optical_frame"
    )
    from robot_runtime_sim.mujoco import camera_mjcf_elements, camera_tf_urdf

    effective = merge_peripherals(load_profile(str(PROFILE))["peripherals"], config["peripherals"])
    assert camera_mjcf_elements(effective)[0][0] == "base_link"
    assert 'pos="0.120000 -0.340000 0.560000"' in camera_mjcf_elements(effective)[0][1]
    assert 'link name="camera_front_optical_frame"' in camera_tf_urdf(effective)

    # Reapplying calibration to an already loaded missing-mount fragment must
    # restore the mount, not crash on its explicit null transform.
    from robot_config.loader import _apply_approved_camera_calibration

    camera["transform"] = None
    _apply_approved_camera_calibration(config)
    assert camera["transform"]["parent_frame"] == "base_link"
    assert [camera["transform"][key] for key in ("x", "y", "z")] == [0.12, -0.34, 0.56]


def test_capture_simulation_composes_raw_odometry_and_camera_info_relays(monkeypatch, tmp_path):
    from robot_runtime.interface_description import build_description
    from robot_runtime.peripherals import fast_lio_nodes
    from robot_runtime_sim.mujoco import camera_plugin_parameters, mujoco_camera_relays

    monkeypatch.setenv("HOME", str(tmp_path))
    config = _load()
    profile = load_profile(str(PROFILE))
    platform = config["simulation"]["platform"]
    assert platform == "mujoco" and platform in profile["sim_platforms"]
    effective = merge_peripherals(profile["peripherals"], config["peripherals"])
    effective_profile = {
        **profile,
        "peripherals": effective,
        "fast_lio": {**profile["fast_lio"], **config["navigation"]["fast_lio"]},
    }
    description = build_description(
        effective_profile,
        robot_description=fixture_urdf(profile),
        use_sim=True,
        sim_platform=platform,
        profile_path=PROFILE,
    )
    interfaces = description["interfaces"]
    plugin = next(iter(camera_plugin_parameters(profile, effective)["mujoco_plugins"].values()))["front_camera"]
    for role, field in (("color", "image_topic"), ("depth", "depth_topic"), ("color_info", "info_topic")):
        assert plugin[field] == interfaces[f"camera.front.{role}"]["endpoint"]
        assert plugin[field] in config["recording"]["topics"]
    relays = mujoco_camera_relays(effective)
    relay_targets = {node._Node__arguments[1] for node in relays}
    for topic in (
        "/camera/front/aligned_depth_to_color/image_raw",
        "/camera/front/depth/camera_info",
        "/camera/front/aligned_depth_to_color/camera_info",
    ):
        assert topic in relay_targets
        assert topic in config["recording"]["topics"]
    for role in ("aligned_depth", "depth_info", "aligned_depth_info"):
        assert interfaces[f"camera.front.{role}"]["endpoint"] in relay_targets
    nodes = fast_lio_nodes(
        effective_profile["fast_lio"],
        bridge_package="lekiwi_robot",
        use_sim=True,
        sim_platform=platform,
    )
    assert len(nodes) == 2
    remappings = {
        perform_substitutions(LaunchContext(), source): perform_substitutions(LaunchContext(), target)
        for source, target in nodes[0]._Node__remappings
    }
    assert remappings["/Odometry"] == "/fast_lio/odometry_raw"
