"""Real Gazebo/MuJoCo interface-description contracts (sim-platform-registry D6).

Moved from ``robot_runtime/test/test_interface_description.py``: every case
that builds a description for a real registered platform needs the plugin's
sensor traits, so it lives where the platforms are installed. The core file
keeps the platform-neutral contract tests and the fake-platform cases.
"""

from __future__ import annotations

import pytest

from robot_runtime.interface_description import (
    build_description,
    description_digest,
    description_sim_platform,
    peripheral_interfaces,
    validate_description,
)


@pytest.fixture
def profile():
    return {
        "runtime": {"name": "test_robot", "version": "1.0.0", "instance_id": "test-1"},
        "capabilities": {"joint.state": {"rate_hz": 50}, "motion.ik": {"endpoints": ["/motion/compute_ik"]}},
        "joints": ["joint_1"],
        "hardware": {"port": "/dev/private-device", "calib_file": "/private/path"},
        "peripherals": [
            {
                "type": "camera",
                "name": "front",
                "driver": "opencv",
                "index": 0,
                "width": 16,
                "height": 12,
                "fps": 20,
                "pixel_format": "mjpeg2rgb",
                "frame_id": "front_optical",
            }
        ],
    }


@pytest.mark.parametrize(
    ("use_sim", "sim_platform", "execution"),
    [
        (True, "gazebo", "simulated"),
        (True, "mujoco", "simulated"),
    ],
)
def test_description_records_sim_platform_and_execution(profile, use_sim, sim_platform, execution):
    doc = build_description(profile, use_sim=use_sim, sim_platform=sim_platform)
    assert doc.get("sim_platform") == sim_platform
    assert doc["execution"] == execution
    validate_description(doc)


_EXCLUDED_MIC = {
    "type": "microphone",
    "name": "respeaker",
    "driver": "alsa",
    "params": {"device": "hw:0,0"},
    "simulation": {"gazebo": {"enabled": False}, "mujoco": {"enabled": False}},
}
_EXCLUDED_SPEAKER = {
    "type": "speaker",
    "name": "speaker",
    "driver": "alsa",
    "params": {"device": ""},
    "simulation": {"gazebo": {"enabled": False}, "mujoco": {"enabled": False}},
}
_EXCLUDED_CAMERA = {
    "type": "camera",
    "name": "front",
    "driver": "realsense",
    "width": 640,
    "height": 480,
    "streams": ["color", "depth"],
    "frame_id": "camera_front_link",
    "transform": {"parent_frame": "base_link"},
    "simulation": {"gazebo": {"enabled": False}, "mujoco": {"enabled": False}},
}
_SIMULATED_WRIST = {
    "type": "camera",
    "name": "wrist",
    "driver": "realsense",
    "width": 640,
    "height": 360,
    "streams": ["color", "depth"],
    "frame_id": "camera_wrist_link",
    "transform": {"parent_frame": "gripper"},
}
_EXCLUSION_INVENTORY = [_EXCLUDED_MIC, _EXCLUDED_SPEAKER, _EXCLUDED_CAMERA, _SIMULATED_WRIST]


def test_mujoco_description_validates_and_keeps_a_stable_digest(profile):
    # the schema enum and the reader accept mujoco (D1), and re-digesting the
    # same document reproduces the digest exactly as for the older platforms
    doc = build_description(profile, use_sim=True, sim_platform="mujoco")
    assert description_sim_platform(doc) == "mujoco"
    assert doc["digest"] == description_digest(doc)
    validate_description(doc)
    again = build_description(profile, use_sim=True, sim_platform="mujoco")
    assert again["digest"] == doc["digest"]


def test_mujoco_depth_stream_dims_come_from_the_rendered_camera(profile):
    # The plugin renders one MJCF camera at width x height for every stream
    # (D5): unequal device depth overrides must not leak into the mujoco
    # contract (finding 7).
    peripherals = [
        {
            "type": "camera",
            "name": "front",
            "driver": "realsense",
            "width": 16,
            "height": 12,
            "depth_width": 8,
            "depth_height": 6,
            "align_depth": True,
            "streams": ["color", "depth"],
        }
    ]
    descriptor = build_description({**profile, "peripherals": peripherals}, use_sim=True, sim_platform="mujoco")
    for stream in ("camera.front.depth", "camera.front.aligned_depth"):
        configured = descriptor["interfaces"][stream]["configured_profile"]
        assert (configured["width"], configured["height"]) == (16, 12), f"{stream} uses the rendered dims"
    # gazebo renders its rgbd sensor at the colour dims/rate too
    # (runtime-deployment-peripherals D8): the depth streams advertise what
    # the sensor actually renders, never the device's depth_* overrides.
    gazebo = build_description({**profile, "peripherals": peripherals}, use_sim=True, sim_platform="gazebo")
    for stream in ("camera.front.depth", "camera.front.aligned_depth"):
        configured = gazebo["interfaces"][stream]["configured_profile"]
        assert (configured["width"], configured["height"], configured["fps"]) == (16, 12, 30)
    # off every simulator the device-specific depth dims are still honoured
    sdk = build_description({**profile, "peripherals": peripherals}, use_sim=True, sim_platform="sdk")
    assert sdk["interfaces"]["camera.front.depth"]["configured_profile"]["width"] == 8


# ---------------------------------------------------------------------------
# lekiwi-gazebo-runtime-platform D5/D6: the lidar contract on Gazebo
# ---------------------------------------------------------------------------


def _lidar_profile(enabled_scan=True, fast_lio=None):
    return {
        "runtime": {"name": "test_robot", "version": "1.0.0"},
        "capabilities": {},
        "joints": ["1"],
        "peripherals": [
            {
                "type": "lidar",
                "name": "mid360",
                "driver": "livox_mid360",
                "xfer_format": 1,
                "frame_id": "livox_frame",
                "pointcloud_topic": "/livox/lidar",
                "imu_topic": "/livox/imu",
                **({"scan_converter": {"enabled": enabled_scan, "scan_topic": "/scan"}} if enabled_scan else {}),
            }
        ],
        "fast_lio": fast_lio or {},
    }


def test_gazebo_advertises_the_point_cloud_regardless_of_xfer_format():
    profile = _lidar_profile(enabled_scan=False)
    gazebo = build_description(profile, use_sim=True, sim_platform="gazebo")["interfaces"]
    # the physical driver's CustomMsg transfer cannot be produced by the
    # simulator: the cloud is advertised as PointCloud2 (D5)
    assert gazebo["lidar.mid360.points"]["message_type"] == "sensor_msgs/msg/PointCloud2"
    assert gazebo["lidar.mid360.imu"]["message_type"] == "sensor_msgs/msg/Imu"
    # hardware and sdk keep the driver's wire format
    hardware = build_description(profile, use_sim=False)["interfaces"]
    assert hardware["lidar.mid360.points"]["message_type"] == "livox_ros_driver2/msg/CustomMsg"
    sdk = build_description(profile, use_sim=True, sim_platform="sdk")["interfaces"]
    assert sdk["lidar.mid360.points"]["message_type"] == "livox_ros_driver2/msg/CustomMsg"


def test_gazebo_scan_advertised_iff_its_source_runs():
    """The scan's source is FAST-LIO's registered body cloud: on Gazebo the
    scan is advertised only when lidar odometry runs (D5); hardware and sdk
    keep the scan_converter-driven advertisement."""
    off = build_description(_lidar_profile(fast_lio={"enabled": False}), use_sim=True, sim_platform="gazebo")
    assert not any(key.startswith("lidar.mid360.scan") for key in off["interfaces"])

    on = build_description(
        _lidar_profile(fast_lio={"enabled": True, "sim_params_file": "/tmp/s.yaml"}),
        use_sim=True,
        sim_platform="gazebo",
    )
    assert on["interfaces"]["lidar.mid360.scan"]["endpoint"] == "/scan"
    # its source stream is advertised with it
    assert on["interfaces"]["localization.cloud"]["endpoint"] == "/cloud_registered_body"

    for use_sim, sim_platform in ((False, None), (True, "sdk")):
        other = build_description(
            _lidar_profile(fast_lio={"enabled": False}), use_sim=use_sim, sim_platform=sim_platform
        )
        assert other["interfaces"]["lidar.mid360.scan"]["endpoint"] == "/scan", (use_sim, sim_platform)


def test_gazebo_scan_rule_covers_every_lidar_with_the_converter_enabled():
    profile = _lidar_profile(fast_lio={"enabled": False})
    interfaces = peripheral_interfaces(profile["peripherals"], sim_platform="gazebo", scan_source_runs=False)
    assert not any(key.startswith("lidar.mid360.scan") for key in interfaces)
    interfaces = peripheral_interfaces(profile["peripherals"], sim_platform="gazebo", scan_source_runs=True)
    assert interfaces["lidar.mid360.scan"]["endpoint"] == "/scan"


def test_gazebo_depth_streams_advertise_the_rendered_colour_dims():
    # D8: the gz rgbd sensor renders one camera at the colour dims/rate; the
    # depth streams advertise exactly that, never the device's depth_* overrides.
    camera = {
        "type": "camera",
        "name": "front",
        "driver": "realsense",
        "width": 1280,
        "height": 720,
        "fps": 15,
        "depth_width": 640,
        "depth_height": 360,
        "depth_fps": 30,
        "align_depth": True,
        "streams": ["color", "depth"],
    }
    profile = {"runtime": {"name": "t", "version": "1"}, "capabilities": {}, "peripherals": [camera]}
    description = build_description(profile, use_sim=True, sim_platform="gazebo")
    for stream in ("camera.front.depth", "camera.front.aligned_depth"):
        configured = description["interfaces"][stream]["configured_profile"]
        assert (configured["width"], configured["height"], configured["fps"]) == (1280, 720, 15)
    # off the simulator the device-specific depth dims are still honoured
    sdk = build_description(profile, use_sim=True, sim_platform="sdk")
    depth = sdk["interfaces"]["camera.front.depth"]["configured_profile"]
    assert (depth["width"], depth["height"], depth["fps"]) == (640, 360, 30)


# ---------------------------------------------------------------------------
# lekiwi-mujoco-runtime-platform D5/D6: the lidar contract on MuJoCo
# ---------------------------------------------------------------------------


def _mujoco_lidar_profile(controllers_path, *, driver="livox_mid360", update_rate=100):
    controllers_path.write_text(
        "controller_manager:\n  ros__parameters:\n    update_rate: " + str(update_rate) + "\n",
        encoding="utf-8",
    )
    profile = _lidar_profile(enabled_scan=False, fast_lio={"enabled": True})
    profile["ros2_control"] = {"controllers_config": str(controllers_path)}
    profile["peripherals"][0]["driver"] = driver
    return profile


def test_mujoco_advertises_the_point_cloud_and_the_imu_rate(tmp_path):
    profile = _mujoco_lidar_profile(tmp_path / "controllers.yaml")
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert interfaces["lidar.mid360.points"]["message_type"] == "sensor_msgs/msg/PointCloud2"
    assert interfaces["lidar.mid360.imu"]["message_type"] == "sensor_msgs/msg/Imu"
    # the IMU rate is the effective controller-manager rate, not a constant (D5.3)
    assert interfaces["lidar.mid360.imu"]["rate_hz"] == 100.0
    # a different controllers file advertises its own rate
    profile = _mujoco_lidar_profile(tmp_path / "controllers2.yaml", update_rate=250)
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert interfaces["lidar.mid360.imu"]["rate_hz"] == 250.0


def test_mujoco_does_not_advertise_lidars_it_does_not_simulate(tmp_path):
    profile = _mujoco_lidar_profile(tmp_path / "controllers.yaml", driver="ldlidar")
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert not any(key.startswith("lidar.") for key in interfaces), "ldlidar is not simulated on MuJoCo (D5)"
    # the same driver stays advertised on Gazebo (it simulates it) and off-sim
    gazebo = build_description(profile, use_sim=True, sim_platform="gazebo")["interfaces"]
    assert gazebo["lidar.mid360.scan"]["message_type"] == "sensor_msgs/msg/LaserScan"
    hardware = build_description(profile, use_sim=False)["interfaces"]
    assert hardware["lidar.mid360.scan"]["message_type"] == "sensor_msgs/msg/LaserScan"


def test_mujoco_without_a_simulated_lidar_needs_no_controllers_config():
    # SO-101-shaped: no lidar at all and no ros2_control.controllers_config —
    # the IMU rate must not be demanded for a profile without an integrated IMU
    profile = _lidar_profile(enabled_scan=False)
    profile["peripherals"] = []
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert not any(key.startswith("lidar.") for key in interfaces)


def test_mujoco_imu_rate_follows_the_effective_inventory_not_the_raw_profile(tmp_path):
    """Review round 1: the IMU rate is computed from the same effective
    inventory (profile + deployment fragment) the description itself uses —
    a fragment that adds or disables the lidar decides whether a rate is
    needed at all, and a disabled lidar never demands a controllers file."""
    # fragment ADDS the lidar: the raw profile has none, the effective list
    # has the MID-360 — the IMU entry carries the controller-manager rate
    bare = _lidar_profile(enabled_scan=False)
    bare["peripherals"] = []
    bare["ros2_control"] = {"controllers_config": str(tmp_path / "controllers.yaml")}
    (tmp_path / "controllers.yaml").write_text(
        "controller_manager:\n  ros__parameters:\n    update_rate: 125\n", encoding="utf-8"
    )
    effective = _lidar_profile(enabled_scan=False, fast_lio={"enabled": True})["peripherals"]
    interfaces = build_description(bare, peripherals=effective, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert interfaces["lidar.mid360.imu"]["rate_hz"] == 125.0

    # fragment DISABLES the lidar: no IMU is advertised and a missing
    # controllers_config must not fail the description
    disabled = _mujoco_lidar_profile(tmp_path / "unused.yaml")
    disabled["peripherals"][0]["enabled"] = False
    interfaces = build_description(disabled, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert not any(key.startswith("lidar.") for key in interfaces)


def test_mujoco_points_rate_honours_the_platform_update_rate(tmp_path):
    """The advertised cloud rate resolves the same precedence as the composed
    MJCF: simulation.mujoco.update_rate over publish_freq (review round 1)."""
    profile = _mujoco_lidar_profile(tmp_path / "controllers.yaml")
    lidar = profile["peripherals"][0]
    lidar["publish_freq"] = 10.0
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert interfaces["lidar.mid360.points"]["rate_hz"] == 10.0  # publish_freq stands

    lidar["simulation"] = {"mujoco": {"update_rate": 20.0}}
    interfaces = build_description(profile, use_sim=True, sim_platform="mujoco")["interfaces"]
    assert interfaces["lidar.mid360.points"]["rate_hz"] == 20.0  # the platform override wins

    # other platforms never read the mujoco section
    gazebo = build_description(profile, use_sim=True, sim_platform="gazebo")["interfaces"]
    assert gazebo["lidar.mid360.points"]["rate_hz"] == 10.0


def test_mujoco_imu_rate_fails_closed_on_a_missing_update_rate(tmp_path):
    profile = _mujoco_lidar_profile(tmp_path / "controllers.yaml")
    (tmp_path / "controllers.yaml").write_text("controller_manager:\n  ros__parameters:\n", encoding="utf-8")
    with pytest.raises(Exception, match="update_rate missing or invalid"):
        build_description(profile, use_sim=True, sim_platform="mujoco")


def test_platform_peripherals_filters_the_excluded_physics_platforms():
    from robot_runtime.interface_description import platform_peripherals

    for platform in ("gazebo", "mujoco"):
        kept = platform_peripherals(_EXCLUSION_INVENTORY, platform)
        assert [entry["name"] for entry in kept] == ["wrist"]
    # hardware and the SDK execution ignore the key entirely
    for passthrough in (None, "", "sdk"):
        assert platform_peripherals(_EXCLUSION_INVENTORY, passthrough) == _EXCLUSION_INVENTORY


def test_build_description_hides_excluded_peripherals_on_physics_platforms():
    profile = {
        "runtime": {"name": "exclusion_probe", "version": "0.0.1"},
        "capabilities": {},
        "joints": ["1"],
        "peripherals": _EXCLUSION_INVENTORY,
        "ros2_control": {"controllers_config": "/tmp/none.yaml"},
    }
    for platform in ("gazebo", "mujoco"):
        description = build_description(profile, use_sim=True, sim_platform=platform)
        assert not any(key.startswith("camera.front.") for key in description["interfaces"])
        assert not any(key.startswith("audio.") for key in description["interfaces"])
        assert any(key.startswith("camera.wrist.") for key in description["interfaces"])
    # the SDK execution sees the full inventory (speaker default-disabled in
    # this fixture: enabling it to assert it is advertised there)
    inventory = [dict(_EXCLUDED_SPEAKER, enabled=True), _EXCLUDED_MIC, _EXCLUDED_CAMERA, _SIMULATED_WRIST]
    sdk = build_description({**profile, "peripherals": inventory}, use_sim=True, sim_platform="sdk")
    assert any(key.startswith("audio.") for key in sdk["interfaces"])
    assert any(key.startswith("camera.front.") for key in sdk["interfaces"])
    physical = build_description({**profile, "peripherals": inventory}, use_sim=False)
    assert any(key.startswith("camera.front.") for key in physical["interfaces"])
