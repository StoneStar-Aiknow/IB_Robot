"""Perception capabilities from the real platforms' sensor traits (D6).

Moved from ``robot_runtime/test/test_peripherals.py``: resolving the
first-party traits goes through the registry, so these cases run where the
platform plugins are installed.
"""

from __future__ import annotations

from robot_runtime.peripherals import perception_capabilities

# ---------------------------------------------------------------------------
# lekiwi-mujoco-runtime-platform D5.4: capabilities from the platform-filtered
# projection
# ---------------------------------------------------------------------------


def _lidar_inventory():
    return [
        {
            "type": "lidar",
            "name": "mid360",
            "driver": "livox_mid360",
            "xfer_format": 1,
            "frame_id": "livox_frame",
            "pointcloud_topic": "/livox/lidar",
            "imu_topic": "/livox/imu",
        },
        {
            "type": "lidar",
            "name": "ld06",
            "driver": "ldlidar",
            "scan_topic": "/ld06/scan",
            "frame_id": "ld06_frame",
        },
    ]


def _traits_for(sim_platform):
    """The first-party platform's effective sensor traits (sim-platform-registry D4)."""
    from robot_runtime.interface_description import effective_sensor_traits

    return effective_sensor_traits({}, None, True, sim_platform)


def test_perception_capabilities_follow_the_platform_filter():
    mujoco = perception_capabilities(_lidar_inventory(), sim_platform="mujoco", traits=_traits_for("mujoco"))
    # MuJoCo stands in for the MID-360 only: its cloud and IMU endpoints
    assert mujoco["perception.lidar"]["topics"] == ["/livox/lidar", "/livox/imu"]
    # Gazebo simulates both drivers; the sdk surface is unfiltered
    gazebo = perception_capabilities(_lidar_inventory(), sim_platform="gazebo", traits=_traits_for("gazebo"))
    assert set(gazebo["perception.lidar"]["topics"]) == {"/livox/lidar", "/livox/imu", "/ld06/scan"}
    sdk = perception_capabilities(_lidar_inventory(), sim_platform="sdk")
    assert set(sdk["perception.lidar"]["topics"]) == {"/livox/lidar", "/livox/imu", "/ld06/scan"}
    hardware = perception_capabilities(_lidar_inventory(), sim_platform=None)
    assert hardware == sdk


def test_perception_capabilities_drop_an_unsupported_only_lidar_on_mujoco():
    ldlidar_only = [_lidar_inventory()[1]]
    assert perception_capabilities(ldlidar_only, sim_platform="mujoco", traits=_traits_for("mujoco")) == {}
    # every other platform still projects it
    assert perception_capabilities(ldlidar_only, sim_platform="gazebo", traits=_traits_for("gazebo"))[
        "perception.lidar"
    ]["topics"] == ["/ld06/scan"]
    assert perception_capabilities(ldlidar_only, sim_platform="sdk")["perception.lidar"]["topics"] == ["/ld06/scan"]
