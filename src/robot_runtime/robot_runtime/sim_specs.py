"""The Gazebo and MuJoCo platform specs (openspec sim-platform-registry D1/D4).

Import-light by contract (D1: stdlib plus ``robot_runtime.sim_platforms``
types only — building an interface description imports a spec module, so it
must never import ``launch``, a simulator package or ``sim_models``). The
embodiment and trait callables stay ``module:attr`` strings resolved lazily
by the dispatch entries in ``launch_support``; the callables themselves live
next to the platform code they read (``robot_runtime.gazebo`` /
``robot_runtime.mujoco``).
"""

from __future__ import annotations

from robot_runtime.sim_platforms import SimPlatform, SimSensorTraits

#: The lidar drivers the Gazebo platform stands in for (D5); drivers outside
#: the set get no simulated sensor and are not advertised on the platform.
_GAZEBO_LIDAR_SIM_DRIVERS = ("livox_mid360", "ldlidar")

#: The lidar drivers whose physical device carries an integrated IMU.
_INTEGRATED_IMU_DRIVERS = ("livox_mid360",)

#: The lidar drivers the MuJoCo platform stands in for (D5) — only the
#: MID-360 in this stage: its ray grid maps to ``mujoco.plugin.lidar`` and
#: its integrated IMU to the co-located MJCF sensor triple; 2D lidars have
#: no MuJoCo counterpart yet.
_MUJOCO_LIDAR_SIM_DRIVERS = ("livox_mid360",)

GAZEBO_SENSOR_TRAITS = SimSensorTraits(
    drop_pointcloud=True,
    depth_dims_follow_color=True,
    depth_fps_follow_color=True,
    forced_encodings={"depth": "32FC1"},
    shared_depth_camera_info=True,
    camera_qos_reliable=False,
    camera_rate=None,
    lidar_sim_drivers=_GAZEBO_LIDAR_SIM_DRIVERS,
    lidar_points_rate=None,
    integrated_imu_drivers=_INTEGRATED_IMU_DRIVERS,
    imu_rate=None,
)

MUJOCO_SENSOR_TRAITS = SimSensorTraits(
    drop_pointcloud=True,
    depth_dims_follow_color=True,
    depth_fps_follow_color=False,
    forced_encodings={"color": "rgb8", "depth": "32FC1"},
    shared_depth_camera_info=False,
    camera_qos_reliable=True,
    camera_rate="robot_runtime.sim_specs:mujoco_camera_rate",
    lidar_sim_drivers=_MUJOCO_LIDAR_SIM_DRIVERS,
    lidar_points_rate="robot_runtime.sim_specs:mujoco_lidar_points_rate",
    integrated_imu_drivers=_INTEGRATED_IMU_DRIVERS,
    imu_rate="robot_runtime.sim_specs:mujoco_imu_rate",
)

GAZEBO_SPEC = SimPlatform(
    name="gazebo",
    package="robot_runtime",
    embodiment="robot_runtime.launch_support:GAZEBO_EMBODIMENT",
    world=None,
    scene_task=None,
    sensor_traits=GAZEBO_SENSOR_TRAITS,
)

MUJOCO_SPEC = SimPlatform(
    name="mujoco",
    package="robot_runtime",
    embodiment="robot_runtime.launch_support:MUJOCO_EMBODIMENT",
    world=None,
    scene_task=None,
    sensor_traits=MUJOCO_SENSOR_TRAITS,
)


# --- lazily resolved trait callables (module:attr targets) -------------------


def mujoco_camera_rate(profile: dict) -> float | None:
    """The one rate every MuJoCo camera stream advertises (D4/D5).

    Delegates to the MuJoCo platform module, which owns the rate algorithm.
    """
    from robot_runtime.mujoco import mujoco_camera_publish_rate

    return mujoco_camera_publish_rate(profile)


def mujoco_lidar_points_rate(periph: dict) -> float | None:
    """The MID-360 points-rate override from ``simulation.mujoco.update_rate``.

    The composed MJCF honours ``simulation.mujoco.update_rate`` over
    ``publish_freq``, so the description must resolve the same precedence —
    never claim a rate the simulation does not produce.
    """
    mujoco_section = (periph.get("simulation") or {}).get("mujoco") or {}
    if isinstance(mujoco_section, dict) and mujoco_section.get("update_rate") is not None:
        return mujoco_section["update_rate"]
    return None


def mujoco_imu_rate(profile: dict, peripherals: list | None) -> float | None:
    """The rate the integrated IMU produces on MuJoCo (D4/D5.3), when one runs.

    The ``imu_sensor_broadcaster`` publishes on the controller-manager clock,
    so the produced rate is the effective ``controller_manager.update_rate``
    of the profile's ``ros2_control.controllers_config``. Returns ``None``
    (no override — no rate is advertised) when the effective inventory
    carries no integrated-IMU lidar the platform simulates or the profile
    has no ``ros2_control`` block (D4: a robot without ros2_control must
    still describe); a profile that declares one but whose controllers
    configuration is missing or unreadable fails closed, exactly as before.
    """
    integrated_here = set(_MUJOCO_LIDAR_SIM_DRIVERS) & set(_INTEGRATED_IMU_DRIVERS)
    effective = peripherals if peripherals is not None else (profile.get("peripherals") or [])
    effective = [entry for entry in effective if entry.get("enabled", True)]
    if not any(
        periph.get("type") == "lidar" and str(periph.get("driver", "")) in integrated_here for periph in effective
    ):
        return None
    import yaml

    from robot_runtime.path_subst import resolve_path

    config = str(((profile.get("ros2_control") or {}).get("controllers_config") or "").strip())
    if not config:
        return None  # D4: a profile without ros2_control declares no IMU rate override
    path = resolve_path(config)
    try:
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read the controllers configuration for the MuJoCo IMU rate: {exc}") from exc
    rate = ((data.get("controller_manager") or {}).get("ros__parameters") or {}).get("update_rate")
    if not isinstance(rate, int | float) or rate <= 0:
        raise ValueError(
            f"controller_manager.update_rate missing or invalid in {path}: the MuJoCo IMU rate cannot be advertised"
        )
    return float(rate)
