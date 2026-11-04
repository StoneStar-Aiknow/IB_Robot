"""The Gazebo platform's default embodiment (openspec sim-platform-registry D3/D6).

Moved from ``robot_runtime.launch_support`` with no logic changes: the
Gazebo description renderer, the embodiment launch composition and the
``SimEmbodiment`` wrapper the platform spec names. The runtime core
(``robot_runtime``) dispatches here through the registry's ``module:attr``
reference and must not import this module.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

from robot_runtime.interface_description import platform_peripherals
from robot_runtime.launch_support import (
    RuntimeProfileError,
    _inject_initial_values,
    _resolve_package_uris,
    profile_controllers,
    render_xacro,
    runtime_facade_node,
    spawner_nodes,
    xacro_value,
)
from robot_runtime.path_subst import resolve_path
from robot_runtime.peripherals import (
    load_peripherals_file,
    merge_peripherals,
    peripheral_tf_nodes_for_absent_frames,
)
from robot_runtime_sim.gazebo import (
    bridge_config_node,
    camera_bridge_entries,
    camera_sensor_urdf,
    imu_bridge_entries,
    imu_settle_filter_nodes,
    lidar_bridge_entries,
    lidar_finite_filter_nodes,
    lidar_imu_sensor_urdf,
)
from robot_runtime_sim.sim_nodes import simulated_joint_current_actions
from robot_runtime_sim.world import gazebo_world_actions

#: The provider-alone minimal empty world (physics + user commands + scene
#: broadcaster + the sensors system so runtime-owned cameras render, a light
#: and a ground plane). Shipped by robot_runtime_sim; the stock gz empty.sdf
#: has no Sensors system plugin, so runtime cameras would never publish in it.
_EMPTY_WORLD_NAME = "empty"


def _empty_world_path() -> str:
    return str(Path(get_package_share_directory("robot_runtime_sim")) / "worlds" / "empty.sdf")


def _on_success(start_actions, success_message: str, failure_reason: str):
    """Run launch actions only when the target process exits successfully."""

    def _handler(event, _context):
        if event.returncode == 0:
            return start_actions
        return [EmitEvent(event=Shutdown(reason=failure_reason))]

    return _handler


def render_gazebo_description(
    profile: dict[str, Any],
    profile_path: Path,
    *,
    peripherals: list | None = None,
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> str:
    """Render the profile's xacro for the Gazebo platform (D5, D7b).

    ``use_sim`` selects the ros2_control simulation plugin (gz
    ``GazeboSimSystem``), the SDK flag stays off, the gz plugin reads the
    profile's controllers parameter file and the effort state interface is
    exposed for the simulated joint-current publisher (D6b). Camera, lidar
    and IMU sensors are injected for the effective peripherals — pass the
    resolved inventory (D5 inventory-first); the profile's own peripherals
    apply when omitted — meshes resolve to absolute ``file://`` URIs and the
    SDK/hardware descriptions are untouched. Each joint in
    ``hardware.reset_positions`` starts, and is held, at that position (D4:
    the position state interface carries the ``initial_value``).
    """
    description = profile.get("description") or {}
    if not description.get("xacro"):
        raise RuntimeProfileError(f"runtime profile missing required key 'description.xacro': {profile_path}")
    ros2_control = profile.get("ros2_control") or {}
    controllers_config = ros2_control.get("controllers_config") or ""
    xacro_args = {str(k): xacro_value(v) for k, v in (description.get("xacro_args") or {}).items()}
    xacro_args.update(
        {
            "use_sim": "true",
            "simulated": "false",  # the SDK virtual transport flag is off under Gazebo
            "sim_plugin": "gz_ros2_control/GazeboSimSystem",
            "sim_effort_state": "true",
            **(
                {"gz_ros2_control_parameters_file": str(Path(resolve_path(controllers_config)).resolve())}
                if controllers_config
                else {}
            ),
        }
    )
    for key, value in (profile.get("hardware") or {}).get("xacro_args", {}).items():
        xacro_args[str(key)] = xacro_value(value)
    robot_description = _inject_initial_values(render_xacro(resolve_path(description["xacro"]), xacro_args), profile)
    effective = platform_peripherals(
        merge_peripherals(profile.get("peripherals"), []) if peripherals is None else peripherals, "gazebo"
    )
    sensors_xml = camera_sensor_urdf(
        effective,
        platform="gazebo",
        spawn_pose=spawn_pose,
        frame_contract=True,
    )
    sensors_xml += lidar_imu_sensor_urdf(effective, platform="gazebo")
    if sensors_xml:
        robot_description = robot_description.replace("</robot>", sensors_xml + "\n</robot>", 1)
    return _resolve_package_uris(robot_description)


def gazebo_embodiment_actions(
    profile: dict[str, Any],
    profile_path: Path,
    peripherals_file: str = "",
    instance_id: str = "",
    *,
    robot_description: str | None = None,
    interface_description: dict | None = None,
    gazebo_world: str = "",
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list:
    """The provider's Gazebo embodiment (D2/D5): no ``ros2_control_node``.

    The effective peripheral inventory (profile defaults merged with the
    deployment fragment, disabled removed) is resolved once, before anything
    else, and drives everything downstream: the sensor generation inside the
    rendered description (when this call renders it), the bridge entries and
    the static transforms for sensor frames absent from the rendered model
    (D5 inventory-first). robot_state_publisher runs on sim time; a minimal
    empty world plus ``/clock`` bridge through the shared launcher (D6) when
    no world is supplied (standalone verification); ``create`` waits for the
    world clock, then spawns the
    entity at the given pose (including yaw); controller spawners target the
    in-simulator controller manager after ``create`` exits; the facade and
    the simulated joint-current publisher complete the embodiment. The world,
    when supplied by generic launch, is not started or bridged here.
    """
    # D5 inventory-first: one effective inventory for sensors, bridges,
    # derived processors and the interface description — filtered by the
    # platform exclusion policy (D3: an excluded peripheral is not simulated).
    peripherals = platform_peripherals(
        merge_peripherals(
            profile.get("peripherals"),
            (load_peripherals_file(peripherals_file) if peripherals_file else {}).get("peripherals"),
        ),
        "gazebo",
    )
    if robot_description is None:
        robot_description = render_gazebo_description(
            profile, profile_path, peripherals=peripherals, spawn_pose=spawn_pose
        )
    ros2_control = profile.get("ros2_control") or {}
    if not ros2_control.get("controllers_config"):
        raise RuntimeProfileError(
            f"runtime profile missing required key 'ros2_control.controllers_config': {profile_path}"
        )
    broadcasters, active, inactive = profile_controllers(profile, "gazebo")
    controller_manager = str(profile["controller_manager"])
    model_name = str(profile["runtime"]["name"])

    actions: list[Any] = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}, {"use_sim_time": True}],
        )
    ]

    world_name = (gazebo_world or "").strip()
    if not world_name:
        # Standalone: bring a minimal empty world so the provider is verifiable
        # in isolation; the world's clock is bridged here because nobody else
        # owns it (the shared launcher honours IBROBOT_GAZEBO_HEADLESS).
        actions.extend(gazebo_world_actions(_empty_world_path(), _EMPTY_WORLD_NAME))
        world_name = _EMPTY_WORLD_NAME

    clock_waiter = Node(
        package="robot_runtime_sim",
        executable="wait_for_clock",
        name="wait_for_gazebo_clock",
        output="screen",
        arguments=["--topic", "/clock", "--timeout", "60.0"],
    )
    x, y, z, yaw = (float(value) for value in spawn_pose)
    create_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-world",
            world_name,
            "-name",
            model_name,
            "-allow_renaming=true",
            "-topic",
            "/robot_description",
            "-x",
            str(x),
            "-y",
            str(y),
            "-z",
            str(z),
            "-Y",
            str(yaw),
        ],
        output="screen",
    )
    spawners = spawner_nodes(broadcasters, active, inactive, controller_manager=controller_manager, use_sim_time=True)
    actions.extend(
        [
            clock_waiter,
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=clock_waiter,
                    on_exit=_on_success(
                        [create_entity], "creating robot entity", "Gazebo clock probe failed before robot spawn"
                    ),
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=create_entity,
                    on_exit=_on_success(
                        spawners, "entity created; spawning controllers", "Gazebo entity creation failed"
                    ),
                )
            ),
            runtime_facade_node(
                profile,
                profile_path,
                peripherals_file,
                use_sim=True,
                sim_platform="gazebo",
                instance_id=instance_id,
                interface_description=interface_description,
                use_sim_time=True,
            ),
            *simulated_joint_current_actions(profile),
        ]
    )
    camera_entries = [
        entry
        for periph in peripherals
        if periph.get("type") == "camera"
        for entry in camera_bridge_entries(periph, full_streams=True)
    ]
    lidar_imu_entries = [
        entry for periph in peripherals for entry in (*lidar_bridge_entries(periph), *imu_bridge_entries(periph))
    ]
    sensor_entries = camera_entries + lidar_imu_entries
    if sensor_entries:
        # One unique config per construction (D9): concurrent launches on
        # different domains or worktrees must not overwrite each other's file.
        fd, sensor_bridge_path = tempfile.mkstemp(prefix="ibrobot_runtime_sensor_bridge_", suffix=".yaml")
        os.close(fd)
        actions.append(bridge_config_node(sensor_entries, sensor_bridge_path))
    # The finite-point filter serves the advertised cloud endpoints of the
    # 3D lidars (D5: the bridge hands the raw gz cloud to the internal
    # topic, the filter republishes the finite points on the endpoint); the
    # IMU settle filter holds the raw IMU back until the physics has settled
    # (zero window and landing impulse dropped), then relays it verbatim.
    actions.extend(lidar_finite_filter_nodes(peripherals))
    actions.extend(imu_settle_filter_nodes(peripherals))
    # Static TF only for sensor frames the rendered model does not contain
    # (the lidar/IMU frames); camera frames stay owned by RSP through their
    # rendered links (D5, D13).
    actions.extend(peripheral_tf_nodes_for_absent_frames(peripherals, robot_description))
    return actions


# ---------------------------------------------------------------------------
# The SimEmbodiment wrapper (sim-platform-registry D3): the platform's
# default embodiment, referenced by its spec as a ``module:attr`` string and
# resolved lazily through the registry, so no platform name appears in the
# dispatch code.
# ---------------------------------------------------------------------------


class _GazeboEmbodiment:
    """The Gazebo platform's default :class:`SimEmbodiment` (D3)."""

    def render_description(
        self,
        profile: dict[str, Any],
        profile_path: Path,
        *,
        peripherals: list | None = None,
        spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        world: str = "",
    ) -> str:
        return render_gazebo_description(profile, profile_path, peripherals=peripherals, spawn_pose=spawn_pose)

    def actions(
        self,
        profile: dict[str, Any],
        profile_path: Path,
        peripherals_file: str = "",
        instance_id: str = "",
        *,
        robot_description: str | None = None,
        interface_description: dict | None = None,
        world: str = "",
        spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    ) -> list:
        return gazebo_embodiment_actions(
            profile,
            profile_path,
            peripherals_file,
            instance_id,
            robot_description=robot_description,
            interface_description=interface_description,
            gazebo_world=world,
            spawn_pose=spawn_pose,
        )


GAZEBO_EMBODIMENT = _GazeboEmbodiment()
