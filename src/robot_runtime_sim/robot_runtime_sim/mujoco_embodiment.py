"""The MuJoCo platform's default embodiment (openspec sim-platform-registry D3/D6).

Moved from ``robot_runtime.launch_support`` with no logic changes: the MuJoCo
description renderer, the embodiment launch composition and the
``SimEmbodiment`` wrapper the platform spec names. The runtime core
(``robot_runtime``) dispatches here through the registry's ``module:attr``
reference and must not import this module.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from xml.dom import minidom

import yaml
from launch.actions import LogInfo
from launch_ros.actions import Node

from robot_runtime.interface_description import platform_peripherals
from robot_runtime.launch_support import (
    RuntimeProfileError,
    _inject_initial_values,
    _platform_xacro_args,
    _resolve_package_uris,
    profile_controllers,
    render_xacro,
    runtime_facade_node,
    spawner_nodes,
    write_controller_manager_params,
    write_node_params,
    xacro_value,
)
from robot_runtime.path_subst import resolve_path
from robot_runtime.peripherals import (
    load_peripherals_file,
    merge_peripherals,
    peripheral_tf_nodes_for_absent_frames,
)
from robot_runtime_sim.mujoco import (
    MUJOCO_MODEL_PLACEHOLDER,
    base_velocity_plugin_parameters,
    camera_plugin_parameters,
    camera_tf_urdf,
    check_simulator_versions,
    free_joint_state_plugin_parameters,
    imu_broadcaster_entries,
    imu_broadcaster_remappings,
    imu_sensor_urdf,
    lidar_plugin_parameters,
    merge_mujoco_plugin_parameters,
    mujoco_camera_relays,
    mujoco_imu_filter_nodes,
    mujoco_lidar_filter_nodes,
    odometry_twist_relay_node,
    render_robot_mjcf,
)
from robot_runtime_sim.sim_nodes import simulated_joint_current_actions


def render_mujoco_description(
    profile: dict[str, Any],
    profile_path: Path,
    peripherals: list | None = None,
    *,
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    scene: str = "",
) -> str:
    """Render the MuJoCo-platform robot description (URDF) for RSP + the plugin (D2/D3/D5).

    Steps: the simulator version check (``check_simulator_versions`` — a
    CameraPlugin-capable ``mujoco_ros2_control`` and a Python ``mujoco``
    matching the simulator's libmujoco are hard preconditions, D2); the
    runtime-private robot MJCF (template + meshes from ``simulation.mujoco``,
    cameras injected, base at the spawn pose including yaw); the composed
    robot+scene model (``compose_mujoco_model``, the domain-keyed ``.mjb`` —
    a missing ``mujoco`` Python package is fatal on the provider path); the
    xacro URDF with ``sim_plugin=mujoco_ros2_control/MujocoSystemInterface``,
    ``simulated=false``, ``mujoco_model=<composed path>``,
    ``auto_register_cameras=false`` (the deprecated sensor auto-registration
    stays off; cameras run through the CameraPlugin parameters, D3),
    ``sim_effort_state=true`` and ``mujoco_headless`` from
    ``IBROBOT_MUJOCO_HEADLESS``; the camera TF (mount + optical frames). The
    URDF carries no camera ``<sensor>`` blocks.

    Xacro-arg precedence (review round 1, item 10): the same chain as the
    Gazebo render — ``description.xacro_args`` <
    ``simulation.mujoco.xacro_args`` (platform-scoped) < this renderer's
    fixed platform keys (``use_sim``, ``simulated``, ``sim_plugin``,
    ``sim_effort_state``, ``mujoco_model``, ``mujoco_headless`` — not
    overridable by the description or platform args) <
    ``hardware.xacro_args`` (last, overrides everything). A profile
    without the platform section renders byte-identically.
    """
    from sim_models.scene_compiler import compose_mujoco_model, mjcf_engine_extensions

    description = profile.get("description") or {}
    if not description.get("xacro"):
        raise RuntimeProfileError(f"runtime profile missing required key 'description.xacro': {profile_path}")
    check_simulator_versions()
    # D3: the MuJoCo sensor generation, plugin parameters and description all
    # project the platform-filtered inventory (an excluded peripheral is not
    # simulated).
    effective_peripherals = platform_peripherals(
        profile.get("peripherals") if peripherals is None else peripherals, "mujoco"
    )
    # D2 render order: the xacro renders ONCE with the platform's arguments
    # and a placeholder mujoco_model; the attachment mount poses come from
    # that document; the composed model path is substituted into the same
    # rendered URDF afterwards (the mujoco_model parameter is the only
    # difference — no second xacro run).
    xacro_args = {str(k): xacro_value(v) for k, v in (description.get("xacro_args") or {}).items()}
    xacro_args.update(_platform_xacro_args(profile, "mujoco"))
    xacro_args.update(
        {
            "use_sim": "true",
            "simulated": "false",  # the SDK virtual transport flag is off under MuJoCo
            "sim_plugin": _MUJOCO_SYSTEM_PLUGIN,
            "sim_effort_state": "true",
            "mujoco_model": MUJOCO_MODEL_PLACEHOLDER,
            # env-derived key: always passed (the xacros default it to
            # false); like every arg here, hardware.xacro_args below still
            # overrides it, and the simulator uses that final value
            "mujoco_headless": "true" if os.environ.get("IBROBOT_MUJOCO_HEADLESS", "0") == "1" else "false",
        }
    )
    for key, value in (profile.get("hardware") or {}).get("xacro_args", {}).items():
        xacro_args[str(key)] = xacro_value(value)
    robot_description = _inject_initial_values(render_xacro(resolve_path(description["xacro"]), xacro_args), profile)
    robot_mjcf = render_robot_mjcf(profile, spawn_pose, effective_peripherals, urdf=robot_description)
    sdf_meshes = ((profile.get("simulation") or {}).get("mujoco") or {}).get("sdf_meshes") or []
    # D2/D3 revision (5b.2): an empty scene composes with the sim_models
    # ``empty`` scene (ground plane + light) — a robot-only model would render
    # on a black void. ``scene=None`` (an explicit robot-only request from a
    # direct API caller) keeps the robot-only composition.
    scene_name = None if scene is None else ((scene or "").strip() or "empty")
    composed = compose_mujoco_model(str(robot_mjcf), scene_name, sdf_meshes)
    if composed.suffix == ".mjb":
        pass  # the extension-free artifact, as before
    elif composed.suffix == ".xml" and mjcf_engine_extensions(composed):
        # D5.1: the model declares engine extensions, so the compiler
        # delivered the XML — a .mjb would bake this process's global plugin
        # slot ids, which the C++ node (its own registration order) cannot
        # read back (spike evidence /tmp/lekiwi-mujoco-evidence/spike/SUMMARY.md);
        # the C++ node compiles the XML in-process with its own registration
        pass
    else:
        raise RuntimeProfileError(
            f"the MuJoCo embodiment needs the mujoco Python package (composition fell back to {composed})"
        )
    robot_description = robot_description.replace(MUJOCO_MODEL_PLACEHOLDER, str(composed), 1)
    cameras_xml = camera_tf_urdf(effective_peripherals, spawn_pose=spawn_pose)
    if cameras_xml:
        robot_description = robot_description.replace("</robot>", cameras_xml + "\n</robot>", 1)
    # D5.3: the integrated-IMU sensors go into the rendered ros2_control
    # block, where the system interface's register_sensors reads them
    imus_xml = imu_sensor_urdf(effective_peripherals)
    if imus_xml:
        robot_description = robot_description.replace("</ros2_control>", imus_xml + "\n  </ros2_control>", 1)
    return _resolve_package_uris(robot_description)


# The operator's key guide for the MuJoCo viewer window (simulate's own
# bindings; the viewer has no API for a custom on-screen label, so the guide
# goes to the launch log).
MUJOCO_VIEWER_KEYS = (
    "MuJoCo viewer keys: Ctrl+A = back to the scene's start view | "
    "Esc = free camera (left-drag rotate, right-drag pan, scroll zoom) | "
    "{cycle} | "
    "Ctrl + right double-click on a body = track that body"
)
_MUJOCO_SYSTEM_PLUGIN = "mujoco_ros2_control/MujocoSystemInterface"


def _simulator_model_path(robot_description: str) -> str | None:
    """The model file the simulator will load, or ``None`` when it cannot be named.

    The ``mujoco_model`` parameter of the MuJoCo hardware component, read the
    way the simulator reads it (see ``_mujoco_hardware_param``), with the
    surrounding whitespace mujoco_ros2_control trims removed. ``None`` for no
    MuJoCo hardware component, a component without ``mujoco_model`` (the
    simulator then loads the model from a topic), or components that name
    different models.
    """
    values = _mujoco_hardware_param(robot_description, "mujoco_model")
    if not values or None in values:
        return None
    # mujoco_ros2_control trims this set of whitespace off the path
    paths = {value.strip(" \t\n\r\f\v") for value in values}
    return paths.pop() if len(paths) == 1 else None


def mujoco_viewer_camera_order(robot_description: str) -> list[str] | None:
    """The fixed cameras in the order the viewer's ']' key cycles them.

    simulate cycles ``mjModel`` cameras by id (']' from the free camera
    starts at id 0, '[' starts at the last id), and the ids follow the
    compiled body tree — they mix recorded and viewer-only cameras, so the
    order is read from the composed model the node will load
    (``_simulator_model_path``, loaded by ``mujoco_model_camera_names``).
    ``None`` when that model cannot be named or read here; the guide is
    informational, so no failure here stops the launch.
    """
    path = _simulator_model_path(robot_description)
    if path is None:
        return None
    try:
        from sim_models.scene_compiler import mujoco_model_camera_names

        names = mujoco_model_camera_names(path)
    except Exception:  # noqa: BLE001 - the guide is informational, never fatal
        return None
    return [name or f"camera {i}" for i, name in enumerate(names)]


def _composed_model_cameras(robot_description: str) -> list[str]:
    """The composed model's camera names for the CameraPlugin parameters.

    The plugin publishes every model camera it is not told to disable, so a
    model that cannot be named or read fails the launch here instead of
    publishing cameras nobody declared.
    """
    path = _simulator_model_path(robot_description)
    if path is None:
        raise RuntimeProfileError(
            "the CameraPlugin is configured but the MuJoCo hardware component names no single "
            "mujoco_model, so its cameras cannot be listed"
        )
    from sim_models.scene_compiler import mujoco_model_camera_names

    try:
        return mujoco_model_camera_names(path)
    except Exception as exc:
        raise RuntimeProfileError(f"cannot list the cameras of the MuJoCo model {path!r}: {exc}") from exc


def _first_text(element) -> str | None:
    """TinyXML2 ``GetText`` on a minidom element.

    The first child text node after any leading comments, as TinyXML2 9.0
    builds them: whitespace-only text nodes are dropped, a CDATA section is a
    text node of its own, and neither a comment nor a CDATA boundary joins
    the text around it. Any other first child means no text.

    Checked against the native GetText on xacro-rendered descriptions, the
    only input here. Raw XML the xacro render would normalise can still
    differ: an empty CDATA section (minidom drops it) or a leading
    character reference to a space or NBSP (Python's ``str.strip`` treats it
    as layout whitespace).
    """
    for node in element.childNodes:
        if node.nodeType == node.COMMENT_NODE:
            continue
        if node.nodeType == node.TEXT_NODE and not node.data.strip():
            continue
        if node.nodeType in (node.TEXT_NODE, node.CDATA_SECTION_NODE):
            return node.data
        return None
    return None


def _child_elements(element, tag: str) -> list:
    return [node for node in element.childNodes if node.nodeType == node.ELEMENT_NODE and node.tagName == tag]


def _mujoco_hardware_param(robot_description: str, name: str) -> list[str | None]:
    """One value of parameter ``name`` per MuJoCo hardware component.

    Read the way the simulator reads the rendered description, following
    ros2_control 2.54's ``parse_control_resources_from_urdf``: each
    ``<ros2_control>`` directly under the root is one component. Every
    ``<hardware>`` child of it, in order, replaces the component's plugin
    (its first ``<plugin>``), and a ``<hardware>`` with ``<param>`` children
    replaces the parameters, so the last ``<hardware>`` names the plugin and
    the last one with parameters supplies them. Only components whose plugin
    is the MuJoCo system count. A repeated parameter keeps its last value;
    each value is TinyXML2's ``GetText`` (see ``_first_text``), and a
    parameter without text is ``""`` as in the parser. ``None`` when the
    component has no such parameter.
    """
    root = minidom.parseString(robot_description).documentElement
    values = []
    for control in _child_elements(root, "ros2_control"):
        plugin, params = None, {}
        for hardware in _child_elements(control, "hardware"):
            plugins = _child_elements(hardware, "plugin")
            plugin = (_first_text(plugins[0]) or "") if plugins else None
            param_elements = _child_elements(hardware, "param")
            if param_elements:
                params = {param.getAttribute("name"): _first_text(param) or "" for param in param_elements}
        if plugin == _MUJOCO_SYSTEM_PLUGIN:
            values.append(params.get(name))
    return values


def mujoco_viewer_is_headless(robot_description: str) -> bool:
    """Whether the simulator runs without the Simulate window.

    ``IBROBOT_MUJOCO_HEADLESS`` sets the ``mujoco_headless`` xacro arg, and
    a profile's ``hardware.xacro_args`` (applied last) can override it, so
    the final value is read from the rendered description's MuJoCo hardware
    blocks (``_mujoco_hardware_param``) with ``hardware_interface::parse_bool``,
    which accepts exactly ``true`` and ``True``. With more than one MuJoCo
    hardware component, a window exists unless every one is headless.
    """
    flags = [value in ("true", "True") for value in _mujoco_hardware_param(robot_description, "headless")]
    return bool(flags) and all(flags)


def mujoco_viewer_hint_actions(robot_description: str) -> list:
    """The viewer key guide as a launch log line; none when running headless."""
    if mujoco_viewer_is_headless(robot_description):
        return []
    order = mujoco_viewer_camera_order(robot_description)
    if order:
        cycle = "] / [ = cycle the fixed cameras: " + " -> ".join(order) + " (] from the first, [ from the last)"
    else:
        cycle = "] / [ = cycle the model's fixed cameras"
    return [LogInfo(msg=MUJOCO_VIEWER_KEYS.format(cycle=cycle))]


# The node name mujoco_ros2_control's hardware interface hard-codes for the
# node that carries the simulator (and CameraPlugin) parameters. Global
# --params-file arguments reach it because it is created with default node
# options and automatically_declare_parameters_from_overrides.
MUJOCO_SIMULATION_NODE = "mujoco_ros2_control_node"


def mujoco_embodiment_actions(
    profile: dict[str, Any],
    profile_path: Path,
    peripherals_file: str = "",
    instance_id: str = "",
    *,
    robot_description: str | None = None,
    interface_description: dict | None = None,
    mujoco_scene: str = "",
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list:
    """The provider's MuJoCo embodiment (D2): one ``mujoco_ros2_control`` process.

    That process is both the simulator and the controller manager and takes
    the place of the provider's own ``ros2_control_node``: robot_state_publisher,
    spawners, the facade, the camera depth fan-out relays and the simulated
    joint-current publisher all run on sim time against it. No world process
    is started here — the composed model already contains the scene (an
    empty ``mujoco_scene`` composes the shared sim_models ``empty`` scene;
    only a direct API caller passing ``scene=None`` gets a robot-only model).
    The spawners start immediately; their
    ``--controller-manager-timeout`` waits for the in-process controller
    manager. ``IBROBOT_MUJOCO_HEADLESS=1`` mirrors the Gazebo headless mode:
    with no display the CameraPlugin renders through EGL (D4), so no
    ``xvfb-run`` wrapper is needed. The cameras (and, for scenes with
    dynamic objects, the FreeJointStatePublisherPlugin the scene task
    observes, issue #195) are configured through a per-launch params file
    keyed under ``MUJOCO_SIMULATION_NODE`` carrying the merged
    ``mujoco_plugins`` tree; MuJoCo plugins load from the executable
    directory and ament resources only, so no ``MUJOCO_PLUGIN_PATH`` is set.
    """
    if robot_description is None:
        robot_description = render_mujoco_description(profile, profile_path, spawn_pose=spawn_pose, scene=mujoco_scene)
    ros2_control = profile.get("ros2_control") or {}
    if not ros2_control.get("controllers_config"):
        raise RuntimeProfileError(
            f"runtime profile missing required key 'ros2_control.controllers_config': {profile_path}"
        )
    controllers_config = resolve_path(ros2_control["controllers_config"])
    peripherals = platform_peripherals(
        merge_peripherals(
            profile.get("peripherals"),
            (load_peripherals_file(peripherals_file) if peripherals_file else {}).get("peripherals"),
        ),
        "mujoco",
    )
    # D5.3: the generated IMU broadcasters ride a GENERATED controllers file
    # merging the profile's shipped configuration (the shipped file is never
    # edited); they join the broadcaster set — always active, in no mode.
    broadcaster_entries = imu_broadcaster_entries(peripherals)
    broadcasters, active, inactive = profile_controllers(profile, "mujoco")
    if broadcaster_entries:
        controllers_config = _merged_controllers_config(controllers_config, broadcaster_entries)
        broadcasters = [*broadcasters, *broadcaster_entries]
    floating_base = ((profile.get("simulation") or {}).get("mujoco") or {}).get("floating_base") or {}
    cm_params = write_controller_manager_params(robot_description)
    mujoco_node_parameters = [cm_params, controllers_config, {"use_sim_time": True}]
    # One params file carries every simulator plugin: the cameras, the lidars,
    # the floating-base drive and, when the composed scene has dynamic
    # objects, the free-joint state publisher the scene task observes
    # (issue #195). Once the CameraPlugin is configured, every camera of the
    # composed model gets an explicit policy, or the plugin streams it
    # (viewer-only cameras are disabled).
    model_cameras = _composed_model_cameras(robot_description) if camera_plugin_parameters(profile, peripherals) else ()
    plugin_parameters = merge_mujoco_plugin_parameters(
        camera_plugin_parameters(profile, peripherals, model_cameras),
        free_joint_state_plugin_parameters(mujoco_scene),
        base_velocity_plugin_parameters(floating_base),
        lidar_plugin_parameters(peripherals),
    )
    if plugin_parameters:
        mujoco_node_parameters.append(write_node_params(MUJOCO_SIMULATION_NODE, plugin_parameters))
    return [
        *mujoco_viewer_hint_actions(robot_description),
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}, {"use_sim_time": True}],
        ),
        Node(
            package="mujoco_ros2_control",
            executable="ros2_control_node",
            output="screen",
            parameters=mujoco_node_parameters,
            # D5.3: process-wide remaps put the IMU broadcasters' ~/imu on the
            # internal /<name>_imu/raw topics (the settle filter's input)
            remappings=imu_broadcaster_remappings(peripherals),
        ),
        *spawner_nodes(
            broadcasters,
            active,
            inactive,
            controller_manager=str(profile["controller_manager"]),
            timeout_s=60.0,
            use_sim_time=True,
        ),
        runtime_facade_node(
            profile,
            profile_path,
            peripherals_file,
            use_sim=True,
            sim_platform="mujoco",
            instance_id=instance_id,
            interface_description=interface_description,
            use_sim_time=True,
        ),
        *simulated_joint_current_actions(profile, sim_platform="mujoco"),
        *mujoco_camera_relays(peripherals),
        # D5.3: the raw lidar cloud and IMU streams land on the advertised
        # endpoints through the shared filters (the cloud gains its zero
        # intensity field; the IMU waits out the spawn settle)
        *mujoco_lidar_filter_nodes(peripherals),
        *mujoco_imu_filter_nodes(peripherals),
        # D3.3: the base's own wheel odometry drives the floating base
        *odometry_twist_relay_node(floating_base),
        # D5: static TF only for sensor frames the rendered model lacks
        *peripheral_tf_nodes_for_absent_frames(peripherals, robot_description),
    ]


def _merged_controllers_config(shipped: Path | str, generated: dict) -> Path:
    """The shipped controllers configuration plus generated entries (D5.3).

    Loads the shipped file, merges the generated controllers' type entries
    under ``controller_manager.ros__parameters`` and their parameter sections
    at the top level, and writes the result to a per-launch temporary file —
    the shipped file itself is never edited.
    """
    shipped_path = Path(shipped)
    data = yaml.safe_load(shipped_path.read_text(encoding="utf-8")) or {}
    manager = data.setdefault("controller_manager", {})
    manager_params = manager.setdefault("ros__parameters", {})
    for name, params in generated.items():
        manager_params[name] = {"type": "imu_sensor_broadcaster/IMUSensorBroadcaster"}
        data[name] = {"ros__parameters": dict(params)}
    out = Path(tempfile.mkdtemp(prefix="runtime_controllers_")) / "controllers.yaml"
    out.write_text(yaml.safe_dump(data, default_flow_style=False), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# The SimEmbodiment wrapper (sim-platform-registry D3): the platform's
# default embodiment, referenced by its spec as a ``module:attr`` string and
# resolved lazily through the registry, so no platform name appears in the
# dispatch code.
# ---------------------------------------------------------------------------


class _MujocoEmbodiment:
    """The MuJoCo platform's default :class:`SimEmbodiment` (D3)."""

    def render_description(
        self,
        profile: dict[str, Any],
        profile_path: Path,
        *,
        peripherals: list | None = None,
        spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
        world: str = "",
    ) -> str:
        scene = None if world is None else ((world or "").strip() or "empty")
        return render_mujoco_description(profile, profile_path, peripherals, spawn_pose=spawn_pose, scene=scene)

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
        return mujoco_embodiment_actions(
            profile,
            profile_path,
            peripherals_file,
            instance_id,
            robot_description=robot_description,
            interface_description=interface_description,
            mujoco_scene=world,
            spawn_pose=spawn_pose,
        )


MUJOCO_EMBODIMENT = _MujocoEmbodiment()
