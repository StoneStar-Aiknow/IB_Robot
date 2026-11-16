"""Launch-time helpers shared by ``<robot>_robot`` runtime launch files.

Runtime launch entries may depend on ``robot_runtime`` and ``ibrobot_msgs``
only (robot-runtime-packaging spec), so the pieces of ros2_control bring-up
that are not robot-specific live here rather than in a generic IB-Robot
launch package: profile path resolution, xacro rendering, the
controller_manager parameter file, and controller spawners derived from the
profile's mode table.

The simulation platforms are open (openspec ``sim-platform-registry``): the
generic dispatch entries (``render_sim_description``,
``sim_embodiment_actions``, ``runtime_stack_actions``) resolve the effective
embodiment through the platform registry and never name a platform. The
Gazebo and MuJoCo implementations live in ``robot_runtime_sim`` and are
reached only through their specs' ``module:attr`` references.
"""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node

from robot_runtime.contract import IDLE_MODE
from robot_runtime.interface_description import build_description, effective_sensor_traits, platform_peripherals
from robot_runtime.path_subst import resolve_path
from robot_runtime.peripherals import (
    audio_nodes,
    load_peripherals_file,
    merge_peripherals,
    peripheral_nodes,
    scan_converter_nodes,
    synthetic_perception_nodes,
)
from robot_runtime.sim_platforms import (
    SDK_PLATFORM,
    SimPlatformError,
    get_sim_platform,
    is_physics_selection,
    is_valid_sim_platform_name,
    resolve_sim_embodiment,
)

logger = logging.getLogger(__name__)


def render_xacro(xacro_path: str, args: dict[str, Any]) -> str:
    """Run xacro and return the URDF string. Raises on failure with xacro's stderr."""
    command = ["xacro", xacro_path] + [f"{key}:={value}" for key, value in args.items()]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"xacro failed for {xacro_path}:\n{result.stderr}")
    return result.stdout


def xacro_value(value: Any) -> str:
    """Render a Python value as a xacro argument (bools lowercase, dicts as JSON)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, dict | list):
        import json

        return json.dumps(value)
    return str(value)


def _inject_initial_values(robot_description: str, profile: dict[str, Any]) -> str:
    """Give the ros2_control position state interfaces an ``initial_value`` (D4).

    On the physics platforms the simulated joints start, and are held, at the
    profile's ``hardware.reset_positions``: each listed joint's position
    state interface in the rendered URDF's ``<ros2_control>`` block gains
    ``<param name="initial_value">`` (gz_ros2_control spawns the joint there;
    mujoco_ros2_control's ``set_initial_pose()`` also sets the position
    actuators' ``ctrl`` from it). Profiles without ``reset_positions`` render
    byte-identical; the hardware/SDK render never calls this.

    Fail-closed validation (review rework): every reset value must be a
    finite number (booleans, NaN and Inf are rejected by name); a reset
    joint the ros2_control block does not declare, or whose position state
    interface is absent, is an error — a reset that silently skips joints
    would start the arm somewhere the profile did not intend; a position
    state interface that already carries parameters (in particular its own
    ``initial_value``) is a conflict the profile author must resolve, not a
    preference the injection makes silently.
    """
    reset = (profile.get("hardware") or {}).get("reset_positions") or {}
    if not reset:
        return robot_description
    import math
    import re

    reset = {str(name): value for name, value in reset.items()}
    for name, value in reset.items():
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise RuntimeProfileError(f"hardware.reset_positions.{name} must be a finite number, got {value!r}")

    # ros2_control joints carry no attributes on <joint>; the URDF's
    # kinematic joints always carry type=… and never match
    joint_re = re.compile(r'<joint name="([^"]+)">(.*?)</joint>', re.DOTALL)
    found: set[str] = set()

    def patch(match: re.Match[str]) -> str:
        name, body = match.group(1), match.group(2)
        if "hardwareInterface" in body:
            # a <transmission>'s joint REFERENCE (no attributes on <joint>,
            # no state interfaces): not the ros2_control block — the real
            # one follows later in the document (the rendered LeKiwi URDF
            # carries both for every joint)
            return match.group(0)
        if name not in reset:
            return match.group(0)
        found.add(name)
        bare = '<state_interface name="position"/>'
        open_tag = '<state_interface name="position">'
        if bare in body:
            patched = body.replace(
                bare,
                '<state_interface name="position">\n'
                f'          <param name="initial_value">{float(reset[name]):.9g}</param>\n'
                "        </state_interface>",
                1,
            )
            return f'<joint name="{name}">{patched}</joint>'
        if open_tag in body:
            if '<param name="initial_value">' in body:
                raise RuntimeProfileError(
                    f"hardware.reset_positions.{name} conflicts with the initial_value the URDF's "
                    "ros2_control block already declares for its position state interface"
                )
            raise RuntimeProfileError(
                f"the position state interface of ros2_control joint {name!r} already carries "
                "parameters; the reset injection only patches the bare interface"
            )
        raise RuntimeProfileError(
            f"hardware.reset_positions.{name} names a joint whose ros2_control block declares "
            "no position state interface"
        )

    patched_description = joint_re.sub(patch, robot_description)
    missing = sorted(set(reset) - found)
    if missing:
        raise RuntimeProfileError(
            f"hardware.reset_positions names joints the ros2_control block does not declare: {missing}"
        )
    return patched_description


def write_node_params(node_name: str, params: dict[str, Any]) -> str:
    """Write ``params`` under ``node_name`` as a ros__parameters YAML file.

    ros2_control_node creates its nodes with the executable-derived names, so
    a parameter file keyed by the node name is the reliable way to hand over
    parameters without a global ``__node`` remapping.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, prefix="runtime_cm_") as handle:
        yaml.safe_dump({node_name: {"ros__parameters": params}}, handle, default_flow_style=False)
        return handle.name


def write_controller_manager_params(robot_description: str, extra_params: dict[str, Any] | None = None) -> str:
    """Write robot_description under the ``controller_manager`` node name.

    ros2_control_node creates a node called ``controller_manager`` while launch
    writes dict parameters under the executable name, so a parameter file with
    the correct key is the reliable way to hand over the description without
    a global ``__node`` remapping.
    """
    params = {"robot_description": robot_description}
    if extra_params:
        params.update(extra_params)
    return write_node_params("controller_manager", params)


def profile_controllers(
    profile: dict[str, Any], sim_platform: str | None = None
) -> tuple[list[str], list[str], list[str]]:
    """Return (broadcasters, initially_active, initially_inactive) controllers from the profile.

    Broadcasters are always active. The initial mode's controllers start
    active; every other controller named by any mode starts inactive so the
    facade can switch activation sets without spawning. ``sim_platform``
    selects the mode table's per-platform overrides (D3.1) through the single
    resolver :func:`robot_runtime.modes.mode_controller_table`, so the
    spawners, the facade and the stop path all see one controller table.
    """
    from robot_runtime.modes import mode_controller_table

    modes = dict(profile.get("modes") or {})
    initial = str(modes.pop("initial", IDLE_MODE))
    broadcasters = [str(c) for c in (profile.get("ros2_control") or {}).get("broadcasters", [])]
    controller_table = mode_controller_table(profile, sim_platform)
    active: list[str] = []
    inactive: list[str] = []
    for name, controllers in controller_table.items():
        for controller in controllers:
            controller = str(controller)
            target = active if str(name) == initial else inactive
            if controller not in target and controller not in active:
                target.append(controller)
    inactive = [c for c in inactive if c not in active]
    return broadcasters, active, inactive


def spawner_nodes(
    broadcasters: list[str],
    active: list[str],
    inactive: list[str],
    controller_manager: str = "controller_manager",
    timeout_s: float = 30.0,
    use_sim_time: bool = False,
) -> list[Node]:
    """Stock controller_manager spawners: broadcasters + active together, inactive with ``--inactive``."""
    nodes: list[Node] = []
    common = ["--controller-manager", controller_manager, "--controller-manager-timeout", str(int(timeout_s))]
    parameters = [{"use_sim_time": use_sim_time}] if use_sim_time else []
    if broadcasters or active:
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                name="runtime_spawner_active",
                arguments=[*broadcasters, *active, *common],
                parameters=parameters,
                output="screen",
            )
        )
    if inactive:
        nodes.append(
            Node(
                package="controller_manager",
                executable="spawner",
                name="runtime_spawner_inactive",
                arguments=[*inactive, "--inactive", *common],
                parameters=parameters,
                output="screen",
            )
        )
    return nodes


def profile_share_path(package: str, profile_name: str) -> Path:
    """Path of ``<package>/profiles/<profile_name>.yaml`` in the installed share directory."""
    return Path(get_package_share_directory(package)) / "profiles" / f"{profile_name}.yaml"


# ---------------------------------------------------------------------------
# Whole-stack composition shared by <robot>_robot runtime launch files
# ---------------------------------------------------------------------------


class RuntimeProfileError(ValueError):
    """A runtime profile lacks a key the launch entry needs (named in the message)."""


def resolve_profile_argument(package: str, raw: str) -> Path:
    """``profile:=`` is a file path or the name of a profile shipped in ``<package>/profiles``."""
    raw = (raw or "").strip()
    if not raw:
        raise RuntimeProfileError(
            f"{package} runtime.launch.py requires profile:=<path-or-name> (robot-runtime-packaging: no default robot)"
        )
    candidate = Path(raw).expanduser()
    if candidate.is_file():
        return candidate
    shipped = profile_share_path(package, raw)
    if shipped.is_file():
        return shipped
    raise RuntimeProfileError(
        f"runtime profile not found: {raw!r} (searched as a path and as {package}/profiles/{raw}.yaml)"
    )


def _parse_bool_argument(value: str, name: str) -> bool:
    text = (value or "").strip().lower()
    if text in ("1", "true", "yes"):
        return True
    if text in ("0", "false", "no"):
        return False
    raise RuntimeProfileError(f"cannot parse {name}:={value!r} as a boolean")


def resolve_sim_selection(
    profile: dict[str, Any],
    use_sim_arg: str,
    sim_platform_arg: str,
    simulated_arg: str,
    *,
    source: str = "",
) -> tuple[bool, str | None]:
    """Resolve the effective simulation selection (runtime-simulation-platform D1).

    Uses generic launch's vocabulary: ``use_sim:=`` selects simulated or
    physical execution, ``sim_platform:=`` selects the simulation platform.
    ``sim_platform:=X`` alone implies simulation on X; ``use_sim:=true``
    alone runs the profile's default platform; ``use_sim:=false`` with a
    platform fails naming both. ``simulated:=`` stays a deprecated alias of
    ``use_sim:=`` (a warning names ``use_sim``); disagreeing with ``use_sim:=``
    fails naming both. A platform the profile does not declare fails naming
    the platform, the declared set and the profile. A physics platform that
    is not registered (sim-platform-registry D2: the effective selection is
    the checkpoint) fails naming the platform and the package to install.
    """
    use_sim_explicit: bool | None = None
    simulated_override = (simulated_arg or "").strip().lower()
    if simulated_override:
        use_sim_explicit = _parse_bool_argument(simulated_override, "simulated")
        logger.warning(
            "simulated:=%s is deprecated; use use_sim:=%s (same effect)",
            simulated_override,
            str(use_sim_explicit).lower(),
        )
    use_sim_override = (use_sim_arg or "").strip().lower()
    if use_sim_override:
        use_sim_value = _parse_bool_argument(use_sim_override, "use_sim")
        if use_sim_explicit is not None and use_sim_explicit != use_sim_value:
            raise RuntimeProfileError(
                f"conflicting selection: use_sim:={use_sim_value} and simulated:={str(use_sim_explicit).lower()}; "
                "pass at most one"
            )
        use_sim_explicit = use_sim_value

    sim_platform = (sim_platform_arg or "").strip().lower()
    if sim_platform and not is_valid_sim_platform_name(sim_platform):
        raise RuntimeProfileError(
            f"unknown sim_platform {sim_platform!r}: expected an identifier matching ^[a-z][a-z0-9_]*$"
        )

    # A platform alone implies simulation; otherwise the profile's selection
    # applies unless an argument overrode it.
    if use_sim_explicit is None:
        # the profile-level `simulated` key stays the legacy alias of `use_sim`
        use_sim = bool(sim_platform) or bool(profile.get("use_sim", profile.get("simulated", False)))
    else:
        use_sim = use_sim_explicit
    if not use_sim:
        if sim_platform:
            raise RuntimeProfileError(
                f"physical execution (use_sim:=false) cannot select sim_platform:={sim_platform!r}; pass at most one"
            )
        return False, None

    platform = sim_platform or str(profile.get("sim_platform") or SDK_PLATFORM)
    supported = [str(entry) for entry in (profile.get("sim_platforms") or [SDK_PLATFORM])]
    if platform not in supported:
        raise RuntimeProfileError(
            f"sim_platform {platform!r} is not supported by this runtime profile (supported: {supported}, "
            f"default: {profile.get('sim_platform', 'sdk')!r}{f'; profile: {source}' if source else ''})"
        )
    if platform != SDK_PLATFORM:
        # D2: registration is required for the effective selection — this is
        # the runtime-entry checkpoint; an unregistered selection fails as a
        # profile error naming the platform and the package to install.
        try:
            get_sim_platform(platform)
        except SimPlatformError as exc:
            raise RuntimeProfileError(str(exc)) from exc
    return True, platform


def render_robot_description(profile: dict[str, Any], profile_path: Path, simulated: bool) -> str:
    """Render the profile's xacro with hardware parameters; fails fast on missing keys or calibration."""
    description = profile.get("description") or {}
    if not description.get("xacro"):
        raise RuntimeProfileError(f"runtime profile missing required key 'description.xacro': {profile_path}")
    hardware = profile.get("hardware") or {}
    xacro_args = {str(k): xacro_value(v) for k, v in (description.get("xacro_args") or {}).items()}
    calib = resolve_path(hardware["calib_file"]) if hardware.get("calib_file") else ""
    xacro_args.update(
        {
            "simulated": xacro_value(simulated),
            "port": str(hardware.get("port", "")),
            "calib_file": calib,
            "reset_positions": xacro_value(hardware.get("reset_positions") or {}),
        }
    )
    for key, value in (hardware.get("xacro_args") or {}).items():
        xacro_args[str(key)] = xacro_value(value)
    if not simulated and calib and not Path(calib).is_file():
        raise RuntimeProfileError(
            f"calibration file not found: {calib} (profile {profile_path}); "
            "run the calibration tool declared by the deployment before launching the runtime"
        )
    return render_xacro(resolve_path(description["xacro"]), xacro_args)


_PACKAGE_URI_PATTERN = None


def _resolve_package_uris(robot_description: str) -> str:
    """Rewrite ``package://<pkg>/...`` mesh URIs to absolute ``file://`` paths.

    A description spawned on the Gazebo platform must be resolvable by a
    simulator server the provider did not configure (D7b): no resource-path
    environment hand-off is needed when the URDF carries absolute file URIs.
    """
    import re

    global _PACKAGE_URI_PATTERN
    if _PACKAGE_URI_PATTERN is None:
        _PACKAGE_URI_PATTERN = re.compile(r"package://(\w+)/")

    def replace(match: re.Match[str]) -> str:
        package = match.group(1)
        try:
            share = Path(get_package_share_directory(package))
        except Exception as exc:  # PackageNotFoundError at render time is fatal
            raise RuntimeProfileError(
                f"description references package://{package}/ but {package} is not in the ament index: {exc}"
            ) from exc
        return share.as_uri() + "/"

    return _PACKAGE_URI_PATTERN.sub(replace, robot_description)


def _platform_xacro_args(profile: dict[str, Any], platform: str) -> dict[str, str]:
    """The profile's ``simulation.<platform>.xacro_args`` (empty when absent).

    Platform-scoped xacro args (review round 1, item 10): a robot's
    platform-specific render arguments live in its PROFILE under
    ``simulation.<platform>.xacro_args``, not in a renderer — a generic
    renderer must not inject robot-specific args into every render. Consumed
    by the physics embodiments (robot_runtime_sim), like ``profile_controllers``.
    """
    section = (profile.get("simulation") or {}).get(platform) or {}
    return {str(key): xacro_value(value) for key, value in (section.get("xacro_args") or {}).items()}


def render_sim_description(
    profile: dict[str, Any],
    profile_path: Path,
    sim_platform: str | None,
    *,
    peripherals: list | None = None,
    simulated: bool = True,
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    world: str = "",
) -> str:
    """Render the profile's robot description for the selection (D3).

    Hardware and the SDK platform keep :func:`render_robot_description`
    (unchanged); every other simulated selection is physics by D1 and renders
    through the effective embodiment — the profile's
    ``simulation.<platform>.embodiment`` override when declared, else the
    platform's default (resolved lazily through the registry, so no platform
    name appears here).
    """
    if not is_physics_selection(simulated, sim_platform):
        return render_robot_description(profile, profile_path, simulated)
    platform = get_sim_platform(str(sim_platform))
    embodiment = resolve_sim_embodiment(profile, profile_path, platform)
    return embodiment.render_description(
        profile, profile_path, peripherals=peripherals, spawn_pose=spawn_pose, world=world
    )


def sim_embodiment_actions(
    profile: dict[str, Any],
    profile_path: Path,
    sim_platform: str | None,
    peripherals_file: str = "",
    instance_id: str = "",
    *,
    robot_description: str | None = None,
    interface_description: dict | None = None,
    world: str = "",
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list | None:
    """The effective embodiment's launch actions for a physics platform (D3).

    Returns the embodiment's actions, or ``None`` for hardware and sdk — they
    have no embodiment and the caller keeps the stock stack for them.
    """
    if not is_physics_selection(True, sim_platform):
        return None
    platform = get_sim_platform(str(sim_platform))
    embodiment = resolve_sim_embodiment(profile, profile_path, platform)
    return embodiment.actions(
        profile,
        profile_path,
        peripherals_file,
        instance_id,
        robot_description=robot_description,
        interface_description=interface_description,
        world=world,
        spawn_pose=spawn_pose,
    )


def runtime_facade_node(
    profile: dict[str, Any],
    profile_path: Path,
    peripherals_file: str = "",
    *,
    use_sim: bool,
    sim_platform: str = "",
    instance_id: str = "",
    interface_description: dict | None = None,
    use_sim_time: bool = False,
) -> Node:
    """The runtime facade node, the one builder both stack paths share (D3).

    The hardware/sdk stack and every physics embodiment construct the facade
    identically apart from the platform identity and the sim-time flag, which
    this helper carries as parameters.
    """
    return Node(
        package="robot_runtime",
        executable="runtime_facade",
        name="runtime_facade",
        output="screen",
        parameters=[
            {
                "profile": str(profile_path),
                "peripherals": peripherals_file,
                "use_sim": use_sim,
                "initial_mode": str((profile.get("modes") or {}).get("initial", IDLE_MODE)),
                **({"sim_platform": sim_platform} if sim_platform else {}),
                **({"use_sim_time": True} if use_sim_time else {}),
                **(
                    {"interface_description_json": json.dumps(interface_description)}
                    if interface_description is not None
                    else {}
                ),
                **({"instance_id": instance_id} if instance_id else {}),
            }
        ],
    )


def runtime_stack_actions(
    profile: dict[str, Any],
    profile_path: Path,
    simulated: bool,
    peripherals_file: str = "",
    instance_id: str = "",
    *,
    robot_description: str | None = None,
    interface_description: dict | None = None,
    sim_platform: str = "",
    sim_world: str = "",
    spawn_pose: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> list:
    """robot_state_publisher + ros2_control_node + controller spawners + runtime facade.

    ``peripherals_file`` (deployment overrides; profile ``peripherals:`` are
    the robot's defaults) additionally composes the sensor driver nodes, or
    synthetic perception publishers in simulated execution (D12), and feeds
    the facade so RuntimeStatus declares the perception surface.
    ``sim_platform`` is forwarded to the facade so its public description
    carries the selected platform. A physics platform dispatches to its
    embodiment (D3): the effective embodiment — the profile's
    ``simulation.<platform>.embodiment`` override when declared, else the
    registered platform's default — renders and composes the stack; hardware
    and sdk are unchanged. ``sim_world`` is the platform-neutral world
    hand-off (the world name on Gazebo, the scene name on MuJoCo).
    """
    if is_physics_selection(simulated, sim_platform):
        return sim_embodiment_actions(
            profile,
            profile_path,
            str(sim_platform),
            peripherals_file,
            instance_id,
            robot_description=robot_description,
            interface_description=interface_description,
            world=sim_world,
            spawn_pose=spawn_pose,
        )
    if robot_description is None:
        robot_description = render_robot_description(profile, profile_path, simulated)
    ros2_control = profile.get("ros2_control") or {}
    if not ros2_control.get("controllers_config"):
        raise RuntimeProfileError(
            f"runtime profile missing required key 'ros2_control.controllers_config': {profile_path}"
        )
    controllers_config = resolve_path(ros2_control["controllers_config"])
    cm_params = write_controller_manager_params(robot_description)
    broadcasters, active, inactive = profile_controllers(profile, sim_platform or None)
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[cm_params, controllers_config],
            remappings=[("~/robot_description", "/robot_description")],
            output="screen",
        ),
        *spawner_nodes(broadcasters, active, inactive, controller_manager=str(profile["controller_manager"])),
        runtime_facade_node(
            profile,
            profile_path,
            peripherals_file,
            use_sim=simulated,
            sim_platform=sim_platform,
            instance_id=instance_id,
            interface_description=interface_description,
        ),
    ]


def peripheral_stack_actions(
    profile: dict, peripherals_file: str, simulated: bool, *, description: dict | None = None, sim_platform: str = ""
) -> list:
    """Sensor peripheral composition for a runtime launch (D12).

    Physical drivers + static TF on real hardware; synthetic perception
    publishers on the same topics on the SDK simulation platform. Under a
    physics platform the camera and lidar streams come from the simulator's
    sensors and bridges (embodiment, D6), so only the host audio devices
    remain here — plus the derived scan converter, which runs on simulation
    time when its source (FAST-LIO's registered body cloud) runs on the
    platform (D5). A lidar the effective traits exclude from the platform's
    simulated surface starts no derived node either (D4: the description, the
    capability surface and the composed nodes project the same inventory —
    a scan converter for a lidar nothing feeds would be a dead node).
    Audio devices are host devices, so their ALSA owners run in EVERY
    execution (the legacy robot_config audio_io entry also launched them
    under simulation).
    """
    fragment = load_peripherals_file(peripherals_file) if peripherals_file else {}
    peripherals = merge_peripherals(profile.get("peripherals"), fragment.get("peripherals"))
    if is_physics_selection(simulated, sim_platform):
        # D3: excluded peripherals are not launched on the platform (the
        # audio owners and the scan converter resolve the filtered inventory)
        peripherals = platform_peripherals(peripherals, sim_platform)
        # D4: the same effective-traits lidar projection the description
        # applies — only a lidar the platform stands in for contributes a
        # derived node.
        traits = effective_sensor_traits(profile, None, simulated, sim_platform)
        if traits is not None and traits.lidar_sim_drivers is not None:
            peripherals = [
                periph
                for periph in peripherals
                if periph.get("type") != "lidar" or str(periph.get("driver", "")) in traits.lidar_sim_drivers
            ]
        fast_lio = {**(profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}
        scan_converters = (
            [
                node
                for periph in peripherals
                if periph.get("type") == "lidar"
                for node in scan_converter_nodes(periph, use_sim=True)
            ]
            if fast_lio.get("enabled", False)
            else []
        )
        return [*audio_nodes(peripherals), *scan_converters]
    if simulated:
        effective = {**profile, "fast_lio": {**(profile.get("fast_lio") or {}), **(fragment.get("fast_lio") or {})}}
        return [
            *synthetic_perception_nodes(
                peripherals, description or build_description(effective, peripherals, use_sim=True, sim_platform="sdk")
            ),
            *audio_nodes(peripherals),
        ]
    return peripheral_nodes(peripherals, use_sim=False)


def motion_launch_arguments(
    profile: dict[str, Any], profile_path: Path, simulated: bool, sim_platform: str, display: bool
) -> dict[str, str]:
    """Launch arguments for a MoveIt-backed motion package (so101_motion/motion.launch.py).

    Every motion node (move_group, motion server, IK workers and the optional
    RViz) runs on the platform's clock: simulation time on a physics platform,
    wall time otherwise (D7).
    """
    motion = profile.get("motion") or {}
    missing = [k for k in ("arm_group_name", "base_link", "ee_link", "shoulder_link") if not motion.get(k)]
    if missing:
        raise RuntimeProfileError(f"runtime profile missing required keys motion.{missing}: {profile_path}")
    trajectory_modes = [
        name
        for name, spec in (profile.get("modes") or {}).items()
        if name != "initial" and (spec or {}).get("allows_trajectory")
    ]
    arm_joints = [str(j) for j in profile.get("arm_joints", profile["joints"])]
    return {
        "use_sim_time": "true" if is_physics_selection(simulated, sim_platform) else "false",
        "display": xacro_value(display),
        "joint_names": " ".join(arm_joints),
        "joint_state_topic": str(motion.get("joint_state_topic", profile["joint_state_topic"])),
        "arm_group_name": str(motion["arm_group_name"]),
        "base_link": str(motion["base_link"]),
        "ee_link": str(motion["ee_link"]),
        "shoulder_link": str(motion["shoulder_link"]),
        "motion_feedback_timeout_s": str(motion.get("motion_feedback_timeout_s", 0.3)),
        "motion_feedback_tolerance_rad": str(motion.get("motion_feedback_tolerance_rad", 0.12)),
        "motion_require_tf_sync": xacro_value(bool(motion.get("motion_require_tf_sync", True))),
        "motion_hardware_feedback_topic": "" if simulated else str(motion.get("motion_hardware_feedback_topic", "")),
        "trajectory_modes": " ".join(trajectory_modes) or "trajectory",
        "ik_worker_count": str(int(motion.get("ik_worker_count", 0))),
    }
