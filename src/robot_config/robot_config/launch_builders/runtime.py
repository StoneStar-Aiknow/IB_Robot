"""Runtime provider launch builder (robot-runtime-packaging: runtime selection by provider name).

When a robot configuration names ``runtime.provider``, robot bring-up
(hardware, controllers, motion services, runtime facade) is delegated to
that provider's ``launch/runtime.launch.py``. Generic launch never references
a robot-specific launch file, model path, or planning-framework launch; it
only resolves the provider through the ament index and gates upper layers on
``RuntimeStatus`` reconciliation (``wait_for_runtime``).

Configuration::

    robot:
      runtime:
        provider: so101_robot          # <provider>/launch/runtime.launch.py
        profile: so101_single_arm      # path, or a profile shipped by the provider
      capabilities:
        requires: [joint.state, joint.trajectory, motion.move_to_pose]

D12: the deployment peripheral inventory (``peripherals:`` and, for lidar
robots, ``navigation.fast_lio``) is serialized into a temp YAML and handed to
the runtime as the ``peripherals`` launch argument -- data passthrough only,
the runtime owns the driver composition.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch.actions import EmitEvent, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

from robot_config.interface_binding import (
    bind_robot_interfaces,
    event_interface_ids,
    required_interface_ids,
    required_interface_requirements,
    semantic_mapping_interface_ids,
)
from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.runtime")

RUNTIME_LAUNCH_FILE = "runtime.launch.py"


def runtime_provider(robot_config: dict[str, Any]) -> str:
    """The configured provider name, or '' when the configuration uses in-config bring-up."""
    return str((robot_config.get("runtime") or {}).get("provider", "") or "").strip()


def runtime_profile(robot_config: dict[str, Any]) -> str:
    return str((robot_config.get("runtime") or {}).get("profile", "") or "").strip()


def required_capabilities(robot_config: dict[str, Any]) -> list[str]:
    return [str(name) for name in ((robot_config.get("capabilities") or {}).get("requires") or [])]


def resolve_runtime_launch(provider: str) -> str:
    """Path of ``<provider>/launch/runtime.launch.py``; fails fast naming provider and search path."""
    if provider == "mock_runtime":
        # The mock runtime is the contract-layer reference implementation
        # shipped inside the robot_runtime package (core-only independence
        # gate provider); its entry is mock_runtime.launch.py.
        try:
            share = get_package_share_directory("robot_runtime")
        except PackageNotFoundError as exc:  # pragma: no cover - robot_config depends on robot_runtime
            raise RuntimeError("robot_runtime is not installed; the mock runtime provider is unavailable") from exc
        launch_path = os.path.join(share, "launch", "mock_runtime.launch.py")
        if os.path.isfile(launch_path):
            return launch_path
    try:
        share = get_package_share_directory(provider)
    except PackageNotFoundError as exc:
        prefixes = os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
        raise RuntimeError(
            f"runtime.provider {provider!r} is not installed (searched AMENT_PREFIX_PATH: {prefixes}). "
            "Install the robot runtime package or fix robot.runtime.provider."
        ) from exc
    launch_path = os.path.join(share, "launch", RUNTIME_LAUNCH_FILE)
    if not os.path.isfile(launch_path):
        raise RuntimeError(
            f"runtime.provider {provider!r} ships no {RUNTIME_LAUNCH_FILE} (looked in {os.path.dirname(launch_path)})"
        )
    return launch_path


def _write_peripherals_fragment(robot_config: dict[str, Any]) -> str:
    """Serialize the deployment peripherals/fast_lio inventory for the runtime.

    The application layer is ``peripherals`` followed by the loader-recorded
    ``runtime._disabled_peripherals`` (runtime-peripheral-overrides D2; a
    missing key means none). The runtime applies this single override layer
    to its profile with ``merge_peripherals``, yielding profile < robot
    configuration; without device fields the serialisation is unchanged
    (byte-identical for shipped configurations).

    Returns "" when the configuration declares nothing beyond profile defaults.
    """
    runtime = robot_config.get("runtime") or {}
    application = list(robot_config.get("peripherals") or []) + list(
        (runtime.get("_disabled_peripherals") if isinstance(runtime, dict) else None) or []
    )
    fast_lio = (robot_config.get("navigation") or {}).get("fast_lio") or {}
    fragment: dict[str, Any] = {}
    if application:
        fragment["peripherals"] = application
    if fast_lio.get("enabled"):
        fragment["fast_lio"] = fast_lio
    if not fragment:
        return ""
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="ibrobot_peripherals_", suffix=".yaml", delete=False, encoding="utf-8"
    ) as handle:
        yaml.safe_dump(fragment, handle, default_flow_style=False, sort_keys=False)
    logger.info(f"peripherals fragment for the runtime: {handle.name}")
    return handle.name


def _cbf_safety_json(robot_config: dict[str, Any]) -> str:
    """Optional X2 data-plane configuration; disabled ancillary fields are inert."""
    config = robot_config.get("cbf_safety")
    if config is None:
        return ""
    if not isinstance(config, dict):
        raise ValueError("robot.cbf_safety must be a mapping")
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("robot.cbf_safety.enabled must be a boolean")
    if not enabled:
        return ""
    if runtime_provider(robot_config) != "aimdk_robot":
        raise ValueError("robot.cbf_safety requires runtime.provider=aimdk_robot")
    mode = config.get("mode", "shadow")
    if mode not in ("shadow", "enforce"):
        raise ValueError("robot.cbf_safety.mode must be shadow or enforce")
    deadline = config.get("deadline_us", 10000)
    guard = config.get("deadline_guard_us", 500)
    if type(deadline) is not int or deadline <= 0:
        raise ValueError("robot.cbf_safety.deadline_us must be a positive integer")
    if type(guard) is not int or not 0 <= guard < deadline:
        raise ValueError("robot.cbf_safety.deadline_guard_us must be an integer in [0, deadline_us)")
    mirror_mode = (robot_config.get("mirror_session") or {}).get("control_mode", "")
    protected = ((robot_config.get("control_modes") or {}).get(mirror_mode) or {}).get("runtime_mode")
    if not isinstance(protected, str) or not protected.strip():
        raise ValueError("robot.cbf_safety requires mirror_session.control_mode to name a control_modes runtime_mode")
    payload = {
        # thumb_mapping and feedback_max_age_s are deployment data: preserve
        # them verbatim, never infer a thumb calibration from joint aliases.
        **config,
        "enabled": True,
        "mode": mode,
        "deadline_us": deadline,
        "deadline_guard_us": guard,
        "protected_mirror_mode": protected.strip(),
    }
    mock = (robot_config.get("motion_imitation_stream") or {}).get("mock_playback") or {}
    velocity, limits = mock.get("arm_velocity_limits"), mock.get("arm_limits")
    if velocity is not None or limits is not None:
        # Reuse the selected profile's arm_pos order without importing the
        # provider package or duplicating its vendor joint layout here.
        from robot_runtime.launch_support import resolve_profile_argument
        from robot_runtime.profile import load_profile

        profile = load_profile(resolve_profile_argument("aimdk_robot", runtime_profile(robot_config)))
        joints = profile.get("arm_joints") or []
        if len(joints) != 14 or len(set(joints)) != 14:
            raise ValueError("robot.cbf_safety requires 14 distinct runtime profile arm_joints")

        def number(value, field):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"motion_imitation_stream.mock_playback.{field} must contain finite numbers")
            return float(value)

        for field, values in (("arm_velocity_limits", velocity), ("arm_limits", limits)):
            if values is not None and (not isinstance(values, dict) or set(values) != set(joints)):
                raise ValueError(f"motion_imitation_stream.mock_playback.{field} must name exactly the 14 arm_joints")
        if velocity is not None:
            ordered = [number(velocity[joint], "arm_velocity_limits") for joint in joints]
            if any(value <= 0 for value in ordered):
                raise ValueError("motion_imitation_stream.mock_playback.arm_velocity_limits must be positive")
            scale = number((mock.get("retarget") or {}).get("vel_scale", 1.0), "retarget.vel_scale")
            if scale <= 0:
                raise ValueError("motion_imitation_stream.mock_playback.retarget.vel_scale must be positive")
            payload["arm_velocity_limits"] = [speed * scale for speed in ordered]
        if limits is not None:
            if any(not isinstance(limits[joint], list | tuple) or len(limits[joint]) != 2 for joint in joints):
                raise ValueError("motion_imitation_stream.mock_playback.arm_limits must contain [lower, upper] pairs")
            lower = [number(limits[joint][0], "arm_limits") for joint in joints]
            upper = [number(limits[joint][1], "arm_limits") for joint in joints]
            if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
                raise ValueError("motion_imitation_stream.mock_playback.arm_limits must have lower < upper")
            # These are the raw URDF bands. The data plane intersects them
            # with its frozen measured bands and applies the safety margin
            # once; duplicating that policy here would add a package dependency.
            payload["arm_lower_limits"], payload["arm_upper_limits"] = lower, upper
    return json.dumps(payload, allow_nan=False)


def generate_runtime_provider_actions(
    robot_config: dict[str, Any],
    *,
    use_sim: bool,
    sim_platform: str = "sdk",
    simulation_startup=None,
    display: bool = False,
    readiness_timeout_s: float = 60.0,
    description_output: str | None = None,
) -> tuple[list, Node]:
    """Return (provider launch include, wait_for_runtime readiness/reconciliation node).

    Generic launch passes its selection through unchanged
    (runtime-simulation-platform D1/D3): ``use_sim:=`` always, plus
    ``sim_platform:=`` when simulated — never the legacy ``simulated`` alias.
    The Gazebo platform additionally hands the provider the resolved world
    name and the full spawn pose from ``simulation_startup``.
    """
    provider = runtime_provider(robot_config)
    if not provider:
        raise RuntimeError("robot.runtime.provider is required for runtime-provider bring-up")
    profile = runtime_profile(robot_config)
    if not profile:
        raise RuntimeError(
            f"robot.runtime.profile is required when runtime.provider={provider!r} "
            "(a path, or a profile name shipped by the provider)"
        )
    launch_path = resolve_runtime_launch(provider)
    peripherals_file = _write_peripherals_fragment(robot_config)
    # Stream-family modes always start idle: policy and teleop must admit
    # explicitly from the observed idle state; only trajectory execution may
    # begin active because it cannot move before a request arrives.
    mode_name = str(robot_config.get("default_control_mode", "") or "")
    mode_config = (robot_config.get("control_modes") or {}).get(mode_name) or {}
    runtime_mode = str(mode_config.get("runtime_mode", "") or "").strip()
    initial_mode = "trajectory" if runtime_mode == "trajectory" else ""
    include_arguments: dict[str, str] = {
        "profile": profile,
        "display": "true" if display else "false",
        "peripherals": peripherals_file,
        **({"initial_mode": initial_mode} if initial_mode else {}),
        **(
            {"instance_id": str(robot_config["runtime"]["instance_id"])}
            if (robot_config.get("runtime") or {}).get("instance_id")
            else {}
        ),
    }
    include_arguments["use_sim"] = "true" if use_sim else "false"
    cbf_safety_json = _cbf_safety_json(robot_config)
    if cbf_safety_json:
        include_arguments["cbf_safety_json"] = cbf_safety_json
    if use_sim:
        include_arguments["sim_platform"] = str(sim_platform).lower()
    platform = str(sim_platform).lower() if use_sim else ""
    if platform == "gazebo":
        # D3: the world owner hands the provider the resolved world name and
        # spawn pose (x, y, z, yaw) — nothing is recomputed here. An empty
        # world name means the provider starts its own minimal empty world.
        startup = simulation_startup
        world_name = str(getattr(startup, "world_name", "") or "")
        spawn_pose = getattr(startup, "spawn_pose", None) or (0.0, 0.0, 0.0, 0.0)
        include_arguments.update(
            {
                "gazebo_world": world_name,
                "spawn_x": str(spawn_pose[0]),
                "spawn_y": str(spawn_pose[1]),
                "spawn_z": str(spawn_pose[2]),
                "spawn_yaw": str(spawn_pose[3]),
            }
        )
        logger.info(
            f"runtime.provider={provider}: including {launch_path} "
            f"(profile={profile}, use_sim=true, sim_platform=gazebo, world={world_name!r}, spawn={spawn_pose}, "
            f"initial_mode={initial_mode or 'profile default'}, peripherals={peripherals_file or 'profile defaults'})"
        )
    elif platform == "mujoco":
        # so101-mujoco-runtime-platform D3: the world owner hands the provider
        # the resolved scene NAME and spawn pose — no file path crosses the
        # boundary and nothing is recomputed. An empty scene name composes the
        # shared sim_models ``empty`` scene (ground + light): the provider
        # renders the robot into a scene either way, never alone.
        startup = simulation_startup
        scene_name = str(getattr(startup, "scene_name", "") or "")
        spawn_pose = getattr(startup, "spawn_pose", None) or (0.0, 0.0, 0.0, 0.0)
        include_arguments.update(
            {
                "mujoco_scene": scene_name,
                "spawn_x": str(spawn_pose[0]),
                "spawn_y": str(spawn_pose[1]),
                "spawn_z": str(spawn_pose[2]),
                "spawn_yaw": str(spawn_pose[3]),
            }
        )
        logger.info(
            f"runtime.provider={provider}: including {launch_path} "
            f"(profile={profile}, use_sim=true, sim_platform=mujoco, scene={scene_name!r}, spawn={spawn_pose}, "
            f"initial_mode={initial_mode or 'profile default'}, peripherals={peripherals_file or 'profile defaults'})"
        )
    else:
        logger.info(
            f"runtime.provider={provider}: including {launch_path} "
            f"(profile={profile}, use_sim={use_sim}"
            + (f", sim_platform={str(sim_platform).lower()}" if use_sim else "")
            + f", initial_mode={initial_mode or 'profile default'}, "
            f"peripherals={peripherals_file or 'profile defaults'})"
        )

    include = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_path),
        launch_arguments=include_arguments.items(),
    )
    waiter = runtime_ready_waiter(
        robot_config, readiness_timeout_s=readiness_timeout_s, description_output=description_output
    )
    return [include], waiter


def runtime_ready_waiter(
    robot_config: dict[str, Any], *, readiness_timeout_s: float = 60.0, description_output: str | None = None
) -> Node:
    """``wait_for_runtime`` for the configured provider, wherever it runs: exits 0 once it is ready."""
    provider = runtime_provider(robot_config)
    if not provider:
        raise RuntimeError("robot.runtime.provider is required to wait for a runtime")
    required = required_capabilities(robot_config)
    # runtime-peripheral-overrides review round 2 (Major 1, corrected): the
    # online semantic-mapping consumer needs the configured camera's colour,
    # aligned-depth and camera-info interfaces to EXIST. The waiter checks
    # them as declared interfaces only — no readiness, freshness or graph
    # gate — because this waiter is the shared barrier for every
    # runtime_ready action on non-binding provider paths, and a slow driver
    # or a late-publishing simulator must not block the whole startup. The
    # IDs do NOT feed required_interface_ids, so provider configurations
    # without require_model keep their non-binding startup flow.
    semantic_interface_ids = semantic_mapping_interface_ids(robot_config)
    # Event streams (audio.wake_event) are declared-only for the same reason:
    # they publish nothing until their event happens, so readiness never comes.
    declared_interface_ids = list(dict.fromkeys([*semantic_interface_ids, *event_interface_ids(robot_config)]))
    interface_ids = required_interface_ids(robot_config) if description_output else []
    instance_id = (robot_config.get("runtime") or {}).get("instance_id")
    return Node(
        package="robot_runtime",
        executable="wait_for_runtime",
        name="wait_for_runtime",
        output="screen",
        arguments=[
            "--runtime-name",
            provider,
            "--timeout",
            str(float(readiness_timeout_s)),
            *(["--required", *required] if required else []),
            *(["--description-output", description_output] if description_output else []),
            *(["--require-interfaces", *interface_ids] if interface_ids else []),
            *(["--declared-interfaces", *declared_interface_ids] if declared_interface_ids else []),
            *(
                ["--interface-requirements", json.dumps(required_interface_requirements(robot_config))]
                if interface_ids
                else []
            ),
            *(["--instance-id", instance_id] if instance_id else []),
        ],
    )


def materialize_interface_config(
    robot_config: dict[str, Any], descriptor: dict, destination: Path, *, require_local_model_artifacts: bool = True
) -> dict:
    """Validate then atomically publish one complete, bound consumer configuration.

    The serialised snapshot drops the framework-private
    ``runtime._disabled_peripherals`` (runtime-peripheral-overrides D2; the
    filtered entries are an in-memory hand-off to the fragment builder). The
    input configuration is not mutated.

    ``require_local_model_artifacts=False`` carries the launch's
    ``with_model_services:=false``: the bound snapshot then validates the
    perception services by metadata, so a host that only calls the remotely
    served endpoints does not need their weights and artifacts.
    """
    from copy import deepcopy

    from robot_config.loader import validate_robot_config_dict

    effective = bind_robot_interfaces(robot_config, descriptor, require_ready=True)
    validate_robot_config_dict(effective, require_local_model_artifacts=require_local_model_artifacts)
    snapshot = {key: value for key, value in effective.items() if key != "_config_path"}
    runtime = snapshot.get("runtime")
    if isinstance(runtime, dict):
        runtime = deepcopy(runtime)
        runtime.pop("_disabled_peripherals", None)
        snapshot["runtime"] = runtime
    encoded = yaml.safe_dump({"robot": snapshot}, sort_keys=False)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, encoding="utf-8", delete=False) as stream:
            temporary_path = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    effective["_config_path"] = str(destination)
    return effective


def generate_bound_runtime_actions(
    robot_config: dict[str, Any],
    construct_consumers,
    *,
    use_sim: bool,
    sim_platform: str = "sdk",
    simulation_startup=None,
    display: bool = False,
    readiness_timeout_s: float = 60.0,
    start_provider: bool = True,
    require_local_model_artifacts: bool = True,
) -> list:
    """Start the provider once; only construct consumers after the live description is bound.

    ``simulation_startup`` carries the world hand-off for a physics transport
    (runtime-simulation-transport D3): the world generic launch started ahead
    of this call is passed to the provider as ``gazebo_world`` plus the full
    spawn pose instead of being recomputed or re-launched.

    ``start_provider=False`` binds to the runtime another host already runs
    (as the X2's grasp pipeline on the Orin does to the robot-side stack).

    ``require_local_model_artifacts=False`` carries the launch's
    ``with_model_services:=false``: the bound snapshot validates the perception
    services by metadata, so a host that only calls the remotely served
    endpoints does not need their weights and artifacts.
    """
    directory = Path(tempfile.mkdtemp(prefix="ibrobot_interfaces_"))
    description_path = directory / "description.yaml"
    effective_path = directory / "robot.yaml"
    try:
        if start_provider:
            providers, waiter = generate_runtime_provider_actions(
                robot_config,
                use_sim=use_sim,
                sim_platform=sim_platform,
                simulation_startup=simulation_startup,
                display=display,
                readiness_timeout_s=readiness_timeout_s,
                description_output=str(description_path),
            )
        else:
            providers = []
            waiter = runtime_ready_waiter(
                robot_config, readiness_timeout_s=readiness_timeout_s, description_output=str(description_path)
            )
    except Exception:
        shutil.rmtree(directory)
        raise

    def on_ready(event, _context):
        if _context.is_shutdown:
            return []
        if event.returncode != 0:
            reason = f"Runtime interface readiness failed (returncode={event.returncode}); consumers were not started"
            logger.error(reason)
            return [EmitEvent(event=Shutdown(reason=reason))]
        try:
            from robot_runtime.interface_description import load_description

            effective = materialize_interface_config(
                robot_config,
                load_description(description_path),
                effective_path,
                require_local_model_artifacts=require_local_model_artifacts,
            )
            logger.info(f"Runtime interfaces bound; consumer snapshot: {effective_path}")
            return construct_consumers(effective)
        except Exception as exc:
            reason = f"Runtime interface binding failed; consumers were not started: {exc}"
            logger.error(reason)
            return [EmitEvent(event=Shutdown(reason=reason))]

    def on_shutdown(_event, _context):
        # Recorders persist this path in dataset/bag metadata for offline conversion.
        if not effective_path.is_file():
            shutil.rmtree(directory, ignore_errors=True)

    return [
        RegisterEventHandler(OnShutdown(on_shutdown=on_shutdown)),
        RegisterEventHandler(OnProcessExit(target_action=waiter, on_exit=on_ready)),
        *providers,
        waiter,
    ]
