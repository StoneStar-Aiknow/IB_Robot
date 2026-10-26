"""Controller readiness and startup-sequence helpers for the registry orchestration.

Migrated verbatim from the legacy ``robot.launch.py`` (openspec change
``refactor-launch-builder-registry``): these are the shared primitives the
preparation module (controller startup timeout) and the generic readiness
orchestrator (barrier selection, success-gated startup sequences) call.
They own no registry state and perform no business filtering.
"""

from __future__ import annotations

from typing import Any

from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch_ros.actions import Node

from robot_config.launch_registry.builder_support import readiness_attribute, runtime_provider_name
from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext
from robot_config.logger_utils import get_colored_logger
from robot_config.utils import parse_bool

logger = get_colored_logger("robot_config.launch_builders.readiness")


def start_actions_on_success(start_actions, success_message: str, failure_reason: str):
    """Run launch actions only when the target process exits successfully."""
    frozen_actions = tuple(start_actions)

    def _handler(event, _context):
        if event.returncode == 0:
            logger.info(success_message)
            return list(frozen_actions)

        logger.error(f"{failure_reason} (returncode={event.returncode})")
        return [EmitEvent(event=Shutdown(reason=failure_reason))]

    return _handler


def serialize_process_startup(processes, sequence_name: str):
    """Start each process only after the previous process exits successfully."""
    frozen_processes = tuple(processes)
    if not frozen_processes:
        return []

    actions = []
    for index in range(len(frozen_processes) - 1):
        current_process = frozen_processes[index]
        next_process = frozen_processes[index + 1]
        actions.append(
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=current_process,
                    on_exit=start_actions_on_success(
                        [next_process],
                        success_message=f"{sequence_name} step {index + 1} completed; starting next step.",
                        failure_reason=f"{sequence_name} step {index + 1} failed; aborting launch.",
                    ),
                )
            )
        )
    actions.append(frozen_processes[0])
    return actions


def controller_readiness_barrier(deferred_spawners, readiness_waiter):
    """Use the single strict group spawner as the owned-startup barrier."""
    if len(deferred_spawners) > 1:
        raise ValueError("Controller startup must use a single controller group spawner.")
    return deferred_spawners[0] if deferred_spawners else readiness_waiter


def resolve_controller_startup_timeout(robot_config: dict, use_sim: bool, sim_platform: str = "") -> float:
    """Resolve controller startup timeout from robot YAML.

    The mapping may define a ``gazebo`` tier for physics transports
    (runtime-simulation-transport D7: world boot + entity spawn + controllers
    need more headroom than the SDK transport); it falls back to ``sim``.
    """
    configured_timeout = robot_config.get("controller_startup_timeout")
    if configured_timeout is None:
        return 120.0 if use_sim else 30.0

    timeout_value = configured_timeout
    if isinstance(configured_timeout, dict):
        profile_key = "sim" if use_sim else "hardware"
        if use_sim and str(sim_platform).lower() == "gazebo":
            timeout_value = configured_timeout.get("gazebo")
            if timeout_value is None:
                timeout_value = configured_timeout.get("sim")
        else:
            timeout_value = configured_timeout.get(profile_key)
        if timeout_value is None:
            timeout_value = configured_timeout.get("default")

    if timeout_value is None:
        raise ValueError("robot.controller_startup_timeout must define a value for the active launch profile.")

    try:
        timeout = float(timeout_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("robot.controller_startup_timeout must be a number or a mapping of numbers.") from exc

    if timeout <= 0.0:
        raise ValueError("robot.controller_startup_timeout must be greater than zero.")

    return timeout


def create_controller_ready_waiter(robot_config: dict, controller_names, use_sim: bool) -> Node:
    """Create a probe that exits when the required controllers are active."""
    timeout = resolve_controller_startup_timeout(robot_config, use_sim)
    return Node(
        package="robot_config",
        executable="wait_for_controllers",
        name="wait_for_active_controllers",
        parameters=[{"use_sim_time": use_sim}],
        arguments=[
            *controller_names,
            "--controller-manager",
            "controller_manager",
            "--timeout",
            str(timeout),
        ],
        output="screen",
    )


def load_launch_robot_config(
    robot_config_name, config_path_override=None, nav_stage="", *, defer_interface_binding=False
):
    """Load robot configuration from YAML file (migrated from robot.launch.py)."""
    from pathlib import Path

    from ament_index_python.packages import get_package_share_directory

    from robot_config.loader import load_robot_config_dict

    try:
        robot_config_share = get_package_share_directory("robot_config")
    except Exception:
        # Mirror the legacy fallback: source-tree launches resolve the config
        # directory relative to this package's parent.
        robot_config_share = str(Path(__file__).resolve().parent.parent.parent)

    if config_path_override:
        config_path = Path(config_path_override)
    else:
        config_path = Path(robot_config_share) / "config" / "robots" / f"{robot_config_name}.yaml"

    logger.info(f"Loading config from: {config_path}")
    logger.info(f"Config exists: {config_path.exists()}")

    robot_config = load_robot_config_dict(
        config_path,
        nav_stage=nav_stage,
        materialize_benchmark_transport=False,
        defer_interface_binding=defer_interface_binding,
    )
    logger.info(f"Loaded robot: {robot_config.get('name', 'UNKNOWN')}")
    logger.info(f"Peripherals: {len(robot_config.get('peripherals', []))}")

    return robot_config


def resolve_config_path(robot_config_name: str, config_path_override: str) -> str:
    """Resolve the ``_config_path`` value stored for downstream modules."""
    from pathlib import Path

    from ament_index_python.packages import get_package_share_directory

    if config_path_override:
        return config_path_override
    try:
        robot_config_share = get_package_share_directory("robot_config")
    except Exception:
        robot_config_share = str(Path(__file__).resolve().parent.parent.parent)
    return str(Path(robot_config_share) / "config" / "robots" / f"{robot_config_name}.yaml")


__all__ = [
    "controller_readiness_barrier",
    "create_controller_ready_waiter",
    "load_launch_robot_config",
    "resolve_config_path",
    "resolve_controller_startup_timeout",
    "serialize_process_startup",
    "start_actions_on_success",
]


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``):
# the BuilderSpec constant and build function moved here from the retired
# ``adapters.py``; the body is a verbatim migration (only the shared-helper
# call sites now resolve through ``builder_support``).
# ---------------------------------------------------------------------------


def build_controller_startup(context: LaunchBuildContext) -> BuildResult:
    """Assemble the controller startup chain and the readiness barrier.

    Consumes the control/simulation typed outputs only. Installs the
    resolved barrier action into the shared context
    (``controller_readiness_barrier``) so the generic orchestrator can
    defer the runtime-ready set behind it. With no barrier the startup
    sequence/waiter actions join the immediate stream in legacy order.
    """
    control = context.control_startup
    simulation = context.simulation_startup
    robot_config = context.robot_config
    use_sim = context.use_sim
    gz_create_entity = simulation.entity_action if simulation is not None else None

    deferred_spawners = list(control.deferred_spawners) if control is not None else []
    controller_names = list(control.controller_names) if control is not None else []
    runtime_ready_waiter = control.runtime_ready_waiter if control is not None else None

    actions: list[Any] = []
    provider_name = runtime_provider_name(context)
    if provider_name:
        # RuntimeStatus reconciliation replaces controller readiness gating.
        controller_ready_waiter = runtime_ready_waiter
    elif parse_bool(context.auto_start_controllers, default=True) and controller_names:
        controller_ready_waiter = readiness_attribute("create_controller_ready_waiter")(
            robot_config, controller_names, use_sim
        )
    else:
        controller_ready_waiter = None
    barrier = readiness_attribute("controller_readiness_barrier")(deferred_spawners, controller_ready_waiter)

    if deferred_spawners:
        startup_sequence = readiness_attribute("serialize_process_startup")(
            list(deferred_spawners), "Controller startup"
        )
        if use_sim and gz_create_entity is not None:
            logger.info("Scheduling controller startup after ros_gz_sim create exits")
            actions.append(
                RegisterEventHandler(
                    event_handler=OnProcessExit(
                        target_action=gz_create_entity,
                        on_exit=readiness_attribute("start_actions_on_success")(
                            startup_sequence,
                            success_message="Robot entity created; starting controller startup sequence.",
                            failure_reason="Robot entity creation failed; aborting launch.",
                        ),
                    )
                )
            )
        elif use_sim:
            actions.extend(startup_sequence)
        else:
            logger.info("Starting controller startup sequence; spawners wait for controller_manager readiness")
            actions.extend(startup_sequence)
    elif controller_ready_waiter is not None:
        actions.append(controller_ready_waiter)

    context.controller_readiness_barrier = barrier
    return BuildResult(actions=actions)


CONTROLLER_STARTUP_BUILDER = BuilderSpec(
    id="controller_startup",
    factory=f"{__name__}:build_controller_startup",
    after=("control", "simulation"),
)
