"""Main robot launch file for robot_config (registry orchestration).

This entry owns three responsibilities only (openspec change
``refactor-launch-builder-registry``):

1. declare the launch arguments contributed by the builtin registry
   (core table + business contributions; the SSOT lives in
   ``launch_builders/launch_options.py``);
2. run the preparation pipeline (``prepare_launch``): config load,
   CLI overrides, safety checks, live-binding target write-back and the
   frozen participation plan;
3. execute the frozen plan through the generic orchestrator
   (``build_prepared_launch``): readiness barrier gating, tracing
   scopes and the live-binding delegation.

All business node construction lives in ``launch_builders`` adapters;
this file deliberately keeps no ``generate_*`` business imports. The
readiness helpers were migrated to ``launch_builders/readiness.py`` and
are re-exported here as thin aliases for the existing direct-call tests.

**CRITICAL**: This workspace uses ROS_DOMAIN_ID=<ID> to avoid conflicts
with other ROS 2 systems:
```bash
export ROS_DOMAIN_ID=<ID>
```

Usage:
    # Basic simulation
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

    # Contract mock simulation
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true sim_platform:=mock control_mode:=model_inference

    # Real hardware
    ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=false

Cleanup: if controllers are already loaded, run ``./scripts/cleanup_ros.sh``.
"""

from launch import LaunchDescription
from launch.actions import OpaqueFunction

from robot_config.dds_environment import check_launch_dds

# Thin aliases over the shared readiness helpers: the implementations have a
# single owner in robot_config.launch_builders.readiness (resolved at call
# time by the orchestrator); these module-level names exist only because
# existing tests call them directly on this module. build_launch_registry is
# the production registry entry: builtin table + ibrobot.launch_builders entry
# point extensions (see builtin_registry and the README's extension section).
from robot_config.launch_builders import readiness as _readiness
from robot_config.launch_builders.builtin_registry import build_launch_registry
from robot_config.launch_builders.launch_options import declare_launch_arguments
from robot_config.launch_builders.orchestrator import build_prepared_launch
from robot_config.launch_builders.preparation import compose_registry_arguments, prepare_launch
from robot_config.launch_builders.tracing import DEFAULT_TRACE_SESSION_NAME  # noqa: F401 (baseline tests read it here)


def _controller_readiness_barrier(deferred_spawners, readiness_waiter):
    return _readiness.controller_readiness_barrier(deferred_spawners, readiness_waiter)


def _create_controller_ready_waiter(robot_config, controller_names, use_sim):
    return _readiness.create_controller_ready_waiter(robot_config, controller_names, use_sim)


def _resolve_controller_startup_timeout(robot_config, use_sim):
    return _readiness.resolve_controller_startup_timeout(robot_config, use_sim)


def _serialize_process_startup(processes, sequence_name):
    return _readiness.serialize_process_startup(processes, sequence_name)


def _start_actions_on_success(start_actions, success_message, failure_reason):
    return _readiness.start_actions_on_success(start_actions, success_message, failure_reason)


def load_robot_config(robot_config_name, config_path_override=None, nav_stage="", *, defer_interface_binding=False):
    """Load robot configuration from YAML (delegates to the readiness module)."""
    from robot_config.launch_builders.readiness import load_launch_robot_config

    return load_launch_robot_config(
        robot_config_name,
        config_path_override,
        nav_stage,
        defer_interface_binding=defer_interface_binding,
    )


def _apply_inference_cli_overrides(context, robot_config: dict, control_mode: str) -> None:
    """Apply an explicit execution-mode override to one named inference pipeline.

    Compatibility wrapper over the preparation pipeline step; reads the two
    launch configurations exactly like the legacy entry did.
    """
    from robot_config.launch_builders.preparation import _apply_inference_overrides

    _apply_inference_overrides(
        {
            "inference_execution_mode": context.launch_configurations.get("inference_execution_mode", ""),
            "inference_pipeline": context.launch_configurations.get("inference_pipeline", ""),
        },
        robot_config,
        control_mode,
    )


def launch_setup(context, *args, loaded_config=None, extra_consumers=None, **kwargs):
    """Prepare the launch through the registry pipeline, then build the plan.

    Args:
        context: Launch context
        loaded_config: Internal preloaded configuration; still normalized and live-bound here
        extra_consumers: Internal callback constructing additional consumers after live logical binding
    """
    # Local DDS admission check (same contract as teleop_operator.launch.py):
    # runs before any config load or provider startup; the live-binding
    # branch re-checks inside the orchestrator's construct_consumers.
    check_launch_dds(context)
    registry = build_launch_registry()
    prepared = prepare_launch(context, registry, loaded_config=loaded_config)
    return build_prepared_launch(prepared, extra_consumers=extra_consumers)


def generate_launch_description():
    """Generate launch description for robot system."""
    registry = build_launch_registry()
    argument_specs = compose_registry_arguments(registry)
    return LaunchDescription(
        [
            *declare_launch_arguments(argument_specs),
            OpaqueFunction(function=launch_setup),
        ]
    )
