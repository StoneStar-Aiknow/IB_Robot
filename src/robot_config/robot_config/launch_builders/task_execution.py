"""Motion-planning launch validation.

This module owns the ``motion_planning`` entry: generic launch ships no
planner, so the entry only validates the ``with_moveit`` resolution and
the runtime provider requirement. The ``task_executor`` entry (the
``task_dispatch/task_executor_node`` wiring) moved to its owning package
``task_dispatch`` (openspec change ``migrate-business-launch-builders``)
and registers through the entry point group ``ibrobot.launch_builders``.
"""

from robot_config.launch_registry.builder_support import resolve_with_moveit, runtime_provider_name
from robot_config.launch_registry.launch_options import ArgumentSpec
from robot_config.launch_registry.registry import BuilderSpec, BuildResult, LaunchBuildContext
from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("robot_config.task_execution")


# ---------------------------------------------------------------------------
# Registry entry (openspec change ``decentralize-launch-builder-registration``):
# the BuilderSpec constant and build function moved here from the retired
# ``adapters.py``; the bodies are verbatim migrations (only the shared with_moveit
# resolution now imports from ``builder_support`` instead of the former
# ``builtin_registry`` helper).
# ---------------------------------------------------------------------------

# Business launch arguments owned by the motion_planning entry (migrated
# verbatim from the former builtin_registry table; task_executor only reads).
_MOTION_PLANNING_ARGUMENTS = (
    ArgumentSpec(
        name="with_moveit",
        declared_default="",
        absent_fallback="",
        description="Enable MoveIt motion planning. If empty, auto-detects from control mode config",
    ),
)


def build_motion_planning(context: LaunchBuildContext) -> BuildResult:
    """with_moveit validation: generic launch ships no planner.

    Always participates: the no-provider hard error is build-time policy
    (failure tolerance), not a participation condition. The with_moveit
    resolution is shared with the task_executor enable predicate.
    """
    with_moveit = resolve_with_moveit(context)
    logger.info(f"with_moveit={with_moveit}")

    provider_name = runtime_provider_name(context)
    if with_moveit and not provider_name:
        raise RuntimeError(
            "motion planning was requested (with_moveit / a moveit control mode) but robot.runtime.provider "
            "is not set: motion services come from the robot runtime, generic launch ships no planner"
        )
    if with_moveit:
        logger.info(f"Motion services provided by runtime.provider={provider_name}")
    else:
        logger.info("Motion planning not requested")
    return BuildResult()


MOTION_PLANNING_BUILDER = BuilderSpec(
    id="motion_planning",
    factory=f"{__name__}:build_motion_planning",
    arguments=_MOTION_PLANNING_ARGUMENTS,
)
