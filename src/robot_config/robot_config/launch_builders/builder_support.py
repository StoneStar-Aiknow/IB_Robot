"""Shared launch-builder helpers used by enable predicates and build functions.

Moved here from the retired ``adapters.py`` / ``builtin_registry.py`` (openspec
change ``decentralize-launch-builder-registration``, design decision 1): the
runtime-provider / display / with_navigation / with_moveit / rerun resolvers
and the readiness lazy-attribute lookup are shared by ``BuilderSpec`` enable
predicates and the per-domain ``build_<id>`` functions, so they live in one
import-light module that depends only on ``registry`` / ``utils`` and the
standard library. Building the registry therefore never pulls business run
dependencies through this module.
"""

from __future__ import annotations

from robot_config.launch_builders.registry import LaunchBuildContext
from robot_config.utils import parse_bool


def readiness_attribute(name: str):
    """Resolve readiness helpers at call time to preserve patch seams."""
    from robot_config.launch_builders import readiness

    return getattr(readiness, name)


def runtime_provider_name(context: LaunchBuildContext) -> str:
    """Read the configured runtime provider name from the prepared config.

    Deliberately reads the dict field directly (same semantics as
    ``runtime.runtime_provider``) instead of importing the runtime
    module: plan enable predicates must not pull business dependencies,
    and build functions keep the source-module monkeypatch seam on their
    own generate_* calls.
    """
    return str((context.robot_config.get("runtime") or {}).get("provider", "") or "").strip()


def resolve_display(context: LaunchBuildContext) -> bool:
    # moveit_display: declared path defaults true; embodied calls pass
    # SetLaunchConfiguration("moveit_display", "false") explicitly; bare
    # launch_setup callers fall back to "false".
    return parse_bool(context.arguments.get("moveit_display", "false"), default=False)


def resolve_with_navigation(context: LaunchBuildContext) -> bool:
    """Overall navigation participation (``with_navigation`` CLI switch).

    An empty CLI value falls back to the prepared ``navigation.enabled``
    YAML flag. Live binding and benchmark materialization never rewrite
    that flag (``bind_robot_interfaces`` only writes runtime /
    robot_model / joints / moveit / contract fields; the benchmark
    materializer only writes the observation transport), so the
    navigation ``enabled`` predicate reads exactly what the former
    adapter-time check read.
    """
    with_navigation_str = context.arguments.get("with_navigation", "")
    if with_navigation_str != "":
        return parse_bool(with_navigation_str, default=False)
    navigation_config = context.robot_config.get("navigation", {})
    return bool(navigation_config.get("enabled", False))


def resolve_with_moveit(context: LaunchBuildContext) -> bool:
    """Overall MoveIt participation (``with_moveit`` CLI switch).

    An empty CLI value falls back to the prepared control mode
    containing ``moveit``. Shared by the task_executor ``enabled``
    predicate and the motion_planning build function (which keeps the
    no-provider hard error as build-time policy).
    """
    with_moveit_str = context.arguments.get("with_moveit", "")
    if with_moveit_str != "":
        return parse_bool(with_moveit_str, default=False)
    return "moveit" in context.control_mode.lower()


def rerun_visualizer_requested(context: LaunchBuildContext) -> bool:
    """``record_visualizer`` participation: any value but ``none``.

    The value is lower()ed first (legacy rule); the rerun/unknown-value
    dispatch and its failure tolerance stay in the build function.
    """
    return context.arguments.get("record_visualizer", "none").lower() != "none"
