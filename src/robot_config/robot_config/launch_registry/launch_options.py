"""Launch argument declaration model with dual entry points.

The registration framework lives in ``robot_config.launch_registry``
(openspec change ``split-launch-registry-package``).

Implements the argument-spec half of openspec change
``refactor-launch-builder-registry`` (design decision 4):

- Every :class:`ArgumentSpec` carries both the ``declared_default`` (used
  by ``generate_launch_description`` via ``DeclareLaunchArgument``) and
  the ``absent_fallback`` (used when ``launch_setup`` is called directly
  — e.g. by the embodied bringup — and the launch configuration key is
  missing). The two defaults are deliberately NOT unified.
- Reading preserves the distinction between a missing key, an empty
  string and a non-empty value; whether an override applies keeps each
  parameter's existing empty-string rule.
- Core parameters are defined once here; business parameters will be
  contributed by builder registration metadata (see
  ``compose_argument_specs``) and are rejected on name collision with
  core arguments or with each other.

This module has no hardware, model or ROS-graph dependencies: generating
the declared argument list must not import business builders.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from launch.actions import DeclareLaunchArgument

from robot_config.launch_builders.tracing import DEFAULT_TRACE_SESSION_NAME


@dataclass(frozen=True)
class ArgumentSpec:
    """One launch argument with dual entry-point defaults.

    ``declared_default`` is what ``generate_launch_description`` declares;
    ``absent_fallback`` is what direct ``launch_setup`` callers observe
    when the launch configuration key is absent. Reading an existing key
    (empty string or not) always returns the raw value.
    """

    name: str
    declared_default: str
    absent_fallback: str
    description: str


# The core table below is migrated verbatim from robot.launch.py:
# declared_default comes from each DeclareLaunchArgument default_value,
# absent_fallback from the corresponding context.launch_configurations.get(...)
# call inside launch_setup. Do not unify the intentionally different
# defaults (robot_config, voice_asr_auto_start, moveit_display).
#
# Business parameters (hand profile, voice ASR, navigation, moveit,
# recording, rerun visualizer, embodied) live on their owning BuilderSpec
# declarations in builtin_registry.py (openspec change
# ``migrate-builtin-builder-metadata``); the core table keeps only the
# parameters consumed by the preparation core or infrastructure entries.
CORE_ARGUMENT_SPECS: tuple[ArgumentSpec, ...] = (
    ArgumentSpec(
        name="robot_config",
        declared_default="so101_single_arm",
        absent_fallback="test_cam",
        description="Robot configuration name (without .yaml extension)",
    ),
    ArgumentSpec(
        name="config_path",
        declared_default="",
        absent_fallback="",
        description="Optional: Full path to robot config file (overrides robot_config)",
    ),
    ArgumentSpec(
        name="use_sim",
        declared_default="",
        absent_fallback="",
        description=(
            "Override virtual embodiment (true/false). Empty infers true for "
            "runtime.target benchmark/simulation and false for hardware."
        ),
    ),
    ArgumentSpec(
        name="sim_platform",
        declared_default="",
        absent_fallback="",
        description="Override simulation.platform from YAML (gazebo, mujoco, or mock). Empty uses YAML.",
    ),
    ArgumentSpec(
        name="auto_start_controllers",
        declared_default="true",
        absent_fallback="true",
        description="Automatically spawn controllers (set to false for debugging)",
    ),
    ArgumentSpec(
        name="control_mode",
        declared_default="",
        absent_fallback="",
        description=(
            "Override control mode from YAML (teleop, model_inference, or moveit_planning). "
            "If empty, uses default_control_mode from config file"
        ),
    ),
    ArgumentSpec(
        name="nav_stage",
        declared_default="",
        absent_fallback="",
        description=(
            "Select a workflow stage declared by the robot config. "
            "Navigation profiles use mapping/navigation; combined mobile-manipulator profiles may also "
            "declare grasp/hybrid. Empty uses default_nav_stage."
        ),
    ),
    ArgumentSpec(
        name="with_inference",
        declared_default="",
        absent_fallback="",
        description=(
            "Enable full execution pipeline (inference + dispatcher). If empty, auto-detects from control mode config"
        ),
    ),
    ArgumentSpec(
        name="inference_pipeline",
        declared_default="",
        absent_fallback="",
        description="Pipeline ID targeted by inference launch overrides.",
    ),
    ArgumentSpec(
        name="inference_execution_mode",
        declared_default="",
        absent_fallback="",
        description="Override the targeted pipeline mode: monolithic or distributed. Empty uses YAML.",
    ),
    ArgumentSpec(
        name="moveit_display",
        declared_default="true",
        absent_fallback="false",
        description="Launch RViz for MoveIt visualization (only used if MoveIt is enabled)",
    ),
    ArgumentSpec(
        name="runtime_target",
        declared_default="",
        absent_fallback="",
        description=(
            "Override the runtime target (hardware, simulation or benchmark). "
            "Empty uses the SSOT runtime.target, then the historical use_sim "
            "fallback, then the default hardware. benchmark + use_sim:=true "
            "stays benchmark and skips ros2_control, simulation backend and "
            "physical perception."
        ),
    ),
    ArgumentSpec(
        name="enable_tracing",
        declared_default="false",
        absent_fallback="false",
        description="Enable ros2_tracing / LTTng for latency analysis (requires ros2-tracing package)",
    ),
    ArgumentSpec(
        name="trace_session_name",
        declared_default=DEFAULT_TRACE_SESSION_NAME,
        absent_fallback=DEFAULT_TRACE_SESSION_NAME,
        description=(
            "Base LTTng trace session name (only used when enable_tracing:=true). "
            "Custom names are never overwritten; the default name auto-suffixes on collision."
        ),
    ),
)


class ArgumentSpecError(ValueError):
    """Raised when composed argument specs collide on names."""


def compose_argument_specs(
    business_specs: Iterable[ArgumentSpec] = (),
    core_specs: Sequence[ArgumentSpec] = CORE_ARGUMENT_SPECS,
) -> tuple[ArgumentSpec, ...]:
    """Merge core and business argument specs, rejecting name collisions.

    Business parameters are contributed by builder registration metadata;
    a business spec may not reuse a core name nor another business name.
    """
    specs: list[ArgumentSpec] = list(core_specs)
    seen = {spec.name for spec in specs}
    for spec in business_specs:
        if not isinstance(spec, ArgumentSpec):
            raise ArgumentSpecError("business argument specs must be ArgumentSpec instances")
        if spec.name in seen:
            raise ArgumentSpecError(f"duplicate launch argument name {spec.name!r}")
        seen.add(spec.name)
        specs.append(spec)
    return tuple(specs)


def read_launch_argument(
    launch_configurations: Mapping[str, Any],
    spec: ArgumentSpec,
) -> str:
    """Read one argument preserving the missing/empty/non-empty distinction.

    A missing key yields ``absent_fallback``; a present key yields its raw
    value (empty string or not) so the existing per-parameter empty-string
    override rules keep working unchanged.
    """
    if spec.name in launch_configurations:
        return launch_configurations[spec.name]
    return spec.absent_fallback


def read_launch_arguments(
    launch_configurations: Mapping[str, Any],
    specs: Iterable[ArgumentSpec] = (),
) -> dict[str, str]:
    """Read every spec (defaulting to the core table) into a plain dict."""
    table = tuple(specs) if specs else CORE_ARGUMENT_SPECS
    return {spec.name: read_launch_argument(launch_configurations, spec) for spec in table}


def declare_launch_arguments(specs: Iterable[ArgumentSpec] = CORE_ARGUMENT_SPECS) -> list:
    """Declare every spec with its ``declared_default`` (no business imports)."""
    return [
        DeclareLaunchArgument(
            name=spec.name,
            default_value=spec.declared_default,
            description=spec.description,
        )
        for spec in specs
    ]
