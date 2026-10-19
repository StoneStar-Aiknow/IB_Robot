"""Explicit builtin launch-builder registration table (ordered references).

Implements design decision 1 of openspec change
``decentralize-launch-builder-registration``: every builtin entry's
``BuilderSpec`` constant and ``build_<id>`` function live in its domain
module; this table only keeps the ordered ``"module:ATTR"`` references.
``build_builtin_registry()`` resolves each reference and registers the spec
on a fresh per-launch registry instance (no process-level global table).

The table order mirrors the legacy ``_construct_robot_nodes`` section order
and is the stable tie-breaker for topological sorting:

control → auxiliary actuators → hand sources → simulation →
controller-startup → perception → benchmark → perception models →
semantic mapping → teleop → audio → ASR → TTS → speech direction →
navigation → execution → motion planning → embodied → task executor →
recording → rerun visualizer → interaction demo → motion imitation
stream → mirror session.

Adding a builtin entry = defining the ``build_<id>`` function and
``<ID>_BUILDER`` constant in its domain module and appending one reference
line here; no other module changes.

Import budget (design decision 2): building the registry imports every
builtin domain module, so those modules keep their cross-package run
dependencies function-local (see ``builder_support.py`` for the shared
enable-predicate helpers); the import-budget test pins the allowed
non-stdlib top-level package set to the registry core + ``launch`` +
``launch_ros`` + ``ament_index_python``.
"""

from __future__ import annotations

from importlib import import_module

from robot_config.launch_builders.registry import LaunchBuilderRegistry

# Ordered builtin builder references ("module:<ID>_BUILDER"). The order is
# the registration order and the stable tie-breaker for topological sorting;
# it must stay identical to the previous inline registration order.
BUILTIN_BUILDERS: tuple[str, ...] = (
    # ---- Infrastructure (typed-output owners) ----
    "robot_config.launch_builders.control:CONTROL_BUILDER",
    "robot_config.launch_builders.control:AUXILIARY_ACTUATORS_BUILDER",
    "robot_config.launch_builders.hand_sources:HAND_SOURCES_BUILDER",
    "robot_config.launch_builders.simulation:SIMULATION_BUILDER",
    "robot_config.launch_builders.readiness:CONTROLLER_STARTUP_BUILDER",
    # ---- Business ----
    "robot_config.launch_builders.perception:PERCEPTION_BUILDER",
    "robot_config.launch_builders.benchmark:BENCHMARK_BUILDER",
    "robot_config.launch_builders.perception_models:PERCEPTION_MODELS_BUILDER",
    "robot_config.launch_builders.semantic_mapping:SEMANTIC_MAPPING_BUILDER",
    "robot_config.launch_builders.teleop:TELEOP_BUILDER",
    "robot_config.launch_builders.audio_io:AUDIO_IO_BUILDER",
    "robot_config.launch_builders.voice_asr:VOICE_ASR_BUILDER",
    "robot_config.launch_builders.voice_tts:VOICE_TTS_BUILDER",
    "robot_config.launch_builders.speech_direction:SPEECH_DIRECTION_BUILDER",
    "robot_config.launch_builders.navigation:NAVIGATION_BUILDER",
    "robot_config.launch_builders.execution:EXECUTION_BUILDER",
    "robot_config.launch_builders.task_execution:MOTION_PLANNING_BUILDER",
    "robot_config.launch_builders.embodied:EMBODIED_BUILDER",
    "robot_config.launch_builders.task_execution:TASK_EXECUTOR_BUILDER",
    "robot_config.launch_builders.recording:RECORDING_BUILDER",
    "robot_config.launch_builders.recording:RERUN_VISUALIZER_BUILDER",
    "robot_config.launch_builders.interaction_demo:INTERACTION_DEMO_BUILDER",
    "robot_config.launch_builders.motion_imitation_stream:MOTION_IMITATION_STREAM_BUILDER",
    "robot_config.launch_builders.mirror_session:MIRROR_SESSION_BUILDER",
)


def build_builtin_registry() -> LaunchBuilderRegistry:
    """Construct the builtin registry for one launch (no shared state)."""
    registry = LaunchBuilderRegistry()
    for reference in BUILTIN_BUILDERS:
        module_name, separator, attribute = reference.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError(f"invalid builtin builder reference {reference!r}; expected 'module:attribute'")
        registry.add(getattr(import_module(module_name), attribute))
    return registry
