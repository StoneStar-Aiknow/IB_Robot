"""Explicit builtin launch-builder registration table (ordered references).

The registration framework lives in ``robot_config.launch_registry``
(openspec change ``split-launch-registry-package``).

Implements design decision 1 of openspec change
``decentralize-launch-builder-registration``: every builtin entry's
``BuilderSpec`` constant and ``build_<id>`` function live in its domain
module; this table only keeps the ordered ``"module:ATTR"`` references.
``build_builtin_registry()`` resolves each reference and registers the spec
on a fresh per-launch registry instance (no process-level global table).

The table order mirrors the legacy ``_construct_robot_nodes`` section order
and is the stable tie-breaker for topological sorting:

control → auxiliary actuators → hand sources → simulation →
controller-startup → perception → benchmark → audio → execution →
motion planning → embodied.

Openspec change ``migrate-business-launch-builders`` moved every business
entry whose owning package exists out of this table:
``perception_models`` (inference_service), ``semantic_mapping``
(semantic_mapping), ``task_executor`` (task_dispatch), ``recording`` and
``rerun_visualizer`` (dataset_tools), ``motion_imitation_stream`` and
``mirror_session`` (manipulation_execution), ``teleop`` (robot_teleop),
``voice_asr`` and ``speech_direction`` (voice_asr_service), ``voice_tts``
(voice_tts_service), ``interaction_demo`` (robot_interaction_demo),
``navigation`` with its nav2 / localization / navigation_command /
voice_nav sub-builders (robot_navigation). They register through the
entry point group ``ibrobot.launch_builders`` — appended after every
builtin entry, sorted by entry point name, because extension entries
always follow the builtin table.

Adding a builtin entry = defining the ``build_<id>`` function and
``<ID>_BUILDER`` constant in its domain module and appending one reference
line here; no other module changes.

Import budget (design decision 2): building the registry imports every
builtin domain module, so those modules keep their cross-package run
dependencies function-local (see ``launch_registry/builder_support.py`` for the
shared enable-predicate helpers); the import-budget test pins the allowed
non-stdlib top-level package set to the registry core + ``launch`` +
``launch_ros`` + ``ament_index_python``.

Cross-package registration (design decision 3): ``build_launch_registry()``
additionally discovers ``BuilderSpec`` entries declared by installed packages
through the entry point group ``ibrobot.launch_builders`` (builtin entries
first, extensions appended sorted by entry point name) — the production entry
``robot.launch.py`` uses it, while ``build_builtin_registry()`` stays
builtin-only for the existing test seams.
"""

from __future__ import annotations

from importlib import import_module

from robot_config.launch_registry.registry import LaunchBuilderRegistry

# Ordered builtin builder references ("module:<ID>_BUILDER"). The order is
# the registration order and the stable tie-breaker for topological sorting;
# it must stay identical to the previous inline registration order.
BUILTIN_BUILDERS: tuple[str, ...] = (
    # ---- Infrastructure (typed-output owners) ----
    "robot_config.launch_builders.control:CONTROL_BUILDER",
    "robot_config.launch_builders.control:AUXILIARY_ACTUATORS_BUILDER",
    # hand_sources moved to robot_teleop and registers through the
    # ibrobot.launch_builders entry point group.
    "robot_config.launch_builders.simulation:SIMULATION_BUILDER",
    # scene_task (so101-mujoco-runtime-platform D8) reads the simulation
    # entry's resolved world and runs behind the runtime readiness barrier.
    "robot_config.launch_builders.scene_task:SCENE_TASK_BUILDER",
    "robot_config.launch_builders.readiness:CONTROLLER_STARTUP_BUILDER",
    # ---- Business ----
    "robot_config.launch_builders.perception:PERCEPTION_BUILDER",
    "robot_config.launch_builders.benchmark:BENCHMARK_BUILDER",
    # perception_models moved to inference_service, semantic_mapping moved
    # to the semantic_mapping package; both register through the
    # ibrobot.launch_builders entry point group.
    # teleop moved to robot_teleop, voice_asr and speech_direction moved to
    # voice_asr_service; all register through the ibrobot.launch_builders
    # entry point group.
    "robot_config.launch_builders.audio_io:AUDIO_IO_BUILDER",
    # voice_tts moved to voice_tts_service and registers through the
    # ibrobot.launch_builders entry point group.
    # navigation (with nav2/localization/navigation_command/voice_nav)
    # moved to robot_navigation and registers through the
    # ibrobot.launch_builders entry point group.
    "robot_config.launch_builders.execution:EXECUTION_BUILDER",
    "robot_config.launch_builders.task_execution:MOTION_PLANNING_BUILDER",
    "robot_config.launch_builders.embodied:EMBODIED_BUILDER",
    # task_executor moved to task_dispatch and registers through the
    # ibrobot.launch_builders entry point group.
    # recording + rerun_visualizer moved to dataset_tools and register
    # through the ibrobot.launch_builders entry point group.
    # interaction_demo moved to robot_interaction_demo and registers
    # through the ibrobot.launch_builders entry point group.
    # motion_imitation_stream moved to manipulation_execution and registers
    # through the ibrobot.launch_builders entry point group.
    # mirror_session moved to manipulation_execution (design decision 4);
    # it registers through the ibrobot.launch_builders entry point group.
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


# --- Cross-package registration via entry points -----------------------------
#
# Design decision 3 of openspec change ``decentralize-launch-builder-registration``:
# installed packages contribute BuilderSpec entries through the entry point
# group ``ibrobot.launch_builders``. The entry point value is a ``module:ATTR``
# reference resolved to a single BuilderSpec or a sequence of them; factories
# and hook factories stay ``module:attribute`` STRINGS resolved lazily on every
# call (the existing monkeypatch seam), never callables.

LAUNCH_BUILDER_ENTRY_POINT_GROUP = "ibrobot.launch_builders"


def _extension_entries(entry_points):
    """Resolve declared extension entry points into validated (name, value, specs).

    ``entry_points`` is the test injection seam; ``None`` means "discover for
    real" through :func:`importlib.metadata.entry_points` with the selection
    API. Accepts any iterable of :class:`importlib.metadata.EntryPoint`-like
    objects exposing ``name`` and ``value``.

    Validation happens here, before any build factory or hook is resolved or
    executed: duplicate entry point names, unresolvable references, non-
    BuilderSpec results and typed-output (``provides``) declarations are all
    rejected as :class:`RegistryError` naming the offending entry point.
    Entries are ordered by (name, value) for deterministic composition.
    """
    from robot_config.launch_registry.registry import BuilderSpec, RegistryError

    if entry_points is None:
        from importlib.metadata import entry_points as discover_entry_points

        entry_points = discover_entry_points(group=LAUNCH_BUILDER_ENTRY_POINT_GROUP)
    entries = list(entry_points)

    # Entry point names must be unique across the whole group: two installed
    # distributions declaring the same name (e.g. a stale egg-info next to a
    # fresh one) is an environment problem the operator must see, not a
    # silently-resolved tie.
    seen_names: dict[str, str] = {}
    for entry in entries:
        name = getattr(entry, "name", None)
        value = getattr(entry, "value", None)
        if not isinstance(name, str) or not name or not isinstance(value, str) or not value:
            raise RegistryError(f"invalid launch-builder entry point {entry!r}; expected name and value strings")
        if name in seen_names:
            raise RegistryError(
                f"duplicate launch-builder entry point name {name!r}: declared twice "
                f"(values {seen_names[name]!r} and {value!r}); entry point names must be unique"
            )
        seen_names[name] = value

    resolved = []
    for entry in sorted(entries, key=lambda entry: (str(entry.name), str(entry.value))):
        name, value = str(entry.name), str(entry.value)
        module_name, separator, attribute = value.partition(":")
        # Strip the optional setuptools extras suffix ("module:attr [e1,e2]").
        attribute = attribute.partition("[")[0].strip()
        if not separator or not module_name or not attribute:
            raise RegistryError(
                f"launch-builder entry point {name!r} has invalid value {value!r}; expected 'module:attribute'"
            )
        try:
            module = import_module(module_name)
        except Exception as exc:
            raise RegistryError(
                f"launch-builder entry point {name!r} ({value!r}) failed to import module {module_name!r}: {exc}"
            ) from exc
        try:
            resolved_attribute = getattr(module, attribute)
        except AttributeError as exc:
            raise RegistryError(
                f"launch-builder entry point {name!r} ({value!r}) resolved to a missing attribute "
                f"{module_name!r}:{attribute}"
            ) from exc
        specs = list(resolved_attribute) if isinstance(resolved_attribute, list | tuple) else [resolved_attribute]
        for spec in specs:
            if not isinstance(spec, BuilderSpec):
                raise RegistryError(
                    f"launch-builder entry point {name!r} ({value!r}) must resolve to a BuilderSpec or a "
                    f"sequence of BuilderSpec, got {type(spec).__name__}"
                )
            if spec.provides:
                raise RegistryError(
                    f"launch-builder entry point {name!r} ({value!r}) declares builder {spec.id!r} with "
                    f"provides={tuple(spec.provides)}; typed infrastructure outputs are reserved for the "
                    "builtin entries and cannot be declared by extensions"
                )
        resolved.append((name, value, specs))
    return resolved


def build_launch_registry(*, entry_points=None) -> LaunchBuilderRegistry:
    """Construct the full launch registry: builtin entries + discovered extensions.

    The builtin table registers first in its fixed order (the stable
    tie-breaker); extension entries are appended afterwards sorted by entry
    point name (then value), so the composition is deterministic across
    runs. Extensions use exactly the same ``add``-time validation as builtin
    entries (duplicate ids, argument-name collisions, hook config-key
    collisions, unknown dependencies); violations are re-raised as
    :class:`RegistryError` naming the offending entry point. Installing an
    extension package does NOT enable its entry: participation is still
    decided by each spec's ``enabled`` predicate / ``mock_skip_key``.

    ``entry_points`` is the test injection seam (an iterable of entry
    points); ``None`` discovers the ``ibrobot.launch_builders`` group through
    :func:`importlib.metadata.entry_points`.
    """
    from robot_config.launch_registry.registry import RegistryError

    registry = build_builtin_registry()
    for name, value, specs in _extension_entries(entry_points):
        for spec in specs:
            try:
                registry.add(spec)
            except RegistryError as exc:
                raise RegistryError(f"launch-builder entry point {name!r} ({value!r}) rejected: {exc}") from exc
    return registry
