"""Explicit ordered launch-builder registry contracts.

The registration framework lives in ``robot_config.launch_registry``
(openspec change ``split-launch-registry-package``).

Implements the registry core for openspec change
``refactor-launch-builder-registry``:

- :class:`BuilderSpec` — pure metadata for one launch builder. The factory
  is referenced as ``"module:attribute"`` and resolved lazily on every
  call, so monkeypatching the source module keeps working as a test seam.
- :class:`LaunchBuildContext` — the prepared state shared by the builders
  of one launch (effective config, resolved runtime target, control mode,
  inference/scheduler state, ``provider_started``). Infrastructure typed
  outputs are propagated into the context as their owning entries build,
  so downstream entries read them without a second orchestration layer.
- :class:`BuildResult` — one launch action segment plus optional typed
  infrastructure outputs (control/simulation/benchmark startup state).
- :class:`LaunchBuilderRegistry` — explicit, instance-scoped registry with
  stable topological ordering over ``after`` edges and hard ``requires``
  participation checks.

No module-level registration side effects: registries are instantiated per
launch (production uses the built-in table from ``builtin_registry``; tests
inject their own instance). Declaration errors (duplicate ids, unknown
dependencies, dependency cycles over the combined ``after``+``requires``
graph) are raised by :meth:`LaunchBuilderRegistry.plan` before any factory
is resolved or called.

Typed-output ownership rules (single owner per output):

- A spec declares ownership via ``provides`` (one of
  ``control_startup`` / ``simulation_startup`` / ``benchmark_startup``);
- two specs declaring the same output are rejected at ``add`` time;
- a spec returning an output it does not declare is rejected at build
  time even when no other entry provided it before.

Registration metadata extensibility (design decision 4): one
``BuilderSpec.add`` can also contribute business launch arguments
(:class:`ArgumentSpec`, merged by the preparation module through
``compose_argument_specs``) and prepare/preflight hooks. Hook phases are
named anchors owned by the preparation module (``load_target``,
``control_mode``, ``config_overrides``, ``inference_override``,
``mock_check``, ``inference_derived``, ``scheduler_check``,
``benchmark_check``, ``resource_check``, ``preflight``); registration
order is the stable execution order within one phase. Hooks are called
as ``hook(robot_config, arguments)``; a hook that additionally accepts a
third positional parameter is called as ``hook(robot_config, arguments,
target_state)`` with the read-only prepared target state
(:class:`HookTargetState`), so preflight hooks can consume the resolved
``use_sim`` / ``control_mode`` without re-deriving them. Name collisions
between contributed arguments (or with core arguments) and cross-spec
config-key collisions for hooks are rejected at ``add`` time, before any
factory or hook is resolved.
"""

from __future__ import annotations

import copy
import importlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from robot_config.launch_registry.launch_options import ArgumentSpec
from robot_config.logger_utils import get_colored_logger
from robot_config.runtime_target import RuntimeTarget

logger = get_colored_logger("robot_config.launch_registry.registry")

VALID_TIMINGS = ("immediate", "runtime_ready")
TYPED_OUTPUT_OWNERS = {
    "control_startup": "control",
    "simulation_startup": "simulation",
    "benchmark_startup": "benchmark",
}
TYPED_OUTPUT_NAMES = tuple(TYPED_OUTPUT_OWNERS)
# Preparation-phase anchors owned by the preparation module (design
# decision 4, ordered anchor table). Registration may attach hooks to
# these named anchors only; safety checkpoints cannot be reordered by
# builder topology. Registration order is the stable tie-breaker within
# one anchor. The business-name anchors (hand_profile, voice_start,
# embodied, voice_override) were collapsed into the generic
# ``config_overrides`` anchor (openspec change
# ``migrate-builtin-builder-metadata``, design decision 4): it has no
# core step and runs the registered business override hooks in
# registration order (builtin embodied first, then extensions sorted
# by entry point name; hand_sources now registers from robot_teleop).
PREPARATION_PHASES = (
    "load_target",  # 0: nav_stage participates; runtime target/backend caps
    "control_mode",  # 1: control_mode override
    "config_overrides",  # 2: business CLI config overrides (registered hooks only)
    "inference_override",  # 3: inference pipeline/execution_mode override
    "mock_check",  # 4: validate_mock_control_mode
    "inference_derived",  # 5: with_inference / teleop fallback derivation
    "scheduler_check",  # 6: scheduler derivation; virtual targets stay direct
    "benchmark_check",  # 7: in-memory transport materialization + closure check
    "resource_check",  # 8: validate_runtime_resources (sees applied hand profile)
    "preflight",  # 9: operator confirmations (e.g. P-pose); interactive, exactly-once
)


class RegistryError(ValueError):
    """Invalid registry declaration detected before any factory runs."""


class CircularDependencyError(RegistryError):
    """The declared dependency graph contains a cycle."""


@dataclass(frozen=True)
class HookSpec:
    """One prepare/preflight hook declaration attached to a phase anchor.

    ``factory`` follows the same ``module:attribute`` lazy-resolution
    rules as builders: resolved on every call, never cached, so the
    source-module monkeypatch seam keeps working.

    ``config_keys`` declares the configuration fields this hook may
    write. Ownership is per spec: the same spec may update its own keys
    in different phases, but two specs declaring the same key are
    rejected at ``add`` time. Override hooks only process configuration
    (no processes/devices/interaction); preflight hooks may prompt the
    operator but must not start processes or open devices and run at
    most once per launch.
    """

    phase: str
    factory: str
    config_keys: Sequence[str] = ()

    def __post_init__(self) -> None:
        if self.phase not in PREPARATION_PHASES:
            raise RegistryError(
                f"hook phase {self.phase!r} is not a named preparation anchor; expected one of {PREPARATION_PHASES}"
            )
        _validate_factory_reference(self.factory)
        if isinstance(self.config_keys, str):
            raise RegistryError("hook config_keys must be a sequence of dotted paths, not a string")
        normalized_keys = tuple(self.config_keys)
        if not normalized_keys:
            raise RegistryError("hook must declare at least one config key it owns")
        for key in normalized_keys:
            if (
                not isinstance(key, str)
                or not key.strip()
                or key != key.strip()
                or key.startswith(".")
                or key.endswith(".")
                or ".." in key
            ):
                raise RegistryError(f"hook config key {key!r} must be a non-empty dotted path")
            if any(not part.strip() or part != part.strip() for part in key.split(".")):
                raise RegistryError(f"hook config key {key!r} must be a non-empty dotted path")
        object.__setattr__(self, "config_keys", normalized_keys)


@dataclass(frozen=True)
class HookTargetState:
    """Read-only prepared target state optionally passed to hooks.

    Hooks are normally called as ``hook(robot_config, arguments)``. A hook
    whose signature additionally accepts a third positional parameter is
    called as ``hook(robot_config, arguments, target_state)`` so preflight
    hooks can consume the resolved runtime-target facts without
    re-deriving them (the preparation core owns the single derivation).
    """

    use_sim: bool = False
    control_mode: str = ""


def _validate_factory_reference(reference: str) -> None:
    """Validate the ``module:attribute`` factory reference format (no import)."""
    module_name, separator, attribute = str(reference).partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise RegistryError(f"invalid factory reference {reference!r}; expected 'module:attribute'")


def resolve_builder_factory(reference: str) -> Callable[..., Any]:
    """Resolve a ``module:attribute`` reference at call time.

    importlib module caching is allowed, but the attribute is looked up on
    every call; the resolved callable is never cached, preserving the
    source-module monkeypatch seam used by the launch tests. The same rule
    applies to prepare/preflight hooks resolved elsewhere in the pipeline.
    """
    module_name, separator, attribute = str(reference).partition(":")
    if not separator or not module_name or not attribute or ":" in attribute:
        raise RegistryError(f"invalid factory reference {reference!r}; expected 'module:attribute'")
    module = importlib.import_module(module_name)
    try:
        factory = getattr(module, attribute)
    except AttributeError as exc:
        raise RegistryError(f"factory reference {reference!r} resolved to a missing attribute") from exc
    if not callable(factory):
        raise RegistryError(f"factory reference {reference!r} resolved to a non-callable object")
    return factory


def isolated_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Deep-copy a caller-provided loaded config for one launch.

    ``launch_setup`` must never mutate the caller's ``loaded_config``; each
    launch works on its own isolated copy.
    """
    return copy.deepcopy(dict(config))


@dataclass(frozen=True)
class BuilderSpec:
    """Pure metadata declaration for one launch builder.

    ``after`` is build ordering: "when the target participates, build it
    first". Edges to non-participating targets are removed before the
    stable ordering pass. ``requires`` is a hard participation dependency:
    the target must be registered and participate in this launch (being
    filtered out by the registry-level enable predicate or mock skip is an
    error). ``requires`` guarantees participation only — a participating
    builder may legally return an empty action list — and never replaces
    process readiness, nor does it contribute ordering edges; ordering
    uses ``after`` only.

    Dependency cycles are rejected on the combined ``after`` + ``requires``
    declaration graph (a pure ``requires`` cycle or an ``after``/``requires``
    mix is equally fatal) before any factory runs.

    ``enabled`` may only read state prepared before provider startup (the
    :class:`LaunchBuildContext` fields); it must not touch live-bound
    effective fields, benchmark run identities, or ``provider_started``.

    ``provides`` declares ownership of typed infrastructure outputs; at
    most one spec may declare each output.
    """

    id: str
    factory: str
    after: Sequence[str] = ()
    requires: Sequence[str] = ()
    timing: str = "immediate"
    enabled: Callable[[LaunchBuildContext], bool] | None = None
    mock_skip_key: str | None = None
    provides: Sequence[str] = ()
    arguments: Sequence[ArgumentSpec] = ()
    hooks: Sequence[HookSpec] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise RegistryError("builder id must be a non-empty string")
        _validate_factory_reference(self.factory)
        if self.timing not in VALID_TIMINGS:
            raise RegistryError(
                f"builder {self.id!r} has invalid timing {self.timing!r}; expected one of {VALID_TIMINGS}"
            )
        object.__setattr__(self, "after", tuple(self.after))
        object.__setattr__(self, "requires", tuple(self.requires))
        object.__setattr__(self, "provides", tuple(self.provides))
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "hooks", tuple(self.hooks))
        if self.id in self.after or self.id in self.requires:
            raise RegistryError(f"builder {self.id!r} cannot declare a dependency on itself")
        for output_name in self.provides:
            if output_name not in TYPED_OUTPUT_NAMES:
                raise RegistryError(
                    f"builder {self.id!r} cannot provide unknown typed output {output_name!r}; "
                    f"expected one of {TYPED_OUTPUT_NAMES}"
                )
        for argument in self.arguments:
            if not isinstance(argument, ArgumentSpec):
                raise RegistryError(
                    f"builder {self.id!r} arguments must be ArgumentSpec instances, got {type(argument).__name__}"
                )
        for hook in self.hooks:
            if not isinstance(hook, HookSpec):
                raise RegistryError(f"builder {self.id!r} hooks must be HookSpec instances, got {type(hook).__name__}")


@dataclass(frozen=True)
class ControlStartup:
    """Typed control-infrastructure output; owned by the single control entry.

    Even when the benchmark target, mock backend or an already-started
    provider means no control processes are created, the control entry
    still participates and returns an (empty) startup state.
    """

    robot_description: Mapping[str, Any] = field(default_factory=dict)
    controller_names: tuple[str, ...] = ()
    deferred_spawners: tuple[Any, ...] = ()
    runtime_ready_waiter: Any = None


@dataclass(frozen=True)
class SimulationStartup:
    """Typed simulation output; owned by the single simulation entry.

    ``entity_action`` is the Gazebo ``ros_gz_sim create`` node consumed by
    the controller-startup entry for the entity -> controller chain
    (provider-less launches).

    ``world_name`` and ``spawn_pose`` carry the world hand-off to a runtime
    provider on a physics transport (runtime-simulation-transport D3):
    generic launch owns the world and resolves the scene, so the provider
    receives the resolved world name and the full robot spawn pose
    (x, y, z, yaw) instead of recomputing them. ``scene_name`` carries the
    scene identity on the MuJoCo transport (so101-mujoco-runtime-platform
    D3/D7): the world owner resolves the scene by name and the provider
    composes its robot MJCF with it. All stay empty/None when generic launch
    owns no world (provider-less or sdk transports).
    """

    entity_action: Any = None
    world_name: str = ""
    spawn_pose: tuple[float, float, float, float] | None = None
    scene_name: str = ""


@dataclass(frozen=True)
class BenchmarkStartup:
    """Materialized run configuration; owned by the benchmark entry alone.

    The orchestrator propagates ``effective_config`` (with the new
    ``_config_path``) and ``run_id`` to execution/task/recording/rerun
    builders, which must not re-materialize or re-assign run identity.
    """

    effective_config: Mapping[str, Any] = field(default_factory=dict)
    config_path: str = ""
    run_id: str = ""


@dataclass
class BuildResult:
    """One builder's output: an action segment plus optional typed outputs.

    ``actions`` may contain any launch action type (Node,
    IncludeLaunchDescription, GroupAction, event actions, ...); an empty
    list is a legal no-op, distinct from the builder being disabled.
    """

    actions: list[Any] = field(default_factory=list)
    control_startup: ControlStartup | None = None
    simulation_startup: SimulationStartup | None = None
    benchmark_startup: BenchmarkStartup | None = None


@dataclass
class LaunchBuildContext:
    """Prepared launch state shared by the builders of one launch.

    Business builders treat the context as read-only. The control and
    simulation startup states appear as the owning entries complete (the
    orchestrator propagates them); when the benchmark entry materializes
    a run configuration, ``robot_config`` is replaced by the materialized
    effective snapshot for all later entries.
    """

    robot_config: Mapping[str, Any]
    launch_context: Any = None
    runtime_target: RuntimeTarget | None = None
    sim_platform: str = ""
    use_sim: bool = False
    node_use_sim_time: bool = False
    control_mode: str = ""
    with_inference: bool = False
    scheduler_enabled: bool = False
    provider_started: bool = False
    control_startup: ControlStartup | None = None
    simulation_startup: SimulationStartup | None = None
    benchmark_startup: BenchmarkStartup | None = None
    # Prepared-only extension fields consumed by the builtin adapters and the
    # readiness orchestrator (not business-writable scratch): the prepared
    # argument table, controller auto-start flag, the prepared
    # ros2_control participation decision, and the barrier action installed
    # by the controller-startup entry for the orchestrator's deferred set.
    arguments: Mapping[str, str] = field(default_factory=dict)
    auto_start_controllers: str = "true"
    sim_backend_needs_ros2_control: bool = False
    controller_readiness_barrier: Any = None

    @property
    def is_benchmark(self) -> bool:
        return self.runtime_target is RuntimeTarget.BENCHMARK

    @property
    def mock_backend_active(self) -> bool:
        return self.use_sim and self.sim_platform == "mock" and not self.is_benchmark


@dataclass(frozen=True)
class LaunchPlan:
    """Immutable, stably-ordered set of participating builders."""

    specs: tuple[BuilderSpec, ...]

    @property
    def immediate_specs(self) -> tuple[BuilderSpec, ...]:
        return tuple(spec for spec in self.specs if spec.timing == "immediate")

    @property
    def runtime_ready_specs(self) -> tuple[BuilderSpec, ...]:
        return tuple(spec for spec in self.specs if spec.timing == "runtime_ready")


@dataclass(frozen=True)
class PlanSegment:
    """One planned builder's action segment, in stable plan order."""

    spec_id: str
    timing: str
    actions: tuple[Any, ...]


@dataclass
class PlanOutputs:
    """Merged build outputs handed to the readiness orchestrator.

    ``actions`` merges every segment in stable plan order — this is the
    observable launch order for the no-barrier case and must not be
    rebuilt by concatenating timing buckets (an early ``runtime_ready``
    entry such as perception would otherwise jump behind later
    ``immediate`` entries such as voice/recording).

    ``segments`` keeps the per-spec structure so the readiness orchestrator
    can split by actual barrier semantics via :meth:`split_for_barrier`
    (which defers the whole ``runtime_ready`` candidate set) or consume
    ``actions`` directly when no barrier exists.
    """

    actions: list[Any] = field(default_factory=list)
    runtime_ready_actions: list[Any] = field(default_factory=list)
    segments: list[PlanSegment] = field(default_factory=list)
    control_startup: ControlStartup | None = None
    simulation_startup: SimulationStartup | None = None
    benchmark_startup: BenchmarkStartup | None = None
    results: dict[str, BuildResult] = field(default_factory=dict)

    def split_for_barrier(self) -> tuple[list[Any], list[Any]]:
        """Return (immediate_actions, deferred_candidates) in plan order."""
        immediate: list[Any] = []
        deferred: list[Any] = []
        for segment in self.segments:
            if segment.timing == "runtime_ready":
                deferred.extend(segment.actions)
            else:
                immediate.extend(segment.actions)
        return immediate, deferred


class LaunchBuilderRegistry:
    """Explicit ordered registry; instantiate one per launch (or per test).

    Registration order is the stable tie-breaker for topological sorting.
    The registry instance is the only mutable state; there is no
    process-level global table, so tests cannot leak registrations into
    subsequent launches.
    """

    def __init__(self) -> None:
        self._specs: dict[str, BuilderSpec] = {}

    def add(self, spec: BuilderSpec) -> BuilderSpec:
        if not isinstance(spec, BuilderSpec):
            raise RegistryError("registry.add expects a BuilderSpec instance")
        if spec.id in self._specs:
            raise RegistryError(f"duplicate builder id {spec.id!r}")
        for output_name in spec.provides:
            for existing in self._specs.values():
                if output_name in existing.provides:
                    raise RegistryError(
                        f"builder {spec.id!r} cannot provide {output_name!r}; already owned by {existing.id!r}"
                    )
        for output_name in spec.provides:
            owner = TYPED_OUTPUT_OWNERS[output_name]
            if spec.id != owner:
                raise RegistryError(f"builder {spec.id!r} cannot provide {output_name!r}; reserved for {owner!r}")
        self._reject_argument_collisions(spec)
        self._reject_hook_config_key_collisions(spec)
        self._specs[spec.id] = spec
        return spec

    def _reject_argument_collisions(self, spec: BuilderSpec) -> None:
        """Reject argument names already contributed by another spec.

        Collisions with the core argument table are rejected by
        ``compose_argument_specs`` when the preparation module merges the
        aggregated contributions; spec-vs-spec collisions are rejected
        here, before any factory or hook is resolved.
        """
        for argument in spec.arguments:
            for existing in self._specs.values():
                if any(argument.name == other.name for other in existing.arguments):
                    raise RegistryError(
                        f"builder {spec.id!r} cannot contribute launch argument {argument.name!r}; "
                        f"already contributed by {existing.id!r}"
                    )

    def _reject_hook_config_key_collisions(self, spec: BuilderSpec) -> None:
        """Reject cross-spec config-key ownership collisions for hooks.

        The same spec may update its own keys in different phases; two
        different specs owning the same key is a declaration error.
        """
        own_keys = {key for hook in spec.hooks for key in hook.config_keys}
        for existing in self._specs.values():
            existing_keys = {key for hook in existing.hooks for key in hook.config_keys}
            colliding = {
                own
                for own in own_keys
                for existing_key in existing_keys
                if own == existing_key or own.startswith(f"{existing_key}.") or existing_key.startswith(f"{own}.")
            }
            if colliding:
                keys = ", ".join(sorted(colliding))
                raise RegistryError(
                    f"builder {spec.id!r} hooks cannot own config keys already owned by {existing.id!r}: {keys}"
                )

    def add_all(self, specs: Iterable[BuilderSpec]) -> None:
        for spec in specs:
            self.add(spec)

    def snapshot(self) -> tuple[BuilderSpec, ...]:
        """Immutable registration snapshot in registration order."""
        return tuple(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)

    def __contains__(self, spec_id: object) -> bool:
        return spec_id in self._specs

    def contributed_arguments(self) -> tuple[ArgumentSpec, ...]:
        """All business argument contributions in registration order.

        The preparation module merges this tuple with the core table via
        ``compose_argument_specs`` (which rejects collisions with core
        argument names) — one registration entry is all a business
        module needs to add both CLI arguments and hooks.
        """
        return tuple(argument for spec in self._specs.values() for argument in spec.arguments)

    def hooks_by_phase(self) -> tuple[tuple[str, tuple[tuple[str, HookSpec], ...]], ...]:
        """Aggregate declared hooks per phase anchor, in anchor order.

        Returns one ``(phase, ((spec_id, hook), ...))`` entry per anchor
        (including empty ones) following :data:`PREPARATION_PHASES`;
        within a phase, registration order is the stable execution
        order. Hook callables are NOT resolved here — the preparation
        module resolves ``hook.factory`` on every call through
        :func:`resolve_builder_factory`, preserving the test seam.
        """
        aggregate: list[tuple[str, tuple[tuple[str, HookSpec], ...]]] = []
        for phase in PREPARATION_PHASES:
            entries = tuple(
                (spec.id, hook) for spec in self._specs.values() for hook in spec.hooks if hook.phase == phase
            )
            aggregate.append((phase, entries))
        return tuple(aggregate)

    def validate_declarations(self) -> None:
        """Validate the full graph without imports, enable predicates or factories."""
        self._validate_declaration_graph(self.snapshot())

    def plan(self, context: LaunchBuildContext) -> LaunchPlan:
        """Validate the declaration graph and produce the immutable plan.

        Ordering of checks (per design):

        1. Full declaration graph (all registered specs, including
           disabled ones): unknown ``after``/``requires`` targets and
           cycles over the combined ``after``+``requires`` graph are
           rejected here.
        2. Enablement: the registry-level enable predicates and mock-skip
           keys decide which specs participate in this launch; every
           non-participating spec is reported once through a skip log
           carrying its id and the owning filter (mock skip key or
           enabled predicate).
        3. Hard ``requires``: every participating spec's requires targets
           must also participate; a non-participating target is an error.
        4. ``after`` edges to non-participating targets are removed; the
           participating graph is ordered stably over the remaining
           ``after`` edges (registration order as tie-breaker).

        No factory is resolved or called during planning.
        """
        specs = self.snapshot()
        self._validate_declaration_graph(specs)
        active: list[BuilderSpec] = []
        for spec in specs:
            # One evaluation per entry: the skip reason is computed once and
            # reported here, replacing the per-adapter skip logs the business
            # entries used to print inside their build bodies.
            reason = self._skip_reason(spec, context)
            if reason is None:
                active.append(spec)
            else:
                logger.info(f"launch builder '{spec.id}' skipped: {reason}")
        active_tuple = tuple(active)
        active_ids = {spec.id for spec in active_tuple}
        for spec in active_tuple:
            for target in spec.requires:
                if target not in active_ids:
                    raise RegistryError(
                        f"builder {spec.id!r} requires {target!r}, which is not participating in this launch"
                    )
        edges = {spec.id: {target for target in spec.after if target in active_ids} for spec in active_tuple}
        ordered_ids = self._stable_topological_order([spec.id for spec in active_tuple], edges)
        by_id = {spec.id: spec for spec in active_tuple}
        return LaunchPlan(specs=tuple(by_id[spec_id] for spec_id in ordered_ids))

    def _skip_reason(self, spec: BuilderSpec, context: LaunchBuildContext) -> str | None:
        """Return why ``spec`` does not participate, or ``None`` when it does.

        The mock-skip declaration wins over the enable predicate (the mock
        backend filters before any business predicate is consulted). The
        reason text distinguishes the two registry-level filters so the
        skip log identifies the owning mechanism.
        """
        if spec.mock_skip_key is not None:
            # Local import: keep module import free of the builder facade's
            # eager imports and avoid any chance of an import cycle.
            from robot_config.launch_builders.hardware_mock import mock_mode_skips_subsystem

            if mock_mode_skips_subsystem(context.mock_backend_active, spec.mock_skip_key):
                return f"mock skip key '{spec.mock_skip_key}'"
        if spec.enabled is not None and not bool(spec.enabled(context)):
            return "enabled predicate"
        return None

    def _is_active(self, spec: BuilderSpec, context: LaunchBuildContext) -> bool:
        return self._skip_reason(spec, context) is None

    def _validate_declaration_graph(self, specs: Sequence[BuilderSpec]) -> None:
        known = {spec.id for spec in specs}
        for spec in specs:
            for target in (*spec.after, *spec.requires):
                if target not in known:
                    raise RegistryError(f"builder {spec.id!r} references unknown dependency {target!r}")
        # Cycle detection over the combined after+requires declaration
        # graph: ordering uses after only, but a requires edge closing a
        # cycle is equally a declaration error.
        union_edges = {spec.id: set(spec.after) | set(spec.requires) for spec in specs}
        self._stable_topological_order([spec.id for spec in specs], union_edges)

    @staticmethod
    def _stable_topological_order(ids: Sequence[str], edges: Mapping[str, set[str]]) -> list[str]:
        """Kahn's algorithm with registration order as the stable tie-breaker."""
        remaining = list(ids)
        placed: list[str] = []
        placed_set: set[str] = set()
        while remaining:
            candidates = [spec_id for spec_id in remaining if edges.get(spec_id, set()) <= placed_set]
            if not candidates:
                cycle = ", ".join(sorted(remaining))
                raise CircularDependencyError(f"circular or unsatisfiable builder dependency order among: {cycle}")
            # ``remaining`` preserves registration order, so the first
            # unlocked candidate is the earliest-registered one.
            next_id = candidates[0]
            placed.append(next_id)
            placed_set.add(next_id)
            remaining.remove(next_id)
        return placed


def execute_plan(plan: LaunchPlan, context: LaunchBuildContext) -> PlanOutputs:
    """Run every planned factory in stable build order and merge outputs.

    Typed outputs are validated against ``provides`` ownership, propagated
    into ``context`` for downstream entries, and merged into
    :class:`PlanOutputs`. The benchmark materialized snapshot replaces
    ``context.robot_config`` for all later entries. Actions merge in plan
    order (see :class:`PlanOutputs`); ``runtime_ready_actions`` records the
    gated candidates, but whether they are actually deferred is decided by
    the readiness orchestrator.

    Business exceptions are re-raised wrapped with the builder id while
    preserving the original exception chain; registry contract violations
    (non-BuildResult returns, undeclared typed outputs) raise
    :class:`RegistryError` directly and recording/rerun tolerance policies
    live in their adapters, not in this wrapper.
    """
    outputs = PlanOutputs()
    for spec in plan.specs:
        factory = resolve_builder_factory(spec.factory)
        try:
            result = factory(context)
        except Exception as exc:
            raise RuntimeError(f"launch builder {spec.id!r} failed: {exc}") from exc
        if not isinstance(result, BuildResult):
            raise RegistryError(f"builder {spec.id!r} factory must return a BuildResult, got {type(result).__name__}")
        outputs.results[spec.id] = result
        _attach_typed_outputs(spec, result, context, outputs)
        outputs.segments.append(PlanSegment(spec.id, spec.timing, tuple(result.actions)))
        outputs.actions.extend(result.actions)
        if spec.timing == "runtime_ready":
            outputs.runtime_ready_actions.extend(result.actions)
    return outputs


def _attach_typed_outputs(
    spec: BuilderSpec,
    result: BuildResult,
    context: LaunchBuildContext,
    outputs: PlanOutputs,
) -> None:
    """Validate ownership, then propagate typed outputs to context and outputs."""
    for output_name in TYPED_OUTPUT_NAMES:
        value = getattr(result, output_name)
        if value is None:
            continue
        if output_name not in spec.provides:
            raise RegistryError(
                f"builder {spec.id!r} returned {output_name} without declaring ownership via 'provides'"
            )
        owner = TYPED_OUTPUT_OWNERS[output_name]
        if spec.id != owner:
            raise RegistryError(f"builder {spec.id!r} cannot provide {output_name}; reserved for {owner!r}")
        expected_type = {
            "control_startup": ControlStartup,
            "simulation_startup": SimulationStartup,
            "benchmark_startup": BenchmarkStartup,
        }[output_name]
        if not isinstance(value, expected_type):
            raise RegistryError(f"builder {spec.id!r} must return {expected_type.__name__} for {output_name}")
        if getattr(outputs, output_name) is not None:
            raise RegistryError(f"builder {spec.id!r} cannot provide {output_name}; another entry already owns it")
        setattr(outputs, output_name, value)
        setattr(context, output_name, value)
    benchmark = outputs.benchmark_startup
    if benchmark is not None:
        # The benchmark materialization is the sole cross-entry config
        # owner: later builders must observe the materialized effective
        # snapshot (with its new ``_config_path``), never their own clone
        # of the pre-benchmark config.
        context.robot_config = benchmark.effective_config
