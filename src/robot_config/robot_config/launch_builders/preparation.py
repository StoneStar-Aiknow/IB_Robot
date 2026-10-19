"""Configuration preparation pipeline for the registry-based launch entry.

Implements design decision 4 of openspec change
``refactor-launch-builder-registry``: the legacy ``launch_setup``
configuration half is split into a fixed, named phase pipeline that ends
before any provider startup or node construction. The result
(:class:`PreparedLaunch`) is consumed by the generic orchestrator and,
through it, the main launch entry.

Phase order (openspec change ``migrate-builtin-builder-metadata`` moved
the business override steps into registered hooks on the generic
``config_overrides`` anchor):

0. load / target — nav_stage participates in config loading; runtime
   target, backend capabilities, provider and simulation platform
   resolution.
1. control-mode — ``control_mode`` CLI override.
2. config-overrides — registered business override hooks only (builtin:
   hand profile -> voice ASR -> embodied); no core step.
3. inference-override — inference pipeline/execution_mode overrides.
4. mock-check — ``validate_mock_control_mode``.
5. inference-derived — ``with_inference`` and teleop fallback derivation.
6. scheduler-check — scheduler derivation; reject simulation+scheduler.
7. benchmark-check — in-memory transport materialization and inference
   closure validation (no run_id allocation here; that belongs to the
   benchmark build adapter).
8. resource-check — ``validate_runtime_resources`` (sees the applied
   hand profile).
9. preflight — registered preflight hooks (builtin: the interactive
   P-pose confirmation), exactly once, before any provider startup.
10a. binding-target — when live binding is required, verify provider
   presence and a non-benchmark target, then write the resolved
   ``runtime.target`` back into this launch's config snapshot and fix
   ``node_use_sim_time=False`` (SDK transports provide no /clock).
10b. plan — validate the declaration/enable graph and freeze the plan.

Registered hooks (``HookSpec``) execute at their declared anchor inside
this pipeline, BEFORE the anchor's core step; their factories resolve at
call time through the real resolver (never cached) and receive
``(robot_config, arguments)`` — or ``(robot_config, arguments,
target_state)`` for hooks that declare the third parameter — so
registered CLI overrides are declarative. Core/business argument and
config-key collisions are rejected before any side-effectful phase
runs. Preflight extension hooks run on an isolated deep copy and must
not modify configuration; the builtin P-pose confirmation stays
interactive, exactly-once, and consumes the read-only
:class:`HookTargetState`. ``controller_startup_timeout`` is
intentionally NOT resolved here: only the control/provider consumption
path resolves it, preserving the legacy benchmark/mock branches that
skip control startup.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from robot_config.interface_binding import required_interface_ids
from robot_config.launch_builders import readiness
from robot_config.launch_builders.launch_options import (
    ArgumentSpec,
    compose_argument_specs,
    read_launch_arguments,
)
from robot_config.launch_builders.registry import (
    HookSpec,
    HookTargetState,
    LaunchBuildContext,
    LaunchBuilderRegistry,
    LaunchPlan,
    isolated_config,
    resolve_builder_factory,
)
from robot_config.logger_utils import get_colored_logger
from robot_config.runtime_target import RuntimeTarget, resolve_runtime_target
from robot_config.utils import parse_bool

logger = get_colored_logger("robot_config.launch_builders.preparation")


class PreparationError(ValueError):
    """Raised when the preparation pipeline cannot proceed."""


# Top-level configuration keys the preparation core owns and derives across
# its phases. Registered hooks may not declare these keys (or any parent
# path of them) as their owned config fields: a late hook rewriting an
# already-derived core field (runtime target, control mode selection,
# scheduler/inference decisions) would invalidate every downstream phase.
# Keys written by the core across phases 0-13b: the business override keys
# (hand profile / voice ASR / embodied) moved to their owning BuilderSpec
# hooks (openspec change ``migrate-builtin-builder-metadata``, design
# decision 4) and are no longer core-forbidden; the cross-spec key
# collision check in the registry keeps other extensions from claiming
# them.
CORE_HOOK_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "_config_path",
        "runtime",  # 13a writes runtime.target; provider/profile live here
        "simulation",  # phase 0 writes simulation.platform
        "default_control_mode",  # phase 1 override
        "control_modes",  # phase 3 pipeline execution_mode override
        "runtime_target",  # reserved core derivation namespace
        "scheduler",  # phase 6 derivation namespace
        "benchmark",  # phase 7 transport materialization / run identity
        "contract",  # binding requirements must not be changed after validation
        "peripherals",  # resource validation and provider ownership
        "controllers",  # control infrastructure resources
    }
)


def _validate_hook_key_ownership(registry: LaunchBuilderRegistry) -> None:
    """Reject hook declarations that touch core-owned config fields.

    Runs before any side-effectful phase (right after argument
    composition). Rules:

    - a hook key that is itself a core key is rejected;
    - a hook key that is a child path of a core key is rejected (e.g.
      ``runtime.target``, ``control_modes.<mode>.inference``);
    - a core key that is a child of a hook key is rejected too (a hook
      owning ``x`` while the core derives ``x.y``).
    Extension hooks own only their own business configuration fields.
    """
    for spec in registry.snapshot():
        for hook in spec.hooks:
            for key in hook.config_keys:
                if key in CORE_HOOK_FORBIDDEN_KEYS:
                    raise PreparationError(f"hook key {key!r} of builder {spec.id!r} is owned by the preparation core")
                for core_key in CORE_HOOK_FORBIDDEN_KEYS:
                    if key.startswith(f"{core_key}.") or core_key.startswith(f"{key}."):
                        raise PreparationError(
                            f"hook key {key!r} of builder {spec.id!r} collides with core-owned "
                            f"configuration path {core_key!r}"
                        )


@dataclass
class PreparedLaunch:
    """Everything the orchestrator needs after configuration preparation.

    ``robot_config`` is the isolated, fully-prepared snapshot for this
    launch (the caller's ``loaded_config`` is never mutated). ``plan`` is
    the frozen participation plan; hooks have already run.
    """

    context: LaunchBuildContext
    plan: LaunchPlan
    robot_config: dict[str, Any]
    launch_context: Any
    runtime_target: RuntimeTarget
    sim_platform: str
    use_sim: bool
    node_use_sim_time: bool
    control_mode: str
    with_inference: bool
    scheduler_enabled: bool
    auto_start_controllers: str
    sim_backend_needs_ros2_control: bool
    runtime_provider_name: str
    arguments: dict[str, str] = field(default_factory=dict)
    argument_specs: tuple[ArgumentSpec, ...] = ()
    live_binding_required: bool = False


def compose_registry_arguments(registry: LaunchBuilderRegistry) -> tuple[ArgumentSpec, ...]:
    """Merge core and registry-contributed arguments, rejecting collisions.

    This runs before any side-effectful phase: name collisions between
    contributed arguments and the core table are fatal here, not later.
    """
    registry.validate_declarations()
    _validate_hook_key_ownership(registry)
    return compose_argument_specs(registry.contributed_arguments())


def _load_config(arguments: Mapping[str, str], loaded_config: Mapping[str, Any] | None) -> dict[str, Any]:
    if loaded_config is not None:
        config = isolated_config(loaded_config)
    else:
        config = readiness.load_launch_robot_config(
            arguments["robot_config"],
            arguments["config_path"] or None,
            arguments["nav_stage"],
            defer_interface_binding=True,
        )
    config["_config_path"] = readiness.resolve_config_path(arguments["robot_config"], arguments["config_path"])
    return config


def _resolve_target_and_backend(arguments: Mapping[str, str], robot_config: dict[str, Any]) -> dict[str, Any]:
    """Design phase 0: runtime target, provider, sim platform, backend caps."""
    use_sim_str = arguments["use_sim"]
    requested_use_sim = None if use_sim_str == "" else parse_bool(use_sim_str, default=False)
    runtime_target = resolve_runtime_target(robot_config, arguments["runtime_target"], requested_use_sim)
    use_sim = runtime_target is not RuntimeTarget.HARDWARE
    is_benchmark = runtime_target is RuntimeTarget.BENCHMARK
    logger.info(f"Runtime target: {runtime_target}")
    logger.info(f"Effective use_sim: {use_sim}")
    if is_benchmark:
        print("[IBROBOT_BENCHMARK][RUNTIME_TARGET] benchmark")
        print("[IBROBOT_BENCHMARK][SKIP] ros2_control")
        print("[IBROBOT_BENCHMARK][SKIP] simulation_backend")
        print("[IBROBOT_BENCHMARK][SKIP] physical_perception")

    from robot_config.launch_builders.runtime import runtime_provider
    from robot_config.launch_builders.sim_backend import get_backend_caps

    runtime_provider_name = runtime_provider(robot_config)
    sim_platform = str(
        robot_config.get("simulation", {}).get("platform", "sdk" if runtime_provider_name else "gazebo")
    ).lower()
    sim_platform_override = arguments["sim_platform"]
    if sim_platform_override:
        logger.info(f"CLI override: simulation.platform={sim_platform_override} (was {sim_platform})")
        sim_platform = sim_platform_override
        robot_config.setdefault("simulation", {})["platform"] = sim_platform
    if runtime_provider_name and not is_benchmark:
        if sim_platform != "sdk":
            raise PreparationError(
                "runtime.provider requires sim_platform:=sdk; physics and legacy mock are unsupported"
            )
        backend_caps = {"provides_clock": False, "needs_ros2_control": False}
    elif is_benchmark:
        backend_caps = {"provides_clock": False, "needs_ros2_control": False}
    else:
        backend_caps = (
            get_backend_caps(sim_platform)
            if use_sim
            else {
                "provides_clock": False,
                "needs_ros2_control": True,
            }
        )
    node_use_sim_time = use_sim and backend_caps["provides_clock"]
    sim_backend_needs_ros2_control = (
        ((not use_sim) or backend_caps["needs_ros2_control"]) and not is_benchmark and not runtime_provider_name
    )
    auto_start_controllers = arguments["auto_start_controllers"]
    if runtime_provider_name:
        auto_start_controllers = "false"
        logger.info(f"runtime.provider={runtime_provider_name}: delegating bring-up to the robot runtime")
    elif use_sim and not backend_caps["needs_ros2_control"] and not is_benchmark:
        auto_start_controllers = "false"
        logger.info(f"simulation.platform={sim_platform}: backend does not use ros2_control")
    return {
        "runtime_target": runtime_target,
        "use_sim": use_sim,
        "is_benchmark": is_benchmark,
        "runtime_provider_name": runtime_provider_name,
        "sim_platform": sim_platform,
        "node_use_sim_time": node_use_sim_time,
        "sim_backend_needs_ros2_control": sim_backend_needs_ros2_control,
        "auto_start_controllers": auto_start_controllers,
        "mock_backend_active": (use_sim and sim_platform == "mock") and not is_benchmark,
    }


def _apply_control_mode_override(arguments: Mapping[str, str], robot_config: dict[str, Any]) -> None:
    """Design phase 1: control_mode override."""
    control_mode_override = arguments["control_mode"]
    if control_mode_override:
        robot_config["default_control_mode"] = control_mode_override


# Design phase 2 (config_overrides): the business CLI config overrides
# (hand profile, voice ASR, embodied) moved to their owning BuilderSpec
# hooks (openspec change ``migrate-builtin-builder-metadata``, design
# decision 4). The anchor has no core step; hooks run in registration
# order (builtin: hand_sources -> voice_asr -> embodied), matching the
# legacy hand -> voice_start -> embodied order. The former voice_override
# step moved ahead of inference_override/mock_check: those steps only
# read control mode and control_modes, never voice_asr, so the final
# configuration is unchanged.


def _apply_inference_overrides(arguments: Mapping[str, str], robot_config: dict[str, Any], control_mode: str) -> None:
    """Design phase 3: inference pipeline/execution_mode CLI overrides.

    Migrated verbatim from the legacy ``_apply_inference_cli_overrides``
    but reading from the prepared argument table instead of the launch
    context.
    """
    execution_mode = arguments["inference_execution_mode"].strip().lower()
    pipeline_id = arguments["inference_pipeline"].strip()
    if not execution_mode:
        if pipeline_id:
            raise PreparationError("inference_pipeline requires inference_execution_mode")
        return
    if execution_mode not in {"monolithic", "distributed"}:
        raise PreparationError("inference_execution_mode must be 'monolithic' or 'distributed'")
    if not pipeline_id:
        raise PreparationError("inference_execution_mode requires inference_pipeline")

    inference = robot_config.get("control_modes", {}).get(control_mode, {}).get("inference", {})
    pipelines = inference.get("pipelines", {})
    if pipeline_id not in pipelines:
        raise PreparationError(
            f"inference_pipeline selects unknown pipeline {pipeline_id!r}; available pipelines: {list(pipelines)}"
        )
    previous = pipelines[pipeline_id].get("execution_mode")
    pipelines[pipeline_id]["execution_mode"] = execution_mode
    logger.info(
        f"CLI override: control_modes.{control_mode}.inference.pipelines.{pipeline_id}."
        f"execution_mode={execution_mode} (was {previous})"
    )


def _check_mock_control_mode(robot_config: dict[str, Any], control_mode: str, mock_backend_active: bool) -> None:
    """Design phase 4: validate_mock_control_mode."""
    if mock_backend_active:
        from robot_config.launch_builders.hardware_mock import validate_mock_control_mode

        validate_mock_control_mode(control_mode)


def _derive_inference(arguments: Mapping[str, str], robot_config: dict[str, Any], control_mode: str) -> bool:
    """Design phase 5: with_inference derivation with teleop fallback."""
    with_inference_str = arguments["with_inference"]
    if with_inference_str != "":
        with_inference = parse_bool(with_inference_str, default=False)
    else:
        control_mode_config = robot_config.get("control_modes", {}).get(control_mode, {})
        with_inference = control_mode_config.get("inference", {}).get("enabled", False)

    if control_mode == "teleop" and with_inference_str == "":
        with_inference = False
        logger.info("Teleop mode: forcing with_inference=False")

    logger.info(f"Final with_inference={with_inference}")
    return with_inference


def _check_scheduler(robot_config: dict[str, Any], control_mode: str, with_inference: bool, runtime_target) -> bool:
    """Design phase 6: scheduler derivation; reject simulation+scheduler."""
    from robot_config.inference_config import scheduler_enabled_from_raw_config

    scheduler_enabled = with_inference and scheduler_enabled_from_raw_config(robot_config, control_mode)
    if scheduler_enabled and runtime_target is RuntimeTarget.SIMULATION:
        raise PreparationError(
            "simulation does not support inference.scheduler.enable=true; set scheduler.enable=false"
        )
    return scheduler_enabled


def _check_benchmark(
    robot_config: dict[str, Any], *, control_mode: str, with_inference: bool, is_benchmark: bool
) -> None:
    """Design phase 7: in-memory transport materialization + closure check.

    No run_id allocation here — the benchmark build adapter owns run
    identity materialization during plan execution.
    """

    def _validate_benchmark_inference_closure() -> None:
        from robot_config.inference_config import InferenceConfigError, parse_inference_config

        benchmark = robot_config.get("benchmark", {})
        evaluation = benchmark.get("evaluation", {}) if isinstance(benchmark, dict) else {}
        if not isinstance(evaluation, dict) or not bool(evaluation.get("enabled", False)):
            return
        if not with_inference:
            raise PreparationError("benchmark evaluation requires with_inference:=true")

        mode_config = robot_config.get("control_modes", {}).get(control_mode, {})
        if not isinstance(mode_config, dict):
            raise PreparationError(f"benchmark evaluation control mode {control_mode!r} is not configured")
        executor = mode_config.get("executor", {})
        if not isinstance(executor, dict) or executor.get("type") != "benchmark":
            raise PreparationError("benchmark evaluation requires control_modes.<mode>.executor.type=benchmark")
        dispatch = mode_config.get("dispatch", {}) or {}
        if not isinstance(dispatch, dict) or dispatch.get("scheduler", "continuous") != "wait_for_feedback":
            raise PreparationError("benchmark evaluation requires dispatch.scheduler=wait_for_feedback")

        try:
            inference = parse_inference_config(robot_config, control_mode)
        except InferenceConfigError as exc:
            raise PreparationError(f"benchmark evaluation inference configuration is invalid: {exc}") from exc
        if not inference.enabled or not inference.pipelines:
            raise PreparationError("benchmark evaluation requires an enabled inference pipeline")
        selected = executor.get("inference_pipeline")
        if selected is None:
            if len(inference.pipelines) != 1:
                raise PreparationError(
                    "benchmark evaluation requires executor.inference_pipeline when multiple pipelines exist"
                )
            selected = next(iter(inference.pipelines))
        if not isinstance(selected, str) or selected not in inference.pipelines:
            raise PreparationError(f"benchmark evaluation selects unknown inference pipeline {selected!r}")
        pipeline = inference.pipelines[selected]
        if pipeline.execution_mode not in {"monolithic", "distributed"}:
            raise PreparationError(f"benchmark evaluation selected pipeline {selected!r} has invalid execution mode")
        transport = pipeline.transport
        required_endpoints = {
            "action_server": transport.action_server,
            "reset_service": transport.reset_service,
            "health_topic": transport.health_topic,
        }
        if pipeline.execution_mode == "distributed":
            required_endpoints.update(
                {
                    "request_topic": transport.request_topic,
                    "result_topic": transport.result_topic,
                    "heartbeat_topic": transport.heartbeat_topic,
                    "video_descriptor_topic": transport.video_descriptor_topic,
                    "video_status_topic": transport.video_status_topic,
                }
            )
        missing = sorted(name for name, value in required_endpoints.items() if not isinstance(value, str) or not value)
        if missing:
            raise PreparationError(
                f"benchmark evaluation selected pipeline {selected!r} is missing required endpoints: {missing}"
            )

    if not is_benchmark:
        return
    from robot_config.benchmark_observation_transport import materialize_benchmark_observation_transport

    materialize_benchmark_observation_transport(robot_config)
    _validate_benchmark_inference_closure()


def _check_resources(robot_config: dict[str, Any], *, use_sim: bool, control_mode: str) -> None:
    """Design phase 8: validate_runtime_resources on the effective config."""
    from robot_config.launch_builders.control import validate_runtime_resources

    validate_runtime_resources(robot_config, use_sim=use_sim, control_mode=control_mode)


# Design phase 9 (preflight): the interactive P-pose confirmation moved to
# the hand_sources preflight hook (openspec change
# ``migrate-builtin-builder-metadata``, design decision 4). Preflight hooks
# run on an isolated deep copy before any provider startup, consume the
# read-only HookTargetState (use_sim / control_mode), and must not modify
# configuration; cancel/EOF still fails the launch in the calling thread.


def _changed_config_paths(before: Any, after: Any, prefix: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """Identify changed YAML paths without treating dots in literal keys as separators."""
    if isinstance(before, dict) and isinstance(after, dict):
        changed: set[tuple[str, ...]] = set()
        for key in before.keys() | after.keys():
            path = (*prefix, str(key))
            if key not in before or key not in after:
                value = after[key] if key in after else before[key]
                if isinstance(value, dict) and value:
                    changed.update(_changed_config_paths({}, value, path))
                else:
                    changed.add(path)
            else:
                changed.update(_changed_config_paths(before[key], after[key], path))
        return changed
    return set() if before == after else {prefix}


def _path_owned(path: tuple[str, ...], owned_keys: tuple[str, ...]) -> bool:
    for key in owned_keys:
        parts = tuple(key.split("."))
        if path[: len(parts)] == parts:
            return True
    return False


def _run_phase_hooks(
    phase: str,
    hooks: tuple[tuple[str, HookSpec], ...],
    robot_config: dict[str, Any],
    arguments: Mapping[str, str],
    target_state: HookTargetState | None = None,
) -> None:
    """Execute hooks transactionally and enforce each hook's config ownership.

    Hooks are called as ``hook(robot_config, arguments)``; a hook whose
    signature additionally accepts a third positional parameter receives
    the read-only prepared target state instead (``hook(robot_config,
    arguments, target_state)``), keeping the legacy two-parameter
    convention compatible. ``target_state`` must be provided for the
    ``preflight`` phase (the builtin P-pose hook consumes it).
    """
    readonly_arguments = MappingProxyType(arguments)
    for spec_id, hook in hooks:
        hook_factory = resolve_builder_factory(hook.factory)
        hook_config = copy.deepcopy(robot_config)
        try:
            _invoke_hook(hook_factory, hook_config, readonly_arguments, target_state)
            changed = _changed_config_paths(robot_config, hook_config)
            if phase == "preflight":
                if changed:
                    raise PreparationError(
                        f"preparation hook {spec_id!r} (preflight) must not modify the configuration"
                    )
            else:
                unauthorized = sorted(
                    ".".join(path) for path in changed if not _path_owned(path, tuple(hook.config_keys))
                )
                if unauthorized:
                    raise PreparationError(
                        f"preparation hook {spec_id!r} ({phase}) modified unowned configuration paths: "
                        f"{', '.join(unauthorized)}"
                    )
                if changed:
                    robot_config.clear()
                    robot_config.update(copy.deepcopy(hook_config))
        except PreparationError:
            raise
        except Exception as exc:
            raise RuntimeError(f"preparation hook {spec_id!r} ({phase}) failed: {exc}") from exc


def _invoke_hook(hook_factory, hook_config, readonly_arguments, target_state) -> None:
    """Call one hook with the legacy two-parameter or the extended signature."""
    if target_state is None:
        hook_factory(hook_config, readonly_arguments)
        return
    try:
        signature = inspect.signature(hook_factory)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        try:
            signature.bind(hook_config, readonly_arguments, target_state)
        except TypeError:
            pass
        else:
            hook_factory(hook_config, readonly_arguments, target_state)
            return
    hook_factory(hook_config, readonly_arguments)


def prepare_launch(
    launch_context,
    registry: LaunchBuilderRegistry,
    *,
    loaded_config: Mapping[str, Any] | None = None,
) -> PreparedLaunch:
    """Run the full preparation pipeline (design phases 0-13b).

    ``loaded_config`` follows the legacy ``launch_setup`` convention: a
    preloaded configuration that is still normalized and live-bound
    here; it is deep-copied first so the caller's mapping is never
    mutated.
    """
    argument_specs = compose_registry_arguments(registry)
    # Argument composition also validates the graph and hook ownership
    # before loading configuration or executing any interactive phase.
    arguments = dict(read_launch_arguments(launch_context.launch_configurations, argument_specs))
    # Legacy normalization (S10): the old entry point normalized these
    # CLI overrides before use; keep the exact strip/lower rules. The
    # hand_profile normalization moved into the hand_sources hook with its
    # ownership (it is applied before apply_hand_profile reads it).
    arguments["sim_platform"] = arguments["sim_platform"].strip().lower()
    arguments["nav_stage"] = arguments["nav_stage"].strip()

    # Declaration-only collision checks ran at registry.add time; the
    # core-vs-business argument merge above is the last gate before any
    # side-effectful phase.
    robot_config = _load_config(arguments, loaded_config)

    # Registered hooks aggregate once; each phase's hooks run BEFORE the
    # corresponding core step (S9), so hook inputs can never observe a
    # stale derived/validated state and cannot rewrite already validated
    # configuration.
    hooks_by_phase = dict(registry.hooks_by_phase())

    _run_phase_hooks("load_target", hooks_by_phase["load_target"], robot_config, arguments)
    target_state = _resolve_target_and_backend(arguments, robot_config)
    hook_target_state = HookTargetState(use_sim=target_state["use_sim"], control_mode="")

    _run_phase_hooks("control_mode", hooks_by_phase["control_mode"], robot_config, arguments, hook_target_state)
    _apply_control_mode_override(arguments, robot_config)
    active_control_mode = robot_config.get("default_control_mode", "model_inference")
    logger.info(f"Active control mode: {active_control_mode}")
    hook_target_state = HookTargetState(use_sim=target_state["use_sim"], control_mode=active_control_mode)

    # config_overrides: no core step; the registered business override
    # hooks (hand profile -> voice ASR -> embodied in builtin registration
    # order) run here, before inference overrides and mock validation.
    _run_phase_hooks("config_overrides", hooks_by_phase["config_overrides"], robot_config, arguments, hook_target_state)

    _run_phase_hooks(
        "inference_override", hooks_by_phase["inference_override"], robot_config, arguments, hook_target_state
    )
    _apply_inference_overrides(arguments, robot_config, active_control_mode)

    _run_phase_hooks("mock_check", hooks_by_phase["mock_check"], robot_config, arguments, hook_target_state)
    _check_mock_control_mode(robot_config, active_control_mode, target_state["mock_backend_active"])

    _run_phase_hooks(
        "inference_derived", hooks_by_phase["inference_derived"], robot_config, arguments, hook_target_state
    )
    with_inference = _derive_inference(arguments, robot_config, active_control_mode)

    _run_phase_hooks("scheduler_check", hooks_by_phase["scheduler_check"], robot_config, arguments, hook_target_state)
    scheduler_enabled = _check_scheduler(
        robot_config, active_control_mode, with_inference, target_state["runtime_target"]
    )

    _run_phase_hooks("benchmark_check", hooks_by_phase["benchmark_check"], robot_config, arguments, hook_target_state)
    _check_benchmark(
        robot_config,
        control_mode=active_control_mode,
        with_inference=with_inference,
        is_benchmark=target_state["is_benchmark"],
    )

    _run_phase_hooks("resource_check", hooks_by_phase["resource_check"], robot_config, arguments, hook_target_state)
    _check_resources(robot_config, use_sim=target_state["use_sim"], control_mode=active_control_mode)

    # Preflight hooks (the builtin P-pose confirmation among them) run on
    # an isolated copy before any provider startup: interactive,
    # exactly-once, never rewriting configuration.
    _run_phase_hooks("preflight", hooks_by_phase["preflight"], robot_config, arguments, hook_target_state)

    # 13a. binding-target: the write-back belongs to the forced snapshot
    # data flow and only applies to the legacy live-binding branch.
    live_binding_required = bool(
        required_interface_ids(robot_config) or (robot_config.get("runtime") or {}).get("require_model")
    )
    node_use_sim_time = target_state["node_use_sim_time"]
    if live_binding_required:
        if not target_state["runtime_provider_name"] or target_state["is_benchmark"]:
            raise PreparationError(
                "Live logical interface binding requires runtime.provider and a non-benchmark target"
            )
        # Provider simulation is an SDK transport, not a Gazebo /clock owner.
        node_use_sim_time = False
        robot_config.setdefault("runtime", {})["target"] = target_state["runtime_target"].value

    context = LaunchBuildContext(
        robot_config=robot_config,
        launch_context=launch_context,
        runtime_target=target_state["runtime_target"],
        sim_platform=target_state["sim_platform"],
        use_sim=target_state["use_sim"],
        node_use_sim_time=node_use_sim_time,
        control_mode=active_control_mode,
        with_inference=with_inference,
        scheduler_enabled=scheduler_enabled,
        arguments=arguments,
        auto_start_controllers=target_state["auto_start_controllers"],
        sim_backend_needs_ros2_control=target_state["sim_backend_needs_ros2_control"],
    )

    # 13b. plan: validate the declaration/enable graph and freeze the
    # immutable plan before any provider factory runs.
    plan = registry.plan(context)

    return PreparedLaunch(
        context=context,
        plan=plan,
        robot_config=robot_config,
        launch_context=launch_context,
        runtime_target=target_state["runtime_target"],
        sim_platform=target_state["sim_platform"],
        use_sim=target_state["use_sim"],
        node_use_sim_time=node_use_sim_time,
        control_mode=active_control_mode,
        with_inference=with_inference,
        scheduler_enabled=scheduler_enabled,
        auto_start_controllers=target_state["auto_start_controllers"],
        sim_backend_needs_ros2_control=target_state["sim_backend_needs_ros2_control"],
        runtime_provider_name=target_state["runtime_provider_name"],
        arguments=arguments,
        argument_specs=argument_specs,
        live_binding_required=live_binding_required,
    )
