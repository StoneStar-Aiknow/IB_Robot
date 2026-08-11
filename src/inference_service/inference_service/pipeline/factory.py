"""Unified construction of validated backend and pipeline instances."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from inference_manifest import CompiledDeployment, TensorBinding, ValidatedManifest
from inference_manifest.json_utils import load_json_strict
from inference_service.backends import BACKEND_REGISTRY, BackendLoadError, BackendRegistry, RuntimeContext
from inference_service.backends.rknn.backend import RKNNBackend
from inference_service.codecs import build_execution_plan, create_policy_codec
from inference_service.model_sessions import AscendOmModelSession, HMMModelSession, RKNNModelSession
from inference_service.pi05_schedule import PI05DenoisingSchedule, load_pi05_schedule, uniform_pi05_schedule
from inference_service.pipeline.manager import InferencePipelineManager
from inference_service.pipeline.pi05 import create_pi05_executor, derive_pi05_topology
from inference_service.pipeline.pi05_hmm import (
    PI05HMMFamilyResource,
    build_embedding_stage,
    build_time_prep_stage,
    load_pi05_policy_config,
)
from inference_service.pipeline.processors import create_lerobot_processor_views
from inference_service.pipeline.runtime import InferencePipeline, _PI05SessionHandle, _RawActionResultAdapter
from inference_service.pipeline.smolvla import (
    SmolVLAFamilyResource,
    create_smolvla_executor,
    derive_smolvla_topology,
    load_smolvla_policy_config,
)

_ALLOWED_PI05_OPTIONS = frozenset({"device_id", "acl_config_path", "random_seed", "curvature_log_path"})
_ALLOWED_HMM_OPTIONS = frozenset({"device_id", "random_seed"})
_ALLOWED_RKNN_OPTIONS = frozenset({"target", "core_mask", "random_seed"})


def create_inference_pipeline(
    pipeline_id: str,
    validated_manifest: ValidatedManifest,
    *,
    request_timeout: float | None = None,
    default_task: str | None = None,
    execution_mode: str = "monolithic",
    runtime_options: Mapping[str, object] | None = None,
    priority_scheduling: bool = False,
    registry: BackendRegistry = BACKEND_REGISTRY,
    model_session_factory=None,
    pi05_diagnostic_schedule: PI05DenoisingSchedule | None = None,
    pi05_diagnostic_schedule_source: str | None = None,
) -> InferencePipeline:
    """Create one pipeline exclusively from a validated manifest and registry.

    Compiled Ascend PI0.5 deployments are routed through ``AscendOmModelSession``
    and the shared ``IterativeStage`` instead of the legacy backend-owned loop.
    Compiled HMM PI0.5, HMM SmolVLA, and RKNN SmolVLA deployments are likewise
    routed through ``HMMModelSession``/``RKNNModelSession`` and the shared
    family executors so compiled backends no longer own family loops.
    ``model_session_factory`` optionally overrides session construction for tests.
    """

    context = RuntimeContext(
        validated_manifest,
        runtime_options=runtime_options or {},
        priority_scheduling=priority_scheduling,
    )
    preprocessor = None
    postprocessor = None
    codec = None
    if isinstance(context.deployment, CompiledDeployment):
        preprocessor, postprocessor = create_lerobot_processor_views()
        codec = create_policy_codec(context.policy)
    handle = _build_session_handle(
        context,
        model_session_factory,
        pi05_diagnostic_schedule,
        pi05_diagnostic_schedule_source,
    )
    if handle is not None:
        return InferencePipeline(
            pipeline_id,
            context,
            pi05_handle=handle,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            codec=codec,
            request_timeout=request_timeout,
            default_task=default_task,
            execution_mode=execution_mode,
        )
    backend = registry.create(context)
    try:
        return InferencePipeline(
            pipeline_id,
            context,
            backend,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            codec=codec,
            request_timeout=request_timeout,
            default_task=default_task,
            execution_mode=execution_mode,
        )
    except Exception:
        backend.close()
        raise


def create_pipeline_manager(
    pipeline_id: str,
    validated_manifest: ValidatedManifest,
    *,
    request_timeout: float | None = None,
    default_task: str | None = None,
    execution_mode: str = "monolithic",
    runtime_options: Mapping[str, object] | None = None,
    priority_scheduling: bool = False,
    registry: BackendRegistry = BACKEND_REGISTRY,
    model_session_factory=None,
    pi05_diagnostic_schedule: PI05DenoisingSchedule | None = None,
    pi05_diagnostic_schedule_source: str | None = None,
) -> InferencePipelineManager:
    pipeline = create_inference_pipeline(
        pipeline_id,
        validated_manifest,
        request_timeout=request_timeout,
        default_task=default_task,
        execution_mode=execution_mode,
        runtime_options=runtime_options,
        priority_scheduling=priority_scheduling,
        registry=registry,
        model_session_factory=model_session_factory,
        pi05_diagnostic_schedule=pi05_diagnostic_schedule,
        pi05_diagnostic_schedule_source=pi05_diagnostic_schedule_source,
    )
    manager = InferencePipelineManager((pipeline,))
    manager.start()
    return manager


def _build_session_handle(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule: PI05DenoisingSchedule | None = None,
    diagnostic_schedule_source: str | None = None,
):
    """Construct the session-driven executor handle for compiled family backends.

    Compiled Ascend/HMM PI0.5 and HMM/RKNN SmolVLA deployments route through the
    matching :class:`ModelSession` and the shared family executor instead of a
    backend-owned loop.  ``model_session_factory`` optionally overrides session
    construction for tests.
    """

    deployment = context.deployment
    if not isinstance(deployment, CompiledDeployment):
        return None
    policy_type = context.policy.policy_type
    backend = deployment.backend
    if policy_type == "pi05" and backend == "ascend":
        return _build_ascend_pi05_handle(
            context, model_session_factory, diagnostic_schedule, diagnostic_schedule_source
        )
    if policy_type == "pi05" and backend == "hmm":
        return _build_hmm_pi05_handle(context, model_session_factory)
    if policy_type == "smolvla" and backend == "hmm":
        return _build_hmm_smolvla_handle(context, model_session_factory)
    if policy_type == "smolvla" and backend == "rknn":
        return _build_rknn_smolvla_handle(context, model_session_factory)
    return None


def _build_ascend_pi05_handle(
    context: RuntimeContext,
    model_session_factory=None,
    diagnostic_schedule: PI05DenoisingSchedule | None = None,
    diagnostic_schedule_source: str | None = None,
):
    """Construct the session-driven PI0.5 executor handle for compiled Ascend."""

    deployment = context.deployment
    assert isinstance(deployment, CompiledDeployment)
    options = _validate_pi05_options(context.runtime_options)
    _validate_pi05_policy_config(context)
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    schedule, schedule_source, schedule_override = _resolve_denoising_schedule(
        context,
        action_binding,
        diagnostic_schedule,
        diagnostic_schedule_source,
        require_velocity=options["curvature_log_path"] is not None,
    )
    _validate_schedule_compatibility(schedule, options["curvature_log_path"])
    num_inference_steps = _pi05_num_inference_steps(context)
    session = _create_ascend_session(context, options, model_session_factory)
    velocity_trace: list[np.ndarray] | None = [] if options["curvature_log_path"] is not None else None
    session_executor = create_pi05_executor(
        plan,
        session,
        schedule,
        _RawActionResultAdapter(action_binding.semantic),
        num_inference_steps=num_inference_steps if schedule is None else None,
        initializer=_make_noise_initializer(plan, options["random_seed"]),
        velocity_trace=velocity_trace,
    )
    schedule_metadata = _schedule_metadata(schedule, schedule_source, schedule_override)
    session_context = _build_session_context(context, options)
    return _PI05SessionHandle(
        session_executor,
        session_context,
        action_binding.semantic,
        schedule_metadata,
        schedule,
        session.capabilities,
        options["curvature_log_path"],
        velocity_trace,
    )


def _build_hmm_pi05_handle(
    context: RuntimeContext,
    model_session_factory=None,
):
    """Construct the session-driven PI0.5 executor handle for compiled HMM.

    The modular HMM PI0.5 topology declares a synthetic ``embedding`` host role
    and a ``time_mlp`` compiled module.  Embedding construction and sinusoidal
    timestep embedding run as executor-owned host stages while TCIM modules
    execute through ``HMMModelSession`` and the shared ``IterativeStage``.
    """

    deployment = context.deployment
    assert isinstance(deployment, CompiledDeployment)
    options = _validate_hmm_options(context.runtime_options)
    policy_config = load_pi05_policy_config(context)
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    num_inference_steps = _hmm_pi05_num_inference_steps(policy_config)
    schedule = uniform_pi05_schedule(num_inference_steps)
    session = _create_hmm_session(context, options, model_session_factory)
    family_resource = PI05HMMFamilyResource(deployment, policy_config)
    embedding_stage = build_embedding_stage(deployment, family_resource, policy_config)
    time_prep_stage = build_time_prep_stage(deployment, policy_config)
    session_executor = create_pi05_executor(
        plan,
        session,
        schedule,
        _RawActionResultAdapter(action_binding.semantic),
        initializer=_make_noise_initializer(plan, options["random_seed"]),
        embedding_stage=embedding_stage,
        time_prep_stage=time_prep_stage,
        extra_components=(family_resource,),
    )
    session_context = _build_hmm_session_context(context, options)
    return _PI05SessionHandle(
        session_executor,
        session_context,
        action_binding.semantic,
        _schedule_metadata(schedule, "uniform", False),
        schedule,
        session.capabilities,
        None,
        None,
    )


def _validate_pi05_options(options: Mapping[str, object]) -> dict[str, object]:
    unknown = sorted(set(options) - _ALLOWED_PI05_OPTIONS)
    if unknown:
        raise BackendLoadError(f"unknown Ascend PI0.5 options: {unknown}", code="invalid_runtime_options")
    device_id = options.get("device_id", 0)
    if type(device_id) is not int or device_id < 0:
        raise BackendLoadError("Ascend device_id must be a non-negative integer", code="invalid_runtime_options")
    acl_config_path = options.get("acl_config_path")
    if acl_config_path is not None and (type(acl_config_path) is not str or not acl_config_path.strip()):
        raise BackendLoadError("Ascend acl_config_path must be a non-empty string", code="invalid_runtime_options")
    random_seed = options.get("random_seed")
    if random_seed is not None and type(random_seed) is not int:
        raise BackendLoadError("Ascend random_seed must be an integer or null", code="invalid_runtime_options")
    curvature_log_path = options.get("curvature_log_path")
    if curvature_log_path is not None and (type(curvature_log_path) is not str or not curvature_log_path.strip()):
        raise BackendLoadError("Ascend curvature_log_path must be a non-empty string", code="invalid_runtime_options")
    if curvature_log_path is not None:
        from pathlib import Path

        path = Path(str(curvature_log_path)).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.open("a", encoding="utf-8").close()
    return {
        "device_id": device_id,
        "acl_config_path": acl_config_path,
        "random_seed": random_seed,
        "curvature_log_path": curvature_log_path,
    }


def _session_action_binding(plan) -> TensorBinding:
    action_role = next(role for role in plan.roles if any(b.semantic == "action" for b in role.bindings.outputs))
    matches = [b for b in action_role.bindings.outputs if b.semantic == "action"]
    if len(matches) != 1:
        raise BackendLoadError(
            "compiled deployment requires exactly one action output binding", code="invalid_action_binding"
        )
    return matches[0]


def _resolve_denoising_schedule(
    context: RuntimeContext,
    action_binding: TensorBinding,
    diagnostic_schedule: PI05DenoisingSchedule | None,
    diagnostic_schedule_source: str | None,
    require_velocity: bool = False,
):
    deployment = context.deployment
    assert isinstance(deployment, CompiledDeployment)
    runtime_name = action_binding.runtime_name or ""
    output_name = next(
        (part for part in reversed(runtime_name.split(":")) if part in {"action", "velocity", "v_t"}),
        runtime_name,
    )
    velocity_mode = output_name in {"velocity", "v_t"}
    if (diagnostic_schedule is not None or require_velocity) and not velocity_mode:
        raise BackendLoadError(
            "Ascend PI0.5 schedule diagnostics require a velocity/v_t Action Expert runtime output",
            code="invalid_runtime_options",
        )
    if diagnostic_schedule is not None:
        return diagnostic_schedule, diagnostic_schedule_source or "diagnostic", True
    artifact = deployment.artifacts.get("denoising_schedule")
    if artifact is None:
        if velocity_mode:
            raise BackendLoadError(
                "Ascend PI0.5 velocity output requires a denoising_schedule artifact",
                code="missing_denoising_schedule",
            )
        return None, None, False
    if artifact.format != "json":
        raise BackendLoadError(
            "Ascend PI0.5 denoising_schedule artifact must use format 'json'", code="invalid_denoising_schedule"
        )
    if not velocity_mode:
        raise BackendLoadError(
            "Ascend PI0.5 denoising_schedule requires a velocity/v_t Action Expert runtime output",
            code="invalid_denoising_schedule",
        )
    try:
        schedule = load_pi05_schedule(context.resolved_artifacts["denoising_schedule"])
    except Exception as exc:
        raise BackendLoadError(
            f"Unable to load PI0.5 denoising schedule {artifact.path}: {exc}", code="invalid_denoising_schedule"
        ) from exc
    return schedule, artifact.path, False


def _validate_pi05_policy_config(context: RuntimeContext) -> None:
    config = _load_policy_config(context)
    for key in ("chunk_size", "max_action_dim", "num_inference_steps"):
        value = config.get(key)
        if type(value) is not int or value < 1:
            raise BackendLoadError(
                f"Ascend PI0.5 requires positive integer {key!r} in LeRobot config",
                code="invalid_policy_config",
            )


def _validate_schedule_compatibility(schedule: PI05DenoisingSchedule | None, curvature_log_path) -> None:
    if curvature_log_path is not None and schedule is None:
        raise BackendLoadError(
            "Ascend PI0.5 curvature log requires a velocity/v_t Action Expert runtime output and schedule",
            code="invalid_runtime_options",
        )


def _pi05_num_inference_steps(context: RuntimeContext) -> int:
    config = _load_policy_config(context)
    value = config.get("num_inference_steps")
    if type(value) is not int or value < 1:
        raise BackendLoadError(
            "Ascend PI0.5 requires positive integer 'num_inference_steps' in LeRobot config",
            code="invalid_policy_config",
        )
    return value


def _load_policy_config(context: RuntimeContext) -> dict[str, object]:
    try:
        value = load_json_strict(context.validated_manifest.bundle_root / "config.json")
    except Exception as exc:
        raise BackendLoadError(f"Unable to read LeRobot config: {exc}", code="invalid_policy_config") from exc
    if not isinstance(value, dict):
        raise BackendLoadError("LeRobot config must be an object", code="invalid_policy_config")
    return value


def _create_ascend_session(
    context: RuntimeContext,
    options: Mapping[str, object],
    model_session_factory=None,
) -> AscendOmModelSession:
    if model_session_factory is not None:
        return model_session_factory(context, options)
    return AscendOmModelSession(
        device_id=int(options["device_id"]),
        priority_scheduling=context.priority_scheduling,
    )


def _build_session_context(context: RuntimeContext, options: Mapping[str, object]) -> RuntimeContext:
    session_options: dict[str, object] = {"device_id": options["device_id"]}
    if options["acl_config_path"] is not None:
        session_options["acl_config_path"] = options["acl_config_path"]
    return RuntimeContext(
        context.validated_manifest,
        runtime_options=session_options,
        priority_scheduling=context.priority_scheduling,
    )


def _validate_hmm_options(options: Mapping[str, object]) -> dict[str, object]:
    unknown = sorted(set(options) - _ALLOWED_HMM_OPTIONS)
    if unknown:
        raise BackendLoadError(f"unknown HMM options: {unknown}", code="invalid_runtime_options")
    device_id = options.get("device_id", 0)
    if type(device_id) is not int or device_id < 0:
        raise BackendLoadError("HMM device_id must be a non-negative integer", code="invalid_runtime_options")
    random_seed = options.get("random_seed")
    if random_seed is not None and type(random_seed) is not int:
        raise BackendLoadError("HMM random_seed must be an integer or null", code="invalid_runtime_options")
    return {"device_id": device_id, "random_seed": random_seed}


def _hmm_pi05_num_inference_steps(policy_config: Mapping[str, object]) -> int:
    value = policy_config.get("num_inference_steps")
    if type(value) is not int or value < 1:
        raise BackendLoadError(
            "HMM PI0.5 requires positive integer 'num_inference_steps' in LeRobot config",
            code="invalid_policy_config",
        )
    return value


def _create_hmm_session(
    context: RuntimeContext,
    options: Mapping[str, object],
    model_session_factory=None,
) -> HMMModelSession:
    if model_session_factory is not None:
        return model_session_factory(context, options)
    return HMMModelSession(device_id=int(options["device_id"]))


def _build_hmm_session_context(context: RuntimeContext, options: Mapping[str, object]) -> RuntimeContext:
    session_options: dict[str, object] = {"device_id": options["device_id"]}
    return RuntimeContext(
        context.validated_manifest,
        runtime_options=session_options,
        priority_scheduling=context.priority_scheduling,
    )


def _make_noise_initializer(plan, random_seed):
    topology = derive_pi05_topology(plan)
    noise_binding = next(
        b
        for role in topology.loop_roles
        for b in plan.role(role).bindings.inputs
        if b.semantic == topology.state_semantic
    )
    rng = np.random.default_rng(random_seed)

    def initializer(values: Mapping[str, object]) -> np.ndarray:
        return np.ascontiguousarray(rng.standard_normal(noise_binding.shape).astype(np.dtype(noise_binding.dtype)))

    return initializer


def _build_hmm_smolvla_handle(
    context: RuntimeContext,
    model_session_factory=None,
):
    """Construct the session-driven SmolVLA executor handle for compiled HMM.

    SmolVLA HMM declares device-resident KV links between ``prefill`` and
    ``action``.  The family executor (``create_smolvla_executor``) runs vision,
    a synthetic host ``embedding`` role, ``prefill``, and the iterative
    ``action`` body through :class:`HMMModelSession`, which keeps the declared
    device links device-resident.  Embedding weights load once into the
    executor-owned :class:`SmolVLAFamilyResource`.
    """

    deployment = context.deployment
    assert isinstance(deployment, CompiledDeployment)
    options = _validate_hmm_options(context.runtime_options)
    policy_config = load_smolvla_policy_config(context)
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    num_inference_steps = _smolvla_num_inference_steps(policy_config)
    session = _create_hmm_session(context, options, model_session_factory)
    resource = SmolVLAFamilyResource(deployment, policy_config)
    session_executor = create_smolvla_executor(
        plan,
        session,
        resource,
        _RawActionResultAdapter(action_binding.semantic),
        num_inference_steps=num_inference_steps,
        initializer=_make_smolvla_noise_initializer(plan, options["random_seed"]),
    )
    return _PI05SessionHandle(
        session_executor,
        _build_hmm_session_context(context, options),
        action_binding.semantic,
        None,
        None,
        session.capabilities,
        None,
        None,
    )


def _build_rknn_smolvla_handle(
    context: RuntimeContext,
    model_session_factory=None,
):
    """Construct the session-driven SmolVLA executor handle for compiled RKNN.

    SmolVLA RKNN declares no device links; prefix/KV tensors are host-visible
    bindings threaded through the executor ``ExecutionFrame``.  Runtime-specific
    tensor conversion stays below :class:`RKNNModelSession`, which exposes only
    semantic role invocation to the shared family executor.
    """

    deployment = context.deployment
    assert isinstance(deployment, CompiledDeployment)
    options = _validate_rknn_options(context.runtime_options)
    policy_config = load_smolvla_policy_config(context)
    plan = build_execution_plan(deployment.execution, deployment.bindings, deployment.device_links)
    action_binding = _session_action_binding(plan)
    num_inference_steps = _smolvla_num_inference_steps(policy_config)
    session = _create_rknn_session(context, options, model_session_factory)
    resource = SmolVLAFamilyResource(deployment, policy_config)
    session_executor = create_smolvla_executor(
        plan,
        session,
        resource,
        _RawActionResultAdapter(action_binding.semantic),
        num_inference_steps=num_inference_steps,
        initializer=_make_smolvla_noise_initializer(plan, options["random_seed"]),
    )
    return _PI05SessionHandle(
        session_executor,
        _build_rknn_session_context(context, options),
        action_binding.semantic,
        None,
        None,
        session.capabilities,
        None,
        None,
    )


def _smolvla_num_inference_steps(policy_config: Mapping[str, object]) -> int:
    value = policy_config.get("num_steps")
    if type(value) is not int or value < 1:
        raise BackendLoadError(
            "SmolVLA requires positive integer 'num_steps' in LeRobot config",
            code="invalid_policy_config",
        )
    return value


def _make_smolvla_noise_initializer(plan, random_seed):
    topology = derive_smolvla_topology(plan)
    noise_binding = next(
        binding for binding in plan.role("action").bindings.inputs if binding.semantic == topology.state_semantic
    )
    rng = np.random.default_rng(random_seed)

    def initializer(values: Mapping[str, object]) -> np.ndarray:
        return np.ascontiguousarray(rng.standard_normal(noise_binding.shape).astype(np.float32))

    return initializer


def _validate_rknn_options(options: Mapping[str, object]) -> dict[str, object]:
    unknown = sorted(set(options) - _ALLOWED_RKNN_OPTIONS)
    if unknown:
        raise BackendLoadError(f"unknown RKNN options: {unknown}", code="invalid_runtime_options")
    return RKNNBackend._validate_runtime_options(options)


def _create_rknn_session(
    context: RuntimeContext,
    options: Mapping[str, object],
    model_session_factory=None,
) -> RKNNModelSession:
    if model_session_factory is not None:
        return model_session_factory(context, options)
    return RKNNModelSession()


def _build_rknn_session_context(context: RuntimeContext, options: Mapping[str, object]) -> RuntimeContext:
    session_options: dict[str, object] = {"target": options["target"], "core_mask": options["core_mask"]}
    return RuntimeContext(
        context.validated_manifest,
        runtime_options=session_options,
        priority_scheduling=context.priority_scheduling,
    )


def _schedule_metadata(schedule, source, override):
    if schedule is None:
        return None
    metadata = {"name": schedule.name, "step_count": schedule.step_count, "source": source}
    if override:
        metadata["override"] = True
    return metadata
