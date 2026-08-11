"""Policy compatibility facade over the unified GenericModelPipeline runtime.

``InferencePipeline`` preserves the existing policy request/result contract and
stateful error semantics while delegating lifecycle, admission, deadline, and
execution to :class:`GenericModelPipeline`.  All control-plane state lives in
``PipelineRuntimeCore``; this module only owns policy-specific preprocessing,
codec binding, action validation, and structured error mapping.
"""

from __future__ import annotations

import copy
import json
import math
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType

import numpy as np

from inference_manifest import CompiledDeployment
from inference_service.backends import (
    BackendAdmissionError,
    BackendHealth,
    BackendResult,
    BackendState,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.codecs import CodecRequest, CodecResult, ExecutionPlan, PolicyCodec, build_execution_plan
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.pi05_schedule import PI05DenoisingSchedule
from inference_service.pipeline.errors import (
    PipelineConfigurationError,
    PipelineNotReadyError,
    PipelineTimeoutError,
    PipelineValidationError,
)
from inference_service.pipeline.executor import SequentialModelExecutor
from inference_service.pipeline.runtime_core import (
    ExecutionError,
    GenericModelPipeline,
    ModelExecutor,
    StageFrame,
)
from inference_service.pipeline.types import PipelineDiagnostics, PipelineResult
from inference_service.pipeline.validation import validate_action_output

Processor = Callable[[Mapping[str, object]], Mapping[str, object]]
Postprocessor = Callable[[object], object]


def _identity_preprocessor(inputs: Mapping[str, object]) -> Mapping[str, object]:
    return inputs


def _identity_postprocessor(action: object) -> object:
    return action


def _snapshot_action(action: object) -> object:
    detached = getattr(action, "detach", None)
    candidate = detached() if callable(detached) else action
    clone = getattr(candidate, "clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(candidate)


def _chunk_size_from_action(action: np.ndarray) -> int:
    shape = getattr(action, "shape", ())
    if len(shape) < 2 or shape[-2] < 1:
        raise PipelineValidationError(
            f"model session action output has invalid shape {shape}",
            pipeline_id="policy",
            code="invalid_action_shape",
        )
    return int(shape[-2])


class _PolicyRequest:
    """Carry policy execution parameters through the unified pipeline boundary."""

    __slots__ = ("_request", "control_inputs", "capture_raw_action")

    def __init__(
        self,
        request: InferenceRequest,
        *,
        control_inputs: Mapping[str, object] | None,
        capture_raw_action: bool,
    ) -> None:
        self._request = request
        self.control_inputs = control_inputs
        self.capture_raw_action = capture_raw_action

    @property
    def request_id(self) -> str:
        return self._request.request_id

    @property
    def inner(self) -> InferenceRequest:
        return self._request

    @property
    def inputs(self) -> Mapping[str, object]:
        return {"_total_start": time.perf_counter()}


class _PolicyPreprocessStage:
    """Policy preprocess: prompt selection, processor, control-input merge, deadline gate."""

    def __init__(self, facade: InferencePipeline) -> None:
        self._facade = facade

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        policy_request = frame.request
        request = policy_request.inner
        selected_prompt = request.prompt if request.prompt is not None else self._facade._default_task
        processor_inputs = dict(request.inputs)
        if selected_prompt is not None:
            processor_inputs["task"] = selected_prompt

        frame.control.raise_if_canceled("preprocess")

        preprocess_start = time.perf_counter()
        canonical_inputs = self._facade._preprocessor(processor_inputs)
        preprocess_latency_ms = (time.perf_counter() - preprocess_start) * 1000.0
        if not isinstance(canonical_inputs, Mapping):
            raise PipelineValidationError(
                f"pipeline {self._facade.pipeline_id!r} preprocessor must return a mapping",
                pipeline_id=self._facade.pipeline_id,
                details={"returned_type": type(canonical_inputs).__name__},
            )
        canonical_inputs = dict(canonical_inputs)
        control_inputs = policy_request.control_inputs
        if control_inputs:
            collisions = sorted(set(canonical_inputs) & set(control_inputs))
            if collisions:
                raise PipelineValidationError(
                    f"pipeline {self._facade.pipeline_id!r} control inputs conflict with preprocessor "
                    f"outputs: {collisions}",
                    pipeline_id=self._facade.pipeline_id,
                    details={"conflicting_inputs": tuple(collisions)},
                )
            canonical_inputs.update(control_inputs)
        self._facade._raise_if_expired(deadline, phase="preprocess", backend_completed=False)

        frame.values["_canonical_inputs"] = canonical_inputs
        frame.values["_selected_prompt"] = selected_prompt
        frame.values["_preprocess_latency_ms"] = preprocess_latency_ms


class _RawActionResultAdapter:
    """Expose the iterative final action semantic as the executor result."""

    def __init__(self, action_semantic: str = "action") -> None:
        self._action_semantic = action_semantic

    def adapt(self, frame: StageFrame) -> object:
        return frame.values[self._action_semantic]

    def adapt_error(self, error: ExecutionError) -> object:
        if error.cause is not None:
            raise error.cause
        raise RuntimeError(error.message)


class _PolicyModelRequestStage:
    """Bind canonical policy values to the model executor request contract."""

    def __init__(
        self,
        facade: InferencePipeline,
    ) -> None:
        self._facade = facade

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        frame.control.raise_if_canceled("model.request")
        canonical_inputs = frame.values["_canonical_inputs"]
        if isinstance(self._facade._context.deployment, CompiledDeployment):
            role_inputs = self._facade._encode_role_inputs(canonical_inputs)
            semantic_values: dict[str, object] = {}
            for bound_inputs in role_inputs.values():
                for tensor in bound_inputs.tensors:
                    semantic_values[tensor.semantic] = tensor.value
        else:
            semantic_values = dict(canonical_inputs)
        model_request = NamedTensorRequest(
            frame.request.inner.request_id,
            MappingProxyType(semantic_values),
            deadline=deadline,
            metadata=frame.request.inner.metadata,
            priority=frame.request.inner.priority,
        )
        frame.values["_model_request"] = model_request
        frame.values["_model_inputs"] = semantic_values
        frame.values.update(semantic_values)
        frame.values["_backend_started"] = True


class _PolicyModelResultStage:
    """Wrap the flattened model stages' action in the policy backend contract."""

    def __init__(
        self,
        facade: InferencePipeline,
        action_semantic: str,
        denoising_schedule_metadata: Mapping[str, object] | None,
        metadata_provider=None,
    ) -> None:
        self._facade = facade
        self._action_semantic = action_semantic
        self._denoising_schedule_metadata = denoising_schedule_metadata
        self._metadata_provider = metadata_provider

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        raw_action = frame.values[self._action_semantic]
        if not hasattr(raw_action, "shape"):
            raise PipelineValidationError(
                f"pipeline {self._facade.pipeline_id!r} model session returned "
                f"{type(raw_action).__name__}, expected a tensor-like action",
                pipeline_id=self._facade.pipeline_id,
            )
        self._facade._raise_if_expired(deadline, phase="backend", backend_completed=True)
        self._facade._ensure_session_ready_after_call()
        metadata = {
            "request_id": frame.request.inner.request_id,
            "deployment_name": self._facade._context.deployment_name,
            "deployment_fingerprint": self._facade._context.deployment_fingerprint,
        }
        if callable(self._metadata_provider):
            metadata.update(self._metadata_provider(frame.request.inner.request_id))
        chunk_size = 1 if metadata.get("action_method") == "select_action" else _chunk_size_from_action(raw_action)
        priority_mapping = self._facade.capabilities.priority_mapping
        if priority_mapping is not None:
            metadata["hardware_priority"] = priority_mapping.map_generic(frame.request.inner.priority)
        if self._denoising_schedule_metadata is not None:
            metadata["denoising_schedule"] = self._denoising_schedule_metadata
        frame.values["_backend_result"] = BackendResult(
            action=raw_action,
            actual_chunk_size=chunk_size,
            backend_latency_ms=0.0,
            metadata=metadata,
        )


class _PolicyCompletionStage:
    def __init__(self, operation: Callable[[], None]) -> None:
        self._operation = operation

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        del frame, deadline
        self._operation()


class _PolicyDecodeStage:
    """Decode/validate/raw-capture stage: action decode, validation, optional raw snapshot."""

    def __init__(self, facade: InferencePipeline) -> None:
        self._facade = facade

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        del deadline
        frame.control.raise_if_canceled("decode")
        backend_result = frame.values["_backend_result"]

        semantic_action = self._facade._decode_backend_action(backend_result)
        validate_action_output(
            semantic_action,
            actual_chunk_size=backend_result.actual_chunk_size,
            action_dimension=self._facade._action_dimension,
            pipeline_id=self._facade.pipeline_id,
            phase="backend",
        )
        raw_action = _snapshot_action(semantic_action) if frame.request.capture_raw_action else None

        frame.values["_semantic_action"] = semantic_action
        frame.values["_raw_action"] = raw_action


class _PolicyPostprocessStage:
    """Postprocess/validate stage: postprocessor, validation, postprocess deadline gate."""

    def __init__(self, facade: InferencePipeline) -> None:
        self._facade = facade

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        frame.control.raise_if_canceled("postprocess")
        backend_result = frame.values["_backend_result"]

        postprocess_start = time.perf_counter()
        action = self._facade._postprocessor(frame.values["_semantic_action"])
        postprocess_latency_ms = (time.perf_counter() - postprocess_start) * 1000.0
        validate_action_output(
            action,
            actual_chunk_size=backend_result.actual_chunk_size,
            action_dimension=self._facade._action_dimension,
            pipeline_id=self._facade.pipeline_id,
            phase="postprocessor",
        )
        self._facade._raise_if_expired(deadline, phase="postprocess", backend_completed=True)

        frame.values["_action"] = action
        frame.values["_postprocess_latency_ms"] = postprocess_latency_ms


class _PolicyResultAdapter:
    """Result adapter: assemble the policy PipelineResult from the completed stage frame."""

    def __init__(self, facade: InferencePipeline) -> None:
        self._facade = facade

    def adapt(self, frame: StageFrame) -> PipelineResult:
        backend_result = frame.values["_backend_result"]
        total_latency_ms = (time.perf_counter() - frame.values["_total_start"]) * 1000.0
        preprocess_latency_ms = frame.values["_preprocess_latency_ms"]
        postprocess_latency_ms = frame.values["_postprocess_latency_ms"]
        latency_metadata = MappingProxyType(
            {
                "total": total_latency_ms,
                "preprocess": preprocess_latency_ms,
                "backend": backend_result.backend_latency_ms,
                "postprocess": postprocess_latency_ms,
            }
        )
        manifest = self._facade._context.validated_manifest.manifest
        deployment = self._facade._context.deployment
        return PipelineResult(
            action=frame.values["_action"],
            actual_chunk_size=backend_result.actual_chunk_size,
            pipeline_id=self._facade.pipeline_id,
            bundle=manifest.bundle.name,
            bundle_uuid=manifest.bundle.uuid,
            bundle_revision=manifest.bundle.revision,
            deployment=self._facade._context.deployment_name,
            deployment_uuid=deployment.uuid,
            deployment_revision=deployment.revision,
            deployment_fingerprint=self._facade._context.deployment_fingerprint,
            backend=self._facade._backend_name(),
            state=self._facade._pipeline.state,
            total_latency_ms=total_latency_ms,
            preprocess_latency_ms=preprocess_latency_ms,
            backend_latency_ms=backend_result.backend_latency_ms,
            postprocess_latency_ms=postprocess_latency_ms,
            raw_action=frame.values["_raw_action"],
            metadata={
                **backend_result.metadata,
                "pipeline_id": self._facade.pipeline_id,
                "bundle": manifest.bundle.name,
                "bundle_uuid": manifest.bundle.uuid,
                "bundle_revision": manifest.bundle.revision,
                "deployment": self._facade._context.deployment_name,
                "deployment_uuid": deployment.uuid,
                "deployment_revision": deployment.revision,
                "deployment_fingerprint": self._facade._context.deployment_fingerprint,
                "backend": self._facade._backend_name(),
                "state": self._facade._pipeline.state.value,
                "latency_ms": latency_metadata,
            },
        )

    def adapt_error(self, error: ExecutionError) -> PipelineResult:
        if error.cause is not None:
            raise error.cause
        raise RuntimeError(error.message)


class _PolicySessionHandle:
    """Configuration bundle the factory passes so the facade can build the executor."""

    def __init__(
        self,
        model_executor,
        session_context: RuntimeContext,
        action_semantic: str,
        schedule_metadata: Mapping[str, object] | None,
        schedule: PI05DenoisingSchedule | None,
        capabilities,
        curvature_log_path: str | None,
        velocity_trace: list | None,
        *,
        preprocessor=None,
        postprocessor=None,
        metadata_provider=None,
    ) -> None:
        self.model_executor = model_executor
        self.session_context = session_context
        self.action_semantic = action_semantic
        self.schedule_metadata = schedule_metadata
        self.schedule = schedule
        self._capability_source = capabilities
        self.curvature_log_path = curvature_log_path
        self.velocity_trace = velocity_trace
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.metadata_provider = metadata_provider

    @property
    def capabilities(self):
        return getattr(self._capability_source, "capabilities", self._capability_source)


def _curvature_scores(velocities: list, eps: float = 1e-6) -> list[float]:
    if not velocities:
        return []
    if len(velocities) == 1:
        return [0.0]
    scores = []
    for current, following in pairwise(velocities):
        current_flat = np.asarray(current).reshape(np.asarray(current).shape[0], -1).astype(np.float32, copy=False)
        following_flat = (
            np.asarray(following).reshape(np.asarray(following).shape[0], -1).astype(np.float32, copy=False)
        )
        difference = np.linalg.norm(following_flat - current_flat, axis=1)
        magnitude = np.linalg.norm(current_flat, axis=1) + eps
        scores.append(float(np.mean(difference / magnitude)))
    scores.append(scores[-1])
    return scores


class InferencePipeline:
    """Policy compatibility facade over :class:`GenericModelPipeline`.

    Preserves ``InferenceRequest``/``PipelineResult``, ``cancel()``, structured
    error mapping, prompt/control inputs, raw-action capture, codec/action
    validation, and stateful fail-closed semantics without owning any
    independent lifecycle, admission, or deadline state.
    """

    def __init__(
        self,
        pipeline_id: str,
        runtime_context: RuntimeContext,
        *,
        executor: ModelExecutor | None = None,
        session_handle: _PolicySessionHandle | None = None,
        preprocessor: Processor | None = None,
        postprocessor: Postprocessor | None = None,
        codec: PolicyCodec | None = None,
        request_timeout: float | None = None,
        default_task: str | None = None,
        execution_mode: str = "monolithic",
    ) -> None:
        if not pipeline_id:
            raise PipelineConfigurationError("pipeline_id must be non-empty")
        if execution_mode != "monolithic":
            raise PipelineConfigurationError(
                f"pipeline {pipeline_id!r} execution mode {execution_mode!r} is not implemented",
                pipeline_id=pipeline_id,
                code="unsupported_execution_mode",
                details={"execution_mode": execution_mode},
            )
        if request_timeout is not None and (not math.isfinite(request_timeout) or request_timeout <= 0):
            raise PipelineConfigurationError(
                f"pipeline {pipeline_id!r} request_timeout must be finite and positive",
                pipeline_id=pipeline_id,
                details={"request_timeout": request_timeout},
            )
        construction_sources = sum(1 for source in (executor, session_handle) if source is not None)
        if construction_sources != 1:
            raise PipelineConfigurationError(
                f"pipeline {pipeline_id!r} requires exactly one of executor or session_handle",
                pipeline_id=pipeline_id,
                code="invalid_pipeline_construction",
            )

        self._pipeline_id = pipeline_id
        self._context = runtime_context
        self._backend = None
        self._executor = executor
        self._session_handle = session_handle
        self._pi05_handle = (
            session_handle
            if runtime_context.policy.policy_type == "pi05"
            and isinstance(runtime_context.deployment, CompiledDeployment)
            else None
        )
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._owns_preprocessor = preprocessor is not None
        self._owns_postprocessor = postprocessor is not None
        self._codec = codec
        self._request_timeout = request_timeout
        self._default_task = default_task
        self._execution_mode = execution_mode
        self._execution_plan: ExecutionPlan | None = None
        self._action_output_role: str | None = None
        self._policy_failure: BackendHealth | None = None
        self._prepare_execution()
        self._bind_processors()

        if session_handle is not None:
            resolved_executor: ModelExecutor = self._build_session_executor(session_handle)
        elif executor is not None:
            resolved_executor = executor
        else:
            resolved_executor = executor
        self._pipeline = GenericModelPipeline(
            pipeline_id,
            runtime_context,
            resolved_executor,
            request_timeout=request_timeout,
            supports_cancellation=self.capabilities.supports_cancellation,
        )

    def _build_session_executor(self, handle: _PolicySessionHandle) -> SequentialModelExecutor:
        model_executor = handle.model_executor
        if not isinstance(model_executor, SequentialModelExecutor):
            raise PipelineConfigurationError(
                f"pipeline {self.pipeline_id!r} session factory must return SequentialModelExecutor",
                pipeline_id=self.pipeline_id,
                code="invalid_session_executor",
            )
        components = list(model_executor.components)
        if self._owns_preprocessor and self._preprocessor is not None:
            components.append(self._preprocessor)
        if self._owns_postprocessor and self._postprocessor is not None:
            components.append(self._postprocessor)
        component_contexts = dict(model_executor.component_contexts)
        component_contexts.update({id(component): handle.session_context for component in model_executor.components})
        return SequentialModelExecutor(
            (
                _PolicyPreprocessStage(self),
                _PolicyModelRequestStage(self),
                *model_executor.stages,
                _PolicyModelResultStage(
                    self,
                    handle.action_semantic,
                    handle.schedule_metadata,
                    handle.metadata_provider,
                ),
                _PolicyDecodeStage(self),
                _PolicyPostprocessStage(self),
                _PolicyCompletionStage(lambda: self._write_curvature_log(handle)),
            ),
            _PolicyResultAdapter(self),
            components=components,
            execution_plan=model_executor.execution_plan,
            component_contexts=component_contexts,
            error_handler=self._record_policy_failure,
            health_override=lambda: self._policy_failure,
            defer_session_execution=True,
        )

    def _write_curvature_log(self, handle: _PolicySessionHandle) -> None:
        if handle.curvature_log_path is None or not handle.velocity_trace:
            return
        record: dict[str, object] = {"curvature_scores": _curvature_scores(handle.velocity_trace)}
        if handle.schedule is not None:
            record["schedule"] = handle.schedule.to_dict()
        path = Path(str(handle.curvature_log_path)).expanduser().resolve()
        try:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        except (OSError, ValueError) as exc:
            raise PipelineValidationError(
                f"pipeline {self.pipeline_id!r} unable to write PI0.5 curvature log {path}: {exc}",
                pipeline_id=self.pipeline_id,
                code="curvature_log_failed",
            ) from exc

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def state(self):
        return self._pipeline.state

    @property
    def runtime_context(self) -> RuntimeContext:
        return self._context

    @property
    def capabilities(self):
        if self._session_handle is not None:
            return self._session_handle.capabilities
        if self._executor is not None and hasattr(self._executor, "capabilities"):
            return self._executor.capabilities
        raise PipelineConfigurationError(
            f"pipeline {self.pipeline_id!r} has no capability source", pipeline_id=self.pipeline_id
        )

    def load(self) -> None:
        self._pipeline.load()

    def infer(
        self,
        request: InferenceRequest,
        *,
        control_inputs: Mapping[str, object] | None = None,
        capture_raw_action: bool = False,
    ) -> PipelineResult:
        policy_request = _PolicyRequest(
            request,
            control_inputs=control_inputs,
            capture_raw_action=capture_raw_action,
        )
        result = self._pipeline.execute(policy_request, deadline=request.deadline)
        if not isinstance(result, PipelineResult):
            raise PipelineValidationError(
                f"pipeline {self.pipeline_id!r} executor returned {type(result).__name__}, expected PipelineResult",
                pipeline_id=self.pipeline_id,
            )
        return result

    def reset(self, deadline: datetime | None = None) -> None:
        self._pipeline.reset(deadline)

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        self._pipeline.cancel(request_id, deadline)

    def diagnostics(self) -> PipelineDiagnostics:
        runtime_diag = self._pipeline.diagnostics()
        identity = runtime_diag.deployment
        return PipelineDiagnostics(
            pipeline_id=runtime_diag.pipeline_id,
            bundle=identity.bundle,
            bundle_uuid=identity.bundle_uuid,
            bundle_revision=identity.bundle_revision,
            deployment=identity.deployment,
            deployment_uuid=identity.deployment_uuid,
            deployment_revision=identity.deployment_revision,
            deployment_fingerprint=identity.deployment_fingerprint,
            backend=identity.backend,
            state=runtime_diag.state,
            backend_health=runtime_diag.executor_health,
            active_requests=runtime_diag.active_requests,
            request_timeout=runtime_diag.request_timeout,
            default_task_configured=self._default_task is not None,
        )

    def health(self) -> PipelineDiagnostics:
        return self.diagnostics()

    def close(self) -> None:
        self._pipeline.close()

    # ------------------------------------------------------------------
    # Policy helpers (preserved from the original implementation)
    # ------------------------------------------------------------------

    @property
    def _action_dimension(self) -> int:
        return self._context.policy.output_features["action"].shape[-1]

    def _backend_name(self) -> str:
        return self._context.deployment.backend

    def _prepare_execution(self) -> None:
        deployment = self._context.deployment
        if not isinstance(deployment, CompiledDeployment):
            self._execution_plan = None
            self._action_output_role = None
            return
        if self._codec is None:
            raise PipelineConfigurationError(
                f"compiled pipeline {self.pipeline_id!r} requires a policy codec",
                pipeline_id=self.pipeline_id,
                code="codec_required",
            )
        self._execution_plan = build_execution_plan(
            deployment.execution,
            deployment.bindings,
            deployment.device_links,
        )
        action_roles = [
            role
            for role in deployment.execution
            if any(binding.semantic == "action" for binding in deployment.bindings[role].outputs)
        ]
        if len(action_roles) != 1:
            raise PipelineConfigurationError(
                f"compiled pipeline {self.pipeline_id!r} must declare exactly one action-producing role",
                pipeline_id=self.pipeline_id,
                code="invalid_action_role",
                details={"action_roles": tuple(action_roles)},
            )
        self._action_output_role = action_roles[0]

    def _encode_role_inputs(self, canonical_inputs: Mapping[str, object]) -> Mapping[str, object]:
        codec, execution_plan = self._require_compiled_codec()
        encode_execution = getattr(codec, "encode_execution", None)
        if callable(encode_execution):
            return encode_execution(CodecRequest(canonical_inputs), execution_plan)
        if len(execution_plan.roles) == 1:
            role = execution_plan.roles[0]
            return {role.name: codec.encode_inputs(CodecRequest(canonical_inputs), role.bindings)}
        raise PipelineConfigurationError(
            f"compiled pipeline {self.pipeline_id!r} requires an execution-aware codec for multiple roles",
            pipeline_id=self.pipeline_id,
            code="execution_codec_required",
            details={"roles": execution_plan.role_names},
        )

    def _ensure_session_ready_after_call(self) -> None:
        if self._session_handle is None:
            return
        health = self._session_handle.model_executor.health()
        if not health.ready:
            raise PipelineNotReadyError(
                f"pipeline {self.pipeline_id!r} session left READY during inference",
                pipeline_id=self.pipeline_id,
                state=health.state.value,
            )

    def _decode_backend_action(self, result: BackendResult) -> object:
        deployment = self._context.deployment
        if not isinstance(deployment, CompiledDeployment):
            return result.action

        codec, execution_plan = self._require_compiled_codec()
        decode_execution = getattr(codec, "decode_execution", None)
        if callable(decode_execution):
            decoded = decode_execution(result.action, execution_plan)
        else:
            if self._action_output_role is None:
                raise PipelineConfigurationError(
                    f"compiled pipeline {self.pipeline_id!r} has no action output role",
                    pipeline_id=self.pipeline_id,
                    code="invalid_action_role",
                )
            decoded = codec.decode_outputs(result.action, deployment.bindings[self._action_output_role])
        if not isinstance(decoded, CodecResult):
            raise PipelineValidationError(
                f"pipeline {self.pipeline_id!r} codec returned {type(decoded).__name__}, expected CodecResult",
                pipeline_id=self.pipeline_id,
            )
        return decoded.action

    def _require_compiled_codec(self) -> tuple[object, ExecutionPlan]:
        if self._codec is None or self._execution_plan is None:
            raise PipelineConfigurationError(
                f"compiled pipeline {self.pipeline_id!r} is missing its codec or execution plan",
                pipeline_id=self.pipeline_id,
                code="codec_required",
            )
        return self._codec, self._execution_plan

    def _load_component(self, component: object) -> None:
        load = getattr(component, "load", None)
        if callable(load):
            load(self._context)

    def _bind_processors(self) -> None:
        if self._preprocessor is None:
            self._preprocessor = _identity_preprocessor
        if self._postprocessor is None:
            self._postprocessor = _identity_postprocessor

    def _raise_if_expired(self, deadline: datetime | None, *, phase: str, backend_completed: bool) -> None:
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            raise self._timeout_error(phase, backend_completed=backend_completed)

    def _timeout_error(self, phase: str, *, backend_completed: bool) -> PipelineTimeoutError:
        completion = " after the backend completed; the late result was discarded" if backend_completed else ""
        return PipelineTimeoutError(
            f"pipeline {self.pipeline_id!r} request deadline expired during {phase}{completion}",
            pipeline_id=self.pipeline_id,
            phase=phase,
            backend_completed=backend_completed,
            cancellation_supported=self.capabilities.supports_cancellation,
        )

    def _record_policy_failure(self, exc: Exception, backend_execution_started: bool) -> None:
        if not hasattr(exc, "operation_started"):
            exc.operation_started = bool(backend_execution_started)
        if not hasattr(exc, "outcome_known"):
            exc.outcome_known = True
        capabilities = self.capabilities
        if isinstance(exc, PipelineTimeoutError):
            if exc.details.get("backend_completed") and capabilities.stateful:
                self._policy_failure = BackendHealth(
                    state=BackendState.FAILED,
                    ready=False,
                    reason_code="deadline_exceeded",
                    message=str(exc),
                )
            return
        non_mutating_rejection = isinstance(exc, BackendAdmissionError) and not exc.operation_started
        if backend_execution_started and not non_mutating_rejection and capabilities.stateful:
            self._policy_failure = BackendHealth(
                state=BackendState.FAILED,
                ready=False,
                reason_code=getattr(exc, "code", "execution_failed"),
                message=str(exc),
            )
