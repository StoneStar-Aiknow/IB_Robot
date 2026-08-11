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
    InferenceBackend,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.codecs import CodecRequest, CodecResult, ExecutionPlan, PolicyCodec, build_execution_plan
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.pi05_schedule import PI05DenoisingSchedule
from inference_service.pipeline.errors import (
    PipelineCanceledError,
    PipelineConfigurationError,
    PipelineLifecycleError,
    PipelineNotReadyError,
    PipelineTimeoutError,
    PipelineValidationError,
)
from inference_service.pipeline.runtime_core import (
    ExecutionControl,
    ExecutionError,
    GenericModelPipeline,
    ModelExecutor,
    StageFrame,
)
from inference_service.pipeline.stages import InferenceStage
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
    if action.ndim < 2 or action.shape[-2] < 1:
        raise PipelineValidationError(
            f"PI0.5 session action output has invalid shape {action.shape}",
            pipeline_id="pi05",
            code="invalid_action_shape",
        )
    return int(action.shape[-2])


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


class _PolicyBackendStage:
    """Backend delegated model stage: codec-bound backend invocation and readiness gate."""

    def __init__(self, facade: InferencePipeline) -> None:
        self._facade = facade

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        frame.control.raise_if_canceled("backend")
        request = frame.request.inner
        backend_inputs = self._facade._prepare_backend_inputs(frame.values["_canonical_inputs"])
        backend_request = InferenceRequest(
            request_id=request.request_id,
            inputs=backend_inputs,
            prompt=frame.values["_selected_prompt"],
            deadline=deadline,
            priority=request.priority,
            metadata={
                **request.metadata,
                "pipeline_id": self._facade.pipeline_id,
                "deployment": self._facade._context.deployment_name,
                "deployment_fingerprint": self._facade._context.deployment_fingerprint,
            },
        )
        try:
            frame.values["_backend_started"] = True
            backend_result = self._facade._backend.infer(backend_request)
        except BackendAdmissionError as exc:
            if exc.code != "deadline_exceeded":
                raise
            raise self._facade._timeout_error("backend_admission", backend_completed=False) from exc
        if not isinstance(backend_result, BackendResult):
            raise PipelineValidationError(
                f"pipeline {self._facade.pipeline_id!r} backend returned {type(backend_result).__name__}, "
                "expected BackendResult",
                pipeline_id=self._facade.pipeline_id,
            )
        self._facade._raise_if_expired(deadline, phase="backend", backend_completed=True)
        self._facade._ensure_backend_ready_after_call()

        frame.values["_backend_result"] = backend_result


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


class _PI05SessionStage:
    """Session-driven iterative PI0.5 execution replacing the legacy backend stage.

    Binds canonical policy inputs to manifest role semantics, drives the shared
    ``IterativeStage`` over an :class:`AscendOmModelSession`, and publishes a
    ``BackendResult`` for the shared decode/postprocess stages.
    """

    def __init__(
        self,
        facade: InferencePipeline,
        session_executor,
        action_semantic: str,
        denoising_schedule_metadata: Mapping[str, object] | None,
    ) -> None:
        self._facade = facade
        self._session_executor = session_executor
        self._action_semantic = action_semantic
        self._denoising_schedule_metadata = denoising_schedule_metadata

    def execute(self, frame: StageFrame, *, deadline: datetime | None) -> None:
        frame.control.raise_if_canceled("pi05.session")
        canonical_inputs = frame.values["_canonical_inputs"]
        role_inputs = self._facade._encode_role_inputs(canonical_inputs)
        semantic_values: dict[str, object] = {}
        for bound_inputs in role_inputs.values():
            for tensor in bound_inputs.tensors:
                semantic_values[tensor.semantic] = tensor.value
        named_request = NamedTensorRequest(
            frame.request.inner.request_id,
            MappingProxyType(semantic_values),
            deadline=deadline,
            priority=frame.request.inner.priority,
        )
        backend_started = True
        try:
            raw_action = self._session_executor.execute(named_request, deadline=deadline, control=frame.control)
            frame.values["_backend_started"] = backend_started
        except PipelineCanceledError:
            raise
        except Exception:
            frame.values["_backend_started"] = backend_started
            raise
        if not isinstance(raw_action, np.ndarray):
            raise PipelineValidationError(
                f"pipeline {self._facade.pipeline_id!r} PI0.5 session returned "
                f"{type(raw_action).__name__}, expected a NumPy action array",
                pipeline_id=self._facade.pipeline_id,
            )
        self._facade._raise_if_expired(deadline, phase="backend", backend_completed=True)
        self._facade._ensure_session_ready_after_call()
        chunk_size = _chunk_size_from_action(raw_action)
        metadata = {
            "request_id": frame.request.inner.request_id,
            "deployment_name": self._facade._context.deployment_name,
            "deployment_fingerprint": self._facade._context.deployment_fingerprint,
        }
        if self._denoising_schedule_metadata is not None:
            metadata["denoising_schedule"] = self._denoising_schedule_metadata
        frame.values["_backend_result"] = BackendResult(
            action=raw_action,
            actual_chunk_size=chunk_size,
            backend_latency_ms=0.0,
            metadata=metadata,
        )


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


class _PolicySequentialExecutor(ModelExecutor):
    """Stage-composed policy executor driving ordered InferenceStage execution."""

    def __init__(self, facade: InferencePipeline) -> None:
        self._facade = facade
        self._stages: tuple[InferenceStage, ...] = (
            _PolicyPreprocessStage(facade),
            _PolicyBackendStage(facade),
            _PolicyDecodeStage(facade),
            _PolicyPostprocessStage(facade),
        )
        self._result_adapter = _PolicyResultAdapter(facade)

    def load(self, context: RuntimeContext) -> None:
        self._facade._load_executor(context)

    def execute(self, request: object, *, deadline: datetime | None, control: ExecutionControl) -> PipelineResult:
        if not isinstance(request, _PolicyRequest):
            raise TypeError("policy executor requires a _PolicyRequest")
        frame = StageFrame(request, values={"_total_start": time.perf_counter()}, control=control)
        try:
            for stage in self._stages:
                stage.execute(frame, deadline=deadline)
            return self._result_adapter.adapt(frame)
        except PipelineCanceledError:
            raise
        except Exception as exc:
            self._facade._record_policy_failure(exc, bool(frame.values.get("_backend_started", False)))
            raise
        finally:
            frame.close()

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        # Cancellable backends delegate to backend.cancel and preserve its structured
        # admission/cancellation errors and stateful fail-closed behavior. Non-cancellable
        # backends are not interrupted here; the shared ExecutionControl flag marks the
        # in-flight request so the completed result is discarded as a late cancellation.
        if self._facade._backend.capabilities.supports_cancellation:
            self._facade._backend.cancel(request_id, deadline=deadline)

    def adapt_error(self, error: ExecutionError) -> PipelineResult:
        return self._result_adapter.adapt_error(error)

    def health(self) -> BackendHealth:
        return self._facade._executor_health()

    def reset(self, deadline: datetime | None = None) -> None:
        self._facade._reset_executor(deadline)

    def close(self) -> None:
        self._facade._close_executor()


class _PI05PolicyExecutor(ModelExecutor):
    """Drive a session-backed PI0.5 iterative executor behind the policy facade.

    Composes the shared preprocess/decode/postprocess/result stages with a
    :class:`_PI05SessionStage` that delegates iterative execution to
    :func:`create_pi05_executor`. Lifecycle, health, and resource ownership are
    delegated to the session executor so the facade never loads a second backend.
    """

    def __init__(
        self,
        facade: InferencePipeline,
        handle: _PI05SessionHandle,
    ) -> None:
        self._facade = facade
        self._handle = handle
        self._session_executor = handle.session_executor
        self._capabilities = handle.capabilities
        self._curvature_log_path = handle.curvature_log_path
        self._velocity_trace = handle.velocity_trace
        self._stages: tuple[InferenceStage, ...] = (
            _PolicyPreprocessStage(facade),
            _PI05SessionStage(facade, handle.session_executor, handle.action_semantic, handle.schedule_metadata),
            _PolicyDecodeStage(facade),
            _PolicyPostprocessStage(facade),
        )
        self._result_adapter = _PolicyResultAdapter(facade)

    @property
    def capabilities(self):
        return self._capabilities

    def load(self, context: RuntimeContext) -> None:
        if self._facade._preprocessor is not None:
            self._facade._load_component(self._facade._preprocessor)
        if self._facade._postprocessor is not None:
            self._facade._load_component(self._facade._postprocessor)
        self._facade._bind_backend_processors()
        self._session_executor.load(self._handle.session_context)
        health = self._session_executor.health()
        if not health.ready:
            raise PipelineNotReadyError(
                f"pipeline {self._facade.pipeline_id!r} PI0.5 session did not become ready after load",
                pipeline_id=self._facade.pipeline_id,
                state=health.state.value,
            )

    def execute(self, request: object, *, deadline: datetime | None, control: ExecutionControl) -> PipelineResult:
        if not isinstance(request, _PolicyRequest):
            raise TypeError("PI0.5 policy executor requires a _PolicyRequest")
        frame = StageFrame(request, values={"_total_start": time.perf_counter()}, control=control)
        try:
            for stage in self._stages:
                stage.execute(frame, deadline=deadline)
            result = self._result_adapter.adapt(frame)
            self._write_curvature_log()
            return result
        except PipelineCanceledError:
            raise
        except Exception as exc:
            self._facade._record_policy_failure(exc, bool(frame.values.get("_backend_started", False)))
            raise
        finally:
            frame.close()

    def _write_curvature_log(self) -> None:
        if self._curvature_log_path is None or not self._velocity_trace:
            return
        scores = _curvature_scores(self._velocity_trace)
        record: dict[str, object] = {"curvature_scores": scores}
        if self._handle.schedule is not None:
            record["schedule"] = self._handle.schedule.to_dict()
        path = Path(str(self._curvature_log_path)).expanduser().resolve()
        try:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
        except (OSError, ValueError) as exc:
            raise PipelineValidationError(
                f"pipeline {self._facade.pipeline_id!r} unable to write PI0.5 curvature log {path}: {exc}",
                pipeline_id=self._facade.pipeline_id,
                code="curvature_log_failed",
            ) from exc

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        cancel = getattr(self._session_executor, "cancel", None)
        capabilities = getattr(self._session_executor, "capabilities", None)
        if callable(cancel) and getattr(capabilities, "supports_cancellation", False):
            cancel(request_id, deadline=deadline)

    def adapt_error(self, error: ExecutionError) -> PipelineResult:
        return self._result_adapter.adapt_error(error)

    def health(self) -> BackendHealth:
        if self._facade._policy_failure is not None:
            return self._facade._policy_failure
        return self._session_executor.health()

    def reset(self, deadline: datetime | None = None) -> None:
        self._session_executor.reset(deadline=deadline)

    def close(self) -> None:
        errors: list[Exception] = []
        components: list[object] = []
        if self._facade._owns_postprocessor and self._facade._postprocessor is not None:
            components.append(self._facade._postprocessor)
        if self._facade._owns_preprocessor and self._facade._preprocessor is not None:
            components.append(self._facade._preprocessor)
        components.append(self._session_executor)
        seen: set[int] = set()
        for component in components:
            if id(component) in seen:
                continue
            seen.add(id(component))
            close = getattr(component, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise PipelineLifecycleError(
                f"pipeline {self._facade.pipeline_id!r} close failed: " + "; ".join(str(error) for error in errors),
                pipeline_id=self._facade.pipeline_id,
                code="close_failed",
                details={"errors": tuple(str(error) for error in errors)},
            )


class _PI05SessionHandle:
    """Configuration bundle the factory passes so the facade can build the executor."""

    def __init__(
        self,
        session_executor,
        session_context: RuntimeContext,
        action_semantic: str,
        schedule_metadata: Mapping[str, object] | None,
        schedule: PI05DenoisingSchedule | None,
        capabilities,
        curvature_log_path: str | None,
        velocity_trace: list | None,
    ) -> None:
        self.session_executor = session_executor
        self.session_context = session_context
        self.action_semantic = action_semantic
        self.schedule_metadata = schedule_metadata
        self.schedule = schedule
        self.capabilities = capabilities
        self.curvature_log_path = curvature_log_path
        self.velocity_trace = velocity_trace


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
        backend: InferenceBackend | None = None,
        *,
        executor: ModelExecutor | None = None,
        pi05_handle: _PI05SessionHandle | None = None,
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
        construction_sources = sum(1 for source in (backend, executor, pi05_handle) if source is not None)
        if construction_sources != 1:
            raise PipelineConfigurationError(
                f"pipeline {pipeline_id!r} requires exactly one of backend, executor, or pi05_handle",
                pipeline_id=pipeline_id,
                code="invalid_pipeline_construction",
            )

        self._pipeline_id = pipeline_id
        self._context = runtime_context
        self._backend = backend
        self._executor = executor
        self._owns_executor = executor is not None
        self._pi05_handle = pi05_handle
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
        self._executor_closed = False
        self._prepare_execution()

        if pi05_handle is not None:
            resolved_executor: ModelExecutor = _PI05PolicyExecutor(self, pi05_handle)
        elif executor is not None:
            resolved_executor = executor
        else:
            resolved_executor = _PolicySequentialExecutor(self)
        self._pipeline = GenericModelPipeline(
            pipeline_id,
            runtime_context,
            resolved_executor,
            request_timeout=request_timeout,
        )

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
        if self._backend is not None:
            return self._backend.capabilities
        if self._pi05_handle is not None:
            return self._pi05_handle.capabilities
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
    # Executor hooks (called by _PolicySequentialExecutor)
    # ------------------------------------------------------------------

    def _load_executor(self, context: RuntimeContext) -> None:
        if self._preprocessor is not None:
            self._load_component(self._preprocessor)
        if self._postprocessor is not None:
            self._load_component(self._postprocessor)
        self._backend.load(context)
        self._bind_backend_processors()
        health = self._backend.health()
        if not health.ready:
            raise PipelineNotReadyError(
                f"pipeline {self.pipeline_id!r} backend did not become ready after load",
                pipeline_id=self.pipeline_id,
                state=health.state.value,
            )

    def _reset_executor(self, deadline: datetime | None) -> None:
        if self._backend.capabilities.stateful and not self._backend.capabilities.resettable:
            raise PipelineLifecycleError(
                f"pipeline {self.pipeline_id!r} backend is stateful but does not support reset",
                pipeline_id=self.pipeline_id,
                code="reset_unsupported",
            )

        backend_reset_error: Exception | None = None
        if self._backend.capabilities.resettable:
            try:
                self._backend.reset(deadline=deadline)
            except Exception as exc:
                backend_reset_error = exc

        processor_reset_error: Exception | None = None
        processor_reset_started = False
        reset_mutated_state = self._backend.capabilities.resettable and backend_reset_error is None
        if backend_reset_error is None:
            seen: set[int] = set()
            for component in (self._preprocessor, self._postprocessor):
                if component is None or id(component) in seen:
                    continue
                seen.add(id(component))
                reset = getattr(component, "reset", None)
                if not callable(reset):
                    continue
                try:
                    self._raise_if_expired(deadline, phase="reset", backend_completed=True)
                    processor_reset_started = True
                    reset()
                    self._raise_if_expired(deadline, phase="reset", backend_completed=True)
                except Exception as exc:
                    if processor_reset_error is None:
                        processor_reset_error = exc
                    break
        reset_error = backend_reset_error or processor_reset_error
        if (processor_reset_error is not None and (processor_reset_started or reset_mutated_state)) or (
            backend_reset_error is not None and not self._backend.health().ready
        ):
            self._policy_failure = BackendHealth(
                state=BackendState.FAILED,
                ready=False,
                reason_code="reset_failed",
                message=str(reset_error) if reset_error is not None else "reset failed",
            )
        if reset_error is not None:
            raise reset_error
        health = self._backend.health()
        if not health.ready:
            raise PipelineNotReadyError(
                f"pipeline {self.pipeline_id!r} backend is not ready after reset",
                pipeline_id=self.pipeline_id,
                state=health.state.value,
            )

    def _close_executor(self) -> None:
        if self._executor_closed:
            return
        self._executor_closed = True
        errors: list[Exception] = []
        for component in self._owned_components_in_close_order():
            close = getattr(component, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise PipelineLifecycleError(
                f"pipeline {self.pipeline_id!r} close failed: " + "; ".join(str(error) for error in errors),
                pipeline_id=self.pipeline_id,
                code="close_failed",
                details={"errors": tuple(str(error) for error in errors)},
            )

    def _executor_health(self) -> BackendHealth:
        if self._policy_failure is not None:
            return self._policy_failure
        return self._backend.health()

    # ------------------------------------------------------------------
    # Policy helpers (preserved from the original implementation)
    # ------------------------------------------------------------------

    @property
    def _action_dimension(self) -> int:
        return self._context.policy.output_features["action"].shape[-1]

    def _backend_name(self) -> str:
        if self._backend is not None:
            return self._backend.name
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

    def _prepare_backend_inputs(self, canonical_inputs: Mapping[str, object]) -> Mapping[str, object]:
        deployment = self._context.deployment
        if not isinstance(deployment, CompiledDeployment):
            return dict(canonical_inputs)

        assert self._codec is not None
        assert self._execution_plan is not None
        encode_execution = getattr(self._codec, "encode_execution", None)
        if callable(encode_execution):
            role_inputs = encode_execution(CodecRequest(canonical_inputs), self._execution_plan)
        elif len(self._execution_plan.roles) == 1:
            role = self._execution_plan.roles[0]
            role_inputs = {role.name: self._codec.encode_inputs(CodecRequest(canonical_inputs), role.bindings)}
        else:
            raise PipelineConfigurationError(
                f"compiled pipeline {self.pipeline_id!r} requires an execution-aware codec for multiple roles",
                pipeline_id=self.pipeline_id,
                code="execution_codec_required",
                details={"roles": self._execution_plan.role_names},
            )
        return {
            "execution_plan": self._execution_plan,
            "role_inputs": MappingProxyType(dict(role_inputs)),
        }

    def _encode_role_inputs(self, canonical_inputs: Mapping[str, object]) -> Mapping[str, object]:
        assert self._codec is not None
        assert self._execution_plan is not None
        encode_execution = getattr(self._codec, "encode_execution", None)
        if callable(encode_execution):
            return encode_execution(CodecRequest(canonical_inputs), self._execution_plan)
        if len(self._execution_plan.roles) == 1:
            role = self._execution_plan.roles[0]
            return {role.name: self._codec.encode_inputs(CodecRequest(canonical_inputs), role.bindings)}
        raise PipelineConfigurationError(
            f"compiled pipeline {self.pipeline_id!r} requires an execution-aware codec for multiple roles",
            pipeline_id=self.pipeline_id,
            code="execution_codec_required",
            details={"roles": self._execution_plan.role_names},
        )

    def _ensure_session_ready_after_call(self) -> None:
        if self._executor is None:
            return
        health = self._executor.health()
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

        assert self._codec is not None
        assert self._execution_plan is not None
        decode_execution = getattr(self._codec, "decode_execution", None)
        if callable(decode_execution):
            decoded = decode_execution(result.action, self._execution_plan)
        else:
            assert self._action_output_role is not None
            decoded = self._codec.decode_outputs(result.action, deployment.bindings[self._action_output_role])
        if not isinstance(decoded, CodecResult):
            raise PipelineValidationError(
                f"pipeline {self.pipeline_id!r} codec returned {type(decoded).__name__}, expected CodecResult",
                pipeline_id=self.pipeline_id,
            )
        return decoded.action

    def _load_component(self, component: object) -> None:
        load = getattr(component, "load", None)
        if callable(load):
            load(self._context)

    def _bind_backend_processors(self) -> None:
        if self._preprocessor is None:
            self._preprocessor = self._borrow_backend_processor("preprocessor", _identity_preprocessor)
        if self._postprocessor is None:
            self._postprocessor = self._borrow_backend_processor("postprocessor", _identity_postprocessor)

    def _borrow_backend_processor(self, name: str, fallback: object) -> object:
        processor = getattr(self._backend, name, None)
        if processor is None:
            return fallback
        if not callable(processor):
            raise PipelineConfigurationError(
                f"pipeline {self.pipeline_id!r} backend {self._backend.name!r} exposed a non-callable {name}",
                pipeline_id=self.pipeline_id,
                code="invalid_backend_processor",
                details={"processor": name, "returned_type": type(processor).__name__},
            )
        return processor

    def _owned_components_in_close_order(self) -> tuple[object, ...]:
        components = [self._backend]
        if self._owns_postprocessor:
            components.append(self._postprocessor)
        if self._owns_preprocessor:
            components.append(self._preprocessor)
        unique: list[object] = []
        seen: set[int] = set()
        for component in components:
            identity = id(component)
            if identity in seen:
                continue
            seen.add(identity)
            unique.append(component)
        return tuple(unique)

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
            cancellation_supported=self._backend.capabilities.supports_cancellation,
        )

    def _ensure_backend_ready_after_call(self) -> None:
        health = self._backend.health()
        if not health.ready:
            raise PipelineNotReadyError(
                f"pipeline {self.pipeline_id!r} backend left READY during inference",
                pipeline_id=self.pipeline_id,
                state=health.state.value,
            )

    def _record_policy_failure(self, exc: Exception, backend_execution_started: bool) -> None:
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
