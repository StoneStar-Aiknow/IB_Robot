"""Composable named inference pipeline with strict lifecycle and diagnostics."""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from types import MappingProxyType

from inference_manifest import CompiledDeployment
from inference_service.backends import (
    BackendAdmissionError,
    BackendCapabilities,
    BackendHealth,
    BackendResult,
    BackendState,
    InferenceBackend,
    InferenceRequest,
    RuntimeContext,
)
from inference_service.codecs import CodecRequest, CodecResult, ExecutionPlan, PolicyCodec, build_execution_plan
from inference_service.pipeline.errors import (
    PipelineConfigurationError,
    PipelineLifecycleError,
    PipelineNotReadyError,
    PipelineTimeoutError,
    PipelineValidationError,
)
from inference_service.pipeline.state import PipelineState, PipelineStateMachine
from inference_service.pipeline.types import PipelineDiagnostics, PipelineResult
from inference_service.pipeline.validation import validate_action_output

Processor = Callable[[Mapping[str, object]], Mapping[str, object]]
Postprocessor = Callable[[object], object]


def _identity_preprocessor(inputs: Mapping[str, object]) -> Mapping[str, object]:
    return inputs


def _identity_postprocessor(action: object) -> object:
    return action


class InferencePipeline:
    """Own one processor/backend instance and expose one named inference route.

    Deadlines are cooperative and never implemented with detached worker threads.
    Backend admission observes the absolute deadline before execution. If an
    uncancellable backend call itself overruns, it is allowed to return and the
    late result is deterministically discarded.
    """

    def __init__(
        self,
        pipeline_id: str,
        runtime_context: RuntimeContext,
        backend: InferenceBackend,
        *,
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

        self._pipeline_id = pipeline_id
        self._context = runtime_context
        self._backend = backend
        self._preprocessor = preprocessor
        self._postprocessor = postprocessor
        self._owns_preprocessor = preprocessor is not None
        self._owns_postprocessor = postprocessor is not None
        self._codec = codec
        self._request_timeout = request_timeout
        self._default_task = default_task
        self._execution_mode = execution_mode
        self._state_machine = PipelineStateMachine()
        self._condition = threading.Condition(threading.RLock())
        self._active_requests = 0
        self._loading = False
        self._resetting = False
        self._execution_plan: ExecutionPlan | None = None
        self._action_output_role: str | None = None

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def state(self) -> PipelineState:
        with self._condition:
            return self._state_machine.state

    @property
    def runtime_context(self) -> RuntimeContext:
        return self._context

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._backend.capabilities

    def load(self) -> None:
        with self._condition:
            if self._state_machine.state is not PipelineState.CREATED:
                raise PipelineLifecycleError(
                    f"pipeline {self.pipeline_id!r} cannot load from state {self._state_machine.state.value}",
                    pipeline_id=self.pipeline_id,
                    code="invalid_load_state",
                )
            self._state_machine.transition(PipelineState.LOADING)
            self._loading = True

        try:
            self._prepare_execution()
            if self._preprocessor is not None:
                self._load_component(self._preprocessor)
            if self._postprocessor is not None:
                self._load_component(self._postprocessor)
            self._backend.load(self._context)
            self._bind_backend_processors()
            health = self._backend.health()
            if not health.ready:
                raise PipelineNotReadyError(
                    f"pipeline {self.pipeline_id!r} backend did not become ready after load",
                    pipeline_id=self.pipeline_id,
                    state=health.state.value,
                )
            with self._condition:
                if self._state_machine.state is PipelineState.CLOSING:
                    raise PipelineLifecycleError(
                        f"pipeline {self.pipeline_id!r} load was interrupted by close",
                        pipeline_id=self.pipeline_id,
                        code="load_interrupted",
                    )
                self._state_machine.transition(PipelineState.READY)
        except Exception:
            with self._condition:
                if self._state_machine.state not in {PipelineState.CLOSING, PipelineState.CLOSED}:
                    self._state_machine.transition(PipelineState.FAILED)
            raise
        finally:
            with self._condition:
                self._loading = False
                self._condition.notify_all()

    def infer(
        self,
        request: InferenceRequest,
        *,
        control_inputs: Mapping[str, object] | None = None,
        capture_raw_action: bool = False,
    ) -> PipelineResult:
        deadline = self._effective_deadline(request.deadline)
        with self._condition:
            self._synchronize_backend_health_locked()
            self._require_ready_locked("infer")
            self._raise_if_expired(deadline, phase="admission", backend_completed=False)
            self._active_requests += 1

        total_start = time.perf_counter()
        try:
            selected_prompt = request.prompt if request.prompt is not None else self._default_task
            processor_inputs = dict(request.inputs)
            if selected_prompt is not None:
                processor_inputs["task"] = selected_prompt

            preprocess_start = time.perf_counter()
            canonical_inputs = self._preprocessor(processor_inputs)
            preprocess_latency_ms = (time.perf_counter() - preprocess_start) * 1000.0
            if not isinstance(canonical_inputs, Mapping):
                raise PipelineValidationError(
                    f"pipeline {self.pipeline_id!r} preprocessor must return a mapping",
                    pipeline_id=self.pipeline_id,
                    details={"returned_type": type(canonical_inputs).__name__},
                )
            canonical_inputs = dict(canonical_inputs)
            if control_inputs:
                collisions = sorted(set(canonical_inputs) & set(control_inputs))
                if collisions:
                    raise PipelineValidationError(
                        f"pipeline {self.pipeline_id!r} control inputs conflict with preprocessor outputs: {collisions}",
                        pipeline_id=self.pipeline_id,
                        details={"conflicting_inputs": tuple(collisions)},
                    )
                canonical_inputs.update(control_inputs)
            self._raise_if_expired(deadline, phase="preprocess", backend_completed=False)

            backend_inputs = self._prepare_backend_inputs(canonical_inputs)
            backend_request = InferenceRequest(
                request_id=request.request_id,
                inputs=backend_inputs,
                prompt=selected_prompt,
                deadline=deadline,
                metadata={
                    **request.metadata,
                    "pipeline_id": self.pipeline_id,
                    "deployment": self._context.deployment_name,
                    "deployment_fingerprint": self._context.deployment_fingerprint,
                },
            )
            try:
                backend_result = self._backend.infer(backend_request)
            except BackendAdmissionError as exc:
                if exc.code != "deadline_exceeded":
                    raise
                raise self._timeout_error("backend_admission", backend_completed=False) from exc
            if not isinstance(backend_result, BackendResult):
                raise PipelineValidationError(
                    f"pipeline {self.pipeline_id!r} backend returned {type(backend_result).__name__}, "
                    "expected BackendResult",
                    pipeline_id=self.pipeline_id,
                )
            self._raise_if_expired(deadline, phase="backend", backend_completed=True)
            self._ensure_backend_ready_after_call()

            semantic_action = self._decode_backend_action(backend_result)
            validate_action_output(
                semantic_action,
                actual_chunk_size=backend_result.actual_chunk_size,
                action_dimension=self._action_dimension,
                pipeline_id=self.pipeline_id,
                phase="backend",
            )
            raw_action = _snapshot_action(semantic_action) if capture_raw_action else None

            postprocess_start = time.perf_counter()
            action = self._postprocessor(semantic_action)
            postprocess_latency_ms = (time.perf_counter() - postprocess_start) * 1000.0
            validate_action_output(
                action,
                actual_chunk_size=backend_result.actual_chunk_size,
                action_dimension=self._action_dimension,
                pipeline_id=self.pipeline_id,
                phase="postprocessor",
            )
            self._raise_if_expired(deadline, phase="postprocess", backend_completed=True)

            total_latency_ms = (time.perf_counter() - total_start) * 1000.0
            with self._condition:
                result_state = self._state_machine.state
            latency_metadata = MappingProxyType(
                {
                    "total": total_latency_ms,
                    "preprocess": preprocess_latency_ms,
                    "backend": backend_result.backend_latency_ms,
                    "postprocess": postprocess_latency_ms,
                }
            )
            return PipelineResult(
                action=action,
                actual_chunk_size=backend_result.actual_chunk_size,
                pipeline_id=self.pipeline_id,
                bundle=self._context.validated_manifest.manifest.bundle.name,
                deployment=self._context.deployment_name,
                deployment_fingerprint=self._context.deployment_fingerprint,
                backend=self._backend.name,
                state=result_state,
                total_latency_ms=total_latency_ms,
                preprocess_latency_ms=preprocess_latency_ms,
                backend_latency_ms=backend_result.backend_latency_ms,
                postprocess_latency_ms=postprocess_latency_ms,
                raw_action=raw_action,
                metadata={
                    **backend_result.metadata,
                    "pipeline_id": self.pipeline_id,
                    "bundle": self._context.validated_manifest.manifest.bundle.name,
                    "deployment": self._context.deployment_name,
                    "deployment_fingerprint": self._context.deployment_fingerprint,
                    "backend": self._backend.name,
                    "state": result_state.value,
                    "latency_ms": latency_metadata,
                },
            )
        except Exception:
            with self._condition:
                self._synchronize_backend_health_locked()
            raise
        finally:
            with self._condition:
                self._active_requests -= 1
                self._condition.notify_all()

    def reset(self) -> None:
        with self._condition:
            self._synchronize_backend_health_locked()
            self._require_ready_locked("reset")
            self._state_machine.transition(PipelineState.RESETTING)
            self._resetting = True
            self._condition.wait_for(lambda: self._active_requests == 0)
            if self._state_machine.state is not PipelineState.RESETTING:
                self._resetting = False
                self._condition.notify_all()
                raise PipelineLifecycleError(
                    f"pipeline {self.pipeline_id!r} reset was interrupted by close",
                    pipeline_id=self.pipeline_id,
                    code="reset_interrupted",
                )

        reset_error: Exception | None = None
        try:
            self._backend.reset()
        except Exception as exc:
            reset_error = exc
        finally:
            with self._condition:
                health = self._backend.health()
                if self._state_machine.state is PipelineState.RESETTING:
                    if health.ready:
                        self._state_machine.transition(PipelineState.READY)
                    else:
                        self._transition_from_backend_failure_locked(health)
                self._resetting = False
                self._condition.notify_all()

        if reset_error is not None:
            raise reset_error
        health = self._backend.health()
        if not health.ready:
            raise PipelineNotReadyError(
                f"pipeline {self.pipeline_id!r} backend is not ready after reset",
                pipeline_id=self.pipeline_id,
                state=health.state.value,
            )

    def cancel(self, request_id: str) -> None:
        with self._condition:
            self._synchronize_backend_health_locked()
            self._require_ready_locked("cancel")
        self._backend.cancel(request_id)

    def diagnostics(self) -> PipelineDiagnostics:
        with self._condition:
            health = self._backend.health()
            self._synchronize_backend_health_locked(health)
            return PipelineDiagnostics(
                pipeline_id=self.pipeline_id,
                bundle=self._context.validated_manifest.manifest.bundle.name,
                deployment=self._context.deployment_name,
                deployment_fingerprint=self._context.deployment_fingerprint,
                backend=self._backend.name,
                state=self._state_machine.state,
                backend_health=health,
                active_requests=self._active_requests,
                request_timeout=self._request_timeout,
                default_task_configured=self._default_task is not None,
            )

    def health(self) -> PipelineDiagnostics:
        return self.diagnostics()

    def close(self) -> None:
        with self._condition:
            if self._state_machine.state is PipelineState.CLOSED:
                return
            if self._state_machine.state is PipelineState.CLOSING:
                self._condition.wait_for(lambda: self._state_machine.state is PipelineState.CLOSED)
                return
            self._state_machine.transition(PipelineState.CLOSING)
            self._condition.wait_for(lambda: not self._loading and not self._resetting and self._active_requests == 0)

        errors: list[Exception] = []
        for component in self._owned_components_in_close_order():
            close = getattr(component, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as exc:
                errors.append(exc)

        with self._condition:
            self._state_machine.transition(PipelineState.CLOSED)
            self._condition.notify_all()
        if errors:
            raise PipelineLifecycleError(
                f"pipeline {self.pipeline_id!r} close failed: " + "; ".join(str(error) for error in errors),
                pipeline_id=self.pipeline_id,
                code="close_failed",
                details={"errors": tuple(str(error) for error in errors)},
            )

    @property
    def _action_dimension(self) -> int:
        return self._context.policy.output_features["action"].shape[-1]

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

    def _effective_deadline(self, request_deadline: datetime | None) -> datetime | None:
        now = datetime.now(timezone.utc)
        configured_deadline = None
        if self._request_timeout is not None:
            configured_deadline = now + timedelta(seconds=self._request_timeout)
        if request_deadline is None:
            return configured_deadline
        if request_deadline.tzinfo is None:
            raise PipelineConfigurationError(
                f"pipeline {self.pipeline_id!r} request deadline must be timezone-aware",
                pipeline_id=self.pipeline_id,
                code="invalid_deadline",
            )
        normalized = request_deadline.astimezone(timezone.utc)
        return normalized if configured_deadline is None else min(normalized, configured_deadline)

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
        with self._condition:
            health = self._backend.health()
            self._synchronize_backend_health_locked(health)
            if not health.ready:
                raise PipelineNotReadyError(
                    f"pipeline {self.pipeline_id!r} backend left READY during inference",
                    pipeline_id=self.pipeline_id,
                    state=health.state.value,
                )

    def _synchronize_backend_health_locked(self, health: BackendHealth | None = None) -> None:
        state = self._state_machine.state
        if state not in {PipelineState.READY, PipelineState.RESETTING, PipelineState.DEGRADED}:
            return
        current_health = health or self._backend.health()
        if current_health.ready:
            if state is PipelineState.DEGRADED:
                self._state_machine.transition(PipelineState.READY)
            return
        self._transition_from_backend_failure_locked(current_health)

    def _transition_from_backend_failure_locked(self, health: BackendHealth) -> None:
        target = (
            PipelineState.DEGRADED
            if health.state in {BackendState.DEGRADED, BackendState.RECOVERING}
            else PipelineState.FAILED
        )
        if self._state_machine.state is target:
            return
        self._state_machine.transition(target)

    def _require_ready_locked(self, operation: str) -> None:
        if self._state_machine.state is not PipelineState.READY:
            raise PipelineNotReadyError(
                f"pipeline {self.pipeline_id!r} cannot {operation} while state is {self._state_machine.state.value}",
                pipeline_id=self.pipeline_id,
                state=self._state_machine.state.value,
            )


def _snapshot_action(action: object) -> object:
    detached = getattr(action, "detach", None)
    candidate = detached() if callable(detached) else action
    clone = getattr(candidate, "clone", None)
    if callable(clone):
        return clone()
    return copy.deepcopy(candidate)
