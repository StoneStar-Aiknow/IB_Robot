"""Model-neutral pipeline lifecycle and request orchestration."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from inference_service.backends import (
    BackendAdmissionError,
    BackendHealth,
    BackendState,
    RuntimeContext,
)
from inference_service.codecs import ExecutionFrame, ExecutionPlan
from inference_service.generic_runtime import DeploymentIdentity, NamedTensorRequest, NamedTensorResult, RuntimeLatency
from inference_service.pipeline.errors import (
    PipelineCanceledError,
    PipelineConfigurationError,
    PipelineLifecycleError,
    PipelineNotReadyError,
    PipelineTimeoutError,
)
from inference_service.pipeline.state import PipelineState, PipelineStateMachine


@runtime_checkable
class ModelExecutor(Protocol):
    """Execute one domain operation independently of ROS transport."""

    def load(self, context: RuntimeContext) -> None: ...

    def execute(self, request: object, *, deadline: datetime | None, control: ExecutionControl) -> object: ...

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None: ...

    def adapt_error(self, error: ExecutionError) -> object: ...

    def health(self) -> BackendHealth: ...

    def reset(self, deadline: datetime | None = None) -> None: ...

    def close(self) -> None: ...


class ExecutionControl:
    """Thread-safe request cancellation state shared by the core and stages."""

    def __init__(self, request_id: str) -> None:
        if not request_id:
            raise ValueError("execution control request_id must be non-empty")
        self.request_id = request_id
        self._canceled = threading.Event()

    @property
    def cancellation_requested(self) -> bool:
        return self._canceled.is_set()

    def cancel(self) -> None:
        self._canceled.set()

    def raise_if_canceled(self, stage: str) -> None:
        if self.cancellation_requested:
            raise PipelineCanceledError(self.request_id, stage=stage)


@dataclass(frozen=True)
class ExecutionError:
    """Domain-neutral normalized execution termination."""

    code: str
    request_id: str
    stage: str
    message: str
    recoverable: bool = False
    details: Mapping[str, object] = field(default_factory=dict)
    cause: Exception | None = None
    partial_result: object | None = None

    def __post_init__(self) -> None:
        if not self.code or not self.request_id or not self.stage or not self.message:
            raise ValueError("execution error code, request_id, stage, and message must be non-empty")
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))


class StageFrame:
    """Request-local domain values plus the manifest-derived execution frame."""

    def __init__(
        self,
        request: object,
        *,
        execution_plan: ExecutionPlan | None = None,
        values: Mapping[str, object] | None = None,
        control: ExecutionControl | None = None,
    ) -> None:
        self.request = request
        self.execution_plan = execution_plan
        self.execution_frame = ExecutionFrame(execution_plan) if execution_plan is not None else None
        self.values: dict[str, object] = dict(values or {})
        self.control = control or ExecutionControl(self._request_id(request))
        self._session_executions: dict[int, object] = {}
        self._session_execution_stack: ExitStack | None = None
        self._closed = False

    def bind_session_execution(self, session: object, execution: object) -> None:
        self._session_executions[id(session)] = execution

    def session_execution(self, session: object) -> object | None:
        return self._session_executions.get(id(session))

    def open_session_execution(self, session: object, request: object, deadline: datetime | None) -> object | None:
        entered = self.session_execution(session)
        if entered is not None:
            return entered
        execution = getattr(session, "execution", None)
        if not callable(execution):
            return None
        if self._session_execution_stack is None:
            self._session_execution_stack = ExitStack()
        scoped_request = NamedTensorRequest(
            request.request_id,
            self.values,
            deadline=deadline,
            priority=getattr(request, "priority", 0),
        )
        entered = self._session_execution_stack.enter_context(execution(scoped_request))
        self.bind_session_execution(session, entered)
        return entered

    def close(self) -> None:
        if self._closed:
            return
        if self.execution_frame is not None:
            self.execution_frame.close()
        if self._session_execution_stack is not None:
            self._session_execution_stack.close()
            self._session_execution_stack = None
        self._session_executions.clear()
        self.values.clear()
        self._closed = True

    @staticmethod
    def _request_id(request: object) -> str:
        request_id = getattr(request, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            raise TypeError("stage frame request must expose a non-empty request_id")
        return request_id


@dataclass(frozen=True)
class PipelineRuntimeDiagnostics:
    """Model-neutral aggregate pipeline diagnostics."""

    pipeline_id: str
    deployment: DeploymentIdentity
    state: PipelineState
    executor_health: BackendHealth
    active_requests: int
    request_timeout: float | None
    last_latency: RuntimeLatency | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def ready(self) -> bool:
        return self.state is PipelineState.READY and self.executor_health.ready


class PipelineRuntimeCore:
    """Own logical pipeline lifecycle while delegating model resources to an executor."""

    def __init__(
        self,
        pipeline_id: str,
        runtime_context: RuntimeContext,
        executor: ModelExecutor,
        *,
        request_timeout: float | None = None,
    ) -> None:
        if not pipeline_id:
            raise PipelineConfigurationError("pipeline_id must be non-empty")
        if request_timeout is not None and (not math.isfinite(request_timeout) or request_timeout <= 0):
            raise PipelineConfigurationError(
                f"pipeline {pipeline_id!r} request_timeout must be finite and positive",
                pipeline_id=pipeline_id,
                details={"request_timeout": request_timeout},
            )
        self._pipeline_id = pipeline_id
        self._context = runtime_context
        self._executor = executor
        self._request_timeout = request_timeout
        self._state_machine = PipelineStateMachine()
        self._condition = threading.Condition(threading.RLock())
        self._control_lock = threading.Lock()
        self._active_requests = 0
        self._active_controls = 0
        self._loading = False
        self._resetting = False
        self._last_latency: RuntimeLatency | None = None
        self._request_controls: dict[str, ExecutionControl] = {}

    @property
    def pipeline_id(self) -> str:
        return self._pipeline_id

    @property
    def runtime_context(self) -> RuntimeContext:
        return self._context

    @property
    def state(self) -> PipelineState:
        with self._condition:
            return self._state_machine.state

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
            self._executor.load(self._context)
            health = self._executor.health()
            if not health.ready:
                raise PipelineNotReadyError(
                    f"pipeline {self.pipeline_id!r} executor did not become ready after load",
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
        except Exception as exc:
            rollback_error: Exception | None = None
            try:
                self._executor.close()
            except Exception as close_exc:
                rollback_error = close_exc
            with self._condition:
                if self._state_machine.state not in {PipelineState.CLOSING, PipelineState.CLOSED}:
                    self._state_machine.transition(PipelineState.FAILED)
            if rollback_error is not None:
                raise PipelineLifecycleError(
                    f"pipeline {self.pipeline_id!r} load failed: {exc}; rollback failed: {rollback_error}",
                    pipeline_id=self.pipeline_id,
                    code="load_failed",
                ) from exc
            raise
        finally:
            with self._condition:
                self._loading = False
                self._condition.notify_all()

    def execute(self, request: object, *, deadline: datetime | None = None) -> object:
        request_deadline = deadline
        if isinstance(request, NamedTensorRequest):
            request_deadline = request.deadline if deadline is None else deadline
        effective_deadline = self._effective_deadline(request_deadline)
        request_id = self._request_id(request)
        control = ExecutionControl(request_id)
        with self._request_operation(effective_deadline, control):
            started = time.perf_counter()
            try:
                control.raise_if_canceled("admission")
                result = self._executor.execute(request, deadline=effective_deadline, control=control)
                control.raise_if_canceled("result")
                self._raise_if_expired(effective_deadline, phase="executor", backend_completed=True)
            except Exception as exc:
                with self._condition:
                    self._synchronize_executor_health_locked()
                normalized = exc
                if isinstance(exc, BackendAdmissionError) and exc.code == "deadline_exceeded":
                    normalized = self._timeout_error("executor", backend_completed=exc.operation_started)
                return self._executor.adapt_error(self._execution_error(normalized, request_id))
            total_ms = (time.perf_counter() - started) * 1000.0
            with self._condition:
                self._synchronize_executor_health_locked()
                self._require_ready_locked("publish execution result")
                result = self._with_pipeline_latency(result, total_ms)
                self._last_latency = self._result_latency(result, total_ms)
            return result

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        if not request_id:
            raise PipelineConfigurationError("cancel request_id must be non-empty", pipeline_id=self.pipeline_id)
        effective_deadline = self._effective_deadline(deadline)
        with self._control_operation("cancel", effective_deadline):
            with self._condition:
                control = self._request_controls.get(request_id)
                if control is not None:
                    control.cancel()
            try:
                self._executor.cancel(request_id, deadline=effective_deadline)
                self._raise_if_expired(effective_deadline, phase="cancel", backend_completed=True)
            except Exception:
                with self._condition:
                    self._synchronize_executor_health_locked()
                raise
            with self._condition:
                self._synchronize_executor_health_locked()
                self._require_ready_locked("complete cancellation")

    def reset(self, deadline: datetime | None = None) -> None:
        effective_deadline = self._effective_deadline(deadline)
        with self._control_operation("reset", effective_deadline):
            with self._condition:
                self._state_machine.transition(PipelineState.RESETTING)
                self._resetting = True
                completed = self._condition.wait_for(
                    lambda: self._active_requests == 0,
                    timeout=self._remaining_seconds(effective_deadline),
                )
                if not completed:
                    self._state_machine.transition(PipelineState.READY)
                    self._resetting = False
                    self._condition.notify_all()
                    raise self._timeout_error("reset admission", backend_completed=False)
            try:
                self._executor.reset(deadline=effective_deadline)
                self._raise_if_expired(effective_deadline, phase="reset", backend_completed=True)
            except Exception:
                with self._condition:
                    self._synchronize_executor_health_locked()
                raise
            finally:
                with self._condition:
                    self._resetting = False
                    if self._state_machine.state is PipelineState.RESETTING:
                        self._state_machine.transition(PipelineState.READY)
                    self._condition.notify_all()

    def diagnostics(self) -> PipelineRuntimeDiagnostics:
        with self._condition:
            health = self._executor.health()
            self._synchronize_executor_health_locked(health)
            return PipelineRuntimeDiagnostics(
                pipeline_id=self.pipeline_id,
                deployment=self._deployment_identity(),
                state=self._state_machine.state,
                executor_health=health,
                active_requests=self._active_requests,
                request_timeout=self._request_timeout,
                last_latency=self._last_latency,
                metadata={"model_kind": self._context.model.kind, "model_family": self._context.model.family},
            )

    def health(self) -> PipelineRuntimeDiagnostics:
        return self.diagnostics()

    def close(self) -> None:
        with self._condition:
            if self._state_machine.state is PipelineState.CLOSED:
                return
            if self._state_machine.state is PipelineState.CLOSING:
                self._condition.wait_for(lambda: self._state_machine.state is PipelineState.CLOSED)
                return
            self._state_machine.transition(PipelineState.CLOSING)
            self._condition.wait_for(
                lambda: (
                    not self._loading
                    and not self._resetting
                    and self._active_requests == 0
                    and self._active_controls == 0
                )
            )
        error: Exception | None = None
        try:
            self._executor.close()
        except Exception as exc:
            error = exc
        finally:
            with self._condition:
                self._state_machine.transition(PipelineState.CLOSED)
                self._condition.notify_all()
        if error is not None:
            raise PipelineLifecycleError(
                f"pipeline {self.pipeline_id!r} close failed: {error}",
                pipeline_id=self.pipeline_id,
                code="close_failed",
                details={"errors": (str(error),)},
            ) from error

    @contextmanager
    def _request_operation(self, deadline: datetime | None, control: ExecutionControl):
        with self._condition:
            self._synchronize_executor_health_locked()
            self._require_ready_locked("execute")
            self._raise_if_expired(deadline, phase="admission", backend_completed=False)
            if control.request_id in self._request_controls:
                raise PipelineLifecycleError(
                    f"pipeline {self.pipeline_id!r} already has active request {control.request_id!r}",
                    pipeline_id=self.pipeline_id,
                    code="duplicate_request_id",
                )
            self._active_requests += 1
            self._request_controls[control.request_id] = control
        try:
            yield
        finally:
            with self._condition:
                self._active_requests -= 1
                self._request_controls.pop(control.request_id, None)
                self._condition.notify_all()

    @contextmanager
    def _control_operation(self, operation: str, deadline: datetime | None):
        timeout = self._remaining_seconds(deadline)
        acquired = self._control_lock.acquire() if timeout is None else self._control_lock.acquire(timeout=timeout)
        if not acquired:
            raise self._timeout_error(f"{operation} admission", backend_completed=False)
        registered = False
        try:
            with self._condition:
                self._synchronize_executor_health_locked()
                self._require_ready_locked(operation)
                self._active_controls += 1
                registered = True
            yield
        finally:
            if registered:
                with self._condition:
                    self._active_controls -= 1
                    self._condition.notify_all()
            self._control_lock.release()

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

    @staticmethod
    def _remaining_seconds(deadline: datetime | None) -> float | None:
        if deadline is None:
            return None
        return max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())

    def _raise_if_expired(self, deadline: datetime | None, *, phase: str, backend_completed: bool) -> None:
        if deadline is not None and datetime.now(timezone.utc) >= deadline:
            raise self._timeout_error(phase, backend_completed=backend_completed)

    def _timeout_error(self, phase: str, *, backend_completed: bool) -> PipelineTimeoutError:
        return PipelineTimeoutError(
            f"pipeline {self.pipeline_id!r} request deadline expired during {phase}",
            pipeline_id=self.pipeline_id,
            phase=phase,
            backend_completed=backend_completed,
            cancellation_supported=False,
        )

    @staticmethod
    def _request_id(request: object) -> str:
        request_id = getattr(request, "request_id", None)
        if not isinstance(request_id, str) or not request_id:
            raise PipelineConfigurationError("pipeline requests must expose a non-empty request_id")
        return request_id

    @staticmethod
    def _execution_error(exc: Exception, request_id: str) -> ExecutionError:
        code = str(getattr(exc, "code", "execution_failed"))
        details = getattr(exc, "details", {})
        if not isinstance(details, Mapping):
            details = {}
        return ExecutionError(
            code=code,
            request_id=request_id,
            stage=str(getattr(exc, "stage", details.get("phase", "pipeline"))),
            message=str(exc),
            recoverable=bool(getattr(exc, "recoverable", False)),
            details=details,
            cause=exc,
        )

    def _synchronize_executor_health_locked(self, health: BackendHealth | None = None) -> None:
        state = self._state_machine.state
        if state not in {PipelineState.READY, PipelineState.RESETTING, PipelineState.DEGRADED}:
            return
        current = health or self._executor.health()
        if current.ready:
            if state is PipelineState.DEGRADED:
                self._state_machine.transition(PipelineState.READY)
            return
        self._transition_from_executor_health_locked(current)

    def _transition_from_executor_health_locked(self, health: BackendHealth) -> None:
        target = (
            PipelineState.DEGRADED
            if health.state in {BackendState.DEGRADED, BackendState.RECOVERING}
            else PipelineState.FAILED
        )
        if self._state_machine.state is not target:
            self._state_machine.transition(target)

    def _require_ready_locked(self, operation: str) -> None:
        if self._state_machine.state is not PipelineState.READY:
            raise PipelineNotReadyError(
                f"pipeline {self.pipeline_id!r} cannot {operation} while state is {self._state_machine.state.value}",
                pipeline_id=self.pipeline_id,
                state=self._state_machine.state.value,
            )

    def _deployment_identity(self) -> DeploymentIdentity:
        manifest = self._context.validated_manifest.manifest
        deployment = self._context.deployment
        return DeploymentIdentity(
            bundle=manifest.bundle.name,
            bundle_uuid=manifest.bundle.uuid,
            bundle_revision=manifest.bundle.revision,
            deployment=self._context.deployment_name,
            deployment_uuid=deployment.uuid,
            deployment_revision=deployment.revision,
            deployment_fingerprint=self._context.deployment_fingerprint,
            backend=deployment.backend,
        )

    def _with_pipeline_latency(self, result: object, total_ms: float) -> object:
        if not isinstance(result, NamedTensorResult):
            return result
        latency = replace(result.latency, total_ms=total_ms)
        return replace(
            result,
            latency=latency,
            metadata={**result.metadata, "pipeline_id": self.pipeline_id, "runtime_state": self.state.value},
        )

    @staticmethod
    def _result_latency(result: object, total_ms: float) -> RuntimeLatency:
        if isinstance(result, NamedTensorResult):
            return result.latency
        return RuntimeLatency(total_ms=total_ms, backend_ms=total_ms)


class GenericModelPipeline:
    """Public model-neutral facade over the sole pipeline runtime core."""

    def __init__(
        self,
        pipeline_id: str,
        runtime_context: RuntimeContext,
        executor: ModelExecutor,
        *,
        request_timeout: float | None = None,
    ) -> None:
        self._core = PipelineRuntimeCore(
            pipeline_id,
            runtime_context,
            executor,
            request_timeout=request_timeout,
        )

    @property
    def pipeline_id(self) -> str:
        return self._core.pipeline_id

    @property
    def runtime_context(self) -> RuntimeContext:
        return self._core.runtime_context

    @property
    def state(self) -> PipelineState:
        return self._core.state

    def load(self) -> None:
        self._core.load()

    def execute(self, request: object, *, deadline: datetime | None = None) -> object:
        return self._core.execute(request, deadline=deadline)

    def reset(self, deadline: datetime | None = None) -> None:
        self._core.reset(deadline)

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        self._core.cancel(request_id, deadline)

    def diagnostics(self) -> PipelineRuntimeDiagnostics:
        return self._core.diagnostics()

    def health(self) -> PipelineRuntimeDiagnostics:
        return self._core.health()

    def close(self) -> None:
        self._core.close()
