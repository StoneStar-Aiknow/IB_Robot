"""Common lifecycle and diagnostics for named-tensor model sessions."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import contextmanager, suppress
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np

from inference_manifest import CompiledDeployment, SemanticTensor
from inference_service.backends.admission import BackendAdmission, ResourceDomainAdmissions
from inference_service.backends.errors import (
    BackendAdmissionError,
    BackendCancellationError,
    BackendCapabilityError,
    BackendError,
    BackendInferenceError,
    BackendLifecycleError,
    BackendLoadError,
    BackendNotReadyError,
)
from inference_service.backends.lifecycle import PartialLoadRollback
from inference_service.backends.types import BackendCapabilities, BackendHealth, BackendState, RuntimeContext
from inference_service.generic_runtime import NamedTensorRequest
from inference_service.unified_runtime import ModelResult, OutcomeEvidence, RuntimeLatency


class ModelSessionExecution:
    """Behavioral request scope coupled to one owning :class:`ModelSession`."""

    def __init__(self, session: ModelSession, request: NamedTensorRequest) -> None:
        self._session = session
        self.request = request

    def invoke(self, role: str, inputs: Mapping[str, object]) -> Mapping[str, object]:
        self._session._raise_if_deadline_expired(self.request.deadline)
        try:
            outputs = self._session._execute_role(role, inputs, self.request)
            self._session._validate_role_values(role, inputs, outputs)
            self._session._raise_if_deadline_expired(self.request.deadline)
        except BackendAdmissionError as exc:
            if exc.operation_started:
                raise
            raise BackendAdmissionError(str(exc), code=exc.code, operation_started=True) from exc
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, BackendError)
                else BackendInferenceError(f"model session {self._session.name!r} runtime failure: {exc}")
            )
            with self._session._condition:
                self._session._record_failure(error)
            if error is exc:
                raise
            raise error from exc
        else:
            with self._session._condition:
                self._session._last_successful_inference_time = datetime.now(timezone.utc)
            return outputs


class ModelSession(ABC):
    """Enforce lifecycle, admission, deadlines, health, and rollback for model execution."""

    def __init__(
        self,
        name: str,
        capabilities: BackendCapabilities,
        *,
        domains: ResourceDomainAdmissions | None = None,
    ) -> None:
        self._name = name
        self._capabilities = capabilities
        self._condition = threading.Condition(threading.RLock())
        self._state = BackendState.CREATED
        self._reason_code: str | None = None
        self._message: str | None = None
        self._recoverable = False
        self._failure_count = 0
        self._last_successful_inference_time: datetime | None = None
        self._active_operations = 0
        self._context: RuntimeContext | None = None
        self._control_lock = threading.Lock()
        self._admission = BackendAdmission(name, capabilities, domains=domains)

    @property
    def name(self) -> str:
        return self._name

    @property
    def capabilities(self) -> BackendCapabilities:
        return self._capabilities

    def _update_loaded_capabilities(
        self,
        *,
        priority_mapping=None,
        **changes: object,
    ) -> None:
        """Refine observational capabilities while the session is loading."""

        with self._condition:
            if self._state is not BackendState.LOADING:
                raise BackendLifecycleError(
                    f"model session {self.name!r} can update capabilities only while loading",
                    code="invalid_capability_update_state",
                )
            self._capabilities = replace(self._capabilities, priority_mapping=priority_mapping, **changes)

    @property
    def runtime_version(self) -> str:
        """Return the version reported by the runtime loaded by this session."""
        return ""

    @staticmethod
    def _runtime_version(runtime: object | None) -> str:
        if runtime is None:
            return ""
        version = getattr(runtime, "__version__", None)
        if version is not None and str(version).strip():
            return str(version).strip()
        getter = getattr(runtime, "get_version", None)
        if callable(getter):
            with suppress(Exception):
                value = getter()
                if isinstance(value, tuple):
                    value = ".".join(str(part) for part in value)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return ""

    def load(self, context: RuntimeContext) -> None:
        if not isinstance(context, RuntimeContext):
            raise TypeError("model sessions require a validated RuntimeContext")
        with self._condition:
            if self._state is not BackendState.CREATED:
                raise BackendLifecycleError(
                    f"model session {self.name!r} cannot load from state {self._state.value}",
                    code="invalid_load_state",
                )
            self._state = BackendState.LOADING

        rollback = PartialLoadRollback()
        try:
            self._load(context, rollback)
            self._context = context
            rollback.commit()
        except Exception as exc:
            rollback_errors = rollback.rollback()
            error = self._load_error(exc, rollback_errors)
            with self._condition:
                self._record_failure(error)
            if error is exc:
                raise
            raise error from exc
        else:
            with self._condition:
                self._state = BackendState.READY

    def infer(self, request: NamedTensorRequest) -> ModelResult:
        if not isinstance(request, NamedTensorRequest):
            raise TypeError("model sessions require a NamedTensorRequest")
        self._require_ready()
        self._validate_request(request)
        with self._admission.admit(request.deadline):
            with self._condition:
                self._require_ready_locked()
                self._active_operations += 1
            execution_started = False
            try:
                self._raise_if_deadline_expired(request.deadline)
                context = self._require_context()
                self._validate_values(request.inputs, context.validated_manifest.manifest.model.inputs, "input")
                started = time.perf_counter()
                execution_started = True
                outputs = self._execute(request)
                backend_ms = (time.perf_counter() - started) * 1000.0
                self._raise_if_deadline_expired(request.deadline)
                self._validate_values(outputs, context.validated_manifest.manifest.model.outputs, "output")
                result = ModelResult(
                    outputs=outputs,
                    latency=RuntimeLatency(total_ms=backend_ms, backend_ms=backend_ms),
                    evidence=OutcomeEvidence.completed("backend"),
                    metadata={
                        "request_id": request.request_id,
                        "interface": context.interface,
                        "model_type": context.model_type,
                        "operation": context.operation,
                        "deployment": context.deployment_name,
                        "deployment_fingerprint": context.deployment_fingerprint,
                        "runtime_state": BackendState.READY.value,
                    },
                )
            except Exception as exc:
                if not execution_started:
                    raise
                if isinstance(exc, BackendAdmissionError):
                    if exc.operation_started:
                        raise
                    raise BackendAdmissionError(str(exc), code=exc.code, operation_started=True) from exc
                error = (
                    exc
                    if isinstance(exc, BackendError)
                    else BackendInferenceError(f"model session {self.name!r} runtime failure: {exc}")
                )
                with self._condition:
                    self._record_failure(error)
                if error is exc:
                    raise
                raise error from exc
            else:
                with self._condition:
                    self._last_successful_inference_time = datetime.now(timezone.utc)
                return result
            finally:
                with self._condition:
                    self._active_operations -= 1
                    self._condition.notify_all()

    @contextmanager
    def execution(self, request: NamedTensorRequest):
        """Admit one request across multiple manifest role invocations."""

        if not isinstance(request, NamedTensorRequest):
            raise TypeError("model sessions require a NamedTensorRequest")
        self._require_ready()
        self._validate_request(request)
        with self._admission.admit(request.deadline):
            with self._condition:
                self._require_ready_locked()
                self._active_operations += 1
            try:
                self._raise_if_deadline_expired(request.deadline)
                context = self._require_context()
                model_inputs = context.validated_manifest.manifest.model.inputs
                if model_inputs:
                    self._validate_values(request.inputs, model_inputs, "input")
                yield ModelSessionExecution(self, request)
            finally:
                with self._condition:
                    self._active_operations -= 1
                    self._condition.notify_all()

    def reset(self, deadline: datetime | None = None) -> None:
        if not self.capabilities.resettable:
            raise BackendCapabilityError(
                f"model session {self.name!r} does not support reset",
                capability="reset",
            )
        with self._admission.exclusive(deadline), self._control_operation("reset", deadline):
            started = False
            try:
                self._raise_if_deadline_expired(deadline)
                started = True
                self._reset()
                self._raise_if_deadline_expired(deadline)
            except Exception as exc:
                if not started:
                    raise
                if isinstance(exc, BackendCapabilityError):
                    raise
                error = exc if isinstance(exc, BackendError) else BackendInferenceError(str(exc), code="reset_failed")
                with self._condition:
                    self._record_failure(error)
                if error is exc:
                    raise
                raise error from exc

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None:
        if not self.capabilities.supports_cancellation:
            raise BackendCapabilityError(
                f"model session {self.name!r} does not support cancellation",
                capability="cancellation",
            )
        if not request_id:
            raise BackendCapabilityError("cancellation requires a request ID", capability="cancellation")
        with self._control_operation("cancel", deadline):
            self._raise_if_deadline_expired(deadline)
            try:
                self._cancel(request_id)
            except BackendCapabilityError:
                raise
            except BackendCancellationError as error:
                if self.capabilities.stateful and error.operation_started and not error.outcome_known:
                    with self._condition:
                        self._record_failure(error)
                raise
            except Exception as exc:
                error = BackendCancellationError(
                    f"model session {self.name!r} cancellation outcome is unknown: {exc}",
                    operation_started=True,
                    outcome_known=False,
                )
                if self.capabilities.stateful:
                    with self._condition:
                        self._record_failure(error)
                raise error from exc
            try:
                self._raise_if_deadline_expired(deadline)
            except BackendAdmissionError as exc:
                error = BackendCancellationError(
                    "model session cancellation deadline expired during execution",
                    code=exc.code,
                    operation_started=True,
                    outcome_known=False,
                )
                if self.capabilities.stateful:
                    with self._condition:
                        self._record_failure(error)
                raise error from exc

    def recover(self) -> None:
        if not self._supports_recovery():
            raise BackendCapabilityError(
                f"model session {self.name!r} does not support recovery",
                capability="recovery",
            )
        with (
            self._admission.exclusive(),
            self._control_operation("recover", None, required_state=BackendState.DEGRADED),
        ):
            with self._condition:
                if not self._recoverable:
                    raise BackendLifecycleError(
                        f"model session {self.name!r} cannot recover from state {self._state.value}",
                        code="invalid_recovery_state",
                    )
                self._state = BackendState.RECOVERING
            try:
                self._recover()
            except Exception as exc:
                error = (
                    exc if isinstance(exc, BackendError) else BackendInferenceError(str(exc), code="recovery_failed")
                )
                with self._condition:
                    self._record_failure(error, allow_recovery=False)
                if error is exc:
                    raise
                raise error from exc
            else:
                with self._condition:
                    self._state = BackendState.READY
                    self._reason_code = "recovered"
                    self._message = "runtime recovered"
                    self._recoverable = False

    def health(self) -> BackendHealth:
        with self._condition:
            return BackendHealth(
                state=self._state,
                ready=self._state is BackendState.READY,
                reason_code=self._reason_code,
                message=self._message,
                recoverable=self._recoverable,
                last_successful_inference_time=self._last_successful_inference_time,
                failure_count=self._failure_count,
            )

    def close(self) -> None:
        with self._condition:
            if self._state is BackendState.CLOSED:
                return
            if self._state is BackendState.CLOSING:
                self._condition.wait_for(lambda: self._state is BackendState.CLOSED)
                return
            self._state = BackendState.CLOSING
            self._condition.wait_for(lambda: self._active_operations == 0)

        error: Exception | None = None
        try:
            self._close()
        except Exception as exc:
            error = exc
        finally:
            self._context = None
            self._admission.close()
            with self._condition:
                if error is not None:
                    self._failure_count += 1
                    self._reason_code = getattr(error, "code", "close_failed")
                    self._message = str(error)
                    self._recoverable = False
                self._state = BackendState.CLOSED
                self._condition.notify_all()
        if error is not None:
            raise error

    @abstractmethod
    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None: ...

    @abstractmethod
    def _execute(self, request: NamedTensorRequest) -> Mapping[str, object]: ...

    def _validate_request(self, request: NamedTensorRequest) -> None:
        del request

    def _execute_role(
        self,
        role: str,
        inputs: Mapping[str, object],
        request: NamedTensorRequest,
    ) -> Mapping[str, object]:
        raise BackendCapabilityError(
            f"model session {self.name!r} does not implement manifest role {role!r}", capability="role_execution"
        )

    @abstractmethod
    def _close(self) -> None: ...

    def _reset(self) -> None:
        raise BackendCapabilityError(f"model session {self.name!r} does not implement reset", capability="reset")

    def _cancel(self, request_id: str) -> None:
        raise BackendCapabilityError(
            f"model session {self.name!r} does not implement cancellation for request {request_id!r}",
            capability="cancellation",
        )

    def _recover(self) -> None:
        raise BackendCapabilityError(f"model session {self.name!r} does not implement recovery", capability="recovery")

    def _require_context(self) -> RuntimeContext:
        if self._context is None:
            raise BackendInferenceError("model session is not fully loaded", code="runtime_not_loaded")
        return self._context

    def _validate_role_values(
        self,
        role: str,
        inputs: Mapping[str, object],
        outputs: Mapping[str, object],
    ) -> None:
        deployment = self._require_context().deployment
        if not isinstance(deployment, CompiledDeployment):
            raise BackendInferenceError("role execution requires a compiled deployment", code="invalid_deployment")
        try:
            bindings = deployment.bindings[role]
        except KeyError as exc:
            raise BackendInferenceError(f"unknown execution role {role!r}", code="unknown_execution_role") from exc
        linked_inputs = {
            link.semantic
            for link in deployment.device_links
            if link.consumer == role and link.transport == "device_pointer"
        }
        host_inputs = tuple(binding for binding in bindings.inputs if binding.semantic not in linked_inputs)
        linked_outputs = {
            link.semantic
            for link in deployment.device_links
            if link.producer == role and link.transport == "device_pointer"
        }
        host_outputs = tuple(binding for binding in bindings.outputs if binding.semantic not in linked_outputs)
        self._validate_values(inputs, host_inputs, f"role_{role}_input")
        self._validate_values(outputs, host_outputs, f"role_{role}_output")

    def _require_ready(self) -> None:
        with self._condition:
            self._require_ready_locked()

    def _require_ready_locked(self) -> None:
        if self._state is not BackendState.READY:
            raise BackendNotReadyError(f"model session {self.name!r} is {self._state.value}")

    def _record_failure(self, error: BackendError, *, allow_recovery: bool = True) -> None:
        self._failure_count += 1
        self._reason_code = error.code
        self._message = str(error)
        self._recoverable = allow_recovery and error.recoverable and self._supports_recovery()
        self._state = BackendState.DEGRADED if self._recoverable else BackendState.FAILED

    def _supports_recovery(self) -> bool:
        return type(self)._recover is not ModelSession._recover

    @contextmanager
    def _control_operation(
        self,
        operation: str,
        deadline: datetime | None,
        *,
        required_state: BackendState = BackendState.READY,
    ):
        timeout = None
        if deadline is not None:
            now = datetime.now(deadline.tzinfo) if deadline.tzinfo is not None else datetime.now()
            timeout = max(0.0, (deadline - now).total_seconds())
        acquired = self._control_lock.acquire() if timeout is None else self._control_lock.acquire(timeout=timeout)
        if not acquired:
            raise BackendAdmissionError(
                f"model session {operation} deadline expired waiting for control admission",
                code="deadline_exceeded",
            )
        registered = False
        try:
            with self._condition:
                if self._state is not required_state:
                    raise BackendNotReadyError(
                        f"model session {self.name!r} cannot {operation} while state is {self._state.value}"
                    )
                self._active_operations += 1
                registered = True
            yield
        finally:
            if registered:
                with self._condition:
                    self._active_operations -= 1
                    self._condition.notify_all()
            self._control_lock.release()

    def _load_error(self, exc: Exception, rollback_errors: tuple[Exception, ...]) -> BackendError:
        suffix = ""
        if rollback_errors:
            suffix = "; rollback errors: " + "; ".join(str(error) for error in rollback_errors)
        if isinstance(exc, BackendError) and not suffix:
            return exc
        code = exc.code if isinstance(exc, BackendError) else "load_failed"
        return BackendLoadError(f"model session {self.name!r} failed to load: {exc}{suffix}", code=code)

    @staticmethod
    def _raise_if_deadline_expired(deadline: datetime | None) -> None:
        if deadline is None:
            return
        now = datetime.now(deadline.tzinfo) if deadline.tzinfo is not None else datetime.now()
        if now >= deadline:
            raise BackendAdmissionError("model request deadline expired", code="deadline_exceeded")

    @staticmethod
    def _validate_values(values: Mapping[str, object], descriptors: tuple[SemanticTensor, ...], direction: str) -> None:
        expected = {descriptor.semantic: descriptor for descriptor in descriptors}
        missing = sorted(set(expected) - set(values))
        unexpected = sorted(set(values) - set(expected))
        if missing or unexpected:
            raise BackendInferenceError(
                f"named {direction} semantics mismatch: missing={missing}, unexpected={unexpected}",
                code=f"{direction}_semantic_mismatch",
            )
        for semantic, descriptor in expected.items():
            value = values[semantic]
            try:
                array = np.asarray(value)
            except Exception as exc:
                raise BackendInferenceError(
                    f"{direction} {semantic!r} is not tensor-like: {exc}", code=f"invalid_{direction}_tensor"
                ) from exc
            expected_dtype = np.dtype(descriptor.dtype)
            if array.dtype != expected_dtype:
                raise BackendInferenceError(
                    f"{direction} {semantic!r} dtype {array.dtype} does not match {expected_dtype}",
                    code=f"{direction}_dtype_mismatch",
                )
            if len(array.shape) != len(descriptor.shape) or any(
                declared != -1 and declared != actual
                for declared, actual in zip(descriptor.shape, array.shape, strict=True)
            ):
                raise BackendInferenceError(
                    f"{direction} {semantic!r} shape {array.shape} does not match {descriptor.shape}",
                    code=f"{direction}_shape_mismatch",
                )
