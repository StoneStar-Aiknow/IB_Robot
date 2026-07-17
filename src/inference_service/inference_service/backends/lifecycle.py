"""Reusable backend lifecycle, rollback, health, and admission enforcement."""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timezone

from inference_service.backends.admission import BackendAdmission, ResourceDomainAdmissions
from inference_service.backends.errors import (
    BackendCapabilityError,
    BackendError,
    BackendInferenceError,
    BackendLifecycleError,
    BackendLoadError,
    BackendNotReadyError,
)
from inference_service.backends.types import (
    BackendCapabilities,
    BackendHealth,
    BackendResult,
    BackendState,
    InferenceRequest,
    RuntimeContext,
)


class PartialLoadRollback:
    """Run registered partial-load cleanup callbacks once in reverse order."""

    def __init__(self) -> None:
        self._callbacks: list[tuple[Callable[..., object], tuple[object, ...], dict[str, object]]] = []
        self._finished = False
        self._lock = threading.Lock()

    def defer(self, callback: Callable[..., object], /, *args: object, **kwargs: object) -> None:
        with self._lock:
            if self._finished:
                raise BackendLifecycleError("cannot register rollback after it has finished")
            self._callbacks.append((callback, args, kwargs))

    def commit(self) -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
            self._callbacks.clear()

    def rollback(self) -> tuple[Exception, ...]:
        with self._lock:
            if self._finished:
                return ()
            self._finished = True
            callbacks = tuple(reversed(self._callbacks))
            self._callbacks.clear()

        errors: list[Exception] = []
        for callback, args, kwargs in callbacks:
            try:
                callback(*args, **kwargs)
            except Exception as exc:  # Cleanup must continue through every registered resource.
                errors.append(exc)
        return tuple(errors)


class LifecycleBackend(ABC):
    """Base backend enforcing the common lifecycle and admission contract."""

    def __init__(
        self,
        name: str,
        capabilities: BackendCapabilities,
        *,
        domains: ResourceDomainAdmissions | None = None,
    ) -> None:
        if not name:
            raise ValueError("backend name must be non-empty")
        self._name = name
        self._capabilities = capabilities
        self._condition = threading.Condition(threading.RLock())
        self._state = BackendState.CREATED
        self._reason_code: str | None = None
        self._message: str | None = None
        self._recoverable = False
        self._last_successful_inference_time: datetime | None = None
        self._failure_count = 0
        self._loading = False
        self._active_operations = 0
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
        resettable: bool,
        stateful: bool,
        supports_attention: bool,
        supports_cancellation: bool = False,
    ) -> None:
        """Refine observational capabilities during load without changing admission."""

        with self._condition:
            if self._state is not BackendState.LOADING:
                raise BackendLifecycleError(
                    f"backend {self.name!r} can update loaded capabilities only while loading",
                    code="invalid_capability_update_state",
                )
            self._capabilities = replace(
                self._capabilities,
                resettable=resettable,
                stateful=stateful,
                supports_attention=supports_attention,
                supports_cancellation=supports_cancellation,
            )

    def load(self, context: RuntimeContext) -> None:
        with self._condition:
            if self._state is not BackendState.CREATED:
                raise BackendLifecycleError(
                    f"backend {self.name!r} cannot load from state {self._state.value}",
                    code="invalid_load_state",
                )
            self._loading = True
            self._transition(BackendState.LOADING)

        rollback = PartialLoadRollback()
        try:
            self._load(context, rollback)
            with self._condition:
                if self._state is not BackendState.LOADING:
                    raise BackendLifecycleError(
                        f"backend {self.name!r} load was interrupted by state {self._state.value}",
                        code="load_interrupted",
                    )
                rollback.commit()
                self._transition(BackendState.READY)
        except Exception as exc:
            rollback_errors = rollback.rollback()
            error = self._load_error(exc, rollback_errors)
            with self._condition:
                if self._state is not BackendState.CLOSING:
                    self._record_failure(error, BackendState.FAILED)
            if error is exc:
                raise
            raise error from exc
        finally:
            with self._condition:
                self._loading = False
                self._condition.notify_all()

    def infer(self, request: InferenceRequest) -> BackendResult:
        self._require_ready("infer")
        with self._admission.admit(request.deadline):
            with self._condition:
                self._require_ready_locked("infer")
                self._active_operations += 1
            try:
                result = self._infer(request)
                if not isinstance(result, BackendResult):
                    raise BackendInferenceError(
                        f"backend {self.name!r} returned {type(result).__name__}, expected BackendResult",
                        code="invalid_backend_result",
                    )
            except Exception as exc:
                error = self._inference_error(exc)
                self._handle_runtime_failure(error)
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

    def reset(self) -> None:
        if not self.capabilities.resettable:
            raise BackendCapabilityError(
                f"backend {self.name!r} does not support reset",
                capability="reset",
            )
        self._require_ready("reset")
        with self._admission.exclusive():
            with self._condition:
                self._require_ready_locked("reset")
                self._active_operations += 1
            try:
                self._reset()
            except Exception as exc:
                error = self._inference_error(exc, code="reset_failed")
                self._handle_runtime_failure(error)
                if error is exc:
                    raise
                raise error from exc
            finally:
                with self._condition:
                    self._active_operations -= 1
                    self._condition.notify_all()

    def cancel(self, request_id: str) -> None:
        if not self.capabilities.supports_cancellation:
            raise BackendCapabilityError(
                f"backend {self.name!r} does not support cancellation",
                capability="cancellation",
            )
        if not request_id:
            raise BackendCapabilityError("cancellation requires a request ID", capability="cancellation")
        self._require_ready("cancel")
        self._cancel(request_id)

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
            self._transition(BackendState.CLOSING)
            self._condition.wait_for(lambda: not self._loading and self._active_operations == 0)

        close_error: Exception | None = None
        try:
            self._close()
        except Exception as exc:  # The backend is still terminal and admission must still close.
            close_error = exc
        finally:
            self._admission.close()
            with self._condition:
                if close_error is not None:
                    self._failure_count += 1
                    self._reason_code = getattr(close_error, "code", "close_failed")
                    self._message = str(close_error)
                    self._recoverable = False
                self._transition(BackendState.CLOSED, preserve_reason=close_error is not None)
                self._condition.notify_all()
        if close_error is not None:
            raise close_error

    def report_runtime_failure(self, error: BackendError) -> None:
        """Move an asynchronously failed runtime out of READY without invoking recovery."""

        with self._condition:
            if self._state is not BackendState.READY:
                raise BackendLifecycleError(
                    f"backend {self.name!r} cannot report runtime failure from state {self._state.value}",
                    code="invalid_failure_state",
                )
            state = BackendState.DEGRADED if error.recoverable else BackendState.FAILED
            self._record_failure(error, state)

    def recover(self) -> None:
        """Run the backend recovery hook after an asynchronously reported recoverable failure."""

        with self._admission.exclusive():
            with self._condition:
                if self._state is not BackendState.DEGRADED or not self._recoverable:
                    raise BackendLifecycleError(
                        f"backend {self.name!r} cannot recover from state {self._state.value}",
                        code="invalid_recovery_state",
                    )
                self._transition(BackendState.RECOVERING, preserve_reason=True)
                self._active_operations += 1
            try:
                self._recover()
            except Exception as exc:
                error = self._inference_error(exc, code="recovery_failed")
                with self._condition:
                    if self._state not in {BackendState.CLOSING, BackendState.CLOSED}:
                        self._record_failure(error, BackendState.FAILED)
                if error is exc:
                    raise
                raise error from exc
            else:
                with self._condition:
                    if self._state is BackendState.RECOVERING:
                        self._transition(BackendState.READY, reason_code="recovered", message="runtime recovered")
            finally:
                with self._condition:
                    self._active_operations -= 1
                    self._condition.notify_all()

    @abstractmethod
    def _load(self, context: RuntimeContext, rollback: PartialLoadRollback) -> None:
        """Allocate and validate runtime resources, registering partial cleanup in rollback."""

    @abstractmethod
    def _infer(self, request: InferenceRequest) -> BackendResult:
        """Execute one admitted request."""

    @abstractmethod
    def _close(self) -> None:
        """Release all fully loaded resources."""

    def _reset(self) -> None:
        raise BackendCapabilityError(f"backend {self.name!r} does not implement reset", capability="reset")

    def _recover(self) -> None:
        raise BackendCapabilityError(f"backend {self.name!r} does not implement recovery", capability="recovery")

    def _cancel(self, request_id: str) -> None:
        raise BackendCapabilityError(
            f"backend {self.name!r} does not implement cancellation for request {request_id!r}",
            capability="cancellation",
        )

    def _handle_runtime_failure(self, error: BackendError) -> None:
        with self._condition:
            if self._state in {BackendState.CLOSING, BackendState.CLOSED}:
                return
            state = BackendState.DEGRADED if error.recoverable else BackendState.FAILED
            self._record_failure(error, state)

    def _record_failure(self, error: BackendError, state: BackendState) -> None:
        self._failure_count += 1
        self._transition(
            state,
            reason_code=error.code,
            message=str(error),
            recoverable=error.recoverable,
        )

    def _transition(
        self,
        state: BackendState,
        *,
        reason_code: str | None = None,
        message: str | None = None,
        recoverable: bool = False,
        preserve_reason: bool = False,
    ) -> None:
        self._state = state
        if not preserve_reason:
            self._reason_code = reason_code
            self._message = message
            self._recoverable = recoverable

    def _require_ready(self, operation: str) -> None:
        with self._condition:
            self._require_ready_locked(operation)

    def _require_ready_locked(self, operation: str) -> None:
        if self._state is not BackendState.READY:
            raise BackendNotReadyError(f"backend {self.name!r} cannot {operation} while state is {self._state.value}")

    def _load_error(self, exc: Exception, rollback_errors: tuple[Exception, ...]) -> BackendError:
        suffix = ""
        if rollback_errors:
            suffix = "; rollback errors: " + "; ".join(str(error) for error in rollback_errors)
        if isinstance(exc, BackendError):
            if not suffix:
                return exc
            return BackendLoadError(
                f"{exc}{suffix}",
                code=exc.code,
                recoverable=exc.recoverable,
            )
        return BackendLoadError(f"backend {self.name!r} failed to load: {exc}{suffix}")

    def _inference_error(self, exc: Exception, *, code: str = "inference_failed") -> BackendError:
        if isinstance(exc, BackendError):
            return exc
        return BackendInferenceError(f"backend {self.name!r} runtime failure: {exc}", code=code)
