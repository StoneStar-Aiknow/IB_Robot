"""Structured errors for backend construction, lifecycle, and execution."""

from __future__ import annotations


class BackendError(RuntimeError):
    """Base error carrying a stable reason code and recoverability flag."""

    def __init__(self, message: str, *, code: str, recoverable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.recoverable = recoverable


class BackendLifecycleError(BackendError):
    def __init__(self, message: str, *, code: str = "invalid_lifecycle") -> None:
        super().__init__(message, code=code)


class BackendNotReadyError(BackendLifecycleError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="not_ready")


class BackendCapabilityError(BackendError):
    def __init__(self, message: str, *, capability: str) -> None:
        super().__init__(message, code="unsupported_capability")
        self.capability = capability


class BackendLoadError(BackendError):
    def __init__(self, message: str, *, code: str = "load_failed", recoverable: bool = False) -> None:
        super().__init__(message, code=code, recoverable=recoverable)


class BackendInferenceError(BackendError):
    def __init__(self, message: str, *, code: str = "inference_failed", recoverable: bool = False) -> None:
        super().__init__(message, code=code, recoverable=recoverable)


class BackendAdmissionError(BackendError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "admission_rejected",
        operation_started: bool = False,
    ) -> None:
        super().__init__(message, code=code)
        self.operation_started = operation_started


class BackendCancellationError(BackendError):
    """Cancellation failed after admission, with explicit outcome certainty."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "cancel_failed",
        operation_started: bool,
        outcome_known: bool,
    ) -> None:
        super().__init__(message, code=code)
        self.operation_started = operation_started
        self.outcome_known = outcome_known


class BackendRegistryError(BackendError):
    def __init__(self, message: str, *, code: str = "registry_error") -> None:
        super().__init__(message, code=code)


class BackendCompatibilityError(BackendRegistryError):
    def __init__(self, message: str, *, code: str = "unsupported_backend_selection") -> None:
        super().__init__(message, code=code)
