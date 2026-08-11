"""Structured errors for pipeline lifecycle, routing, and result validation."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType


class PipelineError(RuntimeError):
    """Base pipeline error with a stable code and immutable diagnostic details."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        pipeline_id: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.pipeline_id = pipeline_id
        self.details = MappingProxyType(dict(details or {}))


class PipelineLifecycleError(PipelineError):
    def __init__(
        self,
        message: str,
        *,
        pipeline_id: str | None = None,
        code: str = "invalid_lifecycle",
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, pipeline_id=pipeline_id, details=details)


class PipelineTransitionError(PipelineLifecycleError):
    def __init__(self, message: str, *, details: Mapping[str, object] | None = None) -> None:
        super().__init__(message, code="illegal_transition", details=details)


class PipelineNotReadyError(PipelineLifecycleError):
    def __init__(self, message: str, *, pipeline_id: str, state: str) -> None:
        super().__init__(
            message,
            code="not_ready",
            pipeline_id=pipeline_id,
            details={"state": state},
        )


class PipelineConfigurationError(PipelineError):
    def __init__(
        self,
        message: str,
        *,
        pipeline_id: str | None = None,
        code: str = "invalid_configuration",
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, pipeline_id=pipeline_id, details=details)


class PipelineValidationError(PipelineError):
    def __init__(
        self,
        message: str,
        *,
        pipeline_id: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="invalid_backend_result",
            pipeline_id=pipeline_id,
            details=details,
        )


class PipelineTimeoutError(PipelineError):
    def __init__(
        self,
        message: str,
        *,
        pipeline_id: str,
        phase: str,
        backend_completed: bool,
        cancellation_supported: bool,
    ) -> None:
        super().__init__(
            message,
            code="deadline_exceeded",
            pipeline_id=pipeline_id,
            details={
                "phase": phase,
                "backend_completed": backend_completed,
                "cancellation_supported": cancellation_supported,
                "timeout_mode": "cooperative_deadline_no_detached_threads",
            },
        )


class PipelineCanceledError(PipelineError):
    def __init__(self, request_id: str, *, stage: str = "pipeline") -> None:
        super().__init__(
            f"inference request {request_id!r} was canceled",
            code="request_canceled",
            details={"request_id": request_id, "stage": stage},
        )
        self.request_id = request_id
        self.stage = stage
        self.recoverable = True


class PipelineNotFoundError(PipelineError):
    def __init__(self, pipeline_id: str, available: tuple[str, ...]) -> None:
        super().__init__(
            f"pipeline {pipeline_id!r} is not configured; available pipelines: {list(available)}",
            code="pipeline_not_found",
            pipeline_id=pipeline_id,
            details={"available_pipelines": available},
        )


class PipelineManagerError(PipelineError):
    def __init__(
        self,
        message: str,
        *,
        code: str,
        pipeline_id: str | None = None,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, pipeline_id=pipeline_id, details=details)
