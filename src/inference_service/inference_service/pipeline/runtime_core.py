"""Small native stage primitives used by policy runtime assemblers."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from inference_service.backends.types import BackendHealth, RuntimeContext
from inference_service.codecs import ExecutionFrame, ExecutionPlan
from inference_service.pipeline.errors import PipelineCanceledError
from inference_service.unified_runtime import ExecutionContext, ModelRequest, RuntimeLatency


@runtime_checkable
class ModelExecutor(Protocol):
    """Native runtime executor protocol consumed by ``ModelRuntimeHandle``."""

    def load(self, context: RuntimeContext) -> None: ...

    def execute(self, request: ModelRequest, context: ExecutionContext) -> object: ...

    def cancel(self, request_id: str, deadline=None) -> None: ...

    def health(self) -> BackendHealth: ...

    def reset(self, deadline=None) -> None: ...

    def close(self) -> None: ...


class ExecutionControl:
    """Stage-facing view of the native cancellation token."""

    def __init__(self, request_id: str, cancellation_token=None) -> None:
        if not request_id:
            raise ValueError("execution control request_id must be non-empty")
        self.request_id = request_id
        self._cancellation_token = cancellation_token
        self._canceled = threading.Event()

    @property
    def cancellation_requested(self) -> bool:
        return self._canceled.is_set() or bool(
            self._cancellation_token is not None and self._cancellation_token.cancelled
        )

    def cancel(self) -> None:
        self._canceled.set()
        if self._cancellation_token is not None:
            self._cancellation_token.cancel("stage cancellation requested")

    def raise_if_canceled(self, stage: str) -> None:
        if self.cancellation_requested:
            raise PipelineCanceledError(self.request_id, stage=stage)


@dataclass(frozen=True)
class ExecutionError:
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
    """Request-local values plus the optional manifest execution frame."""

    def __init__(
        self,
        request: ModelRequest,
        *,
        execution_plan: ExecutionPlan | None = None,
        values: Mapping[str, object] | None = None,
        control: ExecutionControl | None = None,
    ) -> None:
        if not isinstance(request, ModelRequest):
            raise TypeError("StageFrame requires a ModelRequest")
        self.request = request
        self.execution_plan = execution_plan
        self.execution_frame = ExecutionFrame(execution_plan) if execution_plan is not None else None
        self.values: dict[str, object] = dict(values or {})
        self.control = control or ExecutionControl(request_id="stage")
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        if self.execution_frame is not None:
            self.execution_frame.close()
        self.values.clear()
        self._closed = True


@dataclass(frozen=True)
class PipelineRuntimeDiagnostics:
    """Compatibility-free diagnostic value for callers that expose pipeline health."""

    pipeline_id: str
    deployment: object
    state: object
    executor_health: BackendHealth
    active_requests: int
    request_timeout: float | None
    last_latency: RuntimeLatency | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def ready(self) -> bool:
        return bool(getattr(self.executor_health, "ready", False))


__all__ = ["ExecutionControl", "ExecutionError", "ModelExecutor", "PipelineRuntimeDiagnostics", "StageFrame"]
