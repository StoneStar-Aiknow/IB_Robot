"""Model-neutral request and result contracts for named semantic tensors.

Policy backends keep using :class:`InferenceRequest` and :class:`BackendResult`.
These values provide an additive boundary for non-policy model sessions and a
small adapter back to the existing policy result contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

from inference_service.backends import BackendResult, BackendState, InferenceRequest


def _immutable_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class DeploymentIdentity:
    """Stable identity for the selected model deployment."""

    bundle: str
    bundle_uuid: str
    bundle_revision: int
    deployment: str
    deployment_uuid: str
    deployment_revision: int
    deployment_fingerprint: str
    backend: str

    def __post_init__(self) -> None:
        for name in (
            "bundle",
            "bundle_uuid",
            "deployment",
            "deployment_uuid",
            "deployment_fingerprint",
            "backend",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        if self.bundle_revision < 0 or self.deployment_revision < 0:
            raise ValueError("deployment revisions cannot be negative")


@dataclass(frozen=True)
class RuntimeLatency:
    """Measured execution phases in milliseconds."""

    total_ms: float
    backend_ms: float
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0

    def __post_init__(self) -> None:
        values = (self.total_ms, self.backend_ms, self.preprocess_ms, self.postprocess_ms)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("runtime latencies must be finite and non-negative")


@dataclass(frozen=True)
class RuntimeErrorInfo:
    """Stable, serializable failure information for a model request."""

    code: str
    message: str
    recoverable: bool = False
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("error code must be non-empty")
        if not self.message:
            raise ValueError("error message must be non-empty")
        object.__setattr__(self, "details", _immutable_mapping(self.details))


@dataclass(frozen=True)
class NamedTensorRequest:
    """Immutable generic request carrying named semantic input values."""

    request_id: str
    inputs: Mapping[str, object]
    deadline: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    priority: int = 0

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.inputs:
            raise ValueError("inputs must contain at least one named value")
        if any(not isinstance(name, str) or not name for name in self.inputs):
            raise ValueError("input names must be non-empty strings")
        if isinstance(self.priority, bool) or not isinstance(self.priority, int) or self.priority < 0:
            raise ValueError("priority must be a non-negative integer")
        object.__setattr__(self, "inputs", _immutable_mapping(self.inputs))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    def to_inference_request(self) -> InferenceRequest:
        """Adapt to the existing backend request without changing its contract."""

        return InferenceRequest(
            request_id=self.request_id,
            inputs=self.inputs,
            deadline=self.deadline,
            metadata=self.metadata,
            priority=self.priority,
        )


@dataclass(frozen=True)
class NamedTensorResult:
    """Immutable generic result with named semantic outputs and diagnostics."""

    outputs: Mapping[str, object]
    deployment: DeploymentIdentity
    latency: RuntimeLatency
    state: BackendState = BackendState.READY
    metadata: Mapping[str, object] = field(default_factory=dict)
    error: RuntimeErrorInfo | None = None

    def __post_init__(self) -> None:
        if not self.outputs and self.error is None:
            raise ValueError("successful results must contain at least one named output")
        if any(not isinstance(name, str) or not name for name in self.outputs):
            raise ValueError("output names must be non-empty strings")
        if self.error is None and self.state is not BackendState.READY:
            raise ValueError("successful results must be in the READY state")
        object.__setattr__(self, "outputs", _immutable_mapping(self.outputs))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))

    @property
    def ready(self) -> bool:
        return self.error is None and self.state is BackendState.READY

    def to_backend_result(self, *, output: str = "action", actual_chunk_size: int = 1) -> BackendResult:
        """Convert one named output to the legacy policy result shape."""

        if self.error is not None:
            raise ValueError(f"cannot convert failed result: {self.error.code}")
        if output not in self.outputs:
            raise KeyError(f"named output {output!r} is not present")
        return BackendResult(
            action=self.outputs[output],
            actual_chunk_size=actual_chunk_size,
            backend_latency_ms=self.latency.backend_ms,
            metadata={
                **self.metadata,
                "deployment": self.deployment.deployment,
                "deployment_fingerprint": self.deployment.deployment_fingerprint,
                "runtime_state": self.state.value,
                "latency_ms": {
                    "total": self.latency.total_ms,
                    "backend": self.latency.backend_ms,
                    "preprocess": self.latency.preprocess_ms,
                    "postprocess": self.latency.postprocess_ms,
                },
            },
        )
