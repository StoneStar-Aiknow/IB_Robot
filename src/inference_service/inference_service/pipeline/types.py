"""Dependency-light value objects returned by the inference pipeline runtime."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from inference_service.backends import BackendHealth
from inference_service.pipeline.state import PipelineState


def immutable_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class PipelineResult:
    action: object
    actual_chunk_size: int
    pipeline_id: str
    bundle: str
    bundle_uuid: str
    bundle_revision: int
    deployment: str
    deployment_uuid: str
    deployment_revision: int
    deployment_fingerprint: str
    backend: str
    state: PipelineState
    total_latency_ms: float
    preprocess_latency_ms: float
    backend_latency_ms: float
    postprocess_latency_ms: float
    raw_action: object | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.actual_chunk_size < 1:
            raise ValueError("actual_chunk_size must be at least one")
        latencies = (
            self.total_latency_ms,
            self.preprocess_latency_ms,
            self.backend_latency_ms,
            self.postprocess_latency_ms,
        )
        if any(not math.isfinite(value) or value < 0 for value in latencies):
            raise ValueError("pipeline latencies must be finite and non-negative")
        object.__setattr__(self, "metadata", immutable_mapping(self.metadata))


@dataclass(frozen=True)
class PipelineDiagnostics:
    pipeline_id: str
    bundle: str
    bundle_uuid: str
    bundle_revision: int
    deployment: str
    deployment_uuid: str
    deployment_revision: int
    deployment_fingerprint: str
    backend: str
    state: PipelineState
    backend_health: BackendHealth
    active_requests: int
    request_timeout: float | None
    default_task_configured: bool

    @property
    def ready(self) -> bool:
        return self.state is PipelineState.READY and self.backend_health.ready

    @property
    def metadata(self) -> Mapping[str, object]:
        return immutable_mapping(
            {
                "pipeline_id": self.pipeline_id,
                "bundle": self.bundle,
                "bundle_uuid": self.bundle_uuid,
                "bundle_revision": self.bundle_revision,
                "deployment": self.deployment,
                "deployment_uuid": self.deployment_uuid,
                "deployment_revision": self.deployment_revision,
                "deployment_fingerprint": self.deployment_fingerprint,
                "backend": self.backend,
                "state": self.state.value,
                "active_requests": self.active_requests,
                "request_timeout": self.request_timeout,
            }
        )
