"""Hardware-independent backend contracts and immutable runtime values."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from inference_manifest import (
    CompiledDeployment,
    Deployment,
    DeploymentTarget,
    PolicyMetadata,
    ValidatedManifest,
    resolve_bundle_file,
)


def _immutable_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


class BackendState(str, Enum):
    CREATED = "created"
    LOADING = "loading"
    READY = "ready"
    DEGRADED = "degraded"
    RECOVERING = "recovering"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass(frozen=True)
class BackendAdmissionEvidence:
    """Evidence gates for concurrency declarations above conservative defaults."""

    overlapping_calls: bool = False
    output_isolation: bool = False
    failure_isolation: bool = False
    deterministic_cleanup: bool = False
    sdk_initialization: bool = False
    multi_instance_execution: bool = False
    independent_close: bool = False


@dataclass(frozen=True)
class BackendCapabilities:
    resettable: bool = False
    stateful: bool = False
    thread_safe: bool = False
    max_in_flight_per_instance: int = 1
    supports_multiple_instances: bool = False
    resource_domain: str | None = None
    max_in_flight_per_resource_domain: int | None = None
    supports_attention: bool = False
    supports_cancellation: bool = False
    admission_evidence: BackendAdmissionEvidence | None = None

    def __post_init__(self) -> None:
        if self.max_in_flight_per_instance < 1:
            raise ValueError("max_in_flight_per_instance must be at least one")
        if self.max_in_flight_per_instance > 1:
            if not self.thread_safe:
                raise ValueError("max_in_flight_per_instance greater than one requires thread_safe=true")
            self._require_evidence(
                "max_in_flight_per_instance greater than one",
                "overlapping_calls",
                "output_isolation",
                "failure_isolation",
                "deterministic_cleanup",
            )

        if self.resource_domain is not None and not self.resource_domain.strip():
            raise ValueError("resource_domain must be a non-empty string when provided")
        if self.resource_domain is None and self.max_in_flight_per_resource_domain is not None:
            raise ValueError("max_in_flight_per_resource_domain requires resource_domain")
        if self.max_in_flight_per_resource_domain is not None and self.max_in_flight_per_resource_domain < 1:
            raise ValueError("max_in_flight_per_resource_domain must be at least one")

        if self.supports_multiple_instances or self.resource_domain_limit > 1:
            if self.resource_domain_limit > 1 and not self.supports_multiple_instances:
                raise ValueError("a resource-domain limit greater than one requires supports_multiple_instances=true")
            self._require_evidence(
                "multiple backend instances",
                "sdk_initialization",
                "multi_instance_execution",
                "failure_isolation",
                "independent_close",
            )

    @property
    def resource_domain_limit(self) -> int:
        if self.resource_domain is None:
            return 1
        return self.max_in_flight_per_resource_domain or 1

    def _require_evidence(self, declaration: str, *fields: str) -> None:
        missing = [field_name for field_name in fields if not getattr(self.admission_evidence, field_name, False)]
        if missing:
            raise ValueError(f"{declaration} requires conformance evidence for: {', '.join(missing)}")


@dataclass(frozen=True)
class RuntimeContext:
    """Validated manifest selection and local operational options for a backend."""

    validated_manifest: ValidatedManifest
    runtime_options: Mapping[str, object] = field(default_factory=dict)
    resolved_artifacts: Mapping[str, Path] = field(init=False)

    def __post_init__(self) -> None:
        deployment = self.validated_manifest.deployment
        artifacts: dict[str, Path] = {}
        if isinstance(deployment, CompiledDeployment):
            for role, artifact in deployment.artifacts.items():
                artifacts[role] = resolve_bundle_file(self.validated_manifest.bundle_root, artifact.path)
        object.__setattr__(self, "runtime_options", _immutable_mapping(self.runtime_options))
        object.__setattr__(self, "resolved_artifacts", MappingProxyType(artifacts))

    @property
    def deployment(self) -> Deployment:
        return self.validated_manifest.deployment

    @property
    def deployment_name(self) -> str:
        return self.validated_manifest.deployment_name

    @property
    def policy(self) -> PolicyMetadata:
        return self.validated_manifest.policy

    @property
    def target(self) -> DeploymentTarget | None:
        deployment = self.deployment
        return deployment.target if isinstance(deployment, CompiledDeployment) else None

    @property
    def deployment_fingerprint(self) -> str:
        return self.validated_manifest.fingerprint


@dataclass(frozen=True)
class InferenceRequest:
    request_id: str
    inputs: Mapping[str, object]
    prompt: str | None = None
    deadline: datetime | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        object.__setattr__(self, "inputs", _immutable_mapping(self.inputs))
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class BackendResult:
    action: object
    actual_chunk_size: int
    backend_latency_ms: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.actual_chunk_size < 1:
            raise ValueError("actual_chunk_size must be at least one")
        if not math.isfinite(self.backend_latency_ms) or self.backend_latency_ms < 0:
            raise ValueError("backend_latency_ms must be finite and non-negative")
        object.__setattr__(self, "metadata", _immutable_mapping(self.metadata))


@dataclass(frozen=True)
class BackendHealth:
    state: BackendState
    ready: bool
    reason_code: str | None = None
    message: str | None = None
    recoverable: bool = False
    last_successful_inference_time: datetime | None = None
    failure_count: int = 0

    def __post_init__(self) -> None:
        if self.ready != (self.state is BackendState.READY):
            raise ValueError("backend readiness must match the READY state")
        if self.failure_count < 0:
            raise ValueError("failure_count cannot be negative")


@runtime_checkable
class InferenceBackend(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def capabilities(self) -> BackendCapabilities: ...

    def load(self, context: RuntimeContext) -> None: ...

    def infer(self, request: InferenceRequest) -> BackendResult: ...

    def reset(self, deadline: datetime | None = None) -> None: ...

    def cancel(self, request_id: str, deadline: datetime | None = None) -> None: ...

    def health(self) -> BackendHealth: ...

    def recover(self) -> None: ...

    def close(self) -> None: ...
