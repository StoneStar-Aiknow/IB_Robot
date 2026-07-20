"""Common lifecycle contracts and lazy registry for inference backends."""

from inference_service.backends.admission import BackendAdmission, ResourceDomainAdmissions
from inference_service.backends.errors import (
    BackendAdmissionError,
    BackendCancellationError,
    BackendCapabilityError,
    BackendCompatibilityError,
    BackendError,
    BackendInferenceError,
    BackendLifecycleError,
    BackendLoadError,
    BackendNotReadyError,
    BackendRegistryError,
)
from inference_service.backends.lifecycle import LifecycleBackend, PartialLoadRollback
from inference_service.backends.registry import (
    BACKEND_REGISTRY,
    CANONICAL_BACKENDS,
    STATIC_BACKEND_DESCRIPTORS,
    BackendDescriptor,
    BackendRegistry,
)
from inference_service.backends.types import (
    BackendAdmissionEvidence,
    BackendCapabilities,
    BackendHealth,
    BackendResult,
    BackendState,
    InferenceBackend,
    InferenceRequest,
    RuntimeContext,
)

__all__ = [
    "BACKEND_REGISTRY",
    "CANONICAL_BACKENDS",
    "STATIC_BACKEND_DESCRIPTORS",
    "BackendAdmission",
    "BackendAdmissionError",
    "BackendAdmissionEvidence",
    "BackendCapabilities",
    "BackendCancellationError",
    "BackendCapabilityError",
    "BackendCompatibilityError",
    "BackendDescriptor",
    "BackendError",
    "BackendHealth",
    "BackendInferenceError",
    "BackendLifecycleError",
    "BackendLoadError",
    "BackendNotReadyError",
    "BackendRegistry",
    "BackendRegistryError",
    "BackendResult",
    "BackendState",
    "InferenceBackend",
    "InferenceRequest",
    "LifecycleBackend",
    "PartialLoadRollback",
    "ResourceDomainAdmissions",
    "RuntimeContext",
]
