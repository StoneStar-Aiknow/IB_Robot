"""Common lifecycle contracts and lazy registry for inference backends."""

from inference_service.backends.admission import ResourceDomainAdmissions
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
from inference_service.backends.registry import (
    CANONICAL_BACKENDS,
    MODEL_TYPE_OPERATIONS,
    PERCEPTION_FAMILIES,
    STATIC_BACKEND_DESCRIPTORS,
    VALID_INTERFACES,
    BackendDescriptor,
    BackendRegistry,
    ConformanceEvidence,
)
from inference_service.backends.types import (
    BackendAdmissionEvidence,
    BackendCapabilities,
    BackendHealth,
    BackendPriorityMapping,
    BackendState,
    InferenceRequest,
    RuntimeContext,
)

__all__ = [
    "CANONICAL_BACKENDS",
    "MODEL_TYPE_OPERATIONS",
    "ConformanceEvidence",
    "PERCEPTION_FAMILIES",
    "STATIC_BACKEND_DESCRIPTORS",
    "VALID_INTERFACES",
    "BackendAdmissionError",
    "BackendAdmissionEvidence",
    "BackendCapabilities",
    "BackendCancellationError",
    "BackendCapabilityError",
    "BackendCompatibilityError",
    "BackendDescriptor",
    "BackendError",
    "BackendHealth",
    "BackendPriorityMapping",
    "BackendInferenceError",
    "BackendLifecycleError",
    "BackendLoadError",
    "BackendNotReadyError",
    "BackendRegistry",
    "BackendRegistryError",
    "BackendState",
    "InferenceRequest",
    "ResourceDomainAdmissions",
    "RuntimeContext",
]
