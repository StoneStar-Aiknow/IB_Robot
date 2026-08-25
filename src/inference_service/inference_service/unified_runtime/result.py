"""Result and failure boundary imports."""

from .adapters import ModelResultAdapter, ResultAdapter, ResultAdapterProtocol, SuccessResultAdapter
from .contracts import ModelResult, OutcomeEvidence, OutcomeEvidenceTracker, OutcomeState, RuntimeLatency
from .errors import (
    ExecutionFailure,
    ExecutionFailureFactory,
    InvalidRecoveryRequirement,
    RecoveryAction,
    RecoveryRequirement,
    RecoveryScope,
    serialize_execution_failure,
)

__all__ = [
    "ExecutionFailure",
    "ExecutionFailureFactory",
    "InvalidRecoveryRequirement",
    "ModelResult",
    "ModelResultAdapter",
    "OutcomeEvidence",
    "OutcomeEvidenceTracker",
    "OutcomeState",
    "RecoveryAction",
    "RecoveryRequirement",
    "RecoveryScope",
    "ResultAdapter",
    "ResultAdapterProtocol",
    "RuntimeLatency",
    "serialize_execution_failure",
    "SuccessResultAdapter",
]
