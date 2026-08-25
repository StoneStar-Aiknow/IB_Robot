"""Private transition aliases for the retired policy backend contract.

This module is intentionally outside the public ``inference_service.backends``
namespace. New runtime construction uses ModelRuntimeFactory and
ModelRuntimeHandle; only un-migrated policy tests/adapters may import these
aliases while their stage executor is being retired.
"""

from .types import _LegacyBackendProtocol, _LegacyBackendResult

BackendResult = _LegacyBackendResult
InferenceBackend = _LegacyBackendProtocol

__all__ = ["BackendResult", "InferenceBackend"]
