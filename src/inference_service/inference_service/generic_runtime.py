"""Legacy request adapter for named semantic tensors.

New local runtime code uses ``unified_runtime.ModelRequest`` and
``unified_runtime.ModelResult``.  ``NamedTensorRequest`` remains a private
session transport until all service adapters have moved to the typed request
boundary.  Legacy named-result imports are served lazily from
``_legacy_named_tensor`` for the few compatibility fixtures that still need
them; they are not importable from this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType

from inference_service.backends import InferenceRequest


def _immutable_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(value))


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


__all__ = ["NamedTensorRequest"]
