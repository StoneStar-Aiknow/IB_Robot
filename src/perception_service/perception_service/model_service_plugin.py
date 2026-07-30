"""Contract for typed ROS service plugins hosted by the generic model node."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType


@dataclass(frozen=True)
class PluginRuntimeStatus:
    """Family-neutral runtime state projected into ``ModelRuntimeInfo``."""

    state: str
    ready: bool
    failure_reason: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.state:
            raise ValueError("plugin runtime state must be non-empty")
        if self.ready and self.failure_reason:
            raise ValueError("a ready plugin cannot report a failure reason")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class ModelServicePlugin(ABC):
    """Bind one concrete ROS service type to one model adapter and runtime session."""

    service_type: str

    @abstractmethod
    def handle(self, request, response) -> str:
        """Execute one typed request and populate its domain response."""

    @abstractmethod
    def runtime_status(self) -> PluginRuntimeStatus:
        """Return current shared-runtime health without family-specific fields."""

    @abstractmethod
    def close(self) -> None:
        """Release plugin-owned adapter/session references idempotently."""


__all__ = ["ModelServicePlugin", "PluginRuntimeStatus"]
