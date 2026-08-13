"""Manifest-keyed construction registry for model execution sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from inference_service.backends import BACKEND_REGISTRY, BackendLoadError, BackendRegistry, RuntimeContext
from inference_service.model_sessions.base import ModelSession

ModelSessionBuilder = Callable[..., ModelSession]


@dataclass(frozen=True)
class ModelSessionBuilderKey:
    kind: str
    family: str
    operation: str
    backend: str

    @classmethod
    def from_context(cls, context: RuntimeContext) -> ModelSessionBuilderKey:
        model = context.model
        family = context.policy.policy_type if model.kind == "policy" else model.family
        return cls(model.kind, family, model.operation, context.deployment.backend)


class ModelSessionBuilderRegistry:
    """Select and invoke session builders from canonical manifest identity."""

    def __init__(self) -> None:
        self._builders: dict[ModelSessionBuilderKey, ModelSessionBuilder] = {}

    def register(
        self,
        kind: str,
        family: str,
        operation: str,
        backend: str,
        builder: ModelSessionBuilder,
    ) -> None:
        key = ModelSessionBuilderKey(kind, family, operation, backend)
        if key in self._builders:
            raise ValueError(f"model session builder {key!r} is already registered")
        if not callable(builder):
            raise TypeError("model session builder must be callable")
        self._builders[key] = builder

    def get(self, kind: str, family: str, operation: str, backend: str) -> ModelSessionBuilder | None:
        return self._builders.get(ModelSessionBuilderKey(kind, family, operation, backend))

    def create(
        self,
        context: RuntimeContext,
        *,
        allowed_deployments: frozenset[str] | None = None,
        backend_registry: BackendRegistry = BACKEND_REGISTRY,
        override: ModelSessionBuilder | None = None,
        builder_options: Mapping[str, object] | None = None,
    ) -> ModelSession:
        backend_registry.validate(context, allowed_deployments=allowed_deployments)
        key = ModelSessionBuilderKey.from_context(context)
        builder = override or self._builders.get(key)
        if builder is None:
            raise BackendLoadError(f"model session builder {key!r} is unavailable", code="session_builder_unavailable")
        session = builder(context, **dict(builder_options or {}))
        if not isinstance(session, ModelSession):
            raise TypeError(f"model session builder {key!r} must return ModelSession")
        return session


MODEL_SESSION_BUILDER_REGISTRY = ModelSessionBuilderRegistry()


__all__ = [
    "MODEL_SESSION_BUILDER_REGISTRY",
    "ModelSessionBuilder",
    "ModelSessionBuilderKey",
    "ModelSessionBuilderRegistry",
]
