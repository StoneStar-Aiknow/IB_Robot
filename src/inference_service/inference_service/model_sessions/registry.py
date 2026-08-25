"""V3 identity-keyed model-session builder registry."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping

from inference_service.backends import BackendLoadError, BackendRegistry, RuntimeContext
from inference_service.model_sessions.base import ModelSession
from inference_service.unified_runtime.registry import (
    RuntimeDependencyError,
    RuntimeProviders,
    SessionBuilderKey,
    SessionBuilderRegistry,
)

ModelSessionBuilder = Callable[..., ModelSession]
ModelSessionBuilderKey = SessionBuilderKey


class ModelSessionBuilderRegistry(SessionBuilderRegistry):
    """Select a session constructor from ``interface/model_type/operation``.

    The registry deliberately has no ``kind``/``family`` compatibility path.
    A unified-runtime role context can be passed directly; the factory has
    already performed backend/profile conformance validation in that case.
    """

    def create(
        self,
        context: object,
        *,
        allowed_deployments: frozenset[str] | None = None,
        backend_registry: BackendRegistry | None = None,
        providers: RuntimeProviders | None = None,
        override: ModelSessionBuilder | None = None,
        builder_options: Mapping[str, object] | None = None,
    ) -> ModelSession:
        if providers is None:
            raise RuntimeDependencyError(
                "ModelSessionBuilderRegistry requires explicitly injected RuntimeProviders",
                code="runtime_providers_required",
            )
        if not isinstance(providers, RuntimeProviders):
            raise RuntimeDependencyError(
                "providers must be a RuntimeProviders value",
                code="runtime_providers_invalid",
            )
        if isinstance(context, RuntimeContext):
            if backend_registry is None:
                raise BackendLoadError(
                    "BackendRegistry must be explicitly supplied for a RuntimeContext",
                    code="backend_registry_required",
                )
            backend_registry.validate(context, allowed_deployments=allowed_deployments)
        elif allowed_deployments is not None:
            deployment_name = getattr(context, "deployment_name", None)
            if deployment_name not in allowed_deployments:
                raise BackendLoadError(
                    f"deployment {deployment_name!r} is not in the adapter supported deployments "
                    f"{sorted(allowed_deployments)}",
                    code="adapter_deployment_mismatch",
                )

        key = ModelSessionBuilderKey.from_context(context)
        builder = override or self.get(key)
        if builder is None:
            raise BackendLoadError(f"model session builder {key!r} is unavailable", code="session_builder_unavailable")
        options = dict(builder_options or {})
        if "providers" not in options:
            try:
                signature = inspect.signature(builder)
            except (TypeError, ValueError):
                signature = None
            if (
                signature is None
                or "providers" in signature.parameters
                or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
            ):
                options["providers"] = providers
        session = builder(context, **options)
        if not isinstance(session, ModelSession):
            raise TypeError(f"model session builder {key!r} must return ModelSession")
        return session


__all__ = [
    "ModelSessionBuilder",
    "ModelSessionBuilderKey",
    "ModelSessionBuilderRegistry",
]
