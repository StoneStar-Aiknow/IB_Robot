"""Composition-root helpers for inference runtime dependencies."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from inference_service.backends import STATIC_BACKEND_DESCRIPTORS, BackendRegistry
from inference_service.backends.admission import ResourceDomainAdmissions
from inference_service.backends.ascend.acl_runtime import AclRuntimeManager
from inference_service.unified_runtime import (
    RegistrySet,
    RuntimeAssemblerRegistry,
    RuntimeDependencyError,
    RuntimeProviders,
    SessionBuilderRegistry,
)


@dataclass(frozen=True, slots=True)
class RuntimeDependencies:
    """All dependencies a production runtime composition root owns."""

    registry_set: RegistrySet
    providers: RuntimeProviders


def require_runtime_dependencies(
    registry_set: RegistrySet | None,
    providers: RuntimeProviders | None,
    *,
    owner: str,
) -> tuple[RegistrySet, RuntimeProviders]:
    """Validate explicit dependencies at a construction boundary."""

    if registry_set is None:
        raise RuntimeDependencyError(
            f"{owner} requires an explicitly injected RegistrySet",
            code="registry_set_required",
        )
    if providers is None:
        raise RuntimeDependencyError(
            f"{owner} requires explicitly injected RuntimeProviders",
            code="runtime_providers_required",
        )
    if not isinstance(registry_set, RegistrySet):
        raise RuntimeDependencyError(
            f"{owner} registry_set must be a RegistrySet",
            code="registry_set_invalid",
        )
    if not isinstance(providers, RuntimeProviders):
        raise RuntimeDependencyError(
            f"{owner} providers must be a RuntimeProviders value",
            code="runtime_providers_invalid",
        )
    return registry_set, providers


def build_runtime_dependencies(
    register_builtins: Callable[[SessionBuilderRegistry, RuntimeAssemblerRegistry], None] | None = None,
    *,
    backend_registry: BackendRegistry | None = None,
    providers: RuntimeProviders | None = None,
) -> RuntimeDependencies:
    """Build and freeze one process-local runtime composition.

    ``register_builtins`` is the explicit composition callback for the service
    being launched. It runs while all construction registries are mutable,
    before the resulting ``RegistrySet`` is frozen.
    """

    if backend_registry is None:
        backend_registry = BackendRegistry(STATIC_BACKEND_DESCRIPTORS)
    validate_static_registry = getattr(backend_registry, "validate_static_registry", None)
    if callable(validate_static_registry):
        validate_static_registry()

    session_registry = SessionBuilderRegistry()
    assembler_registry = RuntimeAssemblerRegistry()

    if register_builtins is not None:
        if not callable(register_builtins):
            raise TypeError("register_builtins must be callable")
        register_builtins(session_registry, assembler_registry)

    registry_set = RegistrySet(
        backend_registry,
        session_registry,
        assembler_registry,
    )
    registry_set.freeze()

    if providers is None:
        providers = RuntimeProviders.create(AclRuntimeManager(), ResourceDomainAdmissions())
    if not isinstance(providers, RuntimeProviders):
        raise RuntimeDependencyError(
            "providers must be a RuntimeProviders value",
            code="runtime_providers_invalid",
        )
    return RuntimeDependencies(registry_set, providers)


def build_policy_runtime_dependencies(**kwargs: object) -> RuntimeDependencies:
    """Build an isolated composition containing the policy compatibility builders."""

    def register(session_registry: SessionBuilderRegistry, assembler_registry: RuntimeAssemblerRegistry) -> None:
        from inference_service.pipeline.factory import register_policy_session_builders

        register_policy_session_builders(session_registry, assembler_registry)

    return build_runtime_dependencies(register, **kwargs)


def build_model_service_runtime_dependencies(**kwargs: object) -> RuntimeDependencies:
    """Build the explicit composition used by generic model-service plugins."""

    def register(session_registry: SessionBuilderRegistry, assembler_registry: RuntimeAssemblerRegistry) -> None:
        from inference_service.pipeline.factory import register_policy_session_builders

        register_policy_session_builders(session_registry, assembler_registry)
        optional_registrars = (
            ("perception_service.model_session_builders", "register_perception_session_builders"),
            ("voice_asr_service.model_session_builders", "register_speech_direction_session_builder"),
            ("voice_tts_service.model_session_builders", "register_zipvoice_session_builder"),
        )
        for module_name, function_name in optional_registrars:
            try:
                module = __import__(module_name, fromlist=[function_name])
            except ModuleNotFoundError as exc:
                if exc.name and not exc.name.startswith(module_name.split(".", 1)[0]):
                    raise
                continue
            getattr(module, function_name)(session_registry)

    return build_runtime_dependencies(register, **kwargs)


__all__ = [
    "RuntimeDependencies",
    "build_model_service_runtime_dependencies",
    "build_policy_runtime_dependencies",
    "build_runtime_dependencies",
    "require_runtime_dependencies",
]
