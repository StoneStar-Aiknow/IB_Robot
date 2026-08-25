"""Dependency-neutral construction registries for unified runtimes.

This module contains only value objects, callables, and small registry
containers.  It deliberately does not import a backend SDK, a manifest
loader, or a concrete model session.  The composition root supplies those
objects and freezes the resulting :class:`RegistrySet` before construction.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias

from .assembly import RuntimeAssembly, RuntimeProviders

_VALID_INTERFACES = frozenset({"policy", "tensor_model"})
_VALID_CONTRACTS = frozenset(
    {
        "request-direct",
        "request-iterative",
        "stream-direct",
        "stream-iterative",
    }
)
_VALID_VISIBILITY = frozenset({"executor", "session"})
_UNSET = object()


class RuntimeRegistryError(ValueError):
    """Configuration error raised by a construction registry.

    ``code`` is intentionally stable so composition roots and diagnostics can
    distinguish duplicate registration from an invalid cross-registry graph.
    """

    def __init__(self, message: str, *, code: str = "runtime_registry_error") -> None:
        super().__init__(message)
        self.code = code


class RegistryFrozenError(RuntimeRegistryError):
    """Mutation was attempted after a registry or registry set was frozen."""

    def __init__(self, message: str = "registry is frozen") -> None:
        super().__init__(message, code="registry_frozen")


class RuntimeDependencyError(RuntimeRegistryError):
    """A required explicit construction dependency was not supplied."""

    def __init__(self, message: str, *, code: str = "runtime_dependency_error") -> None:
        super().__init__(message, code=code)


def _identifier(value: object, name: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized and not allow_empty:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _interface(value: object) -> str:
    normalized = _identifier(value, "interface")
    if normalized not in _VALID_INTERFACES:
        raise ValueError(f"interface must be one of {sorted(_VALID_INTERFACES)}, got {normalized!r}")
    return normalized


def _json_value(value: object) -> object:
    """Convert a small value object into deterministic JSON-compatible data."""

    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return _json_value(value.model_dump(mode="json", exclude_none=False))
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, tuple | list):
        return [_json_value(item) for item in value]
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return value


def _freeze_value(value: object) -> object:
    """Freeze mapping/list containers while leaving opaque provider objects alone."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})
    if isinstance(value, tuple | list):
        return tuple(_freeze_value(item) for item in value)
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(_json_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _contract_from_value(value: object) -> str:
    """Normalize a string or a manifest/unified execution-contract value."""

    candidate: object = value
    if isinstance(value, Mapping):
        if "name" in value:
            candidate = value["name"]
        elif "contract_name" in value:
            candidate = value["contract_name"]
        elif {"state_scope", "execution_structure"}.issubset(value):
            candidate = f"{value['state_scope']}-{value['execution_structure']}"
    elif not isinstance(value, str):
        candidate = getattr(value, "name", None)
        if candidate is None:
            candidate = getattr(value, "contract_name", None)
        if candidate is None and hasattr(value, "state_scope") and hasattr(value, "execution_structure"):
            candidate = f"{value.state_scope}-{value.execution_structure}"

    if not isinstance(candidate, str):
        raise TypeError("execution_contract must be a canonical string or an execution-contract value")
    normalized = candidate.strip()
    if normalized not in _VALID_CONTRACTS:
        raise ValueError(f"execution_contract must be one of {sorted(_VALID_CONTRACTS)}, got {normalized!r}")
    return normalized


def _visibility_for_contract(contract: str, value: object) -> str | None:
    if value is None:
        normalized: str | None = None
    elif isinstance(value, str):
        normalized = value.strip() or None
    else:
        raise TypeError("orchestration_visibility must be a string or None")

    if contract.endswith("-direct"):
        if normalized is not None:
            raise ValueError("direct execution requires orchestration_visibility=None")
        return None
    if normalized not in _VALID_VISIBILITY:
        raise ValueError("iterative execution requires orchestration_visibility='executor' or 'session'")
    return normalized


def _identity_parts(value: object) -> tuple[str, str, str]:
    if isinstance(value, Mapping):
        source = value
        return (
            _interface(source.get("interface")),
            _identifier(source.get("model_type"), "model_type"),
            _identifier(source.get("operation"), "operation"),
        )
    return (
        _interface(value.interface),
        _identifier(value.model_type, "model_type"),
        _identifier(value.operation, "operation"),
    )


def _backend_from_context(context: object) -> str:
    candidates: list[object] = [context]
    for parent_name in ("runtime_profile", "role_runtime_profile", "profile"):
        parent = getattr(context, parent_name, None)
        if parent is not None:
            candidates.append(parent)
    deployment = getattr(context, "deployment", None)
    if deployment is not None:
        candidates.append(deployment)
        for parent_name in ("runtime_profile", "role_runtime_profile", "profile"):
            parent = getattr(deployment, parent_name, None)
            if parent is not None:
                candidates.append(parent)
    for candidate in candidates:
        backend = getattr(candidate, "backend", None)
        if isinstance(backend, str) and backend.strip():
            return backend.strip()
    raise ValueError("context does not expose a backend for SessionBuilderKey construction")


@dataclass(frozen=True, slots=True)
class SessionBuilderKey:
    """Identity used only to select a concrete model-session builder."""

    interface: str
    model_type: str
    operation: str
    backend: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "interface", _interface(self.interface))
        object.__setattr__(self, "model_type", _identifier(self.model_type, "model_type"))
        object.__setattr__(self, "operation", _identifier(self.operation, "operation"))
        object.__setattr__(self, "backend", _identifier(self.backend, "backend"))

    @classmethod
    def from_identity(cls, identity: object, backend: str) -> SessionBuilderKey:
        interface, model_type, operation = _identity_parts(identity)
        return cls(interface, model_type, operation, backend)

    @classmethod
    def from_context(cls, context: object) -> SessionBuilderKey:
        model = getattr(context, "identity", None)
        if model is None:
            model = getattr(context, "model", None)
        if model is None:
            raise ValueError("context does not expose a model identity")

        if not hasattr(model, "interface") or not hasattr(model, "model_type") or not hasattr(model, "operation"):
            raise ValueError("context model does not expose a v3 identity")
        identity = _identity_parts(model)
        return cls(*identity, _backend_from_context(context))

    def to_dict(self) -> dict[str, str]:
        return {
            "interface": self.interface,
            "model_type": self.model_type,
            "operation": self.operation,
            "backend": self.backend,
        }

    serialize = to_dict

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    to_json = canonical_json

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class ModelRuntimeKey:
    """Canonical role-level runtime assembler selector."""

    interface: str
    model_type: str
    operation: str
    backend: str
    execution_contract: str
    orchestration_visibility: str | None = None

    def __post_init__(self) -> None:
        contract = _contract_from_value(self.execution_contract)
        object.__setattr__(self, "interface", _interface(self.interface))
        object.__setattr__(self, "model_type", _identifier(self.model_type, "model_type"))
        object.__setattr__(self, "operation", _identifier(self.operation, "operation"))
        object.__setattr__(self, "backend", _identifier(self.backend, "backend"))
        object.__setattr__(self, "execution_contract", contract)
        object.__setattr__(
            self, "orchestration_visibility", _visibility_for_contract(contract, self.orchestration_visibility)
        )

    @classmethod
    def from_identity(
        cls,
        identity: object,
        backend: str,
        execution_contract: object,
        orchestration_visibility: str | None = None,
    ) -> ModelRuntimeKey:
        interface, model_type, operation = _identity_parts(identity)
        return cls(
            interface,
            model_type,
            operation,
            backend,
            _contract_from_value(execution_contract),
            orchestration_visibility,
        )

    @classmethod
    def from_session_key(
        cls,
        session_key: SessionBuilderKey,
        execution_contract: object,
        orchestration_visibility: str | None = None,
    ) -> ModelRuntimeKey:
        return cls(
            session_key.interface,
            session_key.model_type,
            session_key.operation,
            session_key.backend,
            _contract_from_value(execution_contract),
            orchestration_visibility,
        )

    @property
    def session_builder_key(self) -> SessionBuilderKey:
        return SessionBuilderKey(self.interface, self.model_type, self.operation, self.backend)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "interface": self.interface,
            "model_type": self.model_type,
            "operation": self.operation,
            "backend": self.backend,
            "execution_contract": self.execution_contract,
            # Always emit this field.  ``None`` is the canonical JSON null for
            # direct contracts.
            "orchestration_visibility": self.orchestration_visibility,
        }

    serialize = to_dict

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    to_json = canonical_json

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class CompositeRuntimeKey:
    """Backend-agnostic top-level selector for a composite runtime."""

    interface: str
    model_type: str
    operation: str
    execution_contract: str
    orchestration_visibility: str | None = None

    def __post_init__(self) -> None:
        contract = _contract_from_value(self.execution_contract)
        object.__setattr__(self, "interface", _interface(self.interface))
        object.__setattr__(self, "model_type", _identifier(self.model_type, "model_type"))
        object.__setattr__(self, "operation", _identifier(self.operation, "operation"))
        object.__setattr__(self, "execution_contract", contract)
        object.__setattr__(
            self, "orchestration_visibility", _visibility_for_contract(contract, self.orchestration_visibility)
        )

    @classmethod
    def from_identity(
        cls,
        identity: object,
        execution_contract: object,
        orchestration_visibility: str | None = None,
    ) -> CompositeRuntimeKey:
        interface, model_type, operation = _identity_parts(identity)
        return cls(interface, model_type, operation, _contract_from_value(execution_contract), orchestration_visibility)

    def to_dict(self) -> dict[str, str | None]:
        return {
            "interface": self.interface,
            "model_type": self.model_type,
            "operation": self.operation,
            "execution_contract": self.execution_contract,
            "orchestration_visibility": self.orchestration_visibility,
        }

    serialize = to_dict

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")

    to_json = canonical_json

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True, slots=True)
class RuntimeRoleSelector:
    """One required composite role identity, independent of its backend."""

    role: str
    interface: str
    model_type: str
    operation: str
    execution_contract: str | None = None
    orchestration_visibility: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identifier(self.role, "role"))
        object.__setattr__(self, "interface", _interface(self.interface))
        object.__setattr__(self, "model_type", _identifier(self.model_type, "model_type"))
        object.__setattr__(self, "operation", _identifier(self.operation, "operation"))
        if self.execution_contract is not None:
            contract = _contract_from_value(self.execution_contract)
            object.__setattr__(self, "execution_contract", contract)
            object.__setattr__(
                self, "orchestration_visibility", _visibility_for_contract(contract, self.orchestration_visibility)
            )
        elif self.orchestration_visibility not in (None, ""):
            raise ValueError("a role selector cannot declare visibility without an execution contract")
        else:
            object.__setattr__(self, "orchestration_visibility", None)

    @classmethod
    def from_runtime_key(cls, role: str, key: ModelRuntimeKey) -> RuntimeRoleSelector:
        return cls(
            role,
            key.interface,
            key.model_type,
            key.operation,
            key.execution_contract,
            key.orchestration_visibility,
        )

    def with_contract(self, execution_contract: object) -> RuntimeRoleSelector:
        return replace(self, execution_contract=_contract_from_value(execution_contract))

    def runtime_key(
        self,
        backend: str,
        *,
        execution_contract: object | None = None,
        orchestration_visibility: str | None | object = _UNSET,
    ) -> ModelRuntimeKey:
        contract = self.execution_contract if execution_contract is None else _contract_from_value(execution_contract)
        if contract is None:
            raise ValueError(f"role selector {self.role!r} does not declare an execution contract")
        visibility = self.orchestration_visibility if orchestration_visibility is _UNSET else orchestration_visibility
        return ModelRuntimeKey(
            self.interface,
            self.model_type,
            self.operation,
            backend,
            contract,
            visibility,  # type: ignore[arg-type]
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "role": self.role,
            "interface": self.interface,
            "model_type": self.model_type,
            "operation": self.operation,
            "execution_contract": self.execution_contract,
            "orchestration_visibility": self.orchestration_visibility,
        }


CompositeRoleSelector = RuntimeRoleSelector
RoleSelector = RuntimeRoleSelector
ModelSessionBuilderKey = SessionBuilderKey


@dataclass(frozen=True, slots=True)
class RoleBackendProfile:
    """One role/backend/profile entry in a composite compatibility matrix."""

    backend: str
    profile_type: type | None = None
    target_runtime: str | None = None
    runtime_abi: str | None = None
    role: str | None = None
    profile: type | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", _identifier(self.backend, "backend"))
        selected_profile = self.profile_type or self.profile
        if selected_profile is not None and not isinstance(selected_profile, type):
            raise TypeError("role compatibility profile_type must be a type")
        object.__setattr__(self, "profile_type", selected_profile)
        if self.role is not None:
            object.__setattr__(self, "role", _identifier(self.role, "role"))
        if self.target_runtime is not None:
            object.__setattr__(self, "target_runtime", _identifier(self.target_runtime, "target_runtime"))
        if self.runtime_abi is not None:
            object.__setattr__(self, "runtime_abi", _identifier(self.runtime_abi, "runtime_abi"))

    @property
    def backend_profile_type(self) -> type | None:
        return self.profile_type

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "backend": self.backend,
            "profile_type": self.profile_type,
            "target_runtime": self.target_runtime,
            "runtime_abi": self.runtime_abi,
        }


RoleProfileCompatibility = RoleBackendProfile
RoleBackendProfileCompatibility = RoleBackendProfile


def _role_profile(value: object, *, role: str | None = None) -> RoleBackendProfile:
    if isinstance(value, RoleBackendProfile):
        if role is None or value.role == role:
            return value
        return replace(value, role=role)
    if isinstance(value, Mapping):
        raw = dict(value)
        if role is not None:
            raw.setdefault("role", role)
        if "profile_type" not in raw and "profile" not in raw and "backend_profile_type" in raw:
            raw["profile_type"] = raw.pop("backend_profile_type")
        return RoleBackendProfile(**raw)
    raise TypeError("composite role compatibility entries must be RoleBackendProfile values or mappings")


@dataclass(frozen=True, slots=True)
class CompositeRuntimeMatrixEntry:
    """One complete supported role/backend/profile combination."""

    roles: Mapping[str, RoleBackendProfile]
    constraints: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.roles, Mapping):
            raise TypeError("matrix entry roles must be a mapping")
        normalized = {
            _identifier(role, "matrix role"): _role_profile(profile, role=role) for role, profile in self.roles.items()
        }
        if not normalized:
            raise ValueError("matrix entry must contain at least one role")
        object.__setattr__(self, "roles", MappingProxyType(normalized))
        object.__setattr__(self, "constraints", _freeze_value(self.constraints))

    @property
    def role_backend_profiles(self) -> Mapping[str, RoleBackendProfile]:
        return self.roles

    def canonical_tuple(self) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (
                role,
                value.backend,
                value.profile_type,
                value.target_runtime,
                value.runtime_abi,
            )
            for role, value in sorted(self.roles.items())
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "roles": {role: value.to_dict() for role, value in sorted(self.roles.items())},
            "constraints": self.constraints,
        }


CompositeRoleCompatibility = CompositeRuntimeMatrixEntry
RoleCompatibilityMatrixEntry = CompositeRuntimeMatrixEntry


def _selector(value: object, *, role: str | None = None) -> RuntimeRoleSelector:
    if isinstance(value, RuntimeRoleSelector):
        if role is None or value.role == role:
            return value
        return replace(value, role=role)
    if isinstance(value, ModelRuntimeKey):
        if role is None:
            raise ValueError("a role is required when converting a ModelRuntimeKey to a selector")
        return RuntimeRoleSelector.from_runtime_key(role, value)
    if isinstance(value, Mapping):
        raw = dict(value)
        if "key" in raw:
            key = raw.pop("key")
            selected_role = raw.pop("role", role)
            if not isinstance(key, ModelRuntimeKey) or selected_role is None:
                raise TypeError("selector key must be a ModelRuntimeKey and role must be supplied")
            return RuntimeRoleSelector.from_runtime_key(selected_role, key)
        if role is not None:
            raw.setdefault("role", role)
        return RuntimeRoleSelector(**raw)
    raise TypeError("required role selectors must be RuntimeRoleSelector values or mappings")


def _matrix_entries(value: object) -> tuple[CompositeRuntimeMatrixEntry, ...]:
    if value is None:
        return ()
    if isinstance(value, CompositeRuntimeMatrixEntry):
        return (value,)
    if isinstance(value, RoleBackendProfile):
        if value.role is None:
            raise ValueError("a standalone role compatibility entry requires role")
        return (CompositeRuntimeMatrixEntry({value.role: value}),)
    if isinstance(value, Mapping):
        # A mapping of role -> list of alternatives is expanded into a
        # deterministic Cartesian matrix.  A mapping of role -> one profile
        # is the common one-row shorthand.
        values = tuple(value.values())
        is_alternatives = bool(values) and all(
            isinstance(item, Sequence) and not isinstance(item, str | bytes | Mapping) for item in values
        )
        if is_alternatives:
            roles = tuple(value)
            options = [tuple(_role_profile(item, role=role) for item in value[role]) for role in roles]
            return tuple(
                CompositeRuntimeMatrixEntry(dict(zip(roles, combination, strict=True)))
                for combination in itertools.product(*options)
            )
        return (CompositeRuntimeMatrixEntry(value),)
    if isinstance(value, str | bytes):
        raise TypeError("role compatibility matrix must be a mapping or sequence")
    entries: list[CompositeRuntimeMatrixEntry] = []
    flat_profiles: list[RoleBackendProfile] = []
    for item in value:  # type: ignore[union-attr]
        if isinstance(item, CompositeRuntimeMatrixEntry):
            entries.append(item)
        elif isinstance(item, RoleBackendProfile):
            flat_profiles.append(item)
        elif isinstance(item, Mapping):
            entries.append(CompositeRuntimeMatrixEntry(item))
        else:
            raise TypeError("invalid composite role compatibility matrix entry")
    if flat_profiles:
        if entries or any(profile.role is None for profile in flat_profiles):
            raise ValueError("flat role compatibility entries require unique role values")
        entries.append(
            CompositeRuntimeMatrixEntry({profile.role: profile for profile in flat_profiles if profile.role})
        )
    return tuple(entries)


def _capability_contract(capabilities: object) -> str | None:
    if capabilities is None:
        return None
    if isinstance(capabilities, Mapping):
        for name in ("execution_contract", "contract", "contract_name"):
            if name in capabilities:
                return _contract_from_value(capabilities[name])
        if {"state_scope", "execution_structure"}.issubset(capabilities):
            return _contract_from_value(capabilities)
        supported = capabilities.get("supported_execution_contracts")
        if supported is not None:
            values = tuple(_contract_from_value(item) for item in supported)
            return values[0] if len(values) == 1 else None
        return None
    for name in ("execution_contract", "contract", "contract_name"):
        value = getattr(capabilities, name, None)
        if value is not None:
            return _contract_from_value(value)
    if hasattr(capabilities, "state_scope") and hasattr(capabilities, "execution_structure"):
        return _contract_from_value(capabilities)
    return None


def _assembler_callable(assembler: object) -> Callable[..., object]:
    if callable(assembler):
        return assembler  # type: ignore[return-value]
    method = getattr(assembler, "assemble", None)
    if callable(method):
        return method
    raise TypeError("runtime assembler must be callable or expose assemble(...)")


def _assembler_contract(assembler: object) -> str | None:
    for name in ("execution_contract", "contract", "contract_name"):
        value = getattr(assembler, name, None)
        if value is not None:
            return _contract_from_value(value)
    capabilities = getattr(assembler, "declared_capabilities", None)
    if capabilities is None:
        capabilities = getattr(assembler, "capabilities", None)
    return _capability_contract(capabilities)


@dataclass(frozen=True, slots=True, init=False)
class RuntimeDescriptor:
    """Public role-level descriptor selected by :class:`ModelRuntimeKey`."""

    key: ModelRuntimeKey
    session_builder_key: SessionBuilderKey
    profile_type: type | None
    assembler: object
    execution_contract: str | None = None
    declared_capabilities: object = field(default_factory=dict)
    target_runtime: str | None = None
    runtime_abi: str | None = None
    supported_target_runtimes: frozenset[str] = frozenset()
    supported_runtime_abis: frozenset[str] = frozenset()

    def __init__(
        self,
        key: ModelRuntimeKey | None = None,
        session_builder_key: SessionBuilderKey | None = None,
        profile_type: type | None = None,
        assembler: object | None = None,
        execution_contract: str | None = None,
        declared_capabilities: object | None = None,
        target_runtime: str | None = None,
        runtime_abi: str | None = None,
        supported_target_runtimes: frozenset[str] = frozenset(),
        supported_runtime_abis: frozenset[str] = frozenset(),
        *,
        runtime_key: ModelRuntimeKey | None = None,
        backend_profile_type: type | None = None,
        capabilities: object | None = None,
        session_builder_reference: SessionBuilderKey | None = None,
    ) -> None:
        if key is None:
            key = runtime_key
        elif runtime_key is not None and key != runtime_key:
            raise ValueError("key and runtime_key disagree")
        if session_builder_key is None:
            session_builder_key = session_builder_reference
        elif session_builder_reference is not None and session_builder_key != session_builder_reference:
            raise ValueError("session_builder_key and session_builder_reference disagree")
        if profile_type is None:
            profile_type = backend_profile_type
        elif backend_profile_type is not None and profile_type is not backend_profile_type:
            raise ValueError("profile_type and backend_profile_type disagree")
        if declared_capabilities is None:
            declared_capabilities = capabilities if capabilities is not None else {}
        elif capabilities is not None and declared_capabilities != capabilities:
            raise ValueError("declared_capabilities and capabilities disagree")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "session_builder_key", session_builder_key)
        object.__setattr__(self, "profile_type", profile_type)
        object.__setattr__(self, "assembler", assembler)
        object.__setattr__(self, "execution_contract", execution_contract)
        object.__setattr__(self, "declared_capabilities", declared_capabilities)
        object.__setattr__(self, "target_runtime", target_runtime)
        object.__setattr__(self, "runtime_abi", runtime_abi)
        object.__setattr__(self, "supported_target_runtimes", supported_target_runtimes)
        object.__setattr__(self, "supported_runtime_abis", supported_runtime_abis)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.key, ModelRuntimeKey):
            raise TypeError("RuntimeDescriptor.key must be a ModelRuntimeKey")
        if not isinstance(self.session_builder_key, SessionBuilderKey):
            raise TypeError("RuntimeDescriptor.session_builder_key must be a SessionBuilderKey")
        if self.profile_type is not None and not isinstance(self.profile_type, type):
            raise TypeError("RuntimeDescriptor.profile_type must be a type")
        _assembler_callable(self.assembler)
        contract = (
            self.key.execution_contract
            if self.execution_contract is None
            else _contract_from_value(self.execution_contract)
        )
        object.__setattr__(self, "execution_contract", contract)
        object.__setattr__(self, "declared_capabilities", _freeze_value(self.declared_capabilities))
        if self.target_runtime is not None:
            object.__setattr__(self, "target_runtime", _identifier(self.target_runtime, "target_runtime"))
        if self.runtime_abi is not None:
            object.__setattr__(self, "runtime_abi", _identifier(self.runtime_abi, "runtime_abi"))
        object.__setattr__(self, "supported_target_runtimes", frozenset(self.supported_target_runtimes))
        object.__setattr__(self, "supported_runtime_abis", frozenset(self.supported_runtime_abis))

    @property
    def backend_profile_type(self) -> type | None:
        return self.profile_type

    @property
    def runtime_key(self) -> ModelRuntimeKey:
        return self.key

    @property
    def session_builder_reference(self) -> SessionBuilderKey:
        return self.session_builder_key

    @property
    def profile(self) -> type | None:
        return self.profile_type

    @property
    def assembler_contract(self) -> str | None:
        return _assembler_contract(self.assembler)

    @property
    def capabilities(self) -> object:
        return self.declared_capabilities

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.to_dict(),
            "session_builder_key": self.session_builder_key.to_dict(),
            "profile_type": self.profile_type,
            "execution_contract": self.execution_contract,
            "declared_capabilities": self.declared_capabilities,
            "target_runtime": self.target_runtime,
            "runtime_abi": self.runtime_abi,
        }


@dataclass(frozen=True, slots=True, init=False)
class CompositeRuntimeDescriptor:
    """Public top-level descriptor and explicit aggregate assembler."""

    key: CompositeRuntimeKey
    assembler: object
    required_role_selectors: Sequence[RuntimeRoleSelector] | Mapping[str, object] = ()
    role_compatibility_matrix: object = ()
    execution_contract: str | None = None
    declared_capabilities: object = field(default_factory=dict)
    required_roles: Sequence[RuntimeRoleSelector] | Mapping[str, object] | None = None
    role_backend_profile_matrix: object | None = None

    def __init__(
        self,
        key: CompositeRuntimeKey | None = None,
        assembler: object | None = None,
        required_role_selectors: Sequence[RuntimeRoleSelector] | Mapping[str, object] = (),
        role_compatibility_matrix: object = (),
        execution_contract: str | None = None,
        declared_capabilities: object | None = None,
        required_roles: Sequence[RuntimeRoleSelector] | Mapping[str, object] | None = None,
        role_backend_profile_matrix: object | None = None,
        *,
        runtime_key: CompositeRuntimeKey | None = None,
        aggregate_assembler: object | None = None,
        capabilities: object | None = None,
        role_selectors: Sequence[RuntimeRoleSelector] | Mapping[str, object] | None = None,
        compatibility_matrix: object | None = None,
    ) -> None:
        if key is None:
            key = runtime_key
        elif runtime_key is not None and key != runtime_key:
            raise ValueError("key and runtime_key disagree")
        if assembler is None:
            assembler = aggregate_assembler
        elif aggregate_assembler is not None and assembler is not aggregate_assembler:
            raise ValueError("assembler and aggregate_assembler disagree")
        if role_selectors is not None:
            if required_role_selectors not in ((), {}, None) and required_role_selectors != role_selectors:
                raise ValueError("required_role_selectors and role_selectors disagree")
            required_role_selectors = role_selectors
        if compatibility_matrix is not None:
            if role_compatibility_matrix not in ((), {}, None) and role_compatibility_matrix != compatibility_matrix:
                raise ValueError("role compatibility matrix aliases disagree")
            role_compatibility_matrix = compatibility_matrix
        if declared_capabilities is None:
            declared_capabilities = capabilities if capabilities is not None else {}
        elif capabilities is not None and declared_capabilities != capabilities:
            raise ValueError("declared_capabilities and capabilities disagree")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "assembler", assembler)
        object.__setattr__(self, "required_role_selectors", required_role_selectors)
        object.__setattr__(self, "role_compatibility_matrix", role_compatibility_matrix)
        object.__setattr__(self, "execution_contract", execution_contract)
        object.__setattr__(self, "declared_capabilities", declared_capabilities)
        object.__setattr__(self, "required_roles", required_roles)
        object.__setattr__(self, "role_backend_profile_matrix", role_backend_profile_matrix)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.key, CompositeRuntimeKey):
            raise TypeError("CompositeRuntimeDescriptor.key must be a CompositeRuntimeKey")
        _assembler_callable(self.assembler)
        contract = (
            self.key.execution_contract
            if self.execution_contract is None
            else _contract_from_value(self.execution_contract)
        )
        object.__setattr__(self, "execution_contract", contract)
        raw_selectors = self.required_role_selectors
        if self.required_roles is not None:
            if raw_selectors not in ((), {}, None) and raw_selectors != self.required_roles:
                raise ValueError("required_role_selectors and required_roles disagree")
            raw_selectors = self.required_roles
        selectors: list[RuntimeRoleSelector] = []
        if isinstance(raw_selectors, Mapping):
            selectors = [_selector(value, role=role) for role, value in raw_selectors.items()]
        elif isinstance(raw_selectors, RuntimeRoleSelector | ModelRuntimeKey):
            selectors = [_selector(raw_selectors)]
        else:
            selectors = [_selector(value) for value in raw_selectors]
        selectors = [
            selector if selector.execution_contract is not None else selector.with_contract(contract)
            for selector in selectors
        ]
        object.__setattr__(self, "required_role_selectors", tuple(selectors))
        object.__setattr__(self, "required_roles", tuple(selectors))
        raw_matrix = self.role_compatibility_matrix
        if self.role_backend_profile_matrix is not None:
            if raw_matrix not in ((), {}, None) and raw_matrix != self.role_backend_profile_matrix:
                raise ValueError("role compatibility matrix aliases disagree")
            raw_matrix = self.role_backend_profile_matrix
        normalized_matrix = _matrix_entries(raw_matrix)
        object.__setattr__(self, "role_compatibility_matrix", normalized_matrix)
        object.__setattr__(self, "role_backend_profile_matrix", normalized_matrix)
        object.__setattr__(self, "declared_capabilities", _freeze_value(self.declared_capabilities))

    @property
    def required_roles_map(self) -> Mapping[str, RuntimeRoleSelector]:
        return MappingProxyType({selector.role: selector for selector in self.required_role_selectors})

    @property
    def role_selectors(self) -> tuple[RuntimeRoleSelector, ...]:
        return tuple(self.required_role_selectors)

    @property
    def runtime_key(self) -> CompositeRuntimeKey:
        return self.key

    @property
    def role_backend_profile_matrix_entries(self) -> tuple[CompositeRuntimeMatrixEntry, ...]:
        return self.role_compatibility_matrix  # type: ignore[return-value]

    @property
    def aggregate_assembler(self) -> object:
        return self.assembler

    @property
    def assembler_contract(self) -> str | None:
        return _assembler_contract(self.assembler)

    @property
    def capabilities(self) -> object:
        return self.declared_capabilities

    @property
    def backend_agnostic(self) -> bool:
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.to_dict(),
            "required_role_selectors": [selector.to_dict() for selector in self.required_role_selectors],
            "role_compatibility_matrix": [entry.to_dict() for entry in self.role_compatibility_matrix],
            "execution_contract": self.execution_contract,
            "declared_capabilities": self.declared_capabilities,
        }


RuntimeAssembler: TypeAlias = Callable[..., RuntimeAssembly]
CompositeRuntimeAssembler: TypeAlias = Callable[..., RuntimeAssembly]


class SessionBuilderRegistry:
    """Dependency-neutral session-builder registry used by composition roots."""

    def __init__(self) -> None:
        self._builders: dict[SessionBuilderKey, Callable[..., object]] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def keys(self) -> tuple[SessionBuilderKey, ...]:
        return tuple(self._builders)

    @property
    def builders(self) -> Mapping[SessionBuilderKey, Callable[..., object]]:
        return MappingProxyType(self._builders)

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RegistryFrozenError("session builder registry is frozen")

    @staticmethod
    def _parse_key(
        key: SessionBuilderKey | str | None,
        model_type: str | Callable[..., object] | None,
        operation: str | None,
        backend: str | None,
        builder: Callable[..., object] | None,
        interface: str | None,
    ) -> tuple[SessionBuilderKey, Callable[..., object]]:
        if isinstance(key, SessionBuilderKey):
            selected_builder = builder
            if selected_builder is None and callable(model_type) and operation is None and backend is None:
                selected_builder = model_type
            if selected_builder is None:
                raise TypeError("register(key, builder) requires a callable builder")
            return key, selected_builder
        selected_interface = interface if interface is not None else key
        if selected_interface is None or model_type is None or operation is None or backend is None or builder is None:
            raise TypeError("register requires interface, model_type, operation, backend, and builder")
        if not callable(builder):
            raise TypeError("session builder must be callable")
        return SessionBuilderKey(selected_interface, model_type, operation, backend), builder

    def register(
        self,
        key: SessionBuilderKey | str | None = None,
        model_type: str | Callable[..., object] | None = None,
        operation: str | None = None,
        backend: str | None = None,
        builder: Callable[..., object] | None = None,
        *,
        interface: str | None = None,
    ) -> SessionBuilderKey:
        selected_key, selected_builder = self._parse_key(key, model_type, operation, backend, builder, interface)
        self._ensure_mutable()
        if selected_key in self._builders:
            raise RuntimeRegistryError(
                f"session builder {selected_key!r} is already registered",
                code="duplicate_session_builder_key",
            )
        self._builders[selected_key] = selected_builder
        return selected_key

    def register_idempotent(
        self,
        key: SessionBuilderKey,
        builder: Callable[..., object],
    ) -> SessionBuilderKey:
        self._ensure_mutable()
        current = self._builders.get(key)
        if current is not None:
            if current is not builder:
                raise RuntimeRegistryError(
                    f"session builder {key!r} is already registered with a different callable",
                    code="duplicate_session_builder_key",
                )
            return key
        return self.register(key, builder)

    def replace(self, key: SessionBuilderKey, builder: Callable[..., object]) -> SessionBuilderKey:
        self._ensure_mutable()
        if not isinstance(key, SessionBuilderKey) or not callable(builder):
            raise TypeError("replace requires a typed session key and callable builder")
        if key not in self._builders:
            raise RuntimeRegistryError(f"session builder {key!r} is not registered", code="session_builder_unavailable")
        self._builders[key] = builder
        return key

    def unregister(self, key: SessionBuilderKey) -> None:
        self._ensure_mutable()
        try:
            del self._builders[key]
        except KeyError as exc:
            raise RuntimeRegistryError(
                f"session builder {key!r} is not registered", code="session_builder_unavailable"
            ) from exc

    def get(
        self,
        key: SessionBuilderKey | str,
        model_type: str | None = None,
        operation: str | None = None,
        backend: str | None = None,
        *,
        interface: str | None = None,
    ) -> Callable[..., object] | None:
        if isinstance(key, SessionBuilderKey):
            selected_key = key
        else:
            selected_interface = interface if interface is not None else key
            if model_type is None or operation is None or backend is None:
                raise TypeError("get requires interface, model_type, operation, and backend")
            selected_key = SessionBuilderKey(selected_interface, model_type, operation, backend)
        return self._builders.get(selected_key)

    def create(self, key_or_context: SessionBuilderKey | object, *args: object, **kwargs: object) -> object:
        if isinstance(key_or_context, SessionBuilderKey):
            key = key_or_context
            context = kwargs.pop("context", None)
        else:
            context = key_or_context
            key = SessionBuilderKey.from_context(context)
        builder = self._builders.get(key)
        if builder is None:
            raise RuntimeRegistryError(
                f"session builder {key!r} is unavailable",
                code="session_builder_unavailable",
            )
        if context is None:
            return builder(*args, **kwargs)
        return builder(context, *args, **kwargs)

    def freeze(self) -> SessionBuilderRegistry:
        self._frozen = True
        return self


class RuntimeAssemblerRegistry:
    """Mutable registry for role and top-level runtime assemblers."""

    def __init__(self) -> None:
        self._runtime_descriptors: dict[ModelRuntimeKey, RuntimeDescriptor] = {}
        self._composite_descriptors: dict[CompositeRuntimeKey, CompositeRuntimeDescriptor] = {}
        self._frozen = False

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def runtime_keys(self) -> tuple[ModelRuntimeKey, ...]:
        return tuple(self._runtime_descriptors)

    @property
    def composite_keys(self) -> tuple[CompositeRuntimeKey, ...]:
        return tuple(self._composite_descriptors)

    @property
    def descriptors(self) -> Mapping[ModelRuntimeKey, RuntimeDescriptor]:
        return MappingProxyType(self._runtime_descriptors)

    @property
    def role_descriptors(self) -> Mapping[ModelRuntimeKey, RuntimeDescriptor]:
        return self.descriptors

    @property
    def composite_descriptors(self) -> Mapping[CompositeRuntimeKey, CompositeRuntimeDescriptor]:
        return MappingProxyType(self._composite_descriptors)

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RegistryFrozenError("runtime assembler registry is frozen")

    def register(
        self,
        descriptor: RuntimeDescriptor | CompositeRuntimeDescriptor | ModelRuntimeKey | CompositeRuntimeKey,
        selected_descriptor: RuntimeDescriptor | CompositeRuntimeDescriptor | None = None,
    ) -> object:
        self._ensure_mutable()
        if selected_descriptor is not None:
            if isinstance(descriptor, ModelRuntimeKey) and isinstance(selected_descriptor, RuntimeDescriptor):
                if descriptor != selected_descriptor.key:
                    raise RuntimeRegistryError(
                        "runtime key and descriptor key disagree", code="descriptor_key_mismatch"
                    )
                descriptor = selected_descriptor
            elif isinstance(descriptor, CompositeRuntimeKey) and isinstance(
                selected_descriptor, CompositeRuntimeDescriptor
            ):
                if descriptor != selected_descriptor.key:
                    raise RuntimeRegistryError(
                        "composite key and descriptor key disagree", code="descriptor_key_mismatch"
                    )
                descriptor = selected_descriptor
            else:
                raise TypeError("descriptor key and descriptor type do not match")
        if isinstance(descriptor, RuntimeDescriptor):
            if descriptor.key in self._runtime_descriptors:
                raise RuntimeRegistryError(
                    f"runtime descriptor {descriptor.key!r} is already registered",
                    code="duplicate_runtime_key",
                )
            self._runtime_descriptors[descriptor.key] = descriptor
            return descriptor.key
        if isinstance(descriptor, CompositeRuntimeDescriptor):
            if descriptor.key in self._composite_descriptors:
                raise RuntimeRegistryError(
                    f"composite runtime descriptor {descriptor.key!r} is already registered",
                    code="duplicate_composite_runtime_key",
                )
            self._composite_descriptors[descriptor.key] = descriptor
            return descriptor.key
        raise TypeError("runtime assembler registry accepts RuntimeDescriptor or CompositeRuntimeDescriptor")

    register_descriptor = register

    def register_runtime(
        self,
        descriptor: RuntimeDescriptor | ModelRuntimeKey,
        selected_descriptor: RuntimeDescriptor | None = None,
    ) -> ModelRuntimeKey:
        if selected_descriptor is not None:
            if not isinstance(descriptor, ModelRuntimeKey) or selected_descriptor.key != descriptor:
                raise RuntimeRegistryError(
                    "runtime key and descriptor key disagree",
                    code="descriptor_key_mismatch",
                )
            descriptor = selected_descriptor
        if not isinstance(descriptor, RuntimeDescriptor):
            raise TypeError("register_runtime requires a RuntimeDescriptor")
        return self.register(descriptor)  # type: ignore[return-value]

    def register_composite(
        self,
        descriptor: CompositeRuntimeDescriptor | CompositeRuntimeKey,
        selected_descriptor: CompositeRuntimeDescriptor | None = None,
    ) -> CompositeRuntimeKey:
        if selected_descriptor is not None:
            if not isinstance(descriptor, CompositeRuntimeKey) or selected_descriptor.key != descriptor:
                raise RuntimeRegistryError(
                    "composite key and descriptor key disagree",
                    code="descriptor_key_mismatch",
                )
            descriptor = selected_descriptor
        if not isinstance(descriptor, CompositeRuntimeDescriptor):
            raise TypeError("register_composite requires a CompositeRuntimeDescriptor")
        return self.register(descriptor)  # type: ignore[return-value]

    def register_idempotent(self, descriptor: RuntimeDescriptor | CompositeRuntimeDescriptor) -> object:
        self._ensure_mutable()
        current = self.get(descriptor.key)
        if current is not None:
            if current != descriptor:
                code = (
                    "duplicate_composite_runtime_key"
                    if isinstance(descriptor, CompositeRuntimeDescriptor)
                    else "duplicate_runtime_key"
                )
                raise RuntimeRegistryError(
                    f"runtime descriptor {descriptor.key!r} is already registered with different meaning",
                    code=code,
                )
            return descriptor.key
        return self.register(descriptor)

    def replace(self, descriptor: RuntimeDescriptor | CompositeRuntimeDescriptor) -> object:
        self._ensure_mutable()
        if isinstance(descriptor, RuntimeDescriptor):
            if descriptor.key not in self._runtime_descriptors:
                raise RuntimeRegistryError(
                    f"runtime descriptor {descriptor.key!r} is not registered",
                    code="runtime_assembler_unavailable",
                )
            self._runtime_descriptors[descriptor.key] = descriptor
            return descriptor.key
        if isinstance(descriptor, CompositeRuntimeDescriptor):
            if descriptor.key not in self._composite_descriptors:
                raise RuntimeRegistryError(
                    f"composite descriptor {descriptor.key!r} is not registered",
                    code="composite_assembler_unavailable",
                )
            self._composite_descriptors[descriptor.key] = descriptor
            return descriptor.key
        raise TypeError("replace requires a runtime descriptor")

    def unregister(self, key: ModelRuntimeKey | CompositeRuntimeKey) -> None:
        self._ensure_mutable()
        target = self._runtime_descriptors if isinstance(key, ModelRuntimeKey) else self._composite_descriptors
        try:
            del target[key]
        except KeyError as exc:
            raise RuntimeRegistryError(
                f"runtime descriptor {key!r} is not registered", code="runtime_assembler_unavailable"
            ) from exc

    def get(self, key: ModelRuntimeKey | CompositeRuntimeKey) -> RuntimeDescriptor | CompositeRuntimeDescriptor | None:
        if isinstance(key, ModelRuntimeKey):
            return self._runtime_descriptors.get(key)
        if isinstance(key, CompositeRuntimeKey):
            return self._composite_descriptors.get(key)
        raise TypeError("runtime assembler lookup requires a typed runtime key")

    def get_runtime(self, key: ModelRuntimeKey) -> RuntimeDescriptor | None:
        value = self.get(key)
        return value if isinstance(value, RuntimeDescriptor) else None

    def get_composite(self, key: CompositeRuntimeKey) -> CompositeRuntimeDescriptor | None:
        value = self.get(key)
        return value if isinstance(value, CompositeRuntimeDescriptor) else None

    get_role = get_runtime

    def require(self, key: ModelRuntimeKey | CompositeRuntimeKey) -> RuntimeDescriptor | CompositeRuntimeDescriptor:
        descriptor = self.get(key)
        if descriptor is not None:
            return descriptor
        code = (
            "composite_assembler_unavailable"
            if isinstance(key, CompositeRuntimeKey)
            else "runtime_assembler_unavailable"
        )
        raise RuntimeRegistryError(f"no runtime assembler is registered for {key!r}", code=code)

    resolve = require

    @staticmethod
    def _validate_assembly_contract(
        descriptor: RuntimeDescriptor | CompositeRuntimeDescriptor,
        assembly: RuntimeAssembly,
    ) -> None:
        actual = getattr(assembly, "execution_contract", None)
        actual_contract = (
            _contract_from_value(actual)
            if actual is not None
            else _capability_contract(getattr(assembly, "declared_capabilities", {}))
        )
        if actual_contract is not None and actual_contract != descriptor.execution_contract:
            raise RuntimeRegistryError(
                f"assembler for {descriptor.key!r} returned contract {actual_contract!r}, "
                f"expected {descriptor.execution_contract!r}",
                code="descriptor_assembly_contract_mismatch",
            )

    @staticmethod
    def _invoke_assembler(
        descriptor: RuntimeDescriptor | CompositeRuntimeDescriptor,
        args: tuple[object, ...],
        providers: RuntimeProviders,
        kwargs: dict[str, object],
    ) -> object:
        callable_assembler = _assembler_callable(descriptor.assembler)
        try:
            signature = inspect.signature(callable_assembler)
        except (TypeError, ValueError):
            selected_args, selected_kwargs = args, {**kwargs, "providers": providers}
            return callable_assembler(*selected_args, **selected_kwargs)
        candidates: list[tuple[tuple[object, ...], dict[str, object]]] = []
        if "descriptor" in signature.parameters and "descriptor" not in kwargs:
            candidates.append((args, {**kwargs, "providers": providers, "descriptor": descriptor}))
        candidates.extend(
            (
                (args, {**kwargs, "providers": providers}),
                ((*args, providers), kwargs),
                (args, kwargs),
            )
        )
        for selected_args, selected_kwargs in candidates:
            try:
                signature.bind(*selected_args, **selected_kwargs)
            except TypeError:
                continue
            return callable_assembler(*selected_args, **selected_kwargs)
        selected_args, selected_kwargs = candidates[0]
        return callable_assembler(*selected_args, **selected_kwargs)

    def assemble(
        self,
        key: ModelRuntimeKey | CompositeRuntimeKey,
        *args: object,
        providers: RuntimeProviders | None = None,
        **kwargs: object,
    ) -> RuntimeAssembly:
        if providers is None and args and isinstance(args[-1], RuntimeProviders):
            args, providers = args[:-1], args[-1]
        if providers is None:
            raise RuntimeDependencyError(
                "RuntimeProviders must be explicitly injected for runtime assembly",
                code="runtime_providers_required",
            )
        descriptor = self.require(key)
        try:
            result = self._invoke_assembler(descriptor, args, providers, dict(kwargs))
        except RuntimeRegistryError:
            raise
        except Exception as exc:
            raise RuntimeRegistryError(
                f"runtime assembler for {key!r} failed: {exc}",
                code="runtime_assembler_failed",
            ) from exc
        if not isinstance(result, RuntimeAssembly):
            raise RuntimeRegistryError(
                f"runtime assembler for {key!r} returned {type(result).__name__}, expected RuntimeAssembly",
                code="invalid_runtime_assembly",
            )
        self._validate_assembly_contract(descriptor, result)
        return result

    create = assemble

    def validate_local(self) -> None:
        for descriptor in self._runtime_descriptors.values():
            if descriptor.key.execution_contract != descriptor.execution_contract:
                raise RuntimeRegistryError(
                    f"runtime descriptor {descriptor.key!r} has a contract mismatch",
                    code="descriptor_contract_mismatch",
                )
            assembler_contract = descriptor.assembler_contract
            if assembler_contract is not None and assembler_contract != descriptor.execution_contract:
                raise RuntimeRegistryError(
                    f"runtime assembler for {descriptor.key!r} declares {assembler_contract!r}, "
                    f"expected {descriptor.execution_contract!r}",
                    code="descriptor_contract_mismatch",
                )
        for descriptor in self._composite_descriptors.values():
            if descriptor.key.execution_contract != descriptor.execution_contract:
                raise RuntimeRegistryError(
                    f"composite descriptor {descriptor.key!r} has a contract mismatch",
                    code="descriptor_contract_mismatch",
                )
            assembler_contract = descriptor.assembler_contract
            if assembler_contract is not None and assembler_contract != descriptor.execution_contract:
                raise RuntimeRegistryError(
                    f"composite assembler for {descriptor.key!r} declares {assembler_contract!r}, "
                    f"expected {descriptor.execution_contract!r}",
                    code="descriptor_contract_mismatch",
                )

    validate = validate_local

    def freeze(self) -> RuntimeAssemblerRegistry:
        self.validate_local()
        self._frozen = True
        return self


def _backend_names(registry: object) -> tuple[str, ...] | None:
    value = getattr(registry, "names", None)
    if callable(value):
        value = value()
    if value is not None:
        return tuple(str(item) for item in value)
    descriptors = getattr(registry, "descriptors", None)
    if isinstance(descriptors, Mapping):
        return tuple(str(item) for item in descriptors)
    return None


def _backend_exists(registry: object, backend: str) -> bool:
    names = _backend_names(registry)
    if names is not None:
        return backend in names
    descriptor = getattr(registry, "descriptor", None)
    if callable(descriptor):
        try:
            descriptor(backend)
        except Exception:
            return False
        return True
    if isinstance(registry, Mapping):
        return backend in registry
    # A deliberately minimal fake can opt out of static backend introspection;
    # profile and session checks still run below.
    return True


def _session_builder(registry: object, key: SessionBuilderKey) -> object | None:
    getter = getattr(registry, "get", None)
    if not callable(getter):
        return None
    try:
        return getter(key)
    except (TypeError, KeyError):
        try:
            return getter(key.interface, key.model_type, key.operation, key.backend)
        except (TypeError, KeyError):
            return None


def _profile_backend(profile_type: type | None) -> str | None:
    if profile_type is None:
        return None
    for name in ("backend_name", "backend"):
        value = getattr(profile_type, name, None)
        if isinstance(value, str):
            return value
    return None


def _profile_matches(expected: type | None, actual: type | None) -> bool:
    if expected is None or actual is None:
        return False
    if expected is actual:
        return True
    try:
        return issubclass(actual, expected) or issubclass(expected, actual)
    except TypeError:
        return False


def _backend_profile_types(registry: object, backend: str) -> tuple[type, ...] | None:
    descriptor = getattr(registry, "descriptor", None)
    if callable(descriptor):
        try:
            descriptor = descriptor(backend)
        except Exception:
            return None
    elif isinstance(registry, Mapping):
        descriptor = registry.get(backend)
    if descriptor is None:
        return None
    for name in ("profile_type", "backend_profile_type"):
        value = getattr(descriptor, name, None)
        if isinstance(value, type):
            return (value,)
    for name in ("profile_types", "supported_profile_types", "backend_profile_types"):
        value = getattr(descriptor, name, None)
        if value is None:
            continue
        if isinstance(value, Mapping):
            value = value.values()
        try:
            types = tuple(item for item in value if isinstance(item, type))
        except TypeError:
            continue
        return types
    return None


class RegistrySet:
    """The exact three-registry dependency passed through a composition root."""

    __slots__ = (
        "_backend_registry",
        "_session_builder_registry",
        "_runtime_assembler_registry",
        "_frozen",
        "_bootstraps",
    )

    def __init__(
        self,
        backend_registry: object | None = None,
        session_builder_registry: object | None = None,
        runtime_assembler_registry: RuntimeAssemblerRegistry | None = None,
        *,
        backend: object | None = None,
        session_builders: object | None = None,
        runtime_assemblers: RuntimeAssemblerRegistry | None = None,
    ) -> None:
        if backend_registry is None:
            backend_registry = backend
        if session_builder_registry is None:
            session_builder_registry = session_builders
        if runtime_assembler_registry is None:
            runtime_assembler_registry = runtime_assemblers
        if backend_registry is None or session_builder_registry is None or runtime_assembler_registry is None:
            raise TypeError("RegistrySet requires backend, session-builder, and runtime-assembler registries")
        self._backend_registry = backend_registry
        self._session_builder_registry = session_builder_registry
        self._runtime_assembler_registry = runtime_assembler_registry
        self._frozen = False
        self._bootstraps: dict[str, object] = {}

    @classmethod
    def mutable(cls, *args: object, **kwargs: object) -> RegistrySet:
        return cls(*args, **kwargs)

    create = mutable

    @property
    def backend_registry(self) -> object:
        return self._backend_registry

    @property
    def session_builder_registry(self) -> object:
        return self._session_builder_registry

    @property
    def runtime_assembler_registry(self) -> RuntimeAssemblerRegistry:
        return self._runtime_assembler_registry

    # Short aliases make composition-root wiring readable without changing the
    # exact three-member shape.
    backends = backend_registry
    backend = backend_registry
    session_builders = session_builder_registry
    session_builder = session_builder_registry
    runtime_assemblers = runtime_assembler_registry
    runtime_assembler = runtime_assembler_registry

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def _ensure_mutable(self) -> None:
        if self._frozen:
            raise RegistryFrozenError("RegistrySet is frozen")

    def _validate_role_descriptor(self, descriptor: RuntimeDescriptor) -> None:
        if descriptor.key.session_builder_key != descriptor.session_builder_key:
            raise RuntimeRegistryError(
                f"runtime descriptor {descriptor.key!r} references a mismatched session builder key",
                code="descriptor_session_key_mismatch",
            )
        if not _backend_exists(self._backend_registry, descriptor.key.backend):
            raise RuntimeRegistryError(
                f"backend {descriptor.key.backend!r} for {descriptor.key!r} is not registered",
                code="backend_not_registered",
            )
        if _session_builder(self._session_builder_registry, descriptor.session_builder_key) is None:
            raise RuntimeRegistryError(
                f"runtime descriptor {descriptor.key!r} references missing session builder "
                f"{descriptor.session_builder_key!r}",
                code="missing_session_builder",
            )
        if descriptor.profile_type is None:
            raise RuntimeRegistryError(
                f"runtime descriptor {descriptor.key!r} does not declare a backend profile type",
                code="role_backend_profile_mismatch",
            )
        profile_backend = _profile_backend(descriptor.profile_type)
        if profile_backend is not None and profile_backend != descriptor.key.backend:
            raise RuntimeRegistryError(
                f"runtime descriptor {descriptor.key!r} uses profile type for backend {profile_backend!r}",
                code="role_backend_profile_mismatch",
            )
        supported_profile_types = _backend_profile_types(self._backend_registry, descriptor.key.backend)
        if supported_profile_types and not any(
            _profile_matches(candidate, descriptor.profile_type) for candidate in supported_profile_types
        ):
            raise RuntimeRegistryError(
                f"runtime descriptor {descriptor.key!r} uses a profile type unsupported by backend "
                f"{descriptor.key.backend!r}",
                code="role_backend_profile_mismatch",
            )
        capability_contract = _capability_contract(descriptor.declared_capabilities)
        if capability_contract is not None and capability_contract != descriptor.execution_contract:
            raise RuntimeRegistryError(
                f"runtime descriptor {descriptor.key!r} declares capabilities for {capability_contract!r}, "
                f"expected {descriptor.execution_contract!r}",
                code="descriptor_contract_mismatch",
            )

    def _validate_composite_descriptor(self, descriptor: CompositeRuntimeDescriptor) -> None:
        selectors = tuple(descriptor.required_role_selectors)
        roles = [selector.role for selector in selectors]
        if not roles or any(not role for role in roles) or len(roles) != len(set(roles)):
            raise RuntimeRegistryError(
                f"composite descriptor {descriptor.key!r} must declare unique required role selectors",
                code="invalid_composite_role_selectors",
            )
        if any(selector.execution_contract != descriptor.execution_contract for selector in selectors):
            raise RuntimeRegistryError(
                f"composite descriptor {descriptor.key!r} has role selectors with inconsistent contracts",
                code="invalid_composite_role_selectors",
            )
        role_names = set(roles)
        entries = tuple(descriptor.role_compatibility_matrix)
        if not entries:
            raise RuntimeRegistryError(
                f"composite descriptor {descriptor.key!r} has no role backend/profile matrix",
                code="composite_role_backend_mismatch",
            )

        seen_rows: set[tuple[tuple[object, ...], ...]] = set()
        selector_by_role = {selector.role: selector for selector in selectors}
        for entry in entries:
            entry_roles = set(entry.roles)
            if entry_roles != role_names:
                raise RuntimeRegistryError(
                    f"composite descriptor {descriptor.key!r} matrix roles {sorted(entry_roles)} do not match "
                    f"required roles {sorted(role_names)}",
                    code="invalid_composite_role_selectors",
                )
            row_key = entry.canonical_tuple()
            if row_key in seen_rows:
                raise RuntimeRegistryError(
                    f"composite descriptor {descriptor.key!r} contains duplicate role compatibility rows",
                    code="duplicate_composite_role_matrix",
                )
            seen_rows.add(row_key)
            for role, compatibility in entry.roles.items():
                selector = selector_by_role[role]
                role_key = selector.runtime_key(
                    compatibility.backend,
                    execution_contract=descriptor.execution_contract,
                    orchestration_visibility=selector.orchestration_visibility,
                )
                role_descriptor = self._runtime_assembler_registry.get_runtime(role_key)
                if role_descriptor is None:
                    raise RuntimeRegistryError(
                        f"composite descriptor {descriptor.key!r} has no role descriptor for {role_key!r}",
                        code="composite_role_backend_mismatch",
                    )
                self._validate_role_descriptor(role_descriptor)
                if not _profile_matches(role_descriptor.profile_type, compatibility.profile_type):
                    raise RuntimeRegistryError(
                        f"composite role {role!r} backend/profile combination is incompatible with {role_key!r}",
                        code="composite_role_backend_mismatch",
                    )
                if compatibility.target_runtime is not None:
                    supported = role_descriptor.supported_target_runtimes
                    if (
                        role_descriptor.target_runtime is not None
                        and role_descriptor.target_runtime != compatibility.target_runtime
                    ):
                        raise RuntimeRegistryError(
                            f"composite role {role!r} target runtime does not match {role_key!r}",
                            code="composite_role_backend_mismatch",
                        )
                    if supported and compatibility.target_runtime not in supported:
                        raise RuntimeRegistryError(
                            f"composite role {role!r} target runtime is not supported by {role_key!r}",
                            code="composite_role_backend_mismatch",
                        )
                if compatibility.runtime_abi is not None:
                    supported_abis = role_descriptor.supported_runtime_abis
                    if (
                        role_descriptor.runtime_abi is not None
                        and role_descriptor.runtime_abi != compatibility.runtime_abi
                    ):
                        raise RuntimeRegistryError(
                            f"composite role {role!r} runtime ABI does not match {role_key!r}",
                            code="composite_role_backend_mismatch",
                        )
                    if supported_abis and compatibility.runtime_abi not in supported_abis:
                        raise RuntimeRegistryError(
                            f"composite role {role!r} runtime ABI is not supported by {role_key!r}",
                            code="composite_role_backend_mismatch",
                        )

        capability_contract = _capability_contract(descriptor.declared_capabilities)
        if capability_contract is not None and capability_contract != descriptor.execution_contract:
            raise RuntimeRegistryError(
                f"composite descriptor {descriptor.key!r} declares capabilities for {capability_contract!r}, "
                f"expected {descriptor.execution_contract!r}",
                code="descriptor_contract_mismatch",
            )

        supported_roles = getattr(descriptor.assembler, "supported_roles", None)
        if supported_roles is not None and set(supported_roles) != role_names:
            raise RuntimeRegistryError(
                f"composite assembler for {descriptor.key!r} does not support the declared role selectors",
                code="invalid_composite_role_selectors",
            )
        supported_selectors = getattr(descriptor.assembler, "supported_role_selectors", None)
        if supported_selectors is not None:
            if isinstance(supported_selectors, Mapping):
                normalized_supported = tuple(_selector(value, role=role) for role, value in supported_selectors.items())
            else:
                normalized_supported = tuple(_selector(value) for value in supported_selectors)
            expected = {
                (
                    selector.role,
                    selector.interface,
                    selector.model_type,
                    selector.operation,
                    selector.execution_contract,
                    selector.orchestration_visibility,
                )
                for selector in selectors
            }
            actual = {
                (
                    selector.role,
                    selector.interface,
                    selector.model_type,
                    selector.operation,
                    selector.execution_contract,
                    selector.orchestration_visibility,
                )
                for selector in normalized_supported
            }
            if actual != expected:
                raise RuntimeRegistryError(
                    f"composite assembler for {descriptor.key!r} does not support the declared role identities",
                    code="invalid_composite_role_selectors",
                )
        supported_combinations = getattr(descriptor.assembler, "supported_role_combinations", None)
        if supported_combinations is not None:
            actual_rows = {_entry.canonical_tuple() for _entry in _matrix_entries(supported_combinations)}
            expected_rows = {entry.canonical_tuple() for entry in entries}
            if actual_rows != expected_rows:
                raise RuntimeRegistryError(
                    f"composite assembler for {descriptor.key!r} does not support the declared backend/profile matrix",
                    code="composite_role_backend_mismatch",
                )

    def validate_cross_registry(self) -> None:
        self._runtime_assembler_registry.validate_local()
        for descriptor in self._runtime_assembler_registry.descriptors.values():
            self._validate_role_descriptor(descriptor)
        for descriptor in self._runtime_assembler_registry.composite_descriptors.values():
            self._validate_composite_descriptor(descriptor)

    validate = validate_cross_registry

    def freeze(self) -> RegistrySet:
        if self._frozen:
            return self
        self.validate_cross_registry()
        for registry in (self._backend_registry, self._session_builder_registry, self._runtime_assembler_registry):
            freezer = getattr(registry, "freeze", None)
            if callable(freezer):
                freezer()
        self._frozen = True
        return self

    def bootstrap(
        self,
        registrar: Callable[[RegistrySet], object] | None = None,
        *,
        bootstrap_id: str = "builtins",
        freeze: bool = False,
    ) -> RegistrySet:
        """Run one named registration callback at most once.

        The callback is recorded only after it succeeds (and, when requested,
        after validation/freeze succeeds), so a failed bootstrap remains
        inspectable and mutable for a corrected attempt.
        """

        name = _identifier(bootstrap_id, "bootstrap_id")
        token = registrar if registrar is not None else object()
        if name in self._bootstraps:
            previous = self._bootstraps[name]
            if previous is not registrar and registrar is not None:
                raise RuntimeRegistryError(
                    f"bootstrap id {name!r} was already used by another registrar",
                    code="bootstrap_conflict",
                )
            if freeze and not self._frozen:
                self.freeze()
            return self
        self._ensure_mutable()
        if registrar is not None:
            if not callable(registrar):
                raise TypeError("bootstrap registrar must be callable")
            registrar(self)
        if freeze:
            self.freeze()
        self._bootstraps[name] = token
        return self

    bootstrap_builtins = bootstrap


def bootstrap_registry_set(
    registry_set: RegistrySet,
    registrar: Callable[[RegistrySet], object] | None = None,
    *,
    bootstrap_id: str = "builtins",
    freeze: bool = False,
) -> RegistrySet:
    """Functional composition-root spelling for idempotent bootstrap."""

    return registry_set.bootstrap(registrar, bootstrap_id=bootstrap_id, freeze=freeze)


bootstrap_builtin_registries = bootstrap_registry_set


__all__ = [
    "CompositeRoleCompatibility",
    "CompositeRoleSelector",
    "CompositeRuntimeAssembler",
    "CompositeRuntimeDescriptor",
    "CompositeRuntimeKey",
    "CompositeRuntimeMatrixEntry",
    "ModelRuntimeKey",
    "ModelSessionBuilderKey",
    "RegistryFrozenError",
    "RegistrySet",
    "RoleBackendProfile",
    "RoleBackendProfileCompatibility",
    "RoleCompatibilityMatrixEntry",
    "RoleProfileCompatibility",
    "RoleSelector",
    "RuntimeAssembler",
    "RuntimeAssemblerRegistry",
    "RuntimeDependencyError",
    "RuntimeDescriptor",
    "RuntimeRegistryError",
    "RuntimeProviders",
    "RuntimeRoleSelector",
    "SessionBuilderKey",
    "SessionBuilderRegistry",
    "bootstrap_registry_set",
    "bootstrap_builtin_registries",
]
