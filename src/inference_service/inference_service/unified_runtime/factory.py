"""Explicit construction of unified model runtimes.

The factory is deliberately separate from the legacy pipeline factory.  It
accepts a validated runtime specification, resolves descriptors from an
injected :class:`RegistrySet`, and transfers a fully assembled
``RuntimeAssembly`` to ``ModelRuntimeHandle`` only after all checks succeed.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .adapters import ResultAdapter
from .assembly import OwnedComponent, RuntimeAssembly, RuntimeProviders
from .handle import ModelRuntimeHandle
from .registry import (
    CompositeRuntimeDescriptor,
    CompositeRuntimeKey,
    ModelRuntimeKey,
    RegistrySet,
    RoleBackendProfile,
    RuntimeDescriptor,
    RuntimeRegistryError,
    RuntimeRoleSelector,
    SessionBuilderKey,
)

_MISSING = object()
_VALID_CONTRACTS = frozenset(
    {
        "request-direct",
        "request-iterative",
        "stream-direct",
        "stream-iterative",
    }
)


class RuntimeFactoryError(RuntimeRegistryError):
    """Stable construction error raised before a handle is exposed."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "runtime_factory_error",
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code)
        self.details = MappingProxyType(dict(details or {}))


AssemblyValidationError = RuntimeFactoryError


def _get(value: object, *names: str, default: object = _MISSING) -> object:
    if value is None:
        return default
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeFactoryError(f"{name} must be a non-empty string", code="invalid_runtime_spec")
    return value.strip()


def _identity_parts(identity: object, *, label: str = "identity") -> tuple[str, str, str]:
    if isinstance(identity, tuple | list) and len(identity) == 3:
        interface, model_type, operation = identity
    else:
        interface = _get(identity, "interface")
        model_type = _get(identity, "model_type")
        operation = _get(identity, "operation")
    try:
        interface = _non_empty_string(interface, f"{label}.interface")
        model_type = _non_empty_string(model_type, f"{label}.model_type")
        operation = _non_empty_string(operation, f"{label}.operation")
    except RuntimeFactoryError:
        raise
    if interface not in {"policy", "tensor_model"}:
        raise RuntimeFactoryError(
            f"{label}.interface must be 'policy' or 'tensor_model'",
            code="invalid_runtime_identity",
        )
    return interface, model_type, operation


def _contract_name(contract: object) -> str:
    if isinstance(contract, str):
        name = contract.strip()
    elif isinstance(contract, Mapping):
        name_value = contract.get("name", contract.get("contract_name"))
        if name_value is None and {"state_scope", "execution_structure"}.issubset(contract):
            name_value = f"{contract['state_scope']}-{contract['execution_structure']}"
        name = str(name_value).strip() if name_value is not None else ""
    else:
        name_value = _get(contract, "name", "contract_name")
        if name_value is _MISSING or name_value is None:
            state_scope = _get(contract, "state_scope")
            execution_structure = _get(contract, "execution_structure")
            if state_scope is not _MISSING and execution_structure is not _MISSING:
                name_value = f"{state_scope}-{execution_structure}"
        name = str(name_value).strip() if name_value is not _MISSING and name_value is not None else ""
    if name not in _VALID_CONTRACTS:
        raise RuntimeFactoryError(
            f"execution_contract must be one of {sorted(_VALID_CONTRACTS)}, got {name!r}",
            code="invalid_execution_contract",
        )
    return name


def _visibility(contract: object, contract_name: str) -> str | None:
    value = _get(contract, "orchestration_visibility", default=None)
    if value is None:
        normalized = None
    elif isinstance(value, str):
        normalized = value.strip() or None
    else:
        raise RuntimeFactoryError(
            "orchestration_visibility must be a string or null", code="invalid_execution_contract"
        )
    if contract_name.endswith("-direct"):
        if normalized is not None:
            raise RuntimeFactoryError(
                "direct execution requires orchestration_visibility=None",
                code="invalid_execution_contract",
            )
        return None
    if normalized not in {"executor", "session"}:
        raise RuntimeFactoryError(
            "iterative execution requires orchestration_visibility='executor' or 'session'",
            code="invalid_execution_contract",
        )
    return normalized


def _canonical_target_runtime(value: object, backend: str) -> str:
    target = _non_empty_string(value, "target.runtime")
    if (
        target.startswith("acl-")
        or target == "om"
        or (target.isdigit() and len(target) == 4)
        or target.endswith("_acl")
        or target.endswith("_om")
        or "_om_" in target
    ):
        raise RuntimeFactoryError(
            "target.runtime must use the canonical runtime family; versioned ACL values belong in target.runtime_abi",
            code="invalid_target_runtime",
        )
    if backend == "ascend" and target != "acl":
        raise RuntimeFactoryError(
            "backend='ascend' requires target.runtime='acl'",
            code="backend_target_mismatch",
        )
    return target


@dataclass(frozen=True, slots=True)
class RuntimeRoleContext:
    """Immutable context passed to a role SessionBuilder or assembler."""

    spec: object
    runtime_spec: object
    role: str
    identity: object
    runtime_key: ModelRuntimeKey
    session_builder_key: SessionBuilderKey
    descriptor: RuntimeDescriptor
    profile: object
    backend: str
    target_runtime: str
    runtime_abi: str | None
    deployment: object | None
    validated_deployment: object | None
    artifact_bindings: object | None
    artifact_handles: Mapping[str, object]
    providers: RuntimeProviders

    @property
    def role_runtime_spec(self) -> object:
        return self.runtime_spec

    @property
    def backend_profile(self) -> object:
        return self.profile

    @property
    def artifacts(self) -> Mapping[str, object]:
        return self.artifact_handles


@dataclass(frozen=True, slots=True)
class _RoleRuntimeInfo:
    role: str
    runtime_spec: object
    profile: object
    backend: str
    target_runtime: str
    runtime_abi: str | None
    identity: object
    runtime_key: ModelRuntimeKey
    session_builder_key: SessionBuilderKey


def _profile_info(
    value: object,
    *,
    role: str,
    backend_override: object = _MISSING,
    target_runtime_override: object = _MISSING,
    runtime_abi_override: object = _MISSING,
) -> tuple[object, str, str, str | None]:
    """Extract a role profile from either a typed spec or a small test value."""

    backend_value = _get(value, "backend", default=_MISSING)
    if backend_value is _MISSING:
        backend_value = backend_override
    profile = _get(value, "backend_profile", "profile", "runtime_profile", default=_MISSING)
    if profile is _MISSING:
        # A typed BackendRuntimeProfile is itself a valid profile value.
        profile = value
    if backend_value is _MISSING:
        backend_value = _get(profile, "backend", "backend_name", default=_MISSING)
    backend = _non_empty_string(backend_value, f"role {role!r} backend")
    profile_backend = _get(profile, "backend", "backend_name", default=_MISSING)
    if profile_backend is not _MISSING and profile_backend is not None and str(profile_backend) != backend:
        raise RuntimeFactoryError(
            f"role {role!r} backend {backend!r} does not match its profile backend {profile_backend!r}",
            code="role_backend_profile_mismatch",
        )
    if (
        backend.startswith("acl-")
        or backend == "om"
        or (backend.isdigit() and len(backend) == 4)
        or backend.endswith("_acl")
        or backend.endswith("_om")
        or "_om_" in backend
    ):
        raise RuntimeFactoryError(
            f"legacy backend identity {backend!r} is not supported by the unified runtime",
            code="invalid_backend",
        )

    target_value = _get(value, "target_runtime", default=_MISSING)
    if target_value is _MISSING:
        target_value = target_runtime_override
    if target_value is _MISSING:
        target = _get(value, "target", default=_MISSING)
        target_value = _get(target, "runtime", default=_MISSING)
    if target_value is _MISSING:
        target_value = _get(profile, "target_runtime", default=_MISSING)
    if target_value is _MISSING:
        # ModelRuntimeSpec normalizes this value for non-Ascend profiles.  The
        # fallback keeps the factory usable with equivalent lightweight specs.
        target_value = "acl" if backend == "ascend" else backend
    target_runtime = _canonical_target_runtime(target_value, backend)

    abi_value = _get(value, "runtime_abi", default=_MISSING)
    if abi_value is _MISSING:
        abi_value = runtime_abi_override
    if abi_value is _MISSING:
        target = _get(value, "target", default=_MISSING)
        abi_value = _get(target, "runtime_abi", default=_MISSING)
    if abi_value is _MISSING:
        abi_value = _get(profile, "runtime_abi", default=None)
    runtime_abi = None if abi_value is None or abi_value is _MISSING else _non_empty_string(abi_value, "runtime_abi")
    if runtime_abi is not None and not any(character.isdigit() for character in runtime_abi):
        raise RuntimeFactoryError(
            "runtime_abi must contain a version identifier",
            code="invalid_runtime_abi",
        )

    if profile is _MISSING or profile is None:
        raise RuntimeFactoryError(
            f"role {role!r} does not provide a typed backend profile",
            code="runtime_profile_missing",
        )
    return profile, backend, target_runtime, runtime_abi


def _find_snapshot(spec: object) -> object | None:
    candidates = [spec]
    for name in ("validated_deployment", "deployment_snapshot", "snapshot", "deployment"):
        value = _get(spec, name, default=_MISSING)
        if value is not _MISSING and value is not None and not any(value is candidate for candidate in candidates):
            candidates.append(value)
    for candidate in candidates:
        if any(
            _get(candidate, name, default=_MISSING) is not _MISSING
            for name in ("top_level_identity", "selected_deployment")
        ):
            return candidate
    return None


def _selected_deployment(spec: object, snapshot: object | None) -> object | None:
    for candidate in (snapshot, spec):
        selected = _get(candidate, "selected_deployment", "deployment", default=_MISSING)
        if selected is not _MISSING and selected is not None and selected is not spec:
            return selected
    return None


def _top_level_identity(spec: object, snapshot: object | None, deployment: object | None) -> object:
    candidates = [
        _get(snapshot, "top_level_identity", "identity", default=_MISSING),
        _get(spec, "top_level_identity", "identity", default=_MISSING),
        _get(deployment, "top_level_identity", "identity", default=_MISSING),
    ]
    manifest = _get(snapshot, "manifest", default=_MISSING)
    model = _get(manifest, "model", default=_MISSING)
    if model is not _MISSING:
        candidates.extend((_get(model, "identity", default=_MISSING), model))
    candidates.extend((_get(spec, "model", default=_MISSING), _get(deployment, "model", default=_MISSING)))
    for candidate in candidates:
        if candidate is not _MISSING and candidate is not None:
            _identity_parts(candidate)
            return candidate
    raise RuntimeFactoryError(
        "runtime spec does not expose a top-level model identity",
        code="runtime_identity_missing",
    )


def _role_identities(spec: object, snapshot: object | None, deployment: object | None) -> Mapping[str, object]:
    for candidate in (snapshot, deployment, spec):
        value = _get(candidate, "role_identities", default=_MISSING)
        if isinstance(value, Mapping):
            return value
    return {}


def _runtime_specs(spec: object, snapshot: object | None, deployment: object | None) -> Mapping[str, object]:
    value = _get(spec, "role_runtime_specs", default=_MISSING)
    if isinstance(value, Mapping):
        return value
    profiles = _get(snapshot, "role_runtime_profiles", default=_MISSING)
    if not isinstance(profiles, Mapping):
        profiles = _get(deployment, "role_runtime_profiles", default=_MISSING)
    return profiles if isinstance(profiles, Mapping) else {}


def _single_runtime_profile(spec: object, snapshot: object | None, deployment: object | None) -> object:
    for candidate in (spec, snapshot, deployment):
        value = _get(candidate, "runtime_profile", "profile", default=_MISSING)
        if value is not _MISSING and value is not None:
            return value
    raise RuntimeFactoryError(
        "single-model runtime spec does not provide runtime_profile",
        code="runtime_profile_missing",
    )


def _execution_contract(spec: object, deployment: object | None, snapshot: object | None) -> object:
    for candidate in (spec, deployment, snapshot):
        value = _get(candidate, "execution_contract", "contract", default=_MISSING)
        if value is not _MISSING and value is not None:
            return value
    raise RuntimeFactoryError(
        "runtime spec does not provide execution_contract",
        code="execution_contract_missing",
    )


def _artifact_data(
    snapshot: object | None, deployment: object | None
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    bindings: object = _MISSING
    handles: object = _MISSING
    for candidate in (snapshot, deployment):
        if bindings is _MISSING:
            bindings = _get(
                candidate, "role_artifact_bindings", "role_to_artifact_bindings", "bindings", default=_MISSING
            )
        if handles is _MISSING:
            handles = _get(
                candidate, "resolved_artifacts", "resolved_artifact_handles", "artifact_handles", default=_MISSING
            )
    return (
        dict(bindings) if isinstance(bindings, Mapping) else {},
        dict(handles) if isinstance(handles, Mapping) else {},
    )


def _registry_value(registry_set: RegistrySet, name: str) -> object:
    value = _get(registry_set, name, default=_MISSING)
    if value is _MISSING or value is None:
        raise RuntimeFactoryError(
            f"RegistrySet does not expose {name}",
            code="registry_set_invalid",
        )
    return value


def _lookup(registry: object, key: object, *, required_code: str, label: str) -> object:
    getter = _get(registry, "get", default=_MISSING)
    if callable(getter):
        try:
            value = getter(key)
        except (TypeError, KeyError):
            value = None
        if value is not None:
            return value
    resolver = _get(registry, "resolve", "require", default=_MISSING)
    if callable(resolver):
        try:
            return resolver(key)
        except RuntimeRegistryError as exc:
            if getattr(exc, "code", None) != required_code:
                raise
    raise RuntimeFactoryError(
        f"no {label} is registered for {key!r}",
        code=required_code,
        details={"key": key.to_dict() if hasattr(key, "to_dict") else repr(key)},
    )


def _assembler_callable(assembler: object) -> Callable[..., object]:
    if callable(assembler):
        return assembler  # type: ignore[return-value]
    method = _get(assembler, "assemble", default=_MISSING)
    if callable(method):
        return method  # type: ignore[return-value]
    raise RuntimeFactoryError("runtime descriptor assembler is not callable", code="invalid_runtime_descriptor")


def _parameter_value(name: str, payload: Mapping[str, object], fallback: list[object]) -> object:
    if name in payload:
        return payload[name]
    normalized = name.lstrip("_").lower()
    aliases = {
        "context": "role_context",
        "buildcontext": "role_context",
        "rolecontext": "role_context",
        "runtimecontext": "role_context",
        "runtimespec": "runtime_spec",
        "modelruntimespec": "runtime_spec",
        "specification": "spec",
        "runtimekey": "runtime_key",
        "sessionbuilderkey": "session_builder_key",
        "backendprofile": "profile",
        "roleassembly": "assembly",
        "assemblies": "role_assemblies",
        "sessions": "sessions",
    }
    selected = aliases.get(normalized, normalized)
    if selected in payload:
        return payload[selected]
    if fallback:
        return fallback.pop(0)
    return _MISSING


def _invoke(target: Callable[..., object], payload: Mapping[str, object], fallback: Iterable[object] = ()) -> object:
    """Invoke a user-supplied builder using its declared parameter names.

    Builders from different backends commonly use ``spec``, ``context``, or
    a small positional signature.  Signature binding lets the factory support
    those shapes without catching a TypeError raised by the builder itself.
    """

    fallback_values = list(fallback)
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return target(*fallback_values)

    args: list[object] = []
    kwargs: dict[str, object] = {}
    positional_names: set[str] = set()
    has_varargs = False
    has_varkw = False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            has_varargs = True
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            has_varkw = True
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            value = _parameter_value(parameter.name, payload, fallback_values)
            if value is _MISSING:
                if parameter.default is inspect.Parameter.empty:
                    raise TypeError(f"cannot supply required builder parameter {parameter.name!r}")
                continue
            args.append(value)
            positional_names.add(parameter.name)
            continue
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
            value = _parameter_value(
                parameter.name,
                payload,
                fallback_values if parameter.default is inspect.Parameter.empty else [],
            )
            if value is _MISSING:
                if parameter.default is inspect.Parameter.empty:
                    raise TypeError(f"cannot supply required builder parameter {parameter.name!r}")
                continue
            if parameter.default is inspect.Parameter.empty:
                args.append(value)
                positional_names.add(parameter.name)
            else:
                kwargs[parameter.name] = value
            continue
        value = _parameter_value(
            parameter.name,
            payload,
            fallback_values if parameter.default is inspect.Parameter.empty else [],
        )
        if value is _MISSING:
            if parameter.default is inspect.Parameter.empty:
                raise TypeError(f"cannot supply required builder parameter {parameter.name!r}")
            continue
        kwargs[parameter.name] = value

    if has_varargs:
        args.extend(fallback_values)
    if has_varkw:
        kwargs.update({name: value for name, value in payload.items() if name not in positional_names})
    return target(*args, **kwargs)


def _profile_matches(expected: type | None, actual: object) -> bool:
    if expected is None:
        return True
    try:
        return isinstance(actual, expected)
    except TypeError:
        return type(actual) is expected


def _descriptor_contract(descriptor: object) -> str | None:
    value = _get(descriptor, "execution_contract", default=None)
    if value is None:
        capabilities = _get(descriptor, "declared_capabilities", "capabilities", default={})
        value = _get(capabilities, "execution_contract", "contract", "contract_name", default=None)
    return None if value is None else _contract_name(value)


def _validate_role_descriptor(
    descriptor: RuntimeDescriptor,
    info: _RoleRuntimeInfo,
) -> None:
    if descriptor.key != info.runtime_key:
        raise RuntimeFactoryError(
            f"runtime descriptor key {descriptor.key!r} does not match resolved key {info.runtime_key!r}",
            code="runtime_descriptor_key_mismatch",
            details={"role": info.role},
        )
    if descriptor.session_builder_key != info.session_builder_key:
        raise RuntimeFactoryError(
            f"runtime descriptor for role {info.role!r} references a different SessionBuilderKey",
            code="descriptor_session_key_mismatch",
        )
    profile_type = _get(descriptor, "profile_type", "backend_profile_type", "profile", default=None)
    if not _profile_matches(profile_type if isinstance(profile_type, type) else None, info.profile):
        raise RuntimeFactoryError(
            f"role {info.role!r} backend/profile is incompatible with runtime descriptor",
            code="role_backend_profile_mismatch",
            details={"role": info.role, "backend": info.backend, "profile": type(info.profile).__name__},
        )
    descriptor_backend = _get(descriptor.key, "backend")
    if descriptor_backend != info.backend:
        raise RuntimeFactoryError(
            f"role {info.role!r} selects backend {info.backend!r}, descriptor requires {descriptor_backend!r}",
            code="role_backend_profile_mismatch",
        )
    target_runtime = _get(descriptor, "target_runtime", default=None)
    supported_targets = _get(descriptor, "supported_target_runtimes", default=frozenset())
    if target_runtime is not None and target_runtime != info.target_runtime:
        raise RuntimeFactoryError(
            f"role {info.role!r} target runtime {info.target_runtime!r} does not match descriptor {target_runtime!r}",
            code="role_backend_profile_mismatch",
        )
    if supported_targets and info.target_runtime not in supported_targets:
        raise RuntimeFactoryError(
            f"role {info.role!r} target runtime {info.target_runtime!r} is unsupported",
            code="role_backend_profile_mismatch",
        )
    descriptor_abi = _get(descriptor, "runtime_abi", default=None)
    supported_abis = _get(descriptor, "supported_runtime_abis", default=frozenset())
    if descriptor_abi is not None and descriptor_abi != info.runtime_abi:
        raise RuntimeFactoryError(
            f"role {info.role!r} runtime ABI {info.runtime_abi!r} does not match descriptor {descriptor_abi!r}",
            code="role_backend_profile_mismatch",
        )
    if supported_abis and info.runtime_abi not in supported_abis:
        raise RuntimeFactoryError(
            f"role {info.role!r} runtime ABI {info.runtime_abi!r} is unsupported",
            code="role_backend_profile_mismatch",
        )
    expected_contract = _descriptor_contract(descriptor)
    if expected_contract is not None and expected_contract != info.runtime_key.execution_contract:
        raise RuntimeFactoryError(
            f"runtime descriptor contract {expected_contract!r} does not match {info.runtime_key.execution_contract!r}",
            code="descriptor_contract_mismatch",
        )


def _selector_values(descriptor: CompositeRuntimeDescriptor) -> tuple[RuntimeRoleSelector, ...]:
    selectors = _get(descriptor, "required_role_selectors", "role_selectors", default=())
    if isinstance(selectors, Mapping):
        values = tuple(selectors.values())
    elif isinstance(selectors, RuntimeRoleSelector):
        values = (selectors,)
    else:
        values = tuple(selectors or ())
    return values


def _validate_composite_selectors(
    descriptor: CompositeRuntimeDescriptor,
    identities: Mapping[str, object],
    contract_name: str,
    visibility: str | None,
) -> None:
    selectors = _selector_values(descriptor)
    if not selectors:
        raise RuntimeFactoryError(
            "composite descriptor has no required role selectors",
            code="invalid_composite_role_selectors",
        )
    selector_roles = [str(_get(selector, "role", default="")) for selector in selectors]
    if len(selector_roles) != len(set(selector_roles)) or set(selector_roles) != set(identities):
        raise RuntimeFactoryError(
            "composite role selectors do not match declared role identities",
            code="invalid_composite_role_selectors",
            details={"selectors": selector_roles, "roles": sorted(identities)},
        )
    for selector in selectors:
        role = str(_get(selector, "role"))
        expected = _identity_parts(selector, label=f"role selector {role!r}")
        actual = _identity_parts(identities[role], label=f"role {role!r}")
        if expected != actual:
            raise RuntimeFactoryError(
                f"composite role selector {role!r} does not match the declared role identity",
                code="composite_role_selector_mismatch",
            )
        selector_contract = _get(selector, "execution_contract", default=None)
        if selector_contract is not None and _contract_name(selector_contract) != contract_name:
            raise RuntimeFactoryError(
                f"composite role selector {role!r} has a different execution contract",
                code="descriptor_contract_mismatch",
            )
        selector_visibility = _get(selector, "orchestration_visibility", default=None)
        if selector_visibility not in (None, "", visibility):
            raise RuntimeFactoryError(
                f"composite role selector {role!r} has a different orchestration visibility",
                code="descriptor_contract_mismatch",
            )


def _matrix_entries(descriptor: CompositeRuntimeDescriptor) -> tuple[object, ...]:
    value = _get(descriptor, "role_compatibility_matrix", "role_backend_profile_matrix", default=())
    if isinstance(value, Mapping):
        return (value,)
    if isinstance(value, RoleBackendProfile):
        role = _get(value, "role", default=None)
        return ({role: value},) if role is not None else ()
    return tuple(value or ())


def _matrix_roles(entry: object) -> Mapping[str, object]:
    roles = _get(entry, "roles", "role_backend_profiles", default=_MISSING)
    if isinstance(roles, Mapping):
        return roles
    return entry if isinstance(entry, Mapping) else {}


def _validate_composite_matrix(
    descriptor: CompositeRuntimeDescriptor,
    role_infos: Mapping[str, _RoleRuntimeInfo],
) -> None:
    entries = _matrix_entries(descriptor)
    if not entries:
        raise RuntimeFactoryError(
            "composite descriptor has no role backend/profile compatibility matrix",
            code="composite_role_backend_mismatch",
        )
    for entry in entries:
        expected_roles = _matrix_roles(entry)
        if set(expected_roles) != set(role_infos):
            continue
        row_matches = True
        for role, info in role_infos.items():
            expected = expected_roles[role]
            expected_backend = _get(expected, "backend", default=_MISSING)
            if expected_backend is not _MISSING and expected_backend != info.backend:
                row_matches = False
                break
            expected_profile = _get(expected, "profile_type", "profile", "backend_profile_type", default=None)
            if isinstance(expected_profile, type) and not _profile_matches(expected_profile, info.profile):
                row_matches = False
                break
            expected_target = _get(expected, "target_runtime", default=None)
            if expected_target is not None and expected_target != info.target_runtime:
                row_matches = False
                break
            expected_abi = _get(expected, "runtime_abi", default=None)
            if expected_abi is not None and expected_abi != info.runtime_abi:
                row_matches = False
                break
        if row_matches:
            return
    raise RuntimeFactoryError(
        "declared role backend/profile combination is not supported by the composite descriptor",
        code="composite_role_backend_mismatch",
        details={
            "roles": {
                role: {
                    "backend": info.backend,
                    "target_runtime": info.target_runtime,
                    "runtime_abi": info.runtime_abi,
                    "profile": type(info.profile).__name__,
                }
                for role, info in sorted(role_infos.items())
            }
        },
    )


def _role_info(
    role: str,
    identity: object,
    runtime_spec: object,
    contract_name: str,
    visibility: str | None,
) -> _RoleRuntimeInfo:
    profile, backend, target_runtime, runtime_abi = _profile_info(runtime_spec, role=role)
    runtime_key = ModelRuntimeKey(
        *_identity_parts(identity, label=f"role {role!r}"),
        backend,
        contract_name,
        visibility,
    )
    return _RoleRuntimeInfo(
        role,
        runtime_spec,
        profile,
        backend,
        target_runtime,
        runtime_abi,
        identity,
        runtime_key,
        runtime_key.session_builder_key,
    )


def _release(resource: OwnedComponent) -> None:
    if resource.release is not None:
        resource.release()
        return
    method_name = resource.release_method
    if method_name is not None:
        method = getattr(resource.resource, method_name, None)
        if callable(method):
            method()
            return
    for name in ("close", "release", "unregister"):
        method = getattr(resource.resource, name, None)
        if callable(method):
            method()
            return


def _deduplicate(entries: Iterable[OwnedComponent]) -> tuple[OwnedComponent, ...]:
    result: list[OwnedComponent] = []
    seen: set[int] = set()
    for entry in entries:
        if id(entry.resource) in seen:
            continue
        seen.add(id(entry.resource))
        result.append(entry)
    return tuple(result)


def _rollback(entries: Iterable[OwnedComponent]) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []
    for entry in reversed(_deduplicate(entries)):
        try:
            _release(entry)
        except BaseException as exc:
            errors.append(exc)
    return tuple(errors)


def _assembly_from_result(
    result: object,
    *,
    session: object | None,
    role: str | None,
    contract: object,
    descriptor: RuntimeDescriptor | CompositeRuntimeDescriptor,
) -> RuntimeAssembly:
    if isinstance(result, RuntimeAssembly):
        assembly = result
    elif isinstance(result, Mapping):
        executor = result.get("runtime_executor", result.get("executor"))
        if executor is None:
            executor = result.get("runtime")
        if executor is None and session is not None:
            executor = session
        if executor is None:
            raise RuntimeFactoryError(
                "assembler result does not contain runtime_executor",
                code="invalid_runtime_assembly",
            )
        assembly = RuntimeAssembly(
            runtime_executor=executor,
            streaming_runtime=result.get("streaming_runtime"),
            session=result.get("session", session),
            role_assemblies=result.get("role_assemblies", {}),
            artifact_bindings=result.get("artifact_bindings", {}),
            request_adapter=result.get("request_adapter", result.get("result_adapter")),
            failure_factory=result.get("failure_factory"),
            processor=result.get("processor"),
            worker=result.get("worker"),
            host_resources=tuple(result.get("host_resources", ())),
            provider_leases=tuple(result.get("provider_leases", ())),
            provider_registrations=tuple(result.get("provider_registrations", ())),
            provider_leases_and_registrations=tuple(result.get("provider_leases_and_registrations", ())),
            owned_components=tuple(result.get("owned_components", ())),
            stateful=bool(result.get("stateful", False)),
            resettable=bool(result.get("resettable", False)),
            state_scope=str(result.get("state_scope", "request")),
            state_bank_mode=result.get("state_bank_mode"),
            max_open_streams=result.get("max_open_streams"),
            cancellation_granularity=str(result.get("cancellation_granularity", "request_boundary")),
            declared_capabilities=result.get("declared_capabilities", {}),
            runtime_id=result.get("runtime_id"),
        )
    else:
        if result is None:
            raise RuntimeFactoryError("assembler returned no runtime", code="invalid_runtime_assembly")
        executor = _get(result, "runtime_executor", "executor", default=result)
        assembly = RuntimeAssembly(runtime_executor=executor, session=session)

    if assembly.ownership_transferred:
        raise RuntimeFactoryError(
            "assembler returned an assembly whose ownership was already transferred",
            code="assembly_ownership_conflict",
        )
    if assembly.runtime_executor is None:
        raise RuntimeFactoryError("RuntimeAssembly requires runtime_executor", code="invalid_runtime_assembly")
    if assembly.execution_contract is None:
        assembly.execution_contract = contract
    capabilities = _get(descriptor, "declared_capabilities", "capabilities", default={})
    declared = dict(capabilities) if isinstance(capabilities, Mapping) else {}
    assembly_capabilities = _get(assembly, "declared_capabilities", default={})
    if isinstance(assembly_capabilities, Mapping):
        declared.update(assembly_capabilities)
    assembly.declared_capabilities = declared
    if role is not None:
        if assembly.session is None:
            assembly.session = session
        role_map = dict(assembly.role_assemblies)
        role_map.setdefault(role, assembly)
        assembly.role_assemblies = role_map
    return assembly


def _validate_assembly_contract(assembly: RuntimeAssembly, contract: object, *, role: str | None = None) -> None:
    expected = _contract_name(contract)
    actual = _get(assembly, "execution_contract", default=None)
    if actual is None or _contract_name(actual) != expected:
        label = f" for role {role!r}" if role is not None else ""
        raise RuntimeFactoryError(
            f"RuntimeAssembly contract{label} does not match {expected!r}",
            code="descriptor_assembly_contract_mismatch",
        )
    capabilities = assembly.declared_capabilities
    stateful = bool(assembly.stateful or _get(capabilities, "stateful", default=False))
    state_scope = _get(contract, "state_scope", default="request")
    if state_scope == "request" and stateful:
        raise RuntimeFactoryError(
            f"request-scoped runtime cannot use a stateful Session{label}",
            code="stateful_request_runtime",
        )


def _assembly_entries(assembly: RuntimeAssembly, *, process_provider_ids: set[int]) -> tuple[OwnedComponent, ...]:
    selected_provider_ids = set(process_provider_ids)
    if isinstance(assembly.providers, RuntimeProviders):
        selected_provider_ids.update(
            (id(assembly.providers.acl_runtime_provider), id(assembly.providers.resource_admission_provider))
        )
    selected_provider_ids.update(id(provider) for provider in assembly.process_providers)
    return tuple(
        entry
        for entry in assembly.component_entries()
        if id(entry.resource) not in selected_provider_ids and not isinstance(entry.resource, RuntimeAssembly)
    )


def _execution_roles(deployment: object | None, fallback: Iterable[str]) -> tuple[str, ...]:
    declared = _get(deployment, "execution", default=())
    if isinstance(declared, Sequence) and not isinstance(declared, str | bytes) and declared:
        return tuple(str(role) for role in declared)
    return tuple(fallback)


class RequestDirectExecutor:
    """Execute one session or a fixed acyclic sequence of role executors."""

    def __init__(
        self,
        targets: Mapping[str, object],
        *,
        execution_roles: Sequence[str] = (),
    ) -> None:
        if not targets:
            raise ValueError("request-direct execution requires at least one target")
        self._targets = dict(targets)
        self._execution_roles = tuple(execution_roles) or tuple(targets)
        missing = [role for role in self._execution_roles if role not in self._targets]
        if missing:
            raise ValueError(f"execution roles have no target: {missing}")

    def execute(self, request, context):
        if len(self._execution_roles) == 1:
            return _invoke_execution(self._targets[self._execution_roles[0]], request, context)
        values = dict(request.inputs)
        final: object = None
        for role in self._execution_roles:
            role_request = type(request)(values, request.metadata)
            final = _invoke_execution(self._targets[role], role_request, context)
            outputs = _outputs(final)
            if isinstance(outputs, Mapping):
                values.update(outputs)
        if isinstance(final, Mapping) and "outputs" in final:
            return final
        return {"outputs": _outputs(final)}


def _outputs(value: object) -> object:
    if isinstance(value, Mapping) and "outputs" in value:
        return value["outputs"]
    return _get(value, "outputs", default=value)


def _invoke_execution(target: object, request: object, context: object) -> object:
    candidate = _get(target, "runtime_executor", default=target)
    for name in ("execute", "forward", "predict", "infer"):
        method = _get(candidate, name, default=_MISSING)
        if callable(method):
            payload = {
                "request": request,
                "model_request": request,
                "context": context,
                "execution_context": context,
                "inputs": request.inputs,
                "metadata": request.metadata,
            }
            return _invoke(method, payload, (request, context))
    if callable(candidate):
        return _invoke(
            candidate, {"request": request, "context": context, "inputs": request.inputs}, (request, context)
        )
    raise TypeError(f"runtime target {type(candidate).__name__} has no execute/forward/predict/infer operation")


class RequestDirectAssembler:
    """Default assembler for request-direct session and fixed-DAG shapes."""

    execution_contract = "request-direct"

    def assemble(
        self,
        session: object | None = None,
        *,
        executor: object | None = None,
        role: str = "__default__",
        role_assemblies: Mapping[str, RuntimeAssembly] | None = None,
        execution_roles: Sequence[str] = (),
        contract: object | None = None,
        artifact_bindings: Mapping[str, object] | None = None,
    ) -> RuntimeAssembly:
        if role_assemblies:
            targets = {
                name: _get(value, "runtime_executor", "executor", default=value)
                for name, value in role_assemblies.items()
            }
            executor = RequestDirectExecutor(targets, execution_roles=execution_roles)
            return RuntimeAssembly(
                runtime_executor=executor,
                role_assemblies=dict(role_assemblies),
                artifact_bindings=dict(artifact_bindings or {}),
                execution_contract=contract or self.execution_contract,
            )
        if executor is not None:
            return RuntimeAssembly(
                runtime_executor=executor,
                execution_contract=contract or self.execution_contract,
                artifact_bindings=dict(artifact_bindings or {}),
            )
        if session is None:
            raise ValueError("request-direct assembler requires a session")
        executor = RequestDirectExecutor({role: session})
        return RuntimeAssembly(
            runtime_executor=executor,
            session=session,
            execution_contract=contract or self.execution_contract,
            artifact_bindings=dict(artifact_bindings or {}),
        )

    __call__ = assemble


DefaultRequestDirectAssembler = RequestDirectAssembler


class RequestIterativeAssembler:
    """Small explicit wrapper for request-iterative executor/session paths."""

    execution_contract = "request-iterative"

    def assemble(
        self,
        session: object | None = None,
        *,
        executor: object | None = None,
        contract: object | None = None,
    ) -> RuntimeAssembly:
        target = executor or session
        if target is None:
            raise ValueError("request-iterative assembler requires an executor or session")
        return RuntimeAssembly(
            runtime_executor=target,
            session=session,
            execution_contract=contract or self.execution_contract,
        )

    __call__ = assemble


def assemble_request_direct(*args: object, **kwargs: object) -> RuntimeAssembly:
    """Functional spelling for the default request-direct assembler."""

    return RequestDirectAssembler().assemble(*args, **kwargs)


def assemble_request_iterative(*args: object, **kwargs: object) -> RuntimeAssembly:
    """Functional spelling for the explicit request-iterative wrapper."""

    return RequestIterativeAssembler().assemble(*args, **kwargs)


def _invoke_role_assembler(
    descriptor: RuntimeDescriptor,
    context: RuntimeRoleContext,
    session: object,
    contract: object,
) -> object:
    assembler = _assembler_callable(getattr(descriptor, "assembler", None))
    payload = {
        "session": session,
        "role": context.role,
        "identity": context.identity,
        "key": context.runtime_key,
        "runtime_key": context.runtime_key,
        "descriptor": descriptor,
        "spec": context.spec,
        "runtime_spec": context.runtime_spec,
        "role_context": context,
        "context": context,
        "profile": context.profile,
        "backend_profile": context.profile,
        "providers": context.providers,
        "deployment": context.deployment,
        "validated_deployment": context.validated_deployment,
        "artifact_bindings": context.artifact_bindings,
        "artifact_handles": context.artifact_handles,
        "contract": contract,
    }
    return _invoke(assembler, payload, (session, context.runtime_spec, context))


def _invoke_composite_assembler(
    descriptor: CompositeRuntimeDescriptor,
    *,
    role_assemblies: Mapping[str, RuntimeAssembly],
    role_contexts: Mapping[str, RuntimeRoleContext],
    spec: object,
    deployment: object | None,
    providers: RuntimeProviders,
    contract: object,
    artifact_bindings: Mapping[str, object],
    execution_roles: Sequence[str],
) -> object:
    assembler = _assembler_callable(getattr(descriptor, "assembler", None))
    payload = {
        "role_assemblies": role_assemblies,
        "assemblies": role_assemblies,
        "sessions": {role: assembly.session for role, assembly in role_assemblies.items()},
        "role_contexts": role_contexts,
        "spec": spec,
        "runtime_spec": spec,
        "deployment": deployment,
        "descriptor": descriptor,
        "providers": providers,
        "contract": contract,
        "artifact_bindings": artifact_bindings,
        "execution_roles": execution_roles,
    }
    return _invoke(assembler, payload, (role_assemblies, role_contexts, spec))


class ModelRuntimeFactory:
    """Construct a ``ModelRuntimeHandle`` from typed runtime data."""

    @staticmethod
    def create(
        spec: object,
        registry_set: RegistrySet,
        providers: RuntimeProviders,
    ) -> ModelRuntimeHandle:
        if registry_set is None:
            raise RuntimeFactoryError("RegistrySet must be explicitly supplied", code="registry_set_required")
        if providers is None:
            raise RuntimeFactoryError("RuntimeProviders must be explicitly supplied", code="runtime_providers_required")
        if not isinstance(registry_set, RegistrySet):
            raise RuntimeFactoryError(
                "registry_set must be a RegistrySet value",
                code="registry_set_invalid",
            )
        if not isinstance(providers, RuntimeProviders):
            raise RuntimeFactoryError(
                "providers must be a RuntimeProviders value",
                code="runtime_providers_invalid",
            )

        snapshot = _find_snapshot(spec)
        deployment = _selected_deployment(spec, snapshot)
        identity = _top_level_identity(spec, snapshot, deployment)
        contract = _execution_contract(spec, deployment, snapshot)
        contract_name = _contract_name(contract)
        visibility = _visibility(contract, contract_name)
        role_identities = dict(_role_identities(spec, snapshot, deployment))
        role_specs = dict(_runtime_specs(spec, snapshot, deployment))
        composite = bool(role_identities or role_specs)
        bindings, artifact_handles = _artifact_data(snapshot, deployment)

        session_registry = _registry_value(registry_set, "session_builder_registry")
        assembler_registry = _registry_value(registry_set, "runtime_assembler_registry")
        rollback_entries: list[OwnedComponent] = []

        try:
            if composite:
                if not role_identities:
                    raise RuntimeFactoryError(
                        "composite runtime requires role_identities",
                        code="role_identities_missing",
                    )
                composite_key = CompositeRuntimeKey(
                    *_identity_parts(identity, label="top-level identity"),
                    contract_name,
                    visibility,
                )
                # This lookup is intentionally first: no role builder may run
                # when the aggregate descriptor is absent.
                composite_descriptor = _lookup(
                    assembler_registry,
                    composite_key,
                    required_code="composite_assembler_unavailable",
                    label="composite runtime assembler",
                )
                if not isinstance(composite_descriptor, CompositeRuntimeDescriptor):
                    raise RuntimeFactoryError(
                        f"registry returned an invalid composite descriptor for {composite_key!r}",
                        code="invalid_runtime_descriptor",
                    )
                _validate_composite_selectors(composite_descriptor, role_identities, contract_name, visibility)
                if set(role_specs) != set(role_identities):
                    missing = sorted(set(role_identities) - set(role_specs))
                    extra = sorted(set(role_specs) - set(role_identities))
                    raise RuntimeFactoryError(
                        f"role runtime specs must match role identities (missing={missing}, extra={extra})",
                        code="role_runtime_spec_missing" if missing else "role_runtime_spec_extra",
                    )

                role_infos: dict[str, _RoleRuntimeInfo] = {}
                for role in sorted(role_identities):
                    try:
                        role_infos[role] = _role_info(
                            role,
                            role_identities[role],
                            role_specs[role],
                            contract_name,
                            visibility,
                        )
                    except RuntimeFactoryError as exc:
                        if exc.code != "role_backend_profile_mismatch":
                            raise
                        raise RuntimeFactoryError(
                            str(exc),
                            code="composite_role_backend_mismatch",
                            details={"role": role, **dict(exc.details)},
                        ) from exc
                _validate_composite_matrix(composite_descriptor, role_infos)
                role_assemblies: dict[str, RuntimeAssembly] = {}
                role_contexts: dict[str, RuntimeRoleContext] = {}
                for role, info in role_infos.items():
                    descriptor = _lookup(
                        assembler_registry,
                        info.runtime_key,
                        required_code="runtime_assembler_unavailable",
                        label=f"role runtime assembler for {role!r}",
                    )
                    if not isinstance(descriptor, RuntimeDescriptor):
                        raise RuntimeFactoryError(
                            f"registry returned an invalid role descriptor for {info.runtime_key!r}",
                            code="invalid_runtime_descriptor",
                        )
                    _validate_role_descriptor(descriptor, info)
                    builder = _lookup(
                        session_registry,
                        info.session_builder_key,
                        required_code="session_builder_unavailable",
                        label=f"SessionBuilder for {role!r}",
                    )
                    role_context = RuntimeRoleContext(
                        spec,
                        info.runtime_spec,
                        role,
                        info.identity,
                        info.runtime_key,
                        info.session_builder_key,
                        descriptor,
                        info.profile,
                        info.backend,
                        info.target_runtime,
                        info.runtime_abi,
                        deployment,
                        snapshot,
                        bindings.get(role),
                        artifact_handles,
                        providers,
                    )
                    session = _invoke(
                        builder,
                        {
                            "spec": spec,
                            "runtime_spec": info.runtime_spec,
                            "role_runtime_spec": info.runtime_spec,
                            "role": role,
                            "identity": info.identity,
                            "key": info.session_builder_key,
                            "session_builder_key": info.session_builder_key,
                            "profile": info.profile,
                            "backend_profile": info.profile,
                            "context": role_context,
                            "role_context": role_context,
                            "providers": providers,
                            "deployment": deployment,
                            "validated_deployment": snapshot,
                            "artifact_bindings": bindings.get(role),
                            "artifact_handles": artifact_handles,
                        },
                        (info.runtime_spec, role_context, spec),
                    )
                    if session is None:
                        raise RuntimeFactoryError(
                            f"SessionBuilder for role {role!r} returned None",
                            code="session_builder_failed",
                        )
                    session_entries = [OwnedComponent(session, f"session:{role}")]
                    rollback_entries.extend(session_entries)
                    result = _invoke_role_assembler(descriptor, role_context, session, contract)
                    role_assembly = _assembly_from_result(
                        result,
                        session=session,
                        role=role,
                        contract=contract,
                        descriptor=descriptor,
                    )
                    role_assemblies[role] = role_assembly
                    role_contexts[role] = role_context
                    rollback_entries.extend(_assembly_entries(role_assembly, process_provider_ids=set()))
                    _validate_assembly_contract(role_assembly, contract, role=role)

                _validate_composite_matrix(composite_descriptor, role_infos)
                aggregate_result = _invoke_composite_assembler(
                    composite_descriptor,
                    role_assemblies=role_assemblies,
                    role_contexts=role_contexts,
                    spec=spec,
                    deployment=deployment,
                    providers=providers,
                    contract=contract,
                    artifact_bindings=bindings,
                    execution_roles=_execution_roles(deployment, role_assemblies),
                )
                assembly = _assembly_from_result(
                    aggregate_result,
                    session=None,
                    role=None,
                    contract=contract,
                    descriptor=composite_descriptor,
                )
                rollback_entries.extend(_assembly_entries(assembly, process_provider_ids=set()))
                _validate_assembly_contract(assembly, contract)
                if any(
                    assembly.runtime_executor is role_assembly.runtime_executor
                    for role_assembly in role_assemblies.values()
                ):
                    raise RuntimeFactoryError(
                        "composite assembler must provide an explicit aggregate runtime_executor",
                        code="composite_executor_missing",
                    )
                assembly.role_assemblies = dict(role_assemblies)
                assembly.artifact_bindings = dict(bindings)
                _finalize_assembly(
                    assembly,
                    role_assemblies.values(),
                    providers=providers,
                    sessions=tuple(role_assembly.session for role_assembly in role_assemblies.values()),
                    spec=spec,
                    snapshot=snapshot,
                    identity=identity,
                    contract=contract,
                    bindings=bindings,
                    artifact_handles=artifact_handles,
                )
            else:
                role = _single_role_name(deployment, bindings)
                runtime_profile = _single_runtime_profile(spec, snapshot, deployment)
                # A ModelRuntimeSpec stores target/runtime fields beside the
                # typed profile, so preserve those overrides when present.
                profile, backend, target_runtime, runtime_abi = _profile_info(
                    runtime_profile,
                    role=role,
                    target_runtime_override=_get(spec, "target_runtime", default=_MISSING),
                    runtime_abi_override=_get(spec, "runtime_abi", default=_MISSING),
                )
                info = _role_info(
                    role,
                    identity,
                    {
                        "backend": backend,
                        "target_runtime": target_runtime,
                        "runtime_abi": runtime_abi,
                        "backend_profile": profile,
                    },
                    contract_name,
                    visibility,
                )
                descriptor = _lookup(
                    assembler_registry,
                    info.runtime_key,
                    required_code="runtime_assembler_unavailable",
                    label="runtime assembler",
                )
                if not isinstance(descriptor, RuntimeDescriptor):
                    raise RuntimeFactoryError(
                        f"registry returned an invalid role descriptor for {info.runtime_key!r}",
                        code="invalid_runtime_descriptor",
                    )
                _validate_role_descriptor(descriptor, info)
                builder = _lookup(
                    session_registry,
                    info.session_builder_key,
                    required_code="session_builder_unavailable",
                    label="SessionBuilder",
                )
                role_context = RuntimeRoleContext(
                    spec,
                    runtime_profile,
                    role,
                    identity,
                    info.runtime_key,
                    info.session_builder_key,
                    descriptor,
                    profile,
                    backend,
                    target_runtime,
                    runtime_abi,
                    deployment,
                    snapshot,
                    bindings.get(role) if role in bindings else next(iter(bindings.values()), None),
                    artifact_handles,
                    providers,
                )
                session = _invoke(
                    builder,
                    {
                        "spec": spec,
                        "runtime_spec": runtime_profile,
                        "role_runtime_spec": runtime_profile,
                        "role": role,
                        "identity": identity,
                        "key": info.session_builder_key,
                        "session_builder_key": info.session_builder_key,
                        "profile": profile,
                        "backend_profile": profile,
                        "context": role_context,
                        "role_context": role_context,
                        "providers": providers,
                        "deployment": deployment,
                        "validated_deployment": snapshot,
                        "artifact_bindings": role_context.artifact_bindings,
                        "artifact_handles": artifact_handles,
                    },
                    (runtime_profile, role_context, spec),
                )
                if session is None:
                    raise RuntimeFactoryError("SessionBuilder returned None", code="session_builder_failed")
                rollback_entries.append(OwnedComponent(session, f"session:{role}"))
                result = _invoke_role_assembler(descriptor, role_context, session, contract)
                assembly = _assembly_from_result(
                    result,
                    session=session,
                    role=role,
                    contract=contract,
                    descriptor=descriptor,
                )
                _finalize_assembly(
                    assembly,
                    (assembly,),
                    providers=providers,
                    sessions=(session,),
                    spec=spec,
                    snapshot=snapshot,
                    identity=identity,
                    contract=contract,
                    bindings=bindings,
                    artifact_handles=artifact_handles,
                )
                rollback_entries.extend(_assembly_entries(assembly, process_provider_ids=set()))

            _validate_final_assembly(assembly, contract, composite=composite)
            # ``ModelRuntimeHandle`` claims only after all factory checks and
            # ownership entries are complete.  A constructor failure is still
            # rolled back by this scope.
            handle = ModelRuntimeHandle(assembly)
            if not assembly.ownership_transferred:
                raise RuntimeFactoryError(
                    "runtime assembly ownership was not transferred", code="assembly_transfer_failed"
                )
            return handle
        except Exception as exc:
            rollback_errors = _rollback(rollback_entries)
            if isinstance(exc, RuntimeFactoryError):
                if rollback_errors:
                    details = {**dict(exc.details), "rollback_errors": tuple(str(error) for error in rollback_errors)}
                    raise RuntimeFactoryError(str(exc), code=exc.code, details=details) from exc
                raise
            if isinstance(exc, RuntimeRegistryError):
                details = {"exception_type": type(exc).__name__}
                if rollback_errors:
                    details["rollback_errors"] = tuple(str(error) for error in rollback_errors)
                raise RuntimeFactoryError(str(exc), code=exc.code, details=details) from exc
            details = {"exception_type": type(exc).__name__}
            if rollback_errors:
                details["rollback_errors"] = tuple(str(error) for error in rollback_errors)
            raise RuntimeFactoryError(
                f"runtime construction failed: {exc}",
                code="runtime_factory_failed",
                details=details,
            ) from exc

    build = create
    create_runtime = create


def _single_role_name(deployment: object | None, bindings: Mapping[str, object]) -> str:
    roles = _get(deployment, "execution", default=())
    if isinstance(roles, Sequence) and not isinstance(roles, str | bytes) and len(roles) == 1:
        role = roles[0]
        if isinstance(role, str) and (not bindings or role in bindings):
            return role
    return "__default__"


def _retain_owned_entry(
    entries: list[OwnedComponent],
    existing_ids: set[int],
    process_provider_ids: set[int],
    resource: object | None,
    name: str,
    release_method: str | None = None,
) -> None:
    if resource is None or id(resource) in existing_ids or id(resource) in process_provider_ids:
        return
    entries.append(OwnedComponent(resource, name, release_method))
    existing_ids.add(id(resource))


def _finalize_assembly(
    assembly: RuntimeAssembly,
    role_assemblies: Iterable[RuntimeAssembly],
    *,
    providers: RuntimeProviders,
    sessions: Sequence[object],
    spec: object,
    snapshot: object | None,
    identity: object,
    contract: object,
    bindings: Mapping[str, object],
    artifact_handles: Mapping[str, object],
) -> None:
    process_provider_ids = {id(providers.acl_runtime_provider), id(providers.resource_admission_provider)}
    provider_entries: list[OwnedComponent] = []
    resource_entries: list[OwnedComponent] = []
    for candidate in (*tuple(role_assemblies), assembly):
        entries = list(_assembly_entries(candidate, process_provider_ids=process_provider_ids))
        existing_ids = {id(entry.resource) for entry in entries}
        owns_nested_lifecycle = getattr(candidate.runtime_executor, "_owns_lifecycle_components", False)
        _retain_owned_entry(
            entries, existing_ids, process_provider_ids, candidate.device_lease, "device_lease", "release"
        )
        if not owns_nested_lifecycle:
            _retain_owned_entry(entries, existing_ids, process_provider_ids, candidate.session, "session")
        _retain_owned_entry(entries, existing_ids, process_provider_ids, candidate.runtime_executor, "runtime_executor")
        _retain_owned_entry(
            entries, existing_ids, process_provider_ids, candidate.streaming_runtime, "streaming_runtime"
        )
        adapter = candidate.request_adapter
        if adapter is None:
            adapter = candidate.result_adapter
        if adapter is None:
            adapter = candidate.adapter
        _retain_owned_entry(entries, existing_ids, process_provider_ids, adapter, "adapter")
        _retain_owned_entry(entries, existing_ids, process_provider_ids, candidate.processor, "processor")
        _retain_owned_entry(entries, existing_ids, process_provider_ids, candidate.worker, "worker")
        for index, resource in enumerate(candidate.host_resources):
            _retain_owned_entry(entries, existing_ids, process_provider_ids, resource, f"host_resource:{index}")
        provider_ids = {
            id(item)
            for item in (
                *candidate.provider_leases,
                *candidate.provider_registrations,
                *candidate.provider_leases_and_registrations,
                candidate.provider_lease,
                candidate.provider_registration,
                candidate.device_lease,
            )
            if item is not None
        }
        for item, name in (
            (candidate.provider_lease, "provider_lease"),
            (candidate.provider_registration, "provider_registration"),
        ):
            if item is not None and id(item) not in existing_ids and id(item) not in process_provider_ids:
                entries.append(OwnedComponent(item, name, "release"))
                existing_ids.add(id(item))
        for item in (
            *candidate.provider_leases,
            *candidate.provider_registrations,
            *candidate.provider_leases_and_registrations,
        ):
            if id(item) not in existing_ids and id(item) not in process_provider_ids:
                entries.append(OwnedComponent(item, "provider_lease_or_registration", "release"))
                existing_ids.add(id(item))
        for entry in entries:
            if id(entry.resource) in provider_ids:
                provider_entries.append(entry)
            else:
                resource_entries.append(entry)
    # A private transition executor may intentionally own a legacy Session's
    # nested lifecycle.  In that case adding the Session as a second handle
    # component would load and close it twice; the assembly still retains the
    # Session field for diagnostics and migration bookkeeping.
    if not getattr(assembly.runtime_executor, "_owns_lifecycle_components", False):
        for session in sessions:
            resource_entries.append(OwnedComponent(session, "session"))
    assembly.owned_components = _deduplicate((*provider_entries, *resource_entries))
    assembly.components = ()
    assembly.providers = providers
    assembly.process_providers = (providers.acl_runtime_provider, providers.resource_admission_provider)
    assembly.artifact_bindings = dict(bindings)
    if assembly.request_adapter is None:
        assembly.request_adapter = ResultAdapter()
    if assembly.identity is None:
        assembly.identity = identity
    if assembly.deployment_fingerprint is None:
        assembly.deployment_fingerprint = _get(snapshot, "deployment_fingerprint", "fingerprint", default=None)
    if assembly.runtime_profile_fingerprint is None:
        assembly.runtime_profile_fingerprint = _get(
            snapshot,
            "runtime_profile_fingerprint",
            "runtime_instance_fingerprint",
            default=None,
        )
    if assembly.artifact_integrity is None:
        assembly.artifact_integrity = _get(snapshot, "integrity_status", "integrity_report", default=None)
    if assembly.runtime_id is None:
        fingerprint = assembly.deployment_fingerprint
        assembly.runtime_id = (
            f"runtime-{str(fingerprint)[:16]}" if fingerprint else f"runtime-{_identity_parts(identity)[1]}"
        )
    assembly.load_context = _get(spec, "_load_context", default=spec)
    del contract, artifact_handles


def _validate_final_assembly(assembly: RuntimeAssembly, contract: object, *, composite: bool) -> None:
    _validate_assembly_contract(assembly, contract)
    expected = _contract_name(contract)
    if assembly.runtime_executor is None:
        raise RuntimeFactoryError("RuntimeAssembly has no runtime_executor", code="invalid_runtime_assembly")
    if str(expected).startswith("stream-") and assembly.streaming_runtime is None:
        raise RuntimeFactoryError(
            "stream-scoped runtime requires a StreamingRuntime",
            code="streaming_runtime_missing",
        )
    if composite and not assembly.role_assemblies:
        raise RuntimeFactoryError(
            "composite RuntimeAssembly must preserve role assemblies",
            code="invalid_runtime_assembly",
        )


__all__ = [
    "AssemblyValidationError",
    "DefaultRequestDirectAssembler",
    "ModelRuntimeFactory",
    "RequestDirectAssembler",
    "RequestDirectExecutor",
    "RequestIterativeAssembler",
    "assemble_request_direct",
    "assemble_request_iterative",
    "RuntimeFactoryError",
    "RuntimeRoleContext",
]
