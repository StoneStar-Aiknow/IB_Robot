"""Typed validation for arbitrary model service instances."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from inference_manifest import (
    MANIFEST_FILENAME,
    ManifestError,
    ValidatedManifest,
    load_inference_manifest,
    semantic_identity_fingerprint,
)

_INSTANCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")
_NODE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENDPOINT_PATTERN = re.compile(r"^/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*$")
_IMPORT_PATH_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*$")
_SERVICE_TYPE_PATTERN = re.compile(r"^[A-Za-z_]\w*/srv/[A-Za-z_]\w*$")
_SERVICE_FIELDS = frozenset(
    {
        "id",
        "enabled",
        "required",
        "bundle_path",
        "deployment",
        "adapter_class",
        "service_type",
        "endpoint",
        "node_name",
        "runtime_options",
    }
)


class PerceptionRuntimeConfigError(ValueError):
    """Model service configuration is invalid."""


@dataclass(frozen=True)
class PerceptionModelServiceConfig:
    instance_id: str
    enabled: bool
    required: bool
    bundle_path: Path | None
    deployment: str | None
    adapter_class: str | None
    service_type: str | None
    endpoint: str | None
    node_name: str
    runtime_options: Mapping[str, object]
    validated_manifest: ValidatedManifest | None


@dataclass(frozen=True)
class PerceptionRuntimeConfig:
    services: tuple[PerceptionModelServiceConfig, ...]

    @property
    def enabled_services(self) -> tuple[PerceptionModelServiceConfig, ...]:
        return tuple(service for service in self.services if service.enabled)

    def configuration_generation(self, role_bindings: Mapping[str, str]) -> int:
        """Return a stable positive generation for the effective role-bound services."""
        by_id = {service.instance_id: service for service in self.enabled_services}
        records = []
        for role, instance_id in sorted(role_bindings.items()):
            service = by_id.get(instance_id)
            if service is None or service.validated_manifest is None:
                raise PerceptionRuntimeConfigError(f"Role {role!r} references unavailable service {instance_id!r}")
            identity = service.validated_manifest.manifest.model.semantic_identity
            records.append(
                {
                    "role": role,
                    "id": instance_id,
                    "required": service.required,
                    "deployment": service.deployment,
                    "deployment_fingerprint": service.validated_manifest.fingerprint,
                    "semantic_identity_fingerprint": semantic_identity_fingerprint(identity) if identity else None,
                    "service_type": service.service_type,
                    "endpoint": service.endpoint,
                    "node_name": service.node_name,
                    "adapter_class": service.adapter_class,
                    "runtime_options": dict(service.runtime_options),
                }
            )
        payload = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        generation = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
        return generation or 1


def parse_perception_runtime_config(robot_config: Mapping[str, Any]) -> PerceptionRuntimeConfig:
    """Validate the ordered model-service list without a family vocabulary."""

    perception = robot_config.get("perception_services", {})
    if not isinstance(perception, Mapping):
        raise PerceptionRuntimeConfigError("perception_services must be a mapping")
    path = "perception_services"
    for legacy_field in ("model_backend", "backend", "device"):
        if legacy_field in perception:
            raise PerceptionRuntimeConfigError(
                f"{path}.{legacy_field} is unsupported; select services[].deployment explicitly"
            )

    raw_services = perception.get("services", [])
    if not isinstance(raw_services, Sequence) or isinstance(raw_services, str | bytes):
        raise PerceptionRuntimeConfigError(f"{path}.services must be a list")
    services = tuple(_parse_service(value, index, path) for index, value in enumerate(raw_services))
    _validate_conflicts(services)
    return PerceptionRuntimeConfig(services)


def _parse_service(value: Any, index: int, parent_path: str) -> PerceptionModelServiceConfig:
    service_path = f"{parent_path}.services[{index}]"
    if not isinstance(value, Mapping):
        raise PerceptionRuntimeConfigError(f"{service_path} must be a mapping")
    unknown = sorted(set(value) - _SERVICE_FIELDS)
    if unknown:
        raise PerceptionRuntimeConfigError(f"{service_path} contains unsupported fields: {unknown}")
    instance_id = _require_string(value, "id", service_path)
    if not _INSTANCE_ID_PATTERN.fullmatch(instance_id):
        raise PerceptionRuntimeConfigError(f"{service_path}.id must match ^[a-z][a-z0-9_]{{0,62}}$")
    for legacy_field in ("backend", "device"):
        if legacy_field in value:
            raise PerceptionRuntimeConfigError(f"{service_path}.{legacy_field} is unsupported; select deployment")

    enabled = value.get("enabled", True)
    required = value.get("required", False)
    if not isinstance(enabled, bool) or not isinstance(required, bool):
        raise PerceptionRuntimeConfigError(f"{service_path}.enabled and .required must be booleans")
    if required and not enabled:
        raise PerceptionRuntimeConfigError(f"{service_path} cannot be required when disabled")

    node_name = value.get("node_name", f"model_service_{instance_id}")
    if not isinstance(node_name, str) or not _NODE_NAME_PATTERN.fullmatch(node_name):
        raise PerceptionRuntimeConfigError(f"{service_path}.node_name must be a valid ROS node name")
    runtime_options = _parse_runtime_options(value.get("runtime_options", {}), service_path)

    bundle_path = deployment = adapter_class = service_type = endpoint = validated_manifest = None
    if enabled:
        bundle_path = _resolve_bundle_path(_require_string(value, "bundle_path", service_path), service_path)
        deployment = _require_string(value, "deployment", service_path)
        adapter_class = _require_string(value, "adapter_class", service_path)
        service_type = _require_string(value, "service_type", service_path)
        endpoint = _require_string(value, "endpoint", service_path)
        if not _IMPORT_PATH_PATTERN.fullmatch(adapter_class):
            raise PerceptionRuntimeConfigError(f"{service_path}.adapter_class must be a canonical module:Class path")
        if not _SERVICE_TYPE_PATTERN.fullmatch(service_type):
            raise PerceptionRuntimeConfigError(f"{service_path}.service_type must use package/srv/Type syntax")
        if not _ENDPOINT_PATTERN.fullmatch(endpoint):
            raise PerceptionRuntimeConfigError(f"{service_path}.endpoint must be an absolute ROS name")
        validated_manifest = _validate_bundle(bundle_path, deployment, service_path)

    return PerceptionModelServiceConfig(
        instance_id=instance_id,
        enabled=enabled,
        required=required,
        bundle_path=bundle_path,
        deployment=deployment,
        adapter_class=adapter_class,
        service_type=service_type,
        endpoint=endpoint,
        node_name=node_name,
        runtime_options=runtime_options,
        validated_manifest=validated_manifest,
    )


def _parse_runtime_options(value: Any, service_path: str) -> Mapping[str, object]:
    path = f"{service_path}.runtime_options"
    if not isinstance(value, Mapping):
        raise PerceptionRuntimeConfigError(f"{path} must be a mapping")
    if any(not isinstance(key, str) or not key for key in value):
        raise PerceptionRuntimeConfigError(f"{path} keys must be non-empty strings")
    try:
        normalized = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise PerceptionRuntimeConfigError(f"{path} must contain JSON-compatible finite values: {exc}") from exc
    return MappingProxyType(normalized)


def _require_string(value: Mapping[str, Any], field: str, path: str) -> str:
    result = value.get(field)
    if not isinstance(result, str) or not result:
        raise PerceptionRuntimeConfigError(f"{path}.{field} must be a non-empty string")
    return result


def _resolve_bundle_path(value: str, service_path: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        workspace_value = os.environ.get("WORKSPACE")
        if not workspace_value:
            raise PerceptionRuntimeConfigError(
                f"{service_path}.bundle_path is relative but WORKSPACE is unset; no repository fallback is allowed"
            )
        workspace = Path(workspace_value).expanduser()
        if not workspace.is_absolute():
            raise PerceptionRuntimeConfigError("WORKSPACE must be absolute for relative model bundle paths")
        candidate = workspace / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise PerceptionRuntimeConfigError(f"{service_path}.bundle_path does not exist: {candidate}: {exc}") from exc
    if not resolved.is_dir():
        raise PerceptionRuntimeConfigError(f"{service_path}.bundle_path is not a directory: {resolved}")
    return resolved


def _validate_bundle(path: Path, deployment: str, service_path: str) -> ValidatedManifest:
    manifest_path = path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise PerceptionRuntimeConfigError(f"{service_path} bundle is missing {MANIFEST_FILENAME}: {manifest_path}")
    try:
        validated = load_inference_manifest(path, deployment)
    except ManifestError as exc:
        raise PerceptionRuntimeConfigError(
            f"{service_path} failed bundle validation for deployment {deployment!r}: {exc}"
        ) from exc
    if validated.manifest.model.interface == "policy":
        raise PerceptionRuntimeConfigError(f"{service_path} cannot expose a policy bundle as a model service")
    return validated


def _validate_conflicts(services: tuple[PerceptionModelServiceConfig, ...]) -> None:
    instance_ids: set[str] = set()
    nodes: dict[str, str] = {}
    endpoints: dict[str, str] = {}
    for service in services:
        if service.instance_id in instance_ids:
            raise PerceptionRuntimeConfigError(f"Duplicate model service id {service.instance_id!r}")
        instance_ids.add(service.instance_id)
        if not service.enabled:
            continue
        previous_service = nodes.get(service.node_name)
        if previous_service is not None:
            raise PerceptionRuntimeConfigError(
                f"Model service node conflict: {service.instance_id!r} and {previous_service!r} use {service.node_name!r}"
            )
        nodes[service.node_name] = service.instance_id
        previous_service = endpoints.get(service.endpoint)
        if previous_service is not None:
            raise PerceptionRuntimeConfigError(
                f"Model service endpoint conflict: {service.instance_id!r} and {previous_service!r} use {service.endpoint!r}"
            )
        endpoints[service.endpoint] = service.instance_id


__all__ = [
    "PerceptionModelServiceConfig",
    "PerceptionRuntimeConfig",
    "PerceptionRuntimeConfigError",
    "parse_perception_runtime_config",
]
