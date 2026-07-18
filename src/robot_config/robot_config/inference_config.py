"""Typed validation for named inference pipelines in robot configuration."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from inference_manifest import MANIFEST_FILENAME, ManifestError, ValidatedManifest, load_inference_manifest

PIPELINE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,62}$")

_NODE_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENDPOINT_PATTERN = re.compile(r"^/(?:[A-Za-z_][A-Za-z0-9_]*)(?:/[A-Za-z_][A-Za-z0-9_]*)*$")
_INFERENCE_FIELDS = frozenset({"enabled", "pipelines"})
_PIPELINE_FIELDS = frozenset(
    {
        "model_path",
        "deployment",
        "execution_mode",
        "request_timeout",
        "default_task",
        "runtime_options",
        "transport",
    }
)
_TRANSPORT_FIELDS = (
    "node_name",
    "cloud_node_name",
    "action_server",
    "reset_service",
    "health_topic",
    "action_topic",
    "request_topic",
    "result_topic",
    "heartbeat_topic",
)
_DISTRIBUTED_TRANSPORT_FIELDS = frozenset({"cloud_node_name", "request_topic", "result_topic", "heartbeat_topic"})


class InferenceConfigError(ValueError):
    """Inference pipeline configuration is invalid."""


@dataclass(frozen=True)
class InferenceTransportConfig:
    """Resolved ROS names for one pipeline."""

    node_name: str
    cloud_node_name: str | None
    action_server: str
    reset_service: str
    health_topic: str
    action_topic: str
    request_topic: str | None
    result_topic: str | None
    heartbeat_topic: str | None


@dataclass(frozen=True)
class InferencePipelineConfig:
    """Validated configuration and bundle metadata for one pipeline."""

    pipeline_id: str
    model_path: Path
    deployment: str
    execution_mode: Literal["monolithic", "distributed"]
    request_timeout: float
    default_task: str
    runtime_options: Mapping[str, object]
    transport: InferenceTransportConfig
    validated_manifest: ValidatedManifest


@dataclass(frozen=True)
class ControlModeInferenceConfig:
    """Typed inference configuration for one robot control mode."""

    control_mode: str
    enabled: bool
    pipelines: Mapping[str, InferencePipelineConfig]


def parse_inference_config(
    robot_config: Mapping[str, Any],
    control_mode: str,
) -> ControlModeInferenceConfig:
    """Parse and validate ``control_modes.<mode>.inference`` before launch construction."""

    if not isinstance(robot_config, Mapping):
        raise InferenceConfigError("robot configuration must be a mapping")
    if not isinstance(control_mode, str) or not control_mode:
        raise InferenceConfigError("control mode must be a non-empty string")

    control_modes = robot_config.get("control_modes", {})
    if not isinstance(control_modes, Mapping):
        raise InferenceConfigError("control_modes must be a mapping")
    if control_mode not in control_modes:
        raise InferenceConfigError(f"control mode {control_mode!r} is not configured")

    mode_path = f"control_modes.{control_mode}"
    mode_config = control_modes[control_mode]
    if not isinstance(mode_config, Mapping):
        raise InferenceConfigError(f"{mode_path} must be a mapping")

    inference_path = f"{mode_path}.inference"
    if "inference" not in mode_config:
        return ControlModeInferenceConfig(
            control_mode=control_mode,
            enabled=False,
            pipelines=MappingProxyType({}),
        )

    inference = mode_config["inference"]
    if not isinstance(inference, Mapping):
        raise InferenceConfigError(f"{inference_path} must be a mapping")
    if "model" in inference:
        raise InferenceConfigError(
            f"{inference_path}.model is a legacy field; configure named entries under {inference_path}.pipelines"
        )
    _reject_unknown_fields(inference, _INFERENCE_FIELDS, inference_path)

    enabled = inference.get("enabled", False)
    if not isinstance(enabled, bool):
        raise InferenceConfigError(f"{inference_path}.enabled must be a boolean")

    raw_pipelines = inference.get("pipelines", {})
    if not isinstance(raw_pipelines, Mapping):
        raise InferenceConfigError(f"{inference_path}.pipelines must be a mapping")
    if enabled and not raw_pipelines:
        raise InferenceConfigError(f"{inference_path}.pipelines must be non-empty when inference is enabled")

    pipelines: dict[str, InferencePipelineConfig] = {}
    for pipeline_id, pipeline_value in raw_pipelines.items():
        validated_id = _validate_pipeline_id(pipeline_id, inference_path)
        pipelines[validated_id] = _parse_pipeline(
            validated_id,
            pipeline_value,
            inference_path,
        )

    _validate_endpoint_conflicts(pipelines)
    return ControlModeInferenceConfig(
        control_mode=control_mode,
        enabled=enabled,
        pipelines=MappingProxyType(pipelines),
    )


def _validate_pipeline_id(pipeline_id: Any, inference_path: str) -> str:
    if not isinstance(pipeline_id, str) or not PIPELINE_ID_PATTERN.fullmatch(pipeline_id):
        raise InferenceConfigError(
            f"{inference_path}.pipelines contains invalid pipeline ID {pipeline_id!r}; "
            "expected ^[a-z][a-z0-9_]{0,62}$"
        )
    return pipeline_id


def _parse_pipeline(
    pipeline_id: str,
    value: Any,
    inference_path: str,
) -> InferencePipelineConfig:
    pipeline_path = f"{inference_path}.pipelines.{pipeline_id}"
    if not isinstance(value, Mapping):
        raise InferenceConfigError(f"{pipeline_path} must be a mapping")
    for legacy_field in ("device", "concurrency"):
        if legacy_field in value:
            raise InferenceConfigError(f"{pipeline_path}.{legacy_field} is not supported")
    _reject_unknown_fields(value, _PIPELINE_FIELDS, pipeline_path)

    model_path_value = _require_string(value, "model_path", pipeline_path)
    deployment = _require_string(value, "deployment", pipeline_path)
    execution_mode = _require_string(value, "execution_mode", pipeline_path)
    if execution_mode not in {"monolithic", "distributed"}:
        raise InferenceConfigError(
            f"{pipeline_path}.execution_mode must be 'monolithic' or 'distributed', got {execution_mode!r}"
        )

    request_timeout = value.get("request_timeout", 5.0)
    if isinstance(request_timeout, bool) or not isinstance(request_timeout, int | float):
        raise InferenceConfigError(f"{pipeline_path}.request_timeout must be a positive finite number")
    request_timeout = float(request_timeout)
    if not math.isfinite(request_timeout) or request_timeout <= 0.0:
        raise InferenceConfigError(f"{pipeline_path}.request_timeout must be a positive finite number")

    default_task = value.get("default_task", "")
    if not isinstance(default_task, str):
        raise InferenceConfigError(f"{pipeline_path}.default_task must be a string")
    runtime_options = _parse_runtime_options(value.get("runtime_options", {}), pipeline_path)

    transport = _parse_transport(pipeline_id, execution_mode, value.get("transport", {}), pipeline_path)
    model_path = _resolve_model_path(model_path_value, pipeline_path)
    validated_manifest = _validate_model_bundle(model_path, deployment, pipeline_path)
    return InferencePipelineConfig(
        pipeline_id=pipeline_id,
        model_path=model_path,
        deployment=deployment,
        execution_mode=execution_mode,
        request_timeout=request_timeout,
        default_task=default_task,
        runtime_options=runtime_options,
        transport=transport,
        validated_manifest=validated_manifest,
    )


def _require_string(value: Mapping[str, Any], field: str, path: str) -> str:
    if field not in value:
        raise InferenceConfigError(f"{path}.{field} is required")
    field_value = value[field]
    if not isinstance(field_value, str) or not field_value:
        raise InferenceConfigError(f"{path}.{field} must be a non-empty string")
    return field_value


def _parse_runtime_options(value: Any, pipeline_path: str) -> Mapping[str, object]:
    options_path = f"{pipeline_path}.runtime_options"
    if not isinstance(value, Mapping):
        raise InferenceConfigError(f"{options_path} must be a mapping")
    invalid_keys = sorted(repr(key) for key in value if not isinstance(key, str) or not key)
    if invalid_keys:
        raise InferenceConfigError(f"{options_path} keys must be non-empty strings: {invalid_keys}")
    try:
        normalized = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise InferenceConfigError(f"{options_path} must contain JSON-compatible finite values: {exc}") from exc
    return MappingProxyType(normalized)


def _parse_transport(
    pipeline_id: str,
    execution_mode: str,
    value: Any,
    pipeline_path: str,
) -> InferenceTransportConfig:
    transport_path = f"{pipeline_path}.transport"
    if not isinstance(value, Mapping):
        raise InferenceConfigError(f"{transport_path} must be a mapping")
    _reject_unknown_fields(value, frozenset(_TRANSPORT_FIELDS), transport_path)
    if execution_mode == "monolithic":
        distributed_overrides = sorted(_DISTRIBUTED_TRANSPORT_FIELDS.intersection(value))
        if distributed_overrides:
            raise InferenceConfigError(
                f"{transport_path} cannot override distributed fields for a monolithic pipeline: "
                f"{distributed_overrides}"
            )

    defaults: dict[str, str | None] = {
        "node_name": f"inference_{pipeline_id}",
        "cloud_node_name": f"inference_{pipeline_id}_cloud" if execution_mode == "distributed" else None,
        "action_server": f"/inference/{pipeline_id}/dispatch",
        "reset_service": f"/inference/{pipeline_id}/reset",
        "health_topic": f"/inference/{pipeline_id}/health",
        "action_topic": f"/actions/{pipeline_id}",
        "request_topic": f"/inference/{pipeline_id}/request" if execution_mode == "distributed" else None,
        "result_topic": f"/inference/{pipeline_id}/result" if execution_mode == "distributed" else None,
        "heartbeat_topic": f"/inference/{pipeline_id}/heartbeat" if execution_mode == "distributed" else None,
    }
    for field, override in value.items():
        if not isinstance(override, str) or not override:
            raise InferenceConfigError(f"{transport_path}.{field} must be a non-empty string")
        defaults[field] = override

    for field in ("node_name", "cloud_node_name"):
        name = defaults[field]
        if name is not None and not _NODE_NAME_PATTERN.fullmatch(name):
            raise InferenceConfigError(
                f"{transport_path}.{field} must be a valid ROS node name containing only letters, digits, and '_'"
            )
    for field in _TRANSPORT_FIELDS[2:]:
        endpoint = defaults[field]
        if endpoint is not None and not _ENDPOINT_PATTERN.fullmatch(endpoint):
            raise InferenceConfigError(
                f"{transport_path}.{field} must be an absolute ROS name with valid slash-separated tokens"
            )

    return InferenceTransportConfig(**defaults)


def _resolve_model_path(value: str, pipeline_path: str) -> Path:
    expanded = Path(value).expanduser()
    if expanded.is_absolute():
        candidate = expanded
    else:
        workspace_value = os.environ.get("WORKSPACE")
        if not workspace_value:
            raise InferenceConfigError(
                f"{pipeline_path}.model_path is relative but WORKSPACE is unset; "
                "no current-directory or repository fallback is allowed"
            )
        workspace = Path(workspace_value).expanduser()
        if not workspace.is_absolute():
            raise InferenceConfigError("WORKSPACE must be an absolute path for relative inference model paths")
        try:
            workspace = workspace.resolve(strict=True)
        except OSError as exc:
            raise InferenceConfigError(f"WORKSPACE cannot be resolved: {workspace}: {exc}") from exc
        candidate = workspace / expanded

    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise InferenceConfigError(f"{pipeline_path}.model_path does not exist: {candidate}: {exc}") from exc
    if not resolved.is_dir():
        raise InferenceConfigError(f"{pipeline_path}.model_path is not a directory: {resolved}")
    return resolved


def _validate_model_bundle(model_path: Path, deployment: str, pipeline_path: str) -> ValidatedManifest:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise InferenceConfigError(f"{pipeline_path} bundle is missing config.json: {config_path}")
    manifest_path = model_path / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise InferenceConfigError(f"{pipeline_path} bundle is missing {MANIFEST_FILENAME}: {manifest_path}")
    try:
        return load_inference_manifest(model_path, deployment)
    except ManifestError as exc:
        raise InferenceConfigError(
            f"{pipeline_path} failed bundle validation for deployment {deployment!r}: {exc}"
        ) from exc


def _reject_unknown_fields(value: Mapping[str, Any], allowed: frozenset[str], path: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise InferenceConfigError(f"{path} contains unsupported fields: {unknown}")


def _validate_endpoint_conflicts(pipelines: Mapping[str, InferencePipelineConfig]) -> None:
    node_owners: dict[str, tuple[str, str]] = {}
    interface_owners: dict[str, tuple[str, str]] = {}
    for pipeline_id, pipeline in pipelines.items():
        transport = pipeline.transport
        local_interfaces: dict[str, str] = {}
        for field in ("node_name", "cloud_node_name"):
            _record_endpoint_owner(node_owners, pipeline_id, field, getattr(transport, field))
        for field in _TRANSPORT_FIELDS[2:]:
            endpoint = getattr(transport, field)
            if endpoint is not None:
                previous_field = local_interfaces.get(endpoint)
                if previous_field is not None:
                    raise InferenceConfigError(
                        f"Inference endpoint conflict: pipeline {pipeline_id!r} {field} and {previous_field} "
                        f"both use {endpoint!r}"
                    )
                local_interfaces[endpoint] = field
            _record_endpoint_owner(interface_owners, pipeline_id, field, endpoint)


def _record_endpoint_owner(
    owners: dict[str, tuple[str, str]],
    pipeline_id: str,
    field: str,
    endpoint: str | None,
) -> None:
    if endpoint is None:
        return
    previous = owners.get(endpoint)
    if previous is not None and previous[0] != pipeline_id:
        previous_pipeline, previous_field = previous
        raise InferenceConfigError(
            f"Inference endpoint conflict: pipeline {pipeline_id!r} {field} and pipeline "
            f"{previous_pipeline!r} {previous_field} both use {endpoint!r}"
        )
    owners[endpoint] = (pipeline_id, field)


__all__ = [
    "ControlModeInferenceConfig",
    "InferenceConfigError",
    "InferencePipelineConfig",
    "InferenceTransportConfig",
    "PIPELINE_ID_PATTERN",
    "parse_inference_config",
]
