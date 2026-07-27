"""Strict unified manifest loading before backend initialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from inference_manifest.errors import ManifestValidationError
from inference_manifest.integrity import (
    deployment_fingerprint,
    verify_bundle_digest,
)
from inference_manifest.json_utils import load_json_strict
from inference_manifest.metadata import PolicyFeature, PolicyMetadata, load_policy_metadata
from inference_manifest.models import CompiledDeployment, Deployment, InferenceManifest, TensorBinding
from inference_manifest.paths import normalize_unique_paths, resolve_bundle_file
from inference_manifest.schema import validate_manifest_schema

MANIFEST_FILENAME = "inference_manifest.json"


@dataclass(frozen=True)
class ValidatedManifest:
    bundle_root: Path
    manifest_path: Path
    manifest: InferenceManifest
    deployment_name: str
    deployment: Deployment
    policy: PolicyMetadata
    fingerprint: str


def load_inference_manifest(bundle_root: str | Path, deployment_name: str) -> ValidatedManifest:
    """Validate one bundle and selected deployment without importing a backend SDK."""

    return _load_inference_manifest(
        bundle_root,
        deployment_name,
        verify_all_bundle_files=True,
        verify_deployment_artifacts=True,
        require_native_weights=None,
    )


def load_inference_manifest_metadata(bundle_root: str | Path, deployment_name: str) -> ValidatedManifest:
    """Validate edge-owned semantics and identity without requiring cloud-only artifacts."""

    return _load_inference_manifest(
        bundle_root,
        deployment_name,
        verify_all_bundle_files=False,
        verify_deployment_artifacts=False,
        require_native_weights=False,
    )


def _load_inference_manifest(
    bundle_root: str | Path,
    deployment_name: str,
    *,
    verify_all_bundle_files: bool,
    verify_deployment_artifacts: bool,
    require_native_weights: bool | None,
) -> ValidatedManifest:
    root = Path(bundle_root)
    manifest_path = root / MANIFEST_FILENAME
    raw = load_json_strict(manifest_path)
    if not isinstance(raw, dict):
        raise ManifestValidationError(f"Inference manifest must be a JSON object: {manifest_path}")

    schema_version = raw.get("schema_version")
    if schema_version != 2:
        guidance = (
            " Schema-v1 artifacts are unsupported; rerun the owning exporter or packager to create a schema-v2 bundle."
            if schema_version == 1
            else ""
        )
        raise ManifestValidationError(
            f"Unsupported schema_version {schema_version!r} in {manifest_path}; supported versions: [2].{guidance}"
        )
    validate_manifest_schema(raw, str(manifest_path))
    manifest = _parse_typed_manifest(raw, manifest_path)

    try:
        deployment = manifest.deployments[deployment_name]
    except KeyError as exc:
        raise ManifestValidationError(
            f"Deployment {deployment_name!r} is not present in {manifest_path}; "
            f"available deployments: {sorted(manifest.deployments)}"
        ) from exc

    bundle_entries = tuple(manifest.bundle.files)
    selected_artifacts = tuple(deployment.artifacts.values()) if isinstance(deployment, CompiledDeployment) else ()
    bundle_paths = set(normalize_unique_paths((entry.path for entry in bundle_entries), "bundle.files"))
    artifact_paths = {artifact.path for artifact in selected_artifacts}
    overlap = sorted(bundle_paths & artifact_paths)
    if overlap:
        raise ManifestValidationError(f"bundle files and deployment artifacts must use distinct paths: {overlap}")

    if require_native_weights is None:
        require_native_weights = any(candidate.backend == "torch" for candidate in manifest.deployments.values())

    if verify_all_bundle_files:
        for entry in bundle_entries:
            resolve_bundle_file(root, entry.path)
    verify_bundle_digest(
        manifest.bundle.uuid,
        manifest.bundle.revision,
        manifest.bundle.name,
        bundle_entries,
        manifest.bundle.digest.value,
    )

    policy = load_policy_metadata(root, require_native_weights=require_native_weights)
    _validate_required_bundle_files(
        manifest,
        policy,
        allow_declared_native_weights=not require_native_weights,
    )

    required_paths = set(policy.required_files)
    entries_to_verify = (
        bundle_entries
        if verify_all_bundle_files
        else tuple(entry for entry in bundle_entries if entry.path in required_paths)
    )
    if not verify_all_bundle_files:
        for entry in entries_to_verify:
            resolve_bundle_file(root, entry.path)

    if verify_deployment_artifacts and isinstance(deployment, CompiledDeployment):
        for artifact in deployment.artifacts.values():
            resolve_bundle_file(root, artifact.path)

    if isinstance(deployment, CompiledDeployment):
        _validate_feature_compatibility(deployment, policy)

    fingerprint = deployment_fingerprint(
        manifest.schema_version,
        manifest.bundle.digest.value,
        deployment_name,
        deployment,
    )
    return ValidatedManifest(
        bundle_root=root.resolve(strict=True),
        manifest_path=manifest_path.resolve(strict=True),
        manifest=manifest,
        deployment_name=deployment_name,
        deployment=deployment,
        policy=policy,
        fingerprint=fingerprint,
    )


def _parse_typed_manifest(raw: dict[str, Any], manifest_path: Path) -> InferenceManifest:
    content = json.dumps(raw, ensure_ascii=False, separators=(",", ":"))
    try:
        return InferenceManifest.model_validate_json(content)
    except ValidationError as exc:
        raise ManifestValidationError(f"Typed manifest validation failed for {manifest_path}: {exc}") from exc


def _validate_required_bundle_files(
    manifest: InferenceManifest,
    policy: PolicyMetadata,
    *,
    allow_declared_native_weights: bool = False,
) -> None:
    declared = {entry.path for entry in manifest.bundle.files}
    required = set(policy.required_files)
    missing = sorted(required - declared)
    if missing:
        raise ManifestValidationError(
            f"Manifest bundle.files omits required LeRobot semantic files: {missing}. "
            "Rerun the owning exporter or packaging workflow."
        )

    optional = {"model.safetensors"} if allow_declared_native_weights else set()
    unexpected_semantic = sorted(path for path in declared - required - optional if _is_reserved_semantic_path(path))
    if unexpected_semantic:
        raise ManifestValidationError(
            f"Manifest bundle.files contains unreferenced LeRobot semantic files: {unexpected_semantic}"
        )


def _is_reserved_semantic_path(path: str) -> bool:
    name = Path(path).name
    return (
        name == "model.safetensors"
        or name.startswith("policy_preprocessor_step_")
        or name.startswith("policy_postprocessor_step_")
    )


def _validate_feature_compatibility(deployment: CompiledDeployment, policy: PolicyMetadata) -> None:
    for role in deployment.execution:
        group = deployment.bindings[role]
        for binding in (*group.inputs, *group.outputs):
            feature = _policy_feature_for_semantic(binding.semantic, policy)
            if feature is None:
                continue
            if not _shape_matches_feature(binding, feature, policy.policy_type):
                raise ManifestValidationError(
                    f"Binding {binding.semantic!r} for role {role!r} has shape {binding.shape} "
                    f"incompatible with LeRobot feature shape {feature.shape}"
                )


def _policy_feature_for_semantic(semantic: str, policy: PolicyMetadata) -> PolicyFeature | None:
    # Tokenizer processors derive these tensors; they are not raw LeRobot input_features.
    if policy.policy_type in {"pi05", "smolvla"} and semantic in {
        "observation.language.tokens",
        "observation.language.attention_mask",
    }:
        return None
    if semantic.startswith("observation."):
        try:
            return policy.input_features[semantic]
        except KeyError as exc:
            raise ManifestValidationError(
                f"Binding references unknown LeRobot input feature {semantic!r}; "
                f"available inputs: {sorted(policy.input_features)}"
            ) from exc
    if semantic == "action":
        return policy.output_features[semantic]
    return None


def _shape_matches_feature(binding: TensorBinding, feature: PolicyFeature, policy_type: str) -> bool:
    expected = feature.shape
    if policy_type in {"pi05", "smolvla"} and feature.type.upper() == "VISUAL" and len(expected) == 3:
        if len(binding.shape) < 3:
            return False
        actual = binding.shape[-3:]
        channel_axis = -1 if binding.layout == "NHWC" else 0
        actual_channels = actual[channel_axis]
        return actual_channels == -1 or actual_channels == expected[0]
    if feature.type.upper() == "VISUAL" and len(expected) == 3 and binding.layout == "NHWC":
        expected = (expected[1], expected[2], expected[0])
    if len(binding.shape) < len(expected):
        return False
    actual_suffix = binding.shape[-len(expected) :]
    if policy_type in {"pi05", "smolvla"} and feature.type.upper() in {"STATE", "ACTION"}:
        return all(actual == -1 or actual >= wanted for actual, wanted in zip(actual_suffix, expected, strict=True))
    return all(actual == -1 or actual == wanted for actual, wanted in zip(actual_suffix, expected, strict=True))
