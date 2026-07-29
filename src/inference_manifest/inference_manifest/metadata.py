"""Read-only discovery of LeRobot policy semantics and local assets."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, StringConstraints, model_validator

from inference_manifest.errors import ManifestValidationError
from inference_manifest.json_utils import load_json_strict
from inference_manifest.paths import normalize_bundle_path, resolve_bundle_file, resolve_bundle_path

_PROCESSOR_FILES = ("policy_preprocessor.json", "policy_postprocessor.json")
_TOKENIZER_REFERENCE_KEYS = frozenset(
    {
        "tokenizer_name",
        "action_tokenizer_name",
        "paligemma_tokenizer_name",
    }
)
_CONFIG_LOCAL_REFERENCE_KEYS = frozenset({"vlm_model_name"})


def _validate_feature_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError("feature shape dimensions must be positive integers")
    return shape


FeatureShape = Annotated[tuple[int, ...], AfterValidator(_validate_feature_shape)]
NonEmptyString = Annotated[str, StringConstraints(strict=True, min_length=1)]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class PolicyFeature(_FrozenModel):
    type: NonEmptyString
    shape: FeatureShape


class ExternalDependency(_FrozenModel):
    source: NonEmptyString
    identifier: NonEmptyString


class PolicyMetadata(_FrozenModel):
    policy_type: NonEmptyString
    input_features: dict[str, PolicyFeature]
    output_features: dict[str, PolicyFeature]
    n_obs_steps: int = Field(default=1, ge=1)
    nominal_chunk_size: int | None = Field(default=None, ge=1)
    max_action_dimension: int | None = Field(default=None, ge=1)
    required_files: tuple[str, ...] = Field(min_length=3)
    external_dependencies: tuple[ExternalDependency, ...] = ()
    native_weights_required: bool = False

    @model_validator(mode="after")
    def require_action_feature(self) -> PolicyMetadata:
        if "action" not in self.output_features:
            raise ValueError("LeRobot config output_features must contain 'action'")
        return self


def load_policy_metadata(bundle_root: Path, require_native_weights: bool = False) -> PolicyMetadata:
    """Read policy metadata directly from the original bundle without rewriting it."""

    root = bundle_root.resolve(strict=True)
    config_path = resolve_bundle_file(root, "config.json")
    config = _load_json_object(config_path, "LeRobot config")

    policy_type = config.get("type")
    if not isinstance(policy_type, str) or not policy_type:
        raise ManifestValidationError(f"LeRobot config must contain a non-empty string 'type': {config_path}")

    input_features = _parse_features(config.get("input_features"), "input_features", config_path)
    output_features = _parse_features(config.get("output_features"), "output_features", config_path)
    n_obs_steps = _optional_positive_int(config, "n_obs_steps", config_path)
    if n_obs_steps is None:
        n_obs_steps = 2 if policy_type == "diffusion" else 1
    chunk_size_key = "n_action_steps" if policy_type == "diffusion" else "chunk_size"
    nominal_chunk_size = _optional_positive_int(config, chunk_size_key, config_path)
    max_action_dimension = _optional_positive_int(config, "max_action_dim", config_path)
    required_files = {"config.json"}
    external_dependencies: set[tuple[str, str]] = set()

    for processor_file in _PROCESSOR_FILES:
        processor_path = resolve_bundle_file(root, processor_file)
        processor = _load_json_object(processor_path, processor_file)
        required_files.add(processor_file)

        for state_file in _find_values(processor, {"state_file"}):
            required_files.add(_require_local_file(root, state_file, "state_file"))
        _discover_references(
            root,
            processor,
            _TOKENIZER_REFERENCE_KEYS,
            required_files,
            external_dependencies,
        )

    _discover_references(
        root,
        config,
        _TOKENIZER_REFERENCE_KEYS | _CONFIG_LOCAL_REFERENCE_KEYS,
        required_files,
        external_dependencies,
    )

    if require_native_weights:
        required_files.add(_require_local_file(root, "model.safetensors", "native Torch policy weights"))

    metadata_value = {
        "policy_type": policy_type,
        "input_features": input_features,
        "output_features": output_features,
        "n_obs_steps": n_obs_steps,
        "nominal_chunk_size": nominal_chunk_size,
        "max_action_dimension": max_action_dimension,
        "required_files": tuple(sorted(required_files)),
        "external_dependencies": tuple(
            ExternalDependency(source=source, identifier=identifier)
            for source, identifier in sorted(external_dependencies)
        ),
        "native_weights_required": require_native_weights,
    }
    return PolicyMetadata.model_validate(metadata_value)


def _optional_positive_int(config: Mapping[str, Any], key: str, config_path: Path) -> int | None:
    value = config.get(key)
    if value is None:
        return None
    if type(value) is not int or value < 1:
        raise ManifestValidationError(f"LeRobot config {key} must be a positive integer when present: {config_path}")
    return value


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    value = load_json_strict(path)
    if not isinstance(value, dict):
        raise ManifestValidationError(f"{description} must be a JSON object: {path}")
    return value


def _parse_features(value: Any, name: str, config_path: Path) -> dict[str, PolicyFeature]:
    if not isinstance(value, dict) or not value:
        raise ManifestValidationError(f"LeRobot config {name} must be a non-empty object: {config_path}")

    features: dict[str, PolicyFeature] = {}
    for semantic, raw_feature in value.items():
        if not isinstance(semantic, str) or not semantic:
            raise ManifestValidationError(f"LeRobot config {name} contains an invalid feature name: {config_path}")
        if not isinstance(raw_feature, dict):
            raise ManifestValidationError(f"LeRobot feature {semantic!r} must be an object: {config_path}")
        try:
            features[semantic] = PolicyFeature.model_validate_json(_json_bytes(raw_feature))
        except ValueError as exc:
            raise ManifestValidationError(f"Invalid LeRobot feature {semantic!r} in {config_path}: {exc}") from exc
    return features


def _json_bytes(value: Any) -> bytes:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def _find_values(value: Any, keys: set[str] | frozenset[str]) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys:
                if not isinstance(item, str) or not item:
                    raise ManifestValidationError(f"Semantic asset reference {key!r} must be a non-empty string")
                yield item
            yield from _find_values(item, keys)
    elif isinstance(value, list):
        for item in value:
            yield from _find_values(item, keys)


def _discover_references(
    bundle_root: Path,
    value: Any,
    keys: set[str] | frozenset[str],
    required_files: set[str],
    external_dependencies: set[tuple[str, str]],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in keys:
                if not isinstance(item, str) or not item:
                    raise ManifestValidationError(f"Semantic asset reference {key!r} must be a non-empty string")
                local_files = _local_reference_files(bundle_root, item, key)
                if local_files is None:
                    external_dependencies.add((key, item))
                else:
                    required_files.update(local_files)
            _discover_references(bundle_root, item, keys, required_files, external_dependencies)
    elif isinstance(value, list):
        for item in value:
            _discover_references(bundle_root, item, keys, required_files, external_dependencies)


def _local_reference_files(bundle_root: Path, reference: str, source: str) -> tuple[str, ...] | None:
    candidate: Path
    if Path(reference).is_absolute():
        try:
            resolved = Path(reference).resolve(strict=True)
        except OSError as exc:
            raise ManifestValidationError(f"Local {source} reference does not exist: {reference!r}") from exc
        try:
            relative = resolved.relative_to(bundle_root)
        except ValueError as exc:
            raise ManifestValidationError(f"Local {source} reference escapes the policy bundle: {reference!r}") from exc
        candidate = bundle_root / relative
    else:
        try:
            normalized = normalize_bundle_path(reference)
        except ValueError as exc:
            raise ManifestValidationError(f"Invalid local {source} reference {reference!r}: {exc}") from exc
        candidate = bundle_root.joinpath(*normalized.split("/"))
        if not candidate.exists() and not candidate.is_symlink():
            return None

    try:
        resolved = resolve_bundle_path(bundle_root, candidate.relative_to(bundle_root).as_posix())
    except ValueError as exc:
        raise ManifestValidationError(f"Invalid local {source} reference {reference!r}: {exc}") from exc
    if resolved.is_file():
        return (resolved.relative_to(bundle_root).as_posix(),)
    if not resolved.is_dir():
        raise ManifestValidationError(f"Local {source} reference is not a file or directory: {reference!r}")
    return _directory_files(bundle_root, resolved, source)


def _directory_files(bundle_root: Path, directory: Path, source: str) -> tuple[str, ...]:
    files: list[str] = []
    visited_directories: set[Path] = set()

    def visit(path: Path) -> None:
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(bundle_root)
        except ValueError as exc:
            raise ManifestValidationError(f"Local {source} directory escapes the bundle root: {path}") from exc
        if resolved in visited_directories:
            raise ManifestValidationError(f"Local {source} directory contains a symlink cycle: {path}")
        visited_directories.add(resolved)
        for child in sorted(path.iterdir(), key=lambda entry: entry.name):
            child_relative = child.relative_to(bundle_root).as_posix()
            child_resolved = resolve_bundle_path(bundle_root, child_relative)
            if child_resolved.is_dir():
                visit(child)
            elif child_resolved.is_file():
                files.append(child_relative)
            else:
                raise ManifestValidationError(f"Unsupported local {source} asset type: {child_relative}")

    visit(directory)
    if not files:
        raise ManifestValidationError(f"Local {source} directory contains no files: {directory}")
    return tuple(files)


def _require_local_file(bundle_root: Path, reference: str, source: str) -> str:
    try:
        normalized = normalize_bundle_path(reference)
        resolve_bundle_file(bundle_root, normalized)
    except ValueError as exc:
        raise ManifestValidationError(f"Invalid required {source} reference {reference!r}: {exc}") from exc
    return normalized
