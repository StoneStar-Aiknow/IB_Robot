"""Installed JSON Schema loading and validation."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import ValidationError
from jsonschema.protocols import Validator
from jsonschema.validators import validator_for

from inference_manifest.errors import ManifestValidationError


@lru_cache(maxsize=1)
def manifest_schema() -> dict[str, Any]:
    resource = files("inference_manifest").joinpath("inference_manifest.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def manifest_validator() -> Validator:
    schema = manifest_schema()
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    return validator_class(schema)


def validate_manifest_schema(value: Any, source: str) -> None:
    errors = sorted(manifest_validator().iter_errors(value), key=lambda error: list(error.absolute_path))
    if errors:
        raise ManifestValidationError(_format_schema_error(_most_specific_error(errors[0]), source))

    # JSON Schema expresses the v3 shape and closed object boundaries.  The
    # typed model performs the cross-field rules (contract combinations,
    # profile/backend pairing, logical state-link safety, and composite role
    # matching) before a caller can proceed to runtime construction.
    from pydantic import ValidationError as PydanticValidationError

    from inference_manifest.models import InferenceManifest

    try:
        InferenceManifest.model_validate_json(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (PydanticValidationError, ValueError) as exc:
        raise ManifestValidationError(f"Typed manifest validation failed for {source}: {exc}") from exc


def _most_specific_error(error: ValidationError) -> ValidationError:
    candidates = [error]
    for child in error.context:
        candidates.append(_most_specific_error(child))
    return max(candidates, key=lambda candidate: (len(candidate.absolute_path), len(candidate.absolute_schema_path)))


def _format_schema_error(error: ValidationError, source: str) -> str:
    location = "$"
    for part in error.absolute_path:
        location += f"[{part}]" if isinstance(part, int) else f".{part}"
    return f"Manifest schema validation failed at {location} in {source}: {error.message}; received {error.instance!r}"
