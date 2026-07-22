"""Canonical atomic writer shared by model conversion tools."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from inference_manifest.errors import ManifestValidationError
from inference_manifest.models import InferenceManifest
from inference_manifest.schema import validate_manifest_schema


def canonical_manifest_bytes(manifest: InferenceManifest | dict[str, Any]) -> bytes:
    if isinstance(manifest, InferenceManifest):
        value = manifest.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    else:
        value = manifest
    validate_manifest_schema(value, "manifest writer input")
    try:
        typed = InferenceManifest.model_validate_json(json.dumps(value, ensure_ascii=False))
    except ValidationError as exc:
        raise ManifestValidationError(f"Typed manifest validation failed for writer input: {exc}") from exc
    canonical = typed.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    return json.dumps(canonical, ensure_ascii=False, indent=2, sort_keys=True).encode() + b"\n"


def write_inference_manifest(path: str | Path, manifest: InferenceManifest | dict[str, Any]) -> Path:
    """Atomically replace a manifest after schema and typed validation."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_manifest_bytes(manifest)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return destination
