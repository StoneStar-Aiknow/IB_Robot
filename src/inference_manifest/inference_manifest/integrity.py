"""Streaming integrity checks and canonical manifest identities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from inference_manifest.errors import ManifestIntegrityError
from inference_manifest.models import BundleFile, Deployment
from inference_manifest.paths import normalize_unique_paths, resolve_bundle_file

_HASH_CHUNK_SIZE = 1024 * 1024
_REGENERATE_GUIDANCE = (
    "Rerun the owning exporter or packaging workflow to regenerate the manifest; do not edit digests manually."
)


def sha256_file(path: Path, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Calculate a file SHA-256 without loading the complete artifact into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(bundle_root: Path, relative_path: str, expected: str, description: str) -> str:
    path = resolve_bundle_file(bundle_root, relative_path)
    actual = sha256_file(path)
    if actual != expected:
        raise ManifestIntegrityError(
            f"SHA-256 mismatch for {description} {relative_path!r}: expected {expected}, actual {actual}. "
            f"{_REGENERATE_GUIDANCE}"
        )
    return actual


def canonical_bundle_digest(files: Iterable[BundleFile]) -> str:
    entries = tuple(files)
    normalized_paths = normalize_unique_paths((entry.path for entry in entries), "bundle.files")
    canonical_entries = sorted(
        ({"path": path, "sha256": entry.sha256} for path, entry in zip(normalized_paths, entries, strict=True)),
        key=lambda entry: entry["path"],
    )
    payload = json.dumps(canonical_entries, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_bundle_digest(files: Iterable[BundleFile], expected: str) -> str:
    actual = canonical_bundle_digest(files)
    if actual != expected:
        raise ManifestIntegrityError(
            f"Bundle digest mismatch: expected {expected}, actual {actual}. {_REGENERATE_GUIDANCE}"
        )
    return actual


def deployment_fingerprint(
    schema_version: int,
    bundle_digest: str,
    deployment_name: str,
    deployment: Deployment,
) -> str:
    if isinstance(deployment, BaseModel):
        deployment_value = deployment.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
    else:
        deployment_value = deployment
    payload = {
        "schema_version": schema_version,
        "bundle_digest": bundle_digest,
        "deployment_name": deployment_name,
        "deployment": deployment_value,
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()
