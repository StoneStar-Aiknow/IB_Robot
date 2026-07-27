"""Canonical identities derived only from manifest structure."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from pydantic import BaseModel

from inference_manifest.errors import ManifestIntegrityError
from inference_manifest.models import BundleFile, Deployment
from inference_manifest.paths import normalize_unique_paths

_REGENERATE_GUIDANCE = (
    "Rerun the owning exporter or packaging workflow to regenerate the manifest; do not edit digests manually."
)


def canonical_bundle_digest(
    bundle_uuid: str,
    bundle_revision: int,
    bundle_name: str,
    files: Iterable[BundleFile],
) -> str:
    """Hash the lightweight bundle declaration without reading bundle files."""

    entries = tuple(files)
    normalized_paths = normalize_unique_paths((entry.path for entry in entries), "bundle.files")
    payload = {
        "format": "ibrobot.bundle-structure-v2",
        "uuid": bundle_uuid,
        "revision": bundle_revision,
        "name": bundle_name,
        "files": sorted(normalized_paths),
    }
    payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_bundle_digest(
    bundle_uuid: str,
    bundle_revision: int,
    bundle_name: str,
    files: Iterable[BundleFile],
    expected: str,
) -> str:
    actual = canonical_bundle_digest(bundle_uuid, bundle_revision, bundle_name, files)
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
        deployment_value = deployment.model_dump(mode="json", exclude_none=True)
    else:
        deployment_value = deployment
    payload = {
        "format": "ibrobot.deployment-structure-v2",
        "schema_version": schema_version,
        "bundle_digest": bundle_digest,
        "deployment_name": deployment_name,
        "deployment": deployment_value,
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(serialized).hexdigest()
