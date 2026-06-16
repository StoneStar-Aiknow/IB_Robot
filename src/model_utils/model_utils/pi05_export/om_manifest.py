# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
# Licensed under the Mulan PSL v2.
"""Writer for the PI05 Ascend OM manifest (``config.om.json``).

The IB-Robot inference runtime loads compiled PI05 policies through a
``config.om.json`` manifest that enumerates the two OM artifacts (the VLM
prefill stage and the action-expert denoising stage) and their execution
order. See ``inference_service/core/compiled_policy.py`` for the consumer.

PI05 export is split across two scripts (VLM and action expert) that run
independently, so this module performs a read-merge-write *upsert*: each run
records its own artifact role while preserving the other, producing a complete
manifest once both scripts have run.
"""

from __future__ import annotations

import json
from pathlib import Path

OM_MANIFEST_BASENAME = "config.om.json"
POLICY_TYPE = "pi05"
BACKEND = "ascend_om"
SCHEMA_VERSION = 1
EXECUTION_ORDER = ["vlm", "action_expert"]
ARTIFACT_ROLES = frozenset(EXECUTION_ORDER)


def _relative_or_absolute_path(path: Path, base_dir: Path) -> str:
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_dir(manifest_dir: str | Path) -> Path:
    directory = Path(manifest_dir).expanduser()
    if not directory.is_absolute():
        directory = (Path.cwd() / directory).resolve()
    return directory


def upsert_pi05_om_manifest(manifest_dir: str | Path, role: str, om_path: str | Path) -> Path:
    """Insert or update one artifact role in the PI05 OM manifest.

    Args:
        manifest_dir: Directory that holds ``config.om.json`` (the policy dir).
        role: Artifact role, one of ``vlm`` / ``action_expert``.
        om_path: Predicted ``.om`` artifact path for this role. Stored relative
            to ``manifest_dir`` when possible, otherwise as an absolute path.

    Returns:
        Path to the written ``config.om.json``.
    """
    if role not in ARTIFACT_ROLES:
        raise ValueError(f"Unknown PI05 OM artifact role {role!r}; expected one of {sorted(ARTIFACT_ROLES)}")

    directory = _resolve_dir(manifest_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / OM_MANIFEST_BASENAME

    artifacts: dict[str, object] = {}
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as f:
            existing = json.load(f)
        existing_backend = str(existing.get("backend", "")).lower().strip()
        if existing_backend and existing_backend != BACKEND:
            raise ValueError(
                f"Existing manifest backend {existing_backend!r} does not match {BACKEND!r}: {manifest_path}"
            )
        existing_policy = str(existing.get("policy_type", "")).lower().strip()
        if existing_policy and existing_policy != POLICY_TYPE:
            raise ValueError(
                f"Existing manifest policy_type {existing_policy!r} does not match {POLICY_TYPE!r}: {manifest_path}"
            )
        raw_artifacts = existing.get("artifacts")
        if isinstance(raw_artifacts, dict):
            # Preserve the original JSON shape of other roles' artifacts. The
            # runtime loader accepts both a bare path string and an object form
            # ({"path": ...}), so coercing values to str() would corrupt object
            # entries into literal "{'path': ...}" strings.
            artifacts = {str(k): v for k, v in raw_artifacts.items()}

    om_path = Path(om_path).expanduser()
    if not om_path.is_absolute():
        om_path = directory / om_path
    artifacts[role] = _relative_or_absolute_path(om_path, directory)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "policy_type": POLICY_TYPE,
        "backend": BACKEND,
        "artifacts": artifacts,
        "execution": list(EXECUTION_ORDER),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    return manifest_path
