"""Verified, generation-indexed skill snapshots for Safety Guard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from threading import RLock
from types import MappingProxyType
from typing import Any

from embodied_common.primitive_contracts import PRIMITIVE_CONTRACT_DIGEST
from skill_catalog.consumer import CatalogIdentity, verify_snapshot_response
from skill_catalog.digest import (
    deep_freeze,
    derive_capability_digest,
    derive_provenance_digest,
    derive_registry_digest,
    to_canonical_json,
)


class SnapshotCacheError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class SnapshotIdentity:
    registry_epoch: str
    generation: int
    registry_digest: str


@dataclass(frozen=True)
class VerifiedSafetySnapshot:
    identity: SnapshotIdentity
    capability_digest: str
    provenance_digest: str
    templates: MappingProxyType
    robot_context: MappingProxyType
    payload: MappingProxyType


class SafetySnapshotCache:
    """Thread-safe exact snapshot cache; validation never performs ROS I/O."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._snapshots: dict[tuple[str, int], VerifiedSafetySnapshot] = {}
        self._current_key: tuple[str, int] | None = None

    @property
    def current_identity(self) -> SnapshotIdentity | None:
        with self._lock:
            snapshot = self._snapshots.get(self._current_key) if self._current_key else None
            return snapshot.identity if snapshot else None

    def activate(
        self,
        *,
        registry_epoch: str,
        generation: int,
        registry_digest: str,
        capability_digest: str,
        provenance_digest: str,
        snapshot_json: str,
        make_current: bool,
    ) -> VerifiedSafetySnapshot:
        if not registry_epoch or generation <= 0 or not registry_digest:
            raise SnapshotCacheError("SKILL_SCHEMA_INVALID", "snapshot identity is incomplete")
        try:
            payload: Any = json.loads(snapshot_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SnapshotCacheError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot payload is not valid JSON") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise SnapshotCacheError("SKILL_SCHEMA_INVALID", "snapshot payload schema_version must be 1")
        if set(payload) != {"schema_version", "registry_preimage", "capability_preimage", "provenance_preimage"}:
            raise SnapshotCacheError("SKILL_SCHEMA_INVALID", "snapshot payload fields do not match v1")
        if to_canonical_json(payload) != snapshot_json:
            raise SnapshotCacheError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot payload is not canonical JSON")
        registry_preimage = payload["registry_preimage"]
        capability_preimage = payload["capability_preimage"]
        provenance_preimage = payload["provenance_preimage"]
        if not all(isinstance(value, dict) for value in (registry_preimage, capability_preimage, provenance_preimage)):
            raise SnapshotCacheError("SKILL_SCHEMA_INVALID", "snapshot preimages must be objects")
        if derive_registry_digest(registry_preimage) != registry_digest:
            raise SnapshotCacheError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "registry digest does not match snapshot")
        if derive_capability_digest(capability_preimage) != capability_digest:
            raise SnapshotCacheError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "capability digest does not match snapshot")
        if derive_provenance_digest(provenance_preimage) != provenance_digest:
            raise SnapshotCacheError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "provenance digest does not match snapshot")
        response = type(
            "SnapshotResponse",
            (),
            {
                "success": True,
                "error_code": "",
                "message": "",
                "registry_epoch": registry_epoch,
                "generation": generation,
                "registry_digest": registry_digest,
                "capability_digest": capability_digest,
                "provenance_digest": provenance_digest,
                "snapshot_json": snapshot_json,
            },
        )()
        try:
            verify_snapshot_response(response, CatalogIdentity(registry_epoch, generation, registry_digest))
        except Exception as exc:
            raise SnapshotCacheError("SKILL_SNAPSHOT_DIGEST_MISMATCH", str(exc)) from exc
        if registry_preimage.get("primitive_contract_digest") != PRIMITIVE_CONTRACT_DIGEST:
            raise SnapshotCacheError(
                "SKILL_SNAPSHOT_DIGEST_MISMATCH", "primitive contract digest does not match local SSOT"
            )
        skills = registry_preimage.get("skills")
        robot_context = registry_preimage.get("robot_context")
        if not isinstance(robot_context, dict):
            raise SnapshotCacheError("SKILL_SCHEMA_INVALID", "registry robot_context must be an object")
        if not isinstance(skills, list):
            raise SnapshotCacheError("SKILL_SCHEMA_INVALID", "registry skills must be an array")
        templates: dict[str, dict[str, Any]] = {}
        for entry in skills:
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("name"), str)
                or not isinstance(entry.get("template"), dict)
            ):
                raise SnapshotCacheError("SKILL_SCHEMA_INVALID", "registry skill entry is invalid")
            name = entry["name"]
            if not name or name in templates:
                raise SnapshotCacheError("SKILL_SCHEMA_INVALID", "registry skill names must be unique and non-empty")
            templates[name] = entry["template"]

        identity = SnapshotIdentity(registry_epoch, generation, registry_digest)
        snapshot = VerifiedSafetySnapshot(
            identity=identity,
            capability_digest=capability_digest,
            provenance_digest=provenance_digest,
            templates=deep_freeze(templates),
            robot_context=deep_freeze(robot_context),
            payload=deep_freeze(payload),
        )
        key = (registry_epoch, generation)
        with self._lock:
            existing = self._snapshots.get(key)
            if existing is not None and existing != snapshot:
                raise SnapshotCacheError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot identity changed in cache")
            self._snapshots[key] = snapshot
            if make_current:
                self._current_key = key
        return snapshot

    def get(self, identity: SnapshotIdentity) -> VerifiedSafetySnapshot:
        with self._lock:
            snapshot = self._snapshots.get((identity.registry_epoch, identity.generation))
            if snapshot is None:
                raise SnapshotCacheError("SKILL_SNAPSHOT_NOT_RETAINED", "exact safety snapshot is not cached")
            if snapshot.identity.registry_digest != identity.registry_digest:
                raise SnapshotCacheError("SKILL_REGISTRY_VERSION_MISMATCH", "snapshot digest does not match")
            return snapshot

    def mark_current(self, identity: SnapshotIdentity) -> None:
        self.get(identity)
        with self._lock:
            self._current_key = (identity.registry_epoch, identity.generation)

    def reconcile(self, registry_epoch: str, retained_generations: set[int], *, keep_recent: int = 2) -> None:
        if keep_recent < 0:
            raise ValueError("keep_recent must be non-negative")
        with self._lock:
            local_generations = sorted(
                (generation for epoch, generation in self._snapshots if epoch == registry_epoch), reverse=True
            )
            keep = set(retained_generations)
            keep.update(local_generations[:keep_recent])
            if self._current_key is not None and self._current_key[0] != registry_epoch:
                self._current_key = None
            self._snapshots = {
                key: value for key, value in self._snapshots.items() if key[0] == registry_epoch and key[1] in keep
            }
