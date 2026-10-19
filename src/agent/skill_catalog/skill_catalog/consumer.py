"""Verified read-only views for catalog snapshot consumers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from skill_catalog.digest import (
    CAPABILITY_PREIMAGE_FIELDS,
    PROVENANCE_PREIMAGE_FIELDS,
    REGISTRY_PREIMAGE_FIELDS,
    deep_freeze,
    derive_capability_digest,
    derive_capability_view_from_registry,
    derive_provenance_digest,
    derive_registry_digest,
    to_canonical_json,
)


class CatalogConsumerError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class CatalogIdentity:
    registry_epoch: str
    generation: int
    registry_digest: str


@dataclass(frozen=True)
class VerifiedCatalogView:
    identity: CatalogIdentity
    capability_digest: str
    provenance_digest: str
    profile_name: str
    aliases: MappingProxyType
    capability_view: MappingProxyType
    enabled_names: frozenset[str]
    planner_visible_names: frozenset[str]
    named_pose_names: tuple[str, ...]
    timeout_policy: MappingProxyType
    robot_context: MappingProxyType


def verify_snapshot_response(snapshot: Any, expected: CatalogIdentity) -> VerifiedCatalogView:
    """Verify a GetSkillSnapshot-like response and expose planner-safe fields."""

    if not bool(snapshot.success):
        raise CatalogConsumerError(
            str(snapshot.error_code or "SKILL_SNAPSHOT_NOT_RETAINED"),
            str(snapshot.message or "exact snapshot is unavailable"),
        )
    actual = CatalogIdentity(
        str(snapshot.registry_epoch),
        int(snapshot.generation),
        str(snapshot.registry_digest),
    )
    if actual != expected:
        raise CatalogConsumerError("SKILL_REGISTRY_VERSION_MISMATCH", "snapshot identity does not match status")
    try:
        payload: Any = json.loads(snapshot.snapshot_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot is not valid JSON") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "registry_preimage", "capability_preimage", "provenance_preimage"}
        or payload.get("schema_version") != 1
    ):
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot schema is invalid")
    if to_canonical_json(payload) != snapshot.snapshot_json:
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot is not canonical JSON")
    registry = payload.get("registry_preimage")
    capability = payload.get("capability_preimage")
    provenance = payload.get("provenance_preimage")
    if not all(isinstance(value, dict) for value in (registry, capability, provenance)):
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot preimages are invalid")
    if (
        set(registry) != REGISTRY_PREIMAGE_FIELDS
        or set(capability) != CAPABILITY_PREIMAGE_FIELDS
        or set(provenance) != PROVENANCE_PREIMAGE_FIELDS
        or registry.get("schema_version") != 1
        or capability.get("schema_version") != 1
        or provenance.get("schema_version") != 1
    ):
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot preimage schema is invalid")
    if derive_registry_digest(registry) != actual.registry_digest:
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "registry digest does not match")
    if derive_capability_digest(capability) != snapshot.capability_digest:
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "capability digest does not match")
    if derive_provenance_digest(provenance) != snapshot.provenance_digest:
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "provenance digest does not match")
    aliases = registry.get("aliases")
    capability_view = capability.get("capability_view")
    robot_context = registry.get("robot_context")
    if not isinstance(aliases, dict) or not isinstance(capability_view, dict) or not isinstance(robot_context, dict):
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot planner views are invalid")
    repeated_fields = ("robot_name", "profile_name", "enabled_skill_names", "planner_visible_skill_names")
    if any(registry.get(field) != capability.get(field) for field in repeated_fields):
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot preimages disagree")
    if robot_context.get("timeout_policy") != capability.get("timeout_policy"):
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot timeout policy disagrees")
    if sorted(robot_context.get("named_poses", {})) != capability.get("named_pose_names"):
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot named poses disagree")
    try:
        derived_capability_view = derive_capability_view_from_registry(registry)
    except (TypeError, ValueError) as exc:
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", str(exc)) from exc
    if to_canonical_json(derived_capability_view) != to_canonical_json(capability_view):
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "snapshot capability view disagrees")
    enabled_names = frozenset(str(name) for name in registry.get("enabled_skill_names", []))
    planner_visible_names = frozenset(str(name) for name in registry.get("planner_visible_skill_names", []))
    if not planner_visible_names <= enabled_names:
        raise CatalogConsumerError("SKILL_SNAPSHOT_DIGEST_MISMATCH", "planner-visible entries are not enabled")
    return VerifiedCatalogView(
        identity=actual,
        capability_digest=str(snapshot.capability_digest),
        provenance_digest=str(snapshot.provenance_digest),
        profile_name=str(capability.get("profile_name", "")),
        aliases=deep_freeze(aliases),
        capability_view=deep_freeze(capability_view),
        enabled_names=enabled_names,
        planner_visible_names=planner_visible_names,
        named_pose_names=tuple(str(name) for name in capability.get("named_pose_names", [])),
        timeout_policy=deep_freeze(capability.get("timeout_policy", {})),
        robot_context=deep_freeze(robot_context),
    )
