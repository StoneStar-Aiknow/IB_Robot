"""Canonical JSON and freeze helpers shared by every digest consumer.

This module is the single source of truth for canonical serialization rules
across ``robot_config``, ``embodied_common``, ``skill_catalog`` and
``skill_library``. It lives in the lowest shared package so that no consumer
needs to import a higher-level business package (which would create a circular
dependency). The rules mirror
``docs/lightweight_skill_package_registry_design_zh.md`` section 6.4 exactly.

Every process (compiler, runtime, CLI, safety) MUST use these helpers so that
the same frozen data always produces byte-identical preimages and digests.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- #
# Canonical JSON                                                              #
# --------------------------------------------------------------------------- #


def to_canonical_json(payload: Any) -> str:
    """Serialize ``payload`` with the canonical rules from section 6.4.

    Rules:
      * keys sorted lexicographically;
      * tuple/list both encoded as JSON arrays;
      * set/frozenset encoded as a sorted array;
      * ``Path`` -> POSIX relative string (caller converts);
      * finite floats only (NaN/Infinity rejected, ``-0.0`` -> ``0.0``);
      * unicode escaped via ``ensure_ascii=True``.
    """

    normalized = _normalize_for_json(payload)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _normalize_for_json(value: Any) -> Any:
    # Treat MappingProxyType (our frozen mapping) as a plain mapping.
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("digest preimage mapping keys must be strings")
        return {key: _normalize_for_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, set | frozenset):
        return [_normalize_for_json(v) for v in sorted(value, key=_sort_key)]
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError("NaN and Infinity are not allowed in digest preimages")
        if value == 0.0:
            return 0.0  # normalize -0.0 -> 0.0
        return value
    if isinstance(value, int):
        return value
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, os.PathLike):
        return _normalize_for_json(os.fspath(value))
    raise TypeError(f"unsupported type in digest preimage: {type(value).__name__}")


def _sort_key(value: Any) -> Any:
    # set/frozenset elements are typically strings; provide a stable fallback.
    if isinstance(value, str):
        return (0, value)
    return (1, str(value))


def sha256_text(payload: str) -> str:
    """Lowercase hex SHA-256 of a UTF-8 encoded string."""

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# Deep freeze (section 9.5)                                                    #
# --------------------------------------------------------------------------- #


def deep_freeze(value: Any) -> Any:
    """Recursively convert mutable containers to read-only ones.

    * ``list`` -> ``tuple`` (elements frozen);
    * ``set``  -> ``frozenset`` (elements frozen);
    * ``dict`` -> ``MappingProxyType`` (values frozen).

    Already-frozen types (tuple, frozenset, MappingProxyType, dataclass
    instances, scalars) are returned as-is or re-wrapped for nested safety.
    """

    return _freeze(value, _freeze_cache={})


def _freeze(value: Any, *, _freeze_cache: dict[int, Any]) -> Any:
    # Frozen dataclasses and PrimitiveDescriptor-like objects are treated as
    # opaque scalars; their fields are assumed already frozen by the owner.
    if isinstance(value, str | int | float | bool) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            raise ValueError("NaN and Infinity are not allowed in frozen catalog data")
        return value
    if isinstance(value, MappingProxyType):
        # Re-freeze contents defensively (cheap if already frozen).
        return MappingProxyType({k: _freeze(v, _freeze_cache=_freeze_cache) for k, v in value.items()})
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v, _freeze_cache=_freeze_cache) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v, _freeze_cache=_freeze_cache) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v, _freeze_cache=_freeze_cache) for v in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze(v, _freeze_cache=_freeze_cache) for v in value)
    # Fall back to the object itself (dataclass instances, enums, etc.).
    return value


__all__ = [
    "SCHEMA_VERSION",
    "deep_freeze",
    "sha256_bytes",
    "sha256_text",
    "to_canonical_json",
]
