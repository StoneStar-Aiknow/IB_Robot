"""Idempotency helpers shared by the Scheduler and pipeline ledgers.

Implements request idempotency keys, canonical fingerprints, UUID4 validation,
and deadline conversion without ROS dependencies.

Key invariants:
  - fingerprint is canonical JSON -> sha256 hex. No Python hash()/repr()/RMW bytes.
  - Time encoded as {"sec": int32, "nanosec": uint32}; strings UTF-8; arrays
    preserve order; integers/booleans not stringified.
  - NaN/Infinity, non-UTF-8, out-of-range Time, undeclared fields are rejected
    BEFORE fingerprinting.
  - effective_deadline_utc is frozen on first ledger miss; replays reuse it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# RFC 4122 UUID4 canonical lowercase text: 8-4-4-4-12.
# variant must be 8/9/a/b; version must be 4.
_UUID4_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
_PIPELINE_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")

_MAX_INT32 = 2_147_483_647
_MAX_UINT32 = 4_294_967_295


class IdempotencyError(ValueError):
    """Raised when a request payload cannot be fingerprinted or fails validation."""


@dataclass(frozen=True)
class CanonicalTime:
    """Canonical encoding of a builtin_interfaces/Time for fingerprinting."""

    sec: int
    nanosec: int

    @classmethod
    def from_msg(cls, msg: Any) -> CanonicalTime:
        sec = int(getattr(msg, "sec", 0))
        nanosec = int(getattr(msg, "nanosec", 0))
        if sec < 0 or nanosec < 0:
            raise IdempotencyError("Time sec/nanosec must be non-negative")
        if sec > _MAX_INT32 or nanosec > _MAX_UINT32:
            raise IdempotencyError("Time field out of range")
        return cls(sec=sec, nanosec=nanosec)


def validate_uuid4(value: str, *, field: str = "id") -> str:
    """Validate RFC 4122 UUID4 canonical lowercase text.

    Non-canonical forms (uppercase, braces, missing variant/version bits) are
    rejected before entering any ledger.
    """
    if not isinstance(value, str) or not _UUID4_RE.fullmatch(value):
        raise IdempotencyError(f"{field} must be canonical lowercase UUID4 text (8-4-4-4-12), got {value!r}")
    return value


def validate_pipeline_id(value: str, *, field: str = "pipeline_id") -> str:
    if not isinstance(value, str) or not _PIPELINE_ID_RE.fullmatch(value):
        raise IdempotencyError(f"{field} must be a non-empty identifier, got {value!r}")
    return value


def _canonicalize(value: Any) -> Any:
    """Recursively canonicalize a value for JSON serialization.

    Rejects NaN/Infinity, non-UTF-8, and unsupported types BEFORE hashing.
    """
    if isinstance(value, int | float):
        if isinstance(value, float) and value != value:  # NaN
            raise IdempotencyError("NaN is not allowed in idempotency payload")
        if isinstance(value, float) and value in (float("inf"), float("-inf")):
            raise IdempotencyError("Infinity is not allowed in idempotency payload")
        return value
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise IdempotencyError(f"non-UTF-8 string in idempotency payload: {exc}") from exc
        return value
    if isinstance(value, Mapping):
        # sort keys for canonical ordering; reject non-string keys.
        items = []
        for key in sorted(value.keys(), key=lambda k: str(k)):
            if not isinstance(key, str):
                raise IdempotencyError(f"non-string mapping key {key!r} in idempotency payload")
            items.append((key, _canonicalize(value[key])))
        return dict(items)
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if hasattr(value, "sec") and hasattr(value, "nanosec"):
        # builtin_interfaces/Time-like; encode as {"sec":..,"nanosec":..}.
        ct = CanonicalTime.from_msg(value)
        return {"sec": ct.sec, "nanosec": ct.nanosec}
    raise IdempotencyError(f"unsupported idempotency payload type {type(value).__name__}")


def canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    """Return the canonical JSON SHA-256 hex fingerprint of a payload.

    Keys sorted, no insignificant whitespace, Time as {"sec","nanosec"}, UTF-8,
    arrays preserve order, integers/booleans not stringified.
    """
    canonical = _canonicalize(payload)
    serialized = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def resolve_entry_deadline_ns(
    deadline_ns: int, *, is_open: bool, default_open_timeout_ns: int, default_request_timeout_ns: int
) -> int:
    """Resolve a zero deadline at the global entry to an absolute deadline.

    Zero at the global entry selects the configured default; Open uses the open
    timeout, while Dispatch and Close use the request timeout. The deadline is
    frozen here on first ledger miss; replays reuse the frozen value.
    """
    if deadline_ns < 0:
        raise IdempotencyError("deadline must be non-negative")
    if deadline_ns > 0:
        return deadline_ns
    now = _now_utc_ns()
    return now + (default_open_timeout_ns if is_open else default_request_timeout_ns)


def deadline_to_monotonic_ns(deadline_utc_ns: int, *, utc_now_ns: int, mono_now_ns: int) -> int:
    """Convert an absolute UTC deadline to a local monotonic deadline.

    Subsequent queuing/execution judgments use only the monotonic deadline; the
    UTC value is only for cross-process pass-through and logging.
    """
    remaining = deadline_utc_ns - utc_now_ns
    if remaining < 0:
        remaining = 0
    return mono_now_ns + remaining


def _now_utc_ns() -> int:
    """Current UTC wall clock in nanoseconds.

    Used only to convert a zero entry deadline into an absolute deadline. It is
    NOT used for in-flight monotonic deadline judgments, which use monotonic time
    after entry to avoid wall-clock skew or back-jumps."""
    return int(datetime.now(tz=timezone.utc).timestamp() * 1_000_000_000)


__all__ = [
    "CanonicalTime",
    "IdempotencyError",
    "canonical_fingerprint",
    "deadline_to_monotonic_ns",
    "resolve_entry_deadline_ns",
    "validate_pipeline_id",
    "validate_uuid4",
]
