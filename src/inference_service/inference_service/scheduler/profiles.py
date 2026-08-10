"""Deadline-aware best-effort profile framework.

Loads offline-measured profiles from a read-only YAML/JSON file and validates
min-samples, max-age, and fingerprint matches for request-level admission.

Profile data (latency_p99_ms, goal_acceptance_p999_ms, sample_count) comes from
real offline benchmarks on the target hardware. The values define a p99
admission SLA, not an absolute completion guarantee. Profiles declare the input
contract and maximum prompt size covered by their measurements; requests outside
that coverage fail closed instead of reusing an unrelated estimate.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

_PROFILE_FIELDS = frozenset(
    {
        "deployment_fingerprint",
        "hardware_fingerprint",
        "profile_compatibility_fingerprint",
        "scope",
        "work_class",
        "closure_key",
        "hardware_priority",
        "input_contract_fingerprint",
        "prompt_bytes_max",
        "goal_acceptance_p999_ms",
        "latency_p99_ms",
        "profiled_at_ns",
        "sample_count",
    }
)
_PROFILE_SCOPE = "global_proxy"
_WORK_CLASSES = frozenset({1, 2})


class ProfileError(Exception):
    """Raised for profile load/validation errors."""


@dataclass(frozen=True)
class ClosureProfile:
    """End-to-end closure profile for a work class."""

    deployment_fingerprint: str
    hardware_fingerprint: str
    profile_compatibility_fingerprint: str
    scope: str  # global_proxy
    work_class: int  # SESSION_CONTROL=1 / ACTION_GENERATION=2
    closure_key: str  # "full_infer"
    hardware_priority: int
    input_contract_fingerprint: str
    prompt_bytes_max: int
    goal_acceptance_p999_ms: float
    latency_p99_ms: float
    profiled_at_ns: int
    sample_count: int


@dataclass
class ProfileRegistry:
    """Validated read-only profile store.

    Loads profiles from a YAML/JSON file at startup, validates fingerprint
    match, min-samples, and max-age. Without valid profiles, priority-0
    admission fails-closed (no_feasible_deadline).

    Thread-safe after load (immutable reads).
    """

    profile_path: str
    profile_min_samples: int
    profile_max_age_days: int
    deployment_fingerprint: str
    hardware_fingerprint: str
    profile_compatibility_fingerprint: str
    now_ns: Callable[[], int]
    _closure_profiles: list[ClosureProfile] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def load(self) -> None:
        """Load and validate the profile file. Fail-closed: invalid/missing/
        expired profiles are dropped; the registry reports empty for those keys."""
        with self._lock:
            self._closure_profiles.clear()
            path = Path(self.profile_path)
            if not path.is_file():
                raise ProfileError(f"profile_path does not exist: {path}")
            try:
                raw = self._read_file(path)
            except Exception as exc:
                raise ProfileError(f"failed to read profile file: {exc}") from exc
            try:
                self._parse_and_validate_locked(raw)
            except Exception:
                self._closure_profiles.clear()
                raise

    def _read_file(self, path: Path) -> dict:
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml

            return yaml.safe_load(text) or {}
        return json.loads(text)

    def _parse_and_validate_locked(self, raw: dict) -> None:
        if not isinstance(raw, dict):
            raise ProfileError("profile document must be an object")
        unknown = sorted(set(raw) - {"closure_profiles"})
        if unknown:
            raise ProfileError(f"unsupported profile fields: {unknown}")
        now = self.now_ns()
        max_age_ns = self.profile_max_age_days * 86_400 * 1_000_000_000
        dep_fp = self.deployment_fingerprint
        hardware_fingerprint = self.hardware_fingerprint
        entries = raw.get("closure_profiles", [])
        if not isinstance(entries, list):
            raise ProfileError("closure_profiles must be an array")
        seen_keys: set[tuple[object, ...]] = set()
        for entry in entries:
            p = self._validate_closure(entry, dep_fp, hardware_fingerprint, now, max_age_ns)
            if p is not None:
                key = (
                    p.scope,
                    p.work_class,
                    p.closure_key,
                    p.hardware_priority,
                    p.input_contract_fingerprint,
                    p.prompt_bytes_max,
                )
                if key in seen_keys:
                    raise ProfileError(f"duplicate closure profile key: {key}")
                seen_keys.add(key)
                self._closure_profiles.append(p)

    def _validate_sample_count(self, entry: dict) -> int:
        value = entry.get("sample_count", 0)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ProfileError("sample_count must be an integer")
        sc = value
        if sc < self.profile_min_samples:
            raise ProfileError(f"sample_count {sc} < min {self.profile_min_samples}")
        return sc

    def _validate_age(self, profiled_at_ns: int, now: int, max_age_ns: int) -> bool:
        if profiled_at_ns <= 0:
            raise ProfileError("profiled_at_ns must be positive")
        if profiled_at_ns > now:
            raise ProfileError("profiled_at_ns must not be in the future")
        return (now - profiled_at_ns) <= max_age_ns

    def _validate_closure(self, entry, dep_fp, hardware_fingerprint, now, max_age_ns):
        if not isinstance(entry, dict):
            raise ProfileError("closure profile entry must be an object")
        unknown = sorted(set(entry) - _PROFILE_FIELDS)
        if unknown:
            raise ProfileError(f"unsupported closure profile fields: {unknown}")
        if (
            entry.get("deployment_fingerprint") != dep_fp
            or entry.get("hardware_fingerprint") != hardware_fingerprint
            or entry.get("profile_compatibility_fingerprint") != self.profile_compatibility_fingerprint
        ):
            return None
        profiled_at_value = entry.get("profiled_at_ns", 0)
        if isinstance(profiled_at_value, bool) or not isinstance(profiled_at_value, int):
            raise ProfileError("profiled_at_ns must be an integer")
        profiled_at = profiled_at_value
        if not self._validate_age(profiled_at, now, max_age_ns):
            return None
        sc = self._validate_sample_count(entry)
        p99_value = entry.get("latency_p99_ms", 0)
        if isinstance(p99_value, bool) or not isinstance(p99_value, int | float):
            raise ProfileError("latency_p99_ms must be numeric")
        p99 = float(p99_value)
        if not math.isfinite(p99) or p99 <= 0:
            raise ProfileError("latency_p99_ms must be finite positive")
        scope = entry.get("scope", _PROFILE_SCOPE)
        if scope != _PROFILE_SCOPE:
            raise ProfileError(f"scope must be {_PROFILE_SCOPE!r}")
        work_class = entry.get("work_class")
        if isinstance(work_class, bool) or not isinstance(work_class, int):
            raise ProfileError("work_class must be an integer")
        if work_class not in _WORK_CLASSES:
            raise ProfileError(f"work_class must be one of {sorted(_WORK_CLASSES)}")
        hardware_priority = entry.get("hardware_priority", 0)
        if isinstance(hardware_priority, bool) or not isinstance(hardware_priority, int):
            raise ProfileError("hardware_priority must be an integer")
        if hardware_priority < 0:
            raise ProfileError("hardware_priority must be non-negative")
        closure_key = entry.get("closure_key")
        if not isinstance(closure_key, str) or not closure_key:
            raise ProfileError("closure_key must be a non-empty string")
        input_contract_fingerprint = entry.get("input_contract_fingerprint", "")
        if not isinstance(input_contract_fingerprint, str):
            raise ProfileError("input_contract_fingerprint must be a string")
        if input_contract_fingerprint and (
            len(input_contract_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in input_contract_fingerprint)
        ):
            raise ProfileError("input_contract_fingerprint must be empty or a lowercase SHA-256 digest")
        prompt_bytes_max = entry.get("prompt_bytes_max", 0)
        if isinstance(prompt_bytes_max, bool) or not isinstance(prompt_bytes_max, int) or prompt_bytes_max < 0:
            raise ProfileError("prompt_bytes_max must be a non-negative integer")
        if work_class == 1 and (input_contract_fingerprint or prompt_bytes_max != 0):
            raise ProfileError("session_control profiles must use empty input contract and prompt_bytes_max=0")
        if work_class == 2 and not input_contract_fingerprint:
            raise ProfileError("action_generation profiles require input_contract_fingerprint")
        acceptance_value = entry.get("goal_acceptance_p999_ms", 0)
        if isinstance(acceptance_value, bool) or not isinstance(acceptance_value, int | float):
            raise ProfileError("goal_acceptance_p999_ms must be numeric")
        acceptance = float(acceptance_value)
        if not math.isfinite(acceptance) or acceptance <= 0:
            raise ProfileError("goal_acceptance_p999_ms must be finite positive")
        return ClosureProfile(
            deployment_fingerprint=dep_fp,
            hardware_fingerprint=hardware_fingerprint,
            profile_compatibility_fingerprint=self.profile_compatibility_fingerprint,
            scope=scope,
            work_class=work_class,
            closure_key=closure_key,
            hardware_priority=hardware_priority,
            input_contract_fingerprint=input_contract_fingerprint,
            prompt_bytes_max=prompt_bytes_max,
            goal_acceptance_p999_ms=acceptance,
            latency_p99_ms=p99,
            profiled_at_ns=profiled_at,
            sample_count=sc,
        )

    # ------------------------------------------------------------------
    # Profile queries used by priority-0 finish estimation.
    # ------------------------------------------------------------------

    def closure_p99_ms(
        self,
        *,
        work_class: int,
        closure_key: str,
        hardware_priority: int,
        input_contract_fingerprint: str,
        prompt_bytes: int,
        scope: str = _PROFILE_SCOPE,
    ) -> float | None:
        """Return the p99 latency for a work class + closure key + scope.
        None if no valid profile (fail-closed)."""
        with self._lock:
            matches = [
                profile
                for profile in self._closure_profiles
                if profile.work_class == work_class
                and profile.closure_key == closure_key
                and profile.hardware_priority == hardware_priority
                and profile.input_contract_fingerprint == input_contract_fingerprint
                and profile.prompt_bytes_max >= prompt_bytes
                and profile.scope == scope
            ]
            if not matches:
                return None
            selected = min(matches, key=lambda profile: profile.prompt_bytes_max)
            return selected.latency_p99_ms

    def goal_acceptance_p999_ms(
        self,
        *,
        work_class: int,
        closure_key: str,
        hardware_priority: int,
        input_contract_fingerprint: str,
        prompt_bytes: int,
        scope: str = _PROFILE_SCOPE,
    ) -> float | None:
        """Return the goal acceptance p99.9 for a work class and scope.
        None if no valid profile."""
        with self._lock:
            matches = [
                profile
                for profile in self._closure_profiles
                if profile.work_class == work_class
                and profile.closure_key == closure_key
                and profile.hardware_priority == hardware_priority
                and profile.input_contract_fingerprint == input_contract_fingerprint
                and profile.prompt_bytes_max >= prompt_bytes
                and profile.scope == scope
            ]
            if not matches:
                return None
            selected = min(matches, key=lambda profile: profile.prompt_bytes_max)
            return selected.goal_acceptance_p999_ms

    @property
    def profile_count(self) -> int:
        with self._lock:
            return len(self._closure_profiles)


__all__ = [
    "ClosureProfile",
    "ProfileError",
    "ProfileRegistry",
]
