"""Unit tests for the profile framework.

Covers ProfileRegistry loading, identity validation, age/sample gates, and
fail-closed queries when no measured profile exists.
"""

from __future__ import annotations

import json
import threading

import pytest

from inference_service.scheduler.profiles import ProfileError, ProfileRegistry


class _Clock:
    def __init__(self, start: int = 10_000_000_000) -> None:
        self.t = start
        self._lock = threading.Lock()

    def now_ns(self) -> int:
        with self._lock:
            return self.t

    def advance(self, ns: int) -> None:
        with self._lock:
            self.t += ns


def _write_profile(path, *, closure=None, closures=None):
    data = {}
    if closure is not None:
        data["closure_profiles"] = [closure]
    elif closures is not None:
        data["closure_profiles"] = closures
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _closure_entry(**kw):
    base = {
        "deployment_fingerprint": "d" * 64,
        "hardware_fingerprint": "a" * 64,
        "profile_compatibility_fingerprint": "t" * 64,
        "scope": "global_proxy",
        "work_class": 2,
        "closure_key": "full_infer",
        "hardware_priority": 0,
        "input_contract_fingerprint": "c" * 64,
        "prompt_bytes_max": 4096,
        "goal_acceptance_p999_ms": 5.0,
        "latency_p99_ms": 50.0,
        "profiled_at_ns": 1,
        "sample_count": 10000,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# ProfileRegistry: load + validation
# ---------------------------------------------------------------------------


def test_load_valid_closure_profile(tmp_path):
    clock = _Clock()
    path = _write_profile(tmp_path / "profile.json", closure=_closure_entry())
    reg = ProfileRegistry(
        profile_path=str(path),
        profile_min_samples=10000,
        profile_max_age_days=30,
        deployment_fingerprint="d" * 64,
        hardware_fingerprint="a" * 64,
        profile_compatibility_fingerprint="t" * 64,
        now_ns=clock.now_ns,
    )
    reg.load()
    assert reg.profile_count == 1
    p99 = reg.closure_p99_ms(
        work_class=2,
        closure_key="full_infer",
        hardware_priority=0,
        input_contract_fingerprint="c" * 64,
        prompt_bytes=0,
        scope="global_proxy",
    )
    assert p99 == 50.0
    assert (
        reg.closure_p99_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=1,
            input_contract_fingerprint="c" * 64,
            prompt_bytes=0,
        )
        is None
    )
    assert (
        reg.closure_p99_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=0,
            input_contract_fingerprint="b" * 64,
            prompt_bytes=0,
        )
        is None
    )


def test_load_rejects_insufficient_samples(tmp_path):
    clock = _Clock()
    path = _write_profile(tmp_path / "profile.json", closure=_closure_entry(sample_count=5000))
    reg = ProfileRegistry(
        profile_path=str(path),
        profile_min_samples=10000,
        profile_max_age_days=30,
        deployment_fingerprint="d" * 64,
        hardware_fingerprint="a" * 64,
        profile_compatibility_fingerprint="t" * 64,
        now_ns=clock.now_ns,
    )
    with pytest.raises(ProfileError, match="sample_count"):
        reg.load()


def test_load_drops_expired_profiles_silently(tmp_path):
    clock = _Clock(start=100_000_000_000_000)  # far future
    path = _write_profile(tmp_path / "profile.json", closure=_closure_entry(profiled_at_ns=1))
    reg = ProfileRegistry(
        profile_path=str(path),
        profile_min_samples=10000,
        profile_max_age_days=1,  # 1 day
        deployment_fingerprint="d" * 64,
        hardware_fingerprint="a" * 64,
        profile_compatibility_fingerprint="t" * 64,
        now_ns=clock.now_ns,
    )
    reg.load()
    assert reg.profile_count == 0  # dropped (expired)
    assert (
        reg.closure_p99_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=0,
            input_contract_fingerprint="c" * 64,
            prompt_bytes=0,
        )
        is None
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("deployment_fingerprint", "e" * 64),
        ("hardware_fingerprint", "b" * 64),
        ("profile_compatibility_fingerprint", "u" * 64),
    ],
)
def test_load_drops_closure_identity_mismatch(tmp_path, field, value):
    clock = _Clock()
    path = _write_profile(tmp_path / "profile.json", closure=_closure_entry(**{field: value}))
    reg = ProfileRegistry(
        profile_path=str(path),
        profile_min_samples=10000,
        profile_max_age_days=30,
        deployment_fingerprint="d" * 64,
        hardware_fingerprint="a" * 64,
        profile_compatibility_fingerprint="t" * 64,
        now_ns=clock.now_ns,
    )
    reg.load()
    assert reg.profile_count == 0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("scope", "unknown", "scope"),
        ("scope", "pipeline", "scope"),
        ("input_contract_fingerprint", "dynamic", "input_contract_fingerprint"),
        ("work_class", 3, "work_class"),
        ("hardware_priority", -1, "hardware_priority"),
        ("goal_acceptance_p999_ms", float("nan"), "goal_acceptance_p999_ms"),
        ("profiled_at_ns", 20_000_000_000, "future"),
    ],
)
def test_load_rejects_invalid_profile_semantics(tmp_path, field, value, error):
    clock = _Clock()
    path = _write_profile(tmp_path / "profile.json", closure=_closure_entry(**{field: value}))
    reg = ProfileRegistry(
        profile_path=str(path),
        profile_min_samples=10000,
        profile_max_age_days=30,
        deployment_fingerprint="d" * 64,
        hardware_fingerprint="a" * 64,
        profile_compatibility_fingerprint="t" * 64,
        now_ns=clock.now_ns,
    )

    with pytest.raises(ProfileError, match=error):
        reg.load()


def test_load_rejects_duplicate_profile_query_key(tmp_path):
    clock = _Clock()
    entry = _closure_entry()
    path = _write_profile(tmp_path / "profile.json", closures=[entry, dict(entry)])
    reg = ProfileRegistry(
        profile_path=str(path),
        profile_min_samples=10000,
        profile_max_age_days=30,
        deployment_fingerprint="d" * 64,
        hardware_fingerprint="a" * 64,
        profile_compatibility_fingerprint="t" * 64,
        now_ns=clock.now_ns,
    )

    with pytest.raises(ProfileError, match="duplicate closure profile key"):
        reg.load()


def test_missing_profile_file_raises():
    clock = _Clock()
    reg = ProfileRegistry(
        profile_path="/nonexistent/path.json",
        profile_min_samples=10000,
        profile_max_age_days=30,
        deployment_fingerprint="d" * 64,
        hardware_fingerprint="a" * 64,
        profile_compatibility_fingerprint="t" * 64,
        now_ns=clock.now_ns,
    )
    with pytest.raises(ProfileError, match="does not exist"):
        reg.load()


# ---------------------------------------------------------------------------
# Fail-closed: no profile -> None
# ---------------------------------------------------------------------------


def test_no_profile_returns_none_for_p99():
    clock = _Clock()
    reg = ProfileRegistry(
        profile_path="/dev/null",  # won't load anything
        profile_min_samples=1,
        profile_max_age_days=30,
        deployment_fingerprint="d" * 64,
        hardware_fingerprint="a" * 64,
        profile_compatibility_fingerprint="t" * 64,
        now_ns=clock.now_ns,
    )
    # load an empty file
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("{}")
        reg.profile_path = f.name
    reg.load()
    assert (
        reg.closure_p99_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=0,
            input_contract_fingerprint="c" * 64,
            prompt_bytes=0,
        )
        is None
    )
    assert (
        reg.goal_acceptance_p999_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=0,
            input_contract_fingerprint="c" * 64,
            prompt_bytes=0,
        )
        is None
    )


def test_scope_default_matches_global_queries(tmp_path):
    clock = _Clock()
    entry = _closure_entry()
    entry.pop("scope")
    path = _write_profile(tmp_path / "profile.json", closure=entry)
    reg = ProfileRegistry(
        profile_path=str(path),
        profile_min_samples=10000,
        profile_max_age_days=30,
        deployment_fingerprint="d" * 64,
        hardware_fingerprint="a" * 64,
        profile_compatibility_fingerprint="t" * 64,
        now_ns=clock.now_ns,
    )

    reg.load()

    assert (
        reg.closure_p99_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=0,
            input_contract_fingerprint="c" * 64,
            prompt_bytes=0,
        )
        == 50.0
    )


def test_query_selects_smallest_profile_covering_prompt_and_fails_outside_coverage(tmp_path):
    clock = _Clock()
    short = _closure_entry(prompt_bytes_max=128, latency_p99_ms=20.0)
    long = _closure_entry(prompt_bytes_max=1024, latency_p99_ms=80.0)
    path = _write_profile(tmp_path / "profile.json", closures=[short, long])
    reg = ProfileRegistry(
        profile_path=str(path),
        profile_min_samples=10000,
        profile_max_age_days=30,
        deployment_fingerprint="d" * 64,
        hardware_fingerprint="a" * 64,
        profile_compatibility_fingerprint="t" * 64,
        now_ns=clock.now_ns,
    )
    reg.load()

    assert (
        reg.closure_p99_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=0,
            input_contract_fingerprint="c" * 64,
            prompt_bytes=100,
        )
        == 20.0
    )
    assert (
        reg.closure_p99_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=0,
            input_contract_fingerprint="c" * 64,
            prompt_bytes=512,
        )
        == 80.0
    )
    assert (
        reg.closure_p99_ms(
            work_class=2,
            closure_key="full_infer",
            hardware_priority=0,
            input_contract_fingerprint="c" * 64,
            prompt_bytes=2048,
        )
        is None
    )
