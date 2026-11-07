"""Tests for the typed ``robot.rt_infer`` configuration block.

The parse is pure (review #6): no filesystem, no host probes, no installed
manifests. Strictness under test: partial role overrides cover every role,
NaN/inf are rejected, unknown fields/roles are rejected, and the canonical
prefix derives from bundle_version instead of a hand-picked path.
"""

from __future__ import annotations

import pytest

from robot_config.rt_infer_config import (
    PearPeriodicConfig,
    RtInferConfigError,
    parse_rt_infer_config,
)


def _enabled(**overrides) -> dict:
    block = {"enable": True, "bundle_version": "2026.11.05"}
    block.update(overrides)
    return {"rt_infer": block}


def test_absent_block_means_disabled():
    config = parse_rt_infer_config({})
    assert config.enable is False
    assert config.bundle_version == ""
    assert config.pear_periodic is None


def test_disabled_block_still_validates_fields():
    with pytest.raises(RtInferConfigError, match="unsupported fields"):
        parse_rt_infer_config({"rt_infer": {"enable": False, "bogus": 1}})


def test_enable_requires_bundle_version():
    with pytest.raises(RtInferConfigError, match="bundle_version"):
        parse_rt_infer_config({"rt_infer": {"enable": True}})


def test_prefix_is_canonical_and_derived():
    config = parse_rt_infer_config(_enabled())
    assert config.prefix == "/opt/rt-infer/2026.11.05"
    assert config.control_tool == "/opt/rt-infer/2026.11.05/scripts/priorityctl.py"


def test_parse_is_pure_on_any_host(monkeypatch):
    """Config parsing never touches the host (review #6): x86 parses fine."""
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    config = parse_rt_infer_config(_enabled())
    assert config.enable is True


def test_partial_role_override_keeps_every_role():
    """A partial override replaces only the named roles (review #6).

    The old implementation iterated only the override's keys, leaving the
    other roles missing and crashing later with KeyError.
    """
    config = parse_rt_infer_config(_enabled(role_priorities={"yolox": 1}))
    assert config.priority_for("pear") == 0
    assert config.priority_for("yolox") == 1
    assert config.priority_for("tts") == 3
    assert config.priority_for("vl") == 3
    config = parse_rt_infer_config(_enabled(role_cpus={"pear": 9}))
    assert config.cpu_for("yolox") == 10
    assert config.cpu_for("pear") == 9


def test_role_defaults_match_candidate10_profile():
    config = parse_rt_infer_config(_enabled())
    assert config.priority_for("pear") == 0
    assert config.priority_for("yolox") == 3
    assert config.priority_for("tts") == 3
    assert config.priority_for("vl") == 3
    assert config.cpu_for("pear") == 13
    assert config.cpu_for("yolox") == 10
    assert config.cpu_for("tts") == 12
    assert config.cpu_for("vl") == 11
    assert config.scheduler_cpu == 15


def test_unknown_role_rejected():
    config = parse_rt_infer_config(_enabled())
    with pytest.raises(RtInferConfigError, match="supported roles"):
        config.priority_for("chassis")
    with pytest.raises(RtInferConfigError, match="unknown roles"):
        parse_rt_infer_config(_enabled(role_priorities={"chassis": 1}))


def test_role_cpu_rejects_scheduler_alias():
    """The scheduler CPU has its own field; it must not appear twice."""
    with pytest.raises(RtInferConfigError, match="unknown roles"):
        parse_rt_infer_config(_enabled(role_cpus={"scheduler": 3}))


def test_priority_out_of_scale_rejected():
    with pytest.raises(RtInferConfigError, match="0-7"):
        parse_rt_infer_config(_enabled(role_priorities={"pear": 9}))


def test_periodic_block_validation():
    with pytest.raises(RtInferConfigError, match="period_hz"):
        parse_rt_infer_config(_enabled(pear_periodic={"period_hz": 0, "budget_ns": 1}))
    with pytest.raises(RtInferConfigError, match="unsupported fields"):
        parse_rt_infer_config(_enabled(pear_periodic={"period_hz": 18, "bogus": 1}))
    with pytest.raises(RtInferConfigError, match="startup_grace_sec"):
        parse_rt_infer_config(_enabled(pear_periodic={"startup_grace_sec": 0}))


def test_periodic_defaults_are_candidate10_values():
    config = parse_rt_infer_config(_enabled(pear_periodic={}))
    assert config.pear_periodic == PearPeriodicConfig(period_hz=18, budget_ns=38_000_000)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, 0, "5", True])
def test_timeouts_reject_non_positive_and_non_finite(bad):
    with pytest.raises(RtInferConfigError, match="readiness_timeout_sec"):
        parse_rt_infer_config(_enabled(readiness_timeout_sec=bad))


def test_nan_is_not_smuggled_through_int_fields():
    # int fields are strictly typed: floats (including NaN) are rejected.
    with pytest.raises(RtInferConfigError):
        parse_rt_infer_config(_enabled(scheduler_cpu=float("nan")))
    with pytest.raises(RtInferConfigError):
        parse_rt_infer_config(_enabled(role_priorities={"pear": float("inf")}))
