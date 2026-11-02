"""Unit tests for the conformance required-set mechanism (lekiwi-nav-grasp D6).

The env-parsed sets are module state of ``test_conformance``; these tests
exercise the pure decision helpers against injected sets, including the
"capability present, required interface missing" case.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import test_conformance as conformance


def _with_sets(monkeypatch, capabilities="", interfaces=""):
    monkeypatch.setattr(conformance, "REQUIRED_CAPABILITIES", conformance._parse_required_set(capabilities))
    monkeypatch.setattr(conformance, "REQUIRED_INTERFACES", conformance._parse_required_set(interfaces))


def test_parse_required_set_ignores_empty_entries(monkeypatch):
    _with_sets(monkeypatch, " joint.state , , motion.fk ,", " camera.wrist.color ")
    assert conformance.REQUIRED_CAPABILITIES == ["joint.state", "motion.fk"]
    assert conformance.REQUIRED_INTERFACES == ["camera.wrist.color"]
    _with_sets(monkeypatch, "", "")
    assert conformance.REQUIRED_CAPABILITIES == [] and conformance.REQUIRED_INTERFACES == []


def test_required_surface_failures_reports_both_namespaces(monkeypatch):
    _with_sets(
        monkeypatch,
        capabilities="joint.state,motion.fk,base.cmd_vel",
        interfaces="camera.wrist.color,lidar.mid360.points",
    )
    failures = conformance.required_surface_failures(
        {"joint.state", "base.cmd_vel"},  # motion.fk missing
        {"camera.wrist.color"},  # lidar.mid360.points missing
    )
    assert failures == {"capabilities": ["motion.fk"], "interfaces": ["lidar.mid360.points"]}
    # nothing required, nothing missing
    _with_sets(monkeypatch)
    assert conformance.required_surface_failures(set(), set()) == {"capabilities": [], "interfaces": []}


def test_capability_present_but_required_interface_missing_still_fails(monkeypatch):
    """The two sets live in different namespaces: a runtime advertising the
    camera capability while the required stream ID is absent must fail."""
    _with_sets(monkeypatch, capabilities="perception.camera", interfaces="camera.wrist.color")
    failures = conformance.required_surface_failures({"perception.camera"}, {"camera.front.color"})
    assert failures == {"capabilities": [], "interfaces": ["camera.wrist.color"]}


def test_requires_skips_optional_but_fails_required_capabilities(monkeypatch):
    _with_sets(monkeypatch, capabilities="joint.state,motion.fk")
    harness = SimpleNamespace(capabilities=lambda: {"joint.state"})
    with pytest.raises(pytest.fail.Exception, match="required capability missing.*motion.fk"):
        conformance.requires(harness, "joint.state", "motion.fk")
    # an undeclared OPTIONAL capability still skips
    with pytest.raises(pytest.skip.Exception, match="does not declare"):
        conformance.requires(harness, "joint.state", "perception.lidar")
    # everything declared: no exception
    conformance.requires(SimpleNamespace(capabilities=lambda: {"joint.state", "motion.fk"}), "joint.state", "motion.fk")
