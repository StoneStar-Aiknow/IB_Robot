"""Certificate matching/fallback and conservative gravity bounds, without hardware."""

import copy
import json
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.compiled_body_geometry import CompiledGeometry
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.fixed_pose_backend import FixedBackend
from aimdk_grasp.hand_restore import RestoreError
from aimdk_grasp.joints import HEAD
from aimdk_grasp.path_certificate import Certificate, LiveConditions, compile_certificate
from aimdk_grasp.path_validation import ValidationGate

from robot_runtime.path_validation import policy_digest


@pytest.fixture
def empty_world():
    """Deliberately empty test world; tests below do not certify X2 hardware."""
    backend = object.__new__(FixedBackend)
    backend.hard_limits = dict.fromkeys(backend.ARM_JOINTS, (-10, 10))
    backend.fingerprint = "empty-test-world"
    backend.library = {
        "digest": "test-library",
        "poses": {"arms_up": {"target": dict.fromkeys(backend.ARM_JOINTS, 0.2)}},
    }
    backend.body_geometry = CompiledGeometry(
        {
            "bodies": [
                {
                    "parent": 0,
                    "transform": np.eye(4).tolist(),
                    "joints": [],
                    "com": [0, 0, 0],
                    "mass": 0,
                }
            ],
            "shapes": [],
            "pairs": [],
        }
    )
    uncertainties = []
    backend.geometry = SimpleNamespace(check_arm_path=lambda *_, **kw: uncertainties.append(kw))
    backend.observation = lambda q: Feedback(10, {**q, **dict.fromkeys(HEAD, 0)}, np.zeros(10), right_hand=np.zeros(10))
    data = compile_certificate(backend, entry_rad=[0.02] * 14, tracking_rad=[0.1] * 14, response_sec=0.25, poll_sec=0.2)
    assert len(uncertainties) == 2
    np.testing.assert_allclose(uncertainties[0]["uncertainty"], 0.395)
    return backend, data


def live_conditions():
    return LiveConditions(time.monotonic(), True, True, 0.5, 0.25)


def route_plan(backend, data):
    route = data["routes"][0]
    plan = backend.plan_for_execution(route["start"], [route["target"]], [0.0])
    return plan, backend.observation(route["start"])


@pytest.mark.parametrize("part", ["hand", "body"])
def test_match_needs_live_assumptions_route_and_entry_separate_from_arrival(empty_world, part):
    backend, data = empty_world
    certificate = Certificate(data, backend)
    plan, observed = route_plan(backend, data)

    def match(f, p=plan):
        return certificate.match(part, f, p, 0.005, live_conditions())

    assert match(observed)[0]
    assert certificate.match(part, observed, plan, 0.005)[0] is None
    altered = copy.deepcopy(observed.positions)
    altered[backend.ARM_JOINTS[0]] += 0.09  # Accepted arrival error is not a certified entry.
    assert match(backend.observation(altered))[0] is None
    altered = copy.deepcopy(plan)
    altered.traj.positions[len(altered.traj.positions) // 2, 1] += 0.001
    assert match(observed, altered)[0] is None
    for side in ("left", "right"):
        assert match(replace(observed, **{f"{side}_hand": None}))[0] is None
    observed.left_hand[0] += 0.006
    assert match(observed)[0] is None


@pytest.mark.parametrize("available", [True, False])
def test_component_reuse_and_online_fallback_are_reported_independently(empty_world, available):
    backend, data = empty_world
    plan, observed = route_plan(backend, data)
    restorer = SimpleNamespace(
        ports=SimpleNamespace(prepare=lambda _: None),
        arm_baseline=lambda _: SimpleNamespace(feedback=observed),
        geometry=backend.geometry,
        guard_arm_motion=lambda *_: SimpleNamespace(feedback=observed),
        validation_conditions=lambda _: live_conditions() if available else None,
    )
    gate = ValidationGate(
        backend,
        {"hand": "prevalidated", "certificate_file": "unit-test-only.json"},
        certificate=Certificate(data, backend),
    )
    gate.check(restorer, object(), plan, 0.005)
    assert gate.report["stages"]["hand_path"]["status"] == ("reused" if available else "executed")
    assert gate.report["stages"]["body_path"]["status"] == "executed"
    assert (gate.monitor is not None) is available
    if not available:
        assert "online fallback" in gate.report["stages"]["hand_path"]["reason"]


@pytest.mark.parametrize("offset,hit", [(0.014, True), (0.016, False), (0.09, False)])
def test_entry_budget_includes_dispatch_tolerance_without_rounding(empty_world, offset, hit):
    backend, data = empty_world
    route = data["routes"][0]
    start = dict(route["start"])
    start[backend.ARM_JOINTS[0]] += offset
    plan = backend.plan_for_execution(start, [route["target"]], [0.0])
    match, _ = Certificate(data, backend).match("hand", backend.observation(start), plan, 0.005, live_conditions())
    assert (match is not None) is hit


@pytest.mark.parametrize("change", ["algorithm", "model", "library", "tube", "file"])
def test_invalid_certificate_is_not_an_online_fallback(empty_world, tmp_path, change):
    backend, data = empty_world
    data = copy.deepcopy(data)
    if change == "tube":
        data["tube_rad"][0] -= 0.001
    elif change != "file":
        data["binding"][
            {"algorithm": "algorithm_digest", "model": "robot_fingerprint", "library": "library_digest"}[change]
        ] = "wrong"
    data["digest"] = policy_digest({k: v for k, v in data.items() if k != "digest"})
    if change == "file":
        path = tmp_path / "certificate.json"
        path.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="changed"):
            Certificate.load({"certificate_file": str(path), "certificate_sha256": "0" * 64}, backend)
    else:
        with pytest.raises(ValueError):
            Certificate(data, backend)


@pytest.mark.parametrize("fault", ["tracking", "head", "stale", "poll", "assumptions"])
def test_monitor_requires_cancel_on_range_or_state_loss(empty_world, monkeypatch, fault):
    from aimdk_grasp import path_certificate

    backend, data = empty_world
    certificate = Certificate(data, backend)
    plan, observed = route_plan(backend, data)
    monitor = certificate.monitor(plan, {"hand": data["routes"][0]})
    clock = [10.0]
    monkeypatch.setattr(path_certificate.time, "monotonic", lambda: clock[0])
    monitor.check(observed, 10.0, live_conditions(), feedback_age_s=0.0)
    if fault == "tracking":
        observed.positions[backend.ARM_JOINTS[0]] += 0.101
    elif fault == "head":
        observed.positions[HEAD[0]] += 0.011
    elif fault == "stale":
        clock[0] += 0.101
    elif fault == "poll":
        clock[0] += 0.201
    with pytest.raises(RestoreError, match="certificate"):
        monitor.check(
            observed,
            10.0,
            None if fault == "assumptions" else live_conditions(),
            feedback_age_s=0.101 if fault == "stale" else 0.0,
        )


def test_no_proof_is_exported_when_error_tube_exceeds_limits(empty_world):
    backend, _data = empty_world
    data = compile_certificate(backend, entry_rad=[20] * 14, tracking_rad=[0.1] * 14, response_sec=0.25)
    assert all(not r["components"] and set(r["refused"]) == {"hand", "body"} for r in data["routes"])
    Certificate(data, backend)  # Valid refusal record, never a positive proof.


def test_monitor_uses_transport_age_without_mixing_ros_and_monotonic_clocks(empty_world):
    backend, data = empty_world
    plan, observed = route_plan(backend, data)
    observed = replace(observed, received_at=1_700_000_000.0)
    monitor = Certificate(data, backend).monitor(plan, {"hand": data["routes"][0]})
    monitor.check(observed, None, live_conditions(), feedback_age_s=0.01)


def test_gravity_enclosure_covers_random_joint_box_and_refuses_unsupported_pivots():
    bodies = [{"parent": 0, "transform": np.eye(4).tolist(), "joints": [], "com": [0, 0, 0], "mass": 0}]
    for index in range(2):
        transform = np.eye(4)
        transform[0, 3] = 0.3
        bodies.append(
            {
                "parent": index,
                "transform": transform.tolist(),
                "com": [0.2, 0, 0],
                "mass": 1,
                "joints": [{"name": f"j{index}", "position": [0, 0, 0], "axis": [0, 1, 0]}],
            }
        )
    model = CompiledGeometry({"bodies": bodies, "shapes": [], "pairs": []})
    center = {"j0": 0.6, "j1": -0.9}
    delta = {"j0": 0.1, "j1": 0.08}
    bound = model.gravity_error(delta)
    nominal = model.forward(center)[1]
    for row in np.random.default_rng(17).uniform(-1, 1, (200, 2)):
        q = {n: v + row[i] * delta[n] for i, (n, v) in enumerate(center.items())}
        torque = model.forward(q)[1]
        assert all(abs(torque[n] - nominal[n]) <= bound[n] for n in center)
    bodies[1]["joints"][0]["position"][0] = 0.1
    with pytest.raises(ValueError, match="zero-pivot"):
        model.gravity_error(delta)
