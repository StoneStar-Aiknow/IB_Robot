"""Independent full-path switches preserve endpoint/entry/state gates."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.feedback import Feedback
from aimdk_grasp.hand_restore import RestoreError
from aimdk_grasp.joints import HEAD, LEFT_ARM, RIGHT_ARM
from aimdk_grasp.path_validation import ValidationGate

from robot_runtime.pose_backend import Plan, Report, Trajectory


@pytest.mark.parametrize("hand,body", [("online", "online"), ("off", "online"), ("online", "off"), ("off", "off")])
def test_independent_full_path_switches_keep_entry_endpoints_and_guards(hand, body):
    calls = []
    entry_radii = []
    joints = LEFT_ARM + RIGHT_ARM
    observed = SimpleNamespace(
        feedback=Feedback(1, dict.fromkeys(joints + HEAD, 0), np.zeros(10), right_hand=np.zeros(10))
    )
    plan = Plan(Trajectory([0, 1, 2], [np.zeros(14), np.full(14, 0.1), np.full(14, 0.2)]), True, [], Report(True))

    def check(part, rows, guard, **options):
        calls.append((part, len(rows)))
        if part == "hand" and len(rows) == 2:
            entry_radii.append(options.get("finger_radius"))
        guard()

    backend = SimpleNamespace(
        ARM_JOINTS=joints,
        prepare_execution=lambda plan, **_: plan,
        check_pose=lambda _, **_kw: calls.append(("endpoint", 1)) or Report(True),
        body_geometry=SimpleNamespace(check_path=lambda _j, r, **kw: check("body", r, **kw)),
    )
    restorer = SimpleNamespace(
        ports=SimpleNamespace(prepare=lambda _: calls.append(("prepare", 1))),
        arm_baseline=lambda _: observed,
        guard_arm_motion=lambda *_: calls.append(("guard", 1)) or observed,
        geometry=SimpleNamespace(check_arm_path=lambda _f, _j, r, **kw: check("hand", r, **kw)),
    )
    gate = ValidationGate(backend, {"hand": hand, "body": body})
    assert gate.check(restorer, object(), plan, 0.005) is observed
    for part, mode in (("hand", hand), ("body", body)):
        assert ((part, 3) in calls) is (mode == "online")
        assert (part, 2) in calls
        assert gate.report["stages"][f"{part}_path"]["status"] == ("executed" if mode == "online" else "skipped")
    assert calls.count(("endpoint", 1)) == 2
    assert ("guard", 1) in calls
    assert entry_radii == ([0.0] if hand == "off" else [None])


@pytest.mark.parametrize("part", ["hand", "body"])
def test_middle_collision_is_rejected_online_and_explicitly_unverified_off(part):
    joints = LEFT_ARM + RIGHT_ARM
    observed = SimpleNamespace(
        feedback=Feedback(1, dict.fromkeys(joints + HEAD, 0), np.zeros(10), right_hand=np.zeros(10))
    )
    plan = Plan(Trajectory([0, 1, 2], np.zeros((3, 14))), True, [], Report(True))

    def collision(rows):
        if len(rows) == 3:
            raise RestoreError("HAND_SWEEP_COLLISION", "mock midpoint collision")

    backend = SimpleNamespace(
        ARM_JOINTS=joints,
        prepare_execution=lambda p, **_: p,
        check_pose=lambda _, **_kw: Report(True),
        body_geometry=SimpleNamespace(check_path=lambda _j, rows, **_: collision(rows) if part == "body" else None),
    )
    restorer = SimpleNamespace(
        ports=SimpleNamespace(prepare=lambda _: None),
        arm_baseline=lambda _: observed,
        guard_arm_motion=lambda *_: observed,
        geometry=SimpleNamespace(check_arm_path=lambda _f, _j, rows, **_: collision(rows) if part == "hand" else None),
    )
    gate = ValidationGate(backend)
    with pytest.raises(RestoreError, match="midpoint"):
        gate.check(restorer, object(), plan, 0.005)
    assert gate.report["stages"][f"{part}_path"]["status"] == "failed"
    gate = ValidationGate(backend, {part: "off"})
    gate.check(restorer, object(), plan, 0.005)
    assert "not verified" in gate.report["stages"][f"{part}_path"]["reason"]


@pytest.mark.parametrize("fault", ["stale", "owner", "cancel", "timeout", "limits", "speed", "endpoint"])
def test_off_never_bypasses_mandatory_refusals(fault):
    joints = LEFT_ARM + RIGHT_ARM
    plan = Plan(Trajectory([0, 1], np.zeros((2, 14))), True, [], Report(True))

    def observe(_):
        raise RestoreError("MOCK_REFUSAL", fault)

    backend = SimpleNamespace(
        ARM_JOINTS=joints,
        prepare_execution=lambda p, **_: Plan(p.traj, fault not in ("limits", "speed"), [], Report(True)),
        check_pose=lambda _, **_kw: Report(fault != "endpoint"),
    )
    restorer = SimpleNamespace(ports=SimpleNamespace(prepare=lambda _: None), arm_baseline=observe)
    if fault == "endpoint":
        restorer.arm_baseline = lambda _: SimpleNamespace(feedback=SimpleNamespace(positions=dict.fromkeys(HEAD, 0)))
    gate = ValidationGate(backend, {"hand": "off", "body": "off"})
    with pytest.raises(RestoreError):
        gate.check(restorer, object(), plan, 0.005)
