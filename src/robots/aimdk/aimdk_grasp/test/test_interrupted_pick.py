"""X2 seven-joint execution, stop in place, then separately admitted retreat."""

from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.arm_gravity import ArmGravity
from aimdk_grasp.arm_stream import x2_arm_executor
from aimdk_grasp.interrupted_pick import InterruptedPick
from aimdk_grasp.joints import HEAD, LEFT_ARM
from test_arm_stream import APPROACH, GRASP, START, SaggingArm

from manipulation_execution.grasp_pipeline.contracts import (
    ArmSegment,
    Effect,
    EffectorAction,
    EffectorOutcome,
    GraspBackends,
    GraspFailure,
    GraspPlan,
    HeldObject,
    Move,
    SegmentKind,
)
from manipulation_execution.grasp_pipeline.sequence import GraspSequence


class Hand:
    def __init__(self):
        self.actions = []

    def released(self):
        return True

    def act(self, *args, **kwargs):
        self.actions.append(args)
        raise AssertionError("recovery must never command a hand")


def fixture():
    arm = SaggingArm(ArmGravity.load(), START)
    executor = x2_arm_executor(arm, gravity=arm.gravity)
    hand = Hand()
    validations = []

    def posture():
        return executor.measured((*arm.channel_joints, *HEAD))

    def validate(steps, nonce, **budget):
        validations.append((steps, nonce))

    recovery = InterruptedPick(executor, hand, posture, validate, clock=arm.monotonic)
    segment = ArmSegment(
        "approach", SegmentKind.TRAJECTORY, LEFT_ARM, np.linspace(START, APPROACH, 40), np.linspace(0.0, 2.5, 40)
    )
    plan = GraspPlan((Move(segment), Effect(EffectorAction.CLOSE)), candidate_index=0, report={"arm_side": "left"})
    backends = GraspBackends(None, None, None, executor, hand, interrupted_pick=recovery)
    return arm, executor, hand, recovery, validations, plan, backends


def interrupt():
    arm, executor, hand, recovery, validations, plan, backends = fixture()
    end = arm.monotonic() + 0.8
    result = GraspSequence(backends).run(plan, deadline=1000, cancelled=lambda: arm.monotonic() >= end)
    executor.finish()
    assert result.error_code == "CANCELED"
    assert not result.at_rest
    assert recovery.status.state == "pending"
    assert hand.actions == [] and validations == []
    assert not np.allclose(arm.measured[:7], START, atol=0.025)
    return arm, executor, hand, recovery, validations, plan


def test_x2_stop_sends_no_retreat_or_release_and_next_recovery_reaches_start():
    arm, executor, hand, recovery, validations, plan = interrupt()
    assert [row["name"] for row in executor.report["segments"]] == ["approach"]
    recovery.recover(recovery.status.nonce, deadline=arm.monotonic() + 20, cancelled=lambda: False)
    assert recovery.status is None
    assert len(validations) == 1
    steps, nonce = validations[0]
    assert steps and nonce
    # Only the executed prefix, not the unexecuted approach target, was reversed.
    assert not np.allclose(steps[0].segment.start, APPROACH, atol=0.05)
    np.testing.assert_allclose(arm.measured[:7], START, atol=0.025)
    assert hand.actions == []


@pytest.mark.parametrize("problem", ["arm_moved", "obstacle", "unavailable", "nonce", "cancel"])
def test_x2_failed_recovery_never_releases_or_starts_a_new_pick(problem):
    arm, executor, hand, recovery, validations, plan = interrupt()
    nonce = recovery.status.nonce
    if problem == "arm_moved":
        arm.measured[0] += 0.1
    elif problem in {"obstacle", "unavailable"}:

        def reject(*args, **kwargs):
            raise GraspFailure("PICK_MANUAL_CHECK", problem)

        recovery.validate = reject
    elif problem == "nonce":
        nonce = "stale"
    before = len(executor.report["segments"])
    with pytest.raises(GraspFailure):
        recovery.recover(nonce, deadline=arm.monotonic() + 20, cancelled=lambda: problem == "cancel")
    assert len(executor.report["segments"]) == before
    assert hand.actions == [] and recovery.status is not None
    with pytest.raises(GraspFailure):
        recovery.begin(plan)


@pytest.mark.parametrize("action", [EffectorAction.CLOSE, EffectorAction.RELEASE])
def test_x2_any_started_close_or_release_requires_manual_check(action):
    arm, executor, hand, recovery, validations, plan, backends = fixture()
    recovery.begin(plan)
    recovery.effect_started(action)
    recovery.end(SimpleNamespace(succeeded=False, at_rest=False, error_code="CANCELED"))
    assert recovery.status.state == "manual"
    with pytest.raises(GraspFailure):
        recovery.recover(recovery.status.nonce, deadline=1000, cancelled=lambda: False)
    assert validations == [] and hand.actions == []


def test_x2_tracking_fault_is_not_automatic_recovery():
    arm, executor, hand, recovery, validations, plan, backends = fixture()
    recovery.begin(plan)
    recovery.end(SimpleNamespace(succeeded=False, at_rest=False, error_code="MOTION_FAILED"))
    assert recovery.status.state == "manual"
    assert validations == []


def carrying_interruption():
    arm, executor, _, recovery, validations, _, backends = fixture()
    events = []

    class GrippingHand:
        closed = False
        grip = ""

        def released(self):
            return not self.closed

        def holding(self):
            return self.grip if self.closed else None

        def snapshot(self):
            return "verified grip"

        def restore(self, snapshot, *, send=True):
            assert snapshot == "verified grip" and not send
            events.append("observe grip")

        def act(self, action, **kwargs):
            if kwargs["cancelled"]():
                return EffectorOutcome(False)
            events.append((action, arm.measured[:7].copy()))
            self.closed = action is EffectorAction.CLOSE
            if self.closed:
                self.closed_at = arm.monotonic()
            return EffectorOutcome(True, held=self.closed)

        def finish(self):
            pass

    hand = GrippingHand()
    recovery.effector = hand

    def segment(name, start, finish):
        return Move(
            ArmSegment(name, SegmentKind.TRAJECTORY, LEFT_ARM, np.linspace(start, finish, 40), np.linspace(0, 2.5, 40))
        )

    lift = GRASP + np.array([0.1, 0.05, 0, -0.1, 0, 0, 0])
    plan = GraspPlan(
        (
            segment("approach", START, APPROACH),
            segment("grasp", APPROACH, GRASP),
            Effect(EffectorAction.CLOSE),
            segment("lift", GRASP, lift),
        ),
        candidate_index=0,
        report={"arm_side": "left", "x2_grasp_head_plan_id": "source-plan"},
    )
    from dataclasses import replace

    result = GraspSequence(replace(backends, effector=hand)).run(
        plan, deadline=1000, cancelled=lambda: hand.closed and arm.monotonic() > hand.closed_at + 0.8
    )
    executor.finish()
    assert result.error_code == "CANCELED" and recovery.status.state == "pending"
    assert len(events) == 1 and events[0][0] is EffectorAction.CLOSE
    recovery.held_factory = lambda moves: HeldObject(
        "interruption-geometry",
        "apple",
        "left",
        LEFT_ARM,
        tuple(arm.measured[:7]),
        tuple(START),
        0.04,
        (0.0, 0.0, 0.0),
        "verified grip",
    )
    return arm, executor, hand, recovery, events


def test_verified_x2_grasp_retraces_lift_releases_at_contact_then_rechecks_empty_return():
    arm, executor, hand, recovery, events = carrying_interruption()
    validations = []

    def validate(steps, nonce, **kwargs):
        validations.append((steps, kwargs.get("held"), hand.closed))
        events.append("validate carrying" if kwargs.get("held") else "validate deposited object")

    recovery.validate = validate
    recovery.recover(recovery.status.nonce, deadline=arm.monotonic() + 40, cancelled=lambda: False)
    assert recovery.status is None
    assert validations[0][0][-1] == Effect(EffectorAction.RELEASE)
    assert validations[0][1] is not None and validations[0][2]
    assert validations[1][0] == (Effect(EffectorAction.RELEASE),) and validations[1][2]
    assert validations[2][1] is None and not validations[2][2]
    release = next(event for event in events if isinstance(event, tuple) and event[0] is EffectorAction.RELEASE)
    np.testing.assert_allclose(release[1], GRASP, atol=0.02)
    np.testing.assert_allclose(arm.measured[:7], START, atol=0.02)
    assert events.index("validate carrying") < events.index(release)
    assert events.index(release) < events.index("validate deposited object")


@pytest.mark.parametrize("problem", ["grip", "geometry", "obstacle", "support", "cancel_during_retreat"])
def test_carrying_recovery_refusal_keeps_hand_closed_and_requires_manual_check(problem):
    arm, executor, hand, recovery, events = carrying_interruption()
    before = len(executor.report["segments"])
    if problem == "grip":
        hand.grip = None
    elif problem == "geometry":
        recovery.held_factory = None
    elif problem in {"obstacle", "support"}:

        def reject(*args, **kwargs):
            raise GraspFailure("PICK_MANUAL_CHECK", problem)

        recovery.validate = reject
    cancellation_at = arm.monotonic() + 0.2 if problem == "cancel_during_retreat" else 1000
    with pytest.raises(GraspFailure):
        recovery.recover(
            recovery.status.nonce, deadline=arm.monotonic() + 40, cancelled=lambda: arm.monotonic() >= cancellation_at
        )
    assert hand.closed and recovery.status.state == "manual"
    assert not any(isinstance(event, tuple) and event[0] is EffectorAction.RELEASE for event in events)
    if problem != "cancel_during_retreat":
        assert len(executor.report["segments"]) == before


def test_after_release_blocked_empty_retreat_does_not_start_a_new_pick():
    arm, executor, hand, recovery, events = carrying_interruption()

    def validate(steps, nonce, **kwargs):
        if kwargs.get("held") is None:
            raise GraspFailure("PICK_MANUAL_CHECK", "deposited object blocks return")

    recovery.validate = validate
    with pytest.raises(GraspFailure):
        recovery.recover(recovery.status.nonce, deadline=arm.monotonic() + 40, cancelled=lambda: False)
    assert not hand.closed and recovery.status.state == "manual"
    np.testing.assert_allclose(arm.measured[:7], GRASP, atol=0.02)


def test_restart_never_replays_a_lost_in_memory_journal_and_manual_reset_is_read_only(tmp_path):
    arm, executor, hand, recovery, validations, plan, backends = fixture()
    state_path = tmp_path / "interruption.json"
    saved = InterruptedPick(
        executor, hand, recovery.posture, recovery.validate, clock=arm.monotonic, state_path=state_path
    )
    saved.begin(plan)
    restarted = InterruptedPick(
        executor, hand, recovery.posture, recovery.validate, clock=arm.monotonic, state_path=state_path
    )
    assert restarted.status.state == "manual"
    with pytest.raises(GraspFailure):
        restarted.recover(restarted.status.nonce, deadline=1000, cancelled=lambda: False)
    with pytest.raises(GraspFailure):
        restarted.confirm_manual_reset(restarted.status.nonce, confirmed_empty=True, note="")
    restarted.confirm_manual_reset(restarted.status.nonce, confirmed_empty=True, note="inspected at original home")
    assert restarted.status is None and hand.actions == []
    assert InterruptedPick(executor, hand, recovery.posture, recovery.validate, state_path=state_path).status is None


def test_restarted_automatic_arm_selection_can_be_inspected_without_commands(tmp_path):
    from aimdk_grasp.dual_arm import ExecutionSelection, SelectedPart

    arm, executor, hand, recovery, validations, plan, backends = fixture()
    state_path = tmp_path / "interruption.json"
    saved = InterruptedPick(executor, hand, recovery.posture, recovery.validate, state_path=state_path)
    saved.begin(plan)
    selection = ExecutionSelection()
    selected_arm = SelectedPart({"left": executor}, selection)
    selected_hand = SelectedPart({"left": hand}, selection)
    restarted = InterruptedPick(selected_arm, selected_hand, recovery.posture, recovery.validate, state_path=state_path)
    assert selection.side == "left" and restarted.status.state == "manual"
    restarted.confirm_manual_reset(restarted.status.nonce, confirmed_empty=True, note="inspected after restart")
    assert restarted.status is None and executor.report["segments"] == [] and hand.actions == []
