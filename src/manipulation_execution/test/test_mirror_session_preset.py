"""Named ownership must drain both public streams and release before resuming."""

from dataclasses import replace

import pytest

from manipulation_execution.mirror_session.hooks import UpperBodyCommand
from manipulation_execution.mirror_session.preset import (
    PresetConfig,
    PresetCoordinator,
    PresetEpoch,
    PresetPhase,
    PresetSnapshot,
)


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def command(value=0.1):
    return UpperBodyCommand((value,) * 14, (value + 0.2,) * 20)


class Port:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.state = PresetSnapshot()
        self.fail = ""

    def begin(self, name, epoch):
        self.calls.append(("begin", name, epoch))
        if self.fail == "begin":
            raise RuntimeError("dispatch failed after possible delivery")
        self.state = PresetSnapshot(epoch=epoch, stage="pending", pending=True, observed_at=self.clock())

    def refresh(self, exit_target=None):
        self.calls.append(("refresh", exit_target))
        if self.fail == "refresh":
            raise RuntimeError("lease publication failed")

    def cancel(self, reason):
        self.calls.append(("cancel", reason))
        if self.fail == "cancel":
            raise RuntimeError("cancel publication failed")

    def stop_to(self, target):
        self.calls.append(("stop_to", target))
        if self.fail == "stop_to":
            raise RuntimeError("stop target publication failed")

    def snapshot(self):
        if self.fail == "snapshot":
            raise RuntimeError("adapter state failed")
        return self.state

    def observe(self, **changes):
        self.state = replace(self.state, observed_at=self.clock(), **changes)

    def accept(self, stage="entering", goal_id="goal-1"):
        self.observe(pending=False, accepted=True, stage=stage, goal_id=goal_id)

    def terminal(self, **changes):
        self.observe(pending=False, terminal=True, **changes)

    def count(self, name):
        return sum(call[0] == name for call in self.calls)


@pytest.fixture
def rig():
    clock = Clock()
    port = Port(clock)
    coordinator = PresetCoordinator(PresetConfig(enabled=True), port, clock)
    return coordinator, port, clock


def enter(rig, *, accepted=True, holding=True):
    coordinator, port, clock = rig
    assert coordinator.request(7)
    assert coordinator.step(7, True).block_public
    clock.advance(0.6)
    assert coordinator.step(7, True).block_public
    assert port.count("begin") == 1
    if accepted:
        port.accept("holding" if holding else "entering")
        coordinator.step(7, True)
    return rig


def exit_normally(rig):
    coordinator, port, clock = rig
    target = command(0.2)
    result = coordinator.step(7, False, target, True)
    assert result.phase == PresetPhase.EXITING
    assert result.block_public and result.resume_seed is None
    assert port.calls[-1] == ("refresh", target)
    return target


def test_normal_entry_hold_and_same_owner_exit_return_one_fresh_seed(rig):
    coordinator, port, clock = enter(rig)
    assert coordinator.phase == PresetPhase.HOLDING
    for _ in range(20):
        clock.advance(0.1)
        port.observe()
        result = coordinator.step(7, True)
        assert result.block_public and result.resume_seed is None
        assert not coordinator.request(7)
    assert port.count("begin") == 1
    assert port.count("refresh") > 10
    target = exit_normally(rig)
    clock.advance(0.1)
    port.terminal(
        success=True,
        release_confirmed=True,
        final_target=target,
        final_target_at=clock(),
        final_target_verified=True,
    )
    result = coordinator.step(7, False)
    assert not result.block_public and result.resume_seed == target
    assert result.release_confirmed and not result.fault
    assert result.phase == PresetPhase.IDLE
    assert coordinator.step(7, False).resume_seed is None
    assert port.count("cancel") == 0


def test_drain_requires_continuously_quiet_arm_and_hand_streams(rig):
    coordinator, port, clock = rig
    coordinator.request(7)
    coordinator.step(7, True)
    clock.advance(0.59)
    coordinator.step(7, True)
    assert port.count("begin") == 0
    coordinator.step(7, True, public_quiet=False)  # either public channel was active
    clock.advance(10.0)
    coordinator.step(7, True)
    assert port.count("begin") == 0
    clock.advance(0.6)
    coordinator.step(7, True)
    assert port.count("begin") == 1


def test_drain_clock_starts_when_both_public_streams_are_observed_quiet(rig):
    coordinator, port, clock = rig
    coordinator.request(7)
    clock.advance(1.0)
    coordinator.step(7, True)
    assert port.count("begin") == 0
    clock.advance(0.6)
    coordinator.step(7, True)
    assert port.count("begin") == 1


def test_human_exiting_during_drain_does_not_dispatch(rig):
    coordinator, port, clock = rig
    coordinator.request(7)
    coordinator.step(7, True)
    clock.advance(0.7)
    result = coordinator.step(7, False)
    assert result.phase == PresetPhase.IDLE
    assert not result.block_public and result.resume_seed is None
    assert port.calls == []


@pytest.mark.parametrize("stage", ["draining", "pending", "entering", "holding", "exiting"])
def test_stop_wins_in_every_stage_and_never_resumes(rig, stage):
    coordinator, port, clock = rig
    if stage == "draining":
        coordinator.request(7)
        coordinator.step(7, True)
    else:
        enter(rig, accepted=stage != "pending", holding=stage in {"holding", "exiting"})
        if stage == "exiting":
            exit_normally(rig)
    refreshes = port.count("refresh")
    result = coordinator.stop("long_cover")
    assert result.resume_seed is None
    clock.advance(0.2)
    if stage != "draining":
        port.observe()
    result = coordinator.step(7, True)
    assert port.count("refresh") == refreshes
    assert port.count("begin") == (0 if stage == "draining" else 1)
    assert port.count("cancel") == (0 if stage == "draining" else 1)
    assert result.block_public == (stage != "draining")
    assert not coordinator.request(7)
    if stage != "draining":
        port.terminal(success=True, release_confirmed=True, final_target=command(), final_target_verified=True)
        result = coordinator.step(7, True)
        assert result.release_confirmed and not result.block_public
        assert result.resume_seed is None
    assert coordinator.phase == PresetPhase.STOPPED


@pytest.mark.parametrize("stage", ["pending", "entering", "holding", "exiting"])
def test_checked_stop_in_each_dispatched_stage_renews_and_returns_only_one_stop_seed(rig, stage):
    coordinator, port, clock = enter(rig, accepted=stage != "pending", holding=stage in {"holding", "exiting"})
    if stage == "exiting":
        exit_normally(rig)
    target = command(0.0)
    result = coordinator.stop("mirror_stop", target)
    assert result.phase == PresetPhase.STOPPING and result.block_public
    assert result.resume_seed is None and result.stop_seed is None
    assert ("stop_to", target) in port.calls and port.count("cancel") == 0
    if stage == "pending":
        port.accept()
    refreshes = port.count("refresh")
    clock.advance(0.11)
    port.observe(stage="stopping")
    result = coordinator.step(7, False, command(0.9), True)
    assert result.phase == PresetPhase.STOPPING
    assert port.count("refresh") == refreshes + 1 and port.calls[-1] == ("refresh", None)
    port.terminal(
        success=True,
        release_confirmed=True,
        stop_target=target,
        stop_target_at=clock(),
        stop_target_verified=True,
    )
    result = coordinator.step(7, False)
    assert result.release_confirmed and not result.block_public and not result.fault
    assert result.phase == PresetPhase.STOPPED and result.stop_seed == target
    assert result.resume_seed is None and port.count("begin") == 1 and port.count("cancel") == 0
    result = coordinator.step(7, False)
    assert result.stop_seed is None and result.resume_seed is None


def test_checked_stop_during_drain_never_dispatches_or_creates_a_seed(rig):
    coordinator, port, clock = rig
    coordinator.request(7)
    coordinator.step(7, True)
    result = coordinator.stop("stop", command())
    assert result.phase == PresetPhase.STOPPED and result.release_confirmed
    assert result.stop_seed is None and result.resume_seed is None and port.calls == []


def test_stop_closed_before_action_dispatch_returns_only_the_untouched_baseline(rig):
    coordinator, port, clock = enter(rig, accepted=False)
    coordinator.stop("stop", command())
    port.terminal(stage="stopped_before_dispatch", accepted=False, success=False, release_confirmed=True)
    result = coordinator.step(7, False)
    assert result.phase == PresetPhase.IDLE and result.release_confirmed and not result.block_public
    assert result.stop_seed is None and result.resume_seed is None and not result.fault


@pytest.mark.parametrize(
    "changes",
    [
        {"stage": "cancelled"},
        {"accepted": True, "goal_id": "goal-1"},
        {"pending": True},
        {"success": True},
        {"release_confirmed": False},
    ],
)
def test_only_explicit_complete_undispatched_stop_proof_can_restore_the_baseline(rig, changes):
    coordinator, port, clock = enter(rig, accepted=False)
    coordinator.stop("stop", command())
    evidence = dict(
        stage="stopped_before_dispatch",
        terminal=True,
        accepted=False,
        pending=False,
        success=False,
        release_confirmed=True,
    )
    port.observe(**(evidence | changes))
    result = coordinator.step(7, False)
    assert result.fault and result.block_public and result.phase == PresetPhase.FAULT
    assert result.stop_seed is None and result.resume_seed is None


def test_pending_uuid_is_bound_before_acceptance_and_cannot_be_replaced_by_stop_result(rig):
    coordinator, port, clock = enter(rig, accepted=False)
    port.observe(goal_id="submitted-owner")
    coordinator.stop("stop", command())
    port.terminal(
        accepted=True,
        goal_id="other-owner",
        success=True,
        release_confirmed=True,
        stop_target=command(),
        stop_target_at=clock(),
        stop_target_verified=True,
    )
    result = coordinator.step(7, False)
    assert result.fault == "preset_goal_identity_changed" and result.block_public
    assert result.stop_seed is None and result.resume_seed is None


def test_an_already_accepted_owner_cannot_later_claim_it_was_never_dispatched(rig):
    coordinator, port, clock = enter(rig)
    coordinator.stop("stop", command())
    port.terminal(stage="stopped_before_dispatch", accepted=False, success=False, release_confirmed=True)
    result = coordinator.step(7, False)
    assert result.fault == "preset_stop_evidence_invalid" and result.block_public
    assert result.stop_seed is None and result.resume_seed is None


def test_known_fault_before_stop_is_cancelled_without_attempting_the_reset_bridge(rig):
    coordinator, port, clock = enter(rig)
    port.observe(fault="runtime_lost")
    result = coordinator.stop("stop", command())
    assert result.fault == "runtime_lost" and result.block_public
    assert port.count("stop_to") == 0 and port.count("cancel") == 1


@pytest.mark.parametrize(
    "change",
    [
        {"success": False},
        {"release_confirmed": False},
        {"stop_target_verified": False},
        {"stop_target": None},
        {"stop_target": command(0.9)},
        {"stop_target_at": -1.0},
        {"stop_target_at": 10.0},
        {"stop_target_at": float("nan")},
        {"accepted": False},
        {"pending": True},
        {"goal_id": "other-owner"},
    ],
)
def test_stop_requires_fresh_exact_verified_reset_evidence(rig, change):
    coordinator, port, clock = enter(rig)
    target = command(0.0)
    coordinator.stop("stop", target)
    evidence = dict(
        success=True,
        release_confirmed=True,
        stop_target=target,
        stop_target_at=clock(),
        stop_target_verified=True,
    )
    port.observe(**(dict(terminal=True, pending=False) | evidence | change))
    result = coordinator.step(7, False)
    assert result.fault and result.block_public
    assert result.stop_seed is None and result.resume_seed is None


@pytest.mark.parametrize("target", [None, UpperBodyCommand((0.0,) * 13, (0.0,) * 20), command(float("inf"))])
def test_unusable_stop_target_retains_immediate_cancel(rig, target):
    coordinator, port, clock = enter(rig)
    result = coordinator.stop("stop", target)
    assert result.phase == PresetPhase.CANCELLING and result.block_public
    assert port.calls[-1] == ("cancel", "stop") and port.count("stop_to") == 0


@pytest.mark.parametrize("failure", ["stop_to", "refresh", "snapshot"])
def test_stop_port_failures_never_create_a_seed_or_revive_the_lease(rig, failure):
    coordinator, port, clock = enter(rig)
    port.fail = failure
    coordinator.stop("stop", command())
    result = coordinator.step(7, False)
    assert result.fault.startswith(f"preset_port_{failure}") and result.block_public
    assert result.stop_seed is None and result.resume_seed is None


@pytest.mark.parametrize("failure", ["stale", "lease", "action_cancel"])
def test_stopping_lost_evidence_or_action_cancellation_cannot_seed(rig, failure):
    coordinator, port, clock = enter(rig)
    coordinator.stop("stop", command())
    if failure == "stale":
        clock.advance(0.6)
        coordinator.step(7, False)
        clock.advance(0.41)
    elif failure == "lease":
        clock.advance(1.01)
        port.observe()
    else:
        port.terminal(success=False, release_confirmed=True)
    result = coordinator.step(7, False)
    assert result.fault and result.block_public
    assert result.stop_seed is None and result.resume_seed is None


def test_changed_stop_target_or_fault_cannot_replace_the_requested_bridge(rig):
    coordinator, port, clock = enter(rig)
    coordinator.stop("stop", command())
    result = coordinator.stop("stop", command(0.8))
    assert result.fault == "preset_stop_target_changed" and result.block_public
    assert port.count("stop_to") == 1 and port.count("cancel") == 1


def test_stop_argument_cancels_before_same_tick_normal_exit_request(rig):
    coordinator, port, clock = enter(rig)
    refreshes = port.count("refresh")
    result = coordinator.step(7, False, command(), True, stop_reason="operator_touch")
    assert port.calls[-1] == ("cancel", "operator_touch")
    assert port.count("refresh") == refreshes
    assert result.block_public and result.resume_seed is None


def test_stop_wins_over_successful_normal_exit_terminal_in_the_same_tick(rig):
    coordinator, port, clock = enter(rig)
    target = exit_normally(rig)
    port.terminal(
        success=True,
        release_confirmed=True,
        final_target=target,
        final_target_at=clock(),
        final_target_verified=True,
    )
    result = coordinator.step(7, False, stop_reason="mirror_stop")
    assert result.phase == PresetPhase.STOPPED
    assert result.release_confirmed and result.resume_seed is None
    assert port.calls[-1] == ("cancel", "mirror_stop")


@pytest.mark.parametrize("stop_target", [None, command(0.0)])
def test_stop_after_completed_exit_returns_verified_resume_seed_for_baseline_wrap(rig, stop_target):
    coordinator, port, clock = enter(rig)
    target = exit_normally(rig)
    clock.advance(0.05)
    port.terminal(
        success=True,
        release_confirmed=True,
        final_target=target,
        final_target_at=clock(),
        final_target_verified=True,
    )
    result = coordinator.stop("operator_touch", stop_target)
    assert result.phase == PresetPhase.IDLE and result.release_confirmed and not result.fault
    assert result.resume_seed == target and result.stop_seed is None
    assert port.count("stop_to") == 0 and port.count("cancel") == 0
    assert coordinator.step(7, False).resume_seed is None


@pytest.mark.parametrize(
    "changes",
    [
        {"success": False},
        {"release_confirmed": False},
        {"final_target_verified": False},
        {"final_target": command(0.9)},
        {"final_target_at": -1.0},
        {"goal_id": "other-owner"},
    ],
)
def test_stop_after_unverified_exit_terminal_does_not_replace_the_fault_with_stopping(rig, changes):
    coordinator, port, clock = enter(rig)
    target = exit_normally(rig)
    evidence = dict(
        success=True,
        release_confirmed=True,
        final_target=target,
        final_target_at=clock(),
        final_target_verified=True,
    )
    port.terminal(**(evidence | changes))
    result = coordinator.stop("operator_touch", command(0.0))
    assert result.fault and result.phase == PresetPhase.FAULT and result.block_public
    assert result.resume_seed is None and result.stop_seed is None and port.count("stop_to") == 0


def test_late_goal_acceptance_after_stop_is_cancelled_without_renewal(rig):
    coordinator, port, clock = enter(rig, accepted=False)
    coordinator.stop("stop_pending")
    refreshes = port.count("refresh")
    clock.advance(0.1)
    port.accept(goal_id="late-goal")
    result = coordinator.step(7, True)
    assert result.block_public
    assert port.count("cancel") == 2
    assert port.count("refresh") == refreshes
    coordinator.step(7, True)
    assert port.count("cancel") == 2


def test_generation_change_cancels_old_action_and_blocks_new_round_until_release(rig):
    coordinator, port, clock = enter(rig)
    result = coordinator.step(8, True)
    assert result.fault == "preset_generation_changed"
    assert result.block_public and result.resume_seed is None
    assert not coordinator.request(8)
    port.terminal(success=True, release_confirmed=True)
    result = coordinator.step(8, True)
    assert result.release_confirmed and result.block_public  # fault never reopens the gate
    assert coordinator.request(8)
    assert coordinator.epoch == PresetEpoch(8, 2)
    assert port.count("begin") == 1  # the new generation must drain again


def test_old_epoch_terminal_cannot_release_or_seed_a_new_action(rig):
    coordinator, port, clock = enter(rig)
    old_epoch = coordinator.epoch
    coordinator.stop()
    port.terminal(release_confirmed=True)
    coordinator.step(7, True)
    assert coordinator.request(8)
    coordinator.step(8, True)
    clock.advance(0.6)
    coordinator.step(8, True)
    port.state = PresetSnapshot(
        epoch=old_epoch,
        accepted=True,
        terminal=True,
        success=True,
        release_confirmed=True,
        goal_id="goal-1",
        final_target=command(),
        final_target_at=clock(),
        final_target_verified=True,
        observed_at=clock(),
    )
    result = coordinator.step(8, False, command(), True)
    assert result.block_public and result.resume_seed is None
    assert not result.release_confirmed


@pytest.mark.parametrize("cause", ["actor_lost", "runtime_lost", "fault", "lease_expired", "shutdown"])
def test_failures_cancel_and_never_resume_even_with_a_later_successful_terminal(rig, cause):
    coordinator, port, clock = enter(rig)
    refreshes = port.count("refresh")
    port.observe(fault=cause)
    result = coordinator.step(7, True)
    assert result.fault == cause and result.block_public
    assert port.count("refresh") == refreshes
    port.terminal(
        success=True,
        release_confirmed=True,
        final_target=command(),
        final_target_at=clock(),
        final_target_verified=True,
    )
    result = coordinator.step(7, True)
    assert result.block_public and result.resume_seed is None


def test_lost_feedback_stops_lease_renewal_and_cancels(rig):
    coordinator, port, clock = enter(rig)
    refreshes = port.count("refresh")
    # Keep the local lease timer fresh, while the server evidence goes stale.
    clock.advance(0.5)
    coordinator.step(7, True)
    clock.advance(0.5)
    coordinator.step(7, True)
    clock.advance(0.01)
    result = coordinator.step(7, True)
    assert result.fault == "preset_status_stale"
    assert result.block_public and result.resume_seed is None
    assert port.count("refresh") == refreshes + 2
    assert port.calls[-1][0] == "cancel"


def test_lapsed_local_lease_is_not_revived_even_if_new_feedback_arrives(rig):
    coordinator, port, clock = enter(rig)
    refreshes = port.count("refresh")
    clock.advance(1.01)
    port.observe()
    result = coordinator.step(7, True)
    assert result.fault == "preset_lease_lapsed"
    assert result.block_public and result.resume_seed is None
    assert port.count("refresh") == refreshes
    assert port.calls[-1] == ("cancel", "lease_lapsed")


def test_terminal_without_release_proof_never_opens_either_public_channel(rig):
    coordinator, port, clock = enter(rig)
    coordinator.stop()
    port.terminal(success=True, release_confirmed=False)
    result = coordinator.step(7, True)
    assert result.block_public and not result.release_confirmed
    assert result.resume_seed is None
    assert not coordinator.request(8)
    port.observe(release_confirmed=True)
    result = coordinator.step(7, True)
    assert not result.block_public and result.release_confirmed
    assert coordinator.request(8)


@pytest.mark.parametrize(
    "change",
    [
        {"release_confirmed": False},
        {"success": False},
        {"final_target_verified": False},
        {"final_target_at": -1.0},
        {"final_target_at": 10.0},
        {"final_target_at": float("nan")},
        {"final_target": None},
        {"final_target": command(0.9)},
        {"final_target": UpperBodyCommand((0.0,) * 14, (0.0,) * 19)},
        {"final_target": UpperBodyCommand((float("inf"),) * 14, (0.0,) * 20)},
        {"goal_id": "another-owner"},
        {"pending": True},
    ],
)
def test_invalid_normal_exit_evidence_never_resumes(rig, change):
    coordinator, port, clock = enter(rig)
    target = exit_normally(rig)
    evidence = dict(
        terminal=True,
        pending=False,
        success=True,
        release_confirmed=True,
        final_target=target,
        final_target_at=clock(),
        final_target_verified=True,
    )
    port.observe(**(evidence | change))
    result = coordinator.step(7, False)
    assert result.fault
    assert result.block_public and result.resume_seed is None


@pytest.mark.parametrize("target,fresh", [(None, True), (command(), False), (command(), 1)])
def test_normal_exit_without_a_fresh_complete_target_cancels(rig, target, fresh):
    coordinator, port, clock = enter(rig)
    refreshes = port.count("refresh")
    result = coordinator.step(7, False, target, fresh)
    assert result.fault == "preset_exit_target_unavailable"
    assert result.block_public and result.resume_seed is None
    assert port.count("refresh") == refreshes


def test_exit_target_is_frozen_and_human_reentry_does_not_restart_the_goal(rig):
    coordinator, port, clock = enter(rig)
    target = exit_normally(rig)
    clock.advance(0.11)
    port.observe()
    result = coordinator.step(7, True, command(0.8), True)
    assert result.phase == PresetPhase.EXITING
    assert port.calls[-1] == ("refresh", target)
    assert port.count("begin") == 1


def test_public_stream_conflict_after_named_dispatch_cancels_the_owner(rig):
    coordinator, port, clock = enter(rig)
    refreshes = port.count("refresh")
    result = coordinator.step(7, True, public_quiet=False)
    assert result.fault == "preset_public_stream_active"
    assert result.block_public and result.resume_seed is None
    assert port.count("refresh") == refreshes
    assert port.calls[-1] == ("cancel", "public_stream_active")


def test_unexpected_terminal_while_human_holds_does_not_restart_or_resume(rig):
    coordinator, port, clock = enter(rig)
    port.terminal(success=True, release_confirmed=True)
    result = coordinator.step(7, True)
    assert result.fault == "preset_unexpected_terminal"
    assert result.resume_seed is None and result.block_public
    assert not coordinator.request(7)
    assert port.count("begin") == 1


@pytest.mark.parametrize("method", ["begin", "refresh", "snapshot", "cancel"])
def test_port_failure_is_nonblocking_and_never_assumes_release(rig, method):
    coordinator, port, clock = rig
    if method in {"begin", "refresh"}:
        port.fail = method
        enter(rig, accepted=False)
    else:
        enter(rig)
        port.fail = method
        if method == "cancel":
            coordinator.stop()
    result = coordinator.step(7, True)
    assert result.fault.startswith(f"preset_port_{method}")
    assert result.block_public and result.resume_seed is None
    assert not result.release_confirmed


@pytest.mark.parametrize("invalid_time", [float("nan"), float("inf"), -1.0])
def test_invalid_or_regressing_clock_cancels_without_renewal(rig, invalid_time):
    coordinator, port, clock = enter(rig)
    refreshes = port.count("refresh")
    clock.now = invalid_time
    result = coordinator.step(7, True)
    assert result.fault == "preset_clock_invalid"
    assert result.block_public and port.count("refresh") == refreshes


@pytest.mark.parametrize("with_port", [True, False])
def test_disabled_configuration_is_completely_inert(with_port):
    clock = Clock()
    port = Port(clock)
    coordinator = PresetCoordinator(PresetConfig(), port if with_port else None, clock)
    assert not coordinator.request(7)
    for generation in (7, 8):
        assert not coordinator.step(generation, True, stop_reason="stop").block_public
        assert not coordinator.stop().fault
    assert port.calls == []


def test_enabled_configuration_without_a_port_is_inert():
    coordinator = PresetCoordinator(PresetConfig(enabled=True))
    assert not coordinator.enabled
    assert not coordinator.request(7)
    assert not coordinator.step(7, True).block_public


@pytest.mark.parametrize(
    "settings",
    [
        {"enabled": "true"},
        {"motion_name": ""},
        {"drain_s": 0.59},
        {"drain_s": True},
        {"lease_period_s": 0.0},
        {"lease_period_s": 1.0},
        {"lease_timeout_s": 1.1},
        {"snapshot_max_age_s": float("nan")},
        {"final_target_max_age_s": -1.0},
    ],
)
def test_invalid_lifecycle_settings_fail_closed(settings):
    with pytest.raises(ValueError):
        PresetConfig(**settings)
