"""Real mirror core with isolated runtime/action ports; no ROS or robot output."""

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from manipulation_execution.mirror_session.core import MirrorSession, Phase, SessionConfig, SessionState
from manipulation_execution.mirror_session.hooks import (
    CoverDetector,
    SessionHooks,
    SessionInputs,
    UpperBodyCommand,
    WrapUpStep,
)
from manipulation_execution.mirror_session.preset import PresetConfig, PresetCoordinator, PresetSnapshot
from manipulation_execution.mirror_session.runtime import RuntimeObservation, WaitReason

IDENTITY = {"id": "fake", "type": "fake", "runtime_name": "isolated", "runtime_version": "1"}


def command(value=0.1):
    return UpperBodyCommand((value,) * 14, (value + 0.2,) * 20)


class Clock:
    now = 10.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def sleep(self, seconds):
        self.advance(seconds)


class Runtime:
    def __init__(self):
        self.calls = []
        self.status = RuntimeObservation("ACTIVE", "idle", False, (), "isolated", "1", IDENTITY, 0.0)

    def get_status(self, timeout_s):
        self.calls.append(("get_status", timeout_s))
        return self.status

    def set_mode(self, mode, timeout_s):
        self.calls.append(("set_mode", mode))
        self.status = replace(self.status, active_mode=mode)
        return True, mode

    def stop(self, policy, timeout_s):
        self.calls.append(("stop", policy))
        self.status = replace(self.status, stop_latched=True)
        return True, "latched"


class Named:
    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.state = PresetSnapshot()
        self.cancelled = threading.Event()

    def begin(self, name, epoch):
        self.calls.append(("begin", name, epoch))
        self.state = PresetSnapshot(epoch=epoch, stage="pending", pending=True, observed_at=self.clock())

    def refresh(self, exit_target=None):
        self.calls.append(("refresh", exit_target))

    def cancel(self, reason):
        self.calls.append(("cancel", reason))
        self.cancelled.set()

    def snapshot(self):
        return self.state

    def observe(self, **changes):
        self.state = replace(self.state, observed_at=self.clock(), **changes)

    def accept(self, stage="holding"):
        self.observe(pending=False, accepted=True, goal_id="one-owner", stage=stage)

    def finish(self, target=None):
        self.observe(
            pending=False,
            terminal=True,
            success=True,
            release_confirmed=True,
            final_target=target,
            final_target_at=self.clock(),
            final_target_verified=target is not None,
        )


class Hooks(SessionHooks):
    def __init__(self):
        self.raw = command(0.8)
        self.post_calls = []
        self.last_input = None
        self.wrap_calls = []
        self.cancel = False

    def retarget(self, inputs):
        self.last_input = inputs
        return self.raw

    def post_process(self, target):
        self.post_calls.append((target, self.last_input.last_sent))
        return command(0.1)

    def cancel_criteria(self, inputs, round_elapsed_s):
        return self.cancel

    def reset_pose(self, arm_at_takeover):
        return command(0.0)

    def wrap_up(self, context):
        self.wrap_calls.append(context)
        return WrapUpStep(command(0.0), True)


class Cover(CoverDetector):
    def update(self, frame):
        return frame == "covered"


def make_rig(*, enabled=True, with_preset=True, target_available=True):
    clock, runtime, sent, hooks = Clock(), Runtime(), [], Hooks()
    seed_flags, publish_fail, holder = [], [False], {}
    named = Named(clock)
    preset = PresetCoordinator(PresetConfig(enabled=enabled), named, clock)
    targets = []

    def target(raw):
        targets.append(raw)
        return raw if target_available else None

    def publish(value):
        seed_flags.append(holder["session"].preset_seed_emitting)
        if publish_fail[0]:
            raise RuntimeError("isolated publisher failed")
        sent.append(value)

    session = MirrorSession(
        SessionConfig("policy_stream", IDENTITY),
        runtime,
        publish,
        hooks,
        Cover(),
        clock,
        clock.sleep,
        preset=preset if with_preset else None,
        preset_target=target,
    )
    holder["session"] = session
    assert session.startup().idle
    runtime.status = replace(runtime.status, active_mode="policy_stream")
    session.observe_arm_position((0.0,) * 14)
    assert session.request("start")[0]
    session.tick()
    assert session.phase == Phase.READY
    assert session.operator_touch()[0]
    runtime.calls.clear()
    hooks.wrap_calls.clear()
    sent.clear()
    return SimpleNamespace(
        session=session,
        named=named,
        clock=clock,
        runtime=runtime,
        sent=sent,
        hooks=hooks,
        preset=preset,
        targets=targets,
        seed_flags=seed_flags,
        publish_fail=publish_fail,
    )


def tick(rig, *, active=True, valid=True, seconds=0.0, camera=None, matcher=True):
    rig.clock.advance(seconds)
    rig.session.observe_status(rig.runtime.status)
    if matcher:
        rig.session.observe_preset(active, valid=valid)
    if rig.named.state.epoch is not None:
        rig.named.observe()
    rig.session.tick(SessionInputs(camera_frame=camera, publish_enabled=True))


def dispatch(rig, *, accept=True):
    tick(rig)
    tick(rig, seconds=0.3)
    tick(rig, seconds=0.3)
    assert sum(call[0] == "begin" for call in rig.named.calls) == 1
    if accept:
        rig.named.accept()
        tick(rig)


def test_normal_handoff_keeps_public_running_and_resumes_once_from_verified_34_value_seed():
    rig = make_rig()
    tick(rig, active=False)
    assert rig.sent[-1] == command(0.1)
    previous_posts = len(rig.hooks.post_calls)
    previous_sends = len(rig.sent)
    dispatch(rig)
    assert rig.session.state == SessionState.MIRRORING and rig.session.phase == Phase.RUNNING
    assert rig.session.holding
    assert len(rig.sent) == previous_sends
    assert len(rig.hooks.post_calls) == previous_posts
    for _ in range(3):
        tick(rig, seconds=0.1)
    assert sum(call[0] == "begin" for call in rig.named.calls) == 1
    rig.hooks.raw = command(0.6)
    tick(rig, active=False)
    assert rig.named.calls[-1] == ("refresh", rig.hooks.raw)
    assert rig.targets[-1] == rig.hooks.raw  # raw target bypasses the pre-preset limiter
    rig.named.finish(rig.hooks.raw)
    tick(rig, active=False)
    assert rig.sent[-1] == rig.hooks.raw
    assert rig.seed_flags[-1] is True
    assert not rig.session.preset_seed_emitting
    assert len(rig.sent) == previous_sends + 1
    assert len(rig.hooks.post_calls) == previous_posts
    tick(rig, active=False)
    assert rig.hooks.post_calls[-1][1] == command(0.6)
    assert rig.seed_flags[-1] is False
    assert rig.session.state == SessionState.MIRRORING and rig.session.phase == Phase.RUNNING


@pytest.mark.parametrize("with_preset,enabled", [(False, True), (True, False)])
def test_absent_or_disabled_preset_leaves_baseline_output_and_callbacks_unchanged(with_preset, enabled):
    rig = make_rig(with_preset=with_preset, enabled=enabled)
    for _ in range(5):
        tick(rig, active=True, valid=False, seconds=0.2)
    assert rig.sent == [command(0.1)] * 5
    assert len(rig.hooks.post_calls) == 5
    assert rig.targets == [] and rig.named.calls == []
    assert rig.session.state == SessionState.MIRRORING


@pytest.mark.parametrize("with_preset,enabled", [(False, True), (True, False)])
def test_absent_or_disabled_preset_keeps_baseline_shutdown_release(with_preset, enabled):
    rig = make_rig(with_preset=with_preset, enabled=enabled)
    outcome = rig.session.shutdown()
    assert outcome.idle and outcome.sent_idle
    assert ("set_mode", "idle") in rig.runtime.calls
    assert not any(call[0] == "stop" for call in rig.runtime.calls)
    assert rig.named.calls == []


def test_enabled_preset_without_any_match_state_keeps_plain_mirroring():
    rig = make_rig()
    for _ in range(10):
        tick(rig, matcher=False, seconds=0.2)
    assert rig.session.state == SessionState.MIRRORING and rig.session.phase == Phase.RUNNING
    assert rig.sent == [command(0.1)] * 10
    assert rig.named.calls == [] and rig.targets == []
    assert not any(call[0] == "stop" for call in rig.runtime.calls)


def test_inactive_invalid_match_state_does_not_trigger_or_end_plain_mirroring():
    rig = make_rig()
    for _ in range(10):
        tick(rig, active=False, valid=False, seconds=0.2)
    assert rig.session.state == SessionState.MIRRORING and rig.session.phase == Phase.RUNNING
    assert rig.sent == [command(0.1)] * 10
    assert rig.named.calls == [] and rig.targets == []
    assert not any(call[0] == "stop" for call in rig.runtime.calls)


def test_old_active_match_state_cannot_trigger_before_a_fresh_confirmation():
    rig = make_rig()
    rig.session.observe_preset(True)
    tick(rig, matcher=False, seconds=0.36)
    assert rig.session.state == SessionState.MIRRORING and rig.sent == [command(0.1)]
    assert rig.named.calls == []
    tick(rig, active=True)
    assert len(rig.sent) == 1  # only a new valid observation can close the public gate
    assert rig.preset.phase.value == "draining"


@pytest.mark.parametrize("invalid", [False, True])
def test_match_state_loss_after_drain_started_still_holds(invalid):
    rig = make_rig()
    tick(rig, active=True)
    tick(rig, active=False, valid=not invalid, seconds=0.36 if not invalid else 0.0, matcher=invalid)
    assert rig.session.state == SessionState.WAIT
    assert rig.session.wait_reason == WaitReason.FAULT
    assert rig.sent == []
    assert not any(call[0] == "begin" for call in rig.named.calls)
    assert ("stop", "HOLD") in rig.runtime.calls


@pytest.mark.parametrize("invalid", [False, True])
def test_matcher_loss_is_a_failed_round_and_hold_never_a_normal_exit(invalid):
    rig = make_rig()
    dispatch(rig)
    sends = len(rig.sent)
    tick(rig, active=False, valid=not invalid, seconds=0.36 if not invalid else 0.0, matcher=invalid)
    assert rig.session.state == SessionState.WAIT
    assert rig.session.wait_reason == WaitReason.FAULT
    assert len(rig.sent) == sends
    assert rig.named.cancelled.is_set()
    assert ("stop", "HOLD") in rig.runtime.calls
    assert not any(call[0] == "set_mode" for call in rig.runtime.calls)
    assert not rig.session.last_release.idle


def test_normal_exit_without_current_full_target_fails_closed():
    rig = make_rig(target_available=False)
    dispatch(rig)
    tick(rig, active=False)
    assert rig.session.state == SessionState.WAIT
    assert rig.session.wait_reason == WaitReason.FAULT
    assert rig.sent == []
    assert ("stop", "HOLD") in rig.runtime.calls
    assert not any(call[0] == "set_mode" for call in rig.runtime.calls)


@pytest.mark.parametrize("cause", ["touch", "cover", "criteria"])
@pytest.mark.parametrize("accepted", [False, True])
def test_named_stop_waits_for_release_then_hold_without_wrap_or_idle(cause, accepted):
    rig = make_rig()
    dispatch(rig, accept=accepted)
    if cause == "touch":
        assert rig.session.operator_touch()[0]
    elif cause == "cover":
        tick(rig, camera="covered")
    else:
        rig.hooks.cancel = True
        tick(rig)
    assert rig.named.cancelled.is_set()
    tick(rig, seconds=0.1)
    assert rig.hooks.wrap_calls == [] and rig.sent == []
    assert not any(call[0] == "set_mode" for call in rig.runtime.calls)
    rig.named.finish()
    tick(rig)
    assert rig.session.state == SessionState.WAIT
    assert rig.hooks.wrap_calls == [] and rig.sent == []
    assert ("stop", "HOLD") in rig.runtime.calls
    assert not rig.session.last_release.idle
    assert "certified reset bridge" in rig.session.detail


def test_public_stop_does_not_claim_success_or_clear_latch_after_named_release():
    rig = make_rig()
    dispatch(rig)
    result = []
    thread = threading.Thread(target=lambda: result.append(rig.session.request("stop")))
    thread.start()
    assert rig.named.cancelled.wait(1.0)
    try:
        tick(rig)
        assert rig.hooks.wrap_calls == [] and rig.sent == []
        rig.named.finish()
        tick(rig)
        thread.join(1.0)
        assert not thread.is_alive()
        assert result and result[0][0] is False
        assert rig.session.stop_final_gap_deg is None
        assert rig.runtime.status.stop_latched
        assert not any(call[0] == "set_mode" for call in rig.runtime.calls)
    finally:
        if thread.is_alive():
            rig.named.finish()
            tick(rig)
            thread.join(1.0)


def test_unconfirmed_release_overruns_to_hold_without_public_wrap():
    rig = make_rig()
    dispatch(rig)
    rig.session.operator_touch()
    tick(rig, seconds=rig.session._config.wrap_up_budget_s + 0.01)
    assert rig.session.state == SessionState.WAIT
    assert rig.hooks.wrap_calls == [] and rig.sent == []
    assert ("stop", "HOLD") in rig.runtime.calls
    assert not any(call[0] == "set_mode" for call in rig.runtime.calls)
    assert not rig.session.request("start")[0]


@pytest.mark.parametrize("cause", ["shutdown", "runtime_mode", "runtime_latch", "runtime_loss"])
def test_revocation_and_shutdown_cancel_before_hold_and_never_publish(cause):
    rig = make_rig()
    dispatch(rig)
    if cause == "shutdown":
        outcome = rig.session.shutdown()
        assert not outcome.idle
    elif cause == "runtime_loss":
        rig.clock.advance(2.01)
        rig.session.tick()
    else:
        changes = {"active_mode": "other"} if cause == "runtime_mode" else {"stop_latched": True}
        rig.runtime.status = replace(rig.runtime.status, **changes)
        rig.session.observe_status(rig.runtime.status)
        rig.session.tick()
    assert rig.named.cancelled.is_set()
    assert rig.session.state == SessionState.WAIT
    assert rig.sent == [] and rig.hooks.wrap_calls == []
    assert ("stop", "HOLD") in rig.runtime.calls
    assert not any(call[0] == "set_mode" for call in rig.runtime.calls)


def test_stop_during_drain_dispatches_nothing_and_preserves_baseline_cancel_wrap():
    rig = make_rig()
    tick(rig)
    rig.session.operator_touch()
    tick(rig)
    assert rig.session.state == SessionState.STANDBY and rig.session.phase == Phase.COOLDOWN
    assert len(rig.hooks.wrap_calls) == 1
    assert rig.named.calls == []
    assert ("stop", "HOLD") not in rig.runtime.calls


def test_new_round_drops_old_match_state_and_uses_its_own_preset_generation():
    rig = make_rig()
    tick(rig)
    first_generation = rig.preset.epoch.generation
    rig.session.operator_touch()
    tick(rig)
    rig.session.observe_preset(True)  # cooldown messages cannot start the next round
    assert rig.session.operator_touch()[0]
    rig.session.tick()
    assert not any(call[0] == "begin" for call in rig.named.calls)
    tick(rig)
    assert rig.preset.epoch.generation == first_generation + 1


def test_stop_wins_over_same_tick_normal_terminal_seed():
    rig = make_rig()
    dispatch(rig)
    tick(rig, active=False)
    rig.named.finish(rig.hooks.raw)
    rig.session.operator_touch()
    tick(rig, active=False)
    assert rig.sent == []
    assert rig.session.state == SessionState.WAIT
    assert ("stop", "HOLD") in rig.runtime.calls


def test_seed_publication_failure_is_caught_and_holds_without_clearing_stop_latch():
    rig = make_rig()
    dispatch(rig)
    tick(rig, active=False)
    rig.named.finish(rig.hooks.raw)
    rig.publish_fail[0] = True
    tick(rig, active=False)
    assert rig.session.state == SessionState.WAIT
    assert rig.session.wait_reason == WaitReason.HOOK_ERROR
    assert not rig.session.preset_seed_emitting
    assert "seed publication failed" in rig.session.detail
    assert ("stop", "HOLD") in rig.runtime.calls
    assert not any(call[0] == "set_mode" for call in rig.runtime.calls)
    assert not rig.session.last_release.idle
