"""v1.2: A pauses on CBF holds, the entry gates, B without the latch. A scripted data plane, no ROS.

``Plane`` publishes the data plane's status as ``dataplane_node.cpp`` does: 20 Hz,
with ``at``, ``safety_epoch``, ``control_seq``, ``arm_owner`` and the CBF's section,
whose counters move every 20 ms CBF call. The old behaviour (every switch off) is
pinned by test_template_motion_blender.py.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import replace

import pytest
from test_template_motion_blender import MIRROR, OFF, P_HEART, TICK, Rig, exit_reason, lines, max_jerkless, max_step

from manipulation_execution.template_motion.blender import MIN_JERK_PEAK, BlendConfig, Phase, outcome_of
from manipulation_execution.template_motion.port import CbfMonitor

PERIOD = 0.05  # the data plane's status
RELEASE_AFTER = 0.2  # the CBF's hold_release_after_s on site
NEW = BlendConfig()


class Plane:
    """The data plane: one CBF call per session tick, a status every PERIOD.

    ``holds`` are [start, end) of non-collision holds (``collisions`` the same, with
    stuck_collision); ``every`` n holds one CBF call in n. stuck_s is 0 on the first
    held call, as the CBF counts it; releasing from RELEASE_AFTER on. ``delay(at)``
    is how late a status made at ``at`` is received (None: lost).
    """

    def __init__(self, now):
        self.epoch, self.owner, self.control_seq = 3, 1, 0
        self.with_epoch = True
        self.mirror_active = True
        self.calls = self.held = 0
        self.run = 0  # held calls in a row
        self.holds, self.collisions = [], []
        self.every = None
        self.next_at = now
        self.delay = lambda at: 0.0
        self.queue = []  # (received at, message)
        self.collision = False

    @staticmethod
    def _within(spans, now):
        return any(start - TICK / 2 <= now < end - TICK / 2 for start, end in spans)

    def status(self, now):
        message = {
            "at": now,
            "control_seq": self.control_seq,
            "arm_owner": self.owner,
            "cbf": {
                "held": self.held,
                "calls": self.calls,
                "stuck_s": round(max(self.run - 1, 0) * TICK, 3),
                "releasing": self.run > 0 and (self.run - 1) * TICK >= RELEASE_AFTER - 1e-9,
                "stuck_collision": self.collision,
                "mirror_active": self.mirror_active,
                "hold_release_after_s": RELEASE_AFTER,
            },
        }
        if self.with_epoch:
            message["safety_epoch"] = self.epoch
        return message

    def tick(self, now):
        """The statuses received this tick, oldest first."""
        self.calls += 1
        self.collision = self._within(self.collisions, now)
        holding = self.collision or self._within(self.holds, now) or bool(self.every and self.calls % self.every == 0)
        self.run = self.run + 1 if holding else 0
        self.held += holding
        if now >= self.next_at - 1e-9:
            self.next_at += PERIOD
            self.control_seq += 1  # every heartbeat; never a revocation
            delay = self.delay(now)
            if delay is not None:  # late, never out of order: one publisher, one subscriber
                at = max([now + delay] + [at for at, _ in self.queue])
                self.queue.append((at, self.status(now)))
        due = [message for at, message in self.queue if at <= now + 1e-9]
        self.queue = [(at, message) for at, message in self.queue if at > now + 1e-9]
        return due


class PauseRig(Rig):
    """The blender rig with v1.2 on (default) and the Plane instead of a scripted cbf section."""

    def __init__(self, config=NEW, warm=True, **kw):
        super().__init__(config, **kw)
        self.cbf = None
        self.plane = Plane(self.now)
        self.extra = []  # statuses received with the next tick's, before them
        self.paused = []  # the blender's paused flag after each tick
        if warm:
            self.run(1.2)  # the held-share gate needs a second of statuses

    def hear(self):
        super().hear()
        for message in self.extra + self.plane.tick(self.now):
            self.cbf_monitor.observe(json.dumps(message), self.now)
        self.extra = []

    def step(self, dt=TICK):
        arm = super().step(dt)
        self.paused.append(self.blender.snapshot()["paused"])
        return arm

    def into_a(self, seconds):
        self.active["heart"] = 7
        assert self.run(1.0, lambda: self.blender.phase is Phase.A)
        self.run(seconds)
        assert self.blender.phase is Phase.A

    def hold(self, start, length, collision=False):
        (self.plane.collisions if collision else self.plane.holds).append((self.now + start, self.now + start + length))

    def spans(self):
        """(first, last) indices into ``out`` of each pause."""
        spans, start = [], None
        for k, paused in enumerate(self.paused):
            if paused and start is None:
                start = k
            if not paused and start is not None:
                spans.append((start, k - 1))
                start = None
        return spans + ([(start, len(self.paused) - 1)] if start is not None else [])


def speed_limit(dist, T_A=1.1):
    return max(NEW.c_peak_speed, MIN_JERK_PEAK * dist / T_A) * 1.02


def accel_limit():
    return max(NEW.a_resume_accel, NEW.c_peak_accel) * 1.02


# -- pause and resume ----------------------------------------------------------------------


def test_a_short_hold_freezes_the_request_and_resumes_with_one_status():
    rig = PauseRig()
    rig.into_a(0.4)
    rig.hold(0.0, 0.1)
    rig.mirror = MIRROR  # still
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.WAIT_ACK)
    (first, last), *more = rig.spans()
    assert not more
    frozen = rig.out[first:last + 1]
    assert len(set(frozen)) == 1  # arm: not a single bit moves
    assert "why=hold" in lines(rig, "template pause")[0]
    resume = lines(rig, "template resume")[0]
    assert "long=False" in resume
    # From A's full speed 0.06 s would take 32 rad/s2: stretched to a_resume_accel, still under the long floor.
    assert NEW.a_short_ramp_s < float(resume.split("ramp=")[1].split()[0]) < NEW.a_resume_ramp_s
    assert max_step(rig.out) <= speed_limit(0.6) * TICK
    assert max_jerkless(rig.out[last:]) <= accel_limit() * TICK * TICK * 1.1  # the pause tick alone may stop dead
    assert rig.started() is not None


def test_the_hand_stands_still_in_a_pause():
    rig = PauseRig()
    rig.active["heart"] = 7
    rig.run(1.0, lambda: rig.blender.phase is Phase.A)
    rig.step()
    hands = []
    rig.hold(0.0, 0.3)
    for _ in range(25):
        rig.step()
        hands.append(rig.hand)
    paused = [h for h, p in zip(hands, rig.paused[-25:], strict=True) if p]
    assert len(paused) > 5 and len(set(paused)) == 1


@pytest.mark.parametrize("gap", [0.0, 0.1, 0.3])
def test_a_resume_is_smooth_whatever_the_person_did_in_the_pause(gap):
    rig = PauseRig()
    rig.into_a(0.4)
    rig.hold(0.0, 0.35)  # long: the CBF releases at 0.2
    assert rig.run(0.2, lambda: rig.blender.snapshot()["paused"])
    rig.mirror = tuple(v + (gap if j in (1, 8) else 0.0) for j, v in enumerate(MIRROR))
    assert rig.run(3.0, lambda: rig.blender.phase is Phase.WAIT_ACK)
    (first, last), = rig.spans()
    assert len(set(rig.out[first:last + 1])) == 1
    resume = lines(rig, "template resume")[0]
    assert "long=True" in resume
    after = rig.out[last:]
    assert max_step(after) <= speed_limit(0.6 + gap) * TICK * 1.05
    assert max_jerkless(after) <= accel_limit() * TICK * TICK * 1.1


def test_releasing_keeps_the_pause_and_a_long_hold_needs_two_clean_statuses():
    rig = PauseRig()
    rig.into_a(0.3)
    rig.hold(0.0, 0.5)
    seen = []
    for _ in range(40):
        rig.step()
        latest = rig.cbf_monitor.latest()
        seen.append((latest.serial, latest.intervening, rig.paused[-1]))
    assert all(paused for serial, intervening, paused in seen if intervening and serial > seen[0][0] + 1)
    # The first clean status after the hold: still paused until the next one (a status read twice counts once).
    clean = [k for k in range(1, len(seen)) if not seen[k][1] and seen[k - 1][1]]
    assert clean
    k = clean[0]
    assert all(seen[j][2] for j in range(k, len(seen)) if seen[j][0] == seen[k][0])
    assert "long=True" in lines(rig, "template resume")[0]


def test_label_lost_or_collision_in_a_pause_or_a_ramp_goes_to_C_continuously():
    for when, what in [(0.1, "label"), (0.1, "collision"), (0.42, "label"), (0.42, "collision")]:
        rig = PauseRig()
        rig.into_a(0.4)
        rig.hold(0.0, 0.3)
        rig.run(0.05)
        rig.mirror = tuple(v + (0.2 if j == 1 else 0.0) for j, v in enumerate(MIRROR))
        rig.run(when)
        if what == "label":
            del rig.active["heart"]
        else:
            rig.hold(0.0, 0.1, collision=True)
        assert rig.run(0.5, lambda: rig.blender.phase is Phase.C), (when, what)
        rig.run(2.5)
        assert exit_reason(rig) == ("label_lost" if what == "label" else "a_cbf_collision")
        assert rig.blender.phase is not Phase.C
        assert max_step(rig.out) <= speed_limit(0.6) * TICK * 1.05, (when, what)
        assert rig.out[-1] == rig.mirror


def test_a_hold_that_outlasts_the_budget_goes_back_and_backs_off():
    rig = PauseRig()
    rig.into_a(0.4)
    rig.hold(0.0, 3.0)
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_pause_budget"
    exit = lines(rig, "template exit")[0]
    assert "outcome=cbf_hold" in exit and "pauses=1" in exit and "backoff=3.0" in exit
    budget = NEW.a_pause_budget_s + NEW.a_pause_budget_per_s * 1.1
    total = float(exit.split(" total=")[1].split()[0])
    assert total <= 1.1 + NEW.a_timeout_margin_s + budget
    del rig.active["heart"]
    rig.run(2.5)
    rig.active["heart"] = 7  # a new episode within the backoff
    rig.run(4.0)
    enter = lines(rig, "template enter")
    assert len(enter) == 2 and "blocked=backoff" in enter[1]


def test_pauses_add_up_and_a_resume_never_resets_the_budget():
    rig = PauseRig()
    rig.into_a(0.2)
    rig.hold(0.0, 0.5)
    rig.hold(0.9, 0.5)
    assert rig.run(2.5, lambda: rig.blender.phase is Phase.C)
    exit = lines(rig, "template exit")[0]
    assert exit_reason(rig) == "a_pause_budget" and "pauses=2" in exit
    assert len(lines(rig, "template resume")) == 1


def test_a_ramp_the_budget_cannot_finish_is_never_begun():
    rig = PauseRig(BlendConfig(a_resume_ramp_s=0.4))  # a resume costs 0.2 s of schedule
    rig.into_a(0.2)
    rig.hold(0.0, 0.97)  # clean after about 1.0 s, 0.15 s of budget left
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.C)
    exit = lines(rig, "template exit")[0]
    assert exit_reason(rig) == "a_pause_budget" and not lines(rig, "template resume")
    budget = NEW.a_pause_budget_s + NEW.a_pause_budget_per_s * 1.1
    assert float(exit.split(" paused_s=")[1].split()[0]) < budget  # left at the resume, not at the end


def test_no_status_A_can_trust_ends_A_at_the_absolute_cap():
    rig = PauseRig()
    rig.into_a(0.3)
    rig.plane.delay = lambda at: None
    t0 = rig.now
    assert rig.run(1.5, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_cbf_untrusted"
    assert rig.now - t0 <= NEW.a_status_age_s + NEW.a_untrusted_max_s + 3 * TICK
    assert "why=cbf_late" in lines(rig, "template pause")[0]
    (first, last), = rig.spans()
    assert len(set(rig.out[first:last + 1])) == 1
    assert "outcome=untrusted" in lines(rig, "template exit")[0]


def test_late_statuses_alone_pause_and_resume():
    rig = PauseRig()
    rig.into_a(0.3)
    rig.plane.delay = lambda at: 0.2  # queued: each one already old
    rig.run(0.3)
    assert rig.blender.snapshot()["paused"]
    rig.plane.delay = lambda at: 0.0
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.WAIT_ACK)


def test_a_hold_heard_late_keeps_the_request_still_before_the_cbf_releases():
    """The CBF releases a non-collision hold after 0.2 s toward our request. Statuses lost for 0.35 s from
    the hold's start: A stops by itself once the newest is a_status_age_s old, so the request the CBF
    releases toward stands still from hold start + 0.12 s on, and never moves before a clean status."""
    rig = PauseRig()
    rig.into_a(0.3)
    start = rig.now + TICK
    rig.hold(TICK, 0.25)
    rig.plane.delay = lambda at: None if start - 1e-9 <= at < start + 0.35 else 0.0
    rig.run(0.6)
    k0 = len(rig.out) - 30  # the hold's first tick
    frozen_from = k0 + round((NEW.a_status_age_s + TICK) / TICK)
    heard = k0 + round(0.35 / TICK)
    window = rig.out[frozen_from:heard + 1]
    assert len(set(window)) == 1
    advance = max(abs(a - b) for a, b in zip(rig.out[k0 - 1], window[0], strict=True))
    assert advance <= speed_limit(0.6) * (NEW.a_status_age_s + 2 * TICK)  # what the CBF's release moves at most
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.WAIT_ACK)


# -- the safety epoch ------------------------------------------------------------------------


def test_heartbeats_moving_control_seq_never_stop_a_resume():
    rig = PauseRig()
    rig.into_a(0.4)
    rig.hold(0.0, 0.3)
    rig.run(0.1)
    rig.plane.control_seq += 50
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.WAIT_ACK)
    assert not lines(rig, "template exit")


def test_an_epoch_change_before_the_start_never_starts():
    rig = PauseRig()
    rig.gate_lag = True  # waits at P for the gate
    rig.into_a(0.2)
    rig.run(0.95)  # past T_A (1.1), within its margin
    assert rig.blender.phase is Phase.A
    rig.plane.epoch += 1
    assert rig.run(0.2, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_epoch_changed" and rig.started() is None
    assert "outcome=safety" in lines(rig, "template exit")[0]


@pytest.mark.parametrize("epochs, reason", [((4, 5), "a_epoch_changed"), ((3, 3), "a_owner_changed")])
def test_a_revoke_and_restore_by_the_same_owner_in_one_batch_never_continues(epochs, reason):
    rig = PauseRig()
    rig.into_a(0.3)
    plane = rig.plane
    revoke = dict(plane.status(rig.now), arm_owner=0, safety_epoch=epochs[0])
    restore = dict(plane.status(rig.now), arm_owner=1, safety_epoch=epochs[1])
    plane.epoch = epochs[1]
    rig.extra = [revoke, restore]
    rig.step()
    assert rig.blender.phase is Phase.C and exit_reason(rig) == reason


def test_an_epoch_change_in_a_pause_ends_it():
    rig = PauseRig()
    rig.into_a(0.3)
    rig.hold(0.0, 0.6)
    rig.run(0.15)
    assert rig.blender.snapshot()["paused"]
    rig.plane.epoch += 1
    assert rig.run(0.2, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_epoch_changed"


def test_revoked_or_stopped_in_B_still_goes_back():
    rig = PauseRig()
    seg_id = to_playing_new(rig)
    rig.publish_port(state="STOPPED", seg_id=seg_id, reason="safety_epoch", P_actual=list(P_HEART))
    rig.step()
    assert rig.blender.phase is Phase.C and exit_reason(rig) == "stopped:safety_epoch"
    assert outcome_of(exit_reason(rig)) == "safety"


def test_no_safety_epoch_no_entry_and_in_A_a_checked_hand_back():
    rig = PauseRig()
    rig.plane.with_epoch = False
    rig.run(0.2)
    rig.active["heart"] = 7
    rig.run(1.0)
    assert rig.blender.phase is Phase.IDLE
    del rig.active["heart"]
    rig.run(0.1)
    assert "reason=epoch_missing" in lines(rig, "template reject")[0]
    rig.plane.with_epoch = True
    rig.into_a(0.3)
    rig.plane.with_epoch = False
    assert rig.run(0.2, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_epoch_missing"
    assert max_step(rig.out) <= speed_limit(0.6) * TICK * 1.05


# -- entry ---------------------------------------------------------------------------------


def test_the_held_share_gate_has_hysteresis():
    rig = PauseRig()
    rig.plane.every = 3  # a third of the calls held, one call at a time: never a stuck status
    rig.run(2.5)
    assert rig.blender.snapshot()["gate"]["open"] is False
    assert "template gate closed" in rig.logs[-1] or lines(rig, "template gate closed")
    rig.active["heart"] = 7
    rig.plane.every = 7  # 14%: between open (0.1) and close (0.2)
    rig.run(3.0)
    assert rig.blender.phase is Phase.IDLE
    rig.plane.every = None
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.A)
    assert lines(rig, "template gate open")
    assert "blocked=hold_ratio" in lines(rig, "template enter")[0]


def test_entry_waits_for_a_second_of_statuses():
    rig = PauseRig(warm=False)
    rig.active["heart"] = 7
    rig.run(0.6)
    assert rig.blender.phase is Phase.IDLE
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.A)
    assert "hold_ratio_unknown" in lines(rig, "template enter")[0]


def test_a_paused_timeout_backs_off_too_and_the_old_config_never_pauses():
    rig = PauseRig(config=OFF)
    rig.into_a(0.3)
    rig.hold(0.0, 0.2)
    assert rig.run(0.3, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_cbf_hold" and not any(rig.paused)
    assert "backoff=" not in lines(rig, "template exit")[0]


# -- B without the latch ---------------------------------------------------------------------


def to_playing_new(rig):
    rig.active["heart"] = 7
    assert rig.run(4.0, lambda: rig.blender.phase is Phase.WAIT_ACK)
    seg_id = rig.started()
    rig.publish_port(state="HOLD", seg_id=seg_id, seg="heart", P_actual=list(P_HEART), hand_P=[0.2] * 20)
    rig.step()
    assert rig.blender.phase is Phase.PLAYING
    return seg_id


def test_a_label_that_comes_back_before_min_hold_keeps_the_hold():
    rig = PauseRig()
    seg_id = to_playing_new(rig)
    triggers = rig.blender.triggers
    rig.run(0.2)
    del rig.active["heart"]
    rig.run(0.2)
    rig.active["heart"] = 7  # a new episode of the same gesture by the same person
    rig.run(1.0)
    assert rig.blender.phase is Phase.PLAYING and not rig.sent("release")
    assert "episode=1->2" in lines(rig, "template relabel")[0] and rig.blender.triggers == triggers
    rig.publish_port(state="EXIT_B", seg_id=seg_id, reason="hold_max", P_actual=list(P_HEART))
    rig.step()
    rig.publish_port(state="DONE", seg_id=seg_id, reason="hold_max", P_actual=list(P_HEART))
    rig.run(4.0)  # C, the cooldown, and the label still up
    assert exit_reason(rig) == "hold_max" and rig.blender.phase is Phase.IDLE
    assert len(rig.sent("start")) == 1  # the adopted episode is never entered on its own
    assert rig.blender.outcomes == {"complete": 1}


def test_a_lost_label_releases_at_min_hold_and_at_once_after_it():
    rig = PauseRig()
    to_playing_new(rig)
    del rig.active["heart"]
    t0 = rig.now
    assert rig.run(1.5, lambda: rig.blender.phase is Phase.WAIT_DONE)
    assert rig.now - t0 == pytest.approx(NEW.min_hold_s, abs=2 * TICK)


def test_a_release_is_never_taken_back():
    rig = PauseRig()
    seg_id = to_playing_new(rig)
    rig.run(1.0)
    del rig.active["heart"]
    rig.step()
    assert rig.blender.phase is Phase.WAIT_DONE
    rig.active["heart"] = 7
    rig.run(0.5)
    assert rig.blender.phase is Phase.WAIT_DONE and len(rig.sent("release")) == 1
    rig.publish_port(state="DONE", seg_id=seg_id, reason="release")
    assert rig.run(0.2, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "label_inactive"


def test_an_actor_change_in_B_releases_and_the_old_config_latches():
    rig = PauseRig()
    to_playing_new(rig)
    rig.run(1.0)
    rig.actor = replace(rig.actor, actor_id=8)
    rig.step()
    assert rig.blender.phase is Phase.WAIT_DONE
    old = PauseRig(config=OFF)
    to_playing_new(old)
    del old.active["heart"]
    old.run(0.2)
    old.active["heart"] = 7
    assert old.run(1.0, lambda: old.blender.phase is Phase.WAIT_DONE)  # latched: released at min_hold


# -- the monitor and the configuration --------------------------------------------------------


def cbf_message(at, held, calls, **cbf):
    return json.dumps({"at": at, "safety_epoch": 1, "arm_owner": 1, "cbf": {"held": held, "calls": calls, **cbf}})


def test_cbf_monitor_after_and_hold_ratio():
    monitor = CbfMonitor()
    assert monitor.hold_ratio(10.0, 2.0) is None
    for k in range(30):  # 1.5 s, a quarter held
        monitor.observe(cbf_message(10.0 + k * 0.05, k * 5 // 4, k * 5), 10.0 + k * 0.05)
    assert [s.serial for s in monitor.after(28)] == [29, 30]
    assert monitor.hold_ratio(11.45, 2.0) == pytest.approx(0.25, abs=0.01)
    assert monitor.hold_ratio(11.45, 4.0) is None  # less than half the window of statuses
    assert monitor.hold_ratio(13.0, 2.0) is None  # nothing for half the window
    monitor.observe(cbf_message(11.5, 0, 3), 11.5)  # the data plane restarted
    assert monitor.hold_ratio(11.5, 2.0) is None and len(monitor.after(0)) == 1
    monitor.observe(json.dumps({"at": 11.55}), 11.55)
    assert monitor.hold_ratio(11.55, 2.0) == 0.0  # the CBF is off


@pytest.mark.parametrize(
    "raw",
    [{"b_label_unlatch": 1}, {"enter_hold_ratio_open": 1.5}, {"enter_hold_ratio_open": 0.3},
     {"a_status_age_s": 0.2}, {"a_untrusted_max_s": 0}, {"a_pause_typo": 1.0}],
)  # fmt: skip
def test_v12_keys_are_checked(raw):
    with pytest.raises(ValueError):
        BlendConfig.from_mapping(raw)


def test_the_off_values_are_accepted():
    raw = {"a_pause_budget_s": 0, "enter_hold_window_s": 0, "load_backoff_s": 0, "b_label_unlatch": False}
    assert BlendConfig.from_mapping(raw) == OFF


# -- a fixed-seed stress: the same events, old and new -------------------------------------------


def stress(config, seed, share, seconds=40.0):
    """Holds of 0.04-0.4 s (clustered) covering about ``share`` of the time, statuses sometimes late,
    lost or batched, a mirror that moves; the label held 5 s at a time. Returns the counts and the
    output, checked for continuity outside pause ticks."""
    rng = random.Random(seed)
    rig = PauseRig(config=config)
    t, end = rig.now, rig.now + seconds
    while t < end:  # holds in clusters
        burst = rng.randint(1, 4)
        for _ in range(burst):
            length = rng.uniform(0.04, 0.4)
            rig.plane.holds.append((t, t + length))
            t += length + rng.uniform(0.02, 0.2)
        t += rng.expovariate(1.0) * (1.0 - share) / max(share, 0.02) * 0.5
    late = {}

    def delay(at):
        key = round(at / PERIOD)
        if key not in late:
            r = rng.random()
            late[key] = None if r < 0.03 else rng.uniform(0.05, 0.15) if r < 0.1 else 0.0
        return late[key]

    rig.plane.delay = delay
    amp, freq = rng.uniform(0.0, 0.3), rng.uniform(0.1, 0.6)
    k, t0 = 0, rig.now
    while rig.now < end:
        if k % 300 == 0:
            rig.active["heart"] = 7
        elif k % 300 == 250:
            rig.active.pop("heart", None)
        rig.mirror = tuple(v + (amp * math.sin(2 * math.pi * freq * (rig.now - t0)) if j == 1 else 0.0)
                           for j, v in enumerate(MIRROR))  # fmt: skip
        play(rig)
        rig.step()
        k += 1
    return rig


def play(rig):
    """A data plane that plays every start: 0.5 s in, 0.5 s held, then DONE (B is not under test here)."""
    seg_id, phase = rig.started(), rig.blender.phase
    if seg_id is None or phase not in (Phase.WAIT_ACK, Phase.PLAYING, Phase.WAIT_DONE):
        return
    if phase is Phase.WAIT_ACK:
        rig.play_t0 = rig.now
    t = rig.now - rig.play_t0
    state = "ENTER_B" if t < 0.5 else "HOLD" if t < 1.0 else "EXIT_B" if t < 1.5 else "DONE"
    rig.publish_port(state=state, seg_id=seg_id, seg="heart", reason="hold_max", P_actual=list(P_HEART),
                     hand_P=[0.2] * 20)  # fmt: skip


@pytest.mark.parametrize("share", [0.1, 0.15, 0.3, 0.5])  # 0.15: step 0 today ([T,T], 10-15% held)
def test_stress_new_completes_at_least_as_often_aborts_less_and_never_jumps(share):
    """The same seeds, old (every switch off) and new; prints the completion / hold curve (-s)."""
    results = {}
    for name, config in (("old", OFF), ("new", NEW)):
        done = aborts = 0
        for seed in range(3):
            rig = stress(config, seed, share)
            outcomes = rig.blender.outcomes
            done += outcomes.get("complete", 0)
            aborts += sum(n for key, n in outcomes.items() if key not in ("complete", "let_go"))
            moving = [k for k in range(1, len(rig.paused)) if not rig.paused[k - 1]]
            steps = [max(abs(a - b) for a, b in zip(rig.out[k - 1], rig.out[k], strict=True)) for k in moving]
            assert max(steps) <= (speed_limit(0.6) + 0.3 * 2 * math.pi * 0.6) * TICK * 1.1
            assert len(lines(rig, "template pause")) <= 2 * rig.blender.entries
        results[name] = (done, aborts)
    print(f"share={share} complete/aborts old={results['old']} new={results['new']}")
    assert results["new"][0] >= results["old"][0] and results["new"][1] <= results["old"][1]
