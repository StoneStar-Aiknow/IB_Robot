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
from test_template_motion_blender import (
    HAND_MIRROR,
    MIRROR,
    OFF,
    P_HEART,
    TICK,
    Rig,
    exit_reason,
    hooked,
    lines,
    max_jerkless,
    max_step,
)

from manipulation_execution.template_motion.blender import (
    A_CAP_ECHO_AGE_S,
    MIN_JERK_PEAK,
    BlendConfig,
    Phase,
    outcome_of,
)
from manipulation_execution.template_motion.port import CbfMonitor

PERIOD = 0.05  # the data plane's status
RELEASE_AFTER = 0.2  # the CBF's hold_release_after_s on site
NEW = BlendConfig()
SHORT = BlendConfig(a_pause_budget_s=0.6, a_pause_budget_per_s=0.5, a_pause_max_s=2.0)  # 43274d30's budget


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
        self.deaf = False  # the tick runs before the subscriptions: nothing is read, the statuses queue
        self.labels_muted = False  # the label publisher went silent
        self.foreign = None  # now -> another data plane's status, received after this one's every tick
        if warm:
            self.run(1.2)  # the held-share gate needs a second of statuses

    def hear(self):
        received = self.extra + self.plane.tick(self.now) + ([self.foreign(self.now)] if self.foreign else [])
        if self.deaf:
            self.extra = received
            return
        if self.labels_muted:
            self._port_now()
        else:
            super().hear()
        for message in received:
            self.cbf_monitor.observe(json.dumps(message), self.now)
        self.extra = []

    def starve(self, seconds):
        """The session is not scheduled for ``seconds`` while the data plane goes on. The first tick
        after it runs before the queued messages are read; the next one reads them all."""
        for _ in range(round(seconds / TICK) - 1):
            self.now += TICK
            self.extra += self.plane.tick(self.now)
        self.deaf = True
        try:
            self.step()
        finally:
            self.deaf = False

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


def safety_stopped(rig):
    """The template ended without a C: a safety exit (plan 3.6)."""
    exits = lines(rig, "template exit")
    return rig.blender.phase is Phase.COOLDOWN and bool(exits) and "back=safety" in exits[-1]


def the_mirror_passes(rig, seconds=0.3):
    """From the safety exit's tick on the blender sends nothing: the gate gets the mirror command, the hand
    the mirror's; no C target, ever (the outer safety path decides what reaches the robot)."""
    assert rig.out[-1] == rig.mirror
    for _ in range(round(seconds / TICK)):
        assert rig.step() is None
        assert rig.out[-1] == rig.mirror and rig.hand == HAND_MIRROR
    assert rig.blender.phase in (Phase.COOLDOWN, Phase.IDLE)


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


@pytest.mark.parametrize("speed", [0.3, 0.6, 1.0])
@pytest.mark.parametrize("hold", [0.1, 0.35])
def test_a_resume_while_the_mirror_moves_stays_within_the_acceleration_limit(speed, hold):
    # x2-wt-18 REPORT 3.1: M eases back to a moving m; r(t) m(t) adds 2 r' m', a short ramp made it 2.5x.
    rig = PauseRig()
    rig.into_a(0.4)
    rig.hold(0.0, hold)
    t0 = rig.now

    def moving():
        rig.mirror = tuple(v + (speed * (rig.now + TICK - t0) if j in (1, 8) else 0.0) for j, v in enumerate(MIRROR))

    for _ in range(round(2.0 / TICK)):
        moving()
        rig.step()
        if rig.blender.phase is not Phase.A:
            break
    (first, last), = rig.spans()
    resume = lines(rig, "template resume")[0]
    # A itself, with this mirror: up to 2 alpha' m' (alpha' at most MIN_JERK_PEAK / T_A); the mirror's own is 0.
    own = 2.0 * MIN_JERK_PEAK / 1.1 * speed
    after = rig.out[last:last + round(0.5 / TICK)]
    assert max_jerkless(after) <= (accel_limit() * 1.1 + own) * TICK * TICK, resume
    assert max_step(after) <= (speed_limit(0.6) + speed) * TICK * 1.05, resume


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


def long_pause_then_silent(rig):
    """A in a long pause (past the release), the hold over and the data plane's statuses not coming."""
    rig.into_a(0.3)
    rig.hold(0.0, 2.0)
    rig.run(0.35)
    assert rig.blender.snapshot()["paused"]
    rig.plane.holds = []
    rig.plane.delay = lambda at: None


def clean_status(rig, at, **top):
    message = rig.plane.status(at)
    message["cbf"].update(stuck_s=0.0, releasing=False, stuck_collision=False)
    message.update(top)
    return message


def tick_with(rig, *messages):
    rig.now += TICK
    rig.hear()
    for message in messages:
        rig.cbf_monitor.observe(json.dumps(message), rig.now)
    rig.blender.step(rig.now, rig.inputs(), rig.mirror)
    return rig.blender.snapshot()["paused"]


def test_a_long_pause_needs_two_individually_fresh_statuses():
    # Codex review 1120 T3: a 0.2 s old one and a fresh one resumed it.
    rig = PauseRig()
    long_pause_then_silent(rig)
    rig.run(0.3)  # silent long enough that a 0.2 s old one is newer than the last received (else: reordered)
    assert tick_with(rig, clean_status(rig, rig.now - 0.2 + TICK), clean_status(rig, rig.now + TICK))
    assert not lines(rig, "template resume")
    assert not tick_with(rig, clean_status(rig, rig.now + TICK))  # the next fresh one makes two
    assert "long=True" in lines(rig, "template resume")[0]


def test_a_counted_status_that_ages_out_before_the_second_no_longer_counts():
    rig = PauseRig()
    long_pause_then_silent(rig)
    rig.run(0.3)
    assert tick_with(rig, clean_status(rig, rig.now + TICK))
    for _ in range(round(NEW.a_status_age_s / TICK) + 1):
        assert tick_with(rig)  # no status: the newest gets old, A stays paused
    assert tick_with(rig, clean_status(rig, rig.now + TICK))  # one fresh: the first is too old to make two
    assert not tick_with(rig, clean_status(rig, rig.now + TICK))


@pytest.mark.parametrize("bad", ["duplicate", "no_epoch", "no_at"])
def test_a_duplicate_or_incomplete_status_does_not_count_towards_a_resume(bad):
    rig = PauseRig()
    long_pause_then_silent(rig)
    rig.run(0.3)
    first = clean_status(rig, rig.now + TICK)
    if bad == "duplicate":
        assert tick_with(rig, first, first)  # read twice: counts once
        assert tick_with(rig, dict(first))  # and again on the next tick
    else:
        broken = dict(first)
        del broken["safety_epoch" if bad == "no_epoch" else "at"]
        assert tick_with(rig, broken, clean_status(rig, rig.now + TICK))
        assert not lines(rig, "template exit")  # the newest is complete: no safety exit
    assert not lines(rig, "template resume")
    assert not tick_with(rig, clean_status(rig, rig.now + TICK))


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
    rig = PauseRig(SHORT)
    rig.into_a(0.4)
    rig.hold(0.0, 3.0)
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_pause_budget"
    exit = lines(rig, "template exit")[0]
    assert "outcome=cbf_hold" in exit and "pauses=1" in exit and "backoff=3.0" in exit
    budget = SHORT.a_pause_budget_s + SHORT.a_pause_budget_per_s * 1.1
    total = float(exit.split(" total=")[1].split()[0])
    assert total <= 1.1 + NEW.a_timeout_margin_s + budget
    del rig.active["heart"]
    rig.run(2.5)
    rig.active["heart"] = 7  # a new episode within the backoff
    rig.run(4.0)
    enter = lines(rig, "template enter")
    assert len(enter) == 2 and "blocked=backoff" in enter[1]


def test_pauses_add_up_and_a_resume_never_resets_the_budget():
    rig = PauseRig(SHORT)
    rig.into_a(0.2)
    rig.hold(0.0, 0.5)
    rig.hold(0.9, 0.5)
    assert rig.run(2.5, lambda: rig.blender.phase is Phase.C)
    exit = lines(rig, "template exit")[0]
    assert exit_reason(rig) == "a_pause_budget" and "pauses=2" in exit
    assert len(lines(rig, "template resume")) == 1


def test_a_ramp_the_budget_cannot_finish_is_never_begun():
    rig = PauseRig(replace(SHORT, a_resume_ramp_s=0.4))  # a resume costs 0.2 s of schedule
    rig.into_a(0.2)
    rig.hold(0.0, 0.97)  # clean after about 1.0 s, 0.15 s of budget left
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.C)
    exit = lines(rig, "template exit")[0]
    assert exit_reason(rig) == "a_pause_budget" and not lines(rig, "template resume")
    budget = SHORT.a_pause_budget_s + SHORT.a_pause_budget_per_s * 1.1
    assert float(exit.split(" paused_s=")[1].split()[0]) < budget  # left at the resume, not at the end


def test_no_status_A_can_trust_ends_A_at_the_absolute_cap():
    rig = PauseRig()
    rig.into_a(0.3)
    rig.plane.delay = lambda at: None
    t0 = rig.now
    assert rig.run(1.5, lambda: safety_stopped(rig))
    assert exit_reason(rig) == "a_cbf_untrusted"
    assert rig.now - t0 <= NEW.a_status_age_s + NEW.a_untrusted_max_s + 3 * TICK
    assert "why=cbf_late" in lines(rig, "template pause")[0]
    (first, last), = rig.spans()
    assert len(set(rig.out[first:last + 1])) == 1
    assert "outcome=untrusted" in lines(rig, "template exit")[0]
    the_mirror_passes(rig)


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


# -- ages judged on punctual ticks (the session starved) -------------------------------------

LATE = 0.06  # every tick this late: the session starved for good


def newest_status_at(rig):
    return rig.cbf_monitor.latest().at


@pytest.mark.parametrize("starved", [0.1, 0.2, 0.4, 0.6])
def test_a_starved_session_never_leaves_A_and_pauses_only_past_twice_the_status_age(starved):
    rig = PauseRig()
    rig.into_a(0.2)
    seen = newest_status_at(rig)
    rig.starve(starved)
    age = rig.now - seen  # of the newest status the late tick knows
    assert rig.blender.snapshot()["paused"] == (age > 2 * NEW.a_status_age_s)
    assert rig.run(3.0, lambda: rig.blender.phase in (Phase.WAIT_ACK, Phase.C))
    assert rig.blender.phase is Phase.WAIT_ACK and exit_reason(rig) == ""  # the label too: 0.6 < 2 x 0.5
    assert len(rig.spans()) <= 1 and not lines(rig, "template exit")


@pytest.mark.parametrize("starved", [0.2, 0.4, 0.6])
def test_a_starved_session_keeps_the_hold_in_B(starved):
    rig = PauseRig()
    to_playing_new(rig)
    rig.run(1.0)  # past min_hold: any release would be sent at once
    rig.starve(starved)
    rig.run(0.5)
    assert rig.blender.phase is Phase.PLAYING and not rig.sent("release")


@pytest.mark.parametrize("starved", [0.3, 0.6])
@pytest.mark.parametrize("answer", ["ENTER_B", "STOPPED", None])
def test_a_session_starved_right_after_the_start_reads_the_port_s_answer_before_no_ack(starved, answer):
    # The late tick runs before the statuses queued while starving: ack_timeout_s is not judged on it
    # (port_suspect would keep templates out for the rest of the round), the next tick reads them.
    rig = PauseRig()
    rig.into_a(0.0)
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.WAIT_ACK)
    seg_id = rig.started()
    if answer is not None:  # queued: seen from the tick after the late one on
        rig.port = {"state": answer, "seg_id": seg_id, "seg": "heart", "P_actual": list(P_HEART), "hand_P": [0.2] * 20,
                    "reason": "keepalive_timeout" if answer == "STOPPED" else ""}  # fmt: skip
    rig.starve(starved)
    assert rig.blender.phase is Phase.WAIT_ACK and not lines(rig, "template exit")
    assert not rig.blender.snapshot()["port_suspect"] and not rig.sent("release")
    rig.step()
    snap = rig.blender.snapshot()
    if answer == "ENTER_B":
        assert rig.blender.phase is Phase.PLAYING and not snap["port_suspect"]
    elif answer == "STOPPED":
        assert exit_reason(rig) == "stopped:keepalive_timeout" and not snap["port_suspect"]
    else:  # no answer even then: the port is suspect, as on a punctual tick
        assert exit_reason(rig) == "no_ack" and snap["port_suspect"]


@pytest.mark.parametrize("dt", [TICK, LATE])
def test_a_status_that_stopped_is_untrusted_by_twice_its_age_at_most(dt):
    rig = PauseRig()
    rig.into_a(0.3)
    rig.plane.delay = lambda at: None
    seen = newest_status_at(rig)
    while not rig.blender.snapshot()["paused"]:
        rig.step(dt)
    limit = NEW.a_status_age_s * (2 if dt > NEW.late_tick_s else 1)
    assert limit < rig.now - seen <= limit + dt + 1e-9
    paused = rig.now
    while rig.blender.phase is Phase.A:
        rig.step(dt)
    assert exit_reason(rig) == "a_cbf_untrusted" and "why=cbf_late" in lines(rig, "template pause")[0]
    assert rig.now - paused <= NEW.a_untrusted_max_s + dt + 1e-9


@pytest.mark.parametrize("dt", [TICK, LATE])
def test_a_port_status_that_stopped_in_B_is_stale_by_twice_its_limit_at_most(dt):
    rig = PauseRig()
    to_playing_new(rig)
    rig.run(0.5)
    rig.port = None
    seen = rig.now
    while not rig.sent("release"):
        rig.step(dt)
    limit = NEW.status_stale_s * (2 if dt > NEW.late_tick_s else 1)
    assert limit < rig.now - seen <= limit + dt + 1e-9
    assert rig.sent("release")[-1]["fast"]
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "status_stale"


@pytest.mark.parametrize("dt", [TICK, LATE])
def test_a_label_that_stopped_is_lost_by_twice_its_stale_s_at_most(dt):
    rig = PauseRig()
    rig.into_a(0.1)
    rig.hold(0.0, 3.0)  # A stays, paused, until the label runs out
    rig.labels_muted = True
    seen = rig.now
    while rig.blender.phase is Phase.A:
        rig.step(dt)
    limit = rig.labels._stale_s * (2 if dt > NEW.late_tick_s else 1)
    assert exit_reason(rig) == "label_lost" and limit < rig.now - seen <= limit + dt + 1e-9


def test_a_late_tick_keeps_only_the_previous_tick_s_verdict():
    rig = PauseRig()
    older = rig.blender._older
    rig.step()
    assert older("kept", 0.6, 0.5) and older("old", 0.6, 0.5)
    rig.step()
    assert older("kept", 0.6, 0.5)  # "old" is not judged on this tick
    rig.step(LATE)
    assert older("kept", 0.6, 0.5) and older("kept", 0.6, 0.5)  # twice in one tick: the same
    assert not older("old", 0.6, 0.5)  # no verdict from the tick before: starvation is no evidence
    assert older("old", 1.01, 0.5) and not older("new", 0.6, 0.5)


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
    assert rig.run(0.2, lambda: safety_stopped(rig))
    assert exit_reason(rig) == "a_epoch_changed" and rig.started() is None
    assert "outcome=safety" in lines(rig, "template exit")[0]
    the_mirror_passes(rig)
    assert rig.started() is None


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
    assert safety_stopped(rig) and exit_reason(rig) == reason
    the_mirror_passes(rig)


def test_an_epoch_change_in_a_pause_ends_it():
    rig = PauseRig()
    rig.into_a(0.3)
    rig.hold(0.0, 0.6)
    rig.run(0.15)
    assert rig.blender.snapshot()["paused"]
    rig.plane.epoch += 1
    assert rig.run(0.2, lambda: safety_stopped(rig))
    assert exit_reason(rig) == "a_epoch_changed"
    the_mirror_passes(rig)


def test_unverifiable_permission_sends_no_C_and_the_gate_gets_the_inner_command():
    # Codex review 1120 T1: mirror_active false for good; the C used to go on moving the request.
    rig = PauseRig()
    rig.into_a(0.3)
    inner, hooks = hooked(rig)
    rig.plane.mirror_active = False
    published, phases = [], []
    for _ in range(round(1.0 / TICK)):
        rig.now += TICK
        rig.hear()
        published.append(hooks.retarget(rig.inputs()))
        phases.append(rig.blender.phase)
        rig.client.pump(rig.now)
    assert safety_stopped(rig) and exit_reason(rig) == "a_cbf_untrusted"
    assert Phase.C not in phases
    stop = phases.index(Phase.COOLDOWN)
    assert stop * TICK <= NEW.a_status_age_s + NEW.a_untrusted_max_s + 3 * TICK
    assert len(set(c.arm for c in published[1:stop])) == 1  # paused until then
    assert all(c.arm == MIRROR and c.hand == (0.3,) * 20 for c in published[stop:])  # the inner's own command
    assert rig.cbf_monitor.latest().mirror_active is False


def test_a_let_go_without_a_trusted_status_waits_paused_then_goes_through_C():
    rig = PauseRig()
    rig.into_a(0.3)
    rig.plane.delay = lambda at: 0.2  # each status already too old
    assert rig.run(0.3, lambda: rig.blender.snapshot()["paused"])
    del rig.active["heart"]
    rig.run(0.1)
    assert rig.blender.phase is Phase.A and rig.blender.snapshot()["paused"]
    assert not lines(rig, "template exit")
    rig.plane.delay = lambda at: 0.0
    assert rig.run(0.5, lambda: rig.blender.phase is Phase.C)
    exit = lines(rig, "template exit")[0]
    assert "reason=label_lost" in exit and " back=c after=label_lost:" in exit and "outcome=let_go" in exit
    rig.run(2.0)
    assert rig.out[-1] == rig.mirror
    assert max_step(rig.out) <= speed_limit(0.6) * TICK * 1.05


def test_a_let_go_never_trusted_is_a_safety_exit_at_the_absolute_cap():
    rig = PauseRig()
    rig.into_a(0.3)
    rig.plane.delay = lambda at: 0.2
    assert rig.run(0.3, lambda: rig.blender.snapshot()["paused"])
    untrusted_at = rig.now
    rig.run(0.2)
    del rig.active["heart"]
    assert rig.run(1.0, lambda: safety_stopped(rig))
    exit = lines(rig, "template exit")[0]
    assert "reason=a_cbf_untrusted" in exit and " after=label_lost:" in exit
    assert rig.now - untrusted_at <= NEW.a_untrusted_max_s + 3 * TICK  # not restarted by the let-go
    the_mirror_passes(rig)


@pytest.mark.parametrize("same_status", [True, False])
def test_a_collision_and_an_epoch_change_in_one_batch_send_no_C(same_status):
    rig = PauseRig()
    rig.into_a(0.3)
    plane = rig.plane
    hit = plane.status(rig.now)
    hit["cbf"].update(stuck_collision=True)
    if same_status:
        rig.extra = [dict(hit, safety_epoch=4)]
    else:
        rig.extra = [hit, dict(plane.status(rig.now), safety_epoch=4)]
    plane.epoch = 4
    rig.step()
    assert safety_stopped(rig) and exit_reason(rig) == "a_epoch_changed"
    assert same_status or " after=a_cbf_collision:" in lines(rig, "template exit")[0]
    the_mirror_passes(rig)


@pytest.mark.parametrize("what", ["collision", "epoch", "nulls"])
def test_an_exit_line_describes_the_status_that_ended_A_and_its_three_ages(what):
    # Codex review 1120 5.2: the raw cbf.reason, arm_age_s and at of the status the exit was read in,
    # not the batch's newest; receipt, status and feedback ages apart; a missing field is unknown, never 0.
    rig = PauseRig()
    rig.into_a(0.3)
    hit = rig.plane.status(rig.now)
    hit["cbf"].update(reason="geometry", arm_age_s=0.042)
    if what == "epoch":
        hit["safety_epoch"] = rig.plane.epoch = 4
    else:
        hit["cbf"]["stuck_collision"] = True
    if what == "nulls":
        del hit["cbf"]["reason"]
        hit["cbf"]["arm_age_s"] = None  # what the data plane sends without feedback
    newest = rig.plane.status(rig.now + TICK / 2)
    newest["cbf"].update(reason="passed", arm_age_s=0.011)
    if what == "epoch":
        newest["safety_epoch"] = 4
    rig.extra = [hit, newest]
    rig.step()
    line = lines(rig, "template exit")[0]
    assert exit_reason(rig) == ("a_epoch_changed" if what == "epoch" else "a_cbf_collision")
    assert "of=trigger recv_age=0.000 status_age=0.020 " in line and f"at={hit['at']:.3f} " in line
    if what == "nulls":
        assert "reason=unknown" in line and "arm_age=unknown" in line
    else:
        assert "reason=geometry" in line and "arm_age=0.042" in line
    assert "reason=passed" not in line and "arm_age=0.011" not in line


def test_a_restarted_data_plane_with_the_same_epoch_is_a_safety_exit():
    rig = PauseRig()
    rig.into_a(0.3)
    rig.plane.calls = rig.plane.held = 0  # counters begin again; the epoch number happens to be equal
    assert rig.run(0.2, lambda: safety_stopped(rig))
    assert exit_reason(rig) == "a_epoch_reset" and "outcome=safety" in lines(rig, "template exit")[0]
    the_mirror_passes(rig)


@pytest.mark.parametrize("what", ["epoch", "owner", "silent"])
def test_a_checked_C_stops_on_what_would_be_a_safety_exit_in_A(what):
    rig = PauseRig()
    rig.into_a(1.0)  # near P: a full-length C
    del rig.active["heart"]
    assert rig.run(0.2, lambda: rig.blender.phase is Phase.C)
    assert " back=c" in lines(rig, "template exit")[0]
    rig.run(0.1)
    if what == "epoch":
        rig.plane.epoch += 1
    elif what == "owner":
        rig.plane.owner = 2
    else:
        rig.plane.delay = lambda at: None
    assert rig.run(0.7, lambda: safety_stopped(rig))
    reason = {"epoch": "a_epoch_changed", "owner": "a_owner_changed", "silent": "a_cbf_untrusted"}[what]
    exit = lines(rig, "template exit")[-1]
    assert f"reason={reason}" in exit and "outcome=repeat" in exit and "phase=C" in exit
    the_mirror_passes(rig)


def via_hooks(rig, seconds):
    """Ticks through TemplateBlendHooks: what it publishes, and the phase after each tick."""
    inner, hooks = hooked(rig)
    published, phases = [], []
    for _ in range(round(seconds / TICK)):
        rig.now += TICK
        rig.hear()
        published.append(hooks.retarget(rig.inputs()))
        phases.append(rig.blender.phase)
        rig.client.pump(rig.now)
    return published, phases


def port_says(rig, seg_id, state, reason="hold_max"):
    rig.publish_port(state=state, seg_id=seg_id, seg="heart", reason=reason, P_actual=list(P_HEART), hand_P=[0.2] * 20)


# T1b: the data plane STOPPED it (its reasons, or one not known here): no C, its CBF path goes on from P_actual.
@pytest.mark.parametrize("why", ["epoch_changed", "revoked", "named_arm", "control_lapsed", "internal_error", "new"])
@pytest.mark.parametrize("phase", [Phase.PLAYING, Phase.WAIT_DONE])
def test_a_playback_the_data_plane_stopped_sends_no_C(why, phase):
    rig = PauseRig()
    seg_id = to_playing_new(rig)
    if phase is Phase.WAIT_DONE:
        del rig.active["heart"]
        assert rig.run(1.5, lambda: rig.blender.phase is Phase.WAIT_DONE)
    port_says(rig, seg_id, "STOPPED", why)
    published, phases = via_hooks(rig, 0.3)
    assert Phase.C not in phases and safety_stopped(rig)
    assert exit_reason(rig) == f"stopped:{why}" and outcome_of(exit_reason(rig)) == "safety"
    assert all(c.arm == MIRROR and c.hand == (0.3,) * 20 for c in published)  # the inner's own command


def test_off_a_stopped_playback_still_goes_back():
    rig = PauseRig(OFF)
    seg_id = to_playing_new(rig)
    port_says(rig, seg_id, "STOPPED", "revoked")
    rig.step()
    assert rig.blender.phase is Phase.C and exit_reason(rig) == "stopped:revoked"


@pytest.mark.parametrize("state", ["DONE", "REJECTED"])
def test_done_or_rejected_goes_through_C_on_a_trusted_status(state):
    rig = PauseRig()
    seg_id = to_playing_new(rig) if state == "DONE" else None
    if seg_id is None:
        rig.active["heart"] = 7
        assert rig.run(4.0, lambda: rig.blender.phase is Phase.WAIT_ACK)
        seg_id = rig.started()
    port_says(rig, seg_id, state)
    rig.step()
    assert rig.blender.phase is Phase.C
    assert " back=c" in lines(rig, "template exit")[-1] and " after=" not in lines(rig, "template exit")[-1]


def test_done_while_the_statuses_are_late_holds_P_actual_then_goes_through_C():
    rig = PauseRig()
    seg_id = to_playing_new(rig)
    rig.plane.delay = lambda at: 0.2
    rig.run(0.2)
    port_says(rig, seg_id, "DONE")
    rig.run(0.3)
    assert rig.blender.phase is Phase.PLAYING and not lines(rig, "template exit")
    assert rig.out[-15:] == [P_HEART] * 15
    rig.plane.delay = lambda at: 0.0
    assert rig.run(0.3, lambda: rig.blender.phase is Phase.C)
    exit = lines(rig, "template exit")[-1]
    assert "reason=done:hold_max" in exit and " back=c after=done:hold_max:" in exit
    rig.run(2.5)
    assert rig.out[-1] == rig.mirror


def test_done_never_trusted_is_a_safety_exit_after_a_untrusted_max_s():
    rig = PauseRig()
    seg_id = to_playing_new(rig)
    rig.plane.delay = lambda at: None
    rig.run(0.2)
    port_says(rig, seg_id, "DONE")
    published, phases = via_hooks(rig, 1.0)
    stop = phases.index(Phase.COOLDOWN)
    assert Phase.C not in phases and NEW.a_untrusted_max_s <= stop * TICK <= NEW.a_untrusted_max_s + 3 * TICK
    exit = lines(rig, "template exit")[-1]
    assert "reason=a_cbf_untrusted" in exit and " back=safety after=done:hold_max:0.5" in exit
    assert all(c.arm == P_HEART for c in published[:stop])
    assert all(c.arm == MIRROR and c.hand == (0.3,) * 20 for c in published[stop:])


@pytest.mark.parametrize("what", ["epoch", "owner"])
def test_a_safety_change_during_B_then_done_sends_no_C(what):
    rig = PauseRig()
    seg_id = to_playing_new(rig)
    if what == "epoch":
        rig.plane.epoch += 1
    else:
        rig.plane.owner = 2
    rig.run(0.1)
    assert rig.blender.phase is Phase.PLAYING  # B is the data plane's: it stops it itself
    port_says(rig, seg_id, "DONE")
    rig.step()
    assert safety_stopped(rig) and exit_reason(rig) == f"a_{what}_changed"
    assert " back=safety after=done:hold_max:0.00" in lines(rig, "template exit")[-1]
    the_mirror_passes(rig)


def test_a_round_end_while_C_waits_for_trust_wraps_up_as_before():
    rig = PauseRig()
    seg_id = to_playing_new(rig)
    rig.plane.delay = lambda at: 0.2
    rig.run(0.2)
    port_says(rig, seg_id, "DONE")
    rig.run(0.1)
    rig.blender.end_round(rig.now)
    assert rig.blender.phase is Phase.STOP_WAIT and exit_reason(rig) == "stop"
    rig.run(0.6)
    assert rig.blender.phase is not Phase.C and "a_cbf_untrusted" not in "".join(lines(rig, "template exit"))


def test_no_safety_epoch_no_entry_and_in_A_a_safety_exit():
    # Without one a status is not this data plane's (CbfMonitor drops it): no status at all.
    rig = PauseRig()
    rig.plane.with_epoch = False
    rig.run(0.6)
    rig.active["heart"] = 7
    rig.run(1.0)
    assert rig.blender.phase is Phase.IDLE
    del rig.active["heart"]
    rig.run(0.1)
    assert "reason=cbf_stale" in lines(rig, "template reject")[0]
    rig.plane.with_epoch = True
    rig.run(2.5)
    rig.into_a(0.3)
    rig.plane.with_epoch = False
    assert rig.run(0.2, lambda: rig.blender.snapshot()["paused"])  # untrusted from a_status_age_s on
    assert rig.run(NEW.a_untrusted_max_s + 0.1, lambda: safety_stopped(rig))
    assert exit_reason(rig) == "a_cbf_untrusted"
    the_mirror_passes(rig)


def old_kit(offset):
    """A colleague's older data plane on the same ROS network (real robot 11-20 17:01): no cbf section,
    no safety_epoch, no arm_owner; ``at`` on its own machine's clock (None: none sent)."""

    def status(now):
        message = {"control_seq": 5, "groups": {}, "faults": {}, "rejected": []}
        if offset is not None:
            message["at"] = now + offset
        return message

    return status


@pytest.mark.parametrize("offset", [None, 0.0, 1000.0, -1000.0])
def test_an_old_data_plane_on_the_network_is_no_evidence_and_no_exit(offset):
    rig = PauseRig()
    rig.plane.every = 2  # half the CBF calls held: the held-share gate is closed
    rig.run(2.5)
    rig.foreign = old_kit(offset)  # newest at every tick: its "CBF off" would read held=0
    rig.active["heart"] = 7
    rig.run(1.0)
    assert rig.blender.phase is Phase.IDLE and rig.blender._gate_ratio == pytest.approx(0.5, abs=0.05)
    assert all("reason=hold_ratio" in line for line in lines(rig, "template reject")), lines(rig, "template reject")
    del rig.active["heart"]
    rig.plane.every = None
    rig.run(3.0)
    rig.active["heart"] = 7
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.A)  # this plane's statuses still count
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.WAIT_ACK), lines(rig, "template exit")
    assert not lines(rig, "template pause") and rig.cbf_monitor.foreign > 0


# -- entry ---------------------------------------------------------------------------------


def test_the_held_share_gate_has_hysteresis():
    rig = PauseRig(replace(NEW, enter_max_wait_s=0.0))  # waits for ever: the gate alone
    rig.plane.every = 2  # half the calls held, one call at a time: never a stuck status
    rig.run(2.5)
    assert rig.blender.snapshot()["gate"]["open"] is False
    assert "template gate closed" in rig.logs[-1] or lines(rig, "template gate closed")
    rig.active["heart"] = 7
    rig.plane.every = 5  # 20%: between open (0.15) and close (0.3)
    rig.run(3.0)
    assert rig.blender.phase is Phase.IDLE
    rig.plane.every = None
    assert rig.run(2.0, lambda: rig.blender.phase is Phase.A)
    assert lines(rig, "template gate open")
    assert "blocked=hold_ratio" in lines(rig, "template enter")[0]


def test_a_trigger_kept_out_enter_max_wait_is_rejected_never_entered_late():
    rig = PauseRig()
    rig.plane.every = 2
    rig.run(2.5)
    rig.active["heart"] = 7  # seen from the next tick; the wait starts there
    rig.plane.every = 5
    rig.run(2.0)
    reject = lines(rig, "template reject")
    assert len(reject) == 1 and "reason=hold_ratio" in reject[0] and "at=enter_max_wait" in reject[0]
    assert NEW.enter_max_wait_s <= float(reject[0].split(" waited=")[1].split()[0]) < NEW.enter_max_wait_s + 0.03
    rig.plane.every = None
    assert rig.run(2.0, lambda: lines(rig, "template gate open"))
    rig.run(0.5)
    assert rig.blender.phase is Phase.IDLE and not lines(rig, "template enter")  # that episode is over
    del rig.active["heart"]
    rig.run(0.3)
    rig.active["heart"] = 7  # the person asks again
    assert rig.run(0.5, lambda: rig.blender.phase is Phase.A)
    assert "waited=" not in lines(rig, "template enter")[0]


def test_a_trigger_entered_within_enter_max_wait_says_how_long_it_waited():
    rig = PauseRig()
    rig.hold(0.0, 0.5)  # 25% of the window at most: the share gate stays open
    rig.run(0.1)  # the status saying so is in
    rig.active["heart"] = 7
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.A)
    enter = lines(rig, "template enter")[0]
    assert "blocked=cbf_hold" in enter and 0.35 <= float(enter.split(" waited=")[1].split()[0]) < 0.6
    assert not lines(rig, "template reject")


@pytest.mark.parametrize("clears", [-TICK, 0.0, TICK])
def test_a_block_that_clears_at_enter_max_wait_never_enters_late(clears):
    # Codex review 1120 T2: the block cleared at 1.52 s and the trigger still went into A.
    rig = PauseRig()
    wait = NEW.enter_max_wait_s
    rig.active["heart"] = 7
    rig.publish_enabled = False
    rig.step()
    first = rig.now
    last_blocked = first + wait + min(clears, 0.0) - TICK  # after: a late tick jumps the threshold (Codex's 1.52)
    while rig.now + TICK <= last_blocked + 1e-9:
        rig.step()
    assert rig.blender.phase is Phase.IDLE and not lines(rig, "template reject")
    rig.publish_enabled = True
    rig.now = first + wait + clears - TICK  # exactly: the next tick is the one the block clears on
    rig.step()
    assert rig.now - first == wait if clears == 0 else math.isclose(rig.now - first, wait + clears)
    if clears < 0:
        enter = lines(rig, "template enter")
        assert rig.blender.phase is Phase.A and len(enter) == 1 and "blocked=publish_off" in enter[0]
        return
    assert rig.blender.phase is Phase.IDLE and not lines(rig, "template enter")
    reject = lines(rig, "template reject")
    assert len(reject) == 1 and "reason=publish_off" in reject[0] and "at=enter_max_wait" in reject[0]
    rig.run(0.5)
    assert rig.blender.phase is Phase.IDLE and not lines(rig, "template enter")  # that episode is over


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
    monitor.observe(json.dumps({"at": 11.55, "safety_epoch": 1}), 11.55)
    assert monitor.hold_ratio(11.55, 2.0) == 0.0  # the CBF is off
    monitor.observe(cbf_message(11.6, 0, 3), 11.6)
    monitor.observe(json.dumps({"at": 11.65}), 11.65)  # another data plane's: no epoch
    assert monitor.hold_ratio(11.65, 2.0) is None and monitor.foreign == 1


@pytest.mark.parametrize(
    "raw",
    [{"b_label_unlatch": 1}, {"enter_hold_ratio_open": 1.5}, {"enter_hold_ratio_open": 0.35},
     {"a_status_age_s": 0.2}, {"a_untrusted_max_s": 0}, {"a_pause_typo": 1.0}],
)  # fmt: skip
def test_v12_keys_are_checked(raw):
    with pytest.raises(ValueError):
        BlendConfig.from_mapping(raw)


def test_the_off_values_are_accepted():
    raw = {"a_pause_budget_s": 0, "enter_hold_window_s": 0, "load_backoff_s": 0, "b_label_unlatch": False,
           "late_tick_s": 0}
    assert BlendConfig.from_mapping(raw) == OFF


# -- a fixed-seed stress: the same events, old and new -------------------------------------------


def stress(config, seed, share, seconds=40.0, rig_type=None):
    """Holds of 0.04-0.4 s (clustered) covering about ``share`` of the time, statuses sometimes late,
    lost or batched, a mirror that moves; the label held 5 s at a time. Returns the counts and the
    output, checked for continuity outside pause ticks."""
    rng = random.Random(seed)
    rig = (rig_type or PauseRig)(config=config)
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


@pytest.mark.parametrize("share", [0.05, 0.1, 0.15])  # 0.15: step 0 today ([T,T], 8-15% held)
def test_stress_up_to_today_s_load_never_gives_up_an_entered_template(share):
    """zll: once entered, no abort but for safety; the triggers kept out are few at 5% (bb: <= 2%)."""
    rejects = triggers = 0
    for seed in range(10):
        rig = stress(NEW, seed, share)
        assert not {key for key in rig.blender.outcomes if key not in ("complete", "let_go")}, rig.blender.outcomes
        rejects += len(lines(rig, "template reject"))
        triggers += rig.blender.triggers
    if share <= 0.05:
        assert rejects <= 0.02 * triggers


@pytest.mark.parametrize("share", [0.1, 0.15, 0.3, 0.5])  # 0.15: step 0 today ([T,T], 8-15% held)
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


# -- (b) a_cap: the session's side, against a data plane that echoes it ----------------------------

CAP = replace(NEW, a_cap=True)
CAP_EXPIRY = 0.3  # the data plane's: no refresh this long and the flag is gone


class CapRig(PauseRig):
    """PauseRig with a data plane that takes a_cap and echoes it in the port status as e8cee8c7 does
    (M0 revision 3: a_cap, and a_cap_epoch / a_cap_attempt / a_cap_age_s of the held message, null
    while the flag is not in force). ``latency`` is the round trip from the session's message to its
    echo. ``echo``: "plane" as the data plane would, None leaves the fields out, False forces it false;
    ``tamper`` edits the echo of a held flag."""

    def __init__(self, config=CAP, latency=0.0, **kw):
        self.latency = latency
        self.echo = "plane"
        self.flags = []  # (effective at, on, epoch, attempt)
        self.read = 0
        self.tamper = lambda echo: echo
        super().__init__(config, **kw)

    def step(self, dt=TICK):
        arm = super().step(dt)
        for raw in self.messages[self.read:]:
            message = json.loads(raw)
            if message["op"] == "a_cap":
                self.flags.append((self.now + self.latency, message["on"], message["epoch"], message["attempt"]))
        self.read = len(self.messages)
        return arm

    def plane_flag(self):
        sent = [flag for flag in self.flags if flag[0] <= self.now + 1e-9]
        none = {"a_cap": False, "a_cap_epoch": None, "a_cap_attempt": None, "a_cap_age_s": None}
        if not sent:
            return none
        at, on, epoch, attempt = sent[-1]
        if not (on and epoch == self.plane.epoch and self.now - at <= CAP_EXPIRY):
            return none
        echo = {"a_cap": True, "a_cap_epoch": epoch, "a_cap_attempt": attempt, "a_cap_age_s": self.now - at}
        return self.tamper(echo)

    def _port_now(self):
        if self.port is None:
            return
        port = dict(self.port)
        if self.echo == "plane":
            port.update(self.plane_flag())
        elif self.echo is not None:
            port["a_cap"] = self.echo
        self.port_monitor.observe(json.dumps({"schema": "x2_template_status/v1", **port}), self.now)


def caps(rig):
    return [(m["on"], m["epoch"]) for m in map(json.loads, rig.messages) if m["op"] == "a_cap"]


def a_cap_wait(rig):
    return [float(line.split(" a_cap_wait=")[1].split()[0]) for line in lines(rig, "template exit")]


def a_waits_at_0(rig, seconds):
    """A, with the flag not confirmed: its clock stays at 0 and the gate gets the mirror command."""
    for _ in range(round(seconds / TICK)):
        rig.step()
        assert rig.blender.phase is Phase.A and rig.blender._clock.at(rig.now) == 0.0
        assert rig.out[-1] == rig.mirror


def test_a_cap_off_sends_nothing_and_needs_no_echo():
    rig = CapRig(NEW, latency=0.1)
    rig.echo = None
    to_playing_new(rig)
    assert not caps(rig) and "a_cap" not in rig.client.snapshot()


def test_a_cap_flags_every_A_tick_in_its_epoch_and_ends_with_one_off_before_the_start():
    rig = CapRig(latency=0.1)
    rig.active["heart"] = 7
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.A)
    ticks = 1
    while rig.blender.phase is Phase.A:
        rig.step()
        ticks += rig.blender.phase is Phase.A
    assert rig.blender.phase is Phase.WAIT_ACK
    assert caps(rig) == [(True, 3)] * ticks + [(False, 3)]
    ops = [json.loads(m)["op"] for m in rig.messages]
    assert ops.index("start") > len(ops) - 1 - ops[::-1].index("a_cap")  # the off went first
    rig.run(1.0)
    assert len(caps(rig)) == ticks + 1


def test_a_cap_no_echo_field_no_entry():
    rig = CapRig()
    rig.echo = None
    rig.active["heart"] = 7
    rig.run(2.0)
    assert rig.blender.entries == 0 and rig.blender.rejected == {"a_cap_missing": 1} and not caps(rig)


def test_a_cap_echo_lost_midway_goes_back_through_C_without_backoff():
    rig = CapRig(latency=0.1)
    rig.into_a(0.3)
    rig.echo = False
    lost = rig.now
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_cap_lost"
    assert CAP.a_cap_lost_s - 1e-9 <= rig.now - lost <= CAP.a_cap_lost_s + 2 * TICK
    exit = lines(rig, "template exit")[0]
    assert "outcome=port" in exit and "backoff=" not in exit
    assert caps(rig)[-1] == (False, 3) and caps(rig).count((False, 3)) == 1
    out = rig.out[-3:]
    assert max_step(out) <= speed_limit(0.6) * TICK  # through C: no jump


@pytest.mark.parametrize("latency", [0.0, 0.25, 0.4])
def test_a_cap_echo_late_at_entry_is_waited_for_without_moving(latency):
    # Codex review 1120 D1: the arm requests come first, the on (or its echo) 0.25 / 0.4 s late.
    rig = CapRig(latency=latency)
    rig.mirror = MIRROR
    rig.active["heart"] = 7
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.A)
    a_waits_at_0(rig, latency)
    assert rig.run(4.0, lambda: rig.blender.phase in (Phase.WAIT_ACK, Phase.C))
    assert rig.blender.phase is Phase.WAIT_ACK
    rig.run(3.0)
    assert latency <= a_cap_wait(rig)[0] <= latency + 2 * TICK
    assert max_step(rig.out) <= speed_limit(0.6) * TICK  # A starts from rest, as it always did


def test_a_cap_a_pause_before_the_first_echo_does_not_start_A():
    rig = CapRig(latency=0.4)
    rig.mirror = MIRROR
    rig.active["heart"] = 7
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.A)
    rig.hold(0.0, 0.06)  # paused and resumed while the flag is not confirmed
    for _ in range(round(0.36 / TICK)):
        rig.step()
        assert rig.blender._clock.at(rig.now) == 0.0
    assert lines(rig, "template pause") and lines(rig, "template resume")
    assert rig.run(0.2, lambda: rig.blender._clock.at(rig.now) > 0.0)
    assert rig.run(4.0, lambda: rig.blender.phase is Phase.WAIT_ACK)


@pytest.mark.parametrize("echo", ["false", "old_attempt", "old_epoch", "old_message", "no_age", "no_attempt"])
def test_a_cap_never_echoed_for_this_A_ends_it_after_the_confirm_window_without_moving(echo):
    # D1: the on lost while the arm requests go on; an echo of an earlier A's flag, of an earlier
    # epoch's, or of a message the data plane got too long ago, confirms nothing.
    rig = CapRig()
    rig.mirror = MIRROR
    if echo == "false":
        rig.echo = False  # an old data plane cannot send the field; one that sends it but never holds the flag
    else:
        key, value = {
            "old_attempt": ("a_cap_attempt", lambda e: e["a_cap_attempt"] - 1),
            "old_epoch": ("a_cap_epoch", lambda e: e["a_cap_epoch"] - 1),
            "old_message": ("a_cap_age_s", lambda e: A_CAP_ECHO_AGE_S + TICK),
            "no_age": ("a_cap_age_s", lambda e: None),
            "no_attempt": ("a_cap_attempt", lambda e: None),  # a data plane that drops the attempt
        }[echo]
        rig.tamper = lambda e: dict(e, **{key: value(e)})
    rig.active["heart"] = 7
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.A)
    entered = rig.now
    a_waits_at_0(rig, CAP.a_cap_confirm_s - 2 * TICK)
    assert rig.run(1.0, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_cap_lost" and rig.now - entered == pytest.approx(CAP.a_cap_confirm_s, abs=TICK)
    assert "a_cap_wait=unknown" in lines(rig, "template exit")[0]
    assert rig.out[-1] == rig.mirror  # C from alpha 0 at rest: the mirror


def test_a_cap_each_A_is_a_new_attempt_and_waits_for_its_own_echo():
    rig = CapRig(latency=0.1)
    for _ in range(2):
        rig.active["heart"] = 7
        for _ in range(round(4.0 / TICK)):  # A, B (played), C
            play(rig)
            rig.step()
        del rig.active["heart"]
        rig.run(1.0)  # cooldown
    attempts = [m["attempt"] for m in map(json.loads, rig.messages) if m["op"] == "a_cap"]
    assert attempts[0] == 1 and attempts[-1] == 2 and attempts == sorted(attempts)
    assert rig.blender.entries == 2 and all(0.1 <= wait <= 0.1 + 2 * TICK for wait in a_cap_wait(rig))


def test_a_cap_an_epoch_change_takes_the_epoch_exit():
    rig = CapRig(latency=0.1)
    rig.into_a(0.3)
    rig.plane.epoch += 1  # the flag is no longer valid either: its epoch is the old one
    assert rig.run(1.0, lambda: safety_stopped(rig))
    assert exit_reason(rig) == "a_epoch_changed"
    assert caps(rig)[-1] == (False, 3)


@pytest.mark.parametrize("starved", [0.2, 0.4, 0.6])
def test_a_cap_a_starved_session_waits_for_the_flag_again(starved):
    rig = CapRig(latency=0.2)
    rig.into_a(0.2)
    rig.starve(starved)  # past 0.3 the data plane dropped the flag; the echo comes back 0.2 s after the next on
    assert rig.run(3.0, lambda: rig.blender.phase in (Phase.WAIT_ACK, Phase.C))
    assert rig.blender.phase is Phase.WAIT_ACK


def test_a_cap_a_silent_port_status_counts_as_no_flag():
    rig = CapRig(latency=0.1)
    rig.into_a(0.2)
    rig.port = None
    silent = rig.now
    assert rig.run(1.5, lambda: rig.blender.phase is Phase.C)
    assert exit_reason(rig) == "a_cap_lost"
    assert rig.now - silent == pytest.approx(A_CAP_ECHO_AGE_S + CAP.a_cap_lost_s, abs=2 * TICK)


def test_a_cap_needs_the_safety_epoch():
    with pytest.raises(ValueError):
        BlendConfig.from_mapping({"a_cap": True, "a_pause_budget_s": 0})
    assert BlendConfig.from_mapping({"a_cap": True}).a_cap


@pytest.mark.parametrize("share", [0.05, 0.15])
def test_stress_a_cap_with_a_working_echo_changes_nothing(share):
    """The data plane echoes 0.1 s after the message: the same entries and completions as a_cap off; each
    A only starts that much later (D1)."""
    for seed in range(3):
        off, on = stress(NEW, seed, share), stress(CAP, seed, share, rig_type=lambda config: CapRig(config, 0.1))
        assert on.blender.outcomes == off.blender.outcomes and on.blender.rejected == off.blender.rejected
        waits = a_cap_wait(on)
        assert waits and all(0.1 <= wait <= 0.1 + 2 * TICK for wait in waits)
