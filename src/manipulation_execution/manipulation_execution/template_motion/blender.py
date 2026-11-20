"""The template state machine: when to leave the mirror, the blends, the playback port. No ROS.

One template per label activation, from the mirror and back to it::

    IDLE --label--> A --at P--> WAIT_ACK --ack--> PLAYING --DONE--> C --> COOLDOWN --> IDLE
                    |            |                  |  (release)            ^
                    +------------+------------------+-> WAIT_DONE ----------+
                         any exit goes to C, which always ends in the mirror

* A: q = m + alpha (P - m), alpha a min-jerk over ``T_A`` (as long as the
  farthest joint needs at ``a_peak_speed``, never shorter than the hand's
  travel to hand_P plus a margin, nor ``a_min_s``; no upper bound, zll
  round 4). The hand's target is hand_P from the first tick, so it gets
  there first. ``m`` is the live mirror command, so the arm keeps following
  the person while it blends.
* At P (the gate's last command within ``arrive_fraction`` of the segment's
  tolerances, arm and hand), ``start``; the data plane plays P -> T -> P on
  its own and the session keeps sending P_actual / hand_P meanwhile.
* C: q = m + beta (anchor - m), beta a quintic from the value, speed and
  acceleration it had down to 0, so leaving A half way is as smooth as
  leaving after the playback; C is stretched until no joint is faster than
  ``c_peak_speed`` nor accelerates harder than ``c_peak_accel`` (or than A
  did when it was left).

Nothing here waits without a deadline: A has ``T_A + a_timeout_margin_s``,
WAIT_ACK ``ack_timeout_s`` (one tick more when it runs out on a late one), the playback ``t_in + hold_max_s + t_out +
play_margin_s`` from its start, a stale-status check and a port-lost check
(fresh statuses that are not this segment's for ``port_lost_s``: the data
plane restarted, or plays something else), WAIT_DONE ``t_out +
done_margin_s``. Every exit goes through C back to the mirror, and every exit
is logged with its reason and the time since the template was entered.

Counting (logs only): a trigger is a label episode of the session's actor
that was not entered yet (nor taken over by the running template); it ends
admitted (``template enter``) or rejected (``template reject``, with the last
reason that kept it out). Every attempt, from entering A, ends in one outcome
(``outcome=`` in its first ``template exit`` line): complete, the person let
go before B, or an abort by reason. Pauses and resumes stay in their attempt
(``pauses=`` / ``paused_s=`` in the exit line).

v1.2 (``X2-TEMPLATE-ROUND1-DESIGN-c5.md``, "v1.2 改动"; each part has an off
value in :class:`BlendConfig` that restores the behaviour above):

* A pauses on a non-collision CBF hold, or while the data plane's newest
  status is older than ``a_status_age_s`` by its own ``at``: the arm request,
  the hand target, A's clock and the mirror reference M all stand still, so
  what the CBF releases after ``hold_release_after_s`` is a still pose. A
  resumes on clean, recent statuses of the same safety epoch, easing M and
  the clock back in over a ramp as long as the pause needs (short holds
  cost little). A leaves for C on a collision, a let-go, or once it is
  ``a_pause_budget_s`` (+ per second of T_A) behind its schedule; C blends
  from the same M, so leaving mid-pause or mid-ramp is continuous.
* Safety exits send no C (plan 3.6): a safety epoch or arm owner change, a
  restarted data plane, no epoch, or ``a_untrusted_max_s`` without a status
  it can trust end the template at once; the mirror command passes unchanged
  and the existing safety path decides what reaches the robot. Any other exit
  goes to C only on a trusted status of A's epoch and owner, waiting for one
  paused (at most ``a_untrusted_max_s``, then a safety exit); that C stops
  the same way on what would be a safety exit in A.
* Entry needs the safety epoch, a recent status, a held share of CBF calls
  below a gate with hysteresis, and no load exit within ``load_backoff_s``;
  a trigger kept out ``enter_max_wait_s`` is rejected, never entered late.
* B is not latched: a label that comes back (also as a new episode of the
  same person's gesture) keeps the hold; a release once sent is never taken
  back.
* Ages are judged on punctual ticks: a tick more than ``late_tick_s`` after
  the previous one (the session was starved, and what it has not read yet
  may be queued) keeps the previous verdict on the A status age, the B port
  status and the labels, up to twice their limit, never beyond. An ack
  timeout that runs out on a late tick is judged on the next one.
* With ``a_cap`` (off by default, until the data plane has (b)): every A tick
  asks the data plane, in A's safety epoch, to aim the CBF's releases of a
  hold at the request of its first flagged tick; ``on: false`` once when A
  ends. The port status echoes ``a_cap``: no field, no entry
  (``a_cap_missing``). Each entry into A is an attempt (from 1, never
  reused), sent with the flag; an echo counts only with A's epoch, this
  attempt, ``a_cap_age_s`` within A_CAP_ECHO_AGE_S and received in A
  (review 1120 D1). A does not move before one: its clock waits at 0, the
  output is the mirror. No such echo ``a_cap_confirm_s`` after entering (or
  after a late tick: the flag may have expired meanwhile), or none
  ``a_cap_lost_s`` in a row after one, goes back through C (``a_cap_lost``,
  no backoff). An owner or epoch change takes the existing exits first.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum

from manipulation_execution.mirror_session.endpoints import ARM_JOINT_COUNT
from manipulation_execution.template_motion.labels import Label, LabelBook
from manipulation_execution.template_motion.port import PLAYING, TERMINAL, CbfMonitor, PlayClient, PortMonitor
from manipulation_execution.template_motion.registry import MIN_JERK_PEAK, Segment

C_STRETCH = 1.25  # C is made this much longer until its peaks are within c_peak_speed / c_peak_accel
C_STRETCH_STEPS = 12
MIRROR_SPEED_SPAN_S = 0.1  # m' for a resume is taken over this much of the mirror command
PERSON_SOURCE = "yolox"  # a PEAR frame of a detected person; center_fallback never enters a template
DEFAULT_HAND_TRAVEL_S = 1.0  # T_A's hand share when the hand output is not known yet
COMPLETE = ("label_inactive", "hold_max", "release")  # the playback ran and the port said DONE
LET_GO = ("label_lost", "actor_changed")  # the person stopped asking before B
LOAD_EXITS = ("a_cbf_hold", "a_pause_budget", "a_cbf_untrusted")  # load_backoff_s follows these (and paused a_timeout)
# v1.2: no C after these: the hand-back cannot be checked, the existing safety path has the gate (plan 3.6)
SAFETY_EXITS = ("a_epoch_changed", "a_epoch_missing", "a_epoch_reset", "a_owner_changed", "a_cbf_untrusted")
A_CAP_ECHO_AGE_S = 0.3  # an a_cap echo older than this tells nothing (sent on change and every 0.1 s)
MIN_JERK_ACCEL = 5.7735  # a min-jerk's peak d2/ds2 (10 / sqrt 3)
RELEASE_AFTER_S = 0.2  # the CBF's hold_release_after_s when its status does not say (on site 0.2)
CLOCK_AHEAD_S = 0.05  # a status made this far after it was received: the two clocks differ
CLOCK_BEHIND_S = 1.0  # or received this long after it was made
PAUSE_LOG_LINES = 2  # pause / resume lines per attempt; the rest are counted in the exit line


def outcome_of(reason: str) -> str:
    """complete, let_go, or the abort's kind: collision, cbf_hold, untrusted, timeout, port, safety, stop, error."""
    if reason in COMPLETE:
        return "complete"
    if reason in LET_GO:
        return "let_go"
    if reason.startswith("error"):
        return "error"
    if reason.startswith(("stopped", "safety", "a_epoch", "a_owner")):
        return "safety"
    if reason == "stop":
        return "stop"
    if reason == "a_cbf_collision":
        return "collision"
    if reason == "a_cbf_untrusted":
        return "untrusted"
    if reason.startswith("a_cbf") or reason == "a_pause_budget":
        return "cbf_hold"
    if reason == "a_timeout":
        return "timeout"
    return "port"


class Phase(str, Enum):
    IDLE = "IDLE"
    A = "A"
    WAIT_ACK = "WAIT_ACK"
    PLAYING = "PLAYING"
    WAIT_DONE = "WAIT_DONE"
    C = "C"
    COOLDOWN = "COOLDOWN"
    STOP_WAIT = "STOP_WAIT"  # the round ended during a playback; the wrap-up waits for the port


_PORT_FREE = ("IDLE", *TERMINAL)


@dataclass(frozen=True)
class BlendConfig:
    port_fresh_s: float = 0.5  # the port's status must be this recent to start
    cbf_fresh_s: float = 0.5  # the data plane's status must be this recent to start
    a_min_s: float = 0.6
    a_peak_speed: float = 1.5  # rad/s, the farthest joint's peak in A
    a_hand_margin_s: float = 0.1  # the hand arrives at least this long before the arm
    a_timeout_margin_s: float = 0.3  # A gives up this long after T_A
    arrive_fraction: float = 0.5  # of start_tol_rad / hand_start_tol, so the port's gate passes
    ack_timeout_s: float = 0.2
    status_stale_s: float = 0.5  # no port status this long while playing: release fast
    port_lost_s: float = 0.3  # only other segments' statuses (or IDLE) this long while playing: the port lost it
    play_margin_s: float = 1.0  # the playback ends at the latest t_in + hold_max_s + t_out + this after its start
    done_margin_s: float = 0.3  # WAIT_DONE ends this long after the exit should have finished
    min_hold_s: float = 0.8  # a lost label releases only after this long in HOLD
    c_duration_s: float = 0.8  # C from beta = 1
    c_peak_speed: float = 1.8  # rad/s; the CBF caps 2.0 (0.04 rad a tick), the rest is the mirror's own motion
    c_peak_accel: float = 6.0  # rad/s2; a C of 0.6 rad in 0.8 s peaks at 5.4
    c_min_s: float = 0.3
    cooldown_s: float = 1.0
    wrap_done_margin_s: float = 0.2  # STOP_WAIT waits t_out_stop plus this for the port
    wrap_return_max_s: float = 2.0
    wrap_reserve_s: float = 0.3  # the return ends this far inside the wrap-up budget
    wrap_min_s: float = 0.3
    wind_down_s: float = 0.5  # keepalive this long after giving a segment up (+ t_out_stop on errors)
    require_person_frame: bool = True
    # v1.2. Off values (bc35e2d3's behaviour): a_pause_budget_s 0 (also turns off the safety epoch and
    # status age checks), enter_hold_window_s 0, load_backoff_s 0, b_label_unlatch false, enter_max_wait_s 0,
    # late_tick_s 0.
    # zll: once entered, a template is never given up for load: standing still beats going back.
    a_pause_budget_s: float = 4.0  # A may fall this far behind its schedule through pauses...
    a_pause_budget_per_s: float = 0.0  # ...plus this much per second of T_A...
    a_pause_max_s: float = 4.0  # ...but never more than this
    a_status_age_s: float = 0.1  # A moves only while the newest status is this recent by its `at` (2 periods)
    a_untrusted_max_s: float = 0.5  # no status A can trust this long: back to the mirror (absolute)
    a_release_guard_s: float = 0.05  # a hold this close to the CBF's release counts as released
    a_resume_ramp_s: float = 0.2  # the shortest resume after a hold the CBF released (or nearly)
    a_short_ramp_s: float = 0.06  # the shortest resume after a short hold
    a_resume_accel: float = 12.0  # rad/s2; a resume is stretched until no joint accelerates harder (or than A does)
    enter_hold_window_s: float = 2.0  # the held share of CBF calls over this long gates the entry
    # X 0.15: step 0 held 8-11% of the calls, clustered; 0.1 kept out 4% of the triggers at 5% (stress test).
    enter_hold_ratio_open: float = 0.15  # opens at or below this share (X)...
    enter_hold_ratio_close: float = 0.3  # ...and closes above this one (2X)
    load_backoff_s: float = 3.0  # no entry this long after an attempt ended for load
    # A trigger not entered this long after the blender first saw it (its label episode active, the
    # person's, not tried yet) is given up: one rejection, never a late entry. 0: it waits for ever.
    enter_max_wait_s: float = 1.5
    b_label_unlatch: bool = True  # B holds while the label is back; a new episode of it is the same request
    late_tick_s: float = 0.04  # a tick this long after the previous one (2 at 50 Hz) judges no age on its own
    # (b): off until the data plane that echoes a_cap ships in the same package, or nothing is ever entered.
    a_cap: bool = False
    a_cap_lost_s: float = 0.15  # in A, the echo false this long in a row after a true one: back through C
    a_cap_confirm_s: float = 0.5  # in A, no echo this long after entering (or a late tick): the same; A waits at 0

    _FLAGS = ("require_person_frame", "b_label_unlatch", "a_cap")
    _MAY_BE_ZERO = (
        "cooldown_s",
        "a_pause_budget_s",
        "a_pause_budget_per_s",
        "enter_hold_window_s",
        "load_backoff_s",
        "enter_max_wait_s",
        "late_tick_s",
    )

    @classmethod
    def from_mapping(cls, raw: Mapping | None) -> BlendConfig:
        """ValueError naming the key for unknown keys or values that are not finite and positive."""
        raw = dict(raw or {})
        known = {f.name: f for f in fields(cls)}
        unknown = sorted(set(raw) - set(known))
        if unknown:
            raise ValueError(f"unknown motion_template keys: {', '.join(unknown)}")
        values = {}
        for name, value in raw.items():
            if name in cls._FLAGS:
                if not isinstance(value, bool):
                    raise ValueError(f"{name} must be true or false")
            elif not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            elif value < 0.0 or (value == 0.0 and name not in cls._MAY_BE_ZERO):
                raise ValueError(f"{name}={value:g} must be positive")
            elif name in ("arrive_fraction", "enter_hold_ratio_open", "enter_hold_ratio_close") and value > 1.0:
                raise ValueError(f"{name} must be within (0, 1]")
            values[name] = value if name in cls._FLAGS else float(value)
        config = cls(**values)
        if config.enter_hold_ratio_open > config.enter_hold_ratio_close:
            raise ValueError("enter_hold_ratio_open must not exceed enter_hold_ratio_close")
        if config.a_status_age_s >= RELEASE_AFTER_S:
            raise ValueError(f"a_status_age_s must stay below the CBF's release after {RELEASE_AFTER_S:g} s")
        if config.a_cap and config.a_pause_budget_s <= 0.0:
            raise ValueError("a_cap needs the safety epoch: a_pause_budget_s must be positive")
        return config


# Which joint counts how much when a label has several segments (xsign: left / right front):
# the closest P wins, wrists at half weight. Per arm: shoulder pitch, roll, yaw, elbow, wrist yaw, pitch, roll.
SELECT_WEIGHTS = (1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5) * 2


def _min_jerk(s: float) -> tuple[float, float, float]:
    """(value, d/ds, d2/ds2) of 10 s^3 - 15 s^4 + 6 s^5, s clamped to [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return (
        s * s * s * (10.0 + s * (-15.0 + 6.0 * s)),
        30.0 * s * s * (1.0 - s) ** 2,
        60.0 * s * (1.0 - s) * (1.0 - 2.0 * s),
    )


@dataclass(frozen=True)
class _Quintic:
    """From (x0, v0, a0) at t = 0 to (0, 0, 0) at t = duration."""

    x0: float
    v0: float
    a0: float
    duration: float

    def at(self, t: float) -> float:
        T = self.duration
        if t >= T:
            return 0.0
        t = max(t, 0.0)
        x0, v0, a0 = self.x0, self.v0, self.a0
        c3 = (-20.0 * x0 - 12.0 * v0 * T - 3.0 * a0 * T * T) / (2.0 * T**3)
        c4 = (30.0 * x0 + 16.0 * v0 * T + 3.0 * a0 * T * T) / (2.0 * T**4)
        c5 = (-12.0 * x0 - 6.0 * v0 * T - a0 * T * T) / (2.0 * T**5)
        return x0 + t * (v0 + t * (a0 / 2.0 + t * (c3 + t * (c4 + t * c5))))

    def peaks(self, samples: int = 64) -> tuple[float, float]:
        """(max |dx/dt|, max |d2x/dt2|), sampled (the curve is smooth; 64 samples are within a percent)."""
        dt = self.duration / samples
        x = [self.at(k * dt) for k in range(samples + 2)]  # at rest after the end
        x.insert(0, self.x0 - self.v0 * dt)  # moving at v0 before the start
        speed = max(abs(b - a) for a, b in zip(x, x[1:], strict=False)) / dt
        accel = max(abs(a - 2.0 * b + c) for a, b, c in zip(x, x[1:], x[2:], strict=False)) / (dt * dt)
        return speed, accel


class _Clock:
    """A's own time tau: wall time, stopped in a pause, then eased back in.

    After a pause the rate goes 0 -> 1 as a min-jerk over the ramp T_r, so
    tau = tau_f + T_r (2.5 s^4 - 3 s^5 + s^6), s = t / T_r: alpha starts again
    from rest, and the ramp costs T_r / 2 of schedule.
    """

    def __init__(self, now: float):
        self._t0, self._tau0, self._ramp, self._stopped = now, 0.0, 0.0, False

    def at(self, now: float) -> float:
        if self._stopped:
            return self._tau0
        t, T = max(now - self._t0, 0.0), self._ramp
        if t >= T:
            return self._tau0 + t - T / 2.0
        s = t / T
        return self._tau0 + T * s**4 * (2.5 + s * (-3.0 + s))

    def rate(self, now: float) -> tuple[float, float]:
        """(dtau/dt, d2tau/dt2)."""
        if self._stopped:
            return 0.0, 0.0
        t, T = now - self._t0, self._ramp
        if t >= T:
            return 1.0, 0.0
        value, slope, _ = _min_jerk(t / T)
        return value, slope / T

    def stop(self, now: float) -> None:
        self._tau0, self._stopped = self.at(now), True

    def go(self, now: float, ramp: float) -> None:
        self._t0, self._ramp, self._stopped = now, ramp, False


class _MirrorRef:
    """M, what A and C blend from: the mirror command m; frozen in a pause, eased back to m (min-jerk) after it."""

    def __init__(self):
        self._frozen: tuple[float, ...] | None = None
        self._ramp: tuple[float, float] | None = None  # (start, duration)

    @property
    def held(self) -> bool:
        return self._frozen is not None and self._ramp is None

    def value(self, now: float, m: tuple[float, ...]) -> tuple[float, ...]:
        if self._frozen is None:
            return m
        if self._ramp is None:
            return self._frozen
        t0, T = self._ramp
        s = (now - t0) / T
        if s >= 1.0:
            self._frozen = self._ramp = None
            return m
        return _lerp(self._frozen, m, _min_jerk(s)[0])

    def freeze(self, now: float, m: tuple[float, ...]) -> None:
        self._frozen, self._ramp = self.value(now, m), None

    def ramp(self, now: float, duration: float) -> None:
        if self._frozen is not None:
            self._ramp = (now, duration)

    def remaining(self, now: float) -> float:
        if self._frozen is None:
            return 0.0
        if self._ramp is None:
            return math.inf
        return max(self._ramp[0] + self._ramp[1] - now, 0.0)


def _known(value, spec: str = "") -> str:
    return "unknown" if value is None else format(value, spec)


def _cbf_ages(cbf, now: float) -> str:
    """The three ages of a status, apart: since it was received, since it was made (its own ``at``),
    and of the arm feedback the CBF call used (``cbf.arm_age_s``). A missing one is unknown, never 0."""
    status_age = None if cbf.at is None else now - cbf.at
    return (
        f"recv_age={now - cbf.received_at:.3f} status_age={_known(status_age, '.3f')} "
        f"arm_age={_known(cbf.arm_age_s, '.3f')}"
    )


def _fmt(counts: Mapping[str, int]) -> str:
    return ",".join(f"{key}:{value}" for key, value in sorted(counts.items())) or "-"


def _lerp(m: Sequence[float], anchor: Sequence[float], weight: float) -> tuple[float, ...]:
    return tuple(a + weight * (b - a) for a, b in zip(m, anchor, strict=False))


def _farthest(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=False))


def _resume_peaks(
    start: Sequence[float],
    m: Sequence[float],
    P: Sequence[float],
    tau_f: float,
    T_A: float,
    ramp: float,
    m_speed: Sequence[float] | None = None,
) -> tuple[float, float]:
    """(max |d/dt|, max |d2/dt2|), any joint, of A's output from a pause through a resume of ``ramp``.

    M eases from ``start`` to ``m``, which moves on at ``m_speed`` (None: stands still), A's clock from
    ``tau_f`` as in :class:`_Clock`; sampled from the still pose before to a little after the ramp.
    """
    samples = 64
    dt = ramp / samples
    m_speed = m_speed or (0.0,) * len(m)

    def out(t: float) -> tuple[float, ...]:
        if t <= 0.0:
            tau, r = tau_f, 0.0
        elif t >= ramp:
            tau, r = tau_f + t - ramp / 2.0, 1.0
        else:
            s = t / ramp
            tau, r = tau_f + ramp * s**4 * (2.5 + s * (-3.0 + s)), _min_jerk(s)[0]
        moved = tuple(a + v * t for a, v in zip(m, m_speed, strict=False))
        return _lerp(_lerp(start, moved, r), P, _min_jerk(tau / T_A)[0])

    x = [out(k * dt) for k in range(-2, samples + 8)]
    speed = max(_farthest(b, a) for a, b in zip(x, x[1:], strict=False)) / dt
    accel = max(
        max(abs(p - 2.0 * q + r) for p, q, r in zip(a, b, c, strict=False))
        for a, b, c in zip(x, x[1:], x[2:], strict=False)
    )
    return speed, accel / (dt * dt)


def _arm(values) -> tuple[float, ...] | None:
    if values is None:
        return None
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None
    return out if len(out) == ARM_JOINT_COUNT and all(map(math.isfinite, out)) else None


class Blender:
    """Called under the session lock from the hooks; ``now`` is ``time.monotonic``, as the monitors'."""

    def __init__(
        self,
        config: BlendConfig,
        segments: Mapping[str, Segment],
        templates: Mapping[str, Sequence[str]],
        labels: LabelBook,
        port: PortMonitor,
        cbf: CbfMonitor,
        client: PlayClient,
        log: Callable[[str], None],
        hand_travel_s: Callable[[Sequence[float], Sequence[float]], float] | None = None,
    ):
        self._config = config
        self._segments = dict(segments)
        self._templates = {label: tuple(names) for label, names in templates.items()}
        for label, names in self._templates.items():
            missing = [name for name in names if name not in self._segments]
            if missing:
                raise ValueError(f"template {label}: unknown segments {missing}")
        self._labels, self._port, self._cbf, self._client = labels, port, cbf, client
        self._log = log
        self._hand_travel_s = hand_travel_s
        self._attempted: set[tuple[str, int]] = set()  # (template, episode); kept across rounds
        self.exits: dict[str, int] = {}
        self.entries = 0
        self.played = 0
        self.triggers = 0  # label episodes of the actor, not entered before
        self.rejected: dict[str, int] = {}  # triggers that ended without an entry, by their last reason
        self.outcomes: dict[str, int] = {}  # one per attempt (entry): outcome_of
        self.aborts: dict[str, int] = {}  # attempts that neither completed nor were let go, by reason
        self._pending: dict[tuple[str, int], list] = {}  # trigger -> [first seen, last reason, reasons seen]
        self._attempt_open = False
        self.pauses = 0  # A's pauses, all attempts
        self._backoff_until = -math.inf  # load_backoff_s after a load exit
        self._gate_open = True  # the held-share gate (enter_hold_*)
        self._gate_ratio: float | None = None
        self._clock_warned = -math.inf
        self._last_exit = ""
        self._hand_out: tuple[float, ...] | None = None
        self._snap: dict = {}
        self._last_step: float | None = None  # this tick's now
        self._prev_step: float | None = None
        self._tick_late = False  # this tick came more than late_tick_s after the previous one
        self._a_cap_sent: tuple[int, int] | None = None  # (b): on: true was sent for (epoch, attempt), no off since
        self._attempts = 0  # entries into A: the a_cap attempt, from 1, never reused
        self.begin_round()

    # -- round --------------------------------------------------------------------

    def begin_round(self) -> None:
        self._phase = Phase.IDLE
        self._round_disabled = False
        self._port_suspect = False
        self._pear_source: str | None = None  # of the newest PEAR frame this round
        self._older_than: dict[object, tuple[bool, float | None]] = {}  # _older's verdicts, with their tick
        self._mirror_seen: list[tuple[float, tuple[float, ...]]] = []  # (now, m) over MIRROR_SPEED_SPAN_S
        self._last_step = None  # the round's first tick is punctual
        self._clear()
        self._refresh_snapshot()

    def _clear(self) -> None:
        self._label: Label | None = None
        self._actor: int | None = None
        self._segment: Segment | None = None
        self._seg_id: int | None = None
        self._since = 0.0  # when the current phase began
        self._entered_at: float | None = None
        self._play_deadline = math.inf
        self._ack_late_seen = False  # WAIT_ACK put off its timeout for one late tick
        self._foreign_since: float | None = None  # newest statuses not this segment's, since
        self._seen_serial = 0
        self._t_a = 0.0
        self._entry_d = 0.0
        self._entry_dist = 0.0
        self._a_elapsed = 0.0
        self._anchor: tuple[float, ...] | None = None
        self._c_curve: _Quintic | None = None
        self._c_dist = 0.0
        self._deadline = math.inf
        self._wait_reason = ""
        self._released = False
        self._label_lost = False
        self._label_lost_at: float | None = None
        self._played_at: float | None = None
        self._hold_since: float | None = None
        self._status_serial = 0
        self._hand_target: tuple[float, ...] | None = None
        self._wrap_from: tuple[float, ...] | None = None
        self._wrap_t0: float | None = None
        self._wrap_duration = 0.0
        # A's pauses (v1.2)
        self._clock = _Clock(0.0)
        self._ref = _MirrorRef()
        self._budget = 0.0  # how far behind its schedule this A may fall
        self._epoch: int | None = None  # the safety epoch and arm owner A was entered in
        self._owner: int | None = None
        self._cbf_serial = 0  # the newest data-plane status A has read
        self._paused = False
        self._pause_since = 0.0
        self._pause_why = ""
        self._pause_long = False  # the CBF released, or was about to: two clean statuses and a full ramp
        self._clean_run = 0  # clean statuses in a row since the pause began
        self._clean_serial = 0
        self._clean_ats: list[float] = []  # their ``at``, oldest first: each must still be fresh to count
        self._untrusted_since: float | None = None
        self._pauses = 0
        self._paused_s = 0.0
        self._pause_lines = 0
        self._exit_pending: str | None = None  # A's exit waits, paused, for a status the hand-back can rest on
        self._exit_pending_since = 0.0
        self._exit_pending_c: tuple | None = None  # after B started: the C waited for (anchor, beta)
        self._c_checked = False  # C from A after a checked hand-back: it stops on what a safety exit would
        self._exit_cbf = None  # the status a CBF exit was read in, for its log line (not the batch's newest)
        # (b)
        self._a_cap_from = 0.0  # the echo is to be true from here on: entering A, or a late tick
        self._a_cap_seen = False  # a true echo since _a_cap_from
        self._a_cap_waited: float | None = None  # entering A to its first echo; A's clock starts there
        self._a_cap_false_since: float | None = None

    def end_round(self, now: float) -> None:
        """The session's round ends (wrap-up next, or WAIT). A playback is released fast."""
        try:
            self._end_round(now)
        finally:
            self._refresh_snapshot()

    def _end_round(self, now: float) -> None:
        self._flush_triggers(now, "round_end")
        self._log(
            f"template tally triggers={self.triggers} entries={self.entries} played={self.played} "
            f"outcomes={_fmt(self.outcomes)} rejected={_fmt(self.rejected)} aborts={_fmt(self.aborts)}"
        )
        if self._phase in (Phase.WAIT_ACK, Phase.PLAYING, Phase.WAIT_DONE):
            if self._seg_id is not None:
                self._client.release(self._seg_id, fast=True)
            self._exit_log("stop", now)
            self._phase, self._since = Phase.STOP_WAIT, now
            self._exit_pending = self._exit_pending_c = None
            self._hand_target = self._status_hand()
            return
        self._a_cap_off()
        if self._phase in (Phase.A, Phase.C):
            self._exit_log("stop", now)
        if self._phase is not Phase.STOP_WAIT:
            self._phase = Phase.IDLE
            self._clear()

    def disable_round(self, reason: str, now: float) -> None:
        """After an error: nothing more this round, back to the mirror at once.

        A playback is released fast and kept alive for ``t_out_stop +
        wind_down_s`` by the port client alone (it does not need the
        blender), so the data plane plays back to P instead of timing out.
        """
        try:
            linger = (self._segment.t_out_stop if self._segment is not None else 0.6) + self._config.wind_down_s
            self._client.wind_down(now, linger)
        except Exception:
            self._client.abandon()
        try:
            self._a_cap_off()
        except Exception:
            pass  # the data plane drops the flag 0.3 s after the last on
        if self._phase not in (Phase.IDLE, Phase.COOLDOWN):
            try:
                self._exit_log(f"error:{reason}", now)
            except Exception:
                self._count_exit(f"error:{reason}")
        self._round_disabled = True
        self._phase = Phase.IDLE
        self._clear()
        self._refresh_snapshot()

    # -- per tick -----------------------------------------------------------------

    def observe_hand_output(self, values: Sequence[float] | None) -> None:
        self._hand_out = None if values is None else tuple(float(v) for v in values)

    def hand_target(self, now: float) -> tuple[float, ...] | None:
        """The hand overlay's goal instead of the mirror's, or None for the mirror's.

        After a round that ended mid-playback the hand stays at hand_P only as
        long as the wrap-up would wait for the port, also when no wrap-up runs.
        """
        segment = self._segment
        if (
            self._phase is Phase.STOP_WAIT
            and segment is not None
            and now - self._since > segment.t_out_stop + self._config.wrap_done_margin_s
        ):
            return None
        return self._hand_target

    @property
    def phase(self) -> Phase:
        return self._phase

    def step(self, now: float, inputs, mirror: Sequence[float] | None) -> tuple[float, ...] | None:
        """The arm to send this tick, or None to pass the mirror command through unchanged.

        ``mirror`` is the newest mirror command (14), ``inputs`` the session's.
        """
        try:
            arm = self._step(now, inputs, mirror)
            if self._phase is Phase.A and self._config.a_cap and self._epoch is not None:
                self._a_cap_sent = (self._epoch, self._attempts)
                self._client.a_cap(True, self._epoch, self._attempts)
            else:
                self._a_cap_off()
            return arm
        finally:
            self._refresh_snapshot()

    def _a_cap_off(self) -> None:
        sent, self._a_cap_sent = self._a_cap_sent, None
        if sent is not None:
            self._client.a_cap(False, *sent)

    def _step(self, now: float, inputs, mirror: Sequence[float] | None) -> tuple[float, ...] | None:
        late_s = self._config.late_tick_s
        self._tick_late = late_s > 0.0 and self._last_step is not None and now - self._last_step > late_s
        self._prev_step, self._last_step = self._last_step, now
        m = _arm(mirror)
        if m is not None:
            self._mirror_seen.append((now, m))
            while len(self._mirror_seen) > 2 and now - self._mirror_seen[1][0] >= MIRROR_SPEED_SPAN_S:
                self._mirror_seen.pop(0)
        frames = getattr(getattr(inputs, "window", None), "frames", ())
        if frames:  # the window holds only the frames since the previous tick
            self._pear_source = getattr(frames[-1], "source", None)
        self._update_gate(now)
        phase = self._phase
        if phase is Phase.IDLE:
            if m is not None:
                self._try_enter(now, inputs, m)
            else:
                self._scan_triggers(now, inputs, "no_mirror")
            return self._blend_a(now, m) if self._phase is Phase.A else None
        self._scan_triggers(now, inputs, "cooldown" if phase is Phase.COOLDOWN else "busy")
        if phase is Phase.COOLDOWN:
            if now - self._since >= self._config.cooldown_s:
                self._phase = Phase.IDLE
                self._clear()
            return None
        if m is None:  # cannot happen once A began (the mirror command is kept), but never blend from nothing
            m = _arm(getattr(inputs.last_sent, "arm", None))
            if m is None:
                return None
        if self._exit_pending_c is not None:
            return self._tick_back_pending(now, m)
        if phase is Phase.A:
            return self._tick_a(now, inputs, m)
        if phase is Phase.WAIT_ACK:
            return self._tick_wait_ack(now, m)
        if phase is Phase.PLAYING:
            return self._tick_playing(now, inputs, m)
        if phase is Phase.WAIT_DONE:
            return self._tick_wait_done(now, m)
        if phase is Phase.C:
            return self._tick_c(now, m)
        return None  # STOP_WAIT outside a wrap-up: the round is over

    # -- IDLE -> A ----------------------------------------------------------------

    def _entry_block(self, now: float, inputs) -> tuple[str | None, object]:
        """Why nothing may be entered this tick (None: it may), and the port status."""
        if self._round_disabled:
            return "round_disabled", None
        if self._port_suspect:
            return "port_suspect", None
        if self._client.busy:
            return "port_busy", None
        if not getattr(inputs, "publish_enabled", False):
            return "publish_off", None
        actor = inputs.actor
        if actor is None or not actor.present or actor.actor_id is None:
            return "no_actor", None
        if self._config.require_person_frame and self._pear_source != PERSON_SOURCE:
            return "no_person_frame", None
        status = self._port.latest()
        if status is None or not 0 <= now - status.received_at <= self._config.port_fresh_s:
            return "port_stale", None
        if status.state not in _PORT_FREE:
            return "port_playing", None
        if self._config.a_cap and status.a_cap is None:
            return "a_cap_missing", None  # a data plane without (b): the flag would be dropped
        cbf = self._cbf.latest()
        if cbf is None or not 0 <= now - cbf.received_at <= self._config.cbf_fresh_s:
            return "cbf_stale", None
        if self._pausing:
            trust = self._trust(cbf, now)
            if trust is not None:
                return trust, None
        if cbf.intervening:
            return "cbf_hold", None
        if now < self._backoff_until:
            return "backoff", None
        if self._config.enter_hold_window_s > 0.0:
            if self._gate_ratio is None:
                return "hold_ratio_unknown", None
            if not self._gate_open:
                return "hold_ratio", None
        return None, status

    @property
    def _pausing(self) -> bool:
        return self._config.a_pause_budget_s > 0.0

    def _older(self, key: object, age: float, limit: float) -> bool:
        """``age > limit``, judged on punctual ticks.

        A late tick runs before the messages queued while the session was
        starved are read: there, an age between ``limit`` and twice it keeps
        the previous tick's verdict on ``key`` (not old: none was made); beyond
        twice it is too old anyway.
        """
        if age <= limit:
            verdict = False
        elif age > 2.0 * limit or not self._tick_late:
            verdict = True
        else:
            verdict, at = self._older_than.get(key, (False, None))
            verdict = verdict and at is not None and at in (self._prev_step, self._last_step)
        self._older_than[key] = (verdict, self._last_step)
        return verdict

    def _label_now(self, template: str, now: float) -> Label | None:
        return self._labels.current(template, now, lambda age, limit: self._older(("label", template), age, limit))

    def _update_gate(self, now: float) -> None:
        """The held-share gate, with hysteresis: closes above enter_hold_ratio_close, opens at or below _open."""
        config = self._config
        if config.enter_hold_window_s <= 0.0:
            return
        ratio = self._gate_ratio = self._cbf.hold_ratio(now, config.enter_hold_window_s)
        if ratio is None:
            return
        if self._gate_open and ratio > config.enter_hold_ratio_close:
            self._gate_open = False
        elif not self._gate_open and ratio <= config.enter_hold_ratio_open:
            self._gate_open = True
        else:
            return
        word = "open" if self._gate_open else "closed"
        self._log(f"template gate {word} held={ratio:.3f} window={config.enter_hold_window_s:.1f}")

    def _trust(self, cbf, now: float) -> str | None:
        """Why the data plane's status ``cbf`` cannot be relied on at ``now`` (None: it can)."""
        if cbf is None:
            return "cbf_none"
        if cbf.at is None:
            return "cbf_no_time"
        if cbf.at > cbf.received_at + CLOCK_AHEAD_S or cbf.received_at - cbf.at > CLOCK_BEHIND_S:
            if now - self._clock_warned >= 1.0:
                self._clock_warned = now
                self._log(f"cbf status clock mismatch at={cbf.at:.3f} received={cbf.received_at:.3f}")
            return "cbf_clock"
        if self._older("cbf_late", now - cbf.at, self._config.a_status_age_s):
            return "cbf_late"
        if cbf.has_cbf and cbf.mirror_active is not True:
            return "mirror_inactive"
        if cbf.safety_epoch is None:
            return "epoch_missing"
        return None

    def _try_enter(self, now: float, inputs, m: tuple[float, ...]) -> None:
        block, status = self._entry_block(now, inputs)
        self._scan_triggers(now, inputs, block, status, m)

    def _scan_triggers(self, now: float, inputs, block: str | None, status=None, m=None) -> None:
        """Note each trigger and why it is kept out; with no ``block``, enter the first one that may be."""
        live = set()
        for template, names in self._templates.items():
            label = self._label_now(template, now)
            if label is None or (template, label.episode) in self._attempted:
                continue
            key = (template, label.episode)
            live.add(key)
            reason = block
            if reason is None:
                actor = inputs.actor
                if label.actor_id != actor.actor_id:
                    reason = "actor_mismatch"
            if reason is None:
                candidates = [
                    self._segments[name] for name in names if status.segments is None or name in status.segments
                ]
                if not candidates:
                    reason = "not_whitelisted"
            if reason is not None:
                self._note_trigger(key, reason, now)
            wait_max = self._config.enter_max_wait_s
            if key in self._pending and wait_max > 0.0 and now - self._pending[key][0] >= wait_max:
                self._attempted.add(key)  # answered late or not at all: not at all (also on the tick it clears)
                live.discard(key)
                self._reject(key, now, " at=enter_max_wait")
                continue
            if reason is not None:
                continue
            pending = self._pending.pop(key, None)
            if pending is None:
                self.triggers += 1
            self._attempted.add(key)
            segment = min(candidates, key=lambda seg: self._weighted(seg.P, m))
            waited = "" if pending is None else f" waited={now - pending[0]:.2f} blocked={','.join(pending[2])}"
            self._enter_a(now, label, actor.actor_id, segment, m, waited)
            live.discard(key)
            block = "busy"  # the rest are still noted, so they are not taken for gone
        for key in [key for key in self._pending if key not in live]:
            self._reject(key, now, "")

    def _note_trigger(self, key: tuple[str, int], reason: str, now: float) -> None:
        pending = self._pending.get(key)
        if pending is None:
            self.triggers += 1
            self._pending[key] = pending = [now, reason, []]
        pending[1] = reason
        if reason not in pending[2]:
            pending[2].append(reason)

    def _reject(self, key: tuple[str, int], now: float, why: str) -> None:
        first, reason, reasons = self._pending.pop(key)
        self.rejected[reason] = self.rejected.get(reason, 0) + 1
        self._log(
            f"template reject {key[0]} episode={key[1]} reason={reason} waited={now - first:.2f} "
            f"reasons={','.join(reasons)}{why}"
        )

    def _flush_triggers(self, now: float, why: str) -> None:
        for key in list(self._pending):
            self._reject(key, now, f" at={why}")

    @staticmethod
    def _weighted(P: Sequence[float], m: Sequence[float]) -> float:
        return max(w * abs(p - q) for w, p, q in zip(SELECT_WEIGHTS, P, m, strict=False))

    def _enter_a(
        self, now: float, label: Label, actor: int, segment: Segment, m: tuple[float, ...], waited: str = ""
    ) -> None:
        config = self._config
        self._clear()
        self._label, self._actor, self._segment = label, actor, segment
        self._entry_d = self._weighted(segment.P, m)
        self._entry_dist = _farthest(segment.P, m)
        hand = DEFAULT_HAND_TRAVEL_S
        if self._hand_out is not None and self._hand_travel_s is not None:
            hand = self._hand_travel_s(self._hand_out, segment.hand_P)
            if not math.isfinite(hand) or hand < 0.0:
                hand = DEFAULT_HAND_TRAVEL_S
        self._t_a = max(
            MIN_JERK_PEAK * self._entry_dist / config.a_peak_speed, hand + config.a_hand_margin_s, config.a_min_s
        )
        self._phase, self._since, self._entered_at = Phase.A, now, now
        self._a_cap_from = now
        self._attempts += 1
        self._clock = _Clock(now)
        if config.a_cap:
            self._clock.stop(now)  # D1: no step towards P before the data plane holds the flag for this A
        cbf = self._cbf.latest()
        if cbf is not None:
            self._epoch, self._owner, self._cbf_serial = cbf.safety_epoch, cbf.arm_owner, cbf.serial
        if self._pausing:
            self._budget = min(
                config.a_pause_budget_s + config.a_pause_budget_per_s * self._t_a, config.a_pause_max_s
            )
        self._hand_target = segment.hand_P
        self.entries += 1
        self._attempt_open = True
        self._log(
            f"template enter {label.template} seg={segment.name} actor={actor} episode={label.episode} "
            f"d={self._entry_d:.3f} dist={self._entry_dist:.3f} T_A={self._t_a:.2f}{waited}"
        )

    # -- A ------------------------------------------------------------------------

    def _blend_a(self, now: float, m: tuple[float, ...]) -> tuple[float, ...]:
        alpha = _min_jerk(self._clock.at(now) / self._t_a)[0]
        return _lerp(self._ref.value(now, m), self._segment.P, alpha)

    def _tick_a(self, now: float, inputs, m: tuple[float, ...]) -> tuple[float, ...] | None:
        config, segment = self._config, self._segment
        if self._exit_pending is not None:
            return self._tick_exit_pending(now, m)
        reason = self._label_check(now, inputs)
        cbf = self._cbf.latest()
        if reason is None and self._pausing:
            reason = self._a_cbf(now, cbf, m)
        elif reason is None and cbf is not None and cbf.received_at > self._since and cbf.intervening:
            reason = "a_cbf_hold"
        elapsed = self._a_elapsed = self._clock.at(now)
        if reason is None and elapsed > self._t_a + config.a_timeout_margin_s:
            reason = "a_timeout"
        if reason is None and now - self._entered_at > self._t_a + config.a_timeout_margin_s + self._budget:
            reason = "a_pause_budget" if self._paused else "a_timeout"
        if reason is None and config.a_cap:
            reason = self._a_cap_lost(now)
        if reason is not None:
            return self._leave_a(now, reason, m)
        if not self._paused and elapsed >= self._t_a and self._arrived(now, inputs, cbf):
            self._a_cap_off()  # before the start: B is the port's, not the CBF's
            try:
                self._seg_id = self._client.start(segment.name, now)
            except RuntimeError:
                return self._hand_back(now, "port_busy", m, segment.P, 1.0)
            status = self._port.latest()
            self._status_serial = self._seen_serial = 0 if status is None else status.serial
            self._play_deadline = now + segment.t_in + segment.hold_max_s + segment.t_out + config.play_margin_s
            self._phase, self._since = Phase.WAIT_ACK, now
            self._ack_late_seen = False
            return segment.P
        return self._blend_a(now, m)

    def _arrived(self, now: float, inputs, cbf) -> bool:
        config, segment = self._config, self._segment
        sent = _arm(getattr(inputs.last_sent, "arm", None))
        if sent is None or _farthest(sent, segment.P) > config.arrive_fraction * segment.start_tol_rad:
            return False
        hand = self._hand_out
        if hand is None or len(hand) != len(segment.hand_P):
            return False
        if _farthest(hand, segment.hand_P) > config.arrive_fraction * segment.hand_start_tol:
            return False
        status = self._port.latest()
        if status is None or not 0 <= now - status.received_at <= config.port_fresh_s:
            return False
        if status.state not in _PORT_FREE or self._client.busy:
            return False
        if self._pausing and (self._trust(cbf, now) is not None or cbf.safety_epoch != self._epoch):
            return False
        return cbf is not None and 0 <= now - cbf.received_at <= config.cbf_fresh_s and not cbf.intervening

    def _a_cap_lost(self, now: float) -> str | None:
        """``a_cap_lost`` when the data plane does not hold the (b) flag for A, else None.

        Runs after the CBF checks, so an owner or epoch change already seen
        takes its own exit. A late tick never concludes: what the session sent
        may have expired while it starved; the echo gets a_cap_confirm_s again.
        """
        config = self._config
        if self._tick_late:
            self._a_cap_from, self._a_cap_seen = now, False
        status = self._port.latest()
        echo = (
            status is not None
            and status.a_cap is True
            and status.received_at > self._a_cap_from  # an echo from before cannot answer this window's flags
            and not self._older("a_cap_echo", now - status.received_at, A_CAP_ECHO_AGE_S)
            and status.a_cap_epoch == self._epoch
            and status.a_cap_attempt == self._attempts  # an echo of an earlier A's flag answers nothing here
            and status.a_cap_age_s is not None
            and status.a_cap_age_s <= A_CAP_ECHO_AGE_S
        )
        if echo:
            self._a_cap_seen, self._a_cap_false_since = True, None
            if self._a_cap_waited is None:
                self._a_cap_waited = now - self._entered_at
                if not self._paused:
                    self._clock.go(now, 0.0)  # alpha leaves 0 at rest: no ramp needed
            return None
        if self._a_cap_false_since is None:
            self._a_cap_false_since = now
        if not self._a_cap_seen:
            return "a_cap_lost" if now - self._a_cap_from >= config.a_cap_confirm_s else None
        return "a_cap_lost" if now - self._a_cap_false_since >= config.a_cap_lost_s else None

    def _leave_a(self, now: float, reason: str, m: tuple[float, ...]) -> tuple[float, ...] | None:
        """To C from where A is: alpha's value, speed and acceleration (0 in a pause); M eases back to m.

        v1.2 (plan 3.6): only on a hand-back it can check now. A safety exit, or
        one seen in the statuses not read yet, sends no C (:meth:`_safety_stop`);
        any other exit without a trusted status of A's epoch and owner waits for
        one, paused, at most ``a_untrusted_max_s``.
        """
        config = self._config
        if self._pausing:
            safety = reason if reason in SAFETY_EXITS else self._safety_change()
            if safety is not None:
                return self._safety_stop(now, safety, "" if safety == reason else reason)
            if not self._hand_back_ok(now):
                if not self._paused:
                    self._pause(now, m, "hand_back", self._cbf.latest())
                if self._exit_pending is None:
                    self._exit_pending, self._exit_pending_since = reason, now
                return self._blend_a(now, m)
            self._c_checked, self._untrusted_since = True, None
        back = ""
        if self._exit_pending is not None:
            back = f" back=c after={self._exit_pending}:{now - self._exit_pending_since:.2f}"
            reason, self._exit_pending = self._exit_pending, None
        elif self._pausing:
            back = " back=c"
        self._a_cap_off()
        s = self._clock.at(now) / self._t_a
        value, speed, accel = _min_jerk(s)
        if s >= 1.0:
            speed = accel = 0.0
        rate, rate_slope = self._clock.rate(now)
        if self._paused:
            self._paused_s += now - self._pause_since
            self._paused = False
        if self._ref.held:  # left in a pause: M goes back to the mirror within C
            gap = _farthest(self._ref.value(now, m), m)
            self._ref.ramp(now, max(config.a_resume_ramp_s, MIN_JERK_PEAK * gap / config.c_peak_speed))
        T = self._t_a
        return self._to_c(
            now,
            reason,
            m,
            self._segment.P,
            value,
            speed * rate / T,
            accel * rate**2 / T**2 + speed * rate_slope / T,
            back,
        )

    @staticmethod
    def _safety_of(state, epoch: int | None, owner: int | None) -> str | None:
        if state.restarted:
            return "a_epoch_reset"  # a restarted data plane counts its epochs again: equal proves nothing
        if state.safety_epoch is not None and state.safety_epoch != epoch:
            return "a_epoch_changed"
        if state.arm_owner != owner:
            return "a_owner_changed"
        return None

    def _safety_change(self) -> str | None:
        """A safety exit in the statuses not read yet (they are read)."""
        for state in self._cbf.after(self._cbf_serial):
            self._cbf_serial = state.serial
            safety = self._safety_of(state, self._epoch, self._owner)
            if safety is not None:
                self._exit_cbf = state
                return safety
        return None

    def _hand_back_ok(self, now: float) -> bool:
        """C may follow A: the newest status is trusted now, of A's safety epoch and arm owner."""
        cbf = self._cbf.latest()
        return self._trust(cbf, now) is None and cbf.safety_epoch == self._epoch and cbf.arm_owner == self._owner

    def _tick_exit_pending(self, now: float, m: tuple[float, ...]) -> tuple[float, ...] | None:
        safety = self._safety_change()
        if safety is not None:
            return self._safety_stop(now, safety, self._exit_pending)
        if self._hand_back_ok(now):
            return self._leave_a(now, self._exit_pending, m)
        since = self._exit_pending_since
        if self._untrusted_since is not None:
            since = min(since, self._untrusted_since)  # a_untrusted_max_s stays absolute
        if now - since > self._config.a_untrusted_max_s:
            return self._safety_stop(now, "a_cbf_untrusted", self._exit_pending)
        return self._blend_a(now, m)  # paused: stands still

    def _safety_stop(self, now: float, reason: str, after: str = "") -> None:
        """Ends the template without a C: from this tick the mirror command passes unchanged and the
        existing safety path (runtime gate, data plane epoch and owner) decides what reaches the robot."""
        self._a_cap_off()
        if self._paused:
            self._paused_s += now - self._pause_since
            self._paused = False
        waited = now - self._exit_pending_since if self._exit_pending or self._exit_pending_c else 0.0
        back = " back=safety" + (f" after={after}:{waited:.2f}" if after else "")
        self._exit_log(reason, now, back)
        self._phase, self._since = Phase.COOLDOWN, now
        self._label = self._segment = None
        self._hand_target = None
        self._exit_pending, self._exit_pending_c, self._c_checked = None, None, False
        return None

    def _hand_back(
        self, now: float, reason: str, m: tuple[float, ...], anchor: tuple[float, ...], beta: float
    ) -> tuple[float, ...] | None:
        """To C from the segment's side of A (from P, or where the port says the arm is).

        v1.2 (plan 3.6, review 1120 T1b): a STOPPED playback (the data plane
        stopped it: epoch_changed, revoked, named_arm, control_lapsed,
        internal_error, or a reason not known here) and a safety change in the
        statuses send no C: the data plane's CBF path goes on from P_actual.
        Any other exit waits for a trusted status of A's epoch and owner, the
        arm kept where it is, at most ``a_untrusted_max_s``.
        """
        if not self._pausing:
            return self._to_c(now, reason, m, anchor, beta)
        pending = self._exit_pending_c is not None
        after = reason if pending else ""
        if reason.startswith("stopped"):
            return self._safety_stop(now, reason)
        safety = self._safety_change()
        if safety is not None:
            return self._safety_stop(now, safety, reason)
        if not self._hand_back_ok(now):
            if not pending:
                self._exit_pending, self._exit_pending_since = reason, now
                self._exit_pending_c = (tuple(anchor), beta)
            return self._exit_pending_c[0]
        back = " back=c"
        if pending:
            back = f" back=c after={after}:{now - self._exit_pending_since:.2f}"
            self._exit_pending = self._exit_pending_c = None
        self._c_checked, self._untrusted_since = True, None
        return self._to_c(now, reason, m, anchor, beta, back=back)

    def _tick_back_pending(self, now: float, m: tuple[float, ...]) -> tuple[float, ...] | None:
        """The hand-back waiting for trust: this playback's STOPPED still wins over the exit waited on (R1)."""
        reason, (anchor, beta) = self._exit_pending, self._exit_pending_c
        status = self._our_status()
        if status is not None and status.state == "STOPPED":
            return self._safety_stop(now, f"stopped:{status.reason}", reason)
        if now - self._exit_pending_since > self._config.a_untrusted_max_s and not self._hand_back_ok(now):
            safety = self._safety_change() or "a_cbf_untrusted"
            return self._safety_stop(now, safety, reason)
        return self._hand_back(now, reason, m, anchor, beta)

    def _a_cbf(self, now: float, cbf, m: tuple[float, ...]) -> str | None:
        """A's reading of the data plane (v1.2): an exit reason, or None after pausing or resuming as due.

        In order: every status since the last tick (a collision, a safety epoch
        or arm owner change in any of them ends A); then whether the newest can
        be trusted now; then hold -> pause, clean -> resume.
        """
        config = self._config
        for state in self._cbf.after(self._cbf_serial):
            self._cbf_serial = state.serial
            safety = self._safety_of(state, self._epoch, self._owner)
            if safety is not None or state.collision:
                self._exit_cbf = state
                return safety or "a_cbf_collision"
            if not self._paused:
                continue
            if state.at is not None and self._clean_ats and state.at <= self._clean_ats[-1]:
                self._clean_serial = state.serial  # the same status again: it neither counts nor breaks the run
                continue
            clean = (
                not state.intervening
                and state.at is not None
                and -CLOCK_BEHIND_S <= state.at - state.received_at <= CLOCK_AHEAD_S
                and now - state.at <= config.a_status_age_s  # T3: each one fresh, not only the newest
                and state.safety_epoch == self._epoch
                and (state.mirror_active is True or not state.has_cbf)
            )
            if clean:
                if state.serial != self._clean_serial + 1:
                    self._clean_ats = []
                self._clean_ats.append(state.at)
                self._clean_serial = state.serial
            else:
                self._clean_ats = []
                release_at = state.hold_release_after_s or RELEASE_AFTER_S
                self._pause_long |= state.releasing or state.stuck_s >= release_at - config.a_release_guard_s
        trust = self._trust(cbf, now)
        if trust == "epoch_missing":
            return "a_epoch_missing"  # the epoch cannot be checked: a safety exit
        if trust is None:
            self._untrusted_since = None
        elif self._untrusted_since is None:
            self._untrusted_since = now
        elif now - self._untrusted_since > config.a_untrusted_max_s:
            return "a_cbf_untrusted"
        holding = trust is None and cbf.intervening
        if not self._paused:
            if trust is not None or holding:
                self._pause(now, m, "hold" if holding else trust, cbf)
            return None
        release_at = (cbf.hold_release_after_s if cbf is not None else None) or RELEASE_AFTER_S
        if holding:  # how long the CBF has held by now
            held_for = cbf.stuck_s + max(now - cbf.at, 0.0)
        else:  # nothing known since the newest status: it may have held since then
            held_for = now - cbf.at if cbf is not None and cbf.at is not None else now - self._pause_since
        self._pause_long |= held_for >= release_at - config.a_release_guard_s
        self._clean_ats = [at for at in self._clean_ats if now - at <= config.a_status_age_s]  # the run's fresh tail
        self._clean_run = len(self._clean_ats)
        if trust is None and not holding and self._clean_run >= (2 if self._pause_long else 1):
            return self._resume(now, m)
        if now - self._entered_at - self._clock.at(now) > self._budget:
            return "a_pause_budget"
        return None

    def _pause(self, now: float, m: tuple[float, ...], why: str, cbf) -> None:
        """Arm, hand, A's clock and M stand still: the CBF's release after hold_release_after_s is this pose."""
        self._ref.freeze(now, m)
        self._clock.stop(now)
        self._paused, self._pause_since, self._pause_why, self._pause_long = True, now, why, False
        self._clean_run, self._clean_ats = 0, []
        hand = self._hand_out
        if hand is not None and len(hand) == len(self._segment.hand_P):
            self._hand_target = hand
        self._pauses += 1
        self.pauses += 1
        if self._pause_lines < PAUSE_LOG_LINES:
            self._pause_lines += 1
            detail = "none" if cbf is None else f"{cbf.detail} {_cbf_ages(cbf, now)}"
            self._log(
                f"template pause {self._label.template} why={why} n={self._pauses} tau={self._clock.at(now):.2f} "
                f"behind={now - self._entered_at - self._clock.at(now):.2f} cbf=[{detail}]"
            )

    def _resume(self, now: float, m: tuple[float, ...]) -> str | None:
        ramp = self._ramp_duration(now, m)
        behind = now - self._entered_at - self._clock.at(now)
        if behind + ramp / 2.0 > self._budget:  # never begin a ramp that the budget cannot finish
            return "a_pause_budget"
        paused = now - self._pause_since
        self._paused_s += paused
        self._ref.ramp(now, ramp)
        if not self._config.a_cap or self._a_cap_waited is not None:
            self._clock.go(now, ramp)  # else alpha stays at 0 until the first echo
        self._paused, self._untrusted_since = False, None
        self._hand_target = self._segment.hand_P
        if self._pause_lines < PAUSE_LOG_LINES:
            self._pause_lines += 1
            self._log(
                f"template resume {self._label.template} n={self._pauses} paused={paused:.2f} "
                f"long={self._pause_long} ramp={ramp:.2f} behind={behind + ramp / 2.0:.2f}"
            )
        return None

    def _ramp_duration(self, now: float, m: tuple[float, ...]) -> float:
        """T_r: M eases from M_f to m and alpha's clock from 0 to 1; stretched until no joint is faster than
        c_peak_speed nor accelerates harder than a_resume_accel (or than A does), m moving on as it did.

        Easing back to a moving m adds 2 r' m' (x2-wt-18 REPORT 3.1): the ramp is stretched for it. What
        the mirror's motion adds to A itself, the mirror's speed and up to 2 alpha' m', is allowed on top.
        """
        config, P, T_A = self._config, self._segment.P, self._t_a
        start, tau_f = self._ref.value(now, m), self._clock.at(now)
        gap = _farthest(start, m)
        floor = config.a_resume_ramp_s if self._pause_long else config.a_short_ramp_s
        ramp = max(floor, MIN_JERK_PEAK * gap / config.a_peak_speed)
        dist = _farthest(P, m)
        m_speed = self._mirror_speed()
        moving = max(map(abs, m_speed or ()), default=0.0)
        speed_limit = max(config.c_peak_speed, MIN_JERK_PEAK * dist / T_A) * 1.02 + moving
        accel_limit = max(config.a_resume_accel, MIN_JERK_ACCEL * dist / T_A**2) * 1.02
        accel_limit += 2.0 * MIN_JERK_PEAK / T_A * moving
        for _ in range(C_STRETCH_STEPS):
            speed, accel = _resume_peaks(start, m, P, tau_f, T_A, ramp, m_speed)
            if speed <= speed_limit and accel <= accel_limit:
                break
            ramp *= C_STRETCH
        return ramp

    def _mirror_speed(self) -> tuple[float, ...] | None:
        """m', each joint, over the last MIRROR_SPEED_SPAN_S of mirror commands (None: not known)."""
        seen = self._mirror_seen
        if len(seen) < 2 or seen[-1][0] - seen[0][0] < MIRROR_SPEED_SPAN_S / 2.0:
            return None
        (t0, a), (t1, b) = seen[0], seen[-1]
        return tuple((y - x) / (t1 - t0) for x, y in zip(a, b, strict=False))

    def _label_check(self, now: float, inputs) -> str | None:
        """Why the person no longer asks for this template, or None.

        With ``b_label_unlatch`` a new episode of the same template by the same
        actor is the same request (the label dropped out and came back): it is
        taken over, not counted as a trigger, and never entered on its own.
        """
        actor = inputs.actor
        if actor is None or actor.actor_id != self._actor:
            return "actor_changed"
        label = self._label_now(self._label.template, now)
        if label is None or label.actor_id != self._actor:
            return "label_lost"
        if label.episode != self._label.episode:
            if not self._config.b_label_unlatch:
                return "label_lost"
            key = (label.template, label.episode)
            self._attempted.add(key)
            if self._pending.pop(key, None) is not None:
                self.triggers -= 1
            self._log(
                f"template relabel {label.template} episode={self._label.episode}->{label.episode} "
                f"phase={self._phase.value}"
            )
            self._label = label
        return None

    # -- the playback ---------------------------------------------------------------

    def _our_status(self):
        status = self._port.latest()
        if status is None or status.seg_id != self._seg_id or status.serial <= self._status_serial:
            return None
        return status

    def _status_arm(self) -> tuple[float, ...]:
        status = self._port.latest()
        if status is not None and status.seg_id == self._seg_id and status.P_actual is not None:
            return status.P_actual
        return self._segment.P

    def _status_hand(self) -> tuple[float, ...]:
        status = self._port.latest()
        if status is not None and status.seg_id == self._seg_id and status.hand_P is not None:
            return status.hand_P
        return self._segment.hand_P

    def _tick_wait_ack(self, now: float, m: tuple[float, ...]) -> tuple[float, ...]:
        status = self._our_status()
        if status is not None and status.state in PLAYING:
            self._phase, self._since = Phase.PLAYING, now
            self._hold_since = now if status.state == "HOLD" else None
            self._played_at = now
            self.played += 1
            self._hand_target = self._status_hand()
            return self._status_arm()
        if status is not None and status.state in TERMINAL:
            return self._hand_back(now, f"{status.state.lower()}:{status.reason}", m, self._segment.P, 1.0)
        if now - self._since > self._config.ack_timeout_s:
            if self._tick_late and not self._ack_late_seen:
                # Judged on punctual ticks: a late one runs before the statuses queued while starving, the next
                # one reads them (port_suspect would keep every template out for the rest of the round).
                self._ack_late_seen = True
                return self._segment.P
            self._client.release(self._seg_id, fast=True)
            self._port_suspect = True
            return self._hand_back(now, "no_ack", m, self._segment.P, 1.0)
        return self._segment.P

    def _port_lost(self, status) -> bool:
        """Fresh statuses, none this segment's, for ``port_lost_s``: the data plane no longer plays it.

        One other segment's status (a refusal of someone else's start) is not
        enough: the port's next periodic status is ours again.
        """
        if status is None or status.serial <= self._seen_serial:
            return False
        self._seen_serial = status.serial
        if status.seg_id == self._seg_id:
            self._foreign_since = None
            return False
        if self._foreign_since is None:
            self._foreign_since = status.received_at
        return status.received_at - self._foreign_since >= self._config.port_lost_s

    def _tick_playing(self, now: float, inputs, m: tuple[float, ...]) -> tuple[float, ...]:
        config, segment = self._config, self._segment
        status = self._port.latest()
        arm = self._status_arm()
        self._hand_target = self._status_hand()
        if status is None or self._older("port_stale", now - status.received_at, config.status_stale_s):
            self._client.release(self._seg_id, fast=True)
            return self._to_wait_done(now, "status_stale", segment.t_out_stop)
        if self._port_lost(status):
            return self._lose_port(now, m, arm)
        if now > self._play_deadline:
            self._client.release(self._seg_id, fast=True)
            return self._to_wait_done(now, "play_timeout", segment.t_out_stop)
        if status.seg_id == self._seg_id:
            if status.state in TERMINAL:
                return self._hand_back(now, f"{status.state.lower()}:{status.reason}", m, arm, 1.0)
            if status.state == "EXIT_B":
                return self._to_wait_done(now, status.reason or "exit", segment.t_out)
            if status.state == "HOLD" and self._hold_since is None:
                self._hold_since = now
        lost = self._label_check(now, inputs) is not None
        if lost and not self._label_lost:
            self._label_lost_at = now
        if config.b_label_unlatch:  # the label came back: hold on (a release once sent stays)
            self._label_lost = lost
        else:
            self._label_lost |= lost
        if self._label_lost and self._hold_since is not None and now - self._hold_since >= config.min_hold_s:
            self._client.release(self._seg_id, fast=False)
            self._released = True
            self._log(
                f"template release {self._label.template} seg={segment.name} hold={now - self._hold_since:.2f} "
                f"label_lost_for={now - self._label_lost_at:.2f} play={now - self._played_at:.2f} "
                f"label=[{self._labels.describe(self._label.template, now)}]"
            )
            return self._to_wait_done(now, "label_inactive", segment.t_out)
        return arm

    def _to_wait_done(self, now: float, reason: str, exit_s: float) -> tuple[float, ...]:
        self._phase, self._since = Phase.WAIT_DONE, now
        self._deadline = now + exit_s + self._config.done_margin_s
        self._wait_reason = reason
        return self._status_arm()

    def _tick_wait_done(self, now: float, m: tuple[float, ...]) -> tuple[float, ...]:
        config, segment = self._config, self._segment
        status = self._port.latest()
        arm = self._status_arm()
        self._hand_target = self._status_hand()
        stale = status is None or self._older("port_stale", now - status.received_at, config.status_stale_s)
        if stale:
            self._deadline = min(self._deadline, now + segment.t_out_stop + config.done_margin_s)
        elif status.seg_id == self._seg_id and status.state in TERMINAL:
            reason = self._wait_reason if status.state == "DONE" else f"{status.state.lower()}:{status.reason}"
            return self._hand_back(now, reason, m, arm, 1.0)
        elif self._port_lost(status):
            return self._lose_port(now, m, arm)
        if now >= self._deadline:
            # The port never said it finished: one more fast release, and the keepalive ends soon after,
            # so the data plane's own keepalive timeout stops it (a port that plays on is never kept alive).
            self._client.wind_down(now, config.wind_down_s)
            if stale:
                reason = "status_stale"
            else:
                reason = "play_timeout" if self._wait_reason == "play_timeout" else "done_timeout"
            return self._hand_back(now, reason, m, arm, 1.0)
        return arm

    def _lose_port(self, now: float, m: tuple[float, ...], arm: tuple[float, ...]) -> tuple[float, ...]:
        # One fast release costs nothing: a port that still has the segment after all lets it go at once.
        self._client.release(self._seg_id, fast=True)
        self._client.abandon()
        return self._hand_back(now, "port_lost", m, arm, 1.0)

    # -- C ------------------------------------------------------------------------

    def _to_c(
        self,
        now: float,
        reason: str,
        m: tuple[float, ...] | None,
        anchor: Sequence[float],
        beta: float,
        speed: float = 0.0,
        accel: float = 0.0,
        back: str = "",
    ) -> tuple[float, ...]:
        config = self._config
        self._exit_log(reason, now, back)
        self._anchor = tuple(anchor)
        dist = 0.0 if m is None else _farthest(self._anchor, self._ref.value(now, m))
        self._c_dist = dist
        duration = max(config.c_duration_s * beta, MIN_JERK_PEAK * beta * dist / config.c_peak_speed, config.c_min_s)
        curve = _Quintic(beta, speed, accel, duration)
        # Leaving A half way, beta still moves toward P and has to turn round. Never faster than
        # c_peak_speed nor harder than c_peak_accel (or than A was when it was left).
        speed_limit = max(config.c_peak_speed, abs(speed) * dist) * 1.02
        accel_limit = max(config.c_peak_accel, abs(accel) * dist) * 1.02
        for _ in range(C_STRETCH_STEPS):
            peak_speed, peak_accel = curve.peaks()
            if peak_speed * dist <= speed_limit and peak_accel * dist <= accel_limit:
                break
            curve = _Quintic(beta, speed, accel, curve.duration * C_STRETCH)
        easing = self._ref.remaining(now)  # A was left in a pause: C ends after M is back on the mirror
        if easing > curve.duration:
            curve = _Quintic(beta, speed, accel, easing)
        self._c_curve = curve
        self._phase, self._since = Phase.C, now
        self._hand_target = None
        if m is None:
            return self._anchor
        return self._blend_c(now, m)

    def _tick_c(self, now: float, m: tuple[float, ...]) -> tuple[float, ...] | None:
        """C after a checked hand-back from A stops (no more C) on what would be a safety exit in A."""
        if self._c_checked:
            safety = self._safety_change()
            if safety is None:
                if self._trust(self._cbf.latest(), now) is None:
                    self._untrusted_since = None
                elif self._untrusted_since is None:
                    self._untrusted_since = now
                elif now - self._untrusted_since > self._config.a_untrusted_max_s:
                    safety = "a_cbf_untrusted"
            if safety is not None:
                return self._safety_stop(now, safety)
        return self._blend_c(now, m)

    def _blend_c(self, now: float, m: tuple[float, ...]) -> tuple[float, ...]:
        t = now - self._since
        beta = min(max(self._c_curve.at(t), 0.0), 1.0)
        arm = _lerp(self._ref.value(now, m), self._anchor, beta)
        if t >= self._c_curve.duration:
            name = self._label.template if self._label is not None else "?"
            total = now - self._entered_at if self._entered_at is not None else 0.0
            self._log(
                f"template mirror {name} c_elapsed={t:.2f} c_T={self._c_curve.duration:.2f} "
                f"c_dist={self._c_dist:.3f} total={total:.2f}"
            )
            self._phase, self._since = Phase.COOLDOWN, now
            self._label = self._segment = None
            self._c_checked = False
        return arm

    # -- wrap-up after a round that ended mid-playback ------------------------------

    def wrap(self, context, hold_arm_command) -> object | None:
        """A ``WrapUpStep`` while STOP_WAIT owns the wrap-up, else None (the inner hooks' wrap-up).

        First P_actual is held (the data plane is playing back to it, fast) until
        the port says it finished or ``t_out_stop + wrap_done_margin_s``; then a
        quintic from P_actual to the reset pose within what is left of the budget.
        ``hold_arm_command(arm)`` builds the ``UpperBodyCommand``.
        """
        try:
            return self._wrap(context, hold_arm_command)
        finally:
            self._refresh_snapshot()

    def _wrap(self, context, hold_arm_command) -> object | None:
        from manipulation_execution.mirror_session.hooks import WrapUpPurpose, WrapUpStep

        if self._phase is not Phase.STOP_WAIT or context.purpose == WrapUpPurpose.APPROACH:
            return None
        config, segment = self._config, self._segment
        elapsed = context.elapsed_s
        if self._wrap_t0 is None:
            status = self._port.latest()
            finished = status is not None and status.seg_id == self._seg_id and status.state in TERMINAL
            if not finished and elapsed < segment.t_out_stop + config.wrap_done_margin_s:
                self._hand_target = self._status_hand()
                return WrapUpStep(hold_arm_command(self._status_arm()), False)
            target = _arm(getattr(context.target, "arm", None))
            self._hand_target = None
            if target is None:
                self._phase = Phase.IDLE
                self._clear()
                return WrapUpStep(None, True)
            self._wrap_from, self._wrap_target, self._wrap_t0 = self._status_arm(), target, elapsed
            self._wrap_duration = max(
                min(config.wrap_return_max_s, context.budget_s - elapsed - config.wrap_reserve_s), config.wrap_min_s
            )
        s = (elapsed - self._wrap_t0) / self._wrap_duration
        if s >= 1.0:
            target = self._wrap_target
            self._phase = Phase.IDLE
            self._clear()
            return WrapUpStep(hold_arm_command(target), True)
        return WrapUpStep(hold_arm_command(_lerp(self._wrap_from, self._wrap_target, _min_jerk(s)[0])), False)

    # -- reporting ----------------------------------------------------------------

    def _count_exit(self, reason: str) -> None:
        key = reason.split(":", 1)[0] if reason.startswith(("rejected", "stopped", "error")) else reason
        self.exits[key] = self.exits.get(key, 0) + 1
        self._last_exit = reason

    def _exit_log(self, reason: str, now: float, back: str = "") -> None:
        self._count_exit(reason)
        name = self._label.template if self._label is not None else "?"
        seg = self._segment.name if self._segment is not None else "?"
        cbf, source = self._exit_cbf, "trigger"
        if cbf is None:
            cbf, source = self._cbf.latest(), "newest"
        self._exit_cbf = None
        cbf_text = "none"
        if cbf is not None:
            cbf_text = (
                f"{cbf.detail} reason={_known(cbf.reason)} of={source} {_cbf_ages(cbf, now)} "
                f"at={_known(cbf.at, '.3f')} epoch={_known(cbf.safety_epoch)} owner={_known(cbf.arm_owner)} "
                f"active={_known(cbf.mirror_active)}"
            )
        label = self._labels.describe(self._label.template, now) if self._label is not None else "none"
        outcome = "repeat"  # a later exit of the same attempt (a stop or an error during C)
        backoff = ""
        if self._attempt_open:
            self._attempt_open = False
            outcome = outcome_of(reason)
            self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
            if outcome not in ("complete", "let_go"):
                key = reason.split(":", 1)[0]
                self.aborts[key] = self.aborts.get(key, 0) + 1
            load = reason in LOAD_EXITS or (reason == "a_timeout" and self._pauses > 0)
            if load and self._config.load_backoff_s > 0.0:
                self._backoff_until = now + self._config.load_backoff_s
                backoff = f" backoff={self._config.load_backoff_s:.1f}"
        paused_s = self._paused_s + (now - self._pause_since if self._paused else 0.0)
        a_cap = f" a_cap_wait={_known(self._a_cap_waited, '.2f')}" if self._config.a_cap else ""
        self._log(
            f"template exit {name} seg={seg} phase={self._phase.value} reason={reason} "
            f"d={self._entry_d:.3f} dist={self._entry_dist:.3f} T_A={self._t_a:.2f} "
            f"a_elapsed={self._a_elapsed:.2f} phase_elapsed={now - self._since:.2f} "
            f"total={now - self._entered_at if self._entered_at is not None else 0.0:.2f} "
            f"outcome={outcome} played={self._played_at is not None} cbf=[{cbf_text}] label=[{label}] "
            f"pauses={self._pauses} paused_s={paused_s:.2f}{backoff}{back}{a_cap}"
        )

    def _refresh_snapshot(self) -> None:
        # One assignment: the status timer reads it from another thread without the session lock.
        self._snap = {
            "phase": self._phase.value,
            "template": None if self._label is None else self._label.template,
            "segment": None if self._segment is None else self._segment.name,
            "seg_id": self._seg_id,
            "entries": self.entries,
            "played": self.played,
            "exits": dict(self.exits),
            "triggers": self.triggers,
            "rejected": dict(self.rejected),
            "outcomes": dict(self.outcomes),
            "aborts": dict(self.aborts),
            "last_exit": self._last_exit,
            "round_disabled": self._round_disabled,
            "port_suspect": self._port_suspect,
            "pauses": self.pauses,
            "paused": self._paused,
            "gate": {"open": self._gate_open, "held": self._gate_ratio},
            "backoff_until": None if math.isinf(self._backoff_until) else self._backoff_until,
        }

    def snapshot(self) -> dict:
        return dict(self._snap)
