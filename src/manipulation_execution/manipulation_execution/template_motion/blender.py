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
WAIT_ACK ``ack_timeout_s``, the playback ``t_in + hold_max_s + t_out +
play_margin_s`` from its start, a stale-status check and a port-lost check
(fresh statuses that are not this segment's for ``port_lost_s``: the data
plane restarted, or plays something else), WAIT_DONE ``t_out +
done_margin_s``. Every exit goes through C back to the mirror, and every exit
is logged with its reason and the time since the template was entered.
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
PERSON_SOURCE = "yolox"  # a PEAR frame of a detected person; center_fallback never enters a template
DEFAULT_HAND_TRAVEL_S = 1.0  # T_A's hand share when the hand output is not known yet


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
    c_peak_speed: float = 1.5
    c_peak_accel: float = 6.0  # rad/s2; a C of 0.6 rad in 0.8 s peaks at 5.4
    c_min_s: float = 0.3
    cooldown_s: float = 1.0
    wrap_done_margin_s: float = 0.2  # STOP_WAIT waits t_out_stop plus this for the port
    wrap_return_max_s: float = 2.0
    wrap_reserve_s: float = 0.3  # the return ends this far inside the wrap-up budget
    wrap_min_s: float = 0.3
    wind_down_s: float = 0.5  # keepalive this long after giving a segment up (+ t_out_stop on errors)
    require_person_frame: bool = True

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
            if name == "require_person_frame":
                if not isinstance(value, bool):
                    raise ValueError("require_person_frame must be true or false")
            elif not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            elif value < 0.0 or (value == 0.0 and name != "cooldown_s"):
                raise ValueError(f"{name}={value:g} must be positive")
            elif name == "arrive_fraction" and value > 1.0:
                raise ValueError("arrive_fraction must be within (0, 1]")
            values[name] = value if name == "require_person_frame" else float(value)
        return cls(**values)


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


def _lerp(m: Sequence[float], anchor: Sequence[float], weight: float) -> tuple[float, ...]:
    return tuple(a + weight * (b - a) for a, b in zip(m, anchor, strict=False))


def _farthest(a: Sequence[float], b: Sequence[float]) -> float:
    return max(abs(x - y) for x, y in zip(a, b, strict=False))


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
        self._last_exit = ""
        self._hand_out: tuple[float, ...] | None = None
        self._snap: dict = {}
        self.begin_round()

    # -- round --------------------------------------------------------------------

    def begin_round(self) -> None:
        self._phase = Phase.IDLE
        self._round_disabled = False
        self._port_suspect = False
        self._pear_source: str | None = None  # of the newest PEAR frame this round
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
        self._foreign_since: float | None = None  # newest statuses not this segment's, since
        self._seen_serial = 0
        self._t_a = 0.0
        self._entry_d = 0.0
        self._entry_dist = 0.0
        self._a_elapsed = 0.0
        self._anchor: tuple[float, ...] | None = None
        self._c_curve: _Quintic | None = None
        self._deadline = math.inf
        self._wait_reason = ""
        self._released = False
        self._label_lost = False
        self._hold_since: float | None = None
        self._status_serial = 0
        self._hand_target: tuple[float, ...] | None = None
        self._wrap_from: tuple[float, ...] | None = None
        self._wrap_t0: float | None = None
        self._wrap_duration = 0.0

    def end_round(self, now: float) -> None:
        """The session's round ends (wrap-up next, or WAIT). A playback is released fast."""
        try:
            self._end_round(now)
        finally:
            self._refresh_snapshot()

    def _end_round(self, now: float) -> None:
        if self._phase in (Phase.WAIT_ACK, Phase.PLAYING, Phase.WAIT_DONE):
            if self._seg_id is not None:
                self._client.release(self._seg_id, fast=True)
            self._exit_log("stop", now)
            self._phase, self._since = Phase.STOP_WAIT, now
            self._hand_target = self._status_hand()
            return
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
            return self._step(now, inputs, mirror)
        finally:
            self._refresh_snapshot()

    def _step(self, now: float, inputs, mirror: Sequence[float] | None) -> tuple[float, ...] | None:
        m = _arm(mirror)
        frames = getattr(getattr(inputs, "window", None), "frames", ())
        if frames:  # the window holds only the frames since the previous tick
            self._pear_source = getattr(frames[-1], "source", None)
        phase = self._phase
        if phase is Phase.IDLE:
            if m is not None:
                self._try_enter(now, inputs, m)
            return self._blend_a(now, m) if self._phase is Phase.A else None
        if phase is Phase.COOLDOWN:
            if now - self._since >= self._config.cooldown_s:
                self._phase = Phase.IDLE
                self._clear()
            return None
        if m is None:  # cannot happen once A began (the mirror command is kept), but never blend from nothing
            m = _arm(getattr(inputs.last_sent, "arm", None))
            if m is None:
                return None
        if phase is Phase.A:
            return self._tick_a(now, inputs, m)
        if phase is Phase.WAIT_ACK:
            return self._tick_wait_ack(now, m)
        if phase is Phase.PLAYING:
            return self._tick_playing(now, inputs, m)
        if phase is Phase.WAIT_DONE:
            return self._tick_wait_done(now, m)
        if phase is Phase.C:
            return self._blend_c(now, m)
        return None  # STOP_WAIT outside a wrap-up: the round is over

    # -- IDLE -> A ----------------------------------------------------------------

    def _try_enter(self, now: float, inputs, m: tuple[float, ...]) -> None:
        if self._round_disabled or self._port_suspect or self._client.busy:
            return
        if not getattr(inputs, "publish_enabled", False):
            return
        actor = inputs.actor
        if actor is None or not actor.present or actor.actor_id is None:
            return
        if self._config.require_person_frame and self._pear_source != PERSON_SOURCE:
            return
        status = self._port.latest()
        if status is None or not 0 <= now - status.received_at <= self._config.port_fresh_s:
            return
        if status.state not in _PORT_FREE:
            return
        cbf = self._cbf.latest()
        if cbf is None or not 0 <= now - cbf.received_at <= self._config.cbf_fresh_s or cbf.intervening:
            return
        for template, names in self._templates.items():
            label = self._labels.current(template, now)
            if label is None or label.actor_id != actor.actor_id or (template, label.episode) in self._attempted:
                continue
            candidates = [self._segments[name] for name in names if status.segments is None or name in status.segments]
            if not candidates:
                continue
            self._attempted.add((template, label.episode))
            segment = min(candidates, key=lambda seg: self._weighted(seg.P, m))
            self._enter_a(now, label, actor.actor_id, segment, m)
            return

    @staticmethod
    def _weighted(P: Sequence[float], m: Sequence[float]) -> float:
        return max(w * abs(p - q) for w, p, q in zip(SELECT_WEIGHTS, P, m, strict=False))

    def _enter_a(self, now: float, label: Label, actor: int, segment: Segment, m: tuple[float, ...]) -> None:
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
        self._hand_target = segment.hand_P
        self.entries += 1
        self._log(
            f"template enter {label.template} seg={segment.name} actor={actor} episode={label.episode} "
            f"d={self._entry_d:.3f} dist={self._entry_dist:.3f} T_A={self._t_a:.2f}"
        )

    # -- A ------------------------------------------------------------------------

    def _blend_a(self, now: float, m: tuple[float, ...]) -> tuple[float, ...]:
        alpha = _min_jerk((now - self._since) / self._t_a)[0]
        return _lerp(m, self._segment.P, alpha)

    def _tick_a(self, now: float, inputs, m: tuple[float, ...]) -> tuple[float, ...]:
        config, segment = self._config, self._segment
        elapsed = now - self._since
        self._a_elapsed = elapsed
        reason = self._label_check(now, inputs)
        cbf = self._cbf.latest()
        if reason is None and cbf is not None and cbf.received_at > self._since and cbf.intervening:
            reason = "a_cbf_hold"
        if reason is None and elapsed > self._t_a + config.a_timeout_margin_s:
            reason = "a_timeout"
        if reason is not None:
            s = elapsed / self._t_a
            value, speed, accel = _min_jerk(s)
            if s >= 1.0:
                speed = accel = 0.0
            return self._to_c(now, reason, m, segment.P, value, speed / self._t_a, accel / self._t_a**2)
        if elapsed >= self._t_a and self._arrived(now, inputs, cbf):
            try:
                self._seg_id = self._client.start(segment.name, now)
            except RuntimeError:
                return self._to_c(now, "port_busy", m, segment.P, 1.0)
            status = self._port.latest()
            self._status_serial = self._seen_serial = 0 if status is None else status.serial
            self._play_deadline = now + segment.t_in + segment.hold_max_s + segment.t_out + config.play_margin_s
            self._phase, self._since = Phase.WAIT_ACK, now
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
        return cbf is not None and 0 <= now - cbf.received_at <= config.cbf_fresh_s and not cbf.intervening

    def _label_check(self, now: float, inputs) -> str | None:
        """Why the person no longer asks for this template, or None."""
        actor = inputs.actor
        if actor is None or actor.actor_id != self._actor:
            return "actor_changed"
        label = self._labels.current(self._label.template, now)
        if label is None:
            return "label_lost"
        if label.episode != self._label.episode or label.actor_id != self._actor:
            return "label_lost"
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
            self.played += 1
            self._hand_target = self._status_hand()
            return self._status_arm()
        if status is not None and status.state in TERMINAL:
            return self._to_c(now, f"{status.state.lower()}:{status.reason}", m, self._segment.P, 1.0)
        if now - self._since > self._config.ack_timeout_s:
            self._client.release(self._seg_id, fast=True)
            self._port_suspect = True
            return self._to_c(now, "no_ack", m, self._segment.P, 1.0)
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
        if status is None or now - status.received_at > config.status_stale_s:
            self._client.release(self._seg_id, fast=True)
            return self._to_wait_done(now, "status_stale", segment.t_out_stop)
        if self._port_lost(status):
            self._client.abandon()
            return self._to_c(now, "port_lost", m, arm, 1.0)
        if now > self._play_deadline:
            self._client.release(self._seg_id, fast=True)
            return self._to_wait_done(now, "play_timeout", segment.t_out_stop)
        if status.seg_id == self._seg_id:
            if status.state in TERMINAL:
                return self._to_c(now, f"{status.state.lower()}:{status.reason}", m, arm, 1.0)
            if status.state == "EXIT_B":
                return self._to_wait_done(now, status.reason or "exit", segment.t_out)
            if status.state == "HOLD" and self._hold_since is None:
                self._hold_since = now
        if not self._label_lost and self._label_check(now, inputs) is not None:
            self._label_lost = True
        if self._label_lost and self._hold_since is not None and now - self._hold_since >= config.min_hold_s:
            self._client.release(self._seg_id, fast=False)
            self._released = True
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
        stale = status is None or now - status.received_at > config.status_stale_s
        if stale:
            self._deadline = min(self._deadline, now + segment.t_out_stop + config.done_margin_s)
        elif status.seg_id == self._seg_id and status.state in TERMINAL:
            reason = self._wait_reason if status.state == "DONE" else f"{status.state.lower()}:{status.reason}"
            return self._to_c(now, reason, m, arm, 1.0)
        elif self._port_lost(status):
            self._client.abandon()
            return self._to_c(now, "port_lost", m, arm, 1.0)
        if now >= self._deadline:
            # The port never said it finished: one more fast release, and the keepalive ends soon after,
            # so the data plane's own keepalive timeout stops it (a port that plays on is never kept alive).
            self._client.wind_down(now, config.wind_down_s)
            return self._to_c(now, "status_stale" if stale else "done_timeout", m, arm, 1.0)
        return arm

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
    ) -> tuple[float, ...]:
        config = self._config
        self._exit_log(reason, now)
        self._anchor = tuple(anchor)
        dist = 0.0 if m is None else _farthest(self._anchor, m)
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
        self._c_curve = curve
        self._phase, self._since = Phase.C, now
        self._hand_target = None
        if m is None:
            return self._anchor
        return self._blend_c(now, m)

    def _blend_c(self, now: float, m: tuple[float, ...]) -> tuple[float, ...]:
        t = now - self._since
        beta = min(max(self._c_curve.at(t), 0.0), 1.0)
        arm = _lerp(m, self._anchor, beta)
        if t >= self._c_curve.duration:
            self._phase, self._since = Phase.COOLDOWN, now
            self._label = self._segment = None
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

    def _exit_log(self, reason: str, now: float) -> None:
        self._count_exit(reason)
        name = self._label.template if self._label is not None else "?"
        seg = self._segment.name if self._segment is not None else "?"
        self._log(
            f"template exit {name} seg={seg} phase={self._phase.value} reason={reason} "
            f"d={self._entry_d:.3f} dist={self._entry_dist:.3f} T_A={self._t_a:.2f} "
            f"a_elapsed={self._a_elapsed:.2f} phase_elapsed={now - self._since:.2f} "
            f"total={now - self._entered_at if self._entered_at is not None else 0.0:.2f}"
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
            "last_exit": self._last_exit,
            "round_disabled": self._round_disabled,
            "port_suspect": self._port_suspect,
        }

    def snapshot(self) -> dict:
        return dict(self._snap)
