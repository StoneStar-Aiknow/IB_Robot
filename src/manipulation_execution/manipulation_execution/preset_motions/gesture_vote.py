"""Vote the hand gesture stream (``x2_hand_gesture/v1``) into heart / holy / xsign states. **No ROS.**

Input: one message per gesture-model run (hand_perception_node, /hri/hand/gesture),
each with the camera stamp, the PearFrame actor, the best class and its score.

- Only the current actor's runs count (``set_actor``; a change, or losing the
  actor, resets everything). A run counts for its class when the score is at
  least ``min_score``; below that it is a ``none`` run.
- The window is the last ``window_s`` of camera time (by ``stamp_ns``), and of
  receive time: a stream that stops ages out instead of freezing the vote.
- A class becomes active when it has at least ``min_frames`` runs in the window
  (or ``min_frames_one_hand`` one-hand runs, ``"crop": "one"``: those come at
  most 5 Hz, so a 0.6 s window holds only 3 of them once there is any jitter)
  AND more than half of the window's runs.
- An active class goes inactive when no run of it has come for ``exit_s``
  (``exit_s_one_hand`` when its last run was a one-hand run), by receive time
  (the node's tick) or by camera time (newer runs of something else), or on an
  actor change / reset. Its runs then leave the window, so entering again takes
  fresh runs (after a short dropout the old ones would re-enter it at once).
- A camera stamp going backwards resets the vote; a repeated stamp is ignored;
  a run older than the ``barrier`` (set on a session change) or staler than
  ``max_source_age_s`` against ROS time never counts.

Reasons are the pose reasons of preset_motions.match_signal, so the matcher node
can put them in ``/preset_motions/state`` as they are.
"""

from __future__ import annotations

import collections
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, fields

SCHEMA = "x2_hand_gesture/v1"
GESTURES = ("heart", "holy", "xsign")
NONE = "none"
_CLASSES = frozenset((*GESTURES, NONE))


@dataclass(frozen=True)
class VoteConfig:
    window_s: float = 0.6
    min_frames: int = 4
    min_frames_one_hand: int = 3  # one-hand runs (<= 5 Hz) that are enough on their own
    min_score: float = 0.5
    exit_s: float = 0.3
    exit_s_one_hand: float = 0.5  # exit_s after a one-hand run: 2.5 periods at 5 Hz, as 0.3 s is at 15 Hz
    max_source_age_s: float = 0.8
    future_tolerance_s: float = 0.1

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v) or v <= 0:
                raise ValueError(f"gesture vote {f.name} must be a finite positive number, got {v!r}")
        for name in ("min_frames", "min_frames_one_hand"):
            if not isinstance(getattr(self, name), int):
                raise ValueError(f"gesture vote {name} must be an integer, got {getattr(self, name)!r}")
        if self.min_score > 1:
            raise ValueError(f"gesture vote min_score must be at most 1, got {self.min_score!r}")


@dataclass(frozen=True)
class VoteState:
    template_name: str
    active: bool
    entered: bool
    exited: bool
    reason: str  # one of match_signal._POSE_REASONS
    votes: int  # runs of this class in the window
    runs: int  # all runs in the window
    source_stamp_ns: int | None  # newest counted run, any class


def _int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse(message) -> tuple[int, int, str, float]:
    """(stamp_ns, actor_id, cls, score) of one gesture message; ValueError if it is not one."""
    if not isinstance(message, Mapping) or message.get("schema") != SCHEMA:
        raise ValueError("not an x2_hand_gesture/v1 message")
    stamp, actor, cls, score = (message.get(k) for k in ("stamp_ns", "actor_id", "cls", "score"))
    if not _int(stamp) or stamp < 0 or not _int(actor):
        raise ValueError("stamp_ns / actor_id must be integers")
    if cls not in _CLASSES:
        raise ValueError(f"unknown class {cls!r}")
    if isinstance(score, bool) or not isinstance(score, int | float) or not 0.0 <= score <= 1.0:
        raise ValueError("score must be in [0, 1]")
    return stamp, actor, cls, float(score)


class GestureVote:
    """Callers serialise all calls (the matcher node's one callback group)."""

    def __init__(self, config: VoteConfig | None = None):
        self.config = config or VoteConfig()
        self.actor_id: int | None = None
        self.barrier_ns = -1
        self.last_reason = ""  # why the last observe did not count, for the status line
        self._frames: collections.deque = collections.deque()  # (stamp_ns, cls, mono_s, one_hand)
        self._last_stamp: int | None = None
        self._active = dict.fromkeys(GESTURES, False)
        self._last_pos: dict[str, tuple[int, float, bool] | None] = dict.fromkeys(GESTURES)  # (stamp, mono, one)
        self._pending_exit: set[str] = set()

    def reset(self, *, barrier: bool = False) -> None:
        """Forget the window; active classes exit. ``barrier``: runs up to the newest seen stamp never count."""
        if barrier and self._last_stamp is not None:
            self.barrier_ns = max(self.barrier_ns, self._last_stamp)
        self._pending_exit |= {c for c, a in self._active.items() if a}
        self._frames.clear()
        self._last_stamp = None
        self._active = dict.fromkeys(GESTURES, False)
        self._last_pos = dict.fromkeys(GESTURES)

    def set_actor(self, actor_id: int | None) -> None:
        if actor_id != self.actor_id:
            self.reset()
            self.actor_id = actor_id

    def observe(self, message, *, now_ros_s: float, now_mono_s: float) -> bool:
        """One gesture message; True if it counted."""
        try:
            stamp, actor, cls, score = parse(message)
        except ValueError:
            self.last_reason = "message_invalid"
            return False
        cfg = self.config
        if self.actor_id is None or actor != self.actor_id:
            self.last_reason = "other_actor"
            return False
        if stamp <= self.barrier_ns:
            self.last_reason = "before_session"
            return False
        age = now_ros_s - stamp / 1e9
        if age > cfg.max_source_age_s or age < -cfg.future_tolerance_s:
            self.last_reason = "source_stale" if age > 0 else "source_future"
            return False
        if self._last_stamp is not None:
            if stamp == self._last_stamp:
                self.last_reason = "duplicate"
                return False
            if stamp < self._last_stamp:
                self.reset()
        self._last_stamp = stamp
        counted = cls if cls != NONE and score >= cfg.min_score else NONE
        one = message.get("crop") == "one"
        self._frames.append((stamp, counted, now_mono_s, one))
        if counted != NONE:
            self._last_pos[counted] = (stamp, now_mono_s, one)
        self.last_reason = ""
        return True

    def update(self, *, now_mono_s: float) -> dict[str, VoteState]:
        """The state of each gesture now (receive time = the caller's monotonic clock)."""
        cfg = self.config
        window_ns = int(cfg.window_s * 1e9)
        frames = self._frames
        if self._last_stamp is not None:
            while frames and frames[0][0] < self._last_stamp - window_ns:
                frames.popleft()
        while frames and now_mono_s - frames[0][2] > cfg.window_s:
            frames.popleft()
        runs = len(frames)
        counts = collections.Counter(c for _, c, _, _ in frames)
        one_hand = collections.Counter(c for _, c, _, one in frames if one)
        out = {}
        for name in GESTURES:
            votes = counts[name]
            was = self._active[name]
            entered = exited = False
            pos = self._last_pos[name]
            exit_s = cfg.exit_s_one_hand if pos is not None and pos[2] else cfg.exit_s
            # A run of this class within exit_s, by both clocks: keeps it active, and is needed to
            # enter (else the window's older votes would re-enter it right after a time exit).
            fresh = (
                pos is not None
                and now_mono_s - pos[1] <= exit_s
                and (self._last_stamp is None or self._last_stamp - pos[0] <= int(exit_s * 1e9))
            )
            enough = votes >= cfg.min_frames or one_hand[name] >= cfg.min_frames_one_hand
            if was and not fresh:
                self._active[name] = False
                exited = True
                # Its runs leave the window: entering again takes min_frames fresh runs.
                self._frames = collections.deque(f for f in self._frames if f[1] != name)
                votes, runs = 0, len(self._frames)
            elif not was and fresh and enough and 2 * votes > runs:
                self._active[name] = entered = True
            if name in self._pending_exit and not self._active[name]:
                exited = True
            active = self._active[name]
            if entered:
                reason = "matched"
            elif active:
                reason = "holding"
            elif exited:
                reason = "pose_exited"
            elif votes:
                reason = "confirming"
            else:
                reason = "pose_not_matched"
            out[name] = VoteState(name, active, entered, exited, reason, votes, runs, self._last_stamp)
        self._pending_exit.clear()
        return out


OVERHEAD = "overhead_heart"
TEMPLATE_ORDER = (OVERHEAD, *GESTURES)  # the latch's tie order, and the publish order


class TemplateLatch:
    """At most one template active at a time.

    ``overhead_heart`` is the pose rule's state, unchanged: it is active exactly
    when the rule is, and takes the latch from a gesture. ``heart`` (the hand
    heart) never counts while the overhead heart does. Among the rest the first
    to activate keeps the latch until it goes inactive; on a tie the earlier
    name in ``TEMPLATE_ORDER`` wins.
    """

    def __init__(self):
        self.latched: str | None = None

    def update(self, raw: Mapping[str, bool]) -> dict[str, bool]:
        raw = {t: raw.get(t) is True for t in TEMPLATE_ORDER}
        raw["heart"] = raw["heart"] and not raw[OVERHEAD]
        if raw[OVERHEAD]:
            self.latched = OVERHEAD
        elif self.latched is None or not raw[self.latched]:
            self.latched = next((t for t in TEMPLATE_ORDER if raw[t]), None)
        return {t: t == self.latched for t in TEMPLATE_ORDER}


class GestureFusion:
    """The matcher node's gesture side: actor, session, vote and latch -> one result per gesture template.

    Results are ``match_state``-ready (``events.match_state``): template_name,
    session_epoch, actor_id, source_stamp_ns, sequence, active, reason.
    """

    def __init__(
        self,
        config: VoteConfig | None = None,
        *,
        actor_max_age_s: float = 1.0,
        remote_source: bool = False,
        source_started_ns: int | None = None,
    ):
        self.vote = GestureVote(config)
        self.latch = TemplateLatch()
        self.actor_max_age_s = float(actor_max_age_s)
        self._actor: tuple[int | None, float] | None = None  # (actor id or None for absent, received mono)
        self._session: tuple[int, bool] | None = None
        self._sequence = dict.fromkeys(GESTURES, 0)
        self.counted = self.ignored = 0
        self.source_gate = None
        self.source_session_token = ""
        if remote_source:
            from .gesture_source import GestureSourceGate

            self.source_gate = GestureSourceGate(
                time.monotonic_ns() if source_started_ns is None else source_started_ns
            )

    def set_session(self, epoch: int, active: bool, *, now_mono_s: float | None = None) -> None:
        """Every (epoch, active) change forgets the vote and bars the runs seen so far."""
        if (epoch, active) != self._session:
            self.vote.reset(barrier=True)
            self._session = (epoch, active)
            if self.source_gate is not None:
                now = time.monotonic() if now_mono_s is None else now_mono_s
                self.source_gate.invalidate(int(now * 1e9))

    def set_source_session_token(self, token: str, *, now_mono_s: float | None = None):
        """A canonical session transition, never a remote subscriber's local epoch."""
        if self.source_gate is not None and self.source_session_token and token != self.source_session_token:
            self._reset_source_votes()
            now = time.monotonic() if now_mono_s is None else now_mono_s
            self.source_gate.invalidate(int(now * 1e9))
        self.source_session_token = token

    def _reset_source_votes(self):
        self.vote.reset()
        if self.latch.latched in GESTURES:
            self.latch.latched = None

    def observe_source(self, text, *, now_mono_s: float) -> bool:
        if self.source_gate is None:
            return False
        accepted, reset = self.source_gate.observe(
            text,
            now_ns=int(now_mono_s * 1e9),
            token=self.source_session_token,
        )
        if reset:
            self._reset_source_votes()
        return accepted

    def observe_actor(self, message, *, now_mono_s: float) -> None:
        """ActorState: the actor counts while present, start-ready and heard within ``actor_max_age_s``."""
        try:
            actor_id = message.actor_id
            if isinstance(actor_id, bool) or not isinstance(actor_id, int):
                raise ValueError
            present = message.actor_present is True and message.start_ready is True and actor_id >= 0
        except (AttributeError, ValueError):
            present, actor_id = False, None
        self._actor = (actor_id if present else None, float(now_mono_s))

    def current_actor(self, now_mono_s: float) -> int | None:
        if self._actor is None or self._actor[0] is None:
            return None
        return self._actor[0] if 0 <= now_mono_s - self._actor[1] <= self.actor_max_age_s else None

    def observe(self, data, *, now_ros_s: float, now_mono_s: float) -> bool:
        """One /hri/hand/gesture message (JSON text or mapping); only while the session is active."""
        if self._session is None or not self._session[1]:
            self.ignored += 1
            return False
        self.vote.set_actor(self.current_actor(now_mono_s))
        try:
            if self.source_gate is not None:
                from embodied_common.hand_gesture_contract import load_json

                message = load_json(data, SCHEMA) if isinstance(data, str) else data
                if not self.source_gate.allows(
                    message,
                    now_ns=int(now_mono_s * 1e9),
                    token=self.source_session_token,
                ):
                    self.ignored += 1
                    return False
            else:
                message = json.loads(data) if isinstance(data, str) else data
                if isinstance(message, Mapping) and ("requester_boot_id" in message or "stream_generation" in message):
                    self.vote.last_reason = "remote_source_not_configured"
                    self.ignored += 1
                    return False
        except (ValueError, TypeError, RecursionError, UnicodeError):
            message = None
        ok = self.vote.observe(message, now_ros_s=now_ros_s, now_mono_s=now_mono_s)
        if ok:
            self.counted += 1
            if self.source_gate is not None:
                self.source_gate.commit(message)
        else:
            self.ignored += 1
        return ok

    def results(self, *, overhead_active: bool, session_active: bool, epoch: int, now_mono_s: float) -> list:
        """One result per gesture template, in ``GESTURES`` order; each call advances their sequences."""
        actor = self.current_actor(now_mono_s)
        if self.source_gate is not None and self.source_gate.tick(int(now_mono_s * 1e9)):
            self._reset_source_votes()
        self.vote.set_actor(actor)
        votes = self.vote.update(now_mono_s=now_mono_s)
        raw = {name: session_active and state.active for name, state in votes.items()}
        raw[OVERHEAD] = session_active and overhead_active
        active = self.latch.update(raw)
        out = []
        for name in GESTURES:
            state = votes[name]
            reason = state.reason
            if state.active and not active[name]:
                reason = "confirming"  # held back by the latch or the overhead heart
            self._sequence[name] += 1
            out.append(
                GestureResult(
                    template_name=name,
                    session_epoch=int(epoch),
                    actor_id=actor,
                    source_stamp_ns=state.source_stamp_ns,
                    sequence=self._sequence[name],
                    active=active[name],
                    reason=reason,
                )
            )
        return out


@dataclass(frozen=True)
class GestureResult:
    template_name: str
    session_epoch: int
    actor_id: int | None
    source_stamp_ns: int | None
    sequence: int
    active: bool
    reason: str
