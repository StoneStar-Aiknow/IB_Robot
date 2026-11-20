"""Client side of the data plane's playback port (``template-play-M0-interface.md``).

* :class:`PortMonitor`: the newest ``/aimdk_dataplane/template_status``
  (``x2_template_status/v1``) with the time it was received;
* :class:`CbfMonitor`: whether the data plane's CBF is holding or releasing,
  from ``/aimdk_runtime/dataplane/status`` (``cbf.stuck_s``, ``releasing``,
  ``stuck_collision``), the status's own time and safety epoch, the recent
  statuses in order (one older than the newest is dropped) and the share of
  held CBF calls over a window;
* :class:`PlayClient`: ``start`` / ``release`` from the blender, ``pump``
  from the node's tick for the keepalive; ``a_cap`` from the blender in A
  (the (b) flag: the CBF's releases target the hold's first request).

The client's obligations (M0 §2.1): a keepalive at least every 50 ms from the
start until the status reports DONE, REJECTED or STOPPED for that ``seg_id``,
also after a release; never a start while a segment is playing. ``abandon``
stops the keepalive at once (a blender error): the data plane then times out
after 150 ms and plays back to P on its own. The keepalive also stops after
``give_up_s`` without any status, so a dead status topic does not keep it
going for ever; by then the data plane has exited on hold_max_s anyway.

All three take their callbacks' messages on any thread; nothing here raises
on a bad message.
"""

from __future__ import annotations

import json
import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace

from manipulation_execution.mirror_session.endpoints import ARM_JOINT_COUNT, HAND_JOINT_COUNT

PLAY_SCHEMA = "x2_template_play/v1"
STATUS_SCHEMA = "x2_template_status/v1"
STATES = ("IDLE", "ENTER_B", "HOLD", "EXIT_B", "DONE", "REJECTED", "STOPPED")
PLAYING = ("ENTER_B", "HOLD", "EXIT_B")
TERMINAL = ("DONE", "REJECTED", "STOPPED")
KEEPALIVE_SLACK_S = 0.005  # a tick that comes a little early still sends: 0.02 at 50 Hz is every tick
CBF_HISTORY = 400  # data-plane statuses kept: 20 s at 20 Hz


@dataclass(frozen=True)
class PortStatus:
    seg_id: int | None
    seg: str | None
    state: str
    reason: str
    P_actual: tuple[float, ...] | None
    hand_P: tuple[float, ...] | None
    segments: tuple[str, ...] | None
    received_at: float
    serial: int  # counts received statuses, so the blender can tell a new one
    a_cap: bool | None = None  # the (b) flag is in force on the data plane; None: the field is not sent


def _vector(values, length: int) -> tuple[float, ...] | None:
    if not isinstance(values, list) or len(values) != length:
        return None
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError):
        return None
    return out if all(map(math.isfinite, out)) else None


def decode_status(data: str, now: float, serial: int) -> PortStatus:
    """ValueError for anything that is not a well-formed status."""
    message = json.loads(data)
    if not isinstance(message, dict) or message.get("schema") != STATUS_SCHEMA:
        raise ValueError("schema")
    state = message.get("state")
    if state not in STATES:
        raise ValueError("state")
    seg_id = message.get("seg_id")
    if seg_id is not None and (not isinstance(seg_id, int) or isinstance(seg_id, bool) or seg_id < 0):
        raise ValueError("seg_id")
    segments = message.get("segments")
    segments = tuple(s for s in segments if isinstance(s, str)) if isinstance(segments, list) else None
    seg = message.get("seg")
    reason = message.get("reason")
    a_cap = message.get("a_cap")
    return PortStatus(
        seg_id=seg_id,
        seg=seg if isinstance(seg, str) else None,
        state=state,
        reason=reason if isinstance(reason, str) else "",
        P_actual=_vector(message.get("P_actual"), ARM_JOINT_COUNT),
        hand_P=_vector(message.get("hand_P"), HAND_JOINT_COUNT),
        segments=segments,
        received_at=now,
        serial=serial,
        a_cap=a_cap if isinstance(a_cap, bool) else None,
    )


class PortMonitor:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest: PortStatus | None = None
        self._serial = 0
        self.malformed = 0

    def observe(self, data: str, now: float) -> None:
        with self._lock:
            try:
                status = decode_status(data, now, self._serial + 1)
            except Exception:
                self.malformed += 1
                return
            self._serial = status.serial
            self._latest = status

    def latest(self) -> PortStatus | None:
        with self._lock:
            return self._latest


@dataclass(frozen=True)
class CbfState:
    intervening: bool  # holding (stuck_s > 0, stuck_collision) or releasing
    received_at: float
    detail: str
    # The rest only describes the status; a field the data plane does not send is None.
    serial: int = 0  # counts received statuses, so the blender can tell a new one
    at: float | None = None  # the data plane's CLOCK_MONOTONIC when it published (time.monotonic's clock)
    stuck_s: float = 0.0
    releasing: bool = False
    collision: bool = False
    safety_epoch: int | None = None  # moves only on mode / owner / admission / clear / safe changes
    control_seq: int | None = None  # moves with every control heartbeat (0.1 s): never a revocation sign
    arm_owner: int | None = None
    mirror_active: bool | None = None
    held: int | None = None  # cumulative held ticks
    calls: int | None = None  # cumulative CBF calls
    hold_release_after_s: float | None = None
    release_episodes: int | None = None
    collision_release_episodes: int | None = None
    has_cbf: bool = False  # the status had a cbf section (without one the CBF is off)
    restarted: bool = False  # its counters went back: the data plane restarted, an equal epoch proves nothing


def _count(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _seconds(value) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value):
        return None
    return float(value)


class CbfMonitor:
    """Without a ``cbf`` section (CBF off) the CBF never intervenes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: CbfState | None = None
        self._serial = 0
        self._history: deque[CbfState] = deque(maxlen=CBF_HISTORY)
        self.malformed = 0
        self.reordered = 0

    def observe(self, data: str, now: float) -> None:
        try:
            message = json.loads(data)
            cbf = message.get("cbf")
            top = {
                "at": _seconds(message.get("at")),
                "safety_epoch": _count(message.get("safety_epoch")),
                "control_seq": _count(message.get("control_seq")),
                "arm_owner": _count(message.get("arm_owner")),
            }
            if cbf is None:
                state = CbfState(False, now, "no cbf", **top)
            else:
                stuck = float(cbf.get("stuck_s") or 0.0)
                releasing = cbf.get("releasing") is True
                collision = cbf.get("stuck_collision") is True
                active = cbf.get("mirror_active")
                state = CbfState(
                    (stuck > 0.0) or releasing or collision,
                    now,
                    f"stuck_s={stuck:.2f} releasing={releasing} stuck_collision={collision}",
                    stuck_s=stuck,
                    releasing=releasing,
                    collision=collision,
                    mirror_active=active if isinstance(active, bool) else None,
                    held=_count(cbf.get("held")),
                    calls=_count(cbf.get("calls")),
                    hold_release_after_s=_seconds(cbf.get("hold_release_after_s")),
                    release_episodes=_count(cbf.get("release_episodes")),
                    collision_release_episodes=_count(cbf.get("collision_release_episodes")),
                    has_cbf=True,
                    **top,
                )
        except Exception:
            with self._lock:
                self.malformed += 1
            return
        with self._lock:
            before = self._latest
            if before is not None and None not in (before.at, state.at) and state.at < before.at:
                self.reordered += 1  # older than the newest: one publisher never does this, a queue might
                return
            self._serial += 1
            restarted = before is not None and None not in (before.calls, state.calls) and state.calls < before.calls
            if restarted:
                self._history.clear()  # the data plane restarted: its counters began again
            self._latest = replace(state, serial=self._serial, restarted=restarted)
            self._history.append(self._latest)

    def latest(self) -> CbfState | None:
        with self._lock:
            return self._latest

    def after(self, serial: int) -> list[CbfState]:
        """The statuses received after the one numbered ``serial``, oldest first (the newest CBF_HISTORY)."""
        with self._lock:
            return [state for state in self._history if state.serial > serial]

    def hold_ratio(self, now: float, window_s: float) -> float | None:
        """Held CBF calls / CBF calls over the last ``window_s``, from the cumulative counters.

        0 while the newest status has no cbf section (the CBF is off). None when
        it cannot be told: no status for half the window, less than half the
        window of statuses since the counters (re)started, or counters missing.
        """
        with self._lock:
            history = list(self._history)
        if not history or now - history[-1].received_at > window_s / 2.0:
            return None
        newest = history[-1]
        if not newest.has_cbf:
            return 0.0
        base = None
        for state in history:  # the newest status at or before the window's start, else the oldest
            if state.received_at <= now - window_s or base is None:
                base = state
        if newest.received_at - base.received_at < window_s / 2.0:
            return None
        if None in (base.held, base.calls, newest.held, newest.calls):
            return None
        calls = newest.calls - base.calls
        if calls <= 0:
            return 0.0  # nothing was filtered: nothing was held
        return min(max((newest.held - base.held) / calls, 0.0), 1.0)


class PlayClient:
    def __init__(
        self,
        publish: Callable[[str], None],
        monitor: PortMonitor,
        keepalive_period_s: float = 0.02,
        give_up_s: float = 10.0,
        first_seg_id: int = 0,
    ):
        self._publish = publish
        self._monitor = monitor
        self._period = float(keepalive_period_s)
        self._give_up_s = float(give_up_s)
        self._lock = threading.Lock()
        self._next_id = int(first_seg_id) % (1 << 31)
        self._seg_id: int | None = None  # keepalive owner; None: nothing to keep alive
        self._last_keepalive = -math.inf
        self._since = -math.inf  # start, or the segment's newest status, whichever is later
        self._linger_until = math.inf  # wind_down: the keepalive ends here at the latest
        self.sent = {"start": 0, "keepalive": 0, "release": 0}
        self.publish_errors = 0

    def _send(self, message: dict) -> None:
        message["schema"] = PLAY_SCHEMA
        try:
            self._publish(json.dumps(message, sort_keys=True))
            self.sent[message["op"]] = self.sent.get(message["op"], 0) + 1
        except Exception:
            self.publish_errors += 1

    @property
    def busy(self) -> bool:
        """A segment is still being kept alive: no new start."""
        with self._lock:
            return self._seg_id is not None

    def start(self, seg: str, now: float) -> int:
        with self._lock:
            if self._seg_id is not None:
                raise RuntimeError(f"segment {self._seg_id} is still playing")
            seg_id, self._next_id = self._next_id, (self._next_id + 1) % (1 << 31)
            self._seg_id, self._last_keepalive, self._since = seg_id, now, now
            self._linger_until = math.inf
            self._send({"op": "start", "seg": seg, "seg_id": seg_id})
            return seg_id

    def release(self, seg_id: int, fast: bool) -> None:
        with self._lock:
            if self._seg_id == seg_id:
                self._send({"op": "release", "seg_id": seg_id, "fast": bool(fast)})

    def a_cap(self, on: bool, epoch: int) -> None:
        """The (b) flag for the data plane's CBF, in the safety epoch A was entered in. No segment needed."""
        with self._lock:
            self._send({"op": "a_cap", "on": bool(on), "epoch": int(epoch)})

    def abandon(self) -> None:
        with self._lock:
            self._seg_id = None

    def wind_down(self, now: float, linger_s: float) -> None:
        """Release the segment fast and keep it alive ``linger_s`` at most, until its terminal status."""
        with self._lock:
            if self._seg_id is None:
                return
            self._send({"op": "release", "seg_id": self._seg_id, "fast": True})
            self._linger_until = min(self._linger_until, now + linger_s)

    def pump(self, now: float) -> None:
        """Node tick: the keepalive, until a terminal status for the segment."""
        status = self._monitor.latest()
        with self._lock:
            if self._seg_id is None:
                return
            # Only the segment's own statuses count: a restarted port's IDLE never keeps it alive.
            if status is not None and status.received_at > self._since and status.seg_id == self._seg_id:
                self._since = status.received_at
                if status.state in TERMINAL:
                    self._seg_id = None
                    return
            if now - self._since > self._give_up_s or now >= self._linger_until:
                self._seg_id = None
                return
            if now - self._last_keepalive >= self._period - KEEPALIVE_SLACK_S:
                self._last_keepalive = now
                self._send({"op": "keepalive", "seg_id": self._seg_id})

    def snapshot(self) -> dict:
        with self._lock:
            return {"seg_id": self._seg_id, **self.sent, "publish_errors": self.publish_errors}
