"""Client side of the data plane's playback port (``template-play-M0-interface.md``).

* :class:`PortMonitor`: the newest ``/aimdk_dataplane/template_status``
  (``x2_template_status/v1``) with the time it was received;
* :class:`CbfMonitor`: whether the data plane's CBF is holding or releasing,
  from ``/aimdk_runtime/dataplane/status`` (``cbf.stuck_s``, ``releasing``,
  ``stuck_collision``);
* :class:`PlayClient`: ``start`` / ``release`` from the blender, ``pump``
  from the node's tick for the keepalive.

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
from collections.abc import Callable
from dataclasses import dataclass

from manipulation_execution.mirror_session.endpoints import ARM_JOINT_COUNT, HAND_JOINT_COUNT

PLAY_SCHEMA = "x2_template_play/v1"
STATUS_SCHEMA = "x2_template_status/v1"
STATES = ("IDLE", "ENTER_B", "HOLD", "EXIT_B", "DONE", "REJECTED", "STOPPED")
PLAYING = ("ENTER_B", "HOLD", "EXIT_B")
TERMINAL = ("DONE", "REJECTED", "STOPPED")
KEEPALIVE_SLACK_S = 0.005  # a tick that comes a little early still sends: 0.02 at 50 Hz is every tick


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


class CbfMonitor:
    """Without a ``cbf`` section (CBF off) the CBF never intervenes."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest: CbfState | None = None
        self.malformed = 0

    def observe(self, data: str, now: float) -> None:
        try:
            cbf = json.loads(data).get("cbf")
            if cbf is None:
                state = CbfState(False, now, "no cbf")
            else:
                stuck = float(cbf.get("stuck_s") or 0.0)
                releasing = cbf.get("releasing") is True
                collision = cbf.get("stuck_collision") is True
                state = CbfState(
                    (stuck > 0.0) or releasing or collision,
                    now,
                    f"stuck_s={stuck:.2f} releasing={releasing} stuck_collision={collision}",
                )
        except Exception:
            with self._lock:
                self.malformed += 1
            return
        with self._lock:
            self._latest = state

    def latest(self) -> CbfState | None:
        with self._lock:
            return self._latest


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
        self._since = -math.inf  # start, or the newest status, whichever is later
        self.sent = {"start": 0, "keepalive": 0, "release": 0}
        self.publish_errors = 0

    def _send(self, message: dict) -> None:
        message["schema"] = PLAY_SCHEMA
        try:
            self._publish(json.dumps(message, sort_keys=True))
            self.sent[message["op"]] += 1
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
            self._send({"op": "start", "seg": seg, "seg_id": seg_id})
            return seg_id

    def release(self, seg_id: int, fast: bool) -> None:
        with self._lock:
            if self._seg_id == seg_id:
                self._send({"op": "release", "seg_id": seg_id, "fast": bool(fast)})

    def abandon(self) -> None:
        with self._lock:
            self._seg_id = None

    def pump(self, now: float) -> None:
        """Node tick: the keepalive, until a terminal status for the segment."""
        status = self._monitor.latest()
        with self._lock:
            if self._seg_id is None:
                return
            if status is not None and status.received_at > self._since:
                self._since = status.received_at
                if status.seg_id == self._seg_id and status.state in TERMINAL:
                    self._seg_id = None
                    return
            if now - self._since > self._give_up_s:
                self._seg_id = None
                return
            if now - self._last_keepalive >= self._period - KEEPALIVE_SLACK_S:
                self._last_keepalive = now
                self._send({"op": "keepalive", "seg_id": self._seg_id})

    def snapshot(self) -> dict:
        with self._lock:
            return {"seg_id": self._seg_id, **self.sent, "publish_errors": self.publish_errors}
