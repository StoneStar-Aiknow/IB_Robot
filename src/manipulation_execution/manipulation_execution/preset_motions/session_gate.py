"""Fresh, ordered mirror-session evidence for pose-event admission only.

This gate does not acquire robot ownership or command motion. After its epoch
changes, the adapter must call ``PoseMatcher.set_session`` and then ``tick``
with the current clocks before feeding another PEAR frame. That establishes
the matcher source-time barrier against a previous round's queued frames.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from numbers import Integral, Real

MAX_STATE_AGE_S = 2.0
FUTURE_TOLERANCE_S = 0.1


def _number(value) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("clock and stamp values must be finite numbers")
    return float(value)


class SessionGate:
    """Accept only fresh MIRRORING/running ownership heartbeats.

    Callers serialize ``observe`` and ``permit`` with PEAR and actor callbacks.
    Duplicate sequence numbers never refresh freshness. A decreased sequence
    invalidates the current epoch and needs a later increasing observation
    before admission can resume.
    """

    def __init__(self):
        self._epoch = 0
        self._active = False
        self._last_seq: int | None = None
        self._stamp_s: float | None = None
        self._received_s: float | None = None
        self._last_mono_s: float | None = None
        self.reason = "awaiting_session"

    def _off(self, reason: str, *, reset_epoch: bool = False) -> tuple[int, bool]:
        self._active = False
        self.reason = reason
        if reset_epoch:
            self._epoch += 1
        return self._epoch, False

    def _clocks(self, now_ros_s, now_mono_s) -> tuple[float, float]:
        ros, mono = _number(now_ros_s), _number(now_mono_s)
        if ros < 0:
            raise ValueError("ROS time must be nonnegative")
        if self._last_mono_s is not None and mono < self._last_mono_s:
            self._off("monotonic_clock_reset", reset_epoch=True)
            self._stamp_s = self._received_s = None
            self._last_seq = None
        self._last_mono_s = mono
        return ros, mono

    def observe(self, state_json, *, now_ros_s: float, now_mono_s: float) -> tuple[int, bool]:
        """Consume one actual state callback, never a repeatedly cached state."""
        try:
            ros, mono = self._clocks(now_ros_s, now_mono_s)
            state = json.loads(state_json) if isinstance(state_json, str) else state_json
            if not isinstance(state, Mapping):
                raise ValueError("mirror state must be a JSON object")
            seq = state["seq"]
            if isinstance(seq, bool) or not isinstance(seq, Integral) or seq < 0:
                raise ValueError("state sequence must be a nonnegative integer")
            stamp = _number(state["stamp"])
            if stamp < 0:
                raise ValueError("state stamp must be nonnegative")
            if type(state["holding"]) is not bool or type(state["releasing"]) is not bool:
                raise ValueError("ownership flags must be booleans")
            if state["state"] not in ("WAIT", "STANDBY", "MIRRORING"):
                raise ValueError("unknown session state")
            if not isinstance(state["phase"], str):
                raise ValueError("session phase must be a string")
        except (KeyError, TypeError, ValueError, OverflowError):
            return self._off("invalid_session")

        if self._last_seq is not None:
            if seq == self._last_seq:
                return self.permit(now_ros_s=ros, now_mono_s=mono)
            if seq < self._last_seq:
                self._last_seq = int(seq)
                self._stamp_s, self._received_s = stamp, mono
                return self._off("sequence_restart", reset_epoch=True)

        self._last_seq = int(seq)
        self._stamp_s, self._received_s = stamp, mono
        age = ros - stamp
        if age < -FUTURE_TOLERANCE_S:
            return self._off("session_future")
        if age > MAX_STATE_AGE_S:
            return self._off("session_stale")
        running = (
            state["state"] == "MIRRORING"
            and state["phase"] == "running"
            and state["holding"]
            and not state["releasing"]
        )
        if not running:
            return self._off("not_running")
        if not self._active:
            self._epoch += 1
        self._active = True
        self.reason = "running"
        return self._epoch, True

    def permit(self, *, now_ros_s: float, now_mono_s: float) -> tuple[int, bool]:
        """Recheck both clocks without refreshing the received heartbeat."""
        try:
            ros, mono = self._clocks(now_ros_s, now_mono_s)
        except (TypeError, ValueError, OverflowError):
            return self._off("invalid_clock")
        if not self._active:
            return self._epoch, False
        if self._stamp_s is None or self._received_s is None:
            return self._off("awaiting_session")
        if ros - self._stamp_s < -FUTURE_TOLERANCE_S:
            return self._off("session_future")
        if ros - self._stamp_s > MAX_STATE_AGE_S:
            return self._off("session_stale")
        if not 0 <= mono - self._received_s <= MAX_STATE_AGE_S:
            return self._off("session_receive_stale")
        return self._epoch, True
