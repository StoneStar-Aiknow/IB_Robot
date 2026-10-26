"""The hand half of every command the session sends: follow while mirroring, neutral otherwise.

The session's state machine and window pipeline only ever produce 20 hand
zeros (``MockPlaybackHooks``); ``mirror_session_node`` passes every command the
gate sends through :meth:`HandOverlay.apply`, which replaces the hand values.
The arm values are not touched.

Per tick:

* goal: the retarget node's hand values (``observe_follow``, fresh within
  ``FOLLOW_FRESH_S``) while ``following`` (MIRRORING/running), else neutral
  (fingers at 0.33 of the fist, thumb at 0.123). A hand whose perception
  stopped is already sent to neutral by hand_core upstream;
* rate: B2, full open<->fist travel in 1.0 s (fingers) / 0.6 s (thumb);
* limits: the real-robot hand ranges of ``hand_limits.py``, right thumb roll
  at most ``RIGHT_ROLL_MAX``.

The first tick after the gate opened (or after a gap) starts from what MC is
commanding (``observe_command``), else the measured hand, else MC's default
fist, so the hand does not jump. If a hand starts as a jammed fist (thumb
|abad| > 0.5 with a pip more than 0.3 below that start, measured within
0.1 s), only its thumb abad moves first, to 0.3; the rest of that hand holds.

Hand feedback (``observe_state``), once it was ever received:

==================  ==========================================
age                  hands
==================  ==========================================
> 0.1 s              unjam paused (that hand holds)
0.1 s - 1.0 s        hold the last command
> 1.0 s              to neutral at B2, until fresh again for 2 s
==================  ==========================================

None of this raises into the gate: a hand problem moves the hands to a
preset, never stops the arms or the session. Constants from video2x2_runtime
570c53c ``upper_body_animation/hand_limits.py`` and ``followme_player.py``.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence

import numpy as np

HAND_COUNT = 20
_MIRROR = np.array([-1, -1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float64)
_LEFT_LO = np.array([-1.117, -0.05, -0.838, 0.0, 0.0, 0.0, -0.17, 0.0, -0.19, 0.0])
_LEFT_HI = np.array([0.03, 1.64, 0.0, 0.16, 1.57, 1.57, 0.0, 1.57, 0.0, 1.57])
HAND_LOW = np.concatenate([_LEFT_LO, np.minimum(_LEFT_LO * _MIRROR, _LEFT_HI * _MIRROR)])
HAND_HIGH = np.concatenate([_LEFT_HI, np.maximum(_LEFT_LO * _MIRROR, _LEFT_HI * _MIRROR)])
RIGHT_ROLL_MAX = 0.87
HAND_HIGH[10] = min(HAND_HIGH[10], RIGHT_ROLL_MAX)
_LEFT_FIST = np.array([-0.5, 1.0, -0.75, 0.0, 1.57, 1.57, 0.0, 1.57, 0.0, 1.57])
HAND_FIST = np.concatenate([_LEFT_FIST, _LEFT_FIST * _MIRROR])
_NEUTRAL_C, _NEUTRAL_S = 0.33, (0.33 - 0.25) / 0.65
_P = np.array([_NEUTRAL_S] * 3 + [0.0, _NEUTRAL_C, _NEUTRAL_C, 0.0, _NEUTRAL_C, 0.0, _NEUTRAL_C])
HAND_NEUTRAL = np.concatenate([_P * _LEFT_FIST, _P * _LEFT_FIST * _MIRROR])
FULL_TRAVEL_S, FULL_TRAVEL_S_THUMB = 1.0, 0.6
_THUMB = np.zeros(HAND_COUNT, dtype=bool)
_THUMB[[0, 1, 2, 10, 11, 12]] = True
_TRAVEL = np.abs(HAND_FIST)
_TRAVEL[[3, 6, 8, 13, 16, 18]] = 1.57

THUMB_ABAD = (1, 11)
PIPS = ((4, 5, 7, 9), (14, 15, 17, 19))
UNJAM_THUMB_ABAD_IN = 0.5
UNJAM_PIP_LAG = 0.3
UNJAM_THUMB_ABAD_OUT = 0.3

FOLLOW_FRESH_S = 0.3
SOURCE_FRESH_S = 1.0  # a command echo / state younger than this may be the start
RESTART_GAP_S = 0.5  # no apply for this long: the next one starts over
STATE_UNJAM_S = 0.1
STATE_HOLD_S = 1.0
STATE_RECOVER_S = 2.0


def _ten(values) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray([float(v) for v in values], dtype=np.float64)
    return array if array.shape == (10,) and np.all(np.isfinite(array)) else None


class HandOverlay:
    def __init__(self, rate_hz: float, neutral: Sequence[float] | None = None):
        self._step = np.where(_THUMB, _TRAVEL / FULL_TRAVEL_S_THUMB, _TRAVEL / FULL_TRAVEL_S) / float(rate_hz)
        self._neutral = np.clip(
            HAND_NEUTRAL if neutral is None else np.asarray(neutral, dtype=np.float64), HAND_LOW, HAND_HIGH
        )
        self._lock = threading.Lock()
        self._follow: np.ndarray | None = None
        self._follow_at = -math.inf
        self._state: list[np.ndarray | None] = [None, None]
        self._state_at = -math.inf
        self._command: list[np.ndarray | None] = [None, None]
        self._command_at = -math.inf
        self._prev: np.ndarray | None = None
        self._last_apply = -math.inf
        self._unjam = [False, False]
        self._state_lost = False
        self._fresh_since: float | None = None
        self.counts = {"starts": 0, "unjams": 0, "held_state": 0, "lost_state": 0, "clamped": 0, "following": 0}
        self.start_source = ""

    # -- inputs (subscription callbacks) ----------------------------------------
    def observe_follow(self, values, now: float) -> None:
        array = np.asarray([float(v) for v in values], dtype=np.float64)
        if array.shape == (HAND_COUNT,) and np.all(np.isfinite(array)):
            with self._lock:
                self._follow, self._follow_at = array, now

    def observe_state(self, left, right, now: float) -> None:
        with self._lock:
            self._state = [_ten(left), _ten(right)]
            if self._state[0] is not None or self._state[1] is not None:
                self._state_at = now

    def observe_command(self, left, right, now: float) -> None:
        with self._lock:
            self._command = [_ten(left), _ten(right)]
            self._command_at = now

    # -- output (the gate's publish, under the session lock) --------------------
    def apply(self, following: bool, now: float) -> tuple[float, ...]:
        with self._lock:
            if self._prev is None or now - self._last_apply > RESTART_GAP_S:
                self._start(now)
            self._last_apply = now
            state_age = now - self._state_at
            if math.isfinite(state_age):
                if state_age > STATE_HOLD_S:
                    if not self._state_lost:
                        self.counts["lost_state"] += 1
                    self._state_lost, self._fresh_since = True, None
                elif state_age <= STATE_UNJAM_S:
                    self._fresh_since = now if self._fresh_since is None else self._fresh_since
                else:
                    self._fresh_since = None
                if self._state_lost and self._fresh_since is not None and now - self._fresh_since >= STATE_RECOVER_S:
                    self._state_lost = False
                if not self._state_lost and state_age > STATE_UNJAM_S:
                    self.counts["held_state"] += 1
                    return tuple(map(float, self._prev))
            live = following and self._follow is not None and now - self._follow_at <= FOLLOW_FRESH_S
            live = live and not self._state_lost
            goal = (self._follow if live else self._neutral).copy()
            self.counts["following"] += int(live)
            for side in (0, 1):
                if not self._unjam[side]:
                    continue
                part = slice(10 * side, 10 * side + 10)
                abad = THUMB_ABAD[side]
                if abs(self._prev[abad]) <= UNJAM_THUMB_ABAD_OUT + 1e-9:
                    self._unjam[side] = False
                    continue
                goal[part] = self._prev[part]
                if math.isfinite(state_age) and state_age <= STATE_UNJAM_S:
                    goal[abad] = math.copysign(UNJAM_THUMB_ABAD_OUT, self._prev[abad])
            delta = goal - self._prev
            stepped = np.where(np.abs(delta) <= self._step, goal, self._prev + np.clip(delta, -self._step, self._step))
            out = np.clip(stepped, HAND_LOW, HAND_HIGH)
            self.counts["clamped"] += int(np.any(out != stepped))
            self._prev = out
            return tuple(map(float, out))

    def _start(self, now: float) -> None:
        start = HAND_FIST.copy()
        sources = []
        for side in (0, 1):
            part = slice(10 * side, 10 * side + 10)
            if now - self._command_at <= SOURCE_FRESH_S and self._command[side] is not None:
                start[part], source = self._command[side], "command"
            elif now - self._state_at <= SOURCE_FRESH_S and self._state[side] is not None:
                start[part], source = self._state[side], "state"
            else:
                source = "fist"
            sources.append(source)
        self._prev = np.clip(start, HAND_LOW, HAND_HIGH)
        self.start_source = "/".join(sources)
        self.counts["starts"] += 1
        self._unjam = [False, False]
        if now - self._state_at > STATE_UNJAM_S:
            return
        for side in (0, 1):
            measured = self._state[side]
            if measured is None:
                continue
            folded = abs(self._prev[THUMB_ABAD[side]]) > UNJAM_THUMB_ABAD_IN
            pips = list(PIPS[side])
            lag = self._prev[pips] - measured[[p - 10 * side for p in pips]]
            if folded and np.any(lag > UNJAM_PIP_LAG):
                self._unjam[side] = True
                self.counts["unjams"] += 1

    def snapshot(self, now: float) -> dict:
        with self._lock:
            return {
                **self.counts,
                "start_source": self.start_source,
                "unjam": list(self._unjam),
                "state_age_s": None if not math.isfinite(self._state_at) else round(now - self._state_at, 3),
                "state_lost": self._state_lost,
                "follow_age_s": None if not math.isfinite(self._follow_at) else round(now - self._follow_at, 3),
                "last": None if self._prev is None else [round(float(v), 3) for v in self._prev],
            }
