"""Window pipeline that routes ``motion_imitation_mock``'s recorded animation through the gate.

It lets the data flow run end to end, the robot included, without a
retargeter. It is not the mirror behaviour.

The mock node keeps computing the arm command (animation, mock safety) and
publishes it on its preview topic only; this session is the one outlet to
the arm and hand channels. Per round:

* ``begin_round`` asks the mock to play from frame 0 (``~/play`` true);
* previews are taken only after that request was acknowledged, so a frame
  of an earlier round never opens a new one;
* ``retarget`` hands each preview to the gate once, hands at 20 zeros
  (the left hand's sign convention is not established); with no new
  preview it passes nothing and the gate holds;
* ``end_round`` pauses the mock (``~/play`` false).

The paused mock publishes nothing, so between rounds no preview is in flight.
Everything else keeps the ``SessionHooks`` default.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence

from manipulation_execution.mirror_session.endpoints import ARM_JOINT_COUNT, HAND_JOINT_COUNT
from manipulation_execution.mirror_session.hooks import SessionHooks, SessionInputs, UpperBodyCommand

# (playing, round) -> None; must not block: it is called under the session lock.
PlayRequest = Callable[[bool, int], None]


class MockPlaybackHooks(SessionHooks):
    """``SessionHooks`` defaults, except a window pipeline fed by the mock's preview."""

    def __init__(self, request_play: PlayRequest):
        self._request_play = request_play
        self._lock = threading.Lock()
        self._round = 0
        self._accepting = False
        self._pending: tuple[float, ...] | None = None
        self.taken = 0
        self.dropped = 0

    def begin_round(self) -> None:
        self._request_play(True, self._next_round())

    def end_round(self) -> None:
        self._request_play(False, self._next_round())

    def _next_round(self) -> int:
        with self._lock:
            self._round += 1
            self._accepting = False
            self._pending = None
            return self._round

    def acknowledge(self, round_id: int, playing: bool, success: bool) -> None:
        """The mock answered ``~/play``. Only a successful play of the current round opens it."""
        with self._lock:
            if playing and success and round_id == self._round:
                self._accepting = True

    def offer(self, arm: Sequence[float]) -> None:
        """A preview message arrived; keep it only while the current round is open."""
        values = tuple(float(v) for v in arm)
        with self._lock:
            if not self._accepting or len(values) != ARM_JOINT_COUNT or not all(map(math.isfinite, values)):
                self.dropped += 1
                return
            self._pending = values

    def retarget(self, inputs: SessionInputs) -> UpperBodyCommand | None:
        """The newest preview, once. The PEAR content is ignored, as in the mock node."""
        with self._lock:
            arm, self._pending = self._pending, None
            if arm is not None:
                self.taken += 1
        if arm is None:
            return None
        return UpperBodyCommand(arm, (0.0,) * HAND_JOINT_COUNT)

    def post_process(self, command: UpperBodyCommand) -> UpperBodyCommand | None:
        """Pass through: no limit is applied here.

        The mock node clamps every command to the joint limits. Its rate limit
        does not cover the first command after a (re)start, so that command
        can step from the held pose to frame 0 of the clip.
        """
        return command

    def snapshot(self) -> dict:
        with self._lock:
            return {"round": self._round, "accepting": self._accepting, "taken": self.taken, "dropped": self.dropped}
