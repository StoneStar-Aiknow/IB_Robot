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

Given a ``reset_pose`` (14 arm values), ``reset_pose`` and ``wrap_up``
return to it instead of the defaults:

* ``reset_pose`` is that pose, hands at 20 zeros; None while no joint state
  was ever received, as the default;
* the approach walks every joint straight toward it at a common speed of at
  least 30 deg/s, fast enough to arrive 0.3 s inside the budget;
* a cancel or stop wrap-up goes from the gate's last command to it on a
  2.0 s quintic, without following the person.

Wrap-up commands do not pass ``post_process``: each path is monotonic per
joint between two poses that are inside the limits, and slower than the
rated speeds. Everything else keeps the ``SessionHooks`` default.
"""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence

from manipulation_execution.mirror_session.endpoints import ARM_JOINT_COUNT, HAND_JOINT_COUNT
from manipulation_execution.mirror_session.hooks import (
    SessionHooks,
    SessionInputs,
    UpperBodyCommand,
    WrapUpContext,
    WrapUpPurpose,
    WrapUpStep,
)

# (playing, round) -> None; must not block: it is called under the session lock.
PlayRequest = Callable[[bool, int], None]

APPROACH_MIN_SPEED_RAD_S = math.radians(30.0)
APPROACH_MARGIN_S = 0.3  # the approach arrives this far inside the wrap-up budget
APPROACH_MAX_STEP_S = 0.1  # one tick never covers more time than this
RETURN_DURATION_S = 2.0  # cancel / stop, as the runtime's blend-out


class MockPlaybackHooks(SessionHooks):
    """``SessionHooks`` defaults, except a window pipeline fed by the mock's preview."""

    def __init__(self, request_play: PlayRequest, reset_pose: Sequence[float] | None = None):
        if reset_pose is not None:
            reset_pose = tuple(float(v) for v in reset_pose)
            if len(reset_pose) != ARM_JOINT_COUNT or not all(map(math.isfinite, reset_pose)):
                raise ValueError(f"reset_pose needs {ARM_JOINT_COUNT} finite values, got {reset_pose!r}")
        self._reset_arm = reset_pose
        self._return_from: tuple[float, ...] | None = None
        self._approach_elapsed: float | None = None
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
        self._return_from = None  # the wrap-up that follows starts from the gate's last command
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

    def reset_pose(self, arm_at_takeover: Sequence[float] | None) -> UpperBodyCommand | None:
        if self._reset_arm is None:
            return super().reset_pose(arm_at_takeover)
        if arm_at_takeover is None:
            return None
        return UpperBodyCommand(self._reset_arm, (0.0,) * HAND_JOINT_COUNT)

    def wrap_up(self, context: WrapUpContext) -> WrapUpStep:
        if self._reset_arm is None:
            return super().wrap_up(context)
        if context.purpose == WrapUpPurpose.APPROACH:
            self._return_from = None
            return self._approach(context)
        self._approach_elapsed = None
        return self._return(context)

    def _approach(self, context: WrapUpContext) -> WrapUpStep:
        """One tick of a straight walk from the last command (or the measured arm) to the reset pose."""
        here = context.last_command.arm if context.last_command is not None else context.measured_arm
        if here is None or len(here) != ARM_JOINT_COUNT:
            return WrapUpStep(None, False)
        previous = self._approach_elapsed
        self._approach_elapsed = context.elapsed_s
        dt = context.elapsed_s if previous is None or previous > context.elapsed_s else context.elapsed_s - previous
        dt = min(max(dt, 0.0), APPROACH_MAX_STEP_S)
        remaining = [goal - now for goal, now in zip(self._reset_arm, here)]
        if not all(map(math.isfinite, remaining)):
            return WrapUpStep(None, False)
        farthest = max(map(abs, remaining))
        time_left = max(context.budget_s - APPROACH_MARGIN_S - context.elapsed_s, dt, 1e-3)
        reach = max(APPROACH_MIN_SPEED_RAD_S, farthest / time_left) * dt
        if farthest <= reach:
            self._approach_elapsed = None
            return WrapUpStep(UpperBodyCommand(self._reset_arm, (0.0,) * HAND_JOINT_COUNT), True)
        arm = tuple(now + max(-reach, min(reach, delta)) for now, delta in zip(here, remaining))
        return WrapUpStep(UpperBodyCommand(arm, (0.0,) * HAND_JOINT_COUNT), False)

    def _return(self, context: WrapUpContext) -> WrapUpStep:
        """Cancel / stop: a quintic from where this wrap-up began to the reset pose."""
        target = UpperBodyCommand(self._reset_arm, (0.0,) * HAND_JOINT_COUNT)
        if self._return_from is None:
            start = context.last_command.arm if context.last_command is not None else context.measured_arm
            if start is None or len(start) != ARM_JOINT_COUNT or not all(map(math.isfinite, start)):
                return WrapUpStep(target, True)
            self._return_from = tuple(float(v) for v in start)
        s = context.elapsed_s / RETURN_DURATION_S
        if s >= 1.0:
            self._return_from = None
            return WrapUpStep(target, True)
        s = max(s, 0.0)
        weight = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))
        arm = tuple(a + (b - a) * weight for a, b in zip(self._return_from, self._reset_arm))
        return WrapUpStep(UpperBodyCommand(arm, (0.0,) * HAND_JOINT_COUNT), False)

    def snapshot(self) -> dict:
        with self._lock:
            return {"round": self._round, "accepting": self._accepting, "taken": self.taken, "dropped": self.dropped}
