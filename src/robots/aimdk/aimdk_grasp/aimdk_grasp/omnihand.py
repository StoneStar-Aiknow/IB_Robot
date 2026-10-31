"""The OmniHand as the grasp pipeline's end effector.

The left hand is streamed on the runtime's ``joint.hand_stream`` (20 native
values: the left hand's ten driven joints in ``LEFT_HAND`` order, then the
right's, sent as the zeros the motion-control contract was verified with). The
last command is kept alive between actions, so the hand holds its target while
the arm moves; after a goal the stream is released and the platform keeps the
last hand target.

Preshape and release open the hand along a straight line, one ``HAND_STEP_RAD``
per frame. The close stops each finger at its first contact and calls the
grasp held when the thumb and one more finger touch: the hand's fault codes are
not used (a finger pressing an object reports one for as long as it grips, and
on the robot fingers reported codes while tracking freely).

Migrated from the X2 pipeline's grasp sequence (move_hand, close_hand) with its
constants.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from manipulation_execution.pick.pipeline.contracts import EffectorAction, EffectorOutcome, Flag
from manipulation_execution.pick.pipeline.streaming import JointSample, received_at_or_stale

from .hand import OPEN_HAND_TOLERANCE_RAD, HandMapping
from .joints import LEFT_HAND

HAND_HZ = 50.0
HAND_STEP_RAD = 0.01
CLOSE_SETTLE_S = 0.5
# Opening holds the open preset until every joint is within the open
# tolerance, up to this long: r3 checked after a fixed 0.5 s while the thumb
# abduction was still 0.034 rad short and stopped (r1: 0.014 short, passed).
OPEN_SETTLE_MAX_S = 3.0
# At the grasp pose the hand may not have room to open completely (r4: thumb
# abduction held 0.035 rad short for 3 s next to the bottle, no touch; it
# opened to 0.010 in free space). Short by at most this much, the retreat
# goes ahead; further short is a blocked hand.
OPEN_SHORT_MAX_RAD = 0.1
CLOSE_MAX_S = 15.0
# Contact-stop close. Each finger closes on its own and freezes at its current
# command once touched: fingertip pad max >= CONTACT_TOUCH (touch1 hand presses
# read 18-157 of 255, idle 0) or its flex joint trails the command by
# CONTACT_LAG_RAD (g3: free closing trailed <= 0.10, the bottle held fingers
# 0.2-0.49 back before squeezing it out). Thumb plus one finger in contact =
# gripped: every finger freezes and that command is held through lift/place.
# A hand faultcode is not contact: in g8/g9 index, pinky and thumb roll
# reported codes while tracking their command (lag <= 0.01) and the thumb
# closed to its full target unblocked, so both runs would have claimed a grip.
CONTACT_TOUCH = 30
# While an object is carried, a contact finger still presses it when its flex
# joint trails the frozen close command by at least this much, or its pad reads
# CONTACT_TOUCH. Held = the thumb and one more finger still pressing. Live
# 2026-10-23: in held grips every contact finger trailed by 0.04-0.16 rad for
# the whole hold; in the two that slipped out (one tipped on release, one flung)
# thumb and fingers crept to within 0.001-0.016 rad of their command 0.5-3 s
# after the close, with the arm still at the grasp pose.
HOLD_LAG_RAD = 0.025
# The grip counts as lost once it has failed this long without a break.
GRIP_LOST_S = 0.2
# Touch readings older than this are not used: when the touch stream stops, the
# last grip's levels would freeze every finger of the next close at once.
TOUCH_MAX_AGE_S = 0.1
CONTACT_LAG_RAD = 0.15
# Native joint indices per finger (LEFT_HAND order), flex joint, touch pad.
FINGERS = {
    "thumb": ((0, 1, 2), 2, "thumb"),
    "index": ((3, 4), 4, "index_finger"),
    "middle": ((5,), 5, "middle_finger"),
    "ring": ((6, 7), 7, "ring_finger"),
    "pinky": ((8, 9), 9, "little_finger"),
}
# The right hand's slots in the 20-value command: the dummy zeros the
# motion-control contract was verified with (mc_upper_body_r13, mc_stop_r14).
RIGHT_DUMMY = (0.0,) * 10


class HandIO(Protocol):
    channel_joints: tuple[str, ...]

    def feedback(self) -> JointSample:
        """Native positions under the stream's ``channel_joints`` names."""
        ...

    def touch(self) -> Mapping[str, float] | None:
        """Max reading per left touch pad (FINGERS pad names), or None without a touch sensor."""
        ...

    def publish(self, positions: Sequence[float]) -> None: ...

    def release(self) -> None: ...

    def refusal(self) -> str | None: ...

    def monotonic(self) -> float: ...

    def wait(self, seconds: float) -> None: ...


class _Failed(Exception):
    pass


@dataclass
class OmniHandEffector:
    """``mapping`` must carry the gripper description's open and close presets."""

    io: HandIO
    mapping: HandMapping
    feedback_max_age_s: float = 0.1
    command: np.ndarray | None = None
    report: dict = field(default_factory=dict)
    #: The last close's grip while it is held: contact fingers, frozen command, closing direction.
    _grip: dict | None = field(default=None, repr=False)
    _lost_since: float | None = field(default=None, repr=False)

    def act(self, action: EffectorAction, *, halted: Flag, cancelled: Flag) -> EffectorOutcome:
        try:
            if self.io.refusal():
                raise _Failed(f"runtime does not admit output: {self.io.refusal()}")
            if action is EffectorAction.CLOSE:
                return self._close(halted, cancelled)
            return self._open(action.value, cancelled)
        except _Failed as exc:
            return EffectorOutcome(False, str(exc))

    def finish(self) -> None:
        """Stop sending; the platform keeps the last hand target."""
        self.io.release()
        self.command = None
        self._grip = None

    def holding(self) -> str | None:
        """While an object is carried: '' while the grip holds, the reason once it is lost, None if unknown.

        The close froze each contact finger's command past where the object
        stopped it; a finger that still presses trails that command or reads
        touch. One whose object slipped out closes onto its command.
        """
        grip = self._grip
        if grip is None:
            return None
        try:
            measured = self._measured()
        except _Failed:
            return None  # the arm's supervision reports stale feedback
        touch = self.io.touch()
        lags, pressing = {}, []
        for finger in grip["contacts"]:
            _, flex, pad = FINGERS[finger]
            lags[finger] = round(float((grip["command"][flex] - measured[flex]) * grip["direction"][flex]), 3)
            if lags[finger] >= HOLD_LAG_RAD or (touch is not None and touch.get(pad, 0.0) >= CONTACT_TOUCH):
                pressing.append(finger)
        if "thumb" in pressing and len(pressing) >= 2:
            self._lost_since = None
            return ""
        now = self.io.monotonic()
        self._lost_since = now if self._lost_since is None else self._lost_since
        if now - self._lost_since < GRIP_LOST_S:
            return ""
        return f"the contact fingers no longer press the object (lag rad: {lags})"

    # -- actions --------------------------------------------------------------------

    def _open(self, name: str, cancelled: Flag) -> EffectorOutcome:
        self._grip = None  # whatever was held is let go
        target = self.mapping.preset("open")
        start = self._measured()
        row = self.report[name] = {}
        if self.command is not None and np.array_equal(self.command, target):
            frames = []  # already commanded open: only wait for the fingers below
        else:
            # Straight line from the measured hand, HAND_STEP_RAD per frame.
            steps = max(1, int(np.ceil(np.max(np.abs(target - start)) / HAND_STEP_RAD)))
            frames = [start + (target - start) * (i + 1) / steps for i in range(steps)]
            frames += [target] * int(CLOSE_SETTLE_S * HAND_HZ)
        for frame in frames:
            self._frame(frame, cancelled, name)
        waited, ended = 0, self.io.monotonic() + OPEN_SETTLE_MAX_S
        while np.max(np.abs(self._measured() - target)) > OPEN_HAND_TOLERANCE_RAD and self.io.monotonic() < ended:
            self._frame(target, cancelled, name)
            waited += 1
        measured = self._measured()
        short = float(np.max(np.abs(target - measured)))
        row.update(frames=len(frames), settle_frames=waited, short_of_target_rad=short, measured=measured.tolist())
        if short > OPEN_SHORT_MAX_RAD:
            return EffectorOutcome(False, f"hand blocked {short:.3f} rad short of open", report=dict(row))
        if short > OPEN_HAND_TOLERANCE_RAD:
            row["open_short"] = True
        return EffectorOutcome(True, report=dict(row))

    def _close(self, halted: Flag, cancelled: Flag) -> EffectorOutcome:
        self._grip = None
        target = self.mapping.preset("close")
        command = self._measured().copy()
        direction = np.sign(target - command)
        frozen: dict[str, dict] = {}
        began = self.io.monotonic()
        while True:
            tick = self.io.monotonic()
            if cancelled():
                raise _Failed("cancelled during close")
            if tick - began > CLOSE_MAX_S:
                raise _Failed(f"close did not finish within {CLOSE_MAX_S:g} s")
            if halted():
                # Hold where the fingers are; the sequence releases them.
                self.report["close"] = dict(
                    gripped=False,
                    interrupted=True,
                    contacts=frozen,
                    seconds=round(tick - began, 3),
                    command=command.tolist(),
                )
                return EffectorOutcome(True, "halted", halted=True, report=dict(self.report["close"]))
            for finger, (joints, _, _) in FINGERS.items():
                if finger not in frozen:
                    for joint in joints:
                        command[joint] += np.clip(target[joint] - command[joint], -HAND_STEP_RAD, HAND_STEP_RAD)
            self._send(command)
            measured = self._measured()
            touch = self.io.touch()
            for finger, (_, flex, pad) in FINGERS.items():
                if finger in frozen:
                    continue
                lag = float((command[flex] - measured[flex]) * direction[flex])
                touched = touch is not None and touch.get(pad, 0.0) >= CONTACT_TOUCH
                if touched or lag >= CONTACT_LAG_RAD:
                    frozen[finger] = dict(
                        at_s=round(tick - began, 3),
                        lag_rad=round(lag, 3),
                        touch=None if touch is None else touch.get(pad),
                        by="touch" if touched else "lag",
                    )
            done = all(
                finger in frozen or np.allclose(command[list(joints)], target[list(joints)])
                for finger, (joints, _, _) in FINGERS.items()
            )
            gripped = "thumb" in frozen and len(frozen) >= 2
            if gripped or done:
                break
            self.io.wait(max(0.0, tick + 1.0 / HAND_HZ - self.io.monotonic()))
        for _ in range(int(CLOSE_SETTLE_S * HAND_HZ)):
            self._frame(command, cancelled, "close")
        if gripped:
            self._grip = dict(contacts=tuple(frozen), command=command.copy(), direction=direction.copy())
            self._lost_since = None
        self.report["close"] = dict(
            gripped=gripped,
            contacts=frozen,
            seconds=round(self.io.monotonic() - began, 3),
            command=command.tolist(),
            measured=self._measured().tolist(),
            touch_after=self.io.touch(),
        )
        return EffectorOutcome(
            True,
            held=gripped,
            confidence=len(frozen) / len(FINGERS) if gripped else 0.0,
            report=dict(self.report["close"]),
        )

    # -- pieces ---------------------------------------------------------------------

    def _measured(self) -> np.ndarray:
        sample = self.io.feedback()
        if not sample.healthy or not 0 <= self.io.monotonic() - sample.received_at <= self.feedback_max_age_s:
            raise _Failed("hand feedback unhealthy or stale")
        left_joints = self.io.channel_joints[: len(LEFT_HAND)]
        if len(left_joints) != len(LEFT_HAND):
            raise _Failed("invalid hand feedback: hand stream describes fewer than ten left joints")
        try:
            return self.mapping.check([sample.positions[name] for name in left_joints])
        except (KeyError, ValueError) as exc:
            raise _Failed(f"invalid hand feedback: {exc}") from None

    def _send(self, left: np.ndarray) -> None:
        left = self.mapping.check(left)
        if self.io.refusal():
            raise _Failed(f"runtime does not admit output: {self.io.refusal()}")
        self.io.publish([*left.tolist(), *RIGHT_DUMMY])
        self.command = left.copy()

    def _frame(self, left: np.ndarray, cancelled: Flag, name: str) -> None:
        tick = self.io.monotonic()
        if cancelled():
            raise _Failed(f"cancelled during {name}")
        self._measured()
        self._send(left)
        self.io.wait(max(0.0, tick + 1.0 / HAND_HZ - self.io.monotonic()))


def left_touch_levels(message) -> dict[str, float] | None:
    """Max reading of each left pad from the runtime's touch array (layout labels ``left_<pad>``)."""
    levels, offset = {}, 0
    for dimension in message.layout.dim:
        cells = list(message.data[offset : offset + dimension.size])
        offset += dimension.size
        if dimension.label.startswith("left_") and cells:
            levels[dimension.label[len("left_") :]] = float(max(cells))
    return levels or None


class RosHandIO:
    """``HandIO`` over the runtime's hand stream, hand state and touch array, bound by interface ID.

    Like the stream's, the touch array is taken when read. Call :meth:`close` before ``node`` is destroyed.
    """

    def __init__(self, node, description: Mapping):
        from std_msgs.msg import Float64MultiArray

        from manipulation_execution.pick.pipeline.ros_stream import RosStreamIO, bind_stream
        from robot_config.contract_utils import qos_profile_from_dict
        from robot_runtime.on_demand import OnDemandSubscriptions

        interfaces = description["interfaces"]
        # The stream names the joints left_thumb_roll..., the state view L_thumb_roll_joint..., in one order.
        aliases = dict(
            zip(
                interfaces["joint.hand_stream"]["joint_names"],
                interfaces["joint.hand_state"]["joint_names"],
                strict=True,
            )
        )
        binding = bind_stream(
            description, stream="joint.hand_stream", feedback=["joint.hand_state"], feedback_names=aliases
        )
        self._stream = RosStreamIO(node, binding)
        self.channel_joints = self._stream.channel_joints
        self._touch: dict[str, float] | None = None
        self._touch_received = float("-inf")
        self._touch_subscriptions = OnDemandSubscriptions(node.context, f"{node.get_name()}_touch")
        touch = interfaces.get("touch.hands")
        if touch is not None:
            self._touch_subscriptions.subscribe(
                Float64MultiArray,
                touch["endpoint"],
                qos_profile_from_dict(touch.get("qos") or {}),
                1,
                self._on_touch,
            )

    def _on_touch(self, message, received: float | None) -> None:
        self._touch, self._touch_received = left_touch_levels(message), received_at_or_stale(received)

    def touch(self) -> Mapping[str, float] | None:
        """The newest touch levels, or None when none is fresh: an old grip's readings are not contact now."""
        self._touch_subscriptions.drain()
        if self.monotonic() - self._touch_received > TOUCH_MAX_AGE_S:
            return None
        return self._touch

    def close(self) -> None:
        self._touch_subscriptions.close()
        self._stream.close()

    def feedback(self) -> JointSample:
        return self._stream.feedback()

    def publish(self, positions: Sequence[float]) -> None:
        self._stream.publish(positions)

    def release(self) -> None:
        self._stream.release()

    def refusal(self) -> str | None:
        return self._stream.refusal()

    def monotonic(self) -> float:
        return self._stream.monotonic()

    def wait(self, seconds: float) -> None:
        self._stream.wait(seconds)
