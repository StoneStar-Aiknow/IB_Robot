"""The OmniHand as the grasp pipeline's end effector.

Either hand is streamed on the runtime's ``joint.hand_stream`` (20 native
values: the left hand's ten driven joints in ``LEFT_HAND`` order, then the
right's in ``RIGHT_HAND`` order). The grasping hand (its :class:`HandMapping`'s
side) takes its ten slots; the other hand holds its fresh measured pose from
the first command until the goal finishes. Missing feedback or drift of that
preserved hand refuses further output. A hand that is not installed is not
preserved: its slots carry ``joints.ABSENT_HAND_COMMAND``. The
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

from manipulation_execution.grasp_pipeline.contracts import EffectorAction, EffectorOutcome, Flag
from manipulation_execution.grasp_pipeline.streaming import JointSample, received_at_or_stale

from .hand import PRESERVED_HAND_DRIFT_RAD, HandMapping, is_open
from .joints import ABSENT_HAND_COMMAND, HAND_PARTS, check_side, other_side

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
# Native joint indices per finger (HAND_PARTS order, either hand), flex joint, touch pad.
FINGERS = {
    "thumb": ((0, 1, 2), 2, "thumb"),
    "index": ((3, 4), 4, "index_finger"),
    "middle": ((5,), 5, "middle_finger"),
    "ring": ((6, 7), 7, "ring_finger"),
    "pinky": ((8, 9), 9, "little_finger"),
}
#: Each hand's offset in the 20-value stream and the hand state view.
HAND_OFFSETS = {"left": 0, "right": len(HAND_PARTS)}


#: Native indices of the thumb's joints (either hand); the other seven are the four fingers.
THUMB = np.zeros(len(HAND_PARTS), dtype=bool)
THUMB[list(FINGERS["thumb"][0])] = True
THUMB.setflags(write=False)
#: Where the thumb waits while the fingers open in the ``thumb_clear`` path:
#: abduction and flexion straightened (native 0), kept this far inside their limits.
THUMB_CLEAR_MARGIN_RAD = 0.05
#: The ways to open a hand, in order of preference. A straight line is the
#: shortest motion; the staged ones exist because the thumb and the bent fingers
#: can block each other (2026-11-08: a straight open from a half-closed right
#: hand stalled the thumb abduction at -1.10 rad under the fingers; after the
#: thumb had been straightened, the same open reached the preset).
OPEN_PATHS = ("together", "fingers_first", "thumb_first", "thumb_clear")


def chained_frames(start, waypoints, step_rad: float = HAND_STEP_RAD) -> np.ndarray:
    """The measured start, then straight lines through ``waypoints``, at most ``step_rad`` per joint per frame."""
    start = np.asarray(start, float)
    points = [np.asarray(point, float) for point in waypoints]
    if start.shape != (len(HAND_PARTS),) or any(point.shape != start.shape for point in points):
        raise ValueError("Ten finite hand values required")
    if not np.isfinite([start, *points]).all():
        raise ValueError("Ten finite hand values required")
    if not np.isfinite(step_rad) or not 0 < step_rad <= HAND_STEP_RAD:
        raise ValueError(f"Hand step must be finite and in (0, {HAND_STEP_RAD}] rad")
    frames, here = [start], start
    for point in points:
        span = float(np.max(np.abs(point - here)))
        if span > 0.0:
            frames.extend(np.linspace(here, point, int(np.ceil(span / step_rad)) + 1)[1:])
            here = point
    if len(frames) == 1:
        frames.append(points[-1] if points else start)  # already there: one frame, as before
    return np.asarray(frames)


def open_path(name: str, start, target, lower, upper, step_rad: float = HAND_STEP_RAD) -> np.ndarray:
    """Frames of one way (:data:`OPEN_PATHS`) to go from ``start`` to ``target``.

    ``together``: every joint on one straight line. ``fingers_first``: the
    thumb holds while the four fingers open, then the thumb opens.
    ``thumb_first``: the reverse. ``thumb_clear``: the thumb straightens out of
    the fingers' way (roll kept), the fingers open, then the thumb comes in to
    its target. ``lower``/``upper`` are the native joint limits.
    """
    start, target = np.asarray(start, float), np.asarray(target, float)
    if name == "together":
        waypoints = [target]
    elif name == "fingers_first":
        waypoints = [np.where(THUMB, start, target), target]
    elif name == "thumb_first":
        waypoints = [np.where(THUMB, target, start), target]
    elif name == "thumb_clear":
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        straight = [HAND_PARTS.index("thumb_abad"), HAND_PARTS.index("thumb_mcp")]
        clear = start.copy()
        clear[straight] = np.clip(
            0.0, lower[straight] + THUMB_CLEAR_MARGIN_RAD, upper[straight] - THUMB_CLEAR_MARGIN_RAD
        )
        waypoints = [clear, np.where(THUMB, clear, target), target]
    else:
        raise ValueError(f"unknown hand open path {name!r}")
    return chained_frames(start, waypoints, step_rad)


def open_paths(start, target, lower, upper, step_rad: float = HAND_STEP_RAD) -> list[tuple[str, np.ndarray]]:
    """Every distinct :func:`open_path`, in :data:`OPEN_PATHS` order; a path equal to an earlier one is left out."""
    paths = []
    for name in OPEN_PATHS:
        frames = open_path(name, start, target, lower, upper, step_rad)
        if not any(frames.shape == other.shape and np.array_equal(frames, other) for _, other in paths):
            paths.append((name, frames))
    return paths


class HandIO(Protocol):
    channel_joints: tuple[str, ...]
    touch_configured: bool

    def feedback(self) -> JointSample:
        """Native positions under the stream's ``channel_joints`` names."""
        ...

    def touch(self) -> Mapping[str, float] | None:
        """Max per grasping-hand pad (FINGERS names), or None without a sensor or fresh data."""
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
    """``mapping`` must carry the gripper description's open and close presets; its side is the hand driven."""

    io: HandIO
    mapping: HandMapping
    feedback_max_age_s: float = 0.1
    #: The other hand is not installed (``backend_options.installed_hands``): its
    #: slots carry ``ABSENT_HAND_COMMAND`` and its feedback is never read.
    absent_other: bool = False
    command: np.ndarray | None = None
    report: dict = field(default_factory=dict)
    #: The last close's grip while it is held: contact fingers, frozen command, closing direction.
    _grip: dict | None = field(default=None, repr=False)
    _lost_since: float | None = field(default=None, repr=False)
    _preserved_hand: np.ndarray | None = field(default=None, repr=False)

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
        self._preserved_hand = None

    def suspend(self) -> None:
        """Stop keepalives for handover while retaining the grip observation."""
        self.io.release()

    def snapshot(self) -> str:
        import json

        if self._grip is None or self.command is None:
            raise ValueError("no verified grip to hand over")
        return json.dumps(
            {
                "contacts": list(self._grip["contacts"]),
                "command": self._grip["command"].tolist(),
                "direction": self._grip["direction"].tolist(),
            }
        )

    def restore(self, state: str) -> None:
        import json

        grip = json.loads(state)
        command = np.asarray(grip["command"], dtype=float)
        direction = np.asarray(grip["direction"], dtype=float)
        contacts = tuple(grip["contacts"])
        if (
            command.shape != (10,)
            or direction.shape != (10,)
            or not np.isfinite([command, direction]).all()
            or "thumb" not in contacts
            or len(contacts) < 2
            or any(name not in FINGERS for name in contacts)
        ):
            raise ValueError("invalid frozen grip state")
        self.mapping.check(command)
        self._measured()  # Require healthy, fresh feedback before publishing.
        self._grip = dict(contacts=contacts, command=command.copy(), direction=direction.copy())
        self._lost_since = None
        self._send(command)

    def released(self) -> bool:
        """Fresh open-hand feedback and no tactile contact are required."""
        touch = self.io.touch()
        return bool(
            is_open(self._measured(), self.mapping.preset("open"))
            and touch is not None
            and all(value < CONTACT_TOUCH for value in touch.values())
        )

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
            self._lost_since = None
            return None  # the arm's supervision reports stale feedback
        touch = self.io.touch()
        lags, pressing = {}, []
        for finger in grip["contacts"]:
            _, flex, pad = FINGERS[finger]
            lags[finger] = round(float((grip["command"][flex] - measured[flex]) * grip["direction"][flex]), 3)
            if lags[finger] >= HOLD_LAG_RAD or (touch is not None and touch.get(pad, 0.0) >= CONTACT_TOUCH):
                pressing.append(finger)
        if "thumb" in pressing and len(pressing) >= 2:
            self.report.setdefault("holding", []).append(dict(at=self.io.monotonic(), lags=lags, pressing=pressing))
            if len(self.report["holding"]) > 3000:
                del self.report["holding"][:1000]
            self._lost_since = None
            return ""
        if touch is None and self.io.touch_configured:
            # An unavailable sensor is not a fresh no-contact reading. Joint
            # lag may still prove a grip above, but cannot disprove its touch.
            self._lost_since = None
            return None
        now = self.io.monotonic()
        self._lost_since = now if self._lost_since is None else self._lost_since
        if now - self._lost_since < GRIP_LOST_S:
            return ""
        return f"the contact fingers no longer press the object (lag rad: {lags})"

    # -- actions --------------------------------------------------------------------

    def _open(self, name: str, cancelled: Flag) -> EffectorOutcome:
        holding = self._grip is not None
        self._grip = None  # whatever was held is let go
        target = self.mapping.preset("open")
        row = self.report[name] = dict(paths=["together"], frames=0, settle_frames=0)
        if self.command is not None and np.array_equal(self.command, target):
            frames = []  # already commanded open: only wait for the fingers below
        else:
            # One straight line: an object in the hand is let go by every finger at once.
            frames = chained_frames(self._measured(), [target])[1:]
        self._open_along(frames, target, cancelled, name)
        if float(np.max(np.abs(target - self._measured()))) > OPEN_SHORT_MAX_RAD and not holding:
            # The thumb and the bent fingers may block each other: straighten the
            # thumb out of the way, open the fingers, then bring the thumb in.
            # Not with an object in the hand, which a straightening thumb could knock.
            row["paths"].append("thumb_clear")
            frames = open_path("thumb_clear", self._measured(), target, self.mapping.lower, self.mapping.upper)
            self._open_along(frames[1:], target, cancelled, name)
        measured = self._measured()
        short = float(np.max(np.abs(target - measured)))
        row.update(short_of_target_rad=short, measured=measured.tolist())
        if short > OPEN_SHORT_MAX_RAD:
            return EffectorOutcome(False, f"hand blocked {short:.3f} rad short of open", report=dict(row))
        if not is_open(measured, target):
            row["open_short"] = True
        return EffectorOutcome(True, report=dict(row))

    def _open_along(self, frames, target: np.ndarray, cancelled: Flag, name: str) -> None:
        """Send ``frames`` and hold ``target`` briefly, then until the hand is open, up to :data:`OPEN_SETTLE_MAX_S`."""
        row = self.report[name]
        frames = list(frames)
        if frames:
            frames += [target] * int(CLOSE_SETTLE_S * HAND_HZ)
        for frame in frames:
            self._frame(frame, cancelled, name)
        row["frames"] += len(frames)
        ended = self.io.monotonic() + OPEN_SETTLE_MAX_S
        while not is_open(self._measured(), target) and self.io.monotonic() < ended:
            self._frame(target, cancelled, name)
            row["settle_frames"] += 1

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

    def _fresh(self):
        sample = self.io.feedback()
        if not sample.healthy or not 0 <= self.io.monotonic() - sample.received_at <= self.feedback_max_age_s:
            raise _Failed("hand feedback unhealthy or stale")
        return sample

    def _measured(self) -> np.ndarray:
        sample = self._fresh()
        side = self.mapping.side
        offset = HAND_OFFSETS[side]
        joints = self.io.channel_joints[offset : offset + len(HAND_PARTS)]
        if len(joints) != len(HAND_PARTS):
            raise _Failed(f"invalid hand feedback: hand stream describes fewer than ten {side} joints")
        try:
            return self.mapping.check([sample.positions[name] for name in joints])
        except (KeyError, ValueError) as exc:
            raise _Failed(f"invalid hand feedback: {exc}") from None

    def _send(self, hand: np.ndarray) -> None:
        hand = self.mapping.check(hand)
        if self.io.refusal():
            raise _Failed(f"runtime does not admit output: {self.io.refusal()}")
        sample = self._fresh()
        if len(self.io.channel_joints) != 2 * len(HAND_PARTS):
            raise _Failed("hand stream must describe both hands")
        preserved_side = other_side(self.mapping.side)
        other_offset = HAND_OFFSETS[preserved_side]
        other_joints = self.io.channel_joints[other_offset : other_offset + len(HAND_PARTS)]
        if self.absent_other:
            if any(name in sample.positions for name in other_joints):
                raise _Failed(f"{preserved_side} hand feedback conflicts with installed_hands")
            # Not installed: nothing to measure or preserve; its slots carry the rest command.
            if self._preserved_hand is None:
                self._preserved_hand = np.array(ABSENT_HAND_COMMAND)
                self.report["preserved_hand"] = dict(side=preserved_side, installed=False)
        else:
            try:
                other = np.asarray([sample.positions[name] for name in other_joints], dtype=float)
            except (KeyError, TypeError, ValueError) as exc:
                raise _Failed(f"invalid preserved hand feedback: {exc}") from None
            if other.shape != (len(HAND_PARTS),) or not np.isfinite(other).all():
                raise _Failed("ten finite preserved hand positions required")
            if self._preserved_hand is None:
                self._preserved_hand = other.copy()
                self.report["preserved_hand"] = dict(side=preserved_side, positions=other.tolist())
            if np.max(np.abs(other - self._preserved_hand)) > PRESERVED_HAND_DRIFT_RAD:
                raise _Failed("preserved hand moved during grasp")
        if self.mapping.side == "left":
            self.io.publish([*hand.tolist(), *self._preserved_hand.tolist()])
        else:
            self.io.publish([*self._preserved_hand.tolist(), *hand.tolist()])
        self.command = hand.copy()

    def _frame(self, hand: np.ndarray, cancelled: Flag, name: str) -> None:
        tick = self.io.monotonic()
        if cancelled():
            raise _Failed(f"cancelled during {name}")
        self._measured()
        self._send(hand)
        self.io.wait(max(0.0, tick + 1.0 / HAND_HZ - self.io.monotonic()))


def touch_levels(message, side: str) -> dict[str, float] | None:
    """Max reading of each pad of ``side``'s hand from the runtime's touch array (layout labels ``<side>_<pad>``)."""
    prefix = f"{check_side(side)}_"
    levels, offset = {}, 0
    for dimension in message.layout.dim:
        cells = list(message.data[offset : offset + dimension.size])
        offset += dimension.size
        if dimension.label.startswith(prefix) and cells:
            levels[dimension.label[len(prefix) :]] = float(max(cells))
    return levels or None


def left_touch_levels(message) -> dict[str, float] | None:
    return touch_levels(message, "left")


class RosHandIO:
    """``HandIO`` over the runtime's hand stream, hand state and touch array, bound by interface ID.

    Like the stream's, the touch array is taken when read. Call :meth:`close` before ``node`` is destroyed.
    """

    def __init__(self, node, description: Mapping, side: str = "left"):
        from std_msgs.msg import Float64MultiArray

        from manipulation_execution.grasp_pipeline.ros_stream import RosStreamIO, bind_stream
        from robot_config.contract_utils import qos_profile_from_dict
        from robot_runtime.on_demand import OnDemandSubscriptions

        self.side = check_side(side)
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
        self.touch_configured = touch is not None
        if touch is not None:
            self._touch_subscriptions.subscribe(
                Float64MultiArray,
                touch["endpoint"],
                qos_profile_from_dict(touch.get("qos") or {}),
                1,
                self._on_touch,
            )

    def _on_touch(self, message, received: float | None) -> None:
        self._touch, self._touch_received = touch_levels(message, self.side), received_at_or_stale(received)

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
