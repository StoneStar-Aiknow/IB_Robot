"""The OmniHand end effector against a simulated hand that an object can block."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from aimdk_grasp.hand import HandMapping
from aimdk_grasp.joints import HAND_SIGNS, LEFT_HAND
from aimdk_grasp.omnihand import (
    CONTACT_LAG_RAD,
    FINGERS,
    GRIP_LOST_S,
    HAND_HZ,
    OmniHandEffector,
    left_touch_levels,
    touch_levels,
)
from aimdk_grasp.robot_model import planning_model
from test_hand import PRESETS, RELEASE

from manipulation_execution.grasp_pipeline.contracts import EffectorAction
from manipulation_execution.grasp_pipeline.ros_stream import RosStreamIO, bind_stream
from manipulation_execution.grasp_pipeline.streaming import JointSample
from robot_runtime.interface_description import build_description
from robot_runtime.profile import load_profile

PROFILE = Path(__file__).resolve().parents[2] / "aimdk_robot" / "profiles" / "x2_ultra_omnihand.yaml"


@pytest.fixture(scope="module")
def mapping():
    return HandMapping(planning_model(), PRESETS)


class SimHand:
    """Native positions follow the command; a blocked finger's joints stop at their block."""

    channel_joints = tuple(f"left_{name.removeprefix('L_').removesuffix('_joint')}" for name in LEFT_HAND)
    channel_joints += tuple(name.replace("left_", "right_", 1) for name in channel_joints)

    def __init__(self, start, *, touch_sensor=True):
        self.t = 10.0
        self.feedback_received_at = self.t
        self.measured = np.array(start, dtype=float)
        self.command = self.measured.copy()
        self.other = np.asarray(RELEASE, dtype=float) * HAND_SIGNS
        self.full_published = []
        self.blocks: dict[int, float] = {}  # joint index -> furthest native position reachable
        self.touching: set[str] = set()  # pads pressed against the object
        self.touch_sensor = touch_sensor
        self.published, self.released, self.refused, self.stale = [], 0, None, False
        self.freeze_feedback_after_publish: int | None = None
        self.feedback_frozen = False

    def block(self, finger, at, *, touch=True):
        joints, flex, pad = FINGERS[finger]
        self.blocks[flex] = at
        if touch:
            self.touching.add(pad)

    def monotonic(self):
        return self.t

    def wait(self, seconds):
        self.t += seconds
        self.measured += (self.command - self.measured) * min(1.0, seconds / 0.02)
        for joint, at in self.blocks.items():
            start = self.published[0][joint] if self.published else self.measured[joint]
            closing = np.sign(at - start) or 1.0
            if (self.measured[joint] - at) * closing > 0:
                self.measured[joint] = at
        if not self.feedback_frozen:
            self.feedback_received_at = self.t

    def feedback(self):
        received_at = self.feedback_received_at - (1.0 if self.stale else 0.0)
        return JointSample(received_at, dict(zip(self.channel_joints, [*self.measured, *self.other], strict=True)))

    def touch(self):
        if not self.touch_sensor:
            return None
        levels = {}
        for _, (_, flex, pad) in FINGERS.items():
            pressed = (
                pad in self.touching and flex in self.blocks and abs(self.measured[flex] - self.blocks[flex]) < 0.02
            )
            levels[pad] = 100.0 if pressed else 0.0
        return levels

    def publish(self, positions):
        assert len(positions) == 20
        self.full_published.append(np.array(positions))
        self.other = np.array(positions[10:])
        self.published.append(np.array(positions[:10]))
        self.command = np.array(positions[:10])
        if self.freeze_feedback_after_publish is not None and len(self.published) >= self.freeze_feedback_after_publish:
            self.feedback_frozen = True

    def release(self):
        self.released += 1

    def refusal(self):
        return self.refused


def act(effector, action, halted=lambda: False, cancelled=lambda: False):
    return effector.act(action, halted=halted, cancelled=cancelled)


def test_preshape_opens_along_a_straight_line_and_skips_when_already_open(mapping):
    hand = SimHand(RELEASE)
    effector = OmniHandEffector(hand, mapping)
    outcome = act(effector, EffectorAction.PRESHAPE)
    assert outcome.succeeded, outcome.reason
    open_hand = mapping.preset("open")
    np.testing.assert_allclose(hand.measured, open_hand, atol=0.025)
    steps = np.abs(np.diff(np.array(hand.published), axis=0)).max()
    assert steps <= 0.01 + 1e-9
    sent = len(hand.published)
    assert act(effector, EffectorAction.PRESHAPE).succeeded
    assert len(hand.published) == sent  # commanded open and measured open: nothing to send


def test_real_hand_state_aliases_are_consumed_under_the_stream_names(mapping):
    description = build_description(load_profile(os.fspath(PROFILE)), use_sim=True)
    interfaces = description["interfaces"]
    stream_names = tuple(interfaces["joint.hand_stream"]["joint_names"])
    state_names = tuple(interfaces["joint.hand_state"]["joint_names"])
    aliases = dict(zip(stream_names, state_names, strict=True))
    binding = bind_stream(
        description,
        stream="joint.hand_stream",
        feedback=["joint.hand_state"],
        feedback_names=aliases,
    )
    stream = RosStreamIO.__new__(RosStreamIO)
    stream.binding = binding
    stream._subscriptions = SimpleNamespace(drain=lambda: None)
    stream._lock = threading.Lock()
    stream._views = {"joint.hand_state": (10.0, dict(zip(state_names, [*RELEASE, *hand_right_pose()], strict=True)))}

    hand = SimHand(RELEASE)
    hand.channel_joints = stream_names
    hand.feedback = stream.feedback
    measured = OmniHandEffector(hand, mapping)._measured()

    assert tuple(stream.feedback().positions) == stream_names
    np.testing.assert_allclose(measured, RELEASE)


def test_thumb_and_one_finger_on_the_object_is_a_grip(mapping):
    hand = SimHand(mapping.preset("open"))
    effector = OmniHandEffector(hand, mapping)
    effector.command = mapping.preset("open")
    close = mapping.preset("close")
    open_hand = mapping.preset("open")
    for finger in ("thumb", "index"):
        flex = FINGERS[finger][1]
        hand.block(finger, open_hand[flex] + 0.4 * (close[flex] - open_hand[flex]))
    outcome = act(effector, EffectorAction.CLOSE)
    assert outcome.succeeded and outcome.held is True
    contacts = outcome.report["contacts"]
    assert set(contacts) >= {"thumb", "index"} and contacts["thumb"]["by"] == "touch"
    assert outcome.confidence == pytest.approx(len(contacts) / 5)


@pytest.mark.parametrize("touch_sensor", [True, False])
def test_a_grip_whose_fingers_close_onto_their_command_has_lost_the_object(mapping, touch_sensor):
    """Pressing fingers trail their frozen command (or read touch); slipped out, they reach it."""
    hand = SimHand(mapping.preset("open"), touch_sensor=touch_sensor)
    effector = OmniHandEffector(hand, mapping)
    assert effector.holding() is None  # nothing gripped yet
    close, open_hand = mapping.preset("close"), mapping.preset("open")
    for finger in ("thumb", "middle"):
        flex = FINGERS[finger][1]
        hand.block(finger, open_hand[flex] + 0.4 * (close[flex] - open_hand[flex]), touch=touch_sensor)
    assert act(effector, EffectorAction.CLOSE).held is True
    for _ in range(20):
        hand.wait(0.02)
        assert effector.holding() == ""  # held: the object keeps the fingers back
    hand.blocks.clear()
    hand.touching.clear()  # it slips out
    hand.wait(0.1)
    assert effector.holding() == ""  # not lost at the first look
    hand.wait(GRIP_LOST_S + 0.01)
    assert "no longer press" in effector.holding()
    assert act(effector, EffectorAction.RELEASE).succeeded and effector.holding() is None


def test_a_close_that_touches_nothing_holds_nothing(mapping):
    hand = SimHand(mapping.preset("open"))
    outcome = act(OmniHandEffector(hand, mapping), EffectorAction.CLOSE)
    assert outcome.succeeded and outcome.held is False and outcome.confidence == 0.0
    np.testing.assert_allclose(hand.command, mapping.preset("close"))


def test_without_touch_a_finger_held_back_is_contact(mapping):
    hand = SimHand(mapping.preset("open"), touch_sensor=False)
    close, open_hand = mapping.preset("close"), mapping.preset("open")
    for finger in ("thumb", "middle"):
        flex = FINGERS[finger][1]
        hand.block(finger, open_hand[flex] + 0.3 * (close[flex] - open_hand[flex]), touch=False)
    outcome = act(OmniHandEffector(hand, mapping), EffectorAction.CLOSE)
    assert outcome.held is True
    assert all(row["by"] == "lag" and row["lag_rad"] >= CONTACT_LAG_RAD for row in outcome.report["contacts"].values())


def test_a_halt_during_the_close_holds_the_fingers_where_they_are(mapping):
    hand = SimHand(mapping.preset("open"))
    effector = OmniHandEffector(hand, mapping)
    calls = [0]

    def halted():
        calls[0] += 1
        return calls[0] > 8

    outcome = act(effector, EffectorAction.CLOSE, halted=halted)
    assert outcome.halted and outcome.held is None
    np.testing.assert_allclose(effector.command, hand.published[-1])
    assert act(effector, EffectorAction.RELEASE).succeeded


def test_an_open_blocked_far_from_the_preset_fails_and_a_little_short_passes(mapping):
    close = mapping.preset("close")
    hand = SimHand(close)
    flex = FINGERS["index"][1]
    hand.block("index", close[flex] - 0.2, touch=False)  # 0.2 rad of the way, then stuck
    outcome = act(OmniHandEffector(hand, mapping), EffectorAction.RELEASE)
    assert not outcome.succeeded and "blocked" in outcome.reason
    hand = SimHand(close)
    open_flex = mapping.preset("open")[flex]
    hand.block("index", open_flex + 0.05 * np.sign(close[flex] - open_flex), touch=False)
    effector = OmniHandEffector(hand, mapping)
    outcome = act(effector, EffectorAction.RELEASE)
    assert outcome.succeeded and outcome.report["open_short"]
    effector.io = SimHand(close)  # the next release is not blocked: it reports nothing short
    outcome = act(effector, EffectorAction.RELEASE)
    assert outcome.succeeded and "open_short" not in outcome.report


@pytest.mark.parametrize("problem", ["refused", "cancelled", "stale"])
def test_the_hand_does_not_move_on_a_refusal_a_cancel_or_stale_feedback(mapping, problem):
    hand = SimHand(RELEASE)
    if problem == "refused":
        hand.refused = "stop latched"
    if problem == "stale":
        hand.stale = True
    outcome = act(OmniHandEffector(hand, mapping), EffectorAction.PRESHAPE, cancelled=lambda: problem == "cancelled")
    assert not outcome.succeeded and not hand.published


def test_open_stops_sending_when_feedback_expires_after_motion_starts(mapping):
    hand = SimHand(mapping.preset("close"))
    hand.freeze_feedback_after_publish = 1
    outcome = act(OmniHandEffector(hand, mapping), EffectorAction.RELEASE)

    assert not outcome.succeeded and "stale" in outcome.reason
    assert len(hand.published) <= int(0.1 * HAND_HZ) + 1


def test_finish_hands_the_last_target_to_the_platform(mapping):
    hand = SimHand(mapping.preset("open"))
    effector = OmniHandEffector(hand, mapping)
    act(effector, EffectorAction.CLOSE)
    effector.finish()
    assert hand.released == 1 and effector.command is None


def test_touch_levels_are_read_per_left_pad():
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension

    message = Float64MultiArray()
    for label, values in (("left_index_finger", [0, 40, 3]), ("left_thumb", [5]), ("right_thumb", [99, 99])):
        message.layout.dim.append(MultiArrayDimension(label=label, size=len(values)))
        message.data.extend(float(value) for value in values)
    assert left_touch_levels(message) == {"index_finger": 40.0, "thumb": 5.0}
    assert left_touch_levels(Float64MultiArray()) is None


def test_touch_readings_older_than_a_frame_or_two_are_not_contact():
    """When the touch stream stops, the last grip's levels must not read as contact on the next close."""
    from aimdk_grasp.omnihand import TOUCH_MAX_AGE_S, RosHandIO

    clock = [100.0]
    io = RosHandIO.__new__(RosHandIO)
    io._touch_subscriptions = SimpleNamespace(drain=lambda: 0)
    io._stream = SimpleNamespace(monotonic=lambda: clock[0])
    io._touch, io._touch_received = {"thumb": 120.0}, 100.0
    assert io.touch() == {"thumb": 120.0}
    clock[0] += TOUCH_MAX_AGE_S + 0.01
    assert io.touch() is None


class RightSimHand(SimHand):
    """The same simulated hand in the right slots of the 20-value stream."""

    def __init__(self, start, **kwargs):
        super().__init__(start, **kwargs)
        self.other = np.asarray(RELEASE, dtype=float)

    def feedback(self):
        received_at = self.feedback_received_at - (1.0 if self.stale else 0.0)
        return JointSample(received_at, dict(zip(self.channel_joints, [*self.other, *self.measured], strict=True)))

    def publish(self, positions):
        assert len(positions) == 20
        self.full_published.append(np.array(positions))
        self.other = np.array(positions[:10])
        self.published.append(np.array(positions[10:]))
        self.command = np.array(positions[10:])


@pytest.fixture(scope="module")
def right_mapping():
    from aimdk_grasp.robot_model import config_directory

    return HandMapping(planning_model(config_directory("right")), PRESETS, side="right")


def hand_right_pose():
    return np.asarray(RELEASE) * HAND_SIGNS


@pytest.mark.parametrize("side", ["left", "right"])
def test_grasp_and_release_preserve_the_other_open_hand(mapping, right_mapping, side):
    active, other = (mapping, right_mapping) if side == "left" else (right_mapping, mapping)
    io = (SimHand if side == "left" else RightSimHand)(active.preset("open"))
    io.other = other.preset("open").copy()
    preserved = io.other.copy()
    effector = OmniHandEffector(io, active)
    close, opened = active.preset("close"), active.preset("open")
    for finger in ("thumb", "index"):
        flex = FINGERS[finger][1]
        io.block(finger, opened[flex] + 0.4 * (close[flex] - opened[flex]))
    assert act(effector, EffectorAction.PRESHAPE).succeeded
    assert act(effector, EffectorAction.CLOSE).held is True
    io.blocks.clear()
    io.touching.clear()
    assert act(effector, EffectorAction.RELEASE).succeeded
    effector.finish()
    other_slice = slice(10, 20) if side == "left" else slice(0, 10)
    for command in io.full_published:
        np.testing.assert_array_equal(command[other_slice], preserved)
    other.require_open(io.other)
    active.require_open(io.measured)
    assert io.released == 1


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("problem", ["missing", "nan", "moved"])
def test_preserved_hand_feedback_failure_stops_output(mapping, right_mapping, side, problem):
    active, other = (mapping, right_mapping) if side == "left" else (right_mapping, mapping)
    io = (SimHand if side == "left" else RightSimHand)(active.preset("open"))
    io.other = other.preset("open").copy()
    effector = OmniHandEffector(io, active)
    if problem == "moved":
        assert act(effector, EffectorAction.PRESHAPE).succeeded
        io.other[0] += 0.03
    else:
        feedback = io.feedback

        def broken_feedback():
            sample = feedback()
            name = io.channel_joints[10 if side == "left" else 0]
            if problem == "missing":
                del sample.positions[name]
            else:
                sample.positions[name] = float("nan")
            return sample

        io.feedback = broken_feedback
    count = len(io.full_published)
    outcome = act(effector, EffectorAction.CLOSE)
    assert not outcome.succeeded and "preserved hand" in outcome.reason
    assert len(io.full_published) == count


@pytest.mark.parametrize("side", ["left", "right"])
def test_finish_discards_the_old_preserved_pose(mapping, right_mapping, side):
    active, other = (mapping, right_mapping) if side == "left" else (right_mapping, mapping)
    io = (SimHand if side == "left" else RightSimHand)(active.preset("open"))
    effector = OmniHandEffector(io, active)
    assert act(effector, EffectorAction.PRESHAPE).succeeded
    effector.finish()
    io.other = other.preset("open").copy()
    assert act(effector, EffectorAction.PRESHAPE).succeeded
    other_slice = slice(10, 20) if side == "left" else slice(0, 10)
    np.testing.assert_array_equal(io.full_published[-1][other_slice], io.other)


@pytest.mark.parametrize("side", ["left", "right"])
def test_grasp_and_release_with_only_the_declared_hand_present(mapping, right_mapping, side):
    active = mapping if side == "left" else right_mapping
    io = (SimHand if side == "left" else RightSimHand)(active.preset("open"))
    original = io.feedback
    active_names = io.channel_joints[:10] if side == "left" else io.channel_joints[10:]

    def feedback():
        sample = original()
        return JointSample(sample.received_at, {name: sample.positions[name] for name in active_names})

    io.feedback = feedback
    effector = OmniHandEffector(io, active, absent_other=True)
    close, opened = active.preset("close"), active.preset("open")
    for finger in ("thumb", "index"):
        flex = FINGERS[finger][1]
        io.block(finger, opened[flex] + 0.4 * (close[flex] - opened[flex]))
    assert act(effector, EffectorAction.PRESHAPE).succeeded
    assert act(effector, EffectorAction.CLOSE).held is True
    io.blocks.clear()
    io.touching.clear()
    assert act(effector, EffectorAction.RELEASE).succeeded
    effector.finish()
    absent_slice = slice(10, 20) if side == "left" else slice(0, 10)
    for command in io.full_published:
        np.testing.assert_array_equal(command[absent_slice], np.zeros(10))
    active.require_open(io.measured)
    count = len(io.full_published)
    io.feedback = original  # A newly attached hand must not receive the absent-hand placeholder.
    outcome = act(effector, EffectorAction.PRESHAPE)
    assert not outcome.succeeded and "installed_hands" in outcome.reason
    assert len(io.full_published) == count


def test_the_right_hand_opens_and_grips_in_its_own_slots(right_mapping):
    hand = RightSimHand(np.asarray(RELEASE) * HAND_SIGNS)
    effector = OmniHandEffector(hand, right_mapping)
    assert act(effector, EffectorAction.PRESHAPE).succeeded
    np.testing.assert_allclose(hand.measured, right_mapping.preset("open"), atol=0.025)
    close, open_hand = right_mapping.preset("close"), right_mapping.preset("open")
    for finger in ("thumb", "ring"):
        flex = FINGERS[finger][1]
        hand.block(finger, open_hand[flex] + 0.5 * (close[flex] - open_hand[flex]))
    outcome = act(effector, EffectorAction.CLOSE)
    assert outcome.succeeded and outcome.held is True
    assert {"thumb", "ring"} <= set(outcome.report["contacts"])
    assert effector.holding() == ""


def test_the_right_hand_state_is_read_from_its_half_of_the_runtime_view(right_mapping):
    description = build_description(load_profile(os.fspath(PROFILE)), use_sim=True)
    interfaces = description["interfaces"]
    stream_names = tuple(interfaces["joint.hand_stream"]["joint_names"])
    state_names = tuple(interfaces["joint.hand_state"]["joint_names"])
    binding = bind_stream(
        description,
        stream="joint.hand_stream",
        feedback=["joint.hand_state"],
        feedback_names=dict(zip(stream_names, state_names, strict=True)),
    )
    stream = RosStreamIO.__new__(RosStreamIO)
    stream.binding = binding
    stream._subscriptions = SimpleNamespace(drain=lambda: None)
    stream._lock = threading.Lock()
    right = np.asarray(RELEASE) * HAND_SIGNS
    stream._views = {"joint.hand_state": (10.0, dict(zip(state_names, [*RELEASE, *right], strict=True)))}

    hand = RightSimHand(right)
    hand.channel_joints = stream_names
    hand.feedback = stream.feedback
    np.testing.assert_allclose(OmniHandEffector(hand, right_mapping)._measured(), right)


def test_touch_levels_are_read_per_pad_of_either_hand():
    from std_msgs.msg import Float64MultiArray, MultiArrayDimension

    message = Float64MultiArray()
    for label, values in (("left_thumb", [5]), ("right_thumb", [99, 98]), ("right_little_finger", [0, 31])):
        message.layout.dim.append(MultiArrayDimension(label=label, size=len(values)))
        message.data.extend(float(value) for value in values)
    assert touch_levels(message, "right") == {"thumb": 99.0, "little_finger": 31.0}
    assert touch_levels(message, "left") == left_touch_levels(message) == {"thumb": 5.0}
