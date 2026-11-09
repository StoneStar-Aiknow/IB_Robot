"""The trigger rules on synthetic inputs: body speed, a held pose, the rhythm, the gestures.

Poses are built so that the speed is known exactly: every body joint turns
about z by the same angle, so the mean joint angle between two frames is the
angle difference and the speed is its rate. Times are camera time.
"""

import math

import numpy as np
import pytest

from manipulation_execution.mirror_photo.config import EventRule, GestureRule, StillRule, UniformRule
from manipulation_execution.mirror_photo.stillness import (
    MAX_GAP_SEC,
    SpeedMeter,
    StillTrigger,
    body_rotations,
)
from manipulation_execution.mirror_photo.triggers import (
    EVENT,
    GESTURE,
    UNIFORM,
    GestureTrigger,
    UniformTrigger,
    event_trigger,
)


def pose(angle: float, *, root: float = 0.0, hands: float = 0.0) -> np.ndarray:
    """``smplx_pose_raw``: body joints turned ``angle`` about z, the root and the hands their own angles."""
    raw = np.zeros((52, 2, 3))
    for joint in range(52):
        theta = root if joint == 0 else angle if joint <= 21 else hands
        c, s = math.cos(theta), math.sin(theta)
        raw[joint, 0] = (c, s, 0.0)
        raw[joint, 1] = (-s, c, 0.0)
    return raw.reshape(-1)


def speeds(meter: SpeedMeter, rate: float, frames: int, *, start: float = 0.0, dt: float = 1 / 30, angle: float = 0.0):
    out = []
    for index in range(frames):
        out.append(meter.update(start + index * dt, pose(angle + rate * index * dt)))
    return out


# -- speed -------------------------------------------------------------------------------


def test_the_rotations_decode_back_to_the_pose():
    rotations = body_rotations(pose(0.3))
    assert rotations.shape == (21, 3, 3)
    expected = np.array([[math.cos(0.3), -math.sin(0.3), 0], [math.sin(0.3), math.cos(0.3), 0], [0, 0, 1]])
    assert np.allclose(rotations, expected)


@pytest.mark.parametrize("raw", [np.zeros(311), np.full(312, np.nan)])
def test_a_malformed_pose_is_a_value_error(raw):
    with pytest.raises(ValueError):
        body_rotations(raw)


def test_a_steady_turn_reads_its_rate():
    meter = SpeedMeter()
    readings = speeds(meter, 0.8, 20)
    assert readings[0] is None
    assert readings[-1] == pytest.approx(0.8, rel=1e-6)


def test_the_root_and_the_hands_do_not_count():
    meter = SpeedMeter()
    for index in range(10):
        reading = meter.update(index / 30, pose(0.0, root=2.0 * index / 30, hands=3.0 * index / 30))
    assert reading == pytest.approx(0.0, abs=1e-6)


def test_the_speed_is_smoothed():
    meter = SpeedMeter()
    speeds(meter, 0.0, 10)
    jump = meter.update(10 / 30, pose(1.0 / 30))  # one frame at 1 rad/s
    assert 0.0 < jump < 1.0


def test_a_gap_in_the_stream_starts_over():
    meter = SpeedMeter()
    speeds(meter, 2.0, 10)
    assert meter.update(9 / 30 + MAX_GAP_SEC + 0.1, pose(0.0)) is None
    assert meter.update(9 / 30 + MAX_GAP_SEC + 0.1 + 1 / 30, pose(0.0)) == pytest.approx(0.0, abs=1e-9)


def test_an_old_or_repeated_frame_changes_nothing():
    meter = SpeedMeter()
    readings = speeds(meter, 0.5, 10)
    assert meter.update(5 / 30, pose(3.0)) is None
    assert meter.update(9 / 30, pose(3.0)) is None
    assert meter.update(10 / 30, pose(0.5 * 10 / 30)) == pytest.approx(readings[-1], rel=1e-6)


# -- a held pose -------------------------------------------------------------------------

RULE = StillRule(v_still=0.25, v_move=0.6, hold_sec=0.4, min_gap_sec=2.0)


def run(trigger: StillTrigger, profile, *, start: float = 0.0, dt: float = 1 / 30):
    """``profile``: speeds, one per frame; returns the fired stamps."""
    fired = []
    for index, speed in enumerate(profile):
        stamp = trigger.update(start + index * dt, speed)
        if stamp is not None:
            fired.append(stamp)
    return fired


def test_still_for_hold_sec_fires_once_on_the_slowest_frame():
    profile = [1.0] * 10 + [0.2, 0.15, 0.05, 0.1] + [0.2] * 20
    fired = run(StillTrigger(RULE), profile)
    assert fired == [pytest.approx(12 / 30)]


def test_still_but_not_for_long_enough_does_not_fire():
    profile = [1.0] * 10 + [0.1] * 10 + [1.0] * 10 + [0.1] * 10
    assert run(StillTrigger(RULE), profile) == []


def test_one_pose_held_long_is_one_candidate():
    assert len(run(StillTrigger(RULE), [1.0] * 5 + [0.1] * 300)) == 1


def test_the_next_pose_needs_a_move_and_the_gap():
    # Pose, a move above v_move, a second pose early (inside min_gap_sec), held on.
    profile = [0.1] * 15 + [0.8] * 5 + [0.1] * 90
    fired = run(StillTrigger(RULE), profile)
    assert len(fired) == 2
    # Re-armed min_gap_sec after the first fired (at 0.4 s), then held hold_sec more.
    assert fired[1] == pytest.approx(0.4 + RULE.min_gap_sec, abs=0.04)


def test_drifting_between_v_still_and_v_move_is_not_a_new_pose():
    profile = [0.1] * 15 + [0.4] * 90 + [0.1] * 30
    assert len(run(StillTrigger(RULE), profile)) == 1


def test_no_speed_breaks_the_stretch():
    profile = [0.1] * 8 + [None] + [0.1] * 8
    assert run(StillTrigger(RULE), profile) == []


def test_the_whole_chain_from_poses():
    meter = SpeedMeter()
    trigger = StillTrigger(RULE)
    fired = []
    angle = 0.0
    for index in range(150):
        stamp = index / 30
        moving = index < 30 or 75 <= index < 90
        angle += (1.5 if moving else 0.0) / 30
        stamp_fired = trigger.update(stamp, meter.update(stamp, pose(angle)))
        if stamp_fired is not None:
            fired.append(stamp_fired)
    assert len(fired) == 2
    assert 1.0 < fired[0] < 2.5
    assert 3.0 < fired[1] < 5.0


# -- rhythm, gestures, events -------------------------------------------------------------


def test_the_rhythm_fires_every_period():
    trigger = UniformTrigger(UniformRule(period_sec=3.0))
    fired = [stamp for stamp in np.arange(0.0, 10.0, 0.1) if trigger.update(float(stamp)) is not None]
    assert fired == pytest.approx([0.0, 3.0, 6.0, 9.0], abs=0.11)
    assert trigger.update(12.0).source == UNIFORM


GESTURES = GestureRule(min_score=0.6, hold_sec=0.2, min_gap_sec=1.5)


def feed(trigger, frames, *, dt=1 / 15):
    out = []
    for index, observations in enumerate(frames):
        out += [(round(index * dt, 3), fired) for fired in trigger.update(index * dt, observations)]
    return out


def test_a_gesture_held_fires_with_its_weight():
    fired = feed(GestureTrigger(GESTURES), [[("Victory", 0.9)]] * 6)
    assert len(fired) == 1
    stamp, trigger = fired[0]
    assert stamp == pytest.approx(0.2, abs=0.07)
    assert (trigger.source, trigger.label, trigger.weight, trigger.score) == (GESTURE, "Victory", 3.0, 0.9)


def test_a_flicker_a_low_score_or_an_unlisted_gesture_does_not():
    frames = (
        [[("Victory", 0.9)], [], [("Victory", 0.9)], []] * 3 + [[("Victory", 0.5)]] * 10 + [[("Open_Palm", 0.99)]] * 10
    )
    assert feed(GestureTrigger(GESTURES), frames) == []


def test_a_gesture_held_on_fires_again_after_the_gap():
    fired = feed(GestureTrigger(GESTURES), [[("Thumb_Up", 0.8)]] * 60)  # 4 s
    stamps = [stamp for stamp, _ in fired]
    assert len(stamps) == 3
    assert all(later - earlier >= 1.5 for earlier, later in zip(stamps, stamps[1:], strict=False))


def test_either_hand_and_each_label_on_its_own():
    frames = [[("Victory", 0.7), ("Thumb_Up", 0.9)]] * 4 + [[("Pointing_Up", 0.4), ("Pointing_Up", 0.8)]] * 4
    labels = sorted(trigger.label for _, trigger in feed(GestureTrigger(GESTURES), frames))
    assert labels == ["Pointing_Up", "Thumb_Up", "Victory"]


def test_an_event_weighs_its_label_or_the_default():
    rule = EventRule(default_weight=2.5, labels={"heart": 4.0})
    heart = event_trigger(rule, "preset_motions", "heart", 1.0, None)
    other = event_trigger(rule, "preset_motions", "wave", 0.5, 12.0)
    assert (heart.source, heart.weight, heart.origin, heart.stamp) == (EVENT, 4.0, "preset_motions", None)
    assert (other.weight, other.score, other.stamp) == (2.5, 0.5, 12.0)
    assert heart.priority and not UniformTrigger(UniformRule()).update(0.0).priority
