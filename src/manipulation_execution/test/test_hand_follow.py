"""x2.hand_follow on the packaged hand_core, its thumb layer, and the session's hand overlay. No ROS, no NPU."""

import numpy as np
import pytest

from manipulation_execution.hand_perception import npu_hands
from manipulation_execution.imitation_retargeting.x2 import thumb_fold
from manipulation_execution.imitation_retargeting.x2.hand_follow import HAND_COUNT, HandFollow, thumb_tuck
from manipulation_execution.mirror_session.hand_overlay import HAND_HIGH, HAND_LOW, HAND_NEUTRAL, HandOverlay

FOLD = (0.55, 0.85, 0.25, 0.60)


def _follow(**kwargs):
    kwargs = {"latency_s": 0.1, "lookahead_s": 0.25, "mirror": True, "out_hz": 50.0, **kwargs}
    return HandFollow(**kwargs)


def _world_hand(thumb_tip=(0.2, 0.0, 0.0)):
    """21 keypoints: palm 0.1 long along y, fingertips 0.8 palm apart at y = 0.2 (past tip_out), thumb tip where asked."""
    a = np.zeros((21, 3))
    a[9] = (0.0, 0.1, 0.0)
    a[17] = (-0.05, 0.1, 0.0)
    for k, tip in enumerate((8, 12, 16, 20)):
        a[tip] = (-0.08 * k, 0.2, 0.0)
    a[4] = thumb_tip
    return a.ravel()


def _run(hf, obs, seconds=2.0, hz=30.0):
    """Feed the same observation for both hands; returns the last tick's values."""
    q = None
    for k in range(int(seconds * hz)):
        t = 100.0 + k / hz
        hf.on_hand(t, t, obs, obs)
        q = hf.tick(t)  # t_out = t - latency_s: inside the frames, not past the newest
    return q


def test_constructs_on_the_packaged_hand_core():
    hand_core = npu_hands.load_extern("", npu_hands.HAND_CORE_PY)
    hf = _follow()
    assert isinstance(hf.stream, hand_core.HandStream)
    assert len(hf.neutral) == HAND_COUNT
    assert all(np.isfinite(hf.neutral))


def test_no_output_before_the_first_hand_frame():
    assert _follow().tick(10.0) is None


def test_open_and_closed_curls_give_different_finite_commands():
    outs = {}
    for name, curl in (("open", 0.2), ("closed", 2.0)):
        q = _run(_follow(gestures=False), ([curl] * 5, "", 0.0))
        assert len(q) == HAND_COUNT and np.all(np.isfinite(q))
        outs[name] = np.array(q)
    assert np.abs(outs["open"] - outs["closed"]).max() > 0.2


def test_a_label_over_the_gate_enters_its_preset():
    hf = _follow(gesture_labels=("Victory",), gesture_theta_in=0.6)
    _run(hf, ([0.5] * 5, "Victory", 0.95), seconds=1.0)
    assert any(mode != "track" for mode, _ in hf.stream.gesture_state())


def test_a_label_under_the_gate_stays_tracking():
    hf = _follow(gesture_labels=("Victory",), gesture_theta_in=0.6)
    _run(hf, ([0.5] * 5, "Victory", 0.3), seconds=1.0)
    assert all(mode == "track" for mode, _ in hf.stream.gesture_state())


def test_thumb_tuck_and_dists_refuse_missing_keypoints():
    assert np.isnan(thumb_tuck(np.full(63, np.nan)))
    assert np.isnan(thumb_tuck(np.zeros(63)))
    assert np.all(np.isnan(thumb_fold.thumb_dists(np.full(63, np.nan))))
    assert np.all(np.isnan(thumb_fold.thumb_dists(np.zeros(63))))


def test_thumb_dists_are_over_palm_length():
    d = thumb_fold.thumb_dists(_world_hand(thumb_tip=(0.0, 0.2, 0.0)))
    assert d.shape == (5,)
    assert d[0] == pytest.approx(0.0)
    assert d[1] == pytest.approx(0.8)


def test_thumb_fold_needs_ordered_thresholds():
    hf = _follow()
    with pytest.raises(ValueError, match="in < out"):
        thumb_fold.ThumbFold(hf.cfg, 50.0, 0.9, 0.5)


def test_thumb_fold_stage_is_the_finger_the_tip_touches():
    far = _follow(gestures=False, thumb_fold=FOLD)
    _run(far, ([0.3] * 5, "", 0.0, _world_hand()))
    assert all(np.isnan(far.fold.stage))

    touching = _follow(gestures=False, thumb_fold=FOLD)
    _run(touching, ([0.3] * 5, "", 0.0, _world_hand(thumb_tip=(0.0, 0.2, 0.0))))
    assert touching.fold.stage == pytest.approx([1.0, 1.0])
    assert touching.fold.closeness == pytest.approx([1.0, 1.0])


def _mid():
    return list((HAND_LOW + HAND_HIGH) / 2.0)


def test_overlay_not_following_ignores_fresh_follow_values_and_goes_to_neutral():
    overlay = HandOverlay(50.0, neutral=list(HAND_NEUTRAL))
    out = None
    for k in range(200):
        now = k * 0.02
        overlay.observe_follow(_mid(), now)
        out = overlay.apply(False, now)
    assert out == pytest.approx(list(np.clip(HAND_NEUTRAL, HAND_LOW, HAND_HIGH)))
    assert overlay.counts["following"] == 0


def test_overlay_following_moves_to_the_follow_values():
    overlay = HandOverlay(50.0, neutral=list(HAND_NEUTRAL))
    out = None
    for k in range(200):
        now = k * 0.02
        overlay.observe_follow(_mid(), now)
        out = overlay.apply(True, now)
    assert overlay.counts["following"] == 200
    assert out == pytest.approx(_mid())


def test_overlay_following_with_stale_values_goes_to_neutral():
    overlay = HandOverlay(50.0, neutral=list(HAND_NEUTRAL))
    overlay.observe_follow(_mid(), 0.0)
    out = None
    for k in range(200):
        out = overlay.apply(True, 1.0 + k * 0.02)
    assert overlay.counts["following"] == 0
    assert out == pytest.approx(list(np.clip(HAND_NEUTRAL, HAND_LOW, HAND_HIGH)))
