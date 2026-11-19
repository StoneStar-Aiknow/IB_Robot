"""Where the gesture model looks (hand_perception/gesture_crop.py): gating, the square, the rate limits."""

import math

import numpy as np
import pytest

from manipulation_execution.hand_perception import gesture_crop as gc

BOX = (100.0, 0.0, 500.0, 1000.0)  # person box 1000 px high
CFG = gc.CropConfig()


def hand(cx, cy, size=40.0):
    """21 points: wrist below, MCPs above; mean exactly (cx, cy); palm = size (wrist->middle MCP)."""
    lm = np.zeros((21, 2))
    lm[gc.WRIST] = (0.0, size / 2)
    lm[gc.MIDDLE_MCP] = (0.0, -size / 2)
    lm[gc.INDEX_MCP] = (-size * 0.3, -size / 2)
    lm[gc.PINKY_MCP] = (size * 0.3, -size / 2)
    lm -= lm.mean(axis=0)
    return lm + (cx, cy)


def test_palm_is_the_longer_of_length_and_1_25_width():
    lm = hand(0, 0, size=40.0)
    assert gc.palm_px(lm) == pytest.approx(40.0)  # width 24 * 1.25 = 30 < 40
    wide = lm.copy()
    wide[gc.INDEX_MCP] = (-30.0, 0.0)
    wide[gc.PINKY_MCP] = (30.0, 0.0)
    assert gc.palm_px(wide) == pytest.approx(75.0)


def test_no_hand_does_not_run():
    plan = gc.plan_crop([], BOX, CFG)
    assert not plan.run and plan.reason == "hands"


def test_non_finite_or_misshapen_hands_are_not_hands():
    bad = hand(300, 300)
    bad[3] = np.nan
    assert gc.plan_crop([bad, np.zeros((20, 2))], BOX, CFG).reason == "hands"


def test_bad_person_box_does_not_run():
    for box in (None, (0, 0, 1, float("nan")), (0, 10, 1, 10), (1, 2, 3)):
        assert gc.plan_crop([hand(300, 300)], box, CFG).reason == "box"


def test_two_hands_far_apart_is_skipped():
    plan = gc.plan_crop([hand(200, 300), hand(200 + 601, 300)], BOX, CFG)
    assert not plan.run and plan.reason == "far" and plan.dist_ratio == pytest.approx(0.601)


def test_two_hands_close_crop_the_union_times_k_square_on_the_long_edge():
    a, b = hand(250, 300), hand(350, 320)
    plan = gc.plan_crop([a, b], BOX, CFG)
    pts = np.concatenate([a, b])
    lo, hi = pts.min(0), pts.max(0)
    side = 1.5 * max(hi - lo)
    assert plan.run and plan.mode == gc.BOTH
    assert plan.side == round(side)
    cx, cy = (lo + hi) / 2
    assert plan.x0 == round(cx - plan.side / 2) and plan.y0 == round(cy - plan.side / 2)
    assert plan.roi == (plan.x0, plan.y0, plan.x0 + plan.side, plan.y0 + plan.side)
    assert plan.dist_ratio == pytest.approx(math.hypot(100, 20) / 1000)
    assert plan.palm == pytest.approx(40.0)


def test_distance_at_the_ratio_still_runs():
    plan = gc.plan_crop([hand(200, 300), hand(800, 300)], BOX, CFG)
    assert plan.run and plan.dist_ratio == pytest.approx(0.6)


def test_one_hand_is_a_square_of_palms_on_the_hand_centre():
    plan = gc.plan_crop([hand(300, 400, size=30.0)], BOX, CFG)
    assert plan.run and plan.mode == gc.ONE and plan.side == 120
    assert plan.x0 == 240 and plan.y0 == 340
    assert math.isnan(plan.dist_ratio)


def test_two_hands_use_the_larger_palm():
    plan = gc.plan_crop([hand(300, 300, size=20.0), hand(340, 300, size=50.0)], BOX, CFG)
    assert plan.palm == pytest.approx(50.0)


def test_square_under_min_side_is_skipped():
    plan = gc.plan_crop([hand(300, 300, size=15.0)], BOX, CFG)  # 4 * 15 = 60 < 64
    assert not plan.run and plan.reason == "small" and plan.side == 60
    assert gc.plan_crop([hand(300, 300, size=16.0)], BOX, CFG).run


def test_absurd_square_is_skipped():
    plan = gc.plan_crop([hand(300, 300, size=2000.0)], (0, 0, 10, 100000), CFG)
    assert not plan.run and plan.reason == "large"


def test_square_is_cut_to_the_image_long_edge_about_its_centre():
    garbage = [hand(5000, -3000, size=2000.0), hand(5200, -2900, size=2000.0)]  # far off the image, huge
    plan = gc.plan_crop(garbage, (0, 0, 10, 100000), CFG, (480, 640))
    assert plan.run and plan.side == 640
    cx, cy = plan.x0 + plan.side / 2, plan.y0 + plan.side / 2
    lo, hi = np.concatenate(garbage).min(axis=0), np.concatenate(garbage).max(axis=0)
    assert (cx, cy) == pytest.approx((lo + hi) / 2, abs=1.0)
    assert gc.extract_square(np.zeros((480, 640, 3), np.uint8), plan.x0, plan.y0, plan.side).nbytes == 640 * 640 * 3
    fits = gc.plan_crop([hand(300, 300)], BOX, CFG, (480, 640))  # under the long edge: as without the size
    assert fits.roi == gc.plan_crop([hand(300, 300)], BOX, CFG).roi == (220, 220, 380, 380)


def test_thresholds_are_parameters():
    cfg = gc.CropConfig(far_ratio=0.3, k=2.0, one_hand_palms=2.0, min_side_px=100.0)
    assert gc.plan_crop([hand(200, 300), hand(600, 300)], BOX, cfg).reason == "far"
    assert gc.plan_crop([hand(300, 300, size=40.0)], BOX, cfg).reason == "small"  # 80 < 100


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "1"])
def test_config_rejects_non_positive_or_non_numbers(bad):
    with pytest.raises(ValueError):
        gc.CropConfig(k=bad)


def test_rate_all_runs_at_most_15hz_by_stamp():
    gate = gc.RateGate(CFG)
    ms = 1_000_000
    assert gate.allow(gc.BOTH, 0)
    gate.mark(gc.BOTH, 0)
    assert not gate.allow(gc.BOTH, 33 * ms)
    assert gate.allow(gc.BOTH, 66 * ms)  # two 30 fps frames, a hair under 1/15 s


def test_rate_one_hand_at_most_5hz_but_both_is_not_held_back_by_it():
    gate = gc.RateGate(CFG)
    ms = 1_000_000
    gate.mark(gc.ONE, 0)
    assert not gate.allow(gc.ONE, 100 * ms)
    assert gate.allow(gc.BOTH, 100 * ms)
    gate.mark(gc.BOTH, 100 * ms)
    assert not gate.allow(gc.ONE, 180 * ms)  # 15 Hz ok, 5 Hz since the last one-hand run not
    assert gate.allow(gc.ONE, 200 * ms)


def test_busy_lets_a_frame_through_at_10hz_by_stamp():
    ms = 1_000_000
    gate = gc.RateGate(CFG)
    assert gate.allow_busy(0)
    gate.mark(gc.BOTH, 1000 * ms)
    assert not gate.allow_busy(1067 * ms)
    assert gate.allow_busy(1096 * ms)  # within the 5 ms slack
    assert gate.allow_busy(900 * ms)  # stamp went back
    assert gc.RateGate(gc.CropConfig(busy_rate_hz=5.0)).allow_busy(0)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), "x", True])
def test_busy_rate_must_be_positive(bad):
    with pytest.raises(ValueError):
        gc.CropConfig(busy_rate_hz=bad)


def test_rate_stamp_going_back_forgets_the_history():
    gate = gc.RateGate(CFG)
    gate.mark(gc.ONE, 10_000_000_000)
    assert gate.allow(gc.ONE, 9_000_000_000)
    assert gate.last_any is None and gate.last_one is None


def test_extract_square_pads_outside_the_image_with_114():
    img = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    sq = gc.extract_square(img, -1, -2, 4)
    assert sq.shape == (4, 4, 3)
    assert (sq[:2] == 114).all() and (sq[:, 0] == 114).all()
    np.testing.assert_array_equal(sq[2:, 1:], img[0:2, 0:3])
    assert sq.base is None or sq.base is not img  # a copy, never a view of the frame


def test_extract_square_fully_outside_is_all_pad():
    img = np.zeros((10, 10, 3), np.uint8)
    assert (gc.extract_square(img, 50, 50, 8) == 114).all()
