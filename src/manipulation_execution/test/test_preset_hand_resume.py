"""Fresh hand targets and verified resume seeds must not jump to old tracking state."""

import numpy as np
import pytest

from manipulation_execution.mirror_session.hand_overlay import HandOverlay


def test_reading_a_fresh_follow_target_does_not_advance_the_hand_output():
    overlay = HandOverlay(50)
    overlay.observe_follow(np.zeros(20), 10.0)
    assert overlay.follow_target(10.1) == (0.0,) * 20
    assert overlay._prev is None
    assert overlay.follow_target(10.31) is None
    assert overlay.follow_target(9.9) is None


def test_reseed_preserves_exact_completed_native_values():
    overlay = HandOverlay(50)
    target = np.zeros(20)
    target[4] = 0.2
    overlay.reseed(target, 10.0)
    assert overlay._prev.tolist() == target.tolist()
    assert overlay._last_apply == 10.0
    overlay.observe_follow(target, 10.0)
    assert overlay.apply(True, 10.01) == tuple(target)


@pytest.mark.parametrize("target", [[0.0] * 19, [float("nan")] * 20, [100.0] * 20])
def test_bad_resume_seed_leaves_the_last_valid_output_untouched(target):
    overlay = HandOverlay(50)
    previous = np.zeros(20)
    overlay.reseed(previous, 10.0)
    with pytest.raises(ValueError):
        overlay.reseed(target, 10.1)
    assert np.array_equal(overlay._prev, previous)
