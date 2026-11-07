"""Image-time joint feedback assembly, without a ROS graph."""

from __future__ import annotations

import time

import numpy as np
import pytest
from aimdk_grasp.feedback import MAX_STATUS_AGE_S, FeedbackHistory
from aimdk_grasp.joints import HEAD, LEFT_ARM, LEFT_HAND, RIGHT_ARM

ARMS = LEFT_ARM + RIGHT_ARM
RIGHT_HAND = tuple(name.replace("L_", "R_", 1) for name in LEFT_HAND)
MS = 1_000_000


def _record(history: FeedbackHistory, stamp_ns: int, *, offset: float = 0.0) -> None:
    history.record("joint.state", stamp_ns, ARMS, [offset + i * 0.01 for i in range(len(ARMS))])
    history.record("joint.head_state", stamp_ns, HEAD, [offset + 0.5, offset + 0.6])
    history.record("joint.hand_state", stamp_ns, LEFT_HAND + RIGHT_HAND, [offset + 0.1] * 10 + [9.0] * 10)


def test_the_samples_nearest_the_image_are_assembled():
    history = FeedbackHistory()
    for index in range(5):
        _record(history, (1000 + 20 * index) * MS, offset=index)

    feedback = history.at(1041 * MS)

    assert feedback.received_at == pytest.approx(1.040)
    assert feedback.positions[LEFT_ARM[0]] == pytest.approx(2.0)
    assert feedback.vector(HEAD) == pytest.approx([2.5, 2.6])
    np.testing.assert_allclose(feedback.left_hand, [2.1] * 10)
    # The head arrives on its own view, so the waist and legs of the head/waist/leg
    # composite are not part of a grasp observation at all. Nothing here needs them,
    # and taking them would mean accepting that view's oldest-group stamp.
    assert "waist_yaw_joint" not in feedback.positions


def test_only_the_left_hand_is_taken_in_the_platform_order():
    history = FeedbackHistory()
    history.record("joint.state", 10 * MS, ARMS, [0.0] * 14)
    history.record("joint.head_state", 10 * MS, HEAD, [0.0, 0.0])
    reversed_hand = tuple(reversed(LEFT_HAND)) + RIGHT_HAND
    history.record("joint.hand_state", 10 * MS, reversed_hand, list(range(10)) + [7.0] * 10)

    np.testing.assert_allclose(history.latest().left_hand, list(reversed(range(10))))


def test_the_right_hand_is_taken_in_its_platform_order_when_the_view_carries_it():
    history = FeedbackHistory()
    history.record("joint.state", 10 * MS, ARMS, [0.0] * 14)
    history.record("joint.head_state", 10 * MS, HEAD, [0.0, 0.0])
    hand = LEFT_HAND + tuple(reversed(RIGHT_HAND))
    history.record("joint.hand_state", 10 * MS, hand, [7.0] * 10 + list(range(10)))
    feedback = history.latest()
    np.testing.assert_allclose(feedback.hand("right"), list(reversed(range(10))))
    np.testing.assert_allclose(feedback.hand("left"), [7.0] * 10)

    history = FeedbackHistory()
    history.record("joint.state", 10 * MS, ARMS, [0.0] * 14)
    history.record("joint.head_state", 10 * MS, HEAD, [0.0, 0.0])
    history.record("joint.hand_state", 10 * MS, LEFT_HAND, [0.0] * 10)
    feedback = history.latest()
    assert feedback.right_hand is None
    with pytest.raises(ValueError, match="lacks right hand"):
        feedback.hand("right")


def test_an_image_without_feedback_within_50_ms_is_refused():
    history = FeedbackHistory()
    _record(history, 1000 * MS)

    assert history.at(1049 * MS)
    with pytest.raises(ValueError, match="within 50 ms"):
        history.at(1051 * MS)


def test_every_view_is_required():
    history = FeedbackHistory()
    history.record("joint.state", 10 * MS, ARMS, [0.0] * 14)
    history.record("joint.head_state", 10 * MS, HEAD, [0.0, 0.0])

    with pytest.raises(ValueError, match=r"No feedback yet from \['joint.hand_state'\]"):
        history.latest()


@pytest.mark.parametrize("sides", [("left",), ("right",), ("left", "right"), ()])
def test_only_explicitly_installed_hands_are_required(sides):
    history = FeedbackHistory(required_hands=sides)
    history.record("joint.state", 10 * MS, ARMS, [0.0] * 14)
    history.record("joint.head_state", 10 * MS, HEAD, [0.0, 0.0])
    names = tuple(name for side in sides for name in (LEFT_HAND if side == "left" else RIGHT_HAND))
    if sides:
        history.record("joint.hand_state", 10 * MS, names, [0.1] * len(names))
    history.set_healthy(True, time.monotonic())
    for feedback in (history.latest(), history.at(10 * MS)):
        assert feedback.healthy and len(feedback.vector(ARMS)) == 14
        for side in ("left", "right"):
            if side in sides:
                np.testing.assert_allclose(feedback.hand(side), [0.1] * 10)
            else:
                with pytest.raises(ValueError, match=f"lacks {side} hand"):
                    feedback.hand(side)


@pytest.mark.parametrize("names", [(), RIGHT_HAND[:-1], LEFT_HAND + RIGHT_HAND])
def test_missing_partial_or_unexpected_hand_feedback_cannot_change_the_mounting_configuration(names):
    history = FeedbackHistory(required_hands=("right",))
    history.record("joint.state", 10 * MS, ARMS, [0.0] * 14)
    history.record("joint.head_state", 10 * MS, HEAD, [0.0, 0.0])
    history.record("joint.hand_state", 10 * MS, names, [0.0] * len(names))
    with pytest.raises(ValueError, match="hand"):
        history.latest()


def test_missing_joints_and_desynchronized_views_are_refused():
    history = FeedbackHistory()
    history.record("joint.state", 10 * MS, LEFT_ARM, [0.0] * 7)
    history.record("joint.head_state", 10 * MS, HEAD, [0.0, 0.0])
    history.record("joint.hand_state", 10 * MS, LEFT_HAND, [0.0] * 10)
    with pytest.raises(ValueError, match="lacks body joints"):
        history.latest()

    history = FeedbackHistory()
    history.record("joint.state", 10 * MS, ARMS, [0.0] * 14)
    history.record("joint.head_state", 10 * MS, HEAD, [0.0, 0.0])
    history.record("joint.hand_state", 200 * MS, LEFT_HAND, [0.0] * 10)
    with pytest.raises(ValueError, match="not synchronized"):
        history.latest()


def test_out_of_order_unknown_and_unstamped_samples_are_dropped():
    history = FeedbackHistory()
    assert history.record("joint.state", 20 * MS, ARMS, [0.0] * 14)
    assert not history.record("joint.state", 20 * MS, ARMS, [1.0] * 14)
    assert not history.record("joint.state", 10 * MS, ARMS, [1.0] * 14)
    assert not history.record("joint.state", 0, ARMS, [1.0] * 14)
    assert not history.record("joint.trajectory", 30 * MS, ARMS, [1.0] * 14)


def test_a_verdict_is_healthy_only_while_it_is_provably_fresh():
    history = FeedbackHistory()
    _record(history, 10 * MS)

    # No status yet: joints alone do not make a healthy runtime.
    assert history.latest().healthy is False

    history.set_healthy(True, time.monotonic())
    assert history.latest().healthy is True

    # A status the middleware did not time can never count as fresh.
    history.set_healthy(True, None)
    assert history.latest().healthy is False


def test_a_status_that_stopped_while_the_views_kept_updating_is_unhealthy():
    """The data plane keeps the views flowing after the runtime's process is gone.

    The joints assemble and stay usable, but the verdict that expired with the
    status must not read as a healthy runtime.
    """
    history = FeedbackHistory()
    _record(history, 10 * MS)
    history.set_healthy(True, time.monotonic() - MAX_STATUS_AGE_S - 1.0)
    _record(history, 40 * MS, offset=1.0)

    assert history.latest().healthy is False
    assert history.at(40 * MS).healthy is False
    assert history.latest().positions[LEFT_ARM[0]] == pytest.approx(1.0)


def test_an_old_active_taken_from_the_drain_is_already_expired():
    """on_demand keeps each sample's arrival time, so a backlog lies about nothing.

    An ACTIVE that waited in the reader queue is delivered with the time it
    arrived; by the time a reader drains it, the verdict is as old as its wait.
    """
    history = FeedbackHistory()
    _record(history, 10 * MS)

    history.set_healthy(True, time.monotonic() - MAX_STATUS_AGE_S - 1.0)

    assert history.latest().healthy is False


def test_an_unhealthy_verdict_is_unhealthy_even_while_it_is_fresh():
    history = FeedbackHistory()
    _record(history, 10 * MS)
    history.set_healthy(False, time.monotonic())
    assert history.latest().healthy is False


def test_the_joint_names_match_what_the_omnihand_runtime_declares():
    """The grasp pipeline's joint lists must be the runtime's, or feedback never assembles."""
    from pathlib import Path

    import yaml
    from aimdk_grasp.joints import BODY_JOINTS

    profile = Path(__file__).resolve().parents[2] / "aimdk_robot" / "profiles" / "x2_ultra_omnihand.yaml"
    declared = yaml.safe_load(profile.read_text(encoding="utf-8"))
    hand = declared["interfaces"]["joint.hand_state"]["joint_names"]
    # The head comes from its own view, not from the head/waist/leg composite:
    # that one is stamped with the oldest of its groups (see joint.head_state in
    # the profile), which would mislabel the head pose the camera is placed by.
    body = list(declared["joints"]) + declared["interfaces"]["joint.head_state"]["joint_names"]

    assert tuple(name for name in hand if name.startswith("L_")) == LEFT_HAND
    assert not [name for name in BODY_JOINTS if name not in body]
