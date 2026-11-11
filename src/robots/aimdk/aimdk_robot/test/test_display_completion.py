"""Display opt-in is explicit; strict completion evidence remains available."""

from types import SimpleNamespace

import pytest
from aimdk_robot.display_completion import display_completion_policy, final_pose_evidence, final_pose_reached


def policy():
    return dict(motions=["self_intro"], mode="playback_only")


@pytest.mark.parametrize("hand_error,arm_error,expected", [(0.08, 0.06, True), (0.3, 0.0, False), (0.0, 0.15, False)])
def test_small_finger_error_does_not_hide_arm_or_large_finger_error(hand_error, arm_error, expected):
    evidence = final_pose_evidence(([arm_error] * 14, 11), ([hand_error] * 20, 11), [0] * 14, [0] * 20, 10, 12)
    assert final_pose_reached(evidence, 0.1, 0.2) is expected
    assert evidence["hand_max_error_rad"] == pytest.approx(hand_error)
    assert evidence["arm_worst_joint"]
    assert not final_pose_reached(evidence, 0.03, 0.03)


@pytest.mark.parametrize("arm_stamp,hand_stamp", [(9, 11), (11, 9), (11, 13)])
def test_cached_or_future_feedback_cannot_confirm_completion(arm_stamp, hand_stamp):
    evidence = final_pose_evidence(([0] * 14, arm_stamp), ([0] * 20, hand_stamp), [0] * 14, [0] * 20, 10, 12)
    assert not final_pose_reached(evidence, 0.1, 0.2)


def test_missing_feedback_preserves_which_view_is_missing():
    evidence = final_pose_evidence(None, ([0] * 20, 11), [0] * 14, [0] * 20, 10, 12)
    assert not final_pose_reached(evidence, 0.1, 0.2)
    assert evidence["arm_max_error_rad"] is None
    assert evidence["hand_age_s"] == 1


@pytest.mark.parametrize(
    "change",
    [{"motions": ["unknown"]}, {"motions": ["handshake"]}, {"mode": "ignore_errors"}, {"arm_tolerance_rad": 99}],
)
def test_bad_display_policy_cannot_disable_validation(change):
    config = policy()
    config.update(change)
    with pytest.raises(ValueError):
        display_completion_policy(
            config, {"self_intro": SimpleNamespace(hands=True), "handshake": SimpleNamespace(hands=False)}
        )


def test_unconfigured_deployments_keep_strict_completion():
    assert display_completion_policy(None, {}) is None


@pytest.mark.parametrize("name", ["wave", "blow_kiss", "self_intro"])
def test_display_opt_in_includes_arm_only_legacy_gestures(name):
    assert (
        display_completion_policy({"motions": [name], "mode": "playback_only"}, {name: SimpleNamespace(hands=False)})[
            "mode"
        ]
        == "playback_only"
    )


@pytest.mark.parametrize("partial,position,expected", [(False, -1.0, True), (True, -1.0, False), (False, 0.0, False)])
def test_goal_tolerance_opt_out_must_be_explicit_and_complete(partial, position, expected):
    from aimdk_robot.display_completion import goal_acceptance_disabled

    joints = ["left", "right"]
    values = [SimpleNamespace(name=name, position=position, velocity=-1.0, acceleration=-1.0) for name in joints]
    assert goal_acceptance_disabled(values[:1] if partial else values, joints) is expected
