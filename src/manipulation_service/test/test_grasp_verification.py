import threading
from types import SimpleNamespace
from typing import Any, cast

from sensor_msgs.msg import Image

from manipulation_service.grasp_verification import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_UNCERTAIN,
    DepthVisibilityStats,
    GraspVerificationInput,
    GraspVerificationWeights,
    evaluate_grasp,
)
from manipulation_service.grasp_verifier_node import GraspVerifierNode


def _input(**overrides):
    data = {
        "gripper_position": 0.12,
        "gripper_closed_position": 0.0,
        "gripper_open_position": 1.0,
        "gripper_fully_open_tolerance": 0.05,
        "gripper_contact_min_opening": 0.08,
        "gripper_no_contact_max_opening": 0.03,
        "gripper_joint": "6",
        "gripper_current_abs_a": 0.12,
        "current_contact_threshold_a": 0.08,
        "wrist_depth": DepthVisibilityStats(
            valid_fraction=0.7,
            near_fraction=0.1,
            median_depth_m=0.35,
            occluded=False,
        ),
        "expected_target_width_m": 0.035,
    }
    data.update(overrides)
    return GraspVerificationInput(**data)


def test_success_when_gripper_stops_open_and_current_rises():
    result = evaluate_grasp(_input())

    assert result.status == STATUS_SUCCESS
    assert result.success is True
    assert result.confidence >= 0.8
    assert any("gripper_contact" in item for item in result.evidence)
    assert any("current_contact" in item for item in result.evidence)


def test_success_for_pc_compatible_so101_marker_contact():
    result = evaluate_grasp(
        _input(
            gripper_position=0.12,
            gripper_current_abs_a=0.702,
            wrist_depth=DepthVisibilityStats(
                valid_fraction=0.1,
                near_fraction=0.9,
                median_depth_m=0.08,
                occluded=True,
            ),
        )
    )

    assert result.status == STATUS_SUCCESS
    assert result.success is True
    assert result.confidence == 1.0
    assert "gripper_contact: stopped before fully closing" in result.evidence


def test_failed_when_gripper_fully_closes_and_current_is_low():
    result = evaluate_grasp(_input(gripper_position=0.005, gripper_current_abs_a=0.01))

    assert result.status == STATUS_FAILED
    assert result.success is False
    assert result.confidence >= 0.6


def test_failed_when_closed_gripper_has_no_contact_evidence():
    result = evaluate_grasp(_input(gripper_position=0.0031, gripper_current_abs_a=0.0))

    assert result.status == STATUS_FAILED
    assert result.success is False
    assert "score=0.65" in result.message
    assert "gripper_contact: fully closed or near closed" in result.evidence
    assert "current_contact: below contact threshold" in result.evidence


def test_open_jaw_after_close_goal_is_never_contact_evidence():
    """Gazebo acceptance fix 3 (e2e/gazebo-155516): the close goal returned
    "success" with the jaw still at the fully-open end (1.0001) and this
    evidence verified the pick at 0.70 — a jaw that never left open holds
    nothing."""
    weights = GraspVerificationWeights(
        gripper_contact_success=0.70,
        current_contact_success=0.0,
        current_contact_failure=0.0,
    )

    result = evaluate_grasp(_input(gripper_position=1.0001, gripper_current_abs_a=None), weights=weights)

    assert result.status == STATUS_UNCERTAIN
    assert result.success is False
    assert "gripper_contact: stopped before fully closing" not in result.evidence
    assert "gripper_contact: jaw at/near fully open after close goal" in result.evidence


def test_open_jaw_with_occluded_wrist_still_does_not_verify():
    # Default weights: the legacy path scored contact 0.55 + occlusion 0.10
    # = 0.65 = the success threshold — occlusion must not rescue a jaw that
    # never left open.
    result = evaluate_grasp(
        _input(
            gripper_position=1.0,
            gripper_current_abs_a=None,
            wrist_depth=DepthVisibilityStats(
                valid_fraction=0.1,
                near_fraction=0.9,
                median_depth_m=0.08,
                occluded=True,
            ),
        )
    )

    assert result.status == STATUS_UNCERTAIN
    assert result.success is False
    assert "gripper_contact: jaw at/near fully open after close goal" in result.evidence


def test_true_contact_unchanged_with_open_reference():
    # A jaw that actually moved from open and stopped early (the run-1
    # blocked close: joint 0.2539) is contact evidence exactly as before.
    weights = GraspVerificationWeights(
        gripper_contact_success=0.70,
        current_contact_success=0.0,
        current_contact_failure=0.0,
    )

    result = evaluate_grasp(_input(gripper_position=0.2539, gripper_current_abs_a=None), weights=weights)

    assert result.status == STATUS_SUCCESS
    assert result.success is True
    assert "gripper_contact: stopped before fully closing" in result.evidence
    assert "gripper_contact: jaw at/near fully open after close goal" not in result.evidence


def test_jaw_past_the_open_tolerance_counts_as_moved():
    # Outside the fully-open band the jaw actually moved from open, so the
    # legacy stopped-before-closing branch applies unchanged.
    result = evaluate_grasp(_input(gripper_position=0.90, gripper_current_abs_a=None))

    assert "gripper_contact: stopped before fully closing" in result.evidence
    assert "gripper_contact: jaw at/near fully open after close goal" not in result.evidence


def test_open_position_unset_keeps_legacy_behaviour():
    # An unconfigured caller (gripper_open_position None) keeps the legacy
    # classification for a jaw that genuinely stopped mid-travel.
    result = evaluate_grasp(_input(gripper_position=0.12, gripper_open_position=None))

    assert result.status == STATUS_SUCCESS
    assert result.success is True
    assert "gripper_contact: stopped before fully closing" in result.evidence


def test_build_input_carries_the_open_position_parameters():
    values = {
        "max_sample_age_s": 2.0,
        "gripper_joint": "6",
        "gripper_closed_position": 0.0,
        "gripper_open_position": 1.0,
        "gripper_contact_min_opening": 0.08,
        "gripper_no_contact_max_opening": 0.03,
        "gripper_fully_open_tolerance": 0.05,
        "current_contact_threshold_a": 0.08,
    }
    harness = SimpleNamespace(
        _lock=threading.Lock(),
        _latest_joint_state=None,
        _latest_joint_current=None,
        _latest_wrist_depth=None,
        _now_ns=lambda: 100,
        _fresh_value=GraspVerifierNode._fresh_value,
        _extract_named_value=GraspVerifierNode._extract_named_value,
        _extract_abs_current=GraspVerifierNode._extract_abs_current,
        get_parameter=lambda name: SimpleNamespace(value=values[name]),
    )

    input_data, note = GraspVerifierNode._build_input(cast(Any, harness), expected_target_width_m=0.0)

    assert note == ""
    assert input_data.gripper_open_position == 1.0
    assert input_data.gripper_fully_open_tolerance == 0.05


def test_wrist_occlusion_alone_does_not_make_grasp_fail():
    result = evaluate_grasp(
        _input(
            gripper_position=None,
            gripper_current_abs_a=None,
            wrist_depth=DepthVisibilityStats(
                valid_fraction=0.1,
                near_fraction=0.9,
                median_depth_m=0.08,
                occluded=True,
            ),
        )
    )

    assert result.status == STATUS_UNCERTAIN
    assert result.success is False
    assert any("wrist_occlusion" in item for item in result.evidence)


def test_custom_weights_can_raise_success_threshold():
    weights = GraspVerificationWeights(success_threshold=0.95)
    result = evaluate_grasp(_input(), weights=weights)

    assert result.status == STATUS_UNCERTAIN
    assert result.success is False


def test_no_current_profile_accepts_explicit_gripper_contact_without_occlusion():
    weights = GraspVerificationWeights(
        gripper_contact_success=0.70,
        current_contact_success=0.0,
        current_contact_failure=0.0,
        success_threshold=0.65,
    )

    result = evaluate_grasp(_input(gripper_current_abs_a=None), weights=weights)

    assert result.status == STATUS_SUCCESS
    assert result.success is True
    assert result.confidence == 0.70


def test_wrist_depth_callback_defers_frame_processing():
    harness = SimpleNamespace(
        _lock=threading.Lock(),
        _latest_wrist_depth=None,
        _now_ns=lambda: 123,
    )
    msg = Image()

    GraspVerifierNode._wrist_depth_cb(cast(Any, harness), msg)

    assert harness._latest_wrist_depth.received_ns == 123
    assert harness._latest_wrist_depth.value is msg
