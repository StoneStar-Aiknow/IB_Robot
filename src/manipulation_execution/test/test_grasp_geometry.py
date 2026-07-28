import math

import numpy as np

from manipulation_execution.grasp_geometry import (
    build_candidate_plan,
    canonicalize_joint5,
    contact_distance_score,
    euler_xyz_matrix,
    fixed_finger_base_side_alignment,
    fixed_finger_envelope_score,
    fixed_finger_robust_gap,
    grasp_axis_errors,
    joint5_closing_axis_correction,
    quaternion_from_matrix,
    quaternion_matrix,
    source_contact_camera,
    target_width_extent_points,
    xyz_within_workspace,
)


def _config():
    return {
        "source_contact_point": [0.0, 0.0, 0.195],
        "adapter": {"source_to_ee_rpy": [math.pi, 0.0, 0.0]},
        "approach_distance_m": 0.08,
        "lift_distance_m": 0.165,
        "target_gripper": {
            "fixed_finger_contact_ee": [-0.014, 0.0, -0.080],
            "closing_axis_ee": [1.0, 0.0, 0.0],
            "fixed_finger_margin_m": 0.006,
            "fixed_finger_margin_max_m": 0.012,
            "fixed_finger_margin_width_ref_m": 0.035,
            "fixed_finger_margin_width_gain": 0.25,
            "width_clearance_m": 0.003,
            "min_width_m": 0.008,
            "max_width_m": 0.080,
            "fallback_width_m": 0.035,
            "width_quality_min": 0.75,
        },
    }


def test_quaternion_matrix_round_trip():
    rotation = euler_xyz_matrix((0.3, -0.2, 1.1))
    quaternion = quaternion_from_matrix(rotation)
    assert np.allclose(quaternion_matrix(quaternion), rotation, atol=1e-8)


def test_grasp_axis_errors_keep_closing_axis_signed():
    target = quaternion_from_matrix(np.eye(3, dtype=np.float64))
    actual = quaternion_from_matrix(euler_xyz_matrix((0.0, 0.0, math.pi)))

    errors = grasp_axis_errors(target, actual, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))

    assert errors.approach_deg == 0.0
    assert math.isclose(errors.closing_deg, 180.0, abs_tol=1e-8)

    symmetric_errors = grasp_axis_errors(
        target,
        actual,
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        closing_axis_180_symmetric=True,
    )
    assert symmetric_errors.approach_deg == 0.0
    assert symmetric_errors.closing_deg == 0.0


def test_grasp_axis_errors_measure_approach_and_closing_independently():
    target = quaternion_from_matrix(np.eye(3, dtype=np.float64))
    actual = quaternion_from_matrix(euler_xyz_matrix((0.0, math.pi / 6.0, 0.0)))

    errors = grasp_axis_errors(target, actual, (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))

    assert math.isclose(errors.approach_deg, 30.0, abs_tol=1e-8)
    assert math.isclose(errors.closing_deg, 30.0, abs_tol=1e-8)


def test_joint5_helpers_select_the_equivalent_closing_axis_branch():
    target = quaternion_from_matrix(euler_xyz_matrix((0.0, 0.0, math.pi / 3.0)))
    actual = quaternion_from_matrix(np.eye(3, dtype=np.float64))

    correction = joint5_closing_axis_correction(
        target,
        actual,
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        closing_axis_180_symmetric=True,
    )

    assert math.isclose(correction, math.pi / 3.0, abs_tol=1e-8)
    assert math.isclose(canonicalize_joint5(2.0 * math.pi / 3.0), -math.pi / 3.0, abs_tol=1e-8)


def test_candidate_plan_preserves_contact_alignment():
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, 3] = [0.2, -0.1, 0.3]
    plan = build_candidate_plan(candidate.reshape(-1), np.eye(4), 0.035, 0.9, _config())
    assert np.allclose(plan.target_contact_base, (0.2, -0.1, 0.495), atol=1e-8)
    assert math.isclose(plan.lift[2] - plan.grasp[2], 0.165)
    assert math.isclose(sum(value * value for value in plan.quaternion), 1.0, rel_tol=1e-8)


def test_candidate_plan_preserves_target_extent_points():
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, 3] = [0.2, -0.1, 0.3]

    plan = build_candidate_plan(
        candidate.reshape(-1),
        np.eye(4),
        0.04,
        1.0,
        _config(),
        width_axis_camera=[1.0, 0.0, 0.0],
        target_width_min_offset_m=-0.02,
        target_width_max_offset_m=0.02,
    )

    assert plan.target_width_min_base is not None
    assert plan.target_width_max_base is not None
    assert np.allclose(plan.target_width_min_base, [0.18, -0.1, 0.3])
    assert np.allclose(plan.target_width_max_base, [0.22, -0.1, 0.3])


def test_target_width_extent_points_apply_camera_to_base_transform():
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, 3] = [0.2, -0.1, 0.3]
    base_to_camera = np.eye(4, dtype=np.float64)
    base_to_camera[:3, 3] = [0.4, 0.2, -0.1]

    points = target_width_extent_points(
        candidate.reshape(-1),
        base_to_camera,
        [0.0, 2.0, 0.0],
        -0.02,
        0.03,
        0.9,
        0.75,
    )

    assert points is not None
    assert np.allclose(points[0], [0.6, 0.08, 0.2])
    assert np.allclose(points[1], [0.6, 0.13, 0.2])


def test_target_width_extent_points_reject_unreliable_width():
    points = target_width_extent_points(
        np.eye(4, dtype=np.float64).reshape(-1),
        np.eye(4, dtype=np.float64),
        [1.0, 0.0, 0.0],
        -0.02,
        0.02,
        0.5,
        0.75,
    )

    assert points is None

    nonfinite_points = target_width_extent_points(
        np.eye(4, dtype=np.float64).reshape(-1),
        np.eye(4, dtype=np.float64),
        [float("nan"), 0.0, 0.0],
        -0.02,
        0.02,
        float("nan"),
        0.75,
    )

    assert nonfinite_points is None


def test_fixed_finger_envelope_prefers_target_behind_front_edge():
    common = {
        "ee_xyz": (0.0, 0.0, 0.0),
        "ee_quaternion": (0.0, 0.0, 0.0, 1.0),
        "fixed_finger_contact_ee": (-0.014, 0.0, -0.080),
        "closing_axis_ee": (1.0, 0.0, 0.0),
        "target_gap_m": 0.010,
        "gap_sigma_m": 0.006,
        "reliable_max_opening_m": 0.072,
        "moving_min_clearance_m": 0.003,
        "fixed_score_weight": 0.80,
    }
    preferred = fixed_finger_envelope_score(
        target_width_min_base=(-0.004, 0.0, -0.080),
        target_width_max_base=(0.009, 0.0, -0.080),
        **common,
    )
    front_contact = fixed_finger_envelope_score(
        target_width_min_base=(-0.014, 0.0, -0.080),
        target_width_max_base=(-0.001, 0.0, -0.080),
        **common,
    )

    assert preferred.fixed_gap_m == 0.010
    assert preferred.score == 1.0
    assert front_contact.fixed_gap_m == 0.0
    assert front_contact.score < preferred.score


def test_fixed_finger_base_side_alignment_rejects_outer_fixed_finger():
    common = {
        "ee_quaternion": (0.0, 0.0, 0.0, 1.0),
        "fixed_finger_contact_ee": (-0.014, 0.0, 0.0),
        "target_width_min_base": (0.19, 0.0, 0.0),
        "target_width_max_base": (0.21, 0.0, 0.0),
        "reference_point_base": (0.0, 0.0, 0.0),
    }
    inward = fixed_finger_base_side_alignment(ee_xyz=(0.194, 0.0, 0.0), **common)
    outward = fixed_finger_base_side_alignment(ee_xyz=(0.226, 0.0, 0.0), **common)

    assert inward.alignment_cos == 1.0
    assert inward.inward_offset_m > 0.0
    assert outward.alignment_cos == -1.0
    assert outward.inward_offset_m < 0.0


def test_fixed_finger_robust_gap_rejects_error_toward_fixed_finger():
    rejected = fixed_finger_robust_gap(
        0.012406,
        0.013099,
        (-0.0029, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        max_target_gap_deficit_m=0.003,
    )
    accepted = fixed_finger_robust_gap(
        0.0105,
        0.0124,
        (0.0057, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0),
        max_target_gap_deficit_m=0.003,
    )

    assert rejected.effective_gap_m < rejected.required_gap_m
    assert rejected.passed is False
    assert accepted.effective_gap_m > accepted.required_gap_m
    assert accepted.passed is True


def test_source_contact_camera_and_centroid_score():
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, 3] = [0.1, -0.2, 0.3]
    contact = source_contact_camera(candidate.reshape(-1), (0.0, 0.0, 0.195))
    assert np.allclose(contact, (0.1, -0.2, 0.495), atol=1e-8)

    distance, score = contact_distance_score(contact, (0.1, -0.2, 0.465), 0.06)
    assert math.isclose(distance, 0.03, abs_tol=1e-8)
    assert math.isclose(score, 0.5, abs_tol=1e-8)


def test_workspace_checks_radius():
    allowed, _ = xyz_within_workspace(
        (0.2, 0.1, 0.2),
        {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.0, 0.5], "max_radius_m": 0.4},
    )
    assert allowed
    allowed, reason = xyz_within_workspace(
        (0.4, 0.4, 0.2),
        {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.0, 0.5], "max_radius_m": 0.5},
    )
    assert not allowed
    assert "radius" in reason
