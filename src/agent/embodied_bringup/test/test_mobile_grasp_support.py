"""The mobile-grasp E2E support unit tests (lekiwi-nav-grasp revision 2, R8).

These tests cover the support module's pure pieces: the planning-scene
construction (the chassis box, its full transform into the arm base frame),
the shared pose/quaternion math, the motion-sample validity rule with its
independent endpoint coverage (review r2 #5), the ground-truth judge (with
its review r1 #1/#6 regressions and the r2 #6/#7 corrections: the base
tilt invariant, the layout-``lift_m`` pre-lift boundary, the
continuous/dense rest window), the continuous invariants (the timestamped
runtime-status coverage/freshness rule and the steady-wall clock watchdog,
r2 #4) and the pick-feedback candidate census (r1 item 15). Synthetic
streams and URDFs only — no simulator, no ROS graph.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from test.mobile_grasp_support import (
    JudgeInput,
    SceneBox,
    chassis_box,
    clock_stall_verdict,
    collision_object_message,
    first_lift_stamp,
    frozen_rest_height,
    judge_hold,
    matrix_from_xyz_rpy,
    motion_audit_window,
    motion_sample_verdict,
    pick_feedback_candidate_rounds,
    point_in_frame,
    quat_in_frame,
    quat_xyzw_to_matrix,
    runtime_status_verdict,
    transform_matrix,
    transform_scene_box,
)

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# The shared pose/quaternion math
# ---------------------------------------------------------------------------


def test_the_shared_pose_helpers_agree_with_each_other():
    yaw90 = matrix_from_xyz_rpy([0.1, -0.2, 0.3], [0.0, 0.0, math.pi / 2])
    assert yaw90[:3, :3] == pytest.approx(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
    half = math.sin(math.pi / 4)
    tf_message = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.1, y=-0.2, z=0.3),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=half, w=math.cos(math.pi / 4)),
        )
    )
    assert transform_matrix(tf_message) == pytest.approx(yaw90)
    assert quat_xyzw_to_matrix((0.0, 0.0, half, math.cos(math.pi / 4))) == pytest.approx(yaw90[:3, :3])
    # a 90-deg-yaw frame at (0.1, -0.2, 0.3): the world point (0.1, 0.8, 0.3) is the
    # frame-relative (0, 1, 0) offset, which the yaw maps to (1.0, 0.0, 0.0)
    assert point_in_frame((0.1, -0.2, 0.3), (0.0, 0.0, half, math.cos(math.pi / 4)), (0.1, 0.8, 0.3)) == pytest.approx(
        (1.0, 0.0, 0.0)
    )
    assert quat_in_frame(IDENTITY_QUAT, IDENTITY_QUAT) == pytest.approx(IDENTITY_QUAT)


# ---------------------------------------------------------------------------
# The planning-scene construction (R6)
# ---------------------------------------------------------------------------


def test_chassis_box_cuts_at_the_arm_mount_underside():
    vertices = np.array([[x, y, z] for x in (-0.15, 0.15) for y in (-0.15, 0.15) for z in (0.0, 0.12)], dtype=float)
    box = chassis_box(vertices, arm_mount_underside_z=0.06)
    assert box.dims[2] == pytest.approx(0.06)  # cut at the arm mount underside
    assert box.centre[2] == pytest.approx(0.03)
    assert box.dims[0] == pytest.approx(0.3)


def test_transform_scene_box_carries_a_nonzero_rotation_and_translation():
    """Review r1 item 4: a centre-only translation would silently drop the
    mount rotation. The full corner transform re-AABBs the rotated box."""
    mount = np.array(
        [
            [math.cos(math.pi / 2), -math.sin(math.pi / 2), 0.0, 0.06],
            [math.sin(math.pi / 2), math.cos(math.pi / 2), 0.0, -0.02],
            [0.0, 0.0, 1.0, 0.02],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    box = SceneBox("chassis", (0.0, 0.0, 0.0), (0.2, 0.1, 0.05), frame="base_link")
    transformed = transform_scene_box(box, mount, "base")
    assert transformed.frame == "base"
    assert transformed.centre == pytest.approx((0.06, -0.02, 0.02))  # the full translation
    assert transformed.dims == pytest.approx((0.1, 0.2, 0.05))  # the yaw's x/y swap, applied
    # a pure translation leaves the dims unchanged
    translated = transform_scene_box(
        box, np.array([[1.0, 0, 0, 0.1], [0, 1.0, 0, 0.2], [0, 0, 1.0, 0.3], [0, 0, 0, 1.0]]), "map"
    )
    assert translated.centre == pytest.approx((0.1, 0.2, 0.3))
    assert translated.dims == pytest.approx((0.2, 0.1, 0.05))


# ---------------------------------------------------------------------------
# Motion-sample validity (R6)
# ---------------------------------------------------------------------------


def test_motion_sample_verdict_rate_gap_and_invalid():
    ok, reason = motion_sample_verdict([0.0, 0.05, 0.10, 0.15], [True, True, True, True])
    assert ok, reason
    ok, reason = motion_sample_verdict(list(np.arange(0, 0.9, 0.09)), [True] * 10)
    assert not ok and "rate" in reason  # 0.09 s gaps (below the limit) but only ~11 Hz overall
    ok, reason = motion_sample_verdict([0.0, 0.05, 0.3, 0.35], [True, True, True, True])
    assert not ok and "gap" in reason
    ok, reason = motion_sample_verdict([0.0, 0.05, 0.10], [True, False, True])
    assert not ok and "invalid" in reason
    ok, reason = motion_sample_verdict([], [])
    assert not ok and "no samples" in reason


def test_motion_sample_verdict_rejects_a_singleton_window():
    """Review r2 #5: a one-sample audit used to pass vacuously (span zero
    skipped the rate rule) — an executed motion must leave a stream, not a
    point."""
    ok, reason = motion_sample_verdict([1.0], [True])
    assert not ok and "insufficient" in reason


def test_motion_sample_verdict_requires_strictly_increasing_stamps():
    ok, reason = motion_sample_verdict([0.0, 0.05, 5.0, 5.05], [True] * 4)
    assert not ok and "gap" in reason  # head and tail only: the middle is missing
    ok, reason = motion_sample_verdict([0.0, 0.05, 0.03, 0.08], [True] * 4)
    assert not ok and "strictly increase" in reason  # backwards stamps
    ok, reason = motion_sample_verdict([0.0, 0.05, 0.05, 0.10], [True] * 4)
    assert not ok and "strictly increase" in reason  # a repeated stamp


def _audit_samples(motion_samples: int = 50) -> list[tuple[float, list[float]]]:
    static = [(float(t), [0.1, 0.2, 0.3]) for t in np.arange(0.0, 10.0, 0.05)]
    motion = [(10.0 + 0.05 * i, [0.1 + 0.01 * (i + 1), 0.2, 0.3]) for i in range(motion_samples)]
    return static + motion


def test_the_audit_window_retains_the_sample_just_before_the_onset():
    """Review r2 #5: the window includes the LAST AT-REST sample before the
    first arm motion — the onset transition itself is part of the audit, so
    a buffer that starts mid-motion cannot masquerade as the onset."""
    samples = _audit_samples()
    window, reason = motion_audit_window(samples, start_s=0.0, end_s=samples[-1][0])
    assert reason == ""
    assert len(window) == 51  # 50 moved samples + the last at-rest one
    assert window[0][1] == [0.1, 0.2, 0.3]  # the at-rest sample just before the onset
    assert window[1][1][0] == pytest.approx(0.11)  # the first moved sample


def test_a_lost_head_fails_the_audit_window():
    """Review r2 #5: losing the initial observation makes a later delta
    look like the first motion — the recording's own start marker (an
    independent execution marker) must anchor the buffer's head."""
    samples = _audit_samples()[40:]  # the recording starts 2 s late: no pre-motion rest
    window, reason = motion_audit_window(samples, start_s=0.0, end_s=_audit_samples()[-1][0])
    assert window == []
    assert "head" in reason


def test_a_lost_tail_fails_the_audit_window():
    """Review r2 #5: losing the final seconds leaves a short contiguous
    prefix that used to pass — the recording's end marker anchors the tail."""
    full = _audit_samples()
    samples = full[:-20]  # the last second of the recording is lost
    window, reason = motion_audit_window(samples, start_s=0.0, end_s=full[-1][0])
    assert window == []
    assert "tail" in reason


def test_no_arm_motion_is_explicit_no_motion_evidence():
    """Review r2 #2/#5: an arm that never moved is NOT a coverage failure —
    the empty window with a covered buffer is the explicit no-motion
    evidence a failed pick needs (and an empty buffer is a missing
    recording, never a vacuous pass)."""
    static = [(float(t), [0.1, 0.2, 0.3]) for t in np.arange(0, 5.0, 0.05)]
    window, reason = motion_audit_window(static, start_s=0.0, end_s=4.95)
    assert window == [] and reason == "no arm motion over the covered window"
    # a sub-epsilon jitter at the rest pose is not arm motion either
    jitter = [(float(t), [0.1, 0.2, 0.3]) for t in range(10)] + [(10.0, [0.1005, 0.2, 0.3])]
    window, reason = motion_audit_window(jitter, start_s=0.0, end_s=10.0)
    assert window == [] and reason == "no arm motion over the covered window"
    # the recording itself is missing
    window, reason = motion_audit_window([], start_s=0.0, end_s=1.0)
    assert window == [] and "no arm samples" in reason


# ---------------------------------------------------------------------------
# The judge (R7)
# ---------------------------------------------------------------------------


def _judge_input(**overrides) -> JudgeInput:
    base = {
        "z_rest": 0.095,
        "hold_seconds": 3.0,
        "lift_m": 0.05,
        "max_object_rotation_in_gripper_deg": 10.0,
        "object_samples": [(t, (1.0, 0.4, 0.145), IDENTITY_QUAT) for t in np.arange(0, 3.2, 0.1)],
        "object_in_gripper_samples": [(t, (0.01, 0.0, -0.02), IDENTITY_QUAT) for t in np.arange(0, 3.2, 0.1)],
        "hold_point_in_gripper": (0.0, 0.0, 0.0),
        "object_table_region": ((0.9, 1.4), (0.1, 0.7)),
        "base_planar_speeds": [(t, 0.001) for t in np.arange(0, 3.2, 0.5)],
        "base_roll_pitch_deg": [(0.0, 0.0), (0.2, -0.1)],
    }
    base.update(overrides)
    return JudgeInput(**base)


def test_the_judge_accepts_a_held_lifted_object():
    ok, reason = judge_hold(_judge_input())
    assert ok, reason


def test_the_judge_rejects_the_negative_cases():
    # lifted but resting on the wrist: outside the grasp region
    ok, reason = judge_hold(
        _judge_input(object_in_gripper_samples=[(t, (0.09, 0.0, 0.0), IDENTITY_QUAT) for t in np.arange(0, 3.2, 0.1)])
    )
    assert not ok and "grasp region" in reason
    # lifted then slipping: motion within the gripper frame
    ok, reason = judge_hold(
        _judge_input(
            object_in_gripper_samples=[
                (t, (0.005 + 0.004 * t, 0.0, -0.01), IDENTITY_QUAT) for t in np.arange(0, 3.2, 0.1)
            ]
        )
    )
    assert not ok and "slipping" in reason
    # a gap in the samples splitting off the trailing window
    samples = [(t, (0.01, 0.0, -0.02), IDENTITY_QUAT) for t in [0.0, 0.1, 0.2, 0.6, 0.7, 3.0]]
    ok, reason = judge_hold(_judge_input(object_in_gripper_samples=samples))
    assert not ok and "gap" in reason
    # knocked off the table before the lift
    ok, reason = judge_hold(_judge_input(object_samples=[(0.5, (0.2, 0.4, 0.02), IDENTITY_QUAT)] * 30))
    assert not ok and "table region" in reason
    # the base moved during the attempt
    ok, reason = judge_hold(_judge_input(base_planar_speeds=[(1.0, 0.05)]))
    assert not ok and "base moved" in reason
    # not lifted high enough
    ok, reason = judge_hold(
        _judge_input(object_samples=[(t, (1.0, 0.4, 0.12), IDENTITY_QUAT) for t in np.arange(0, 3.2, 0.1)])
    )
    assert not ok and "lift" in reason


def test_the_judge_records_base_tilt_without_a_verdict():
    """Maintainer decision 2026-11-03 (review r2 #6 follow-up): base
    roll/pitch is RECORD ONLY — the judge applies no tilt bound (the spec
    judges the base by planar speed alone); the wrapper records the max
    over the hold window into metrics. A heavily tilted base therefore no
    longer fails the hold by itself, and the base-stillness rule keeps its
    bite."""
    tilted = _judge_input(base_roll_pitch_deg=[(0.0, 0.0), (2.0, 9.5)])
    ok, reason = judge_hold(tilted)
    assert ok, reason  # 9.5 deg of tilt is recorded, not judged
    ok, reason = judge_hold(_judge_input(base_planar_speeds=[(1.0, 0.05)]))
    assert not ok and "base moved" in reason  # planar speed still fails


def test_the_judge_holds_on_the_trailing_window_past_a_mid_span_jitter_gap():
    """The design's hold is 'after the result, hold_s of sim time': a
    stream-jitter hole in the transport phase splits the carried history,
    and the TRAILING contiguous run is what must cover hold_s — with the
    0.1 s gap rule unchanged inside it."""
    carried = [(t, (0.01, 0.0, -0.02), IDENTITY_QUAT) for t in np.arange(0, 2.0, 0.1)]
    carried += [(t, (0.01, 0.0, -0.02), IDENTITY_QUAT) for t in np.arange(2.5, 6.0, 0.1)]  # the 0.5 s hole
    world = [(t, (1.0, 0.4, 0.145), IDENTITY_QUAT) for t in np.arange(2.4, 6.1, 0.1)]
    ok, reason = judge_hold(
        _judge_input(object_in_gripper_samples=carried, object_samples=world, base_planar_speeds=[])
    )
    assert ok, reason
    # but a hole INSIDE the trailing window still fails the gap rule
    carried = [(t, (0.01, 0.0, -0.02), IDENTITY_QUAT) for t in [3.0, 3.1, 3.2, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.1, 4.2]]
    ok, reason = judge_hold(
        _judge_input(
            object_in_gripper_samples=carried,
            object_samples=[(t, (1.0, 0.4, 0.145), IDENTITY_QUAT) for t in np.arange(2.9, 4.3, 0.1)],
            base_planar_speeds=[],
        )
    )
    assert not ok and "gap" in reason


# ---------------------------------------------------------------------------
# Review r2 #7: the frozen rest window, the first lift at the layout's lift_m
# ---------------------------------------------------------------------------


def test_frozen_rest_height_is_the_median_of_a_continuous_dense_window():
    samples = [(0.05 * i, 0.145 + (0.001 if i % 2 else -0.001)) for i in range(40)]  # 20 Hz, 2 s
    assert frozen_rest_height(samples, window_s=1.0) == pytest.approx(0.145)
    # an empty stream freezes nothing
    assert frozen_rest_height([]) is None


def test_frozen_rest_height_rejects_a_sparse_or_gapped_window():
    """Review r2 #7: two samples a second apart are not a second of fresh
    pre-motion data — a sparse or gapped trailing window freezes nothing
    (the caller keeps waiting and fails explicitly)."""
    sparse = [(0.0, 0.145), (1.0, 0.146)]
    assert frozen_rest_height(sparse) is None
    # dense enough but with a 0.5 s hole inside the trailing window
    gapped = [(0.05 * i, 0.145) for i in range(20, 29)] + [(1.9, 0.145)]
    assert len([z for t, z in gapped if 0.9 <= t <= 1.9]) >= 10  # the window is not sparse...
    assert frozen_rest_height(gapped) is None  # ...but it is not continuous


def test_frozen_rest_height_ignores_a_later_drop():
    """The freeze happens BEFORE any arm motion: a later drop below the rest
    height can never lower it (the old whole-run minimum could)."""
    samples = [(0.05 * i, 0.145) for i in range(10)]
    frozen = frozen_rest_height(samples)
    assert frozen == pytest.approx(0.145)
    dipped = samples + [(0.5 + 0.05 * i, 0.08) for i in range(20)]
    assert frozen_rest_height(dipped, window_s=1.0) == pytest.approx(0.08)  # the trailing window dips
    assert frozen == pytest.approx(0.145)  # but the frozen value never moved


def test_first_lift_stamp_closes_the_pre_lift_window_at_the_layout_lift():
    """Review r2 #7: the pre-lift window closes at the LAYOUT's lift_m —
    a bounce below the required lift height never closes it."""
    samples = [
        (0.0, (1.0, 0.4, 0.145), IDENTITY_QUAT),
        (0.1, (1.0, 0.4, 0.165), IDENTITY_QUAT),  # a 0.020 m bounce < lift_m (0.05)
        (0.2, (1.0, 0.4, 0.30), IDENTITY_QUAT),
        (0.3, (1.0, 0.4, 0.28), IDENTITY_QUAT),
    ]
    assert first_lift_stamp(samples, z_rest=0.145, detect_m=0.05) == pytest.approx(0.2)  # not the bounce
    assert first_lift_stamp(samples[:2], z_rest=0.145, detect_m=0.05) is None  # the bounce alone never lifts
    assert first_lift_stamp(samples, z_rest=0.30, detect_m=0.05) is None  # never rises off that surface


def test_a_bounce_below_the_required_lift_keeps_the_knocked_off_monitoring():
    """Review r2 #7 regression: with the pre-lift boundary at a hard-coded
    0.01 m, a small bounce disabled the knocked-off monitoring before the
    required lift — the object could be knocked off the table behind it."""
    rest = [(t, (1.0, 0.4, 0.145), IDENTITY_QUAT) for t in np.arange(0, 1.0, 0.1)]
    bounce = [(t, (1.0, 0.4, 0.165), IDENTITY_QUAT) for t in np.arange(1.0, 1.4, 0.1)]  # +0.020 < lift_m
    knocked = [(t, (0.6, 0.4, 0.165), IDENTITY_QUAT) for t in np.arange(1.4, 2.0, 0.1)]
    carried = [(t, (0.01, 0.0, -0.02), IDENTITY_QUAT) for t in np.arange(2.0, 5.2, 0.1)]
    held = [(t, (0.6, 0.4, 0.30), IDENTITY_QUAT) for t in np.arange(2.0, 5.2, 0.1)]
    ok, reason = judge_hold(
        _judge_input(
            z_rest=0.145,
            object_samples=rest + bounce + knocked + held,
            object_in_gripper_samples=carried,
            base_planar_speeds=[],
        )
    )
    assert not ok and "table region" in reason  # the bounce never closed the pre-lift window


def test_the_judge_accepts_a_legal_transport_off_the_table():
    """Review r1 item 7 regression: the production pick TRANSPORTS the
    object out of the table footprint — the table-region (knocked-off)
    rule applies only BEFORE the first lift; the old whole-stream check
    failed every legal carry that left the region."""
    rest = [(t, (1.0, 0.4, 0.145), IDENTITY_QUAT) for t in np.arange(0, 1.0, 0.1)]
    rise = [(t, (0.9, 0.4, 0.22), IDENTITY_QUAT) for t in np.arange(1.1, 1.4, 0.1)]
    carried_out = [(t, (0.6, 0.4, 0.30), IDENTITY_QUAT) for t in np.arange(1.5, 5.0, 0.1)]
    carried = [(t, (0.01, 0.0, -0.02), IDENTITY_QUAT) for t in np.arange(1.5, 5.0, 0.1)]
    ok, reason = judge_hold(
        _judge_input(
            z_rest=0.145,
            object_samples=rest + rise + carried_out,
            object_in_gripper_samples=carried,
            base_planar_speeds=[],
        )
    )
    assert ok, reason  # the carry out of the region is legitimate after the first lift
    # the same transport with the object knocked off BEFORE the lift still fails
    knocked = [(t, (0.6, 0.4, 0.145), IDENTITY_QUAT) for t in np.arange(0.4, 1.0, 0.1)]
    ok, reason = judge_hold(
        _judge_input(
            z_rest=0.145,
            object_samples=rest + knocked + rise + carried_out,
            object_in_gripper_samples=carried,
            base_planar_speeds=[],
        )
    )
    assert not ok and "table region" in reason


def test_an_object_dropped_below_the_rest_height_lowers_nothing():
    """Review r1 item 7 regression: the rest height is FROZEN before any arm
    motion, so a dip below it during the run cannot lower the bar the lift
    rule measures against (the old whole-run minimum let a 0.045 m "lift"
    pass against its own dip)."""
    rest = [(t, (1.0, 0.4, 0.145), IDENTITY_QUAT) for t in np.arange(0, 1.0, 0.1)]
    dip = [(t, (1.0, 0.4, 0.08), IDENTITY_QUAT) for t in np.arange(1.0, 1.3, 0.1)]
    held = [(t, (1.0, 0.4, 0.19), IDENTITY_QUAT) for t in np.arange(1.6, 4.8, 0.1)]  # 0.045 above REST
    carried = [(t, (0.01, 0.0, -0.02), IDENTITY_QUAT) for t in np.arange(1.6, 4.8, 0.1)]
    ok, reason = judge_hold(
        _judge_input(
            z_rest=0.145,
            object_samples=rest + dip + held,
            object_in_gripper_samples=carried,
            base_planar_speeds=[],
        )
    )
    assert not ok and "lift" in reason  # measured against the FROZEN 0.145, not the 0.08 dip


# ---------------------------------------------------------------------------
# Review r1 #1: the collision-object messages carry each box's OWN frame
# ---------------------------------------------------------------------------


def test_the_collision_object_message_keeps_every_boxs_own_frame():
    from moveit_msgs.msg import CollisionObject

    chassis_in_base = SceneBox("chassis", (0.0225, 0.06, -0.0235), (0.2241, 0.2158, 0.007), frame="base")
    message = collision_object_message(chassis_in_base, CollisionObject.ADD)
    assert message.header.frame_id == "base"
    assert message.id == "chassis"
    assert (message.primitive_poses[0].position.x, message.primitive_poses[0].position.y) == (
        pytest.approx(0.0225),
        pytest.approx(0.06),
    )
    assert message.primitives[0].dimensions[0] == pytest.approx(0.2241)
    # a box in any other frame keeps that frame through the message
    baselink_box = SceneBox("chassis", (0.12, -0.14, -0.033), (0.03, 0.03, 0.06), frame="base_link")
    message = collision_object_message(baselink_box, CollisionObject.ADD)
    assert message.header.frame_id == "base_link"
    assert message.primitive_poses[0].orientation.w == 1.0
    # a REMOVE needs only the id + frame
    removal = collision_object_message(baselink_box, CollisionObject.REMOVE)
    assert removal.id == "chassis" and removal.operation == CollisionObject.REMOVE
    assert not removal.primitives


# ---------------------------------------------------------------------------
# Review r1 #6: the judge's orientation math
# ---------------------------------------------------------------------------


def _quat_of_yaw_deg(degrees):
    half = math.radians(degrees) / 2.0
    return (0.0, 0.0, math.sin(half), math.cos(half))


def test_the_judge_accepts_a_rigidly_held_tilted_cube_and_rejects_jaw_rotation():
    """The rigidity rule (Revision-2 corrections): a cube whose orientation
    in the GRIPPER frame is constant passes at ANY world tilt; an object
    rotating in the jaws (> 10 deg in the gripper frame) fails even though
    its world position never drifts."""
    tilt114 = (math.sin(math.radians(114) / 2), 0.0, 0.0, math.cos(math.radians(114) / 2))
    carried = [(t, (0.004, 0.0, -0.096), tilt114) for t in np.arange(0, 3.2, 0.1)]
    world = [(t, (1.0, 0.4, 0.145), tilt114) for t in np.arange(0, 3.2, 0.1)]
    ok, reason = judge_hold(
        _judge_input(
            object_in_gripper_samples=carried,
            object_samples=world,
            hold_point_in_gripper=(0.004, 0.0, -0.096),
            base_planar_speeds=[],
        )
    )
    assert ok, reason
    # the same hold with the object rotating 20 deg in the jaws: fails
    rotating = [(t, (0.004, 0.0, -0.096), _quat_of_yaw_deg(20.0 * t / 3.1)) for t in np.arange(0, 3.2, 0.1)]
    ok, reason = judge_hold(
        _judge_input(
            object_in_gripper_samples=rotating,
            object_samples=world,
            hold_point_in_gripper=(0.004, 0.0, -0.096),
            base_planar_speeds=[],
        )
    )
    assert not ok and "rotated" in reason


# ---------------------------------------------------------------------------
# Review r2 #4: the continuous invariants
# ---------------------------------------------------------------------------


def test_runtime_status_verdict_requires_timestamped_coverage_and_freshness():
    ok, reason = runtime_status_verdict([(float(t), "ACTIVE", False) for t in np.arange(0, 30, 1.0)])
    assert ok, reason
    # the old vacuous pass on an empty stream ("0 status samples, all ACTIVE") is gone
    ok, reason = runtime_status_verdict([])
    assert not ok and "no runtime-status samples" in reason
    # a coverage hole: the status stream went silent mid-window
    samples = [(float(t), "ACTIVE", False) for t in [0.0, 1.0, 2.0, 6.0, 7.0]]
    ok, reason = runtime_status_verdict(samples)
    assert not ok and "coverage gap" in reason
    # stale at the end: the last sample is far behind the window's end
    samples = [(float(t), "ACTIVE", False) for t in np.arange(0, 10, 1.0)]
    ok, reason = runtime_status_verdict(samples, end_sim_s=20.0)
    assert not ok and "stale" in reason
    # the non-ACTIVE and latch rejections (r1 item 8) keep their bite
    ok, reason = runtime_status_verdict([(0.0, "ACTIVE", False), (1.0, "DEGRADED", False)])
    assert not ok and "ACTIVE" in reason
    ok, reason = runtime_status_verdict([(0.0, "ACTIVE", False), (1.0, "ACTIVE", True)])
    assert not ok and "latch" in reason


def test_clock_stall_verdict_is_a_steady_wall_watchdog():
    advancing = [(w, w) for w in np.arange(0, 10, 0.1)]  # RTF 1
    ok, reason = clock_stall_verdict(advancing)
    assert ok, reason
    slow = [(w, 0.3 * w) for w in np.arange(0, 10, 0.1)]  # RTF 0.3, still advancing
    ok, reason = clock_stall_verdict(slow)
    assert ok, reason
    # five seconds of repeated sim time at 10 Hz: the old adjacent-interval check passed it
    repeated = [(i / 10, 20.0) for i in range(51)]
    ok, reason = clock_stall_verdict(repeated)
    assert not ok and "stalled" in reason
    # five seconds of silence followed by a slightly advanced stamp: the old check passed it too
    silent = [(0.0, 20.0), (5.0, 20.01), (5.1, 20.11)]
    ok, reason = clock_stall_verdict(silent)
    assert not ok and "dropped out" in reason
    # terminal silence: no sample and no advance before the window's end
    terminal = [(w, w) for w in np.arange(0, 5, 0.1)]
    ok, reason = clock_stall_verdict(terminal, end_wall_s=8.0)
    assert not ok and "terminal" in reason
    # a rewind is a reset, detected directly on /clock (no tolerance)
    ok, reason = clock_stall_verdict([(0.0, 5.0), (0.1, 4.9)])
    assert not ok and "rewound" in reason
    # an empty window is missing evidence, never a pass
    ok, reason = clock_stall_verdict([])
    assert not ok and "no /clock samples" in reason


# ---------------------------------------------------------------------------
# Review r1 item 15: the candidate census from the executor's own feedback
# ---------------------------------------------------------------------------


def test_pick_feedback_candidate_rounds_counts_per_selection_round():
    lines = [
        "PICK_FEEDBACK phase=selecting progress=0.320 attempt=0 detail=evaluating 1000 candidates",
        "PICK_FEEDBACK phase=selecting progress=0.320 attempt=1 detail=preparing execution candidate 169 (1/3)",
        "PICK_FEEDBACK phase=selecting progress=0.320 attempt=1 detail=evaluating 1000 candidates",
        "PICK_FEEDBACK phase=selecting progress=0.320 attempt=2 detail=evaluating 1000 candidates",
        "PICK_FEEDBACK phase=approach progress=0.480 attempt=2 detail=approaching candidate 612",
    ]
    assert pick_feedback_candidate_rounds(lines) == [1000, 1000, 1000]
    # "unavailable" when the captured feedback carries no evaluating line
    assert (
        pick_feedback_candidate_rounds(
            ["PICK_FEEDBACK phase=approach progress=0.480 attempt=2 detail=approaching candidate 612"]
        )
        is None
    )
    assert pick_feedback_candidate_rounds([]) is None
