"""Behavior tests for human geometry, freshness and pose-match transitions."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manipulation_execution.imitation_retargeting.rotations import encode_rotation_6d
from manipulation_execution.imitation_retargeting.x2._vendor.neutral_fk import NeutralFK
from manipulation_execution.imitation_retargeting.x2._vendor.smplx2robot.source import torso_coordinates
from manipulation_execution.preset_motions import HeartBounds, HeartRuleConfig, MatcherConfig, PoseMatcher
from manipulation_execution.preset_motions.matcher import _evaluate, pose_features


def _stamp(t):
    sec, nanosec = divmod(round(t * 1_000_000_000), 1_000_000_000)
    return SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec))


def _actor(t, actor_id=0, *, present=True):
    return SimpleNamespace(header=_stamp(t), actor_id=actor_id, actor_present=present, start_ready=present)


def _pear(t, sequence, raw, *, actor_id=0, source="yolox"):
    return SimpleNamespace(
        header=_stamp(t),
        sequence=sequence,
        schema="pear_parameter_v1",
        source=source,
        actor_id=actor_id,
        smplx_pose_raw=raw,
    )


def _identity_raw():
    return encode_rotation_6d(np.tile(np.eye(3), (52, 1, 1)))


def _rotation_between(source, target):
    a = source / np.linalg.norm(source)
    b = target / np.linalg.norm(target)
    axis = np.cross(a, b)
    sine = np.linalg.norm(axis)
    cosine = np.clip(np.dot(a, b), -1.0, 1.0)
    if sine < 1e-10:
        if cosine > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-10:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return Rotation.from_rotvec(axis / np.linalg.norm(axis) * np.pi).as_matrix()
    return Rotation.from_rotvec(axis / sine * np.arctan2(sine, cosine)).as_matrix()


def _heart_raw(root=None):
    """Encode a real neutral-skeleton pose with raised, inward-bent forearms."""
    fk = NeutralFK()
    rotations = np.tile(np.eye(3), (52, 1, 1))
    rest = fk(
        {
            "global_orient": rotations[:1],
            "body_pose": rotations[1:22],
            "left_hand_pose": rotations[22:37],
            "right_hand_pose": rotations[37:52],
        }
    )
    torso = torso_coordinates(rest[None])[0]
    for sign, shoulder, elbow, wrist in ((1, 16, 18, 20), (-1, 17, 19, 21)):
        upper = torso.T @ np.array([0.0, sign * 0.6, 0.8])
        forearm = torso.T @ np.array([0.0, -sign * 0.75, 0.66])
        rotations[shoulder] = _rotation_between(fk.offsets[elbow], upper)
        elbow_global = _rotation_between(fk.offsets[wrist], forearm)
        rotations[elbow] = rotations[shoulder].T @ elbow_global
    if root is not None:
        rotations[0] = root
    return encode_rotation_6d(rotations)


def _points(*, wrist_height=1.0, wrist_span=0.8):
    points = np.zeros((145, 3))
    points[0] = [0, 0, -0.8]
    points[12] = [0, 0, 0.2]
    points[15] = [0, 0, 0.5]
    points[13], points[14] = [0, 0.2, 0], [0, -0.2, 0]
    points[16], points[17] = [0, 0.5, 0], [0, -0.5, 0]
    points[18], points[19] = [0, 0.9, 0.7], [0, -0.9, 0.7]
    points[20], points[21] = [0, wrist_span / 2, wrist_height], [0, -wrist_span / 2, wrist_height]
    return points


def _matcher():
    matcher = PoseMatcher()
    matcher.set_session(epoch=1, active=True)
    matcher.tick(now_ros_s=100.0, now_mono_s=0.0)
    return matcher


def _feed(matcher, dt, sequence, raw=None, *, actor_id=0):
    t = 100.0 + dt
    matcher.observe_actor(_actor(t, actor_id), now_mono_s=dt)
    return matcher.observe_pear(
        _pear(t, sequence, _heart_raw() if raw is None else raw, actor_id=actor_id),
        now_ros_s=t,
        now_mono_s=dt,
    )


def _activate(matcher):
    results = [_feed(matcher, i / 10, i) for i in range(1, 8)]
    assert results[-1].active
    assert sum(result.entered for result in results) == 1
    return results[-1]


def test_body_features_ignore_translation_scale_and_tilt():
    points = _points()
    original = pose_features(points)
    rotation = Rotation.from_euler("xyz", [0.4, -0.3, 0.8]).as_matrix()
    transformed = pose_features(points @ rotation.T * 1.7 + [3.0, -2.0, 4.0])
    for name, value in original.items():
        if name != "shoulder_width_m":
            assert transformed[name] == pytest.approx(value)
    assert transformed["shoulder_width_m"] == pytest.approx(1.7)


@pytest.mark.parametrize("span", [0.2, 0.8, 1.3])
def test_large_and_small_overhead_arcs_are_positive(span):
    assert _evaluate(pose_features(_points(wrist_span=span)), HeartBounds())[0]


def test_chest_heart_and_straight_up_arms_are_not_overhead_heart():
    assert not _evaluate(pose_features(_points(wrist_height=0.1)), HeartBounds())[0]
    straight = _points(wrist_height=1.5, wrist_span=1.0)
    straight[18], straight[19] = [0, 0.5, 0.7], [0, -0.5, 0.7]
    assert not _evaluate(pose_features(straight), HeartBounds())[0]


def test_real_6d_decode_fk_features_and_temporal_confirmation():
    matcher = _matcher()
    first = _feed(matcher, 0.1, 1)
    assert first.reason == "confirming"
    assert first.features["left_wrist_height"] >= 0.5
    result = _activate(matcher)
    assert result.template_name == "overhead_heart"
    assert result.source_stamp_ns == 100_700_000_000
    assert 0.5 <= result.score <= 1.0
    assert not _feed(matcher, 0.8, 8).entered


def test_real_fk_positive_remains_positive_under_root_tilt():
    matcher = _matcher()
    raw = _heart_raw(Rotation.from_euler("xyz", [0.4, 0.1, -0.6]).as_matrix())
    results = [_feed(matcher, i / 10, i, raw) for i in range(1, 8)]
    assert results[-1].active


def test_negative_pose_must_persist_then_exits_once():
    matcher = _matcher()
    _activate(matcher)
    initial = _feed(matcher, 0.8, 8, _identity_raw())
    assert initial.active and initial.reason == "exit_confirming"
    results = [_feed(matcher, i / 10, i, _identity_raw()) for i in range(9, 15)]
    assert not results[-1].active
    assert sum(result.exited for result in results) == 1


def test_receive_gap_after_activation_does_not_erase_exit_confirmation():
    matcher = _matcher()
    _activate(matcher)
    first = _feed(matcher, 1.2, 8, _identity_raw())
    assert first.active and first.reason == "exit_confirming"
    second = _feed(matcher, 1.7, 9, _identity_raw())
    assert second.exited and not second.active and second.reason == "pose_exited"


def test_lost_source_tick_revokes_without_another_frame():
    matcher = _matcher()
    _activate(matcher)
    matcher.observe_actor(_actor(101.6), now_mono_s=1.6)
    result = matcher.tick(now_ros_s=101.6, now_mono_s=1.6)
    assert not result.active and result.exited and result.reason == "pose_stale"
    assert not matcher.tick(now_ros_s=101.7, now_mono_s=1.7).exited


def test_lost_actor_revokes_immediately_and_delayed_actor_pose_cannot_restore():
    matcher = _matcher()
    _activate(matcher)
    matcher.observe_actor(_actor(100.8, present=False), now_mono_s=0.8)
    result = matcher.tick(now_ros_s=100.8, now_mono_s=0.8)
    assert result.exited and not result.active
    matcher.observe_actor(_actor(100.9, 1), now_mono_s=0.9)
    delayed = matcher.observe_pear(_pear(100.85, 8, _heart_raw()), now_ros_s=100.95, now_mono_s=0.95)
    assert delayed.reason == "actor_mismatch" and not delayed.active


def test_actor_change_restarts_confirmation():
    matcher = _matcher()
    for i in range(1, 5):
        assert not _feed(matcher, i / 10, i).active
    result = _feed(matcher, 0.5, 5, actor_id=1)
    assert not result.active
    for i in range(6, 11):
        result = _feed(matcher, i / 10, i, actor_id=1)
    assert result.active and result.entered


def test_stop_epoch_blocks_cached_frames_and_preserves_single_exit_event():
    matcher = _matcher()
    _activate(matcher)
    stopped = matcher.set_session(epoch=1, active=False)
    assert stopped.exited and not stopped.active
    assert not matcher.set_session(epoch=1, active=False).exited
    matcher.set_session(epoch=2, active=True)
    matcher.tick(now_ros_s=101.0, now_mono_s=1.0)
    matcher.observe_actor(_actor(100.8), now_mono_s=1.0)
    old = matcher.observe_pear(_pear(100.8, 8, _heart_raw()), now_ros_s=101.0, now_mono_s=1.0)
    assert old.reason == "source_before_session" and not old.active


def test_session_heartbeat_does_not_reset_confirmation():
    matcher = _matcher()
    results = []
    for i in range(1, 8):
        matcher.set_session(epoch=1, active=True)
        results.append(_feed(matcher, i / 10, i))
    assert results[-1].active and sum(item.entered for item in results) == 1


def test_burst_of_buffered_camera_frames_cannot_satisfy_receive_duration():
    matcher = _matcher()
    for i in range(1, 8):
        t = 100 + i / 10
        matcher.observe_actor(_actor(t), now_mono_s=0.7)
        result = matcher.observe_pear(_pear(t, i, _heart_raw()), now_ros_s=100.7, now_mono_s=0.7)
        assert not result.active


def test_duplicate_actor_does_not_refresh_heartbeat():
    matcher = _matcher()
    matcher.observe_actor(_actor(100.1), now_mono_s=0.1)
    matcher.observe_actor(_actor(100.1), now_mono_s=1.2)
    result = matcher.tick(now_ros_s=101.2, now_mono_s=1.2)
    assert result.reason == "actor_stale"


def test_producer_sequence_restart_requires_new_full_confirmation():
    matcher = _matcher()
    _activate(matcher)
    result = _feed(matcher, 0.8, 1)
    assert result.exited and not result.active
    results = [_feed(matcher, (i + 7) / 10, i) for i in range(2, 8)]
    assert results[-1].active
    assert sum(result.entered for result in results) == 1


def test_producer_restart_can_reconfirm_when_first_packet_was_lost():
    matcher = _matcher()
    _activate(matcher)
    restarted = _feed(matcher, 0.8, 2)
    assert restarted.exited and not restarted.active
    results = [_feed(matcher, (i + 6) / 10, i) for i in range(3, 9)]
    assert results[-1].active
    assert sum(result.entered for result in results) == 1


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda msg: setattr(msg, "source", "center_fallback"), "invalid_source"),
        (lambda msg: setattr(msg, "schema", "unknown"), "invalid_source"),
        (lambda msg: setattr(msg, "smplx_pose_raw", np.zeros(312)), "invalid_pose"),
        (lambda msg: setattr(msg, "smplx_pose_raw", [np.nan] * 312), "invalid_pose"),
        (lambda msg: setattr(msg, "smplx_pose_raw", [0.0] * 311), "invalid_pose"),
    ],
)
def test_bad_input_never_activates(mutate, reason):
    matcher = _matcher()
    matcher.observe_actor(_actor(100.1), now_mono_s=0.1)
    message = _pear(100.1, 1, _heart_raw())
    mutate(message)
    result = matcher.observe_pear(message, now_ros_s=100.1, now_mono_s=0.1)
    assert not result.active and result.reason == reason


def test_stale_future_and_duplicate_frames_do_not_keep_pose_alive():
    matcher = _matcher()
    _activate(matcher)
    duplicate = matcher.observe_pear(_pear(100.7, 7, _heart_raw()), now_ros_s=100.8, now_mono_s=0.8)
    assert duplicate.active and duplicate.reason == "duplicate_or_out_of_order"
    future = matcher.observe_pear(_pear(101.5, 8, _heart_raw()), now_ros_s=100.9, now_mono_s=0.9)
    assert future.exited and future.reason == "source_future"
    stale = matcher.observe_pear(_pear(100.1, 9, _heart_raw()), now_ros_s=101.0, now_mono_s=1.0)
    assert not stale.active and stale.reason == "source_stale"


def test_ros_clock_jump_clears_confirmation_and_actor_history():
    matcher = _matcher()
    _feed(matcher, 0.1, 1)
    result = matcher.tick(now_ros_s=90.0, now_mono_s=0.2)
    assert not result.active and result.reason == "clock_reset"
    assert matcher.tick(now_ros_s=90.1, now_mono_s=0.3).reason == "actor_stale"


def test_invalid_and_non_hysteretic_configuration_is_rejected():
    with pytest.raises(ValueError):
        MatcherConfig(enter_confirm_s=float("nan"))
    with pytest.raises(ValueError):
        MatcherConfig(max_frame_age_s=0)
    with pytest.raises(ValueError):
        MatcherConfig(min_enter_frames=True)
    with pytest.raises(ValueError):
        HeartRuleConfig(keep=replace(HeartBounds(), wrist_height_min=0.6))
    with pytest.raises(ValueError):
        HeartBounds(elbow_angle_min_deg=160, elbow_angle_max_deg=150)
