import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manipulation_execution.imitation_retargeting.rotations import encode_rotation_6d
from manipulation_execution.imitation_retargeting.streaming import create_stream
from manipulation_execution.imitation_retargeting.x2 import followme
from manipulation_execution.imitation_retargeting.x2._vendor import followme_core as core

NEUTRAL = [0.4, 0.0, 0.0, -1.2, 0.0, 0.0, 0.0] * 2


def _raw(body_index=None, axis=0, angle=0.0):
    rotations = np.tile(np.eye(3), (52, 1, 1))
    if body_index is not None:
        euler = [0.0, 0.0, 0.0]
        euler[axis] = angle
        rotations[body_index + 1] = Rotation.from_euler("XYZ", euler).as_matrix()
    return encode_rotation_6d(rotations).reshape(-1)


def test_round_parks_on_the_neutral_pose_then_follows():
    stream = create_stream("x2_followme", base_pose=NEUTRAL)
    stream.start_round()
    samples = []
    for i in range(60):
        t = i * 0.05
        stream.push_pose(t, stream.retarget(t, i, _raw()))
        samples += stream.poll(t + 0.2)
    assert samples, "the gate never opened"
    assert all(s.shape == (14,) for s in samples)
    np.testing.assert_allclose(samples[0], NEUTRAL)
    assert stream.status()["phase"] in ("park", "blend_in", "live")


def test_status_reports_how_far_the_output_runs_behind_the_camera():
    stream = create_stream("x2_followme", base_pose=NEUTRAL, latency_s=0.4)
    stream.start_round()
    assert stream.status()["offset_ms"] is None and stream.status()["age_ms"] is None
    for i in range(60):
        t = i * 0.05
        stream.push_pose(t, np.array(NEUTRAL))
        stream.poll(t + 0.2)            # every frame 200 ms old when polled
    status = stream.status()
    assert status["age_ms"] == 200.0 and status["offset_ms"] == 600.0


def _wave(t):
    return np.array(NEUTRAL) + 0.3 * np.sin(t) * np.array([1, 0, 1, 1, 0, 0, 0] * 2)


def _follow(stream, age, until, start=0.0, frame_dt=0.1):
    """Frames every ``frame_dt`` of camera time, each arriving ``age(t)`` later (None: lost),
    polled at 50 Hz from ``start`` to ``until``. One (now, samples, status) per poll."""
    frames = [(t + age(t), t) for t in np.arange(0.0, until, frame_dt) if age(t) is not None]
    frames.sort()
    out, i = [], 0
    for now in np.arange(start, until, 0.02):
        while i < len(frames) and frames[i][0] <= now:
            stream.push_pose(frames[i][1], _wave(frames[i][1]))
            i += 1
        out.append((now, stream.poll(now), stream.status()))
    return out


def _stalled_until(t_end):
    # PEAR at ~3 Hz and 0.8 s old while the buffer fills, then 10 Hz and 0.15 s old.
    return lambda t: (0.8 if round(t * 10) % 3 == 0 else None) if t < t_end else 0.15


def test_a_stall_while_filling_is_caught_up_within_five_seconds():
    stream = create_stream("x2_followme", base_pose=NEUTRAL, latency_s=0.4)
    stream.start_round()
    polls = _follow(stream, _stalled_until(2.0), 20.0)
    offsets = [s["offset_ms"] for _, _, s in polls if s["offset_ms"] is not None]
    assert offsets[0] > 1100.0                      # calibrated in the stall
    caught = [now for now, _, s in polls if s["catchup_rate"] > 0.0]
    assert caught and caught[-1] - caught[0] < 5.0
    final = polls[-1][2]
    assert final["catchups"] == 1 and final["catchup_rate"] == 0.0
    assert abs(final["offset_ms"] - final["offset_target_ms"]) < 1.0
    assert final["offset_ms"] < 700.0
    assert (np.diff(offsets) <= 0.0).all()                 # only ever down
    live = [samples for now, samples, s in polls if s["phase"] == "live"]
    assert all(len(samples) == 1 for samples in live)          # still one sample per tick
    steps = np.abs(np.diff(np.array([p for samples in live for p in samples]), axis=0)).max()
    assert steps < 0.3 * 0.02 * 1.5, steps                     # the wave at most 1.2x fast, no jump
    assert final["starved_frames"] == polls[len(polls) // 2][2]["starved_frames"]


def test_without_a_stall_the_offset_never_moves():
    stream = create_stream("x2_followme", base_pose=NEUTRAL, latency_s=0.4)
    stream.start_round()
    rng = np.random.default_rng(0)
    ages = {round(t, 1): a for t, a in zip(np.arange(0.0, 30.0, 0.1), rng.uniform(0.12, 0.3, 300))}
    polls = _follow(stream, lambda t: ages[round(t, 1)], 30.0)
    offsets = {s["offset_ms"] for _, _, s in polls if s["offset_ms"] is not None}
    assert len(offsets) == 1
    assert polls[-1][2]["catchups"] == 0 and polls[-1][2]["caught_up_ms"] == 0.0


def test_a_second_stall_or_a_reentry_while_catching_up_stops_it():
    stream = create_stream("x2_followme", base_pose=NEUTRAL, latency_s=0.4)
    stream.start_round()
    polls = _follow(stream, _stalled_until(2.0), 7.0)
    assert polls[-1][2]["catchup_rate"] > 0.0
    # PEAR stalls again: the ages climb back over the offset and the catch-up stops.
    again = _follow(stream, lambda t: 0.15 if t < 6.9 or t > 10.0 else (1.2 if t > 9.0 else None), 10.5, start=7.0)
    stopped = [s for _, _, s in again if s["catchup_rate"] == 0.0]
    assert stopped and stopped[0]["offset_ms"] <= stopped[0]["offset_target_ms"] + 0.1
    offsets = [s["offset_ms"] for _, _, s in polls + again if s["offset_ms"] is not None]
    assert (np.diff(offsets) <= 0.0).all()
    # A re-entry recalibrates from scratch and drops the catch-up; the counts carry on.
    caught = again[-1][2]["caught_up_ms"]
    stream.start_round(from_pose=again[-1][1][-1] if again[-1][1] else NEUTRAL)
    status = stream.status()
    assert status["catchup_rate"] == 0.0 and status["offset_ms"] is None and status["offset_target_ms"] is None
    assert status["catchups"] >= 1 and status["caught_up_ms"] == caught


def test_a_round_from_a_pose_parks_on_it_then_blends_into_the_person():
    stream = create_stream("x2_followme", base_pose=NEUTRAL)
    inside = np.array([1, 0, 1, 1, 1, 0, 0] * 2)  # joints with room either way of neutral
    held = np.array(NEUTRAL) + 0.3 * inside
    person = np.array(NEUTRAL) - 0.2 * inside
    stream.start_round(from_pose=held)
    samples = []
    for i in range(80):
        t = i * 0.05
        stream.push_pose(t, person)
        samples += stream.poll(t + 0.2)
    first, last = np.array(samples[0]), np.array(samples[-1])
    np.testing.assert_array_equal(first, held)
    np.testing.assert_allclose(last, person, atol=1e-9)
    steps = np.abs(np.diff(np.array(samples), axis=0)).max()
    assert steps < 0.02, steps          # 0.5 rad over a 2 s quintic blend, no jump
    stream._stream.starved_frames = 7
    stream.start_round(from_pose=person)
    assert stream.status()["starved_frames"] == 7      # a re-entry keeps the round's counts
    stream.start_round()
    assert np.array_equal(stream._stream.base, NEUTRAL)
    assert stream.status()["starved_frames"] == 0      # a new round does not


def test_branch_guard_survives_trimmed_history():
    # A wrist channel turns through +pi after more single-frame pushes than the
    # history keeps; the guard must still widen the window and unwrap it.
    stream = create_stream("x2_followme", base_pose=NEUTRAL, mirror=False)
    ref = stream._stream.retargeter
    body_index, axis = ref.guard.channels[0]
    angles = [*np.linspace(2.0, 3.0, 12), *np.linspace(3.05, 3.6, 8), *np.linspace(3.5, 2.5, 6)]
    frames = [stream._frame(i * 0.05, i, _raw(body_index, axis, a)) for i, a in enumerate(angles)]

    def run(max_anchor_frames):
        retargeter = core.StreamRetargeter(stream._vendor, ref.kinematics, ref.ledger,
                                           max_anchor_frames=max_anchor_frames, mirror=False)
        return np.array([retargeter.push(f) for f in frames]), retargeter

    full, _ = run(1200)
    trimmed, retargeter = run(10)
    assert len(retargeter._frames) < len(frames) and retargeter.guard.trips > 0
    assert not retargeter.notes  # the cap itself never bit
    np.testing.assert_array_equal(trimmed, full)


def _held_trip(stream, frames_past_the_cut):
    # One wrist channel turns through +pi and stays past it, so the guard stays tripped.
    body_index, axis = stream._stream.retargeter.guard.channels[0]
    angles = [*np.linspace(2.5, 3.0, 6), *[3.3] * frames_past_the_cut]
    for i, angle in enumerate(angles):
        stream.retarget(i * 0.05, i, _raw(body_index, axis, angle))


def test_a_held_trip_keeps_the_window_and_history_bounded(monkeypatch):
    monkeypatch.setattr(followme, "MAX_ANCHOR_FRAMES", 10)
    stream = create_stream("x2_followme", base_pose=NEUTRAL, mirror=False)
    retargeter = stream._stream.retargeter
    windows = []
    animation = retargeter._animation
    monkeypatch.setattr(retargeter, "_animation",
                        lambda frames: windows.append([f.frame_id for f in frames]) or animation(frames))
    _held_trip(stream, 40)
    assert retargeter.guard.trips > 0 and retargeter.notes
    assert max(len(w) for w in windows) == 10 and len(retargeter._frames) <= 10
    assert all(w == list(range(w[-1] - len(w) + 1, w[-1] + 1)) for w in windows)
    assert len(retargeter.notes) <= followme.NOTES_KEPT


def test_reset_input_keeps_the_sessions_guard_counts_and_notes(monkeypatch):
    monkeypatch.setattr(followme, "MAX_ANCHOR_FRAMES", 10)
    stream = create_stream("x2_followme", base_pose=NEUTRAL, mirror=False)
    _held_trip(stream, 20)
    before = stream.status()
    assert before["branch_guard_trips"] > 0 and before["notes"]
    stream.reset_input()
    after = stream.status()
    assert stream._stream.retargeter.max_anchor_frames == 10
    for key in ("branch_guard_trips", "branch_guard_worst", "notes"):
        assert after[key] == before[key], key


def test_jittery_ticks_still_get_one_sample_each():
    # Ticks a millisecond either side of the 50 Hz grid, as a ROS timer delivers them.
    stream = create_stream("x2_followme", base_pose=NEUTRAL)
    stream.start_round()
    for i in range(12):
        stream.push_pose(i * 0.05, stream.retarget(i * 0.05, i, _raw()))
    now = 0.6
    while not stream.poll(now):
        now += 0.02
    counts = [len(stream.poll(now + k * 0.02 + (0.001 if k % 2 else -0.001))) for k in range(1, 40)]
    assert counts == [1] * len(counts)


def test_a_mapping_process_gives_the_in_process_poses_and_counts():
    body_index, axis = create_stream("x2_followme", base_pose=NEUTRAL)._stream.retargeter.guard.channels[0]
    raws = [_raw(body_index, axis, a) for a in np.linspace(2.6, 3.6, 30)]
    local = create_stream("x2_followme", base_pose=NEUTRAL)
    remote = create_stream("x2_followme", base_pose=NEUTRAL, separate_process=True)
    try:
        for stream in (local, remote):
            stream.reset_input()
        poses = [(local.retarget(i * 0.05, i, r), remote.retarget(i * 0.05, i, r)) for i, r in enumerate(raws)]
        for mine, theirs in poses:
            np.testing.assert_array_equal(mine, theirs)
        local_status, remote_status = local.status(), remote.status()
        assert remote_status["branch_guard_trips"] > 0
        for key in ("branch_guard_trips", "branch_guard_worst", "retargeted", "input_resets", "shoulder_ik", "notes"):
            assert remote_status[key] == local_status[key], key
        with pytest.raises(ValueError, match="smplx_pose_raw"):
            remote.retarget(2.0, 99, np.zeros(5))
        np.testing.assert_array_equal(remote.retarget(2.05, 100, raws[-1]), local.retarget(2.05, 100, raws[-1]))
    finally:
        remote.close()
    assert remote._child._process.poll() == 0


# elbow_spin_to_wrist: the elbow's twist about the forearm bone moves onto the wrist.
LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST = 18, 19, 20   # indices into the 52 rotations
SHOULDERS_AND_ELBOWS = [0, 1, 2, 3, 7, 8, 9, 10]
WRISTS = [4, 5, 6, 11, 12, 13]


def _people(count, seed=0):
    # Random poses of every joint, up to 1.5 rad each, as (52, 3, 3) local rotations.
    rng = np.random.default_rng(seed)
    rotvecs = rng.normal(size=(count, 52, 3))
    rotvecs *= rng.uniform(0.0, 1.5, (count, 52, 1)) / np.linalg.norm(rotvecs, axis=-1, keepdims=True)
    return Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_matrix().reshape(count, 52, 3, 3)


def _posed(stream, rotations):
    # The NeutralFK points and global rotations of a (52, 3, 3) pose.
    fk = stream._fk
    points = fk({"global_orient": rotations[0:1], "body_pose": rotations[1:22],
                 "left_hand_pose": rotations[22:37], "right_hand_pose": rotations[37:52]})
    return points.astype(np.float64), fk._rot.copy()


def _mapped(rotations, **params):
    stream = create_stream("x2_followme", base_pose=NEUTRAL, **params)
    return stream.retarget(0.0, 0, encode_rotation_6d(rotations).reshape(-1))


def _twist(rotation, axis):
    # The twist about the axis, within half a turn.
    quat = Rotation.from_matrix(rotation).as_quat()
    return (2.0 * np.arctan2(quat[:3] @ axis, quat[3]) + np.pi) % (2.0 * np.pi) - np.pi


@pytest.mark.parametrize("gain, rest", [(1.0, 0.0), (0.5, -0.4)])
def test_moving_the_elbow_spin_keeps_every_point_and_global_rotation(gain, rest):
    stream = create_stream("x2_followme", base_pose=NEUTRAL)
    for rotations in _people(20):
        moved = followme.elbow_spin_to_wrist(rotations, stream._forearm_axes, gain, rest)
        changed = np.flatnonzero(np.abs(moved - rotations).max(axis=(1, 2)) > 0)
        assert set(changed) <= set(sum(followme._FOREARMS, ())), changed
        points, rots = _posed(stream, rotations)
        moved_points, moved_rots = _posed(stream, moved)
        np.testing.assert_allclose(moved_points, points, atol=1e-9)
        # The elbows' own global rotations lose the twist; from the wrists down nothing moves.
        others = [j for j in range(55) if j not in (LEFT_ELBOW, RIGHT_ELBOW)]
        np.testing.assert_allclose(moved_rots[others], rots[others], atol=1e-9)
        # The elbow keeps the share not moved, measured from the rest twist.
        for (elbow, _), axis in zip(followme._FOREARMS, stream._forearm_axes):
            phi, rest_about = _twist(rotations[elbow], axis), np.sign(axis[0]) * rest
            assert abs(_twist(moved[elbow], axis) - (phi - gain * (phi - rest_about))) < 1e-9


def test_with_the_spin_moved_only_the_wrist_channels_change():
    for rotations in _people(5, seed=1):
        off, on = _mapped(rotations), _mapped(rotations, elbow_spin_to_wrist=True)
        np.testing.assert_allclose(on[SHOULDERS_AND_ELBOWS], off[SHOULDERS_AND_ELBOWS], atol=1e-5)
        assert np.abs(on[WRISTS] - off[WRISTS]).max() > 1e-3


def test_an_elbow_at_pears_rest_twist_maps_as_before():
    # PEAR twists a still elbow by ELBOW_REST_TWIST_DEG; moving that turned the palms on the robot.
    stream = create_stream("x2_followme", base_pose=NEUTRAL)
    rotations = np.tile(np.eye(3), (52, 1, 1))
    for (elbow, wrist), axis in zip(followme._FOREARMS, stream._forearm_axes):
        rest = np.sign(axis[0]) * np.radians(followme.ELBOW_REST_TWIST_DEG)
        rotations[elbow] = (Rotation.from_rotvec(1.2 * np.cross(axis, [0.0, 0.0, 1.0])).as_matrix()
                            @ Rotation.from_rotvec(rest * axis).as_matrix())
        rotations[wrist] = Rotation.from_euler("XYZ", [0.3, 0.2, -0.1]).as_matrix()
    off = _mapped(rotations)
    for gain in (0.5, 1.0):
        np.testing.assert_allclose(_mapped(rotations, elbow_spin_to_wrist=True, elbow_spin_gain=gain), off,
                                   atol=1e-6)
    # Moved from no twist instead, the wrist turns: the test can fail.
    plain = followme.elbow_spin_to_wrist(rotations, stream._forearm_axes)
    assert np.abs(stream.retarget(0.0, 0, encode_rotation_6d(plain).reshape(-1)) - off).max() > np.radians(10.0)


def test_a_gain_of_zero_maps_as_off():
    for rotations in _people(3, seed=3):
        np.testing.assert_array_equal(_mapped(rotations, elbow_spin_to_wrist=True, elbow_spin_gain=0.0),
                                      _mapped(rotations))


@pytest.mark.parametrize("gain", [-0.1, 1.1])
def test_a_gain_outside_zero_to_one_is_refused(gain):
    with pytest.raises(ValueError, match="elbow_spin_gain"):
        create_stream("x2_followme", base_pose=NEUTRAL, elbow_spin_to_wrist=True, elbow_spin_gain=gain)


@pytest.mark.parametrize("elbow, robot_wrist_yaw", [(LEFT_ELBOW, 11), (RIGHT_ELBOW, 4)])
def test_an_elbow_twist_turns_the_mirrored_wrist_as_the_wrist_euler_x_does(elbow, robot_wrist_yaw):
    # Mirrored, her left arm drives the robot's right. The wrist's own Euler-X turn is
    # the ledger's checked reading; the elbow's twist must now land the same way.
    axis = create_stream("x2_followme", base_pose=NEUTRAL)._forearm_axes[elbow - LEFT_ELBOW]
    bent = Rotation.from_rotvec(1.0 * np.cross(axis, [0.0, 0.0, 1.0])).as_matrix()
    rest = np.tile(np.eye(3), (52, 1, 1))
    rest[elbow] = bent
    by_elbow, by_wrist = rest.copy(), rest.copy()
    by_elbow[elbow] = bent @ Rotation.from_rotvec(np.radians(30.0) * axis).as_matrix()
    # The bone runs along +x on her left and -x on her right: +30 deg about it is Euler-X of that sign.
    by_wrist[elbow + 2] = Rotation.from_euler("XYZ", [np.sign(axis[0]) * np.radians(30.0), 0.0, 0.0]).as_matrix()
    whole = dict(elbow_spin_to_wrist=True, elbow_spin_gain=1.0)
    base = _mapped(rest, **whole)[robot_wrist_yaw]
    turned = _mapped(by_elbow, **whole)[robot_wrist_yaw] - base
    reference = _mapped(by_wrist, **whole)[robot_wrist_yaw] - base
    assert abs(reference) > np.radians(25.0)
    assert abs(turned - reference) < np.radians(3.0), np.degrees([turned, reference])
    # At the default gain half of the elbow's turn arrives.
    half = (_mapped(by_elbow, elbow_spin_to_wrist=True)[robot_wrist_yaw]
            - _mapped(rest, elbow_spin_to_wrist=True)[robot_wrist_yaw])
    assert abs(half - 0.5 * reference) < np.radians(3.0), np.degrees([half, reference])
    # Off, the elbow's twist never reaches the wrist: the loss this fixes.
    assert abs(_mapped(by_elbow)[robot_wrist_yaw] - _mapped(rest)[robot_wrist_yaw]) < np.radians(1.0)


def test_an_elbow_bent_half_a_turn_is_left_alone():
    stream = create_stream("x2_followme", base_pose=NEUTRAL, elbow_spin_to_wrist=True)
    rotations = np.tile(np.eye(3), (52, 1, 1))
    across = np.cross(stream._forearm_axes[0], [0.0, 0.0, 1.0])
    rotations[LEFT_ELBOW] = Rotation.from_rotvec(np.pi * across / np.linalg.norm(across)).as_matrix()
    moved = followme.elbow_spin_to_wrist(rotations, stream._forearm_axes)
    np.testing.assert_array_equal(moved[LEFT_ELBOW], rotations[LEFT_ELBOW])
    np.testing.assert_array_equal(moved[LEFT_WRIST], rotations[LEFT_WRIST])
    assert np.isfinite(stream.retarget(0.0, 0, encode_rotation_6d(rotations).reshape(-1))).all()


def test_the_mapping_process_moves_the_spin_too_and_status_says_so():
    raws = [encode_rotation_6d(r).reshape(-1) for r in _people(10, seed=2)]
    spin = dict(elbow_spin_to_wrist=True, elbow_spin_gain=0.7)
    local = create_stream("x2_followme", base_pose=NEUTRAL, **spin)
    remote = create_stream("x2_followme", base_pose=NEUTRAL, separate_process=True, **spin)
    # Had the child missed the gain it would map as the default does, which differs.
    default = create_stream("x2_followme", base_pose=NEUTRAL, elbow_spin_to_wrist=True)
    try:
        for i, r in enumerate(raws):
            pose = remote.retarget(i * 0.05, i, r)
            np.testing.assert_array_equal(pose, local.retarget(i * 0.05, i, r))
            assert np.abs(pose - default.retarget(i * 0.05, i, r)).max() > 1e-3
        assert (remote.status()["elbow_spin_to_wrist"], remote.status()["elbow_spin_gain"]) == (True, 0.7)
    finally:
        remote.close()
    status = create_stream("x2_followme", base_pose=NEUTRAL).status()
    assert (status["elbow_spin_to_wrist"], status["elbow_spin_gain"]) == (False, followme.ELBOW_SPIN_GAIN)


# The shoulders: _vendor/shoulder_ik solves pitch/roll/yaw for her upper arm's direction.
LEFT_SHOULDER, RIGHT_SHOULDER = 16, 17               # indices into the 52 rotations
ROLLS = [1, 8]
# Branch A: |roll| <= 90 deg, plus the 0.0002 deg the mirror adds to roll (Trajectory.mirrored).
ROLL_BOUND = np.pi / 2 + np.radians(0.001)


def _arms_raised(stream, elevation, azimuth):
    # Both upper arms swung (no twist) from hanging to `elevation` deg up, `azimuth` deg
    # from her front towards her side (0 in front, 90 out to the side), mirror images.
    e, a = np.radians(elevation), np.radians(azimuth)
    left = np.array([np.sin(e) * np.sin(a), -np.cos(e), np.sin(e) * np.cos(a)])  # SMPL-X: left, up, front
    rotations = np.tile(np.eye(3), (52, 1, 1))
    for shoulder, elbow, her in ((LEFT_SHOULDER, LEFT_ELBOW, left), (RIGHT_SHOULDER, RIGHT_ELBOW, left * [-1, 1, 1])):
        rest = stream._fk.offsets[elbow] / np.linalg.norm(stream._fk.offsets[elbow])
        axis = np.cross(rest, her)
        angle = np.arctan2(np.linalg.norm(axis), rest @ her)
        rotations[shoulder] = Rotation.from_rotvec(angle * axis / np.linalg.norm(axis)).as_matrix()
    return encode_rotation_6d(rotations).reshape(-1)


def _pointing_error_deg(stream, pose, raw):
    # Per robot arm, the angle between its upper arm and hers (both arms raised alike).
    retargeter = stream._stream.retargeter
    animation = retargeter._animation([stream._frame(0.0, 0, raw)])
    kinematics = retargeter.kinematics
    kinematics.set_arm(pose)
    errors = []
    for side in ("left", "right"):
        her = followme.readouts.twist_indicators(animation, side)["bone"][-1]
        robot = kinematics.anchor(f"{side}_elbow_joint") - kinematics.anchor(f"{side}_shoulder_roll_joint")
        errors.append(np.degrees(np.arccos(np.clip(robot @ her / np.linalg.norm(robot), -1.0, 1.0))))
    return np.array(errors)


@pytest.mark.parametrize("mirror", [False, True])
@pytest.mark.parametrize("azimuth", [20.0, 45.0, 90.0])
def test_arms_raised_overhead_point_overhead(mirror, azimuth):
    # The vendor's readouts counted the lift twice past the horizontal: overhead its
    # roll read ~150 deg too and the arm pointed 100-140 deg off, behind her.
    stream = create_stream("x2_followme", base_pose=NEUTRAL, mirror=mirror)
    vendor = core.StreamRetargeter(stream._vendor, stream._stream.retargeter.kinematics,
                                   stream._stream.retargeter.ledger, mirror=mirror)
    for i, elevation in enumerate(np.linspace(1.0, 160.0, 90)):   # 3 s, slow enough for 360 deg/s
        raw = _arms_raised(stream, elevation, azimuth)
        pose, before = stream.retarget(i / 30, i, raw), vendor.push(stream._frame(i / 30, i, raw))
        assert np.abs(pose[ROLLS]).max() <= ROLL_BOUND
    assert _pointing_error_deg(stream, pose, raw).max() < 15.0
    assert _pointing_error_deg(stream, before, raw).min() > 90.0


def test_shoulder_roll_stays_on_one_branch_for_any_pose():
    stream = create_stream("x2_followme", base_pose=NEUTRAL)
    for i, rotations in enumerate(_people(200, seed=4)):
        pose = stream.retarget(i / 15, i, encode_rotation_6d(rotations).reshape(-1))
        assert np.abs(pose[ROLLS]).max() <= ROLL_BOUND, i


def _motion(count, seed, scale):
    # Smooth random motion of every joint at 15 Hz, as 6D rows; scale 2 crosses the wrist's Euler cut.
    rng = np.random.default_rng(seed)
    knots = rng.normal(size=(count // 15 + 2, 52, 3)) * scale
    t = np.arange(count) / 15.0
    rotvecs = np.stack([np.stack([np.interp(t, np.arange(len(knots)), knots[:, j, k]) for k in range(3)], -1)
                        for j in range(52)], 1)
    return t, encode_rotation_6d(Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_matrix()).reshape(count, -1)


def test_the_shoulder_speedups_keep_the_output():
    # followme_feaebe22.npz is this exact run on feaebe22, before the shoulder solver skipped
    # yaw_need for candidates that cannot win and reused its answers; there the twist cost
    # decided on 138 of the 540 solver steps. Bit for bit on x86. On the 310P (aarch64) both
    # feaebe22 and the patched code land up to 5e-7 rad off x86, identical to each other; skipping
    # a candidate that could win or reusing the wrong answer moves joints by 0.2 rad or more.
    golden = np.load(Path(__file__).resolve().parent / "data" / "x2_shoulder_ik" / "followme_feaebe22.npz")
    for seed, scale, mirror in ((0, 0.6, True), (1, 2.0, True), (2, 1.2, False)):
        t, raw = _motion(90, seed, scale)
        stream = create_stream("x2_followme", base_pose=NEUTRAL, mirror=mirror, elbow_spin_to_wrist=True,
                               elbow_spin_gain=1.0)
        poses = []
        for i, (ti, row) in enumerate(zip(t, raw, strict=True)):
            if i == 60:
                stream.reset_input()
            poses.append(stream.retarget(ti, i, row))
        np.testing.assert_allclose(poses, golden[f"seed{seed}"], rtol=0, atol=1e-5, err_msg=f"seed {seed}")


def test_her_arms_reads_what_twist_indicators_does():
    stream = create_stream("x2_followme", base_pose=NEUTRAL)
    retargeter = stream._stream.retargeter
    _, raw = _motion(30, 3, 2.0)
    animation = retargeter._animation([stream._frame(i / 15, i, row) for i, row in enumerate(raw)])
    arms = followme._her_arms(animation, "plane")
    for side in ("left", "right"):
        full = followme.readouts.twist_indicators(animation, side)
        assert arms[side].keys() == {"bone", "plane", "conditioning"}
        for key, value in arms[side].items():
            np.testing.assert_array_equal(value, full[key])
    assert followme._her_arms(animation, "spin")["left"].keys() == full.keys()


def test_shoulder_geometry_is_the_ports():
    # The port exported geometry_x2.json from the same x2_ultra.xml (md5 6992648c...).
    stream = create_stream("x2_followme", base_pose=NEUTRAL)
    data = json.loads((Path(__file__).resolve().parent / "data" / "x2_shoulder_ik" / "geometry_x2.json").read_text())
    for side, geometry in stream._shoulders.items():
        port = followme.ShoulderGeometry.from_dict(data["sides"][side])
        for field in ("pitch_axis", "roll_axis", "rest_dir", "pitch_limit", "roll_limit", "yaw_limit"):
            np.testing.assert_allclose(getattr(geometry, field), getattr(port, field), atol=1e-9, err_msg=field)


def test_a_wrong_shoulder_geometry_is_refused(monkeypatch):
    stream = create_stream("x2_followme", base_pose=NEUTRAL)
    retargeter = stream._stream.retargeter
    anchor = retargeter.kinematics.anchor
    # The upper arm measured from the pitch joint, as the port warns against: 1.4 deg off.
    monkeypatch.setattr(retargeter.kinematics, "anchor",
                        lambda name: anchor(name.replace("shoulder_roll", "shoulder_pitch")))
    with pytest.raises(RuntimeError, match="shoulder geometry"):
        followme.shoulder_geometry(retargeter.kinematics, retargeter.ledger)


def test_reset_input_starts_new_shoulder_solvers_and_keeps_their_counts():
    stream = create_stream("x2_followme", base_pose=NEUTRAL)
    for i, elevation in enumerate(np.linspace(1.0, 160.0, 20)):
        stream.retarget(i / 30, i, _arms_raised(stream, elevation, 90.0))
    before = stream.status()["shoulder_ik"]
    assert before["left"]["frames"] == 20 and before["left"]["rate_limited"] > 0
    old = stream._stream.retargeter.solvers
    stream.reset_input()
    assert stream.status()["shoulder_ik"] == before
    solvers = stream._stream.retargeter.solvers
    assert all(solvers[side] is not old[side] and solvers[side].stats["frames"] == 0 for side in old)
    # A fresh solver does not rate-limit its first frame: the arm lands at once.
    pose = stream.retarget(5.0, 20, _arms_raised(stream, 1.0, 90.0))
    assert _pointing_error_deg(stream, pose, _arms_raised(stream, 1.0, 90.0)).max() < 20.0
    assert stream.status()["shoulder_ik"]["left"]["frames"] == 21
