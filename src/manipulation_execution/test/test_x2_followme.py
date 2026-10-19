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
        retargeter = core.StreamRetargeter(ref._v, ref.kinematics, ref.ledger,
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
        for key in ("branch_guard_trips", "branch_guard_worst", "retargeted", "input_resets", "notes"):
            assert remote_status[key] == local_status[key], key
        with pytest.raises(ValueError, match="smplx_pose_raw"):
            remote.retarget(2.0, 99, np.zeros(5))
        np.testing.assert_array_equal(remote.retarget(2.05, 100, raws[-1]), local.retarget(2.05, 100, raws[-1]))
    finally:
        remote.close()
    assert remote._child._process.poll() == 0
