import numpy as np
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
