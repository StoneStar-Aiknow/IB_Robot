import numpy as np
from scipy.spatial.transform import Rotation

from manipulation_execution.imitation_retargeting.rotations import encode_rotation_6d
from manipulation_execution.imitation_retargeting.streaming import create_stream
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
