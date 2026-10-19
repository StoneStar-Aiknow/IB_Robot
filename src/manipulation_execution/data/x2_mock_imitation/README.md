# X2 mock imitation animation

A recorded upper-body animation that closes the motion-imitation data flow
without a retargeter. The mock retargeter plays it instead of computing a
trajectory from the PEAR stream, so the whole chain
(camera -> YOLOX -> PEAR -> retarget -> safety -> arm command) can be exercised
end to end while each stage is replaced independently.

A retargeter that replaces the mock keeps the playback node's output contract;
this file is then no longer read.

## Provenance

Produced by the `smplx2robot` delivery pipeline (clip
`rec_20260914_212729`), which retargets a recorded human performance to the X2
arms and fits it to the vendor speed table. Only joint angles are shipped -- no
images, video or body-shape parameters of the recorded person.

The `.npz` and `.json` are the producer's output byte for byte. The
`.provenance.json` is too, except that `source_npz` was reduced from an
absolute path on the producing machine to the bare file name; its
`source_sha256` still identifies the exact upstream input.

Its provenance reports a peak joint speed of 53.8 % of the vendor rating and
no sample outside the joint limits. Collision, communication timeout and
emergency stop were not assessed by the producer: the provenance states that
the clip is not a hardware clearance.

The manifest and the `.npz` must travel together: the manifest names the data
file and the arrays inside it.

## Format

`aimdk-upper-body-animation` version 1, `representation: sampled`,
`rate_hz: 50.0`. Four arrays, all float64:

| array | shape | meaning |
|---|---|---|
| `timestamp_s` | (616,) | seconds from 0, exact 0.02 steps, last 12.30 |
| `arm_pos` | (616, 14) | **radians**, positional order = the runtime profile's `arm_joints` |
| `head_pos` | (616, 2) | constant zeros, not a considered value |
| `hand_pos` | (616, 2) | constant zeros, `hand_sub_mode: 1` |

Column order of `arm_pos`, confirmed against the vendor URDF and the platform's
own `arm_pos` indexing:

```
0 left_shoulder_pitch   1 left_shoulder_roll   2 left_shoulder_yaw   3 left_elbow
4 left_wrist_yaw        5 left_wrist_pitch     6 left_wrist_roll
7 right_shoulder_pitch  8 right_shoulder_roll  9 right_shoulder_yaw 10 right_elbow
11 right_wrist_yaw     12 right_wrist_pitch   13 right_wrist_roll
```

Frame 0 and the last frame are the same base pose, held exactly (the producer
asserts the endpoints to 1e-9 rad and inserts a 0.5 s dwell), so the clip loops
without a discontinuity.

`head_pos` and `hand_pos` are zeros because the producer's schema refuses a
manifest without them, **not** because zero is a meaningful target for either.
Do not forward them to the head or the grippers.

## Limits are not in here

The joint position and velocity limits the mock safety clamps against are
application configuration, not part of this asset or of the runtime profile. They
are set in the robot configuration that launches the mock
(`robot_config/config/robots/aimdk_x2_magicmirror.yaml`,
`motion_imitation_stream.mock_playback.arm_limits`), cited to the vendor URDF
`X2_URDF-v1.3.0`. The same numbers appear in this clip's upstream
`_smoothed.npz` measured table and agree to four decimals.
