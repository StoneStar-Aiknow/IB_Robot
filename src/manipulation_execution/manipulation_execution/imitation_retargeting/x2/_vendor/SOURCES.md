# Vendored from video2x2_runtime

Upstream: video2x2_runtime, commit 0717710. Copied byte for byte; the two
files marked PATCHED were then edited in later commits, and each edit carries
an "IB-Robot patch" comment.

| here | upstream | sha256 of the upstream file |
| --- | --- | --- |
| `data/smplx_neutral_skeleton.npz` | `ros2_ws/src/hri_pear_stream/hri_pear_stream/data/smplx_neutral_skeleton.npz` | `f00ce4a2629636dcfe6e3811af907153962d21ebf91df1ff16cf4118dde1c422` |
| `followme_core.py` (PATCHED) | `ros2_ws/src/hri_pear_stream/hri_pear_stream/followme_core.py` | `e6ee4297df44ec4970fa4d82ffbe16a8d6e8b92faf1f242592a8218aa45925ab` |
| `neutral_fk.py` | `ros2_ws/src/hri_pear_stream/hri_pear_stream/neutral_fk.py` | `daa2bdef76350f84031b51da6cc848dd1c704cd201913354c77381e7028432d1` |
| `smplx2robot/data/x2_retarget_ledger.json` | `vendor/smplx2robot/src/smplx2robot/data/x2_retarget_ledger.json` | `0739dd3a9bf78f7f6a6492d1481136e8fcf9b6a4c662dfbc7518daf7669de89b` |
| `smplx2robot/__init__.py` | `vendor/smplx2robot/src/smplx2robot/__init__.py` | `4ae7429076759d438bd73d1facd3c862b8863b7c64b4e1175e19ebfefe415cb2` |
| `smplx2robot/ledger.py` | `vendor/smplx2robot/src/smplx2robot/ledger.py` | `f5350b5ec390a8a4f1923aa5cc31f373aa97b18842dc1bd32ce95c8bb6be3272` |
| `smplx2robot/paths.py` | `vendor/smplx2robot/src/smplx2robot/paths.py` | `ef51a2ceac88070926fcd3933daee353e8b05abe32888db1e94fe3ab7b979b3a` |
| `smplx2robot/pear.py` | `vendor/smplx2robot/src/smplx2robot/pear.py` | `1ce07b739594c1d384b552a4a05eee62d4cbb8661c1085e5d6cb43fb4fedb777` |
| `smplx2robot/postprocess.py` | `vendor/smplx2robot/src/smplx2robot/postprocess.py` | `ae1b3c63a4d274d05e889b402b051f3fd30d6cd087e03dab3b2cae13b4255d24` |
| `smplx2robot/readouts.py` | `vendor/smplx2robot/src/smplx2robot/readouts.py` | `f791a66e2c62fcb357a78147548e1b35be03c88b7e372b1f31e7d974c8c68b79` |
| `smplx2robot/retarget.py` | `vendor/smplx2robot/src/smplx2robot/retarget.py` | `02bbb1c182cfb34fed0780fa0928a19d9e407fef15b4b0384b0845f0715204d5` |
| `smplx2robot/robot.py` (PATCHED) | `vendor/smplx2robot/src/smplx2robot/robot.py` | `68922cfcdc950577b9fea6e2fc1160359d267eb200cd117ef512f9283e38fdea` |
| `smplx2robot/source.py` | `vendor/smplx2robot/src/smplx2robot/source.py` | `dd92cc6b43cbe0e3162fe3d31d742f9a86f4ee563572eb45d416b6202df10451` |

Patches:

* `smplx2robot/robot.py`: `X2Kinematics` runs on `x2.kinematics.MjcfKinematics`
  (numpy) instead of mujoco, and loads the package's `x2/model/x2_ultra.xml`
  by default instead of searching for the vendor drop. Its public methods and
  results are unchanged (`test_x2_kinematics.py` pins the FK to mujoco).
* `followme_core.py`: `_import_vendor` imports the copy beside it instead of
  inserting `SMPLX2ROBOT_DIR` into `sys.path`; the per-frame mirror FK
  self-check is dropped (`x2.followme` runs it once at startup); and
  `StreamRetargeter` numbers frames absolutely across the trimmed history. Upstream
  numbered them by list position, so after `max_anchor_frames` pushes (about a
  minute of tracking) the branch guard never widened its window again and a wrist
  crossing ±π stepped by 2π (`test_x2_followme.py` pins this). When the window
  hits `max_anchor_frames` it also drops the frames before it, which no later
  window reaches; upstream kept every frame while the guard stayed tripped.

Not copied: the rest of `smplx2robot` (cli, export, delivery, safety, preview,
compare, verify), which the streaming path does not import.

## Dexterous hand (video2x2_runtime 570c53c)

Copied byte for byte from video2x2_runtime commit 570c53c; only `hand_core.py`
is PATCHED, and each edit carries an "IB-Robot patch" comment. Before this
copy the same files were loaded at run time from `/root/x2-dexhand-run/extern`;
a non-empty `extern_dir` still loads them from a directory
(`hand_perception/npu_hands.load_extern`).

| here | upstream | sha256 of the upstream file | sha256 here |
| --- | --- | --- | --- |
| `hand_core.py` (PATCHED) | `ros2_ws/src/hri_pear_stream/hri_pear_stream/hand_core.py` | `44b213dd1c4bdb64da23dfe6ab722e08e573eee049cad2655af7051bafbfc0c6` | `897c53a639a151179efed6e9948f49aeb053a734de0c4e5f8c8fd7db14d2960c` |
| `hand_keypoints.py` | `ros2_ws/src/hri_pear_stream/hri_pear_stream/hand_keypoints.py` | `ca407f8a5b3e0808e0c06c0c3a4f3be99f566b78f0e16266ab841fba5959f5a1` | same |
| `data/dexhand_calib.json` | `ros2_ws/src/hri_pear_stream/hri_pear_stream/data/dexhand_calib.json` | `9a1f6d22ec1d41f67c5b5e223d1a17c0e2340de15d2a25fb06d493d7fa2655ec` | same |
| `data/dexhand_calib_mediapipe.json` | `ros2_ws/src/hri_pear_stream/hri_pear_stream/data/dexhand_calib_mediapipe.json` | `7efaebe19ee20adbf37852b9f4f84e547c1940e0b9dbb435b986f8b15f84c4f4` | same |
| `data/dexhand_robot.json` | `ros2_ws/src/hri_pear_stream/hri_pear_stream/data/dexhand_robot.json` | `73ee9eb70415da23c38393e3a12f73c02d7b6098dbe6a6eb34c566c9b4ff7507` | same |
| `mp_hand_pipeline.py` | `tools/mediapipe_npu/mp_hand_pipeline.py` | `317228b526832f2d9423109e50aa34e0f46e15888bdb02daf463fade0a622414` | same |

Patches (`hand_core.py`):

* `HandStream(gesture_labels=...)`: the labels the gesture layer accepts come
  from the caller; the default is upstream's `GESTURE_LABELS`.
* `_gesture_layer`: a gesture already active on the first `emit` starts from
  the tracked value instead of failing on `g.q is None`.

Not copied: `data/dexhand_calib_wilor.json` (WiLoR, not used) and
`thumb_safety.py` (imported by `hand_core` only when `thumb_3d` is set, which
this package never does).
