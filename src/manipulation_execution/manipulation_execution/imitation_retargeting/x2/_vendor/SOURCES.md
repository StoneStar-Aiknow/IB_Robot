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
  crossing ±π stepped by 2π (`test_x2_followme.py` pins this).

Not copied: the rest of `smplx2robot` (cli, export, delivery, safety, preview,
compare, verify), which the streaming path does not import.
