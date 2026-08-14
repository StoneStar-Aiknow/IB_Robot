# FAST-Calib ROS 2 patch stack

Upstream repository: `https://github.com/TommyBrownson/FAST-Calib_Ros2.git`

Pinned commit: `7747dfc6109c04b4bf81d2e3661e41626c8392e1`

This commit contains the ROS 1 to ROS 2 migration without the later sample bag,
Jetson capture, projection, hardware-specific configuration, or generated cache
files. Apply the files in `series.txt` order with `git apply`.

Device intrinsics, target dimensions, scene ROIs, data paths, and calibration
results are runtime inputs and deliberately do not belong to this patch stack.
Absolute paths visible only on unchanged diff context lines are defaults from
the pinned upstream commit; this stack neither adds nor relies on those paths.

The provenance `patch_sha256` is the SHA-256 of `git diff --binary HEAD` after
applying the complete series with `git apply --index`, not a hash of any patch
file's serialized text. The index step ensures added files are included.

Patch responsibilities:

- `0001`: parameterize target detection while preserving the clean base defaults,
  fail when no plane is found, and recover a missing center only when exactly
  three measured centers form one unique configured L pattern.
- `0002`: add a fail-closed `observation_only` mode that writes exactly four
  camera and four LiDAR centers to `observation.yaml`, then exits normally.
- `0003`: require a near-right-angle measured L and the complete configured
  rectangle distance spectrum before recovering one missing LiDAR center.
- `0004`: replace per-cluster circle fitting with whole-plane circular-gap
  evidence and unique configured rectangle selection. This handles hole rims
  connected to board edges or scan texture while failing closed on missing or
  ambiguous patterns.

Multi-scene correspondence, joint fitting, holdout evaluation, approval, and
deployment remain in IB_Robot.
