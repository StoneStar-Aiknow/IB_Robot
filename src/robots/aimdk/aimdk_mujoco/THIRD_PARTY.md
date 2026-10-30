# Third-Party Sources

This file records third-party provenance, attribution and distribution limits.
Usage, configuration and verification instructions belong in [README.md](README.md).
This package contains no vendor binaries, message definitions or policy weights.
Its source license does not grant rights to third-party models or assets.

## Implementation References

`physics.py` adapts the numerical policy contract (joint ordering, observation
history, gains and default pose) documented by AgiBot's AimDK
`extra/mc-rl/x2_config/` and `rl/sim2sim/sim2sim_x2.py`.
The reference's root README declares Apache-2.0 and credits AgiBot (2026), with
its publishing attribution marked provisional. Its license also retains
`Copyright 2024 The Isaac Lab Project Developers`.
The IB-Robot adaptation uses a separate ROS control loop and simulation lifecycle.
No Unitree or Isaac Lab training code is included.

`bmimic.py` independently implements the numerical interface described by the
locally supplied AimDK `extra/x2_rl_deploy` example and ONNX metadata.
No vendor source, libraries, weights or motion tables are copied into this package.
No separate license was found within that example; its availability is not
treated as permission to redistribute its code or weights.

## Robot Assets

Robot meshes and MJCF come from the official public repository:

- Source: <https://github.com/AgibotTech/agibot_x2_urdf>
- Revision: `77f43eb0904dae4c48ccd9154fee824f8ffd4d38`
- License: MulanPSL-2.0; its unmodified text is retained in the prepared asset directory.
- Original files are unmodified and checked against official Git blob hashes.

Asset preparation retrieves only the public robot files and their license, not
policy weights. These assets are separate from the source distribution and remain
subject to their own license.

## Policy Weights

Policy weights and motion tables are not included in this source distribution.
Users must obtain them separately and verify authorization for their use and
redistribution. Do not commit or upload them without the necessary rights.
This package's license and ONNX Runtime's license do not license the weights.

## Runtime Dependencies

MuJoCo (Apache-2.0), ONNX Runtime (MIT), NumPy (BSD-3-Clause) and ROS 2 components
are separately installed dependencies, not vendored code. Their own license
terms apply.
