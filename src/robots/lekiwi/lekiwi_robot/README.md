# lekiwi_robot

The LeKiwi robot runtime: an SO-101 arm on a three-wheel omni base, sharing one
Feetech bus. It is the only package that knows LeKiwi's chassis; applications
reach it through the robot runtime contract (`robot_runtime.contract`) and
`/runtime/get_status`.

## Members and ownership

| Package | Owns |
| --- | --- |
| `lekiwi_robot` (this package) | Runtime profiles, `runtime.launch.py`, the base node, the FAST-LIO odom bridge |
| `lekiwi_sdk` | Pure chassis kinematics and wheel odometry, `OmniBase`, `MobileManipulator` |
| `lekiwi_hardware` | Thin ros2_control adapter: arm positions (IDs 1-6) and wheel velocities (IDs 7-9) on one bus |
| `lekiwi_description` | Assembled URDF and the MID-360 mount (`config/lekiwi_mid360_mount.yaml`) |
| `lekiwi_calibration` | MID-360 / front camera extrinsic calibration workflow |
| `feetech_sdk` | Feetech servo access, FTServo pinned at `06fd3356` (shared with SO-101) |
| `so101_description`, `so101_sdk`, `so101_motion` | The arm. The LeKiwi arm *is* an SO-101; nothing is forked, and the independence gate lists these three by name |

Arm limits come from the calibration file; the URDF only carries the mechanical
envelope. Wheels have no position limits, only `base.max_wheel_radps`.

## The chassis lives in the profile

The profile's `base:` section is the only source of chassis geometry:
`wheel_joints`, `wheel_radius`, `base_radius`, `max_wheel_radps`,
`control_frequency`, `odom_frame`, `base_frame`, `publish_tf`, and
`navigation_enabled_on_startup` (defaults to `false`). A missing key fails
startup with its name (`lekiwi_robot/base_profile.py`); there are no built-in
LeKiwi constants to fall back on.

`capabilities.base.cmd_vel` (`max_vx`, `max_vy`, `max_wz`, `staleness_s`) is
published in `RuntimeStatus`. The declared limits must be realizable: the
runtime computes the envelope from the SDK kinematics and rejects a profile
whose limits exceed it, naming the axis. For the shipped geometry that is
0.2657 m/s in x, 0.2301 m/s in y and 1.84 rad/s in yaw; the profiles publish
0.26 / 0.23 / 1.5.

## Base node (`lekiwi_robot/base_node.py`)

- `/cmd_vel` (`Twist`) -> `base_velocity_controller/commands` (wheel rad/s)
  through `lekiwi_sdk_py.body_to_wheel_velocities`; commands beyond
  `max_wheel_radps` are scaled down preserving direction. Commands are zeroed
  after `staleness_s` without a fresh one.
- `/joint_states` wheel positions -> `/odom`, plus `odom -> base_link` TF.
  Positions are unwrapped with `lekiwi_sdk_py.wheel_position_deltas`. When the
  runtime also runs FAST-LIO with its TF enabled, FAST-LIO owns that TF and the
  base node publishes `/odom` only, so the transform has one publisher.

### Mode and navigation gate

Two independent conditions admit a base command, and the runtime stop latch
overrides both:

| Layer | Set by | Prevents |
| --- | --- | --- |
| `base_navigation` mode (`allows_base: true`) | `/runtime/set_mode` | Arm and base controllers claiming the same joints |
| Navigation gate (`base.navigation_gate`) | `/motion_mode/set_navigation_enabled` (`SetBool`), acknowledged on `/motion_mode/base_navigation_enabled` | A planner driving the base before an operator armed it |

The gate starts closed and fails closed until the first `RuntimeStatus`
arrives. Enabling or disabling it zeroes the target and publishes a zero wheel
command before the acknowledgment; nothing is queued across a closed gate.
Both names are absolute. The former application handshake used the same names
relative (they coincide only in the root namespace) plus an enable topic,
which the runtime does not serve.

## Arm motion

`so101_motion` plans for the SO-101 arm alone. The runtime renders
`motion.model_xacro` (the SO-101 arm URDF) with this profile's hardware
arguments and hands it to move_group, which consumes the arm-only
`arm_joint_state_broadcaster`. Base-only profiles declare no `motion` section
and start no motion services.

## Supported profiles

Only `lekiwi_lidar` (base only: modes `idle`, `base_navigation`) and
`lekiwi_nav_grasp` (arm + base: adds `stream`, `trajectory`) are shipped. The
other LeKiwi configurations in `robot_config` still use the legacy
provider-less bring-up.

## Running standalone

```bash
ros2 launch lekiwi_robot runtime.launch.py profile:=lekiwi_lidar [simulated:=true]
```

`simulated:=true` keeps the production controller and hardware-adapter path
and replaces only the serial bus with the SDK's simulated transport.

Conformance and the independence gate:

```bash
scripts/verify_runtime_independence.sh runtime lekiwi_robot --profile lekiwi_lidar
scripts/verify_runtime_independence.sh runtime lekiwi_robot --profile lekiwi_nav_grasp
```

The base velocity, odometry, staleness and navigation-gate tests are in scope
and must execute, not skip. Use a fresh `ROS_DOMAIN_ID` (at most 232): a stale
runtime left on a shared domain answers for the one under test.
