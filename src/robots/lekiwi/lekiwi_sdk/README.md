# lekiwi_sdk

ROS-free LeKiwi mobile manipulator SDK: omni base velocity control and odometry, composed with the SO-101 arm on one shared Feetech bus.

## Responsibility

- `OmniBase`: wheel velocity commands (rad/s), body-velocity IK (cmd_vel semantics), wheel-FK odometry, proportional overspeed scaling
- Pure, bus-free kinematics shared with the runtime base node: `body_to_wheel_velocities`, `wheel_deltas_to_body`,
  `integrate_pose`, and `wheel_position_deltas` -- wheel positions are raw 12-bit encoder angles that wrap once per
  revolution, so odometry must difference them along the shortest signed path, never directly
- `MobileManipulator`: one shared bus carrying `so101::Arm` (position motors) + `OmniBase` (wheel motors), with independently addressable scoped operations
- Base-only configuration (no arm calibration required)

## Prohibited

- ROS dependencies (links `feetech_sdk` + `so101_sdk`)
- Navigation planning (Nav2/PnC integration lives in the robot runtime)

## Python Bindings

`lekiwi_sdk_py` (pybind11): exposes `SimBus` (standalone simulated bus), `OmniBase`, `BaseGeometry`, `BasePose` and the
pure kinematics functions above for Python runtime nodes and tools.

## Testing

gtest cases (`test_omni_base`, `test_mobile_manipulator`) cover velocity round-trip, odometry across the encoder wrap,
overspeed scaling, subsystem independence, base-only mode and protected-feedback rejection. Odometry tests drive the
simulated bus with a manual clock (`bus.sim().use_manual_clock()` / `advance_clock()`), so motion depends on simulated
time, not on the read rate.

## Protection feedback

Transport success does not imply healthy motor feedback. OmniBase::read_wheels rejects
invalid, missing or protected wheel samples before updating either output array.
update_odometry returns false without changing pose or odometry history on those failures.
The caller must handle the failure: returning false does not itself stop wheel motors.
