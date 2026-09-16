# lekiwi_sdk

ROS-free LeKiwi mobile manipulator SDK: omni base velocity control and odometry, composed with the SO-101 arm on one shared Feetech bus.

## Responsibility

- `OmniBase`: wheel velocity commands (rad/s), body-velocity IK (cmd_vel semantics), wheel-FK odometry, proportional overspeed scaling
- `MobileManipulator`: one shared bus carrying `so101::Arm` (position motors) + `OmniBase` (wheel motors), with independently addressable scoped operations
- Base-only configuration (no arm calibration required)

## Prohibited

- ROS dependencies (links `feetech_sdk` + `so101_sdk`)
- Navigation planning (Nav2/PnC integration lives in the robot runtime)

## Python Bindings

`lekiwi_sdk_py` (pybind11): exposes `SimBus` (standalone simulated bus), `OmniBase`, `BaseGeometry`, `BasePose` for Python runtime nodes and tools.

## Testing

14 gtest cases covering velocity round-trip, odometry, overspeed scaling, subsystem independence, base-only mode.
