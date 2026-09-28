# lekiwi_robot

The LeKiwi robot runtime: SO-101 arm on an omni-wheel base sharing one
Feetech bus (`robot-runtime-packaging` spec, design D1).

## Members

`lekiwi_description`, `so101_description` (arm meshes), `lekiwi_sdk` /
`so101_sdk` / `feetech_sdk`, `lekiwi_hardware` (thin ros2_control adapter,
scoped bus operations), `so101_motion` (the arm is an SO-101; MoveIt consumes
the arm-only joint state broadcaster), `robot_runtime`, `ibrobot_msgs`, and
the **base node** in this package.

## Base node (`lekiwi_robot/base_node.py`)

Relocated from the generic `robot_navigation` cmd_vel bridge:

- `/cmd_vel` → `base_velocity_controller/commands` using the SDK's pure
  kinematics (`lekiwi_sdk_py.body_to_wheel_velocities`), accepted only while
  navigation is enabled, the runtime mode allows base commands
  (`base_navigation`), and no stop is latched; zeroed on staleness.
- `/joint_states` wheel feedback → `/odom` + TF (`lekiwi_sdk_py.wheel_deltas_to_body`).
- Navigation gating (`base.navigation_gate`): `/motion_mode/set_navigation_enabled`
  (`SetBool`) clears pending commands and stops the base before acknowledging;
  `/motion_mode/base_navigation_enabled` heartbeat. Names unchanged for
  navigation stacks; only the implementation moved.

## Running standalone

```bash
ros2 launch lekiwi_robot runtime.launch.py profile:=lekiwi_lidar [simulated:=true]
```

Modes: `idle`, `stream`, `trajectory`, `base_navigation` (see `profiles/lekiwi_lidar.yaml`).
Conformance: as for `so101_robot`, with `lekiwi_robot` / `lekiwi_lidar`; the
base and navigation-gating tests are in scope.
