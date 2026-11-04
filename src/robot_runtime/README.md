# robot_runtime

Robot runtime contract layer: the canonical capability vocabulary, the runtime
facade that serves the contract over a ros2_control execution stack, the mock
runtime reference implementation, launch-time capability reconciliation, the
runtime profile loader, and the parameterized conformance suite every runtime
must pass.

## Responsibility

- **Public interface schema** (`schemas/interface_description.schema.json`):
  versioned effective ROS endpoints, identity/digest, camera source profiles,
  QoS and independently observed states. See `docs/robot_interface_schema.md`.
  `runtime_interfaces` exports/validates YAML; `/runtime/get_status` exposes
  the same document in `RuntimeStatus.interface_description_json`.

- **Contract surface** (`contract.py`): canonical, runtime-neutral topic and
  service names (`/runtime_status`, `/runtime/set_mode`, `/runtime/stop`,
  `/motion/compute_fk`, `/motion/move_to_pose`, ...), lifecycle and outcome
  vocabularies. Generic consumers depend on these names only.
- **Capability vocabulary** (`capabilities.py`): frozen, extension-only
  registry (`joint.*`, `gripper.*`, `base.*`, `motion.*`, `runtime.stop`,
  reserved X2 names) plus the launch-time reconciliation check.
- **Runtime profile** (`profile.py`): the single source of runtime parameters
  (identity, modes → controller sets, declared capabilities with parameters,
  stop latency bounds, command channels, trajectory endpoints, hardware
  components). Missing required keys fail fast naming the key and the file.
- **Mode model** (`modes.py`): named controller activation sets with
  validated transitions and per-channel rejected-command counters. It does
  not arbitrate commands: "one command source per joint group" is enforced
  by the execution stack (controller_manager activation), never re-implemented here.
- **Runtime facade** (`facade_node.py`): non-real-time node that maps
  `SetRuntimeMode` onto `controller_manager/switch_controller`, publishes
  `RuntimeStatus` (1 Hz + on change), and implements `StopRuntime` with the
  contract's ordered guarantees — cancel trajectory goals → idle activation
  set → (TORQUE_OFF) hardware component deactivation — measuring each step.
- **Mock runtime** (`mock_runtime_node.py`): full-contract in-memory
  implementation with deterministic planar-chain kinematics and a base
  profile. Baseline for the conformance suite and the provider used by the
  core-only independence gate.
- **Reconciliation** (`wait_for_runtime.py`): replaces `wait_for_controllers`
  as the upper-layer start condition; fails fast naming missing capabilities.
- **On-demand subscriptions** (`on_demand.py`): subscriptions live on a node no
  executor spins, so samples wait in KEEP_LAST queues until a reader drains
  them, bounded by the reader's declared depth. Each drained sample carries
  the time the middleware received it — not the time it was taken — so a
  freshness gate sees the real message age. A sample whose receive time the
  middleware did not record is delivered with `None` and must be refused by
  any freshness gate: pretending it was just received would pass stale data.
  The reader node follows the launch's namespace and topic remaps like any
  node; only its name is pinned, so it never takes its owner's.
- **Launch support** (`launch_support.py`): profile path resolution
  (`$(find)` / `$(env)`), xacro rendering, controller_manager parameter
  files, spawners derived from the mode table, and the shared stack
  composition used by `<robot>_robot` launch entries.
- **MuJoCo embodiment** (`mujoco.py`, in `launch_support.py`): the provider's
  single `mujoco_ros2_control` process — robot+scene model composition,
  simulator version checks, the CameraPlugin parameter tree, and, for
  scenes with dynamic objects (issue #195), a
  `FreeJointStatePublisherPlugin` instance publishing every dynamic body's
  live pose on `/sim/<scene>/free_joint_states` for the scene task's
  object observation.
- **Simulated sensors** (`gazebo.py`): on the Gazebo platform the runtime
  stands in for the declared camera, lidar and IMU peripherals — it renders
  the sensors into the model at the peripheral's mount, bridges the gz
  streams onto the advertised ROS endpoints, and serves those endpoints
  through small normalizers: the finite-point cloud filter (Gazebo reports
  no-hit beams as ±inf) and the IMU settle filter (the pre-settle zero
  window and landing impulse never reach the endpoint). The derived scan
  converter composes on simulation time when its source runs there.
- **Conformance suite** (`test/test_conformance.py`): the executable contract.

## Prohibited

- Carrying joint commands or implementing trajectory interpolation,
  arbitration, or watchdogs (ros2_control controllers own execution).
- Robot-specific logic of any kind (that lives in `<robot>_robot` runtimes).
- Depending on a motion-planning framework (runtimes implement the motion
  services; this package only names them).

## Running the conformance suite

Headless against the mock runtime (spun in-process):

```bash
python3 -m pytest src/robot_runtime/test/test_conformance.py -q
```

Against a real runtime through its own launch entry (design D8 — the suite
never substitutes a stand-in stack):

```bash
CONFORMANCE_LAUNCH="ros2 launch so101_robot runtime.launch.py profile:=so101_single_arm simulated:=true" \
CONFORMANCE_PROFILE=<path> \
python3 -m pytest src/robot_runtime/test/test_conformance.py -q
```

Tests are scoped by the capabilities the target declares in `RuntimeStatus`;
every assertion message names the requirement and the observed deviation.

### Conformance start pose (`conformance.start_positions`)

Arm-moving motion tests start from a declared pose (issue #194): before each
of them the suite sends one trajectory to the profile's optional top-level
`conformance.start_positions` mapping (`{joint: radians}`, validated by
`load_profile` — known joint names, finite numbers) and asserts arrival
within the suite's goal tolerance. A profile that does not declare it gets
today's behaviour (no reset, no guessed default); a runtime without a
trajectory capability is never reset. Declared joints go to their values,
every other arm joint holds its current position. The key exists so physics
platforms with a work surface at the mount plane (MuJoCo, Gazebo) can pick a
start pose from which the suite's blind `before + delta` motions stay clear
of the surface — the value is the robot owner's choice, never hardcoded in
the suite.

## Independence gates

`scripts/verify_runtime_independence.sh` (repo root) automates both
directions of the robot-runtime-packaging acceptance: `runtime <robot>_robot`
builds the closure in isolation and certifies it, `core` builds the generic
set with every robot package excluded and composes against the mock runtime.

## Stop guarantees (facade)

| Policy | Guarantees, in order | Reached via |
|---|---|---|
| `HOLD` | cancel in-flight trajectories and reject streaming; idle activation set with joints holding; lifecycle `STOPPED`, latched | `<action>/_action/cancel_goal` (cancel-all) → `switch_controller` (STRICT) |
| `TORQUE_OFF` | `HOLD` guarantees, then joint torque released | + `set_hardware_component_state(inactive)` → adapter `on_deactivate` → SDK safe-stop |

The latch clears only on `SetRuntimeMode("idle")`; after `TORQUE_OFF` the
facade reactivates the hardware components before clearing.

## Simulation platform behaviours (robot-agnostic)

Three profile-level behaviours are part of this package's public surface and
apply to every robot runtime, not only one robot:

- **Per-platform peripheral exclusion (issue #205 D3).** A peripheral entry
  may declare `simulation.<platform>.enabled: false` for a physics platform
  (`gazebo`, `mujoco`). On that platform the device is not launched, not
  simulated and not advertised (`/runtime_status`, the interface
  description, capabilities); on hardware and `sdk` executions the key is
  ignored and the inventory passes through. `simulation` and each
  `simulation.<platform>` section must be a mapping when present — an
  explicit `null` (or any non-mapping) is an error naming the peripheral
  and the key, because through the layer merge it would silently erase the
  base's exclusion flags and resurrect an excluded device. An exclusion
  survives a same-driver deep merge, carries over a driver change, and is
  lifted only by an explicit `simulation.<platform>.enabled: true`.
- **MJCF attachments (issue #205 D2).** A profile's
  `simulation.mujoco.attachments` merges sub-model MJCFs (e.g. an arm from
  another package's template) into the host robot model under a named
  parent body. The schema is validated at MJCF render time with unknown
  keys as errors (the platform-neutral profile loader does not gate on it;
  a bad attachment fails closed at the first render). The mount pose is the
  rendered URDF's fixed-joint chain `urdf_parent_link → urdf_child_link`
  relative to `parent_body` (the parent may be nested anywhere in the host
  tree and must be uniquely named); the attachment's meshes and named
  `<default>` classes are namespaced, every cross-model name clash is an
  error, and the host's `<compiler>`/`<option>` govern. Cameras, lidars and
  IMUs are injected after the merge, so a `parent_frame` may name an
  attachment body. Profiles without attachments render byte-identical
  output.
- **Initial joint positions (issue #205 D4).** On both physics platforms
  the render sets the ros2_control `initial_value` of the position state
  interface for every joint in `hardware.reset_positions`, so the robot
  spawns at — and, with `mujoco_ros2_control`, the position actuators'
  `ctrl` also hold — the declared pose instead of zero. A reset joint
  missing from ros2_control, a non-finite value or an existing
  `initial_value` conflict is an error. Profiles without
  `reset_positions`, and the hardware/`sdk` render paths, are unchanged.
