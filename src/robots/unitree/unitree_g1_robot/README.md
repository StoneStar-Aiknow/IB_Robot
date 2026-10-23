# unitree_g1_robot — Unitree G1 runtime

The G1's independently deployable runtime unit: it serves the **public runtime
contract** (`/runtime_status`, `/runtime/set_mode`, `/runtime/get_status`,
`/runtime/stop`, `/joint_states`, `/imu/data`, the declared streaming command
channels and `/cmd_vel`) by bridging to the **vendor DDS tier** of the Unitree
SDK (`rt/lowstate`, `rt/arm_sdk`, `rt/dex3/*` and the `rt/api/*` RPC services)
through pure ROS 2 messages only. Generic IB-Robot packages bind the G1 exactly
as they bind SO-101, LeKiwi or the X2 and never see a vendor topic, api id,
FSM id or motor index.

Planning and rationale: `openspec/changes/unitree-g1-runtime-migration/`
(`design.md` decisions D1–D12 are referenced from the profile and this README).

## Responsibilities

- Translate contract commands into vendor DDS messages and vendor state into
  `RuntimeStatus`, the public interface description, `/joint_states` and
  `/imu/data`.
- Own the vendor mode machine: `stream` is the arm_sdk handover, `locomotion`
  is `SetFsmId(start)` confirmed by `GetFsmId`, and the G1 boots in `idle`
  only — non-idle modes are entered through the mode service, never claimed at
  startup.
- Expose the platform's own interaction surfaces — speech, the RGB LED, named
  arm actions, FSM postures — on neutral, project-owned endpoints.
- Compose the peripherals the vendor SDK does **not** publish (the D435 camera
  and the MID-360 lidar) from the shared drivers (design D8), and declare only
  the endpoints those drivers actually serve.

## What the wrapper exposes

Generic consumers see only the left column; the right column never leaves this
package.

| Public surface | Type | Platform mechanism |
|---|---|---|
| `/runtime/set_mode`, `/runtime/get_status`, `/runtime/stop`, `/runtime_status` | contract | `rt/api/{sport,arm,voice}/{request,response}` RPC (`SetFsmId` 7101, `GetFsmId` 7001, `SwitchToUserCtrl` 7110, `SetVelocity` 7105, …), `rt/lowstate.mode_machine` |
| `/joint_states` (43 joints, URDF names and order) | `sensor_msgs/JointState` | `rt/lowstate` motor states + `rt/lf/dex3/{left,right}/state`, projected through the vendor motor-index remap |
| `/unitree_g1/arm/commands`, `/unitree_g1/waist/commands` | `Float64MultiArray` | `rt/arm_sdk`: the shared-weight 17-joint `LowCmd` frame (weight = motor_cmd[29]'s q slot, design D4) |
| `/unitree_g1/hand_left/commands`, `/unitree_g1/hand_right/commands` | `Float64MultiArray` | `rt/dex3/{left,right}/cmd` `HandCmd` (Dex3, 7 joints per hand) |
| `/cmd_vel` | `geometry_msgs/Twist` | `SetVelocity` RPC re-issued at `command_duration_s` while a fresh Twist keeps arriving (locomotion only) |
| `/motion/execute_named` | `ibrobot_msgs/action/ExecuteNamedMotion` | `arm` service `ExecuteAction` (preset table checked against `GetActionList` at startup, design D7) and `SetFsmId` postures (sit, squat, stand_up) |
| `/speech/speak` | `ibrobot_msgs/srv/SpeakText` | `voice` service TTS call (neutral-only: the vendor call carries no language, priority or interrupt, so requests setting them are refused, not dropped) |
| `/led/set_pattern` | `ibrobot_msgs/srv/SetLedPattern` | `voice` service `SetRgbLed` (one static colour: only `PATTERN_SOLID` and neutral priority map) |
| `/imu/data` | `sensor_msgs/Imu` | republished from `rt/lowstate.imu_state` (frame_id stays unset until the URDF frame is confirmed, gate 6.1) |
| `/camera/head_front/**` | `sensor_msgs/Image`, `CameraInfo` | composed `realsense2_camera` driver (design D8) |
| `/livox/lidar`, `/livox/imu` | `PointCloud2`, `Imu` | composed `livox_ros_driver2` MID-360 driver (design D8) |

### Vendor QoS is part of the endpoint contract

Each `rt/*` endpoint carries the reliability/durability/depth the profile
records under `vendor.qos` (design D9). A BEST_EFFORT publisher cannot serve a
RELIABLE subscription — ROS 2 reports the incompatibility once and then
silently discards every message — so the QoS table is profile data, and when a
firmware changes one, the profile is corrected rather than the code. The
shipped values are assumptions from DDS high-rate-stream conventions, marked
`unverified` until measured with `ros2 topic info -v` on the unit (gate 6.1).

## DDS: CycloneDDS for the whole deployment, one interface

The vendor tier speaks CycloneDDS, so **the whole deployment — this runtime
and every application node that calls its services — must run
`rmw_cyclonedds_cpp`**: service calls do not interoperate across RMW
implementations (topic data flows, request/response does not), so the RMW is
set once by the deployment, never piecemeal. The launch does not set
`RMW_IMPLEMENTATION` itself — a runtime pinned alone would strand the
application nodes on the ambient implementation, and an environment action
inside an included launch leaks into whatever the parent launch does after
the include. Instead, in **real transport** the launch verifies that the
environment already carries `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` and fails
closed with a named error when it does not; **simulated transport** performs
no check (the mock serves whatever RMW the ambient environment provides).

The `network_interface` launch argument names the NIC the robot is wired to —
a deployment fact, never hardcoded, and CycloneDDS configuration only:

- non-empty: the launch writes a minimal CycloneDDS XML binding DDS to that
  one interface and points `CYCLONEDDS_URI` at it;
- empty (default): no `CYCLONEDDS_URI` is set; CycloneDDS keeps its default
  configuration (or one provided externally).

Find the interface with `ip link` (the vendor's own `setup.sh` carries an
`enp3s0` placeholder — do not trust it). CycloneDDS fails loudly when the named
interface does not exist, so a typo names itself at startup.

Do not mix RMW implementations across endpoints that must talk to each other:
every process that joins the robot's graph — this runtime, debugging tools,
recorders — must run `rmw_cyclonedds_cpp` with a matching `ROS_DOMAIN_ID`, and
`ROS_LOCALHOST_ONLY` must be unset or `0` on the robot-facing host. The
repository's test gates set `ROS_LOCALHOST_ONLY=1` on purpose; do not carry
that into the robot-facing shell.

## Prohibited

- **No controller manager, no ros2_control hardware component, no controller
  spawners.** The vendor owns balance and whole-body control (design D1/D11).
- **No `rt/lowcmd` publishing.** The low-level expert tier has no timeout or
  fail-safe protection at this bridge; the profile records it as
  `vendor.expert_joint_tier` with `enabled: false` and the runtime refuses a
  profile that enables it (`projection.expert_tier_gate`). Turning it on
  forfeits the vendor's balance control and is a deliberate deployment act.
- **No publishing vendor topics from outside the bridge.** Everything on
  `rt/*` bypasses the bridge's admission (mode gates, limits, staleness,
  stop latch); a second publisher defeats all of it.
- **No re-driving vendor-published sensors.** The bridge declares and
  republishes them on neutral endpoints; it does not relay them. The D435 and
  the MID-360 are *not* vendor sensors — the SDK publishes nothing for them —
  which is why the runtime composes the shared drivers for those two (design
  D8).

## Vendored material (in-tree, no overlay)

Unlike the X2/AimDK runtime, everything vendor-side is vendored in this
repository — there is no developer-supplied overlay to build or source, and
the independence gate runs without external dependencies:

| Package | Upstream | License |
|---|---|---|
| `unitree_hg`, `unitree_go`, `unitree_api` | `unitree_ros2` at upstream `668d1ec5` (pure `.msg` sources, no binaries) | BSD-3-Clause |
| `unitree_g1_description` | `unitree_ros` `g1_29dof_with_hand_rev_1_0`: the URDF and exactly the meshes it references | BSD 3-Clause |

Each carries a `PROVENANCE.md` recording the upstream commit and any local
rewrite (the description's mesh paths). The bridge imports vendor types only
through `vendor_gateway`, which re-exports them and refuses to start with a
named message when they are missing.

## Profiles

| Profile | Robot | End effector | Joints |
|---|---|---|---|
| `profiles/g1_29dof_dex3.yaml` | G1, 29 DoF | Dex3 dexterous hands (7 joints each) | 43 observable: 12 legs, 3 waist, 14 arms, 14 fingers |

The profile is the single source of runtime parameters: public joint
projection, modes and their vendor actions, command channels, declared
capabilities, stop policy, the vendor endpoint/QoS/api-id tables, and the
peripheral defaults. The `mode_machine` generation (5) is checked against the
first `LowState` at startup — a unit from another generation (15/16/18, 5010
wrists) faults instead of running against a URDF that does not describe it
(design D10).

### What the profile deliberately does not declare

- **No public model.** No URDF is passed to `build_description`, so no `model`
  block is exported and consumers needing LeRobot normalization metadata fail
  closed on the G1 (design D6). `robot_state_publisher` renders the vendored
  URDF for TF only — the D435/MID-360 mounts and RViz — and policy inference
  and dataset recording wait for `refactor-embodiment-control-bindings`.
- **No `gripper.1d`.** Dex3 is a 7-joint-per-hand dexterous hand
  (`hand.multi_joint`); a Dex3 finger is not a scalar aperture and no openness
  value is synthesised from the 7 measured positions (design D3).
- **No trajectory / FK / IK / move-to capabilities** — the vendor tier this
  change bridges exposes no joint-trajectory endpoint.
- **No teleoperation** — the public teleoperation projection binds exactly one
  arm group and one gripper; the G1 has two of each, and no public model.
- **No `power.state` / `interaction.audio_playback`** — which vendor topic
  publishes the pack state is unverified, and the voice service plays named
  streams rather than a documented PCM topic (both gate 6.1). They return with
  their sources confirmed on hardware.
- **No `base.odom`** — the fast-lio lidar odometry chain stays disabled until
  the MID-360 network facts are confirmed (gate 6.1).
- **No LED priority levels** — `SetRgbLed` carries no priority parameter.

### Unverified values name their gate

Every value that cannot be established from the vendored SDK headers, IDL and
URDF carries an `unverified` note naming the hardware gate that will confirm
it (tasks 6.1–6.4); a profile contract test holds the shipped profile to that
marking. Nothing unverified is silently presented as fact.

## Not exposed here

- **Embodied primitives and skills.** `wave`, `shake_hand`, the postures and
  speech as skill primitives would change the shared primitive contract and
  the skill executor — code SO-101 and LeKiwi execute. That coordination is
  specified in the OpenSpec change (§5); the base G1 deployment
  (`unitree_g1.yaml`) ships with `embodied.enabled: false`, and the optional
  `unitree_g1_skills.yaml` that would reuse `runtime_named_motion` is
  deliberately not shipped in this stage. The capabilities themselves are
  available now through `/motion/execute_named` and `/speech/speak`.
- **Voice ASR/TTS services.** The vendor voice RPC carries an ASR api id, but
  no ASR surface is bridged or verified; the audio contract itself waits on
  gate 6.1.

## Status

The runtime is implemented and tested against `unitree_vendor_mock`, which
speaks the real vendored IDL on the topic names and with the QoS the profile
declares. The package suite is 206 tests; the capability-scoped conformance
suite passes 9/9 in simulated transport.

**No hardware contact yet.** Everything that moves the robot — mode switches,
the arm_sdk handover, streaming, locomotion, stop latencies — is a
user-executed gate recorded in the OpenSpec change (tasks §6), and the profile
values those gates will measure (ramp rates, gains, the velocity envelope, the
stop bounds, the endpoint QoS) are placeholders, not measurements.

## Tests

```bash
source .shrc_local
# The bridge tests isolate themselves on domain 148 and the root conftest
# allocates a per-process domain, but keep the run off shared domains anyway.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/robots/unitree/unitree_g1_robot/test -q

# Formal acceptance runs through colcon from the workspace source tree:
colcon test --base-paths src --packages-select unitree_g1_robot
colcon test-result --verbose
```

## Running against the mock

```bash
ros2 launch unitree_g1_robot runtime.launch.py profile:=g1_29dof_dex3 simulated:=true
```

The vendor mock stands in for the robot's DDS tier and the contract layer's
synthetic publishers stand in for the composed peripherals, so the full public
surface is live without hardware. `robot_state_publisher` also runs, so RViz
can display the mock's joint states. To exercise the DDS wiring itself, run
the stack on CycloneDDS and name an interface — the launch then binds
CycloneDDS to it:

```bash
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
ros2 launch unitree_g1_robot runtime.launch.py profile:=g1_29dof_dex3 \
    simulated:=true network_interface:=lo     # DDS smoke: CYCLONEDDS_URI generated and honoured
```

The conformance suite runs against the mock unchanged:

```bash
ROS_DOMAIN_ID=149 \
CONFORMANCE_LAUNCH="ros2 launch unitree_g1_robot runtime.launch.py profile:=g1_29dof_dex3 simulated:=true" \
CONFORMANCE_PROFILE="$(ros2 pkg prefix unitree_g1_robot)/share/unitree_g1_robot/profiles/g1_29dof_dex3.yaml" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/robot_runtime/test/test_conformance.py -q -rs
```

## Running against the robot (user-executed gates)

Nothing in this section has been executed against hardware; it is the order
the OpenSpec change prescribes (tasks §6). Keep the robot on a gantry until
the locomotion gate.

### 1. Network

- Export the RMW first — the launch fails closed in real transport without
  it (see *DDS* above):

  ```bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  ```

- Wire the host to the robot's Ethernet port and identify the interface
  (`ip link`). Launch with `network_interface:=<nic>`; confirm the nodes came
  up with the generated `CYCLONEDDS_URI` (a bad interface name fails loudly
  at startup).
- `ROS_LOCALHOST_ONLY` must be unset or `0`, and every robot-facing process
  must run `rmw_cyclonedds_cpp` (see *DDS* above).
- Prove the link with the vendor's own topics before touching this runtime:

  ```bash
  ros2 topic list | grep '^/rt/'                          # vendor DDS graph visible
  ros2 topic echo /rt/lowstate --qos-reliability best_effort --once
  ```

  No `/rt/*` topics means power, cable, interface or RMW — not this package.

- Pass the deployment peripherals fragment with the MID-360's real network
  facts (the profile's are placeholders; the stock
  `user_config_path` template it carries needs no override):

  ```bash
  ros2 launch unitree_g1_robot runtime.launch.py profile:=g1_29dof_dex3 \
      network_interface:=<nic> peripherals:=<fragment.yaml>
  ```

  ```yaml
  # fragment.yaml
  peripherals:
    - type: lidar
      name: chest_front
      lidar_ip: <the MID-360's address>
      host_ip: <this host's address on that link>
  fast_lio:
    enabled: false   # flips to true only with gate 6.1 closed and a params_file
  ```

### 2. Start, and confirm the platform accepted the runtime

Start the robot per the vendor's guide, then:

```bash
ros2 service call /runtime/get_status ibrobot_msgs/srv/GetRuntimeStatus
```

`lifecycle: ACTIVE` means the first `LowState` arrived, both Dex3 state streams
are publishing, the reported `mode_machine` matches the profile (5), the RPC
peers matched, and the arm service answered `GetActionList`. Anything else
names its cause in `faults` — a generation mismatch is `FAULTED` and never
heals, a quiet Dex3 stream is named per side. Then check the read-only surface
before commanding anything:

```bash
ros2 topic hz /joint_states
ros2 topic echo /imu/data --qos-reliability best_effort --once
ros2 topic echo /runtime_status --once
```

Gate 6.1 closes here: record the measured endpoint QoS and the real RPC topic
names back into the profile.

### 3. Motion, in this order (gates 6.2–6.4)

1. **Modes** — `idle → stream → idle → locomotion → idle` through
   `/runtime/set_mode`; `stream` must answer only after the arm_sdk handover
   and `locomotion` only after `GetFsmId` reports the start FSM. Ask for a
   transition the vendor refuses and confirm the vendor code is surfaced.
2. **Upper-body stream** — in `stream`, publish the *measured* arm pose on
   `/unitree_g1/arm/commands` (no motion expected), then a single joint
   ±0.1 rad at 50 Hz; watch `rejected_counts` in `/runtime_status`. Stop
   publishing: the joints must hold their latched targets, not drift and not
   return to zero. Repeat for the waist and hand channels.
3. **Stop** — `/runtime/stop HOLD` while streaming: the latch engages, `stream`
   is refused, `idle` clears. Record `cancel_latency_s` / `idle_latency_s`.
   `TORQUE_OFF` while standing must be refused (fail-closed to sit/squat).
4. **Locomotion** — off the gantry, in `locomotion`, publish `/cmd_vel` at a
   low velocity, then stop publishing: the runtime's staleness behaviour means
   the base may continue for up to `command_duration_s` (1 s) — see known
   limitations. Leaving locomotion zeroes the velocity explicitly and the
   vendor must confirm it.
5. **Write the numbers back** — the profile's `runtime.stop` bounds, ramp
   rates, gains and the velocity envelope are placeholders until measured;
   replace them with the observed values, with margin, and tick the §6 items
   in the OpenSpec tasks.

### 4. Same suite, real transport

Once 1–3 pass by hand, the contract suite runs unchanged against the robot
(the same command as the mock section with `simulated` omitted). Be aware of
what it commands: the streaming check moves arm joints and the hold check
expects them to stay there.

## Known limitations

- **Process signals do not stop the robot.** SIGINT/SIGTERM exit the spin
  loop without running the stop sequence — no zero velocity, no weight ramp
  — and the rclpy teardown occasionally prints an `RCLError` while executor
  threads are still inside callbacks. The process still exits; the message is
  noise, not a failed stop. `/runtime/stop` is the supported stop.
- **A quiet `/cmd_vel` is not zeroed explicitly.** When the Twist stream stops
  the bridge stops re-issuing, but the last `SetVelocity` keeps its effect for
  its own duration, so the base may keep moving for up to `command_duration_s`
  (1 s) after publishing stops. Mode switches and stops do send the explicit
  zero.
- **`power.state` and `interaction.audio_playback` are undeclared** until
  their vendor sources are confirmed on hardware (gate 6.1).
- **The fast-lio chain is wired but dormant.** `fast_lio.enabled` stays false;
  enabling it additionally requires a `params_file` in the configuration, and
  the `fast_lio_odom_bridge` executable arrives with the fast-lio enablement
  follow-up (the LeKiwi pattern the launch follows names this package as the
  bridge host).
- **The shipped lidar network values are placeholders.** The MID-360 entry
  carries the stock `user_config_path` template but conservative IPs; real
  transport needs the deployment fragment shown above.
