# unitree_g1_robot — Unitree G1 runtime

Operator runbook: [宇树 G1 启动与操作命令](../../../../docs/unitree_g1_operation_commands.md)
archives the supervised Gateway/Web startup, Gateway discovery, CLI use, and timeout boundaries.

## Preset-only development and supervised validation

`g1_29dof_arm_presets` is a separate surface for the 23 statically listed G1
firmware preset actions requested by the operator (IDs 1, 11–13, 15, 17–30, 33,
34, 36 and 99). It has no hand subscriptions, hand writers, arm-stream writer,
locomotion commands or posture actions, and starts in idle. Physical preflight
uses a loco client for read-only FSM queries. The application configuration is
`unitree_g1_arm_presets`; its catalog exposes the 23 static
`runtime_named_motion` skills with stable names, Chinese `display_name` and
`tags: [preset, unitree_action_id_<N>]` (`release_arm` also carries
`control_ownership`). `arms_horizontal` keeps its existing ID 15 binding
(`both_hands_up`); `release_arm` is ID 99. These 23 are the static preset set:
the full `g1_29dof_dex3` profile keeps its existing 9 `vendor.arm_actions`
mappings, and the runtime advertises only the `GetActionList`-confirmed subset,
which may be smaller or empty.

Operator on-site confirmation on 2026-11-10 covers two labels in that table. ID
15 `arms_horizontal` (firmware `both_hands_up`, both hands raised to the chest)
keeps its existing `双臂平举` display and its ID 15 mapping. ID 17 is a two-hand
applause gesture: the public skill key is `clap` (matching the full dex3
profile), while the firmware string `clamp` remains the exact `action_name` and
`action_id` 17, so exact `GetActionList` matching and filtered dispatch do not
change. Only the catalog display metadata describes applause: `display_name:
鼓掌`, and the summaries/aliases use clap/applaud instead of clamp/hard grip, so
a planner never receives the old clamping meaning. Workflows written against the
old preview must replace `skill_name: clamp` with `clap`. This rename changes the
registry digest; old previews are rejected, so reload the registry and prepare a
new preview instead of reusing the old goal. This record is operator-reported
on-site confirmation, not a camera recording and not an Agent real-robot test; no
G1, Gateway, Web, or SDK process was started, stopped, or restarted to change
these labels, and no action was dispatched.

The catalog display tags are generated from this profile, not hand-written.
The offline tool lives outside this package because the robot package must not
write an agent asset: `python3 tools/g1_preset_presentation.py --check` (add
`--write` to regenerate) rewrites only `capability.tags` in
`src/agent/skill_catalog/config/skills/<skill>/manifest.yaml`. The profile's
`vendor.arm_actions` stays the only source of the numbers, and
`tools/test_g1_preset_presentation.py` fails when a profile id and the shipped
tag drift apart. The tags are display-only: they change no binding, recovery
policy, FSM rule or admission result, and the CLI/Web show them as badges
without ever selecting a firmware action from a number.

**Catalog visibility is not readiness or admission.** The catalog lists all 23
static entries so the Web UI and planner can describe them, but the current
Gateway/catalog contract has no per-action `ready` flag. The runtime advertises
only the `GetActionList`-confirmed subset; it can be smaller than 23 and can be
empty when the query times out, fails, or the firmware omits an action. An empty
subset never falls back to the 23-entry static table. The runtime enforces the
two firmware limits reported by the operator's GET-only query before
`ExecuteAction`:

- ID 1 `turn_back_wave` requires FSM 500/501. The current deployment's
  `expected_fsm_id` is 802, so admission rejects it with
  `NAMED_MOTION_PLATFORM_NOT_READY` and the reason before dispatch; the runtime
  never calls `SetFsmId` to make it available.
- IDs 20/21/28/29/30/36 report `mode_machine` 5/6. Admission reads the newest
  LowState `mode_machine` and rejects when the observed value is outside that
  list or feedback is stale.

No other firmware limit is encoded. The actionable "not available now" signal is
the pre-submit rejection and its reason, not a catalog `ready=false`.

**Evidence scope.** The operator reports having manually exercised all 23
firmware actions (operator-reported). The Agent has one earlier software-path
success for action 15; the camera was disconnected at that time, so physical
completion was not confirmed. This README/catalog does not claim Agent
real-robot validation of the 23 actions.

**Firmware naming boundary.** The `action_name` strings are the exact names the
operator's 2026-11-07/09 GET-only `GetActionList` returned. Startup matches
id+name exactly, so a different firmware that renames an action advertises
fewer motions (fail-closed); refresh the profile from a new report rather than
matching by id.

**Physical execution remains disabled by default.** The per-run confirmation is
a launch argument, not a persisted profile file. The shipped profile and the
currently running deployment still carry the earlier single-action (15)
profile; do not change the global `expected_fsm_id` just to make
`turn_back_wave` admissible. From the source workspace:

```bash
source .shrc_local
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=unitree_g1_arm_presets use_sim:=false \
  runtime_network_interface:=<robot-facing-nic> \
  preset_operator_confirmed:=true \
  preset_expected_fsm_id:=<observed-fsm-id>
```

`preset_operator_confirmed` defaults to `false` and fails closed. The launch
requires a non-empty, non-loopback `runtime_network_interface` and an expected
FSM id (the explicit argument, or an audited
`vendor.operator_validation.expected_fsm_id` in the profile). It passes those
three facts to the node as ROS parameters for this run only: no temporary or
run-scoped profile is written, and the node re-checks the same gate so a stale
`confirmed: true` profile cannot bypass the per-run confirmation. The source
profile is never rewritten. Launch and node enforce the boundary. Before
dispatch the runtime reads service ownership, FSM, fresh feedback and the
per-action limit for each independent request. A successful RPC completion
permits a later request; concurrent dispatch is rejected. Timeout, transport
error or exception latches an unknown outcome and prevents further dispatch
without claiming that the robot stopped. No automatic retry is made.
`release_arm` (99) is a control-ownership operation, not a normal gesture; it is
not inserted automatically into recovery/finishing and is sent only on an
explicit workflow request. Its `never_retry` policy and description are
operational guidance, not a provable semantic guarantee that a planner cannot
propose it. This is not a verified software emergency stop; the on-site operator
owns the confirmed physical stopping method.
The offline launch requires simulated transport and explicitly binds both mock
and runtime SDKs to loopback on domain 148. No Dex3 URDF or peripheral driver is
launched for this surface. `runtime.stop` is not advertised as a capability:
the stop service closes local admission but reports physical stop unverified,
and the latch cannot be cleared. A successful mock action response is software
integration evidence, not real-robot FSM-802 compatibility or stop acceptance.

Offline entry (after environment setup and build):

```bash
source .shrc_local
ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 ros2 launch unitree_g1_robot runtime.launch.py \
  profile:=g1_29dof_arm_presets use_sim:=true network_interface:=lo
```

This launches the mock/runtime pair only. The application configuration binds
the named actions through the Gateway when the embodied stack is composed; it
does not provide a direct-SDK shortcut. The supervised physical path was used
for one Gateway request for action 15 that reported software success; the camera
was disconnected, so physical completion was not confirmed. A later
operator-reported statement covers all 23 actions, but it is not Agent
real-robot evidence. No action 99 was sent by this Agent. This observation does
not establish software cancellation, stop, or physical completion of any preset.

> Arm-control correction (2026-10-28): stream entry now uses read-only service
> ownership and FSM checks, never `SwitchToUserCtrl` (a full-body takeover).
> `ai_sport` must already run and `g1_arm_example` must already be stopped;
> this runtime does not switch those services. The arm frame includes waist
> joints. Stop/exit can publish zero weight but cannot acknowledge remote
> handover: admission stays closed (the stop latch) and the unacknowledged
> handover is reported — `ARM_HANDOVER_UNCONFIRMED` in `faults` and in the
> response message. A stream→idle switch still succeeds, so the public mode
> contract holds; `SetRuntimeMode("idle")` then clears the latch and the
> warning, and stream re-entry always runs the full admission again.
> Real-hardware streaming acceptance remains blocked. See `ARM_CONTROL_AUDIT.md`
> and `test/BRIDGE_COVERAGE.md`; historical claims below are not acceptance evidence.

The G1's independently deployable runtime unit: it serves the **public runtime
contract** (`/runtime_status`, `/runtime/set_mode`, `/runtime/get_status`,
`/runtime/stop`, `/joint_states`, `/imu/data`, the declared streaming command
channels and `/cmd_vel`) by bridging to the **vendor DDS tier** of the official
Unitree SDK (`unitree_sdk2py`: `rt/lowstate`, `rt/arm_sdk`, `rt/dex3/*` and the
`rt/api/*` RPC services). The SDK is reached only through the ROS-free
`unitree_g1_adapter` package — `vendor.py` is the single module that imports a
vendor symbol — and this node's public surface is pure ROS 2 messages. Generic
IB-Robot packages bind the G1 exactly as they bind SO-101, LeKiwi or the X2 and
never see a vendor topic, api id, FSM id or motor index.

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
| `/runtime/set_mode`, `/runtime/get_status`, `/runtime/stop`, `/runtime_status` | contract | `sport`, `arm`, `voice`, `robot_state` RPC; stream uses service/FSM queries and weight frames, not full-body takeover |
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

## DDS: two layers, two bindings

Two DDS layers meet in this package and they are configured separately:

- **the vendor tier**, inside the official SDK. The SDK owns its own
  cyclonedds-python participants, so that layer is independent of the ROS RMW
  and keeps working under `rmw_cyclonedds_cpp` or `rmw_fastrtps_cpp`. The
  launch neither sets nor verifies `RMW_IMPLEMENTATION`; it only tells both
  nodes which domain and NIC to bind (the parameters below).
- **the deployment's own ROS graph**, between this runtime and every node that
  calls its services. That graph is the deployment's choice of RMW: pick one
  implementation for all of it and keep a matching `ROS_DOMAIN_ID`, because
  service calls do not interoperate across RMW implementations (topics flow,
  request/response does not). The runtime serves plain ROS 2 services and
  topics, so it works under either implementation — it only has to be the same
  one as its callers'.

The `network_interface` launch argument names the NIC the robot is wired to —
a deployment fact, never hardcoded. It has two separate jobs:

- **vendor tier binding**: in real transport the launch passes the name to the
  runtime and mock nodes as `vendor_network_interface`, and the SDK builds its
  own CycloneDDS domain configuration from it (`Domain(id, config)`), which
  overrides `CYCLONEDDS_URI` — so the environment variable is *not* how the
  vendor tier is isolated. Simulated transport never uses the argument: it
  binds vendor domain `148` on `lo`, and both nodes refuse to start unless the
  (domain, NIC) pair is consistent — a nonzero (isolated) domain with loopback,
  or domain `0` (the robot's production domain) with the explicit robot-facing
  NIC. Real transport without `network_interface` therefore fails closed before
  any vendor channel opens, instead of letting the SDK autodetermine a NIC.
- **the deployment's own ROS participants**: when non-empty, the launch writes
  a minimal CycloneDDS XML binding them to that one interface and points
  `CYCLONEDDS_URI` at it; when empty, no `CYCLONEDDS_URI` is set and CycloneDDS
  keeps its default (or an externally provided) configuration.

For the arm-preset physical path the launch takes the robot-facing NIC from the
per-run `runtime_network_interface` argument instead (non-empty, not `lo`); a
disagreeing `network_interface` or profile
`vendor.operator_validation.network_interface` is rejected as a conflict.
`network_interface` keeps its existing driver meaning everywhere else.

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

## Vendor material (official SDK + in-tree description, no overlay)

Unlike the X2/AimDK runtime, everything vendor-side is available without a
developer-supplied overlay, and the independence gate runs without external
dependencies:

| Material | Upstream | License |
|---|---|---|
| `unitree_sdk2py` | `unitreerobotics/unitree_sdk2_python` pinned at `814556d1` (installed at setup time by `scripts/setup/install_unitree_sdk.sh`; not committed) | BSD-3-Clause |
| `unitree_g1_description` | `unitree_ros` `g1_29dof_with_hand_rev_1_0`: the URDF and exactly the meshes it references | BSD 3-Clause |

The description carries a `PROVENANCE.md` recording the upstream commit and
any local rewrite (the mesh paths). The three formerly vendored ROS message
packages (`unitree_hg`, `unitree_go`, `unitree_api` from `unitree_ros2`) were
removed in Issue #171 PR 2: the vendor frames now come from the official
SDK's IDL through `unitree_g1_adapter`, and the runtime refuses to start with
a guided error when the SDK is missing.

## Profiles

| Profile | Surface | Robot | End effector | Joints |
|---|---|---|---|---|
| `profiles/g1_29dof_dex3.yaml` | full runtime contract; existing 9 `vendor.arm_actions` mappings | G1, 29 DoF | Dex3 dexterous hands (7 joints each) | 43 observable: 12 legs, 3 waist, 14 arms, 14 fingers |
| `profiles/g1_29dof_arm_presets.yaml` | 23 static preset actions (IDs 1, 11–13, 15, 17–30, 33, 34, 36, 99); the runtime publishes only the `GetActionList`-confirmed subset (possibly empty, no fallback); the deployed profile still exposes only action 15 | G1, 29 DoF | none (no hand, stream or base surface) | 29 observed, no command channels |

The profile is the single source of runtime parameters: public joint
projection, modes and their vendor actions, command channels, declared
capabilities, stop policy, the vendor endpoint/RPC/FSM tables, the arm
admission and commissioning boundaries, the simulated-transport domain, and the
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
  specified in the OpenSpec change (§5), so the base deployment
  (`unitree_g1.yaml`) still ships with `embodied.enabled: false`. What does
  ship is the fixed preset path: `config/robots/unitree_g1_arm_presets.yaml`
  enables the embodied stack with `arm_surface: false` and a static
  `runtime_named_motion` catalog of 23 preset skills (including
  `arms_horizontal`, action 15, and `release_arm`, action 99). A generic
  runtime-discovered catalog is not part of this change; the catalog is a static
  allow table, never a discovery result.
- **Voice ASR/TTS services.** The vendor voice RPC carries an ASR api id, but
  no ASR surface is bridged or verified; the audio contract itself waits on
  gate 6.1.

## Status

The runtime is implemented and tested against `unitree_vendor_mock`, which
speaks the real official-SDK IDL on the topic names the profile declares. The
package suite is 268 tests; the capability-scoped conformance suite runs
against the same launch (see *Tests*).

**Hardware evidence is limited and must be read with the preset scope above.**
For the full Dex3 runtime, everything that moves the robot — mode switches, the
arm_sdk handover, streaming, locomotion, stop latencies — is a user-executed
gate recorded in the OpenSpec change (tasks §6), and the profile values those
gates will measure (ramp rates, gains, the velocity envelope, the stop bounds)
are placeholders, not measurements. For the preset surface, the operator
reports having manually exercised all 23 actions (operator-reported); the Agent
has one earlier action-15 software-path success with the camera disconnected
and physical completion unconfirmed. The vendor tier's QoS is the SDK's own,
not this profile's.

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
ros2 launch unitree_g1_robot runtime.launch.py profile:=g1_29dof_dex3 use_sim:=true
```

`use_sim` / `sim_platform` select the transport; `simulated:=true` is the
deprecated alias the launch still accepts with a warning.

The vendor mock stands in for the robot's DDS tier and the contract layer's
synthetic publishers stand in for the composed peripherals, so the full public
surface is live without hardware. `robot_state_publisher` also runs, so RViz
can display the mock's joint states. To exercise the DDS wiring itself, run
the stack on CycloneDDS and name an interface — the launch then binds
CycloneDDS to it:

```bash
ros2 launch unitree_g1_robot runtime.launch.py profile:=g1_29dof_dex3 \
    use_sim:=true network_interface:=lo      # DDS smoke: CYCLONEDDS_URI generated and honoured
```

The conformance suite runs against the mock unchanged:

```bash
ROS_DOMAIN_ID=149 \
CONFORMANCE_LAUNCH="ros2 launch unitree_g1_robot runtime.launch.py profile:=g1_29dof_dex3 use_sim:=true" \
CONFORMANCE_PROFILE="$(ros2 pkg prefix unitree_g1_robot)/share/unitree_g1_robot/profiles/g1_29dof_dex3.yaml" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest src/robot_runtime/test/test_conformance.py -q -rs
```

## Running against the robot (user-executed gates)

Nothing in this section has been executed against hardware; it is the order
the OpenSpec change prescribes (tasks §6). Keep the robot on a gantry until
the locomotion gate.

### 1. Network

- Wire the host to the robot's Ethernet port and identify the interface
  (`ip link`). Launch with `network_interface:=<nic>`: the launch hands that
  NIC to the vendor tier as `vendor_network_interface` and fails closed when
  it is missing or loopback (see *DDS* above). A wrong interface name fails
  loudly in CycloneDDS at startup.
- Pick one RMW for the whole deployment and export it before launching —
  the launch does not set or check `RMW_IMPLEMENTATION` itself, but the
  runtime and every node that calls its services must agree on it:

  ```bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  ```

- `ROS_LOCALHOST_ONLY` must be unset or `0` on the robot-facing host, and
  every robot-facing process must run that same RMW (see *DDS* above).
- The vendor tier runs inside the SDK on its own participants, so prove the
  vendor link from the runtime's own log first: `Startup` reaches `ACTIVE`
  only after `rt/lowstate` and both Dex3 streams arrive, and `faults` names
  whichever one is missing. The SDK's vendor graph is not visible to
  `ros2 topic list` (rclpy cannot register its IDL types).

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

Gate 6.1 closes here: record the real RPC topic names and the vendor service
behaviour back into the profile.

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
   is refused, `ARM_HANDOVER_UNCONFIRMED` stays visible in `faults`, and a
   following `idle` clears both. Record `cancel_latency_s` / `idle_latency_s`.
   `TORQUE_OFF` while standing must be refused (fail-closed to sit/squat); that
   refusal closes admission too, so clear it with `idle` before the next motion.
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
  — and `/runtime/stop` is the supported stop. The signal path exits
  cleanly and bounded after a successful teardown (exit code 0; when the
  signal's `rclpy.shutdown()` races the executor's wait-set rebuild the
  resulting `RCLError` is logged as one line instead of a traceback — the
  process-level tests pin this). A supervisor restart is not a substitute
  for the stop service.
- **A failed teardown exits 3, not 0.** The signal path waits at most 2 s
  for callbacks already executing. If that wait expires, the SDK close
  raises, or `destroy_node` raises, the runtime leaves the remaining
  resources open (a failed close does not destroy the node; its handlers may
  still be reached by the SDK reader threads), prints one diagnostic naming
  the failed step, and exits with status 3 instead of reporting a completed
  cleanup. A supervisor must treat 3 as a failed teardown and restart the
  process; the process exit is the isolation boundary, not a physical stop.
  If the vendor library holds the GIL (Issue #207), even that bounded exit
  cannot run, so run the runtime under a supervisor that escalates to SIGKILL
  after its own timeout. The vendor mock follows the same contract in tests.
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
