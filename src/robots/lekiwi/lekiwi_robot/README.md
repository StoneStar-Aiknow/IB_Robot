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
published in `RuntimeStatus`. The declared limits must be finite positive and
realizable: the runtime computes the envelope from the SDK kinematics and
rejects a profile whose limits are non-finite/non-positive or exceed it,
naming the axis. For the shipped geometry that is
0.2657 m/s in x, 0.2301 m/s in y and 1.84 rad/s in yaw; the profiles publish
0.26 / 0.23 / 1.5.

## Base node (`lekiwi_robot/base_node.py`)

- `/cmd_vel` (`Twist`) -> `base_velocity_controller/commands` (wheel rad/s)
  through `lekiwi_sdk_py.body_to_wheel_velocities`; commands beyond
  `max_wheel_radps` are scaled down preserving direction, and the envelope
  clamps each axis to the published limits first (a command with a
  non-finite component is rejected and zeroes the target — min/max would
  otherwise turn NaN into +limit). Commands are zeroed
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
`lekiwi_nav_grasp` (arm + base: adds `stream`, `trajectory`) are shipped. Both
run on the `sdk` simulated transport; `lekiwi_nav_grasp` additionally runs its
navigation stage on both physics platforms (`sim_platforms: [sdk, gazebo,
mujoco]`, lekiwi-nav-grasp-sim-platforms). The other LeKiwi configurations in
`robot_config` still use the legacy provider-less bring-up.

## Running standalone

```bash
ros2 launch lekiwi_robot runtime.launch.py profile:=lekiwi_lidar [simulated:=true]
```

`simulated:=true` keeps the production controller and hardware-adapter path
and replaces only the serial bus with the SDK's simulated transport.

## Gazebo simulation

The Gazebo platform (`sim_platform:=gazebo`) stands in for the physical
sensors: the runtime renders `gpu_lidar` and IMU sensors into the model at
the MID-360 mount (`lekiwi_mid360_mount.yaml`), bridges them through
`ros_gz`, and serves the advertised endpoints (`/livox/lidar`,
`/livox/imu`) on simulation time. Use a fresh `IGN_PARTITION` **and** the
same value in `GZ_PARTITION` (plus `IBROBOT_GAZEBO_HEADLESS=1` without a
display) so runs on a shared host stay isolated: this Fortress install
(`ign gazebo`, gz-transport11) reads only `IGN_PARTITION` and ignores
`GZ_PARTITION`, while newer gz reads `GZ_PARTITION` — setting both covers
either stack.

The launch entries the verification runs use:

```bash
# provider alone (sensors, admission, wheels): task 2.5
ros2 launch lekiwi_robot runtime.launch.py profile:=lekiwi_lidar \
  sim_platform:=gazebo

# composed through the robot_config entry: task 4.2
ros2 launch robot_config robot.launch.py robot_config:=lekiwi_lidar \
  use_sim:=true sim_platform:=gazebo

# navigation (FAST-LIO + Nav2/AMCL): task 5.2 stages the lekiwi_lidar
# profile and the navigation overlay side by side, then launches
ros2 launch robot_config robot.launch.py config_path:=<staged>/lekiwi_navi_sim.yaml \
  use_sim:=true sim_platform:=gazebo nav_stage:=navigation
```

TF ownership in simulation (one publisher per transform):

| Transform | Owner |
| --- | --- |
| `odom → base_link` | wheel odometry (`base_node`), on simulation time; the FAST-LIO bridge publishes `/odometry/filtered` with `publish_tf: false` |
| `map → odom` | AMCL (the navigation stage); never a static transform |
| `base_link → livox_frame` | static TF from the Gazebo embodiment (the rendered model has no sensor link); camera frames stay on `robot_state_publisher` |
| FAST-LIO's own TF | isolated on `/fast_lio/tf_raw` |

The simulated IMU publishes SI units (m/s², rad/s); at rest it reads gravity
(`|acc| ≈ 9.81` m/s² along +z of the sensor frame). Gazebo's pre-settle
samples — the zero window before the physics starts and the landing impulse
when the entity drops onto the ground — are held back by
`robot_runtime/imu_settle_filter` until gravity has been stable for a full
run, so FAST-LIO's gravity initialization only ever sees valid samples.

`lekiwi_sensor_calib` (the MID-360 / camera extrinsic calibration workflow)
needs the physical sensors and is not supported on Gazebo; the simulated
sensors always sit exactly at the mount.

### Lidar-odometry regression gate (D14)

`robot_navigation/test/test_lidar_odometry_gate.py` is the committed FAST-LIO
accuracy gate for this runtime. It lives in `robot_navigation`, not here:
it launches the generic application entry, and this package may not declare
test dependencies on generic packages (the independence gate below audits
the execution closure; test dependencies only reach its test closure, never
the runtime's execution surface). It is
parametrized over profile (`lekiwi_lidar`, `lekiwi_nav_grasp`) as well as
platform: the mobile manipulator runs the SAME thresholds over the SAME
route, staged from the gate's own
`robot_navigation/test/config/lekiwi_nav_grasp_lio_sim.yaml` overlay, with
the stowed-arm envelope asserted for the whole route (see the nav_grasp
section below). With no Nav2 goal active, it
drives a scripted closed
route on **`/cmd_vel_nav`** — the production chain's entry point — so the
velocity smoother and the collision monitor stay in the loop
(`controller → /cmd_vel_nav → velocity smoother → /cmd_vel_smoothed →
collision monitor → /cmd_vel`). The gate checks that `/cmd_vel` is published
only by the collision monitor and by Nav2's behavior server (its recovery
behaviors write the channel directly, bypassing the smoother and the
monitor — a property of the shipped chain; with no goal active the
behavior server stays silent, and the gate's message-level pairing proves
the monitor is the only publisher that actually drives the base). Every
non-zero command must be relayed unchanged.

The route (west half of the room, 7.0 m of translation, two in-place turns,
two lateral segments, closed; commanded at vx 0.10 / vy 0.08 / wz 0.35 —
below the smoother limits):

| # | Command | From → to (x, y, yaw) |
| --- | --- | --- |
| 1 | forward 0.8 m | (−1.5, −1.5, 0°) → (−0.7, −1.5, 0°) |
| 2 | rotate +90° | → (−0.7, −1.5, 90°) |
| 3 | forward 2.7 m | → (−0.7, 1.2, 90°) |
| 4 | lateral +y 0.8 m (body) | → (−1.5, 1.2, 90°) |
| 5 | rotate −90° | → (−1.5, 1.2, 0°) |
| 6 | lateral −y 2.7 m (body) | → (−1.5, −1.5, 0°) |

Segments close on simulator ground-truth progress (each stops commanding at
the velocity smoother's stopping distance), and every metric is measured in
simulation time. FAST-LIO's `/odometry/filtered`, aligned to ground truth at
the route start, is scored against the committed thresholds
`min(cap, max(2 × worst of five measured runs, floor))`; the spec caps are
ATE RMSE ≤ 0.10 m, max error ≤ 0.20 m, final error ≤ 0.15 m, final yaw error
≤ 3°. Wheel odometry (`/odom`) and AMCL (`/amcl_pose`) are printed and
loosely bounded, never part of the FAST-LIO verdict. Health during the
route: `/odometry/filtered` must carry at least 95 % of the `/livox/lidar`
frames FAST-LIO consumed over the same simulation-time span (the whole
drive, final settle included), with no adjacent stamp gap above 0.35 s —
Gazebo's `gpu_lidar` occasionally renders an empty scan, which FAST-LIO
skips; the fraction floor absorbs those, the gap bound catches a real
stall; no `fastlio_mapping` death or respawn,
no odometry sample stamped ahead of the latest `/clock` received before it (the odom bridge's own rejection log must stay empty), and the TF ownership table above holds
on the live graph. Plots and raw samples go to the test's log directory.

```bash
NAV_TEST_PROFILE=full colcon test --packages-select robot_navigation \
  --pytest-args -k lidar_odometry_gate
```

The gate skips with a reason listing every missing capability on platforms
without ground truth, profile support, lidar simulation or a scene template,
and — independently — when the platform's simulator is unavailable
(`ign gazebo` or `gz sim` for Gazebo; the MuJoCo ament resources and a
matching Python `mujoco` for MuJoCo). Acceptance runs set
`IBROBOT_REQUIRE_SIM_PLATFORMS=gazebo,mujoco` to turn any such skip into a
failure. The environment probe and the require-or-skip rule are shared
through `robot_runtime_sim.testing.sim_environment` (one copy for this gate, the
chassis gate below and robot_config's navigation e2e). The overlay is staged
per platform (`robot_navigation`'s `test/config/lekiwi_lio_sim.yaml` pins
`platform: gazebo`; the gate rewrites the staged copy).

### Navigation end-to-end test (D15)

`robot_config/test/test_navigation_simulation.py` is the Nav2 end-to-end
check of the same navigation stage: it sends `NavigateToPose` goals (to a
point and back to the spawn) and a cancel-based stop, asserting the D3 stop
thresholds and single-publisher `map -> odom` ownership (AMCL) on simulator
ground truth. It is parametrized over the first-party physics platforms
(`gazebo`, `mujoco`) — goals,
thresholds and assertions are identical on Gazebo and MuJoCo — and over the
staging base (`lekiwi_lidar`, `lekiwi_nav_grasp`): the mobile manipulator
runs the same goals through its own overlay
(`robot_config/test/config/lekiwi_nav_grasp_navi_sim.yaml`, whose Nav2
parameters carry the measured `robot_radius: 0.28` footprint — the stowed
arm's planar extent 0.240 m / conservative 0.277 m vs the base-only 0.22)
with the stowed-arm envelope asserted throughout every navigation phase.
The staged overlay copy (not the shipped one) carries the platform under
test. The same capability/environment/require rule as the gate applies
(`IBROBOT_REQUIRE_SIM_PLATFORMS` turns a skip of a listed platform into a
failure); the shared rule lives in `robot_runtime_sim.testing.sim_environment`
(robot_config must not import lekiwi_robot's test assets).

```bash
NAV_TEST_PROFILE=full colcon test --packages-select robot_config \
  --pytest-args -k test_navigation_simulation
```

Conformance and the independence gate:

```bash
scripts/verify_runtime_independence.sh runtime lekiwi_robot --profile lekiwi_lidar
scripts/verify_runtime_independence.sh runtime lekiwi_robot --profile lekiwi_nav_grasp
```

The base velocity, odometry, staleness and navigation-gate tests are in scope
and must execute, not skip. Use a fresh `ROS_DOMAIN_ID` (at most 232): a stale
runtime left on a shared domain answers for the one under test.

## MuJoCo simulation

The MuJoCo platform (`sim_platform:=mujoco`) is a physics backend of both
shipped profiles (introduced as the lidar runtime's second backend,
lekiwi-mujoco-runtime-platform #193; `lekiwi_nav_grasp` followed in
lekiwi-nav-grasp-sim-platforms). It runs the same production chain as Gazebo —
`base_node` gating, the MID-360 mount, FAST-LIO, the scan converter — with the
sensors simulated by MuJoCo itself (the `mujoco_3d_lidar` engine extension for
the cloud, the MJCF sensor triple for the integrated IMU); under
`lekiwi_nav_grasp` the composed model additionally hosts the SO-101 arm MJCF
(`so101_description/mujoco/so101.xml.template`) attached under `base_link` at
the URDF-derived pose (D2/D8):

```bash
# provider alone, headless (no display): sensors, admission, wheels
IBROBOT_MUJOCO_HEADLESS=1 ros2 launch lekiwi_robot runtime.launch.py \
  profile:=lekiwi_lidar sim_platform:=mujoco [sim_world:=nav_room \
  spawn_x:=-1.5 spawn_y:=-1.5 spawn_z:=0.05]

# navigation (FAST-LIO + Nav2/AMCL) through the robot_config entry,
# from a staged overlay whose simulation.platform is mujoco
ros2 launch robot_config robot.launch.py config_path:=<staged>/lekiwi_lio_sim.yaml \
  use_sim:=true sim_platform:=mujoco nav_stage:=navigation
```

An empty `sim_world` composes the shared `empty` scene. Use a fresh
`ROS_DOMAIN_ID` and kill stray `ros2_control_node` processes afterwards; the
composed model lands under `/tmp/ibrobot_sim_models/<ROS_DOMAIN_ID>/` (an
extension-declaring model is delivered as XML — the C++ node compiles it
in-process; extension-free models stay `.mjb`).

### Chassis model

- The robot MJCF (`lekiwi_description/mujoco/lekiwi_base.xml.template`) is
  base-only: three frictionless sphere wheels (`condim 1`, priority 2 over
  the scenes' support planes) with velocity actuators named `wheel_7/8/9`;
  the ros2_control joints stay `7/8/9` (SimpleTransmission mappings, MuJoCo
  branch only — a `PidROS` topic token may not start with a digit).
- The chassis is a KINEMATIC base: `base_node`'s wheel odometry (`/odom`)
  drives the floating base through `robot_runtime/odometry_twist_relay`
  (validated: frame, freshness, order) and the simulator's
  `BaseVelocityPlugin`. There is no wheel slip — wheel odometry tracks the
  ground truth to sub-millimetre — and walls do not stop the driven axes
  reliably, so navigation must avoid geometry as on hardware. The IMU
  accelerometer reads dynamics accelerations only and misses the override's
  velocity steps; between the 10 Hz lidar corrections FAST-LIO absorbs the
  resulting ≤ ~15 mm per-frame prediction error (measured: ATE RMSE
  ≤ 0.0044 m over the gate route).
- `empty` is a SMOKE scene: its invisible support plane sits at z = 0 while
  MuJoCo rays skip alpha-0 geoms, so the lidar sees the checker floor at
  z = −0.274 — sensors publish and FAST-LIO runs, but no accuracy claim is
  made there; quantitative LIO acceptance runs in `nav_room` (its single
  visible floor is both the support and the observation surface).

### Controller difference on MuJoCo (D3.1)

A released MuJoCo velocity actuator keeps its last target (upstream
`mujoco_ros2_control` 0.1.2: `perform_command_mode_switch` clears control
flags, `write()` keeps the staged `ctrl`), so `base_velocity_controller` is
**never deactivated** on MuJoCo: the profile's mode table keeps it active in
`idle` through `controllers_by_platform.mujoco`, and `base_node` is the gate
on that platform — it already zeroes its target on a non-base mode, a stop
latch, a closed navigation gate or stale `/cmd_vel`, then publishes one zero
command. Hardware and Gazebo keep the ordinary controller lifecycle, and
every stop path is verified on ground truth by
`lekiwi_robot/test/test_mujoco_stop_gate.py`.

Residual risk (accepted, MuJoCo only): another publisher on
`/base_velocity_controller/commands` could move the base in `idle` —
hardware blocks that by deactivation. TORQUE_OFF's hardware lifecycle
transition is a no-op upstream (with ros2_control 2.54.0 an INACTIVE
component is still written), so `base_node`'s zero keeps reaching the
actuators — verified for that version only.

### Clocks and the IMU rate

The embodiment runs on simulation time (`use_sim_time`), including
`base_node`'s control timer and the relay's freshness window; command
staleness itself is measured on `base_node`'s monotonic wall clock, so at a
low real-time factor the zero lands at the first control tick after
`staleness_s` wall seconds (no exact bound is claimed). The simulated IMU
publishes on the controller-manager clock — ~100 Hz with the shipped
configuration — and the advertised rate derives from the effective
controller-manager `update_rate` of the generated controllers file (no
second hard-coded rate).

### Screenshots (GUI, non-headless)

The MuJoCo viewer window plus RViz showing the FAST-LIO cloud/map and the
odometry path (acceptance evidence, `DISPLAY=:1`). The
`lekiwi_nav_grasp` navigation-stage variant (map, cloud, Nav2 path and the
wrist camera during a `NavigateToPose`) is documented with the mobile
manipulator section below — it supersedes this base-only recipe for #205
evidence:

```bash
# terminal 0 — the fragment (fast_lio.enabled only)
printf 'fast_lio:\n  enabled: true\n' > /tmp/lekiwi_fastlio.yaml

# terminal 1 — the runtime with the viewer (no IBROBOT_MUJOCO_HEADLESS)
ros2 launch lekiwi_robot runtime.launch.py profile:=lekiwi_lidar \
  sim_platform:=mujoco sim_world:=nav_room spawn_x:=-1.5 spawn_y:=-1.5 \
  peripherals:=/tmp/lekiwi_fastlio.yaml

# terminal 2 — RViz with the FAST-LIO topics (/cloud_registered with a decay
# time, /odometry/filtered arrows, the robot model; fixed frame odom)
rviz2

# terminal 3 — capture the two windows (imagemagick; the X11 compositor keeps
# window pixels readable even when occluded — no hand-written XWD decoder):
xwininfo -root -tree                      # find the window ids
import -window <id> shot.png
```

## The mobile manipulator in simulation (`lekiwi_nav_grasp`)

The whole robot — omni base + SO-101 arm + wrist RealSense + MID-360 — runs
its navigation stage on **both physics platforms** (lekiwi-nav-grasp-sim-
platforms). The arm always starts and holds at `hardware.reset_positions`
(stowed) on Gazebo and MuJoCo: the render path injects the ros2_control
`initial_value` of each joint's position state interface, and on MuJoCo
`set_initial_pose()` additionally stages the position actuators' `ctrl` from
the same values on the first control cycle (verified across controller
release and world reset). Navigating with a non-stowed arm is **outside the
validated envelope** (D7): the runtime admits `base_navigation` with the arm
anywhere (no pose admission exists today — asserted by the stop cross gate
so a later admission check changes it deliberately), the gates assert the
stowed pose throughout every navigation phase, and Nav2 plans with the
measured stowed footprint (`robot_radius: 0.28` in
`robot_navigation/config/nav2_params_navgrasp_sim.yaml`, the overlay-only
parameter variant — the shipped base-only config keeps 0.22).

```bash
# provider alone (sensors, admission, wheels, arm, MoveIt stack), either platform
IBROBOT_MUJOCO_HEADLESS=1 ros2 launch lekiwi_robot runtime.launch.py \
  profile:=lekiwi_nav_grasp sim_platform:=mujoco [sim_world:=nav_room]
IBROBOT_GAZEBO_HEADLESS=1 IGN_PARTITION=<fresh> GZ_PARTITION=<fresh> \
  ros2 launch lekiwi_robot runtime.launch.py \
  profile:=lekiwi_nav_grasp sim_platform:=gazebo

# navigation (FAST-LIO + Nav2/AMCL) through the robot_config entry, from a
# staged nav_grasp overlay whose simulation.platform carries the platform
ros2 launch robot_config robot.launch.py \
  config_path:=<staged>/lekiwi_nav_grasp_navi_sim.yaml \
  use_sim:=true sim_platform:=mujoco nav_stage:=navigation
```

### Excluded peripherals (D3)

The microphone (`respeaker`) and speaker are **host ALSA devices with no
simulated stand-in**: no physics platform simulates an audio source or sink,
so the profile excludes both on Gazebo and MuJoCo
(`simulation.<platform>.enabled: false`) and neither is advertised nor
claimed there. The test overlays
(`robot_config/test/config/lekiwi_nav_grasp_navi_sim.yaml`,
`robot_navigation/test/config/lekiwi_nav_grasp_lio_sim.yaml`) therefore turn
off every application consumer bound to those interfaces — `speech_direction`
(`audio.capture`) and `voice_tts` (`audio.play`) — plus `voice_asr`,
`embodied` and RViz as the base-only overlays do; the `unknown_interface`
binding check stays strict, so a consumer left enabled for an excluded
interface fails the launch instead of silently skipping. The wrist camera
stays in the runtime inventory although the navigation stage's
`peripheral_names` does not list it (`peripheral_names` only selects fragment
entries — the runtime's own inventory is the profile's).

### Stop semantics: the platform difference (D5)

- **MuJoCo / hardware / SDK**: a released position actuator keeps its last
  goal (`mujoco_ros2_control` 0.1.2 keeps a released interface's `ctrl`; the
  physical servos keep their last goal), so a stop mid-convergence lets the
  arm settle at the **last accepted target** — the cross gate asserts it
  within 0.05 rad within 1 s of simulation time.
- **Gazebo**: the settle point is **not deterministic**. `gz_ros2_control`
  0.7.19 implements position control as a proportional-velocity servo, and
  `perform_command_mode_switch` clears the whole control method with
  `joint_control_method &= (VELOCITY & EFFORT)` — a bitwise AND of disjoint
  masks, i.e. 0 — so a released joint falls to the zero-velocity fallback
  only on the **next** controller-manager `write()` cycle; under CPU
  contention the last `JointVelocityCmd` persists in gz's entity-component
  system and the joint coasts past the command by a load-dependent amount
  (measured: up to ~0.7 rad past the target; the control group
  `so101_single_arm` — untouched by this change — reproduces it). The cross
  gate therefore asserts on Gazebo what actually holds: later stream
  commands never move the arm (≤ 1e-6 rad; ≤ 1e-9 measured), the arm ends at
  rest (max joint velocity over the final second ≤ 1e-6 rad/s; ~1e-11
  measured), and after the stop the arm controllers are inactive claiming no
  interface — **no landing-point number is asserted**. The upstream issue
  draft (`gz_system.cpp` line references, timing measurements, suggested
  fix) lives in the #205 evidence directory.

### The MuJoCo controller difference, arm included (D1/D3.1)

The base-only rule above extends to the arm: the profile's mode table keeps
`base_velocity_controller` **active in every mode on MuJoCo**
(`controllers_by_platform.mujoco`), because a released MuJoCo actuator keeps
its last target — `base_node` is the gate there. The arm and gripper
controllers follow the ordinary lifecycle on both platforms (released in
`idle`); on MuJoCo their released position actuators keep the last goal
(the settle-at-target semantics above), on Gazebo the zero-velocity hold
applies (with the release-race caveat above).

### Mobile manipulation in simulation (the grasp stage, #205)

The navigate-then-grasp E2E runs the **production `pick_object` pipeline** in
the `nav_grasp_room` scene — `nav_room`'s navigation geometry plus a static
low table (0.4 × 0.5 × 0.015 m slab, top 0.015 m above the floor, west edge
at map x = 1.0) and one dynamic magenta cube (0.03 m edge, centre 0.025 m
inside the edge; it settles to z = 0.03).
`sim_models/scenes/nav_grasp_room/layout.yaml` is the single authority, and
its `mobile_manipulation:` section carries the task data: the object prompt
("magenta cube") and the success thresholds (lift ≥ 0.05 m, hold 3 s, ≤ 10°
rotation and ≤ 0.01 m drift in the gripper frame).

**What the E2E exercises** (`embodied_bringup/test/test_mobile_pick_simulation.py`,
revision 2): the production pipeline end to end — the `PickObject` action →
`pick_executor_node` → GraspGen `grasp_planner` + `grasp_verifier` + the
grounded-SAM2 detection service, every arm motion as a Gateway primitive
through `skill_executor`/`task_executor` (`move_to_configuration`; the
gripper through the gripper trajectory controller), and the supervised-direct
pick through the production `pick_action_client`. Inference runs on **CUDA**
(the shipped CUDA fixture's deployment values, each annotated with its source
line in the overlay). The stage is a **test overlay staged at test time**
(`lekiwi_nav_grasp_pick_sim`: peripherals `[wrist, mid360]`, the grasp
detection service on CUDA, HRI and semantic mapping off) — **no shipped
config changes**; the skill catalog is a development staging catalog
(test-only profile with `pick_object` and the skills its implementation
needs). Around the pick: `NavigateToPose` with the no-motion-recovery BT and
the `/cmd_vel` attribution audit, `SetRuntimeMode("trajectory")`, the
chassis-only planning scene (the table is deliberately absent — production
never puts it in MoveIt's scene; the maintainer decision of 2026-11-02), the
R6 audit of every executed arm sample against `/check_state_validity` on
EVERY attempted execution — a failed pick that never moved the arm records
explicit no-motion evidence — the timestamped runtime-status and sim-clock
invariants, and the all-ground-truth judge (a continuous one-second frozen
rest window, the layout's `lift_m` pre-lift boundary, the model-TCP grasp
region, the base roll/pitch tilt). The verdict is assigned only after the
invariants, the audit, the judge and the whole cleanup complete: cleanup
failures stay hard failures even behind an approved failure code.

**Simulation-only settings** (hardware, SDK and MoveIt renders stay
byte-identical — golden tests pin this):

- MuJoCo runs the grasp stage with `noslip_iterations: 20`,
  `cone: elliptic` (the profile's `simulation.mujoco.option` override): with
  the host defaults the jaw contacts fired at 0.01–0.07 N and the cube was
  not holdable; with the override the hold drift is 0.0 mm. The scene also
  declares its default free-camera view (`<statistic>` centred on the
  staging/table area, extent 0.75), so the MuJoCo viewer opens ~1.1 m from
  the robot at the table — a view-only setting, no physics effect.
- Gazebo renders the jaws as the same CoACD convex hulls MuJoCo simulates
  (`jaw_collision: hulls`, set only by the Gazebo render path of
  `so101_base.xacro`) and sets `gripper_effort: 1.0` — the URDF default
  (10) ejects the cube through the hull faces on Gazebo (measured sweep:
  10/2.75/2.0/1.5 eject, 1.0 captures 5/5).
- The test overlay stages a **test-only Nav2 parameter variant** (generated
  at staging time, never committed): final-approach inflation radius 0.12 —
  the shipped 0.20 stalls MPPI at the cost cliff in front of the goal and
  stops the base 0.35 m from the table edge, where every top-confidence
  GraspGen candidate fails preparation IK; 0.12 stops ~0.27 m from the edge.
  The staging goal is the layout's single definition —
  `mobile_manipulation.staging_pose` (0.785, 0.40), read through
  `mobile_manipulation_spec` — which brings the base to ~0.24 m from the
  cube (the real robot's validated pick band).

**Hardware-visible production fixes shipped with the stage** (all need
on-robot verification):

1. **Simulation time in the embodied nodes and the perception model service**
   (R3): every embodied node that exchanges stamped data gets `use_sim_time`
   from the launch's clock decision on physics platforms.
2. **`pick_action_client` ROS-clock budgets** (R3): `DispatchBinding.task_budget`
   is built from the node's ROS clock (wall time remains for
   process/watchdog timeouts) — budgets read on the ROS clock must be
   written on it.
3. **IK workers get the runtime arm description** (our own !550 regression):
   `motion.launch` passes the same robot description it gives the primary
   `move_group` to `ik_workers.launch`, so the workers plan in `arm_root`
   like the primary instead of rendering their own `world`-rooted model.
4. **`motion.fk` advertises the global endpoint only** (`/motion/compute_fk`):
   the motion server registers per-worker IK only, so the worker FK
   advertisement could never be served and failed the pick executor's
   service gate at pick start.
5. **Planning failures are per-candidate** (2026-11-02): the motion server
   classifies a MoveIt planning-phase abort (no trajectory execution
   started) as `PLANNING_FAILED`, and the Gateway reports the distinct
   `PRIMITIVE_ARM_PLANNING_FAILED` code, which the pick executor retries
   with the next candidate; a failure after motion started stays
   non-retryable.
6. **The runtime-status heartbeat owns its callback group** (2026-11-02):
   a slow readiness check can no longer delay the facade's heartbeat past
   the Gateway's freshness budget (measured on Gazebo: the blocked-close
   load starved the shared callback group and the next primitive was
   refused as stale).
7. **The grasp verifier's open-jaw rule** (2026-11-02): a jaw reading at or
   near the fully-open reference after a close goal is never contact
   evidence — it scores as failure evidence ("jaw at/near fully open after
   close goal"); "stopped before fully closing" applies only to a jaw that
   actually moved from open.

**Known Gazebo issues** (maintainer decision 2026-11-02: the pick E2E is
**accepted on MuJoCo**; on Gazebo the run still happens and only the two
measured pick-failure classes below are reported as known issues —
`pytest.xfail` raised only AFTER the continuous invariants, the R6 audit and
the whole cleanup complete, with both modes in the reason; every other
failure — startup, navigation, the invariants (runtime status, sim clock,
ground-truth resets or missing/unsynchronized evidence), the arm-sample
audit, cleanup, models — stays a hard failure that overrides the xfail, and
a Gazebo pass is a plain pass; reach accuracy and every other Gazebo test
stay required). The 2026-11-02 acceptance sequence: 6 runs, 0 passes. Two
modes:

- **Retry-budget exhaustion at the stop-pose band** (base 0.177–0.267 m from
  the table edge): the marginal-candidate preparation codes (IK orientation,
  contact realign/compensation over-limit) — including deep-descend
  candidates whose **moving-jaw CoACD hull wedges the table slab at the
  fully-open end** so the close goal never moves the jaw (measured
  e2e/gazebo-180245, -183150: desired ramps 1.0 → 0.0 while the joint stays
  at exactly 1.0001; the tip hull rests on the 0.015 m slab and the closing
  arc drives it deeper — self-locking; the 1.0 N·m gripper effort cannot
  break it; the zero-tolerance JTC reports the goal successful). Root cause:
  the production FK tabletop filter checks the raw-STL jaw mesh
  (`tabletop_clearance_m: -0.025`) while Gazebo's physics uses the CoACD
  hulls whose faces sit up to ~14 mm outside the raw STL. The
  `GRASP_UNCERTAIN` member of this class is granted only on the verifier's
  specific jaw-at-open wedge evidence carried in the PickObject failure
  message; a missing/stale-sensor "No usable grasp-verification evidence"
  or a generic fused-uncertain close is a hard failure.
- **A verified-contact close that does not survive the transport swing**: the
  jaw blocks on the cube at q6 ≈ 0.40 (MuJoCo's validated blocking band) and
  the verifier confirms contact (the `verification_status` enum), but the
  effort-1.0 grip slips and the cube is never carried (e2e/gazebo-174538,
  -183757, -184202). This class requires the judge's PAIRED gripper-frame
  evidence showing the object outside the jaws; missing or unsynchronized
  ground-truth evidence is a hard failure. MuJoCo holds the same blocking
  band through the full transport.

**Known Gazebo modes that stay hard failures** (documented 2026-11-03,
review r2 — none of these is an approved xfail class; all measured on the
fixed tree, runs/e2e/r2fix-gate-*):

- **The Gateway's runtime-status admission can refuse under heavy host load**
  (`CONTROL_MODE_MISMATCH`, "runtime status is stale or its clock is
  unsynchronized"): the Gateway's admission applies a zero-tolerance lower
  bound on the status SOURCE age (`0 <= source_age <= 3.0 s`), and under load
  the Gateway's own `/clock` view can lag the facade's heartbeat stamps.
  Measured (review r2, the instrumented legs): 1 pick-phase occurrence in 5
  Gazebo legs — at host load 31.86, right after a verified 3.9-s wall close;
  the heartbeat itself was fresh at the E2E probe throughout (receipt wall
  gaps ≤ 2.75 s vs the 3.0-s budget, source-stamp ages ≤ 7 ms), and the
  Gateway-side refusal log measured `receipt_age` 1.1–1.5 s (healthy) with
  `source_age` −6 → −12 s (the Gateway's clock view behind the facade's
  stamps). Legs at load 2.87–23.07 never showed it; MuJoCo (RTF ~1.0) never.
  A follow-up issue owns the skew-tolerance decision (the analogous
  `action_dispatch` admission carries one); the refusal log stays in the
  admission path as diagnostics.
- **A marginal base-planar-speed reading** (`0.021 m/s` vs the spec's
  `0.02 m/s` bound, ONE sample during the descend) rejected an otherwise
  verified-contact pick (r2fix-gate-2). The spec number stands — the judge
  stays strict.
- **Arm self-contact grazes at the observe-move start** can fail the R6
  audit: the two recorded invalid states from the r1 run gazebo-shot-155025
  were replayed through `/check_state_validity` (r2,
  runs/e2e/gazebo-r2state-replay/state_replay.json): both are
  `gripper ↔ upper_arm` SELF-CONTACTS (depths 0.12 mm and 0.04 mm) with no
  contact against the chassis box — the earlier "chassis AABB graze"
  attribution is refuted. The audit stays strict; the base's roll/pitch tilt
  is recorded (max over the hold window) but judged by planar speed only
  (maintainer decision 2026-11-03).

**Running the tests** (a clean shell per the project rules —
`source .shrc_local` / the worktree variant; fresh `ROS_DOMAIN_ID` per solo
run; batch `colcon test` keeps the domain unset; both simulators headless
except screenshot runs):

```bash
# the pick E2E on the accepted platform (formal verdict: colcon test)
mkdir -p /tmp/grasp_artifacts
ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 IBROBOT_MUJOCO_HEADLESS=1 \
IBROBOT_REQUIRE_SIM_PLATFORMS=mujoco \
IBROBOT_E2E_RUN_DIR=/tmp/grasp_artifacts \
python3 -m pytest src/agent/embodied_bringup/test/test_mobile_pick_simulation.py -k mujoco

# the pick E2E on Gazebo — still runs; only the two approved pick-failure
# classes above xfail at their verdict (retry-budget exhaustion with the
# listed preparation codes or the jaw-at-open wedge GRASP_UNCERTAIN; a
# true-contact grasp whose transport physically slips, with paired
# gripper-frame evidence) — everything else stays a hard failure
ROS_DOMAIN_ID=219 ROS_LOCALHOST_ONLY=1 IBROBOT_GAZEBO_HEADLESS=1 \
IGN_PARTITION=<fresh> GZ_PARTITION=<fresh> \
IBROBOT_REQUIRE_SIM_PLATFORMS=gazebo \
IBROBOT_E2E_RUN_DIR=/tmp/grasp_artifacts \
python3 -m pytest src/agent/embodied_bringup/test/test_mobile_pick_simulation.py -k gazebo

# reach accuracy against simulator ground truth (D4) — still required on
# BOTH platforms (4 targets, <= 0.015 m / <= 10 deg each)
ROS_DOMAIN_ID=216 ROS_LOCALHOST_ONLY=1 IBROBOT_MUJOCO_HEADLESS=1 \
IBROBOT_REQUIRE_SIM_PLATFORMS=gazebo,mujoco \
python3 -m pytest src/robots/lekiwi/lekiwi_robot/test/test_navgrasp_reach_accuracy.py -k mujoco

# the formal verdict for the package (domain unset; the root conftest
# allocates one per pytest process)
colcon test --packages-select embodied_bringup
```

Acceptance (2026-11-02): the pick E2E passed **twice in a row on MuJoCo**
(mujoco-153742, -154120) and the regression run after the Gazebo fixes
passed (mujoco-184750: judge OK 3.56 s hold, 0.1° in-gripper, 0.0 mm drift,
0.313 m lift, RTF 1.002); Gazebo 0/6 with the two known modes above. Reach
accuracy passes on both platforms. Evidence: the #205 change's
`acceptance.md`.

### Hardware-visible changes (D8/D13 — all need on-robot verification)

This change is sim-focused, but three things are visible on the physical
`lekiwi_nav_grasp` and must be verified on the robot:

1. **TF: the arm's mid-chain link `world` is renamed `arm_root`** on every
   LeKiwi render (hardware, SDK, Gazebo, MuJoCo; sdformat rejects a
   mid-chain frame named `world`, spike 1.1). The fixed joint
   `base_to_arm` (base_link → the renamed root, rpy 0 0 1.5708, xyz 0.06
   −0.02 0.02) is unchanged; the hardware URDF render is byte-identical to
   the pre-change baseline apart from the rename and one comment
   (`test_hardware_render_regression.py` pins this against a golden
   generated from `2d534791`). No in-repository consumer referenced the
   `world` frame (searched `lekiwi_*`, `robot_config`), but an external
   one may.
2. **MoveIt plans in `arm_root` instead of `world`**: at the baseline
   `2d534791` the full robot and the arm-only motion model both used
   `world`; this change renames **both to `arm_root` at the same time**
   (`so101_base.xacro` `root_link` arg), so move_group's planning frame
   always matches the public model's arm chain. The two must move
   together: during development an intermediate state that renamed only
   the full robot's chain left move_group planning in `world` while the
   TF no longer published it — FRAME_TRANSFORM failures on SDK execution,
   caught by robot_config's application e2e.
3. **MID-360 mount extrinsics now match `lekiwi_lidar`**: the profile's
   top-level mid360 `transform` (none → the measured mount
   `lekiwi_mid360_mount.yaml`: base_link → livox_frame, xyz −0.08 0 0.20,
   yaw 90°) and `fast_lio.body_to_base_translation/rotation` (absent →
   the bridge defaults were identity; now the mount inverse
   [0, −0.08, −0.2], yaw −90°). The robot_config navigation/hybrid stage
   fragments do **not** override `body_to_base_*` (policy only: enabled /
   planar_output), so the profile values are what the FAST-LIO odom bridge
   uses on hardware: the odometry frame is now conjugated by the mount
   instead of remaining sensor-rotated — the same correction `lekiwi_lidar`
   already carries (#192/#193, `88f2ee23`). Effect on hardware: the
   `livox_frame` static TF appears at the measured mount, and FAST-LIO's
   `odom` frame aligns with the base instead of the yawed sensor frame.

### nav_grasp screenshot evidence (both platforms)

```bash
# terminal 0 — stage the nav_grasp navigation overlay for the platform
# (the staging helper pattern; or copy test/config/lekiwi_nav_grasp_navi_sim.yaml
# next to robot_config's config/robots/lekiwi_nav_grasp.yaml and set
# robot.nav_stages.navigation.simulation.platform)

# terminal 1 — the navigation stage WITH the viewer: no *_HEADLESS, DISPLAY set
DISPLAY=:1 ros2 launch robot_config robot.launch.py \
  config_path:=<staged>/lekiwi_nav_grasp_navi_sim.yaml \
  use_sim:=true sim_platform:=mujoco nav_stage:=navigation

# terminal 2 — RViz: map + /cloud_registered (decay time) + Nav2 path/costmap
# + a wrist camera image panel (/camera/wrist/image_raw), fixed frame map
DISPLAY=:1 rviz2

# terminal 3 — send a goal, then capture both windows mid-navigation
ros2 action send_goal /navigate_to_pose nav2_msgs/action/NavigateToPose \
  "{pose: {header: {frame_id: map}, pose: {position: {x: 1.0, y: 1.0}}}}"
xwininfo -root -tree                      # find the window geometries
# the simulators render through OpenGL: import -window on their windows comes
# out black under Xvfb (no compositor), so capture the ROOT framebuffer and
# crop to the window's ABSOLUTE geometry (WxH+AbsX+AbsY from xwininfo).
# /cloud_registered is stamped in FAST-LIO's camera_init frame whose TF the
# navigation stage isolates on /fast_lio/tf_raw; the bridge's conjugation
# fixes camera_init -> odom at exactly fast_lio.body_to_base, so a
# visualization-only static_transform_publisher publishing that transform
# lets RViz render the cloud (see the #205 evidence script
# evidence_screenshot.sh, which automates all of this).
import -window root root.png
convert root.png -crop <WxH+X+Y> +repage mujoco_navgrasp_nav.png
convert root.png -crop <WxH+X+Y> +repage rviz_navgrasp_nav.png
```

## Known simulation issues

- **Jaw contact jam at the folded spawn pose (Gazebo).** With the arm in the
  folded `reset_positions` configuration the gripper jaw is wedged by a
  contact constraint and cannot track position commands (the JTC open aborts
  with GOAL_TOLERANCE_VIOLATED while the joint stays at 0.0). Moving the arm
  out of the spawn pose (the production pick's observe-first ordering) frees
  it; MuJoCo is unaffected. The pick E2E's observe-first ordering avoids it.
- **Open-jaw table wedge at deep-descend candidates (Gazebo).** The
  production FK tabletop filter checks the raw-STL jaw mesh while Gazebo's
  physics uses the CoACD hulls (faces up to ~14 mm outside the STL), so a
  descend admitted by the filter can rest the moving-jaw tip hull on the
  0.015 m table slab at the fully-open end; the closing arc drives the tip
  deeper (self-locking) and the 1.0 N·m gripper effort cannot break the wedge
  — the close goal never moves the joint while the zero-tolerance JTC
  reports success (measured e2e/gazebo-180245, -183150: desired 1.0 → 0.0
  with the joint frozen at exactly 1.0001). The verifier's open-jaw rule
  turns this into a retryable uncertain verdict, never a false success.
- **The effort-1.0 grip does not survive the transport swing (Gazebo).** A
  close that blocks on the cube at q6 ≈ 0.40 (MuJoCo's validated blocking
  band) verifies as true contact, but the cube slips out during the
  transport to the container pose — the object is never carried (judge:
  carried 0; e2e/gazebo-174538, -183757, -184202). MuJoCo holds the same
  blocking band through the full transport. The pick E2E is accepted on
  MuJoCo; on Gazebo its failure is reported as these known issues
  (`xfail(strict=False)`, 2026-11-02).
- **Per-worker FK is not served (hardware-visible).** The motion server
  registers the global `/motion/compute_fk` plus per-worker **IK** only
  (`/<namespace>/motion/compute_ik`); `lekiwi_nav_grasp` therefore advertises
  `motion.fk` with the global endpoint only
  (`/ik_worker_0/motion/compute_fk` was advertised but never served, which
  failed the pick executor's SERVICE_UNAVAILABLE gate at pick start).

