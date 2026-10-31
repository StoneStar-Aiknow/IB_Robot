# MC Release Adapter

`profiles/x2_mc.yaml` describes the MC-only interface, checked against the separately delivered
`mc-x86-v1.0.0-20260522`, robot `lx2501_3_t2d5`. It is opt-in: the existing
`x2_ultra` and `x2_ultra_omnihand` profiles describe a different firmware
interface and are unchanged. Do not interchange them based on the robot name.

The profile is a runtime contract, not a vendor release selector or simulator
startup recipe. `vendor.motion_api: mc` selects the supported action vocabulary;
`runtime.type: x2_mc` identifies the public runtime variant. The release date
records the source of interface facts here, not another robot type.

## Source Of Facts

The developer-supplied distribution was inspected at `/home/oee/x2_aimdk0908`:

- `mc/bin/mc_param/src/robot/lx2501_3_t2d5/mc.yaml` and `mc_sim.yaml`: enabled endpoints.
- `action_setting.yaml` and `sim/action_setting.yaml`: registered actions.
- `classic/animation_player.yaml`: animation identifiers and supported areas.
- SDK `GetMcAction.srv` and `GetMcLocomotionSpeedStatus.srv`: query protocols.
- MC startup logs: service names and `nav_msgs/msg/Odometry` leg-odometry type.

Only interface/configuration facts are encoded in the adapter. No vendor SDK,
implementation, binary, animation CSV, URDF or model is redistributed. The vendor
overlay remains an external dependency with its own licensing requirements.

## Implemented Surface

| Public interface | MC interface | Scope |
|---|---|---|
| `/runtime/set_mode` | `SetMcAction` | idle -> `STAND_DEFAULT`, locomotion -> `LOCOMOTION_DEFAULT`, step -> `LOCOMOTION_STEP`; waits for state confirmation |
| `/cmd_vel` | `/aima/mc/locomotion/velocity` | Registered input source, ACTIVE/mode/navigation gates, limits, input timeout, explicit zero on stop |
| `/motion/get_state` | `GetMcAction` | Actual platform action name, ID and raw action status; not cached local mode |
| `/motion/get_locomotion_profiles` | `GetMcLocomotionSpeedStatus` | Current and none/low/medium/high/step profiles, signed per-axis bounds; read-only |
| `/motion/execute_named` | `SetMcPresetMotion`, `GetMcPresetMotionState` | 21 named presets, task completion polling, configured target validation |
| `/motion/execute_named` | `SetMcAction` | `sit_down`, `lie_down`; waits for platform body pose |
| `/odom` | `/aima/mc/leg_odometry` | Preserves timestamp, frames, pose, twist and covariance; does not generate TF |
| `/joint_states`, `/aimdk/body_joint_states` | HAL joint feedback | Read-only arm/body feedback; not independent arm/leg control |
| `/runtime/stop` | Local admission latch, zero velocity, optional `SetMcAction` | HOLD does not cancel a vendor animation; TORQUE_OFF is downgraded to damping, not zero torque |

The existing input-source registration/query and MC common-state subscription
remain in use. The release profile deliberately does not require camera, PMU,
audio or interaction services: MC + MuJoCo alone does not provide that stack.
The adapter now calls seven of the seventeen configured motion-business
services: `SetMcAction`, `SetMcInputSource`, `GetCurrentInputSource`,
`SetMcPresetMotion`, `GetMcPresetMotionState`, `GetMcAction` and
`GetMcLocomotionSpeedStatus`. The ten remaining services are listed below.

Named presets: wave, handshake, raise_hand, blow_kiss, clap, thumbs_up, victory,
heart, arms_out, chest_wave, salute, welcome, cheer, bow, light_wave, hug,
arms_crossed, nod, shake_head, look_around and bye. Not every animation in the
vendor resource table is exposed. Adding a preset is not hardware acceptance.

## Explicit Gaps

| MC surface | Reason not exposed |
|---|---|
| `SetMcMotion`, `GetMcMotions`, `SetMcMotionSeq`, `RegisterCustomMotion`, `SetMcMotionStreaming` | No complete corresponding service IDL found in the supplied public SDK; service registration alone does not specify a protocol |
| `SetRobotRuntimeModel`, `GetRobotRuntimeModel` | Complete service IDL unavailable; common-state runtime-model fields are not an equivalent setter/query |
| `ActivateMcSecureAction`, `SetMcSecureMotion` | Protocol and authorization/safety semantics unavailable; not replaced by a generic raw RPC tunnel |
| `SetMcLocomotionSpeedMode` | Setter IDL unavailable; speed-profile query is not speed-mode control |
| `/aima/mc/body_pose`, `/aima/teleop_bridge/vr_data`, `/aima/mc/joint/retargeting` | MC-side types exist, but the public SDK lacks complete matching messages and control semantics |
| `/mc/upper_body_command`, `HEAD_ONLY`, `UPPERBODY_REMOTE_SPLIT` | SDK contains the API, but this MC package has no matching receiver/action evidence; absent from this profile |
| Whole-body/RL/test actions, generic stand-up | No public execution/input contract in this adapter; distinct recovery actions are not treated as aliases |
| MC debug topics, retarget frame, framework management RPC | Not bridged; outside the public robot execution contract |

Input-source MODIFY/DELETE/DISABLE are not public administration APIs. The
runtime owns its own source and does not let callers modify other controllers.
Vendor protection, action-transition rules and balance remain MC responsibilities.
This adapter does not bypass them through HAL command publication.

## Use And Verification

Build from the repository root, after provisioning the vendor overlay:

```bash
source .shrc_local
source /path/to/aimdk/install/local_setup.bash
./scripts/build.sh --aimdk
```

Select `profile:=x2_mc` on `aimdk_robot/runtime.launch.py`. With the
default `use_sim:=false`, the bridge expects an externally running vendor stack.
`use_sim:=true sim_platform:=sdk` selects the repository protocol mock.
The launch entry starts the bridge and its independent data plane, not vendor MC
or simulator processes, and rejects
`sim_platform:=mujoco`. Launching the bridge does not automatically switch the
MC action or enable velocity output.

Keep deployment choices in the existing `use_sim`, `sim_platform`, `instance_id`
and `initial_mode` launch arguments. `initial_mode` selects local admission only;
it does not execute a vendor action. Query endpoints come from `interfaces`,
without a second public-endpoint table under `vendor`.

The shipped mode graph contains only idle, locomotion and step. It does not
encode the seated simulator sequence `SIT_JOINT_DEFAULT` -> `SIT_UP_DEFAULT` ->
`STAND_DEFAULT`. If the externally managed vendor simulator needs that sequence,
perform and verify it with the vendor's own tooling before requesting runtime
motion. It is not an automatic startup procedure, a generic recovery skill or
a validated real-robot transition. Action-name confirmation alone does not prove
that a recovery movement has finished or the robot is balanced.

The stop service remains available, but this profile does not advertise the
`runtime.stop` capability: its schema requires latency bounds, and none have
been measured for this adapter. Do not substitute placeholder values. HOLD is
a local admission/output stop; TORQUE_OFF requests damping and reports the
downgrade. Neither is an accepted bounded physical stop.

For standalone physics, use [aimdk_mujoco](../aimdk_mujoco/README.md). That package
does not run the proprietary MC or validate this adapter's runtime contract.
MC leg odometry, named-motion completion, locomotion displacement and HOLD
behavior remain unverified against the real vendor stack. Simulation ground
truth is not substituted for MC leg odometry.

Queries use `ibrobot_msgs/srv/GetMotionState` and
`ibrobot_msgs/srv/GetLocomotionProfiles`. They return failure on service absence,
vendor errors, or invalid speed bounds; they do not substitute cached success.

```bash
source .shrc_local
source /path/to/aimdk/install/local_setup.bash
timeout 240 colcon test --merge-install --packages-select aimdk_robot
colcon test-result --test-result-base build/aimdk_robot --verbose
```

Tests use real generated IDL with a localhost-isolated protocol mock. This is
not evidence of end-to-end operation with the proprietary MC, physics correctness,
safe action interruption, or measured stop latency. Those require a separate
controlled acceptance session. HOLD closes local output only: it must never be
presented as a physical emergency stop or as cancellation of every MC action.
