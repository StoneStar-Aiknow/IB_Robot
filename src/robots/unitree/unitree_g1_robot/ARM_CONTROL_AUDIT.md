# G1 arm-control verification — 2026-10-28

Scope: documentation and source inspection only. No robot connection, service
switch, mode change, or motion command was made during this audit. This is not
hardware acceptance. Existing bridge-test skips are not passing coverage.

## Sources

Official documentation was retrieved through the public documentation directory
API (`https://robot-api.unitree.com/doc?space=G1_developer&locale=en`) and its
returned CDN document URLs. The website itself renders a JavaScript shell.

- [Sport Services Interface](https://support.unitree.com/home/en/G1_developer/sport_services_interface)
  — [document body](https://doc-cdn.unitree.com/11/255/en/11_255_en), directory update 2026-07-13.
- [Arm Action Interface](https://support.unitree.com/home/en/G1_developer/arm_action_interface)
  — [document body](https://doc-cdn.unitree.com/11/1004/en/11_1004_en), directory update 2025-11-17.
- [Arm Control Routine](https://support.unitree.com/home/en/G1_developer/arm_control_routine)
  — [document body](https://doc-cdn.unitree.com/11/247/en/11_247_en), directory update 2024-09-27.
- [Motion Switcher Interface](https://support.unitree.com/home/en/G1_developer/motion_witcher_service_interface)
  — [document body](https://doc-cdn.unitree.com/11/993/en/11_993_en), directory update 2025-11-12.
- Installed official Python SDK at commit
  `814556d15970dd2ecf1c9984e845ca02ab07e206`, especially
  `example/g1/high_level/g1_arm7_sdk_dds_example.py` and
  `unitree_sdk2py/g1/loco/g1_loco_client.py`.

The documentation describes supported interfaces, not the currently installed
robot firmware or its present state. Those remain to be established separately.

## Verified interface distinctions

1. **`rt/arm_sdk` depends on built-in motion control.** Sport Services describes
   upper-body DDS control while the built-in controller remains active. Debug
   mode exits built-in motion control and invalidates high-level services.
   Supported DDS states are Locked Stance, Movement Control 1 and Movement
   Control 2. The Arm Control Routine says debug mode is unnecessary; its test
   recommendation is suspension and locked standing. That recommendation is not
   authorization to change this robot's posture or mode.
2. **`SwitchToUserCtrl()` is full-body handover.** Sport Services assigns it to
   User Development Mode with `rt/user_lowcmd` and user control of all motors.
   It is not documented as an arm-only enable call. The pinned arm7 Python
   example calls neither this method nor `ReleaseMode`/`SelectMode`.
3. **Preset actions and streaming have separate ownership requirements.** Arm
   Action documentation requires disabling `g1_arm_example` before independent
   arm-topic development. This is a service change, not a read-only query, and
   must not be performed automatically during verification.
4. **7404 means the current FsmID cannot trigger the requested action.** It does
   not establish voice-only ownership or that the SDK cannot execute presets.
   Motion Switcher documents 7002 as service busy and 7004 as unsupported mode
   name. These codes do not establish supported alternative modes for this unit.
5. **The arm frame also covers the waist.** Sport Services assigns motor indices
   12–28 to waist/upper limbs and index 29's q to shared weight. The pinned
   example commands 14 arm and three waist joints, initially toward zero, then
   shoulder-roll/elbow targets. Running that example unchanged would exceed an
   arm-only, preserve-current-posture request. The earlier ad hoc positive
   shoulder-pitch trajectory was not verified by this example.
6. **Normal release is a weight ramp, not proof of an emergency stop.** The
   documented example reduces the shared weight while publishing. The reviewed
   documentation does not establish an arm-stream packet-loss watchdog or a
   guarantee that interrupted publishers release control.

## Current implementation findings

| Finding | Source location | Consequence |
| --- | --- | --- |
| Stream entry calls `switch_to_user_ctrl` | `unitree_g1_robot/runtime_node.py:919–947` | Full-body handover is incorrectly used as an arm-stream prerequisite; this blocks arm-only execution. |
| Exit only waits for local weight zero | `runtime_node.py:949–976` | It cannot prove restoration from the full-body user-control mode entered above. |
| Weight is updated before writer success; failures only log | `runtime_node.py:1104–1123` | A failed terminal zero-weight publication can still satisfy `_await_weight_zero`; local completion must not be reported as verified remote handover. |
| Stop always sends zero base velocity before checking handover | `runtime_node.py:641–730` | Arm-only stopping is coupled to a whole-body RPC; a failed RPC returns before handover verification, although the timer may continue ramping. |
| Both Dex3 feeds are mandatory at startup | `runtime_node.py:409–440` | Missing hand feeds block ACTIVE even for arm-only work; hand identity/profile compatibility must be established, not bypassed. |
| No `g1_arm_example` ownership check in stream entry | `runtime_node.py:919–947` | The documented service-conflict prerequisite is not established by this path. |
| CRC is fixed at zero and passed through | `projection.py:358–395`; `unitree_g1_adapter/vendor.py:186–213` | The pinned Python example computes CRC before every write (lines 173–174). This discrepancy needs resolution; it does not prove a firmware rejection or explain RPC code 7404. |
| Bridge suite skips unconditionally | `test/test_runtime_bridge.py:49–63` | Prior passing mock/unit tests do not establish end-to-end handover or stop correctness. |

## Required next implementation work

- Remove the full-body takeover assumption from the arm-stream design and its
  tests; define read-only admission for documented supported states and ownership.
- Keep legs under the built-in controller. Explicitly address the shared waist
  slots; holding their measured values still participates in upper-body control
  and is not proof that body posture cannot change.
- Align frame construction with the pinned SDK, including CRC; validate offline.
- Make stop/handover outcomes distinguish local admission closure, publication
  failure, and remotely verified state. Do not retry unknown-outcome motion RPCs.
- Confirm actual hand hardware and introduce the correct capability/profile
  configuration before changing startup requirements.
- Restore meaningful bridge tests and run formal package tests through
  `colcon test` before adding the G1 Gateway action and attempting hardware work.

No implementation, service ownership, firmware state, or physical behavior was
changed by this audit. Do not use this report as a motion authorization.

## Follow-up implementation (same day)

The findings above describe the pre-fix baseline. The follow-up removes the
full-body RPC from stream entry, adds read-only service/FSM admission, computes
CRC on the completed IDL message, and retains local weight until publication
succeeds. A refused/exceptional write latches a fault and is not automatically
retried. Arm-only HOLD does not send base velocity commands. Successful local
zero publication is explicitly not remote handover: exit/stop report uncertainty
and keep the latch. Real-hardware acceptance is still blocked on a verified
handover/stop contract and actual hardware/profile compatibility.

The pinned B2 RobotStateClient cannot import because it refers to missing
`rpc.client_internal`. The adapter uses the official SDK Client transport and
the B2 API constants for a read-only ServiceList shim (no service switching,
no SDK-file modification, no Go2 API-version substitution).

The blanket bridge skip was removed and missing harness teardown restored.
See `test/BRIDGE_COVERAGE.md` for the corrected coverage limitations.

## Operator hardware clarification

The operator confirmed on 2026-10-28 that this G1 uses Revo2 hands, not Dex3.
Hand model identity is therefore supplied by the operator, not inferred from
USB chips or absent DDS topics. The shipped Dex3 profile is incompatible with
this deployment. Port/side mapping, slave IDs, baud rate, firmware and
feedback/stop behavior still need verification. The read-only hardware check
log and the Revo2 hand adapter move out to the hand-port follow-up change.
