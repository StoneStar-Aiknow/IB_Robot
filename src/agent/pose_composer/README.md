# pose_composer — natural-language arm poses for the AgiBot X2

Say *"双手向前伸直"*, *"做一个弯曲手臂的姿势"* or *"右手摸头顶"* and the X2 moves its
arms there. No per-pose trajectory is written in advance.

```
sentence ─► language model ─► pose description ─► solver ─► safety ─► path ─► runtime joint.trajectory
            (spec.SYSTEM_PROMPT)  (per arm: directions     (joint    (limits,   (planner,   (FollowJointTrajectory,
                                   or "hand to body point") angles)  clearance)  via poses)   replayed on the arm stream)
```

The language model never writes joint angles. It describes each arm either by
**directions** (`upper` = shoulder→elbow, `fore` = elbow→wrist, body frame) or by a
**body point** (`hand_to` one of `spec.LANDMARKS`, with an optional `offset_cm`
and `elbow` hint). The solver owns the kinematics; every frame is projected into
the safe set (joint travel, ≥2 cm from the body, 3 cm at the hands, gravity
torque), and the path between frames is checked and, when the straight path
would brush the body, routed through a via pose. The runtime validates the
trajectory once more against its own `vendor.arm.limits`.

## How the representation was chosen

In simulation (official X2 MuJoCo model, 33 instructions, 3 samples each, an
image judge calibrated against independent ratings), with DeepSeek-V4.1-Flash:

| Representation | T1–T3 weighted | held-out T6 | raw output safe | p50 |
|---|---|---|---|---|
| joint angles written by the model | 59–64 % | – | 85–87 % | 7–10 s |
| directions only | 82 % | 62 % | 100 % | 4.7 s |
| hand positions only | 79 % | 71 % | 100 % | 5.1 s |
| **directions or body point (this package)** | **92 %** | **79 %** | **100 %** | **3.3 s** |
| fixed pose library (reference) | 86 % | 25 % | 86 % | 2.6 s |

Known gaps: palm orientation is not described (palms-together reads as
"hands crossed"); the X2 shoulder barely adducts, so hands cannot reach the
ears or the chin squarely and such poses are reported as approximate.

## With the Agent

`robot_config` `aimdk_x2_free_pose` adds the `free_pose` skill to the X2 gesture
deployment. The planner puts the user's words for the pose in the step's
`target_name`; nothing else about the step changes.

```
ibrobot_agent ──(plan starts with free_pose)──► ~/compose  (ComposeArmPose)   composed while the plan is prepared,
     │                                                                          shown as a pose_preview event; a refused
     ▼                                                                          or unreachable pose is answered here
Gateway ─► pose_composer executor ──────────────► ~/execute (ExecuteArmPose)   runs that composition (or composes again
                                                                                 and says so), path from the measured pose
                                                                                 ─► runtime joint.trajectory
```

A stop request is forwarded through the Gateway; the runtime holds the arms
where they are. Only a pose that is the plan's first step is composed with
the plan; a later one is composed when it runs.

```bash
ros2 run aimdk_robot aimdk_fetch_x2_mjcf            # official model into ~/.cache/ibrobot (MulanPSL-2.0)
export ROS_DOMAIN_ID=<not the robot's> ROS_LOCALHOST_ONLY=1
AIMDK_MOCK_PHYSICS_MJCF=<scene.xml> ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=aimdk_x2_free_pose control_mode:=named_motion use_sim:=true authorize_motion:=true
ros2 run ibrobot_agent ibrobot_agent_chat --robot-scope aimdk_x2_skills
```

The composer's model is the `pose_composer.llm` block (OpenAI-compatible,
keys only by `api_key_env`).

## Pose memory

Read-only lookup has its own default-on switch, independent of writable memory:

```yaml
pose_composer:
  lookup:
    enabled: true  # default: query fixed seeds or confirmed stored poses before calling the Agent model
  memory:
    enabled: false # writable pose memory; fixed mode keeps this off
```

`lookup.enabled: false` disables the lookup endpoint and the Agent shortcut; it
does not disable explicit composition/execution or writable feedback when memory
is enabled. Fixed mode queries only its compiled, read-only library. A hit skips
the Agent model; a miss follows normal planning and the fixed executor still
rejects unknown actions. Lookup never creates a memory database or saves a pose.

With `pose_composer.memory.enabled`, poses the robot has done are kept in one
SQLite file per robot (default `~/.local/share/ibrobot/pose_memory/<robot>.sqlite3`),
shared by every conversation and kept across restarts. Only the composer
writes it.

```
miss ──generate──► candidate ──two successes + explicit save confirmation──► active
                      │                                                                  │
                      │ unused 7 days, or 2 bad verdicts                                 │ prompt, model, robot
                      ▼                                                                  ▼ model, travel or rest changes
                   retired ◄──────── no longer solvable ◄────────────────────────── stale ──re-solved (no model call)──► active
seeds (pinned, never evicted) ──solved at startup──► pinned
explicit memory-management rejection ──► those words never recall that pose again
```

* **Two tables.** `poses` holds one pose (description, joint targets of the arms
  it moves, a signature: where the moved elbow, wrist and hand are), `aliases`
  maps normalized phrasings to it. A new pose whose signature is within 5 cm of
  an existing one is that pose: the new phrasing becomes an alias.
* **Recall.** `~/lookup` answers only confirmed (active or pinned) poses, and the
  Agent then skips planning; `~/compose` also requires confirmed or pinned poses.
  Candidates are isolated from execution recall. An arm the pose does not move keeps where it is. Relative
  wording ("再高一点") is never remembered or recalled; context-dependent poses
  (`depends_on_current` in the model's answer) are not remembered.
* **Explicit save.** Two successfully completed, similar absolute poses produce
  a save question. Only an explicit positive answer to that question within 60
  seconds confirms the session's last achieved pose. A different task, timeout,
  or negative answer never saves it. Cancellation/failure does not count as a
  success. Approximate absolute poses can be staged but the question explicitly
  asks about the achieved approximate pose. Seed poses remain pinned.
* **Feedback protocol.** `offer` opens a confirmation only for an eligible candidate;
  `status` inspects it without extending the deadline; `good` accepts an offered
  candidate; `next_task` dismisses it without saving. `bad` rejects its phrases.
  The existing Agent interprets natural save replies from pending context in its
  normal planning call. Storage decisions are bound to the pending pose ID and
  rechecked before writing. There is no confirmation phrase table. Declining
  storage uses `next_task`, not `bad`; uncertain replies preserve the offer.
* **Every reuse is checked again**: projected into the current travel, path
  planned from the measured pose, and validated by the runtime.
* **Seeds**: `seeds/x2_poses.yaml`, descriptions only, solved against the
  robot's own travel; bump `version` to reload. Every change is in `audit`.

Maintenance (writes need the composer stopped; reads any time):

```bash
ros2 run pose_composer pose_memory --path <db> list [--all] | show <id> | audit
ros2 run pose_composer pose_memory --path <db> pin|unpin|delete <id> | ban <id> <phrase>
ros2 run pose_composer pose_memory --path <db> export out.yaml | import in.yaml   # imports are re-solved at startup
```

## Requirements

The workspace venv's `mujoco` and `scipy`; the official model fetched as above;
the OmniHand runtime profile, which declares `joint.trajectory`,
`vendor.arm.limits` and the measured rest pose.

## Small-motion commissioning

`aimdk_x2_free_pose_voice_commissioning` combines the voice entry with the
free-pose composer. Start with `authorize_motion:=false` while checking audio
and request routing. Its planner exposes only `free_pose`; preset gestures are
hidden and excluded by the Agent allowlist. Pose memory is disabled so that
commissioning exercises generation rather than a remembered pose.

The optional `pose_composer.max_joint_delta_rad` and
`pose_composer.max_hand_displacement_m` settings bound every planned trajectory
point relative to the measured start, including detours. This profile uses
0.06 rad and 0.025 m. An excessive excursion is refused as `POSE_MOTION_LIMIT`,
both during preparation and when the path is rebuilt immediately before
execution. Zero disables the corresponding extra limit; other profiles retain
their existing behavior. These are model-based bounds on the commanded path,
in addition to the existing safety and runtime validation.

The voice entry accepts ASR commands without an application-specific prefix.
The vendor's own wake behavior is configured separately on the interaction unit.
Replies require separately provisioned `SynthesizeSpeech` and `PlayAudioFile`
services at `/commissioning/tts/synthesize` and `/commissioning/tts/play`.
Those services must return playback success only after audio playback ends;
the voice bridge then acknowledges the presented plan. The profile does not
start a TTS model on the X2 or assume its built-in speaker works.

Relative descriptions (`depends_on_current`) preserve the measured wrist
angles; they do not silently reset the moving arm's wrist to neutral while
asking for a small translation. Absolute descriptions retain the neutral-wrist
default. Static gravity load uses zero-velocity `qfrc_bias`, rather than
`qfrc_inverse`, which can include contact/constraint reactions. The 30% rated
effort threshold is unchanged; collision checking remains separate. See the
[MuJoCo force definitions](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html).

### X2 pose backend

The Agent owns language parsing, dialogue and pose-memory policy. X2 model
loading, body landmarks, FK/IK, safety projection and joint-path generation
live in `aimdk_robot.pose_backend`. Shared result types and joint-limit
intersection live in `robot_runtime.pose_backend`. The former physical module
imports remain compatibility facades, with no duplicate solver implementation.
The backend supports one robot model per process.

The backend uses the existing CPU L-BFGS-B optimizer for target solving and
safety projection. Saved joint targets remain valid across this code extraction;
model, effective safe-limit or rest-pose changes still invalidate their fingerprint.

All targets and recalled poses pass the same clearance, gravity-load, joint-limit,
path-speed and commissioning-displacement checks. The actual clipped/downsampled
command path is checked again, including its transition from the measured start.
Malformed/non-finite trajectories fail closed. Units are SI; the body frame is
x forward, y left, z up. Optional top-level robot YAML `arm_limits` must cover all
arm joints and is intersected with runtime vendor limits; neither source can
widen the other. Without it, the complete runtime profile limits apply.

Grasp keeps its cuRobo planner and mirror keeps its existing retargeting path.
No hardware action is performed by backend setup or offline unit tests.

### Configured preparation poses and hands

`return_to_task_pose(target_name="grasp_ready")` uses the startup task-pose
snapshot without an LLM or memory lookup. An opt-in `task_poses.file` maps
`pick_object` to its preparation pose; optional `hands: {left: open, right: open}`
adds sequential opening. The snapshot is bound into catalog identity. Already
reached parts skip motion but require fresh stable feedback; failure or
cancellation blocks grasp perception and execution.
The Gateway prepares the runtime mode and owns one task for preparation and
grasp. This node coordinates the deadline, cancellation, measured arm arrival,
and hand backend; `aimdk_grasp.hand_restore` implements takeover, checked hand
interpolation and two seconds of released-state observation using public runtime
interfaces. It neither switches modes nor clears stop latches. Ordinary free
poses and task poses without `hands` keep their behavior.

Hardware needs both `hardware_approved` and `unloaded_operation_approved`, plus
verified models. Neither flag proves that a hand is empty. Both side-specific
grasp models now supplement the arm planner's rigid dummy hands with measured
articulated-finger geometry along the final command path. Pairwise interpolation
bounds cancel shared ancestor motion and include finger/head preservation error.
Missing models or insufficient clearance refuse output; finger/head drift during
arm execution requests cancellation. The raise-then-return Mock uses real planning
and both hand models with simulated language and actuator feedback; it verifies
that both hands actually open after return. It does not validate hardware or scene
clearance. Phase times and completed/skipped hands appear in the action
result on both success and failure. See [configuration and limits](../../../docs/x2_task_poses_zh.md).

### Fixed seed library on devices without MuJoCo

Fixed previews build bounded candidates without repeating nominal full-path
geometry. Dispatch always reads new measured state. Configure full-path checks
independently in the startup SSOT (default for both is `online`):

```yaml
pose_composer:
  mode: fixed
  path_validation:
    hand: online  # online / off / prevalidated
    body: online
```

`off` skips only that component's complete arm-path prediction. Endpoint body
and gravity checks, initial-segment uncertainty, ownership/freshness, limits,
speed, drift monitoring, stable arrival and the independent opening sweep remain.
Reports explicitly say `skipped`, including on failure. Changing this policy
changes configuration and delegated-executor identities; it cannot be overridden
by an Agent request. Generated mode rejects this block. The offline compiler
always uses strict `plan_safe`, independently of deployment switches.
Write the YAML value as `hand: "off"` / `body: "off"`: PyYAML interprets an
unquoted `off` as a boolean, which this policy deliberately rejects.

`prevalidated` additionally requires `certificate_file`; launch freezes its file
SHA256 and the executor validates the internal model/library/algorithm binding.
Missing or corrupt artifacts fail startup. Entry, shape or directed-route misses
fall back to the corresponding online check, never to `off`. A matching record
monitors tracking and requests cancellation on range loss; unconfirmed stopping
blocks subsequent pose executions in that process.

`pose_compile_validation` exports interval checks for HOME ↔ arms_up, including
entry, tracking, interpolation, finger/head and monitoring-response uncertainty,
and conservative gravity upper bounds. It uses the same `--profile`, `--limits`
and `--library` inputs plus `--entry-rad`, `--tracking-rad`, `--response-sec` and
`--output`. Exit code 2 means no component could be proven; refusal reasons remain
in the artifact. A digest is integrity evidence, not hardware approval.

**Physical reuse remains unavailable on the current X2 transport**: it has no
producer of fresh `LiveConditions` proving standing lower body, unloaded hands,
enforced speed and a verified cancellation bound. It therefore falls back online.
The tested X2 error tube (0.02 rad entry, 0.025 rad tracking, 0.1 s feedback age,
0.2 s polling, 0.25 s response) also exceeds joint limits. Do not reduce these
budgets merely to make a certificate pass. See the
[design and acceptance record](../../../docs/x2_path_validation_design_zh.md).

Agent events include request-local `planning_timing`; action results include
`preparation` and `path_validation` stage status/duration/policy identity. Offline
timing comparison is available through `scripts/benchmark_x2_path_validation.py`;
it never creates a ROS node or sends a hardware command.

Set `pose_composer.mode: fixed`, `fixed_library: /absolute/path/fixed_poses.json`,
and `memory.enabled: false`. Omit `llm` and `mjcf_path`. The existing generated
mode remains the default. Fixed mode matches only compiled seed IDs and exact
aliases; unknown descriptions and relative adjustments fail with
`POSE_NOT_IN_LIBRARY`, without an LLM fallback. `return_to_task_pose` uses the
same fixed execution backend and the configured hand-opening state machine.

Compile an artifact **on a workstation** with the official model available:

```bash
source .shrc_local
ros2 run pose_composer pose_compile_library \
  --mjcf /path/to/official/scene.xml \
  --profile src/robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml \
  --limits /path/to/effective_arm_limits.yaml \
  --seeds src/agent/pose_composer/seeds/x2_poses.yaml \
  --pose arms_up --output /path/to/fixed_poses.json
```

The compiler exports joint targets, convex body geometry and mass data, and
checks HOME-to-seed and seed-to-HOME paths with both backends. Seeds with
keep-current semantics are refused. Unselected seeds remain disabled. Keep the
artifact with deployment assets; it contains geometry derived from the vendor
model and is not a new repository model asset. Configuration hashes the complete
file, while its internal fingerprint binds the robot profile, limits and hand
geometry. A changed artifact or mismatched model prevents startup.

Board execution uses NumPy/SciPy, with no MuJoCo, IK or online pose generation.
The X2 geometry and fixed path implementation live in `aimdk_grasp`; the Agent
package selects its backend and coordinates the task and Gateway interfaces.
Declare `pose_composer.interfaces.trajectory` as the logical action
`joint.trajectory_arm` (`control_msgs/action/FollowJointTrajectory`) and
`interfaces.status` as `runtime.get_status` (`ibrobot_msgs/srv/GetRuntimeStatus`),
each with its `kind` and `type`. This keeps launch live-bound when grasp is disabled;
the runtime description supplies the actual endpoints and joint-state topic.
Each request builds a minimum-jerk straight joint-space path from fresh measured
arms to the target (maximum 0.5 rad/s). Body clearance, gravity, real articulated
hands, both hands against each other and the standing-leg envelope remain
checked by default. An unsafe direct path is refused; no alternate path is invented.
The final check uses the measured fingers and head. Start drift above 0.005 rad
prevents dispatch, and the admitted start uncertainty is included in the first
segment's geometry bounds. Finger/head drift during movement requests cancellation.
Finger preservation allows 0.1 rad; the same bound covers preparation, release
confirmation and articulated collision uncertainty. Head preservation remains
0.005 rad. Arm arrival tolerance and finger preservation are separate checks.
Action results contain a short preparation/policy summary, capped at 300
characters; full preparation and path-validation reports are written to the node
log. Previously compiled clearance results must be revalidated when this bound changes.
With hand path validation explicitly `off`, the mandatory entry check covers
the measured finger shape and arm start uncertainty, without certifying future
finger drift; the 0.1 rad runtime finger monitor remains active. `online` checks
the full 0.1 rad envelope and may reject the earlier open-hand library because
of wrist/thumb clearance. Existing offline certificates cannot certify the wider
envelope merely because their nominal targets are unchanged.
The model assumes the same fixed standing lower-body posture as the existing
backend; it does not provide scene sensing, dynamic-load guarantees or hardware
acceptance. Fixed mode requires the hand models even for arm-only seed execution.

Online planning builds a joint-limit/speed-checked candidate; the final transport
gate checks body, gravity and actual fingers/head once before dispatch. The
offline `plan_safe` API still performs full geometry validation. Hand collision
checks use conservative AABB separation before exact sphere distances, bounded
narrow-phase batches and rigid-pair reuse only when changing joints are shared
ancestors. Mesh arrays are cached as immutable model data and gravity arithmetic
is batched. Missing final geometry, a collision, cancellation or stale/non-finite
feedback still prevents dispatch.

Run `colcon test --merge-install --packages-select pose_composer` with
`IBROBOT_FIXED_LIBRARY` pointing to the compiled artifact. On a board without
MuJoCo, select `--pytest-args test/test_fixed_poses.py`; this suite mocks transport
and feedback while checking the real CPU geometry and both hand-opening paths.
The workstation additionally runs `test_compiled_geometry_reference.py` against
MuJoCo FK, gravity and collision distances.

### HOME-bound seeds (seed file version 2)

The passive arm of `right_hand_on_head`, `salute`, `thinking`, `right_forward`
and `left_forward` now uses `{task_pose: grasp_ready}`. It moves to the configured
HOME preparation target; it does not preserve the current arm pose. HOME means
the task-pose snapshot, not zero joint angles or a newly inferred hanging pose.
The composer resolves both anchors before IK and restores their exact values
after safety projection; an incompatible active target is refused.

`arms_down` / **双臂放下** binds both arms to the same task pose. Execution reuses
the `return_to_task_pose` preparation runner, including its hardware/unloaded
approvals, current hand configuration, automatic opening, settle and release
checks. No second copy of HOME angles or hand-opening code is maintained.

Supply the same task-pose YAML used by the deployment when compiling:

```bash
source .shrc_local
ros2 run pose_composer pose_compile_library \
  --mjcf /path/to/official/scene.xml --profile /path/to/runtime_profile.yaml \
  --limits /path/to/arm_limits.yaml --task-poses /path/to/task_poses.yaml \
  --seeds src/agent/pose_composer/seeds/x2_poses.yaml \
  --pose arms_up --pose arms_down --pose right_forward --pose left_forward \
  --pose right_hand_on_head --pose salute --pose thinking \
  --output /path/to/candidate/fixed_poses.json \
  --catalog-root /path/to/candidate/catalog
```

`--catalog-root` updates an existing **candidate copy** of the catalog, deriving
the allowed `free_pose.target_name` enum and aliases from compiled records only.
It does not reload a running Gateway. The new artifact embeds its frozen
task-pose snapshot; the executor rejects a different or missing runtime snapshot.
`--hand-path-check` defaults to `online`: compilation checks the complete
reference open-hand path with the current finger preservation envelope. For a
deployment that explicitly uses `path_validation.hand: off`, pass
`--hand-path-check off` to check the reference hand entry with 0.005 rad arm
start uncertainty, without checking the full hand path or future finger drift.
The artifact records this scope in `compile_validation`; body paths, static
gravity, joint limits and speeds are still checked. This option never changes
the live policy or the runtime's finger monitor.

Changing HOME or its preparation settings requires recompilation. Generated-mode
seed memory also fingerprints this snapshot. Existing version-1 artifacts with
no named anchors remain loadable.

The current commissioning arrival policy checks one fresh joint-state sample
after a successful trajectory terminal, instead of waiting for `stable_sec`.
It still enforces the configured position tolerance, feedback freshness,
cancellation, deadline and measured-pose safety before opening either hand.
The result records `stable_window_verified: false`; it does not certify a
continuous stability interval. An arrival mismatch remains a failed execution.

Near-body seed targets include explicit non-contact offsets for articulated
open-hand clearance: head-top `[0, -5, 5]`, salute `[7, -3, 0]`, thinking
`[8, -3, 0]` cm, in body-frame x-forward/y-left/z-up coordinates, added to their
landmark offsets. They stop outside the head/chin instead of touching the body.
The ambiguous `张开双手` alias was removed from the side-raise seed; spreading
arms does not itself open fingers.

The existing candidate contains 12 of 13 seeds. `hands_on_hips` remains excluded:
its direct HOME roundtrip fails wrist/body and articulated-hand clearance.
That candidate's compilation used the earlier 0.025 rad finger envelope; it does
not certify the widened 0.1 rad envelope. The default compiler and online checks
use the widened envelope and refuse insufficient clearance. Explicit compilation
with the hand check `off` records its narrower entry-only validation. Compilation
checks both HOME directions and the 0.005 rad entry uncertainty. These checks cover HOME ↔ each seed; arbitrary
seed-to-seed transitions are not certified. Return to `grasp_ready` before
switching gestures, or retain online path validation. Offline/Mock success is
not hardware acceptance of the new gestures.
