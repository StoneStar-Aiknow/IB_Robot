# Optional mirror collision filter

The C++ filter runs after the data plane selects the complete arm/hand command.
It is disabled by default and applies only to the configured mirror context.
Other modes and named-motion owners retain the original command path.

No vendor SDK source, assets or binaries are included. The ROS node uses its
existing locally installed `aimdk_msgs` overlay; the numerical kernel and tests
do not require that overlay, ROS execution, MuJoCo or a running robot.

## Configuration and calibration

`robot.cbf_safety.enabled` enables the optional configuration. `mode: shadow`
preserves the original command and reports diagnostics; `mode: enforce` filters
the arm target or repeats a verified complete arm/hand target. The protected
runtime mode is derived from the mirror configuration, not a second mode switch.

The library remains disabled by default. The feature branch profile currently
sets `enabled: true` and `mode: enforce` at the operator's explicit request to
test the live flow with the configured candidate mapping. Shadow is not active
collision protection. Physical calibration remains outstanding.

Real commands require an explicit conversion to the frozen hand URDF:

```yaml
thumb_mapping:
  verified: false
  command_scale: [1, 1, 1, 1]
  command_offset: [0, 0, 0, 0]
  feedback_scale: [1, 1, 1, 1]
  feedback_offset: [0, 0, 0, 0]
```

These identity values illustrate the schema only. They are **not a calibration**.
Each array is ordered left roll, left abad, right roll, right abad, with
`q_urdf = scale * q_native + offset`. `mapping_required` means no conversion is
supplied. `verified` records calibration
provenance; a false value is visible in status and the diagnostic reason but
does not block a configured candidate conversion in enforce mode. There is no
separate trial-authorization switch. Commands and actual feedback may need
different mappings. Using a candidate does not establish physical calibration.

`hands_installed: [true, false]` is the user-confirmed physical setup: two arms
and a left OmniHand, with no right hand. This setting is explicit and defaults
to both hands for existing callers. Only installed hands provide measured
roll/abad feedback. The original hand20 command layout stays unchanged.
Pairs containing an absent palm or thumb are excluded; all body and upper-arm/
forearm pairs remain active, with the same frozen dimensions and transforms.
Reported SDK types must match the configuration (OmniHand type 1, absent type 0).
A lost installed-hand state, unexpected hardware type, or newly attached hand
cannot silently disable collision checks. Recalibrate before changing hardware.
Absent-side diagnostic angles are null, never invented measurements. This v0.5
adapter requires at least one installed OmniHand.

Only arm14 is optimized. Accepted hand20 values are unchanged. MCP bending is
ignored in the straight-thumb collision approximation, never in the command.

## Runtime behavior

- Actual arm/thumb feedback anchors the swept geometry check. Command rate
  limits use the last published target, avoiding drift from measured tracking lag.
- The selected runtime profile supplies the nominal travel bands through
  `arm_feedback_lower_limits` and `arm_feedback_upper_limits`. This is separate
  from the tighter commanded operator envelope. Travel bands constrain goals;
  encoder feedback can cross them slightly under tracking error or quantization.
  `check_arm_feedback_goal_range` defaults to true for existing callers. The live
  mirror profile explicitly sets it to false: arm travel bands constrain goals,
  and cannot reject finite encoder observations before evaluating their geometry.
  Finite samples remain raw, including values outside the native bands. Nonfinite
  feedback, invalid configured intervals, stale/skewed data and unverified geometry
  cannot authorize a new target. Within an active unchanged session, existing
  committed commands may still be retransmitted. Target limits and rate caps remain.
  `feedback_limit_tolerance_rad` is an explicit observation-only allowance (default
  zero, feature trial 0.017453292519943295 rad / one degree; allowed range 0..0.1 rad).
  For arms this allowance matters only when the optional goal-range check is on;
  it is not another threshold to tune in the live mirror's arm input path.
  Raw measured angles, including this excess, enter every FK and swept check.
  No observation is clipped. Installed-hand feedback uses the same native-angle
  allowance, scaled into its own model observation bounds. Hand targets retain
  strict native/model bounds. Desired/output arm targets retain the original
  intersection, one-degree command margin and rate limits. Native profiles and
  frozen collision shapes are not edited. This allowance is a deployment
  setting, not evidence of encoder accuracy or calibration.
- Effective application/measured limits and their one-degree margin are checked.
  Each command step is no larger than 0.04 rad or the original velocity allowance.
- FK and frozen geometry are unchanged. The endpoint projection now operates
  inside `command_joint_limits`: application limits intersected with the rate
  box around the last published target. It cannot propose an oversized correction
  and rely on the wrapper rejecting the whole frame. Final guards keep the same box.
- Ordinary solve, input-quality or deadline failures retransmit the complete
  committed arm14/hand20 pair from the same active epoch. This is continued
  position-command transmission, not a claim of freshly verified geometry.
  No valid initial command means no invented pose; stop, mode, ownership, epoch
  and hardware authorization exits still retain their original authority.
- Exact-target verification never optimizes a hold toward a lagging measurement.
  Safe initial states retain strict configured clearance checks. Runtime recovery
  permits already-violated pairs to withdraw or move tangentially without worsening
  their margin until they leave the barrier; other pairs retain full clearance.
  An existing proxy overlap may only lessen, never deepen. Partial recovery is
  allowed over multiple rate-limited frames, with `recovery_verified` reported.
  All accepted paths use at least 64 segments. Bounded Jacobian repairs at a
  failed path sample share the original deadline and iteration budget.
- A newly requested thumb target is checked with arms held at raw measured
  coordinates, so a fast hand cannot rely on an arm that has not moved yet.
  An unsafe new gesture may wait as a complete hand20 while verified arm motion
  withdraws; it is retried next frame. Non-root finger changes remain unaffected.
  `hand_held` reports this temporary ordering intervention; gesture angles are
  not internally rewritten or permanently latched.
- The hand-first query receives at most one quarter of the remaining compute
  window after reserving the publication guard. A sub-budget timeout defers the
  complete new hand20 target while arm verification/projection continues with
  the old hand target. Whole-frame exhaustion still retransmits the committed
  pair. This does not change hand angles, collision clearance or motion limits.
- Feedback reception continues while the existing command thread solves outside
  the core lock. Admission/epoch, feedback age and the whole tick budget are
  checked again before publication, reserving the configured guard for message
  construction/publication. A late or newly stale result uses its prepared
  committed pair. This is bounded best-effort software, not a hard-real-time proof.
  Publication cannot be undone: if the OS or transport overruns afterwards,
  retain the actual published history and report `publish_overruns`/`max_tick_us`.
- Node and wrapper share one absolute `steady_clock` deadline from the command
  tick, so snapshot preparation consumes the same budget as solving. Numerical
  callers may omit `CbfSnapshot.publication_deadline` and retain the configured
  relative budget. An unchanged projected target reuses the kernel's endpoint/
  path verification for that exact snapshot; replacing its arm target with a
  preferred reference still requires final verification. No cross-frame proof
  cache is used, and the independent command-box/publication checks remain.
- Startup, ready/cooldown, live and same-path return commands are covered. A
  blocked return cannot bypass protection or prevent the existing stop/release.
  Upstream timers are not proof that the physical arm reached its reset pose.

The existing data-plane status adds `cbf` only when enabled: mode, mapping state,
active context, calls, actual accepted/published/modified/held/withheld counts,
kernel and node deadlines, reason, distance and filter/tick timing. No new node,
topic, service, solver thread or per-frame disk logging is introduced.
The existing node also prints a throttled status line every two seconds for the
operator: mirror activity, published frames, geometric corrections, holds,
withheld frames, maximum tick time and the current reason. This uses the existing
launch log and creates no additional recorder process.

Status also exposes the last frame's `available_budget_us`, `hand_check_us`,
`verification_us` and `projection_us`, plus maxima for the three compute stages.
`verification_reuses` counts avoided duplicate projected-target rechecks, even
if a later publication gate holds the frame; it is not a published-frame count.
`hand_budget_holds` counts whole-gesture deferrals caused by the hand sub-budget.
The two-second log includes these stage times and the reuse count. Stage totals
exclude node snapshot, lock reacquisition and publication overhead; use `tick_us`
and its maximum for the command tick, rather than summing unrelated maxima.

## Invalid-input fault log

`CBF_INPUT_FAULT` records the first rejected kernel input of a failure episode.
It names the failed predicate and joint/index, and includes the actual value
and bounds when applicable. The same record retains the measured arm14, original
request, exact kernel target, model thumb6 current/target, compact native thumb
feedback, native target slots and model thumb bounds. All angles are radians;
values on an absent hand are kernel placeholders, not measured feedback.

`stage` identifies `hold`, `nominal`, `projection` or `final`. A
`current_arm_out_of_range` issue concerns measured feedback versus the model
input domain, not proof that the requested command exceeded its limits.
The log includes `kernel_arm_feedback_limits14`, the exact observation bounds
used in that call, including the configured observation allowance. It also
includes `feedback_limit_tolerance_rad` and `kernel_thumb_feedback_limits4`,
separate from the strict target `kernel_thumb_limits4`. Missing explicit native
bounds and a zero allowance retain the strict legacy kernel defaults; the
provider derives native bounds only while CBF is enabled.
`check_arm_feedback_goal_range` is reported in status and fault records. When
false, the displayed arm bands remain configuration diagnostics, not a gate on
the current encoder samples; commanded positions remain strictly bounded.
`distance_gradient_invalid` instead identifies an invalid geometry/gradient
report. Nonfinite, stale or skewed feedback cannot authorize new targets. Thumb
feedback outside its configured observation allowance is also rejected for new
verification. These failures do not suppress same-epoch committed-pair retransmission.

The first frame is retained while the same issue/index persists. Recovery or a
different issue starts a new event. Full records are emitted through the existing
launch log at most once per two seconds, with no additional recorder process and
no per-frame logging. Existing status adds only the invalid-frame count and last
fault event/issue; the full input arrays are not republished at status frequency.

## Operator flow check

After deployment and isolated tests pass, use the feature run directory:

```bash
ssh root@10.42.0.5
bash /root/x2-feat-cbf-safety-run/start.sh
bash /root/x2-feat-cbf-safety-run/mirror.sh start
```

The operator controls the mirrored motion. This starts physical arm/hand control;
it is a live effect test with a candidate conversion, not calibration acceptance.
Inspect the existing log for `CBF enabled: mode=enforce` and then
`CBF status: ... mirror_active=true`; calls and published counts must increase.
Only `geometry_modified` counts collision-driven intervention, rather than
ordinary rate limiting. Keep `mapping_verified=false` truthful while the actual
joint-angle alignment is checked after the trial.

Stop the mirror with `mirror.sh stop`; finish the full session with the existing
`stop.sh`, which returns the robot to its vendor standing mode before stopping
processes. Do not use the untouched summit baseline to assess this feature.

## Isolated board tests

Deploy the committed SHA using the unchanged feature deployment package, then:

```bash
bash /root/x2-feat-cbf-safety-wt/src/robots/aimdk/aimdk_dataplane/tools/run_cbf_colcon_tests.sh \
  /root/x2-feat-cbf-safety-pkg-SHA
```

The runner checks the exact SHA, takes the existing build/test lock, refuses a
running robot launch, and loads the package environment in a clean shell. It
builds only this package's C++ tests through `scripts/build.sh`, then uses
`colcon test` and `colcon test-result` for C++, the CBF cases in the shared
`robot_config` test file (`-k cbf`), and the runtime launch test file.
Test discovery mirrors the existing openEuler build exclusion of `fast_calib`
and `sim_models`. It does not install simulation dependencies or claim to
validate the unrelated Gazebo/MuJoCo scene tests on this physical-robot board.
Tests use domain 81 and localhost only. Logs and the tested SHA stay below the
feature run directory. No real-robot startup is part of this command.

The numerical probe also supports `--replay 1000 --left-hand-only` and a
left-only JSON input with `arm14` and `left_hand10`. Such input cannot be
mistaken for a complete measured hand20. Captured arm and hand messages taken
sequentially are only a posture comparison, not proof of runtime feedback skew.

Physical validation requires separate authorization and the feature run tree,
not the untouched baseline. Verify enforce is active before assessing collision
prevention. Numerical fixtures, especially their identity hand mapping, do not
establish a real SDK calibration or continuous physical collision safety.
