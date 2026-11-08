# Frozen CBF kernel import

Source: `/media/ubuntu/C2BE25D9BE25C6AF1/CBF/cbf/`.

At import, the eight `include/cbf/*.hpp` headers and five `src/*.cpp`
implementations were copied byte for byte, keeping their directory structure
and `x2::cbf` namespace. The source directory is not a Git repository; the
original SHA-256 inventory below identifies that import snapshot without
inventing a commit ID. It is not a claim that every current local file remains
byte-identical: the bounds, exact-target verification and explicit installed-hand
adaptations are documented below.

The kernel uses C++17 and the C++ standard library only. No ROS or AimDK SDK
headers, vendor source, URDF, STL, binaries, or bulk validation data are included.
The package's build integration is handled separately in its CMake configuration.

## Coordinate and hand-target contract

All kernel fixtures use the frozen model's URDF coordinates. The real SDK to
frozen-URDF mapping remains unverified. In particular, an extraction of raw hand
command slots is not a unit, sign, offset, or joint-order conversion:
`extract_thumb_commands()` continues to return `mapping_valid == false`.
The integration must supply an explicit mapping; raw slot extraction alone is
not a conversion. The operator-requested live flow check uses the configured
candidate while keeping its calibration provenance explicitly unverified.
The default frozen thumb pose is an offline fixture, not a measured SDK pose.

Callers must honor `thumb_target_accepted`. A fallback can have
`nonlinear_verified == true` because the actual held pose is safe while
`thumb_target_accepted == false` rejects the requested thumb motion. The proposed
hand target must not be forwarded on that result. `pass_through_hand_and_arm()`
only copies command values; it does not authorize a hand target or validate the
mapping. A false `nonlinear_verified` flag does not certify even the held pose.

## Explicit thumb input bounds

The initial bounds/verification API adaptation changed `include/cbf/rigid_thumb.hpp`,
`include/cbf/safety_filter.hpp`, and `src/safety_filter.cpp` within this kernel.
It adds an optional four-by-two bounds argument to `valid_rigid_thumb_pose()`
and `SafetyFilterInput.thumb_joint_limits_urdf`; both default to the unchanged
`kRigidThumbLimits`. Current and target thumb validation use the supplied bounds.
Every bound must be finite with `lower <= upper` and a finite span no greater
than two pi; NaN, infinity, reversed intervals, and excessive spans are rejected.
The existing angle tolerance and strict default domain are preserved. FK,
collision geometry, distance calculations and generated constants retain their
imported values. Later command-box and recovery changes are documented below. Runtime mapping and
diagnostic integration live outside these three kernel files.

A caller may explicitly supply the motion domain obtained by mapping its native
command/feedback ranges into model coordinates. Passing this input check is not
a collision-safety certificate or proof of SDK calibration. The four root-axis
native limits used by the tests come from
`manipulation_execution/mirror_session/hand_overlay.py`, slots `[0, 1, 10, 11]`:
lower `[-1.117, -0.05, -0.03, -1.64]`, upper `[0.03, 1.64, 0.87, 0.05]`.
The historical scale `[-1, 1, 1, -1]` and offset
`[0, -1.6424, -0.576, -1.6424]` remain inferred. The feature profile explicitly
uses that candidate for the operator-controlled live flow check; numerical
fixtures do not establish its physical calibration.
For example, native open zero maps outside some independent-URDF nominal bands;
only explicit mapped bounds admit that input domain. Defaults are not widened.

The integration fixture keeps `verified: false`. Shadow preserves the original
arm/hand command while reporting `thumb_current_urdf` and `thumb_target_urdf`
with an `unverified_mapping/` reason. Enforce uses the supplied conversion and
retains the same unverified diagnostic prefix. Calibration provenance is not
a second motion admission gate; absent mapping arrays still produce
`mapping_required`. Feedback validity and all geometric acceptance flags remain
mandatory, and a rejected hand target cannot accompany the new arm target.
Inverse-mapping the old safe URDF baseline provides a native fixture; it does
not establish a verified SDK pose or authorize real motion.

## Exact target and sampled-path verification

`SafetyFilterInput.check_current_arm_goal_range` defaults to true for backward
compatibility. A live caller can set it false when encoder observations must be
evaluated geometrically outside the configured goal bands. The kernel still
requires finite current angles and valid configured intervals, and it never clips
those observations before FK/path evaluation. Nominal/output command limits and
all collision checks retain their original semantics. This is independent of
thumb current/target validation; it does not authorize invalid hand commands.

`verify_arm_target(input, config)` is an additional entry point declared in
`include/cbf/safety_filter.hpp` and implemented in `src/safety_filter.cpp`.
It checks the supplied current pose, exact nominal target, and sampled arm/thumb
path with the existing FK, distance, and clearance routines. It does not run
the QP or optimize a hold target. Ordinary success returns exactly `q_nominal`,
status `Normal`, `nonlinear_verified == true`, and `thumb_target_accepted == true`.
Invalid input, an unverified/worsening path, or deadline exhaustion cannot authorize the
requested hand target; callers must inspect the status and acceptance flags.

This is geometric verification in model URDF coordinates, not SDK calibration
or a command-rate limiter. The caller owns verified mapping and native limits,
feedback freshness/skew, admission/ownership, application joint limits, command
step/rate limits, publication budget checks and truthful publication history. In particular,
tracking lag is measured about the actual pose while command steps are bounded
about the previously published target. After those caller checks, a safe exact
bounded nominal path can be accepted without QP optimization. An unsafe path
may go through the original CBF projection, followed by strict verification of
the projected output. A previously accepted target can be checked without another QP;
the adapter distinguishes this new check from merely retransmitting an old command.
FK and distance calculations are unchanged. The projection now clips its
iterates to `SafetyFilterInput.command_joint_limits` (a caller-supplied subset
of the original arm limits), including the published-command rate box.

## Bounded recovery and hand ordering

`SafetyFilterConfig.allow_airwall_recovery` defaults to false for existing
strict callers. The live adapter enables it. If the initial pose is already
inside a barrier, each affected pair must have nondecreasing sampled margin
until it reaches zero; unaffected pairs retain their original clearance.
Partial withdrawal is allowed. An initially penetrating proxy may withdraw
but may not penetrate further. Successful recovery sets `recovery_verified`
and status `AirwallRecovery` or `PenetrationRecovery`; its endpoint margin
may still be negative. `nonlinear_verified` then refers to the explicitly
checked state-relative recovery constraints, not a false static-clearance claim.

If the projected endpoint fails a recovery-path sample, a bounded repair uses
the endpoint derivative of `d(t)-d(t_previous)` and clips to the same command
box. Repairs use the existing path-recovery iteration count and remaining
deadline; an exhausted solve never authorizes a new target.

`verify_thumb_target_with_arm_held()` checks only whether the new thumb can
move at the actual measured arm pose. Its returned arm coordinates are a
geometry query, **not an arm command**, and may be outside the goal envelope.
The adapter can defer a whole gesture while the arm withdraws, then retry it
once feedback permits the hand-first path. This avoids assuming synchronized
arm and hand actuators without adding a process or a hardware dependency.

The mirror adapter now gives this hand-only query a bounded sub-budget, leaving
time for verified arm withdrawal if the new whole gesture must wait. The node
and adapter share an absolute monotonic publication deadline. The kernel also
checks its remaining budget immediately before and after recovery-gradient
queries and consistently reports `DeadlineFallback` on exhaustion.

Within a solve, the last nonlinear-recovery FK/distance report is reused only
when the output candidate is exactly equal to the joint array used for that
report. A changed final candidate is evaluated again. An unchanged projected
output also reuses the kernel's existing verification in the adapter; an arm
replacement still gets a fresh verification. These are same-input computation
reuses, not cached safety claims across frames. Geometry, distance formulas,
path resolution, clearance, iteration limits and command bounds are unchanged.

## Focused test adaptation

`../test/test_cbf_kernel.cpp` adapts deterministic cases from the original
`tests/phase_c_tests.cpp` and `tests/rigid_thumb_tests.cpp`: natural-standing
acceptance, the existing `+0.04` shoulder request, invalid nominal input and an
invalid thumb mapping, exhausted deadline, the exact penetrating thumb fixture
with a zero arm step, and unchanged hand data with an invalid SDK mapping flag.
The raw hand passthrough values come from the original test and are not physical
geometry or approved robot commands. No random stress tests are imported.

Four additional focused cases cover strict default versus explicit mapped thumb
bounds, both current/target validation and invalid bounds, short/long C++ config
arrays, and unverified native mapping diagnostics in shadow versus enforce
refusal. They assert unchanged hand commands; historical conversions remain
explicitly unverified fixtures.

Three exact-verifier cases cover the unchanged natural-standing target with
the measured shoulder 0.06 rad lower, the known penetrating thumb and invalid
inputs, and an exhausted deadline. The lag case requires exact nominal output
and successful acceptance flags, with zero projection iterations. The existing
`CbfSafety` TrackingLag test and its assertions are retained unchanged.

Ordinary test cases use a 100000 microsecond budget to separate correctness from
timing performance; the deadline case deliberately uses `1e-9` microseconds and
zero guard. Production filter defaults remain unchanged. The three new verifier
cases were not compiled or run during this editing task; the integrator owns
build wiring and subsequent acceptance through `colcon test`.

## Explicit installed-hand selection

`SafetyFilterInput`, `KinematicsInput` and `KinematicsOutput` carry
`hands_installed`, ordered left/right and defaulting to `{true, true}`. An
absent hand excludes only pairs containing that side's palm or thumb. The
frozen pair table, all proxy dimensions, wrist mounts and arm transforms are
unchanged. The distance minimum, gradient constraints, recovery selection,
intermediate path samples and exact verifier all honor the same selection.
Excluded distance entries use a finite sentinel and never compete for the
minimum; they are excluded pairs, not measurements. The real integration must
explicitly configure installed hardware and reject incompatible reported types.

The adaptation touches `kinematics.hpp`, `safety_filter.hpp`, `kinematics.cpp`,
`distance_gradient.cpp` and `safety_filter.cpp`. Tests retain both-hand defaults
and cover exclusion of an absent right hand while left-hand and arm collisions
remain detectable. Runtime feedback is required only for installed hands; no
missing native angle is filled in as measured zero.

## Native observations versus commanded joint bounds

`SafetyFilterInput.current_arm_joint_limits` defaults to the unchanged generated
operator limits for existing callers. The provider adapter supplies the selected
native profile's observation bounds separately. Accepted measured angles are
kept unchanged in FK, gradients and swept checks. Candidate/output commands
still use the generated command bounds and the adapter's original, possibly
tighter application limits and margin. No arbitrary feedback tolerance widens
the commanded range. Unaccepted fallback values remain finite and command-bounded;
callers must continue to honor acceptance flags and paired-history validation.

The captured right-shoulder frame had feedback 0.052634239196777344 rad versus
the operator upper bound 0.05236 rad; the actual provider bound is 0.061 rad.
Regression fixtures replay that exact arm/thumb input and preserve strict
default-domain rejection and rejection of illegal command targets.

## Invalid input diagnostics

`SafetyFilterResult` retains a diagnostic predicate, index, value and bounds at
existing invalid-input return sites. The diagnostic index is an arm joint or
thumb pose slot for angle errors, and a primitive-pair index for an invalid
distance/gradient report. The fields themselves do not alter rejection,
numerical tolerance, QP behavior, geometry or sampled-path acceptance. The
observation domain is selected explicitly as described above.

These fields describe a failed input; they do not authorize the returned arm or
hand target. The adapter retains the first failing input and stage for its native
node log. Tests exercise the old rejection together with the diagnostic fields,
and the original source inventory below remains an import record.

## Source SHA-256 inventory

Paths below are relative to the source `cbf/` directory. The 13 header and
implementation entries identify the original import snapshot, not current
whole-file equality. The final two entries identify the original source test
fixtures. Current hashes for the three adapted API files follow separately.

```text
ba565784d93e95ea4164a4b2eb3d7e0549d59ee1d238bd3ba2fcf315ee0e7211  include/cbf/command_adapter.hpp
1dc8e541fd3f734c036bee7fc8acd9ee32da9e4e8b81299a1cc2868520ee8e74  include/cbf/distance_gradient.hpp
83faf28713561a379535b66a4e8443f919d9341dbc548f6f0e1fe7e32dca7031  include/cbf/generated_constants.hpp
a942ffd9302845b75a93f181863dc7280ac8abc9fba772dd77a79a4a0ba9ab6b  include/cbf/kinematics.hpp
4c9d5aa118abf8a0d9aa3287c8a12ac932b46023e8fbb6c5082a8fb494b3f3fa  include/cbf/math.hpp
0356f54e7991fb23db147ed4e628f72d09a87d368f05e74b80a7bd90d42666c1  include/cbf/proxy_model.hpp
ad7c0d622fe870cb24bd0fe3fe4008d2611255cef9722f6450be5c58d69f311a  include/cbf/rigid_thumb.hpp
ea432f95d207517231d4f7532a111a8db816831c611dddb5749ffdae55418353  include/cbf/safety_filter.hpp
0e1396c58fe31145b84d6a35511e39c80316434d4b1c3d25dc75dec006620676  src/command_adapter.cpp
c01ea5f689d3e58b4cff5ba8a216044a1f0fb0e689b637b0e852b11c245aae16  src/distance_gradient.cpp
8bdea2f4fcc28f19cd8c761ecb4577ba3bdbbe1e10dd7c0d0375bfc8b90d1ae5  src/kinematics.cpp
774f15f57667dfc2a3b0ef826e06d66aeb3733666b1fd68fe4c2aaabeb830a2b  src/proxy_model.cpp
e39a001c7e43b00ff984f8e0786c450df4d113b8858e8cd5c0f68fd0721b5913  src/safety_filter.cpp
3aec0429bc3fae431c3067a71b8e0b1029cdd2c41c9be31929cc5649ce1e5738  tests/rigid_thumb_tests.cpp
10ec49bcf72c9df2be873d2f600485460221a38cd928eb097f099fe87f98c8d5  tests/phase_c_tests.cpp
```

## Current adapted API SHA-256 inventory

These target hashes were read after the bounds, exact-target, installed-hand,
diagnostic and native-observation-domain
API changes; they record current bytes, not a
compilation, test result, or mapping verification. The original import inventory
above is retained separately.

```text
5e9adcc2fc08e0b20039c2d79f5bacd99b98d4cf479444b7e0dba48a6d375e5c  include/cbf/rigid_thumb.hpp
ddba3757879f3a1f47d3b0a5ffe98acfdbb8ed2b649f0a4a7de3d1a1bac51721  include/cbf/kinematics.hpp
7bd36fddab98c258d95c74add2263fcd822ac0bd6641663d8cde7838777e3543  include/cbf/safety_filter.hpp
056b215dab5cffda262c9318b79012e36e5c3c114e2348f156de5d59f5d476be  src/kinematics.cpp
fc735b2c236e71fc151517b1e25da4e7b67573a38228ed60967af72d8051de9f  src/distance_gradient.cpp
0bd0dea581d4d3ece1c5675825c0b6d003f9a07a49c8d597b59a377f96552722  src/safety_filter.cpp
```
