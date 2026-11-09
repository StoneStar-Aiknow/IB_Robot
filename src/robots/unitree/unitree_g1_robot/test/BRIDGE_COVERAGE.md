# Runtime bridge coverage

## Pre-commit verification of supervised commissioning changes

The final implementation was rerun through `colcon test` on 2026-10-28:
runtime 229 passed with zero skips; adapter 28 passed with one SDK-absence
guard skip; catalog 118 passed with four existing migrated-profile skips.
All three package runs completed successfully. ROS used localhost/domain 146
and the vendor wire fixtures explicitly used loopback/domain 148. These results
supersede the earlier targeted-only result for the commissioning gate.
Dual-platform Docker verification remains deferred for the WIP PR; human code
review and software physical-stop acceptance remain pending.

The unconditional collection skip in `test_runtime_bridge.py` has been removed.
The harness factory now tears down every runtime it creates. Previously it
returned a factory without the promised teardown, leaving nodes and SDK
resources alive across tests. The mock records received requests before applying
silent-response injection so timeout/no-resend assertions observe actual receipt.
Both wire-test modules use domain 148 because the official SDK channel factory
cannot switch domains within a pytest process. Both harness factories now clean
up their nodes and resources; the suites are run sequentially on loopback.

## Coverage layers

- `test_runtime_bridge.py`: real runtime callbacks, ROS services/actions and a
  subprocess vendor mock using the official SDK wire types on test domain 148.
- `test_arm_stream_safety.py`: actual bridge methods with inert I/O, exercising
  ownership/FSM refusal, stop during admission, publication failure, no automatic
  retry, local-vs-remote handover distinction, and latch retention.
- `test_vendor_gateway.py` and adapter tests: completed IDL frame CRC, field
  projection, read-only service discovery and SDK result normalization.

SDK-dependent tests still skip when the SDK is absent. Such skips are missing
coverage, not successful validation. Formal package results must come from
`colcon test`; a passing bare-pytest reproduction is only a diagnostic result.

## Corrected interpretation of earlier failures

Earlier hangs and crashes do not establish a vendor-only root cause, and process
separation does not establish production immunity. Fixture cleanup defects and
test composition must be considered alongside SDK behavior. The earlier claim
that launch/conformance checks replaced all bridge coverage is withdrawn.

## Limits

The mock demonstrates software behavior, not real robot acceptance. A successful
zero-weight publication has no remote arm-handover acknowledgment. Stream stop
and exit retain the latch and report this uncertainty; these tests must not
expect an acknowledged physical stop or silently clear that latch.

## Formal verification — 2026-10-28

`python3 -m colcon test --merge-install --base-paths src --packages-select
unitree_g1_adapter unitree_g1_robot --executor sequential
--event-handlers console_direct+ --return-code-on-test-failure` completed with
exit status 0 after sourcing this worktree's `.shrc_local`.

- `unitree_g1_robot`: 216 passed, zero skipped, including 29 wire bridge cases
  and 22 inert-I/O arm safety regressions.
- `unitree_g1_adapter`: 28 passed, one skipped. The skip exercises SDK absence
  and is inapplicable because the SDK is installed; real SDK frame tests ran.
- Tests used `ROS_LOCALHOST_ONLY=1`, ROS domain 42, vendor test domain 148,
  and CycloneDDS explicitly bound to loopback with multicast disabled.

These are offline package results. No hardware motion, firmware compatibility,
remote handover, or dual-platform Docker acceptance is claimed.

## Preset-only follow-up — 2026-10-28

After adding the operator-selected `arms_horizontal` catalog entry and separate
`g1_29dof_arm_presets` runtime profile, formal `colcon test` results are:

- runtime: 226 passed, zero skipped (30 bridge cases). The new bridge case
  reaches ACTIVE with both hand feeds absent, executes only action 15 against
  the mock, rejects recovery/stream commands, sends no base/FSM commands, and
  refuses to claim a verified physical stop.
- adapter: 28 passed, one SDK-absence guard test skipped.
- catalog: 118 passed, four existing migrated-profile tests skipped; includes
  compilation of the new one-skill profile and never-retry binding.
- robot_config targeted selection: six passed, one unrelated existing skip;
  1622 tests deselected. This is not a full robot_config suite result.

The mock and all wire-test SDK facades now explicitly initialize on `lo`.
An SDK-supplied Domain configuration can override environment configuration;
`CYCLONEDDS_URI` alone is not the isolation claim.

Catalog compilation and mock/runtime action execution were checked separately;
a live full-Gateway action execution has not been claimed. Physical launch and
node initialization remain disabled for this profile. No robot was contacted
for these tests, and no automatic action 99 was issued.

## Full-stack live binding check — 2026-10-28

Started `embodied_bringup/embodied_pipeline.launch.py` with
`robot_config:=unitree_g1_arm_presets use_sim:=true control_mode:=named_motion
authorize_motion:=false with_perception:=false`. The ROS graph used domain 146
and localhost-only discovery; both vendor SDK processes bound to `lo`, domain 148.

The real launch pipeline waited for the mock-backed runtime to reach ACTIVE,
bound `motion.named`, and generated a consumer configuration snapshot. Passing
that generated snapshot via `robot-skill --config-path ...` produced:

- `status`: exit 0, `ok=true`, `control_plane_state=READY`,
  `control_plane_ready=true`, `motion_authorized=false`.
- `arms_horizontal`: present and planner-visible, execution readiness false
  with `MOTION_NOT_AUTHORIZED` (expected for this read-only check).
- `list-skills`: exit 0, exactly the `arms_horizontal` skill.

This resolves the earlier `description_required` invocation failure for the
offline deployment. The original YAML is launch input; CLI commands use the
snapshot generated by that specific launch, not a hand-authored substitute.
The processes were stopped after inspection and launch exited 0. The snapshot
is historical evidence, not a currently running Gateway or reusable identity.

Local evidence: `/home/lhj/orca/tmp-sdk-verify/gateway-binding/` contains the
launch log, status/list output and generated snapshot path. No execute,
confirm, recovery, service-switch or physical-robot query was issued in this
check. Full-Gateway action execution and hardware enablement remain unverified.

## Full-Gateway mock execution — 2026-10-28

Follow-up to the binding-only check: launched the same isolated simulated
deployment with `authorize_motion:=true` (ROS domain 146, localhost-only;
vendor mock/runtime explicitly on `lo`, domain 148). No physical deployment
was started or queried. The generated consumer snapshot was used by every CLI
call; no original-config shortcut was used.

1. `status`: READY, motion authorized, only `arms_horizontal` exposed and ready.
2. One `run-workflow` request, ID `g1-mock-horizontal-20261028-001`, with exactly
   `[{"schema_version":1,"skill_name":"arms_horizontal"}]`.
3. CLI emitted the prepared workflow presentation followed by terminal
   `state=SUCCEEDED`, `success=true`, exit 0. Execution ID:
   `72f6e46b-5360-4689-95a7-5c87282add2c`.
4. Subsequent `status`: READY, `busy=false`, empty active task ID.
5. `validate release_arm`: exit 2, `UNKNOWN_SKILL`; no recovery execution request.
6. The owned launch group was stopped and launch exited 0.

Evidence is retained in
`/home/lhj/orca/tmp-sdk-verify/gateway-execution/` (launch log, snapshot path,
before/after status, workflow presentation/terminal, and rejection output).
This full-stack check did not enable a vendor wire recorder; exact vendor-side
action-15/no-base/no-mode/no-auto-recovery assertions remain evidence from the
separate bridge regression, not new wire observations from this run.

The full Gateway mock happy path and unknown-skill refusal are now verified.
This does not verify physical movement, FSM-802 action compatibility, physical
stop/cancellation, or transport-failure handling through the full Gateway.
Physical preset launch remains disabled. All test processes were stopped; the
saved snapshot and registry identity describe this completed run only.
