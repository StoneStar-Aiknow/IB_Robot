---
name: om-convert
description: "Unified Ascend OM conversion and optimization entry point for LeRobot policy bundles. Use whenever users mention 'convert to OM', 'Ascend OM', 'ATC', 'OM performance', '转OM', '转换OM', '昇腾模型转换', or '优化OM'. Supports existing ACT/PI0.5 workflows and exploratory ports for policy families supported by LeRobot but not yet by IB-Robot. Collects decisions up front, validates the bundle and target SoC, creates an isolated worktree for new ports, coordinates multi-host accuracy validation, and optionally performs measured optimization."
---

# Ascend OM Conversion And Optimization

This is the only user-facing Ascend OM skill. It supports:

1. converting a policy with an existing IB-Robot Ascend workflow;
2. porting a LeRobot-supported policy family that IB-Robot cannot yet convert to OM;
3. optionally optimizing a validated Ascend deployment.

The supported input is a LeRobot policy bundle whose final output is robot `action`. A new policy must
already be supported by the current LeRobot revision. Bare Torch modules, arbitrary checkpoints, and
policy families not supported by LeRobot are out of scope.

## Internal References

Read only the references needed for the selected route:

| Purpose | Reference |
|---------|-----------|
| Existing ACT conversion | `references/act.md` |
| Existing PI0.5 conversion | `references/pi05.md` |
| New LeRobot policy port | `references/port-new-policy.md` |
| Multi-host target workflow | `references/multi-host-validation.md` |
| Accuracy gates | `references/accuracy.md` |
| OM performance baseline | `references/benchmark.md` |
| Optional optimization | `references/optimize.md` |

Do not expose these references as separate skills.

## Upfront Intake

Ask for all foreseeable user decisions in one compact intake before running expensive commands or
editing source. Reuse values already supplied and do not ask redundant questions. Use the `question`
tool when available; otherwise ask directly. Group choices into one interaction where practical.

### Required Intake

Collect:

| Field | Requirement |
|-------|-------------|
| Intent | `Convert only` or `Convert and then optimize`; for an existing OM, `Optimize only` is valid. |
| Decision mode | `Autonomous (Recommended)` or `Approval required`, as defined below. |
| Policy family | User-selected family; a path-only request still requires confirmation after reading `config.json.type`. |
| Bundle path | Explicit user-provided path; never discover or choose a checkpoint automatically. |
| Target SoC | Exact ATC `soc_version`, resolved through the platform probe below. |
| Accuracy limits | Torch-vs-ONNX maximum absolute error, mean L1, and minimum cosine; Torch-vs-OM mean L1 and minimum cosine; or explicit `report only`. |
| Observation source | Existing versioned batch, local LeRobot dataset, or deterministic random smoke input. |
| Torch host | Local/remote, SSH target when remote, IB-Robot path, bundle path, deployment, and device. |
| Conversion host | Local/remote, SSH target when remote, IB-Robot path, bundle path, and experiment directory. |
| Ascend host | Local/remote, SSH target when remote, IB-Robot path, deployed bundle path, deployment, and ACL device ID. |
| Task and seed | Exact task for language policies and deterministic seed/noise settings. |
| Remote execution | Whether the agent may execute commands over the user's existing SSH setup. Never request credentials. |

If all roles use one host, ask once and record that the topology is shared. Host paths are independent:
never assume the same absolute path exists on two machines.

For PI05 also collect its PaliGemma asset, pipeline, FastGELU, and schedule choices from `pi05.md` in
the same upfront intake. For a new policy, collect whether source/runtime changes may include a new
codec and Ascend execution strategy; this is allowed by default in autonomous mode.

### Decision Mode

Ask at the start:

> When a complex, unanticipated decision appears, should I choose the recommended path, continue, and
> record the decision, or stop and wait for your approval?

Offer:

- `Autonomous (Recommended)`: choose the safest reversible option that preserves semantics, continue,
  and append the decision, evidence, alternatives, and rollback to the experiment ledger.
- `Approval required`: stop before each material unanticipated decision and present the recommendation,
  alternatives, evidence, expected cost, and rollback.

Decision mode does not override these hard approval gates:

- approximate math or relaxed user accuracy thresholds;
- quantization;
- changing preprocessing, tokenizer semantics, action semantics, schedule, step count, or valid input
  range;
- destructive replacement of the user's bundle or existing deployment;
- credentials, privileged host changes, or access outside the supplied machines and paths;
- committing, pushing, or deleting a worktree.

In autonomous mode, normal reversible engineering choices do not interrupt execution. Examples:
opset selection after toolchain inspection, static wrapper structure, naming internal roles, narrow ATC
compatibility rewrites, test organization, and reverting a failed candidate. Record each non-trivial
choice in `reports/decisions.jsonl` with:

```json
{"decision":"...","recommendation":"...","evidence":["..."],"alternatives":["..."],"rollback":"..."}
```

If an unanticipated choice is neither clearly reversible nor covered above, approval mode stops;
autonomous mode also stops rather than guessing.

## Bundle And Family Resolution

The user must explicitly provide the bundle path. Resolve a supplied relative path to an absolute path
and report it. Require readable regular files:

- `config.json`;
- `model.safetensors`;
- `policy_preprocessor.json`;
- `policy_postprocessor.json`.

Read `config.json`. The raw `type` must be canonical and must agree with the user's selected family.
Do not route from directory or weight names. If only a path was supplied, ask the user to confirm the
family inferred from `type`.

Route:

| Family status | Action |
|---------------|--------|
| ACT | Follow `act.md`, then shared accuracy and benchmark references. |
| PI05 | Follow `pi05.md`, then shared accuracy and benchmark references. |
| LeRobot supports it but IB-Robot has no Ascend workflow | Follow `port-new-policy.md`. |
| LeRobot does not support it | Stop as out of scope. |

For an unknown IB-Robot family, prove LeRobot support from the current revision by locating its config,
policy/model implementation, factory/registry path, `from_pretrained()` loading, processor handling,
and action output. A `config.json.type` string alone is not proof.

## Platform Probe And `soc_version`

Before compiling, inspect the conversion host. For a local host run in the workspace:

```bash
source .shrc_local
command -v npu-smi
npu-smi info
command -v atc
atc --version
```

Read `/etc/os-release`. For a remote conversion host, run the equivalent probe through the approved
SSH target and workspace path.

Resolve the exact ATC value:

| Result | Rule |
|--------|------|
| Ubuntu conversion host | Require explicit user-selected `soc_version`, even if a local NPU exists. Reuse an exact target already given. |
| Non-Ubuntu with an unambiguous full chip revision | State and use the derived candidate, unless the user selected a cross-target. |
| Probe missing, failed, or family-only such as `310P` | Ask for exact `soc_version`; never guess the revision. |

Preserve the complete revision, such as `Ascend310P1` versus `Ascend310P3`. Record host OS, NPU
evidence, ATC version, target, and resolution source.

## Shared Execution Rules

- Load `source .shrc_local` before every project command; after a build, also load
  `source install/setup.sh` for runtime commands.
- Generate Torch targets before testing OM accuracy. Target generation and OM deployment are expected
  to occur on different machines; follow `multi-host-validation.md`.
- Use one versioned observation batch and the same task, seed, targets, raw targets, and noises across
  Torch, ONNX, and OM checks.
- Perform only two numerical gates: Torch vs ONNX and Torch vs Ascend OM. Follow `accuracy.md`.
- Benchmark every final OM with `ais_bench --loop 50` and compute the invocation-weighted sum. Follow
  `benchmark.md`.
- Do not run `hardware_mock`, LTTng, tracing, or trace-summary validation in conversion or optimization.
- Conversion defaults to conservative FP16. Quantization is never a default conversion step.
- For a new policy, all exploratory source edits must occur in a dedicated worktree created according
  to `port-new-policy.md`.
- Do not create a permanent deployment for every optimization candidate. Preserve reports and
  reproduction commands; publish only the selected result.

## Default Stop Point

For a LeRobot-supported policy not previously supported by IB-Robot, the default task ends after:

1. an isolated worktree contains the required exporter/runtime changes;
2. IB-Robot runs the original Torch deployment and generates reusable targets;
3. Torch vs ONNX is evaluated against the agreed limits;
4. conservative FP16 OM, ACL ABI, Manifest, and runtime loading are complete;
5. Torch vs Ascend OM is evaluated against the same baseline;
6. each OM has an `ais_bench --loop 50` report and the weighted total is calculated;
7. reports, decisions, source diff, and reproduction commands are retained.

Then stop and ask whether to continue with `optimize.md`, unless the upfront intent explicitly selected
`Convert and then optimize`. Even then, optimization may begin only after the baseline passes or the
user explicitly accepts report-only accuracy results.

## Success And Failure Language

- ONNX export success does not imply OM success.
- ATC success without exact ACL-inspected ABI is a compiler artifact, not a deployment.
- A deployment without Torch-generated targets has not passed OM accuracy validation.
- Report-only metrics are not an accuracy pass until the user accepts them.
- Performance optimization is successful only when accuracy remains accepted and loop-50 weighted
  latency improves.
- Never claim mock, robot-task, or real-hardware validation; those are outside this workflow.
