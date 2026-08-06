# Port A New LeRobot Policy To Ascend OM

Use this workflow only when the current LeRobot revision supports the policy family and IB-Robot has
no complete Ascend OM conversion/runtime workflow for it. The policy bundle must ultimately produce
robot `action`.

## Entry Contract

Require the shared intake from `../SKILL.md`, including decision mode, explicit bundle path, three-host
topology, accuracy limits, observation source, task/seed, and exact `soc_version`.

Prove LeRobot support by finding:

- the policy configuration and canonical `type`;
- the policy/model implementation;
- factory or registry loading;
- `from_pretrained()` or equivalent bundle loading;
- processor integration;
- an action output feature and inference path.

If any of these is absent, stop: the family is not supported by this workflow.

Then classify IB-Robot support:

- `Torch supported, OM unsupported`: first run the existing production Torch deployment. Reuse the
  existing codec, registry entry, processors, and bundle-local remote assets. Focus changes on export,
  Ascend bindings, and execution strategy.
- `IB-Robot unsupported`: add the smallest production Torch codec/registration required to load the
  LeRobot bundle and generate targets before implementing OM support.

Prefer a supplied Torch-supported bundle over reconstructing one. It may already vendor otherwise
remote assets such as tokenizer files, avoiding unnecessary downloads and revision drift.

## Isolated Worktree Gate

Never explore in the user's current worktree. First record the current root, branch, and `HEAD`, then
create a timestamped branch and sibling worktree:

```bash
git worktree add -b "om-port/RESOLVED_POLICY-RESOLVED_TIMESTAMP" \
    "RESOLVED_WORKTREE_PARENT/om-port-RESOLVED_POLICY-RESOLVED_TIMESTAMP" \
    HEAD
```

Before creating it:

- inspect `git worktree list` and branch names to avoid collisions;
- verify the parent exists and is the intended worktree root;
- do not stash, reset, clean, or modify the source worktree;
- do not base the experiment on a different branch unless the user explicitly requests it.

If creation fails, stop. Do not fall back to editing the current worktree. Do not commit, push, remove,
or prune the experiment worktree without explicit user instruction.

Create an experiment directory outside Git-tracked source when possible:

```text
<experiment>/
├── current/
├── reports/
│   ├── decisions.jsonl
│   ├── torch-vs-onnx/
│   ├── torch-vs-om/
│   ├── ais-bench/
│   └── experiments.jsonl
├── inputs/
├── final/
└── reproduce.sh
```

Large weights, ONNX, OM, ACL dumps, observations, and profiler data must not be accidentally tracked.

## Phase 1: Run The Original Torch Policy

Before writing an exporter, make IB-Robot run the original Torch bundle through its production policy
loading path. Reuse existing generic Torch deployment packaging and processor infrastructure. Add only
the family-specific codec/registration needed for correct observation and action semantics.

Validate:

- bundle metadata and processors load without external ambiguity;
- native weights are loaded from `model.safetensors`;
- all required observation features are present;
- inference returns finite action with the configured shape;
- task and control inputs such as noise/timestep/cache are explicit;
- a loading failure cannot silently continue with randomly initialized weights.

If changes to `libs/lerobot` are unavoidable, keep them isolated and later use the
`ibrobot-lerobot-patch` workflow. Never commit a raw submodule pointer as part of ordinary work.

Generate the reusable Torch target before OM validation, following `multi-host-validation.md` and
`accuracy.md`.

## Phase 2: Analyze The Deployment Boundary

Write a short analysis into `reports/port-analysis.md`:

- source policy call path;
- preprocessing and postprocessing;
- ordered external tensor inputs with shape and dtype;
- ordered outputs and action semantics;
- dynamic dimensions and the static deployment values selected;
- language, history, noise, timestep, cache, or recurrent state;
- Python/data-dependent control flow;
- unsupported or custom operators;
- single-graph versus multi-role proposal;
- codec reuse or required new codec;
- required Manifest bindings and Ascend execution strategy.

Prefer one graph. Split only when correctness or repeated-computation structure requires it. Do not
split merely because PI05 does.

## Phase 3: Portable ONNX Baseline

Implement model-family-specific export under `src/model_utils/model_utils/<family>_export/` unless a
small extension to an existing exporter is clearly correct. Keep reusable Manifest, ABI, batch, and
comparison behavior in shared modules.

The first ONNX must:

- use static batch 1 and static deployment shapes;
- use FP16 inputs, weights, and operations by default, with documented FP32 islands only where FP16
  changes correctness or the target kernel requires them;
- avoid approximate and NPU-only operators;
- expose explicit stable input/output names;
- return only deployment tensors;
- preserve action semantics and all control inputs;
- treat external-data files as part of the ONNX artifact;
- pass `onnx.checker` and feasible shape inference;
- be reproducible from one recorded command.

For multiple roles, record role order, handoff tensors, and invocation counts. Validate every external
output and the final action, following `accuracy.md`. Do not proceed to ATC until Torch vs ONNX passes
the user limits or the user explicitly accepts report-only results.

## Phase 4: Conservative FP16 OM

Compile the portable design with the resolved exact `soc_version`. Start with:

- static shapes;
- FP16 ONNX;
- `--precision_mode_v2=origin` initially; after accuracy passes, optionally test
  ATC-default `--precision_mode_v2=fp16` against the same targets;
- do not pass the literal value `default`;
- no legacy `--precision_mode` argument;
- no quantization;
- no approximate operators;
- no schedule/step/action changes;
- no performance-motivated graph split;
- only narrow, explained ATC compatibility rewrites.

Classify ATC failures before modifying code: unsupported op/opset, shape inference, dynamic dimension,
domain/parser, dtype, external data, or compiler bug. Add a regression test for every non-trivial graph
rewrite. In autonomous mode choose the safest semantics-preserving fix and record it; in approval mode
stop before a material architecture change.

Scan the full ATC log for warnings similar to `does not hit high priority library`. Treat them as
performance and possible precision risks: record the operator, dtype, format/layout, shape, and chosen
fallback kernel. Do not ignore the warning because compilation succeeded.

## Phase 5: ACL ABI, Manifest, And Runtime

Use Python ACL to inspect each exact final OM. ABI must contain compiler-reported input/output names,
indices, dtypes, and shapes. Never infer ABI from ONNX, reuse ABI from another build, or compile again
after ABI inspection.

Reuse Manifest helpers and the generic ACL model runner. Add only the model-specific pieces needed for:

- policy codec;
- semantic tensor bindings;
- role order and device links;
- loops or state transitions;
- final action output.

Update the owning package README whenever public CLI, runtime responsibility, configuration, tensor
contract, or known limitations change. Add tests for exporter behavior, ABI/Manifest packaging, codec,
backend strategy, and production Manifest loading.

## Phase 6: Accuracy And Baseline Performance

Follow:

- `multi-host-validation.md` to move observation batches and Torch targets;
- `accuracy.md` for Torch vs OM;
- `benchmark.md` for every final OM with loop 20 and the weighted total.

Do not run hardware mock, tracing, or LTTng.

## Default Completion

Stop after conservative conversion, both accuracy comparisons, and the loop-20 baseline. Report:

- source and experiment worktree paths/branches/SHAs;
- changed files and dirty diff summary;
- bundle and deployment identities;
- portable ONNX and final OM/ABI paths and hashes;
- Torch-vs-ONNX and Torch-vs-OM verdicts;
- per-role and weighted latency;
- decision ledger and reproduction script;
- tests run and remaining gaps;
- experience hits-among-evaluated and catalog coverage from `experience-ledger.md`.

Ask whether to continue with `optimize.md` unless optimization was selected in the upfront intent.
