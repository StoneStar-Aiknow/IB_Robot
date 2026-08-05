# Optional Ascend OM Optimization

This workflow is optional. Do not enter it merely because conversion succeeded. Enter only when the
upfront intent selected optimization or the user confirms after the conservative baseline.

## Optimization Intake

Optimization may happen days after conversion. Always refresh the Torch and Ascend host topology before
planning: SSH targets, IB-Robot paths/revisions, bundle/deployment paths, observation/target/raw-target/
noise package, processor/tokenizer and other bundle-local external-asset paths and hashes, device IDs,
and baseline report locations. Then ask unresolved optimization choices together:

- target performance goal, if any;
- whether approximate math is allowed;
- whether static input ranges/shapes may change;
- whether graph splitting, cache, or parallel roles are allowed;
- whether quantization is allowed;
- accuracy threshold changes, if requested.

Default all risky choices to disabled. Quantization is not a mandatory rung and is never enabled by
autonomous mode without explicit user selection.

Before changing code, present an optimization ladder plan tailored to the current model and baseline.
For each proposed rung show evidence, expected gain, accuracy risk, artifacts/commands, and the stop or
rollback condition. In approval mode wait for confirmation; in autonomous mode record the plan and
continue with the recommended first rung.

## Candidate Discipline

For each candidate:

```text
one principal hypothesis
-> source/export change
-> required accuracy gates
-> ATC + exact ABI
-> ais_bench --loop 20 for every role
-> weighted total
-> keep or rollback
```

Append to `reports/experiments.jsonl`: candidate ID, hypothesis, evidence, source diff/commit, commands,
artifact hashes, authoritative validation-package identity, accuracy, latency, verdict, and rollback.

Optimization candidates produce outputs and metrics only; they never regenerate action targets. If
different hypotheses produce nearly identical unexpected accuracy failures, pause the ladder. Do not
attribute the failure to a graph region until the accepted baseline reproduces under the exact same
validation package and command contract. Treat cross-directory target/noise composition or a
non-reproducing baseline as a harness/provenance failure, invalidate affected verdicts, and rerun them
after correction.

Do not create a permanent Manifest deployment for every candidate. Use `current/` for working artifacts,
retain reports and reproduction commands, delete rejected large ONNX/OM files when no longer needed,
and publish only the selected final generation.

Keep one active large candidate. After each verdict, immediately delete rejected ONNX, OM, ABI,
temporary bundles, tensor dumps, and large `msprof` payloads after extracting the summary needed for
the report. Check experiment size and file count regularly. Reports, hashes, source diffs, and
reproduction commands are the durable record, not every binary candidate.

## Optimization Ladder

The order is evidence-driven. Skip a rung when profiling shows it is irrelevant.

An earlier `attempted_no_gain` or regression lowers priority only for a materially similar graph and target.
It is not a permanent skip rule. Re-evaluate when policy family, shapes, compiler, dtype, or exact SoC changes.

For Ascend310P, read `ascend310p-optimization.md` before proposing the ladder. Default to opset 17 and
end-to-end FP16 with only accuracy-proven FP32 islands. Give PFA low priority and prefer standard-ONNX
rank-3 BMM rewrites.

### 0. Freeze The Baseline

Require accepted Torch-vs-ONNX, accepted Torch-vs-OM, all OM loop-20 logs, weighted totals, input and
artifact hashes, toolchain versions, and invocation counts.

Also freeze one authoritative native-Torch validation package. Record its manifest and checksum identity
in every candidate entry. Before the first candidate, rerun the accepted baseline against that package
and require the recorded accuracy to reproduce. This control prevents a stale or mismatched target/noise
package from being misdiagnosed as an optimization regression.

### 1. Remove Data-Path And Structural Waste

Inspect repeated dataset/buffer creation, avoidable H2D/D2H, host round-trips between roles, unused
outputs copied to host, missing buffer reuse, repeated invariant prefix/embedding/cache work, and
duplicated image encoding. Prefer reusable generic ACL changes over model-specific copies.

For serial OM roles, bind producer outputs and consumer inputs to one ACL device allocation through
Manifest device links, following PI05. Keep allocation ownership centralized and verify normal inference
performs no D2H/H2D for linked tensors.

### 2. Profile And Rank ROI

Use `msprof` as the primary diagnostic profiler; final performance remains ais_bench loop 20. Check AICPU
fallback, host/runtime overhead, OP-type cumulative duration, normalized source-module names, and
`mac_ratio`, `mte1/mte2`, and vector ratios.

Scan ATC logs before profiling. Warnings similar to `xxx op does not hit high priority library` are
high-signal: the op may be using a low-priority fallback because dtype, format/layout, shape, or op
form is ineligible. Record every warning and correlate it with `msprof`. Prefer FP16 because it more
often reaches high-priority kernels, but prove accuracy and do not globally force numerically sensitive
operations to FP16.

Rank by measured share, removability, expected benefit, implementation cost, accuracy risk, and risk of
being invalidated by later quantization/layout changes. Do not optimize a cheap non-bottleneck merely
because a fused operator exists.

Re-rank after every large gain using invocation-weighted latency. A repeatedly invoked Action Expert can
become dominant only after VLM optimization.

For repeated camera encoders, include an internal camera-batch candidate when core utilization can improve.
PI05 has shown meaningful gains from this structure. A weak Ascend310P3 result does not rule it out on
Ascend310P1, whose additional cores may benefit more from a larger vision batch. Preserve the external camera
ABI and token order, then decide from exact-SoC loop-20 and final-action measurements.

### 2A. Prefer Opset 17 And FP16 On 310P

For Ascend310P, first export opset 17 and inspect whether standard `LayerNormalization` survives. Opset
<=16 can decompose LayerNorm into AI CPU primitives. This is not universal: roles without eligible
LayerNorm may show no gain.

Make QK, PV, Linear, and other MatMul paths FP16 by default. Test rank-3 batch/head flattening together
with FP16; an FP32 rank-3 candidate can regress even when rank-3 plus FP16 wins. Retain FP32 only for
islands required by final-action accuracy.

### 3. Eliminate Exact Glue

Try semantics-preserving removal of unused outputs, Cast/Slice chains, materialized Expand/Tile,
redundant shape graphs, and avoidable layout conversion. For MQA, `num_kv_heads == 1` may allow MatMul
broadcast instead of `repeat_kv`; do not apply this to GQA where the head dimension cannot broadcast.

### 4. Diagnose Fusion-Induced Precision Drift

When portable ONNX passes but OM fails accuracy, compile one diagnostic candidate with ATC graph and
UB fusion disabled using a toolchain-valid fusion-switch configuration. Preserve the exact switch file
and command. This is a diagnosis, not an automatic final deployment:

- accuracy restored: identify the offending pass or pattern before replacing operators;
- accuracy still bad: fusion is unlikely to be the primary cause; continue precision localization;
- performance is irrelevant for this diagnostic candidate.

Do not invent a fusion-switch syntax; inspect the installed CANN documentation/examples. If the
toolchain cannot disable all relevant fusion safely, record the limitation instead of claiming the
experiment ran.

### 5. Use Exact NPU Fused Operators

A candidate requires all three:

1. target SoC has a kernel;
2. CANN ONNX plugin has a parser in a compatible opset/domain;
3. model subgraph has identical mathematics, layout, optional inputs, and output semantics.

Check CANN ops-info and `libops_all_onnx_plugin.so`; do not infer support from torch_npu API presence.
Reusable PI05 examples are exact RoPE/RMSNorm and, where weight order and formula match, GeGLU.

On Ascend310P, PromptFlashAttention is low priority. Warn that the parser may require private
`NPUPromptFlashAttention` and P1/P3 deployments commonly cannot use a portable PFA path. Prefer standard
ONNX BMM unless the user accepts custom-domain lock-in and exact target support is proven.

Beware hidden costs: ND-only fused operators can add TransData, alignment can add Pad/Slice, bool ops
can fall to AICPU, and an available parser may have no target kernel. Measure the complete role.

After an NPU custom-domain replacement, standard ORT may no longer run. Keep the portable ONNX
equivalence report, prove the replacement subgraph separately where possible, mark custom-graph ORT
as not applicable, and require Torch-vs-OM.

### 6. Remove Dtype Islands

Target measured fp16/fp32 Cast islands around attention, softmax, norm, reductions, masks, and MatMul.
Keep finite mask sentinels valid in the chosen dtype. A local FP16 rewrite is approximate unless proven
exact and therefore must pass the applicable accuracy gates.

### 7. Scan Static Shape/Tiling Choices

Only scan dimensions whose valid business range permits change, such as padding/sequence length, image
resolution, or chunk shape. Benchmark every point with loop 20. ATC tiling has discontinuous slow bands;
never predict winners from divisibility, powers of two, or prime factors. Changing valid input range or
truncating data is a hard approval gate.

### 8. Approximate Operators

Consider FastGELU or similar approximations only after user approval and hotspot evidence. Different
GELU approximations are not interchangeable merely because cosine is high at one layer. Validate final
action against the original Torch targets.

### 9. Optional Quantization

Run only if the user explicitly selected quantization. It is useful when GEMM/BatchMatMul and weight
movement dominate after glue removal.

Requirements:

- representative real LeRobot dataset observations;
- recorded calibration provenance;
- no random calibration fallback;
- start with selected linear layers;
- exclude sensitive norm, attention BMM, final action head, and empirically sensitive layers by default;
- validate each quantized role and the complete action;
- rerun loop 20 and weighted totals.

Quantization may rebuild layout and TransData decisions, so postpone fragile manual layout work until
the quantized graph is final.

### 10. Split, Parallelize, Or Cache

Attempt only when measured repeated computation or reusable state justifies additional roles. Include
launch, synchronization, device-link, and transfer overhead in the weighted result. Validate every role
handoff and final action. Do not copy PI05's split topology to unrelated policies.

### 11. Algorithmic Changes

Step count, schedule, tokenizer limits, action chunk semantics, or cache semantics are behavior changes,
not compiler optimizations. They always require explicit approval and separately identified reports.

Do not recommend statically unrolling all denoising steps into one OM as a normal optimization. It fixes
step count, makes schedule changes inconvenient, and can multiply graph/artifact size. Optimize the
single-step Expert first; for explicitly requested runtime work, prefer shared device buffers and a
device-resident state/Euler loop while keeping step count configurable.

## Completion

Select the fastest candidate that meets accepted accuracy and operational constraints. Publish only
that candidate, rerun final accuracy and loop-20 reports against final hashes, and provide `reproduce.sh`.
If no candidate improves weighted mean beyond noise, keep the conservative baseline and report that
optimization produced no accepted gain.

End with the hits-among-evaluated and catalog-coverage section from `experience-ledger.md`.
