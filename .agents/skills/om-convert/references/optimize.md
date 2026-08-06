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

## Requests For More Optimization Options

Treat open-ended follow-ups such as "what else can we optimize?", "any other approach?", "还有什么优化方案",
"还有没有其他方案", and "还有没有更多方案" as a request to audit unused experience, not as permission to
invent an unbounded new ladder.

Before answering or starting another candidate:

1. reopen `reports/experience-ledger.json` and apply the experience gap review from
   `experience-ledger.md`;
2. compare every `not_evaluated` experience with the latest `msprof`, ATC warnings, graph structure,
   exact SoC/toolchain, accuracy results, and weighted bottleneck;
3. revisit `attempted_no_gain` and `not_applicable` only when recorded assumptions have materially
   changed, and state the changed evidence;
4. list remaining applicable experiences in ROI order with expected mechanism, evidence, accuracy risk,
   approval requirement, and the experiment that would accept or reject each one;
5. explicitly state when the catalog has no remaining evidence-backed candidate, then separate any new
   speculative idea from accumulated experience.

Do not merely repeat successful or rejected candidates. Do not silently expand scope into quantization,
approximate math, shape-range changes, or algorithm changes; their existing approval gates still apply.
Update the ledger after each newly evaluated candidate so a later follow-up uses current evidence.

## Optimization Ladder

The order is evidence-driven. Skip a rung when profiling shows it is irrelevant.

An earlier `attempted_no_gain` or regression lowers priority only for a materially similar graph and target.
It is not a permanent skip rule. Re-evaluate when policy family, shapes, compiler, dtype, or exact SoC changes.

For Ascend310P, read `ascend310p-optimization.md` before proposing the ladder. Default to opset 17 and
end-to-end FP16 with only accuracy-proven FP32 islands. Establish an `origin` ATC baseline first, then
optionally compile the ATC-default `fp16` candidate against the same targets; do not pass the literal
value `default`. Give PFA low priority and prefer standard-ONNX rank-3 BMM rewrites.

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

### 2B. ATC Precision Mode Philosophy

Use two deliberate stages rather than treating compiler precision modes as interchangeable:

```text
origin
-> conservative compiler precision baseline
-> establish accepted Torch-vs-OM accuracy

fp16 (ATC default; omit the option or pass --precision_mode_v2=fp16)
-> performance candidate
-> compare against the same authoritative Torch targets
```

Use `origin` first when the deployment contract or accuracy behavior is not established. Once `origin`
passes, try ATC-default `fp16` when precision selection may expose more high-performance FP16 kernels or
remove unnecessary precision barriers. This is especially useful when the ONNX graph already preserves
explicit FP32 islands for Softmax, masks, reductions, or other sensitive stages: ATC can optimize other
eligible operators without changing the graph.

The `fp16` candidate must pass the unchanged accuracy and deployment gates:

1. same observations, targets, raw targets, task, seed, and persisted noise;
2. same Torch-vs-OM mean-L1 and cosine limits as `origin`;
3. exact ABI and same-SoC deployment validation;
4. complete-role `ais_bench --loop 20` and invocation-weighted latency comparison.

Do not pass the literal `--precision_mode_v2=default`. In the tested ATC interface, `fp16` is the default
value. Prefer the explicit, reproducible spelling:

```bash
--precision_mode_v2=fp16
```

If the option is omitted to exercise the compiler default, record that omission and the exact ATC help
output. Never use the legacy `--precision_mode` option. If `fp16` fails accuracy or produces non-finite
outputs, keep `origin`, record the failure, and do not relax thresholds or regenerate targets.

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

Trace dtype propagation beyond the local primitives. A cheap FP32 producer can force an expensive downstream
MatMul/BMM onto a low-priority FP32 kernel. Compare producer output dtype and consumer dtype/rank/layout in
both ONNX and `msprof`; do not estimate value from the producer's own duration alone.

A proven Ascend310P pattern is exact RoPE replacement with `NPURotaryMul`: keep frequency and Sin/Cos
calculation in FP32, cast cos/sin once to the Q/K dtype, and require FP16 rotated Q/K outputs. In one SmolVLA
graph this changed fifteen text QK BMMs from about 26-28 ms each to a high-performance FP16 path, reducing
the `BatchMatMulV2` category by about 405 ms while `RotaryMul` itself consumed only 0.176 ms. Treat this as a
downstream kernel-eligibility gain, not as direct fused-operator latency.

When reporting attribution, separate:

- direct operator savings;
- downstream dtype/layout/kernel-selection savings;
- other graph/compiler changes;
- measured ablation evidence versus mechanism-based inference.

Do not claim precise independent contribution from a combined candidate without single-variable controls.

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
