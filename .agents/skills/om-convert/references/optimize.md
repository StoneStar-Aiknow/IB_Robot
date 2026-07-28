# Optional Ascend OM Optimization

This workflow is optional. Do not enter it merely because conversion succeeded. Enter only when the
upfront intent selected optimization or the user confirms after the conservative baseline.

## Optimization Intake

Reuse the initial intake and ask only for unresolved optimization choices together:

- target performance goal, if any;
- whether approximate math is allowed;
- whether static input ranges/shapes may change;
- whether graph splitting, cache, or parallel roles are allowed;
- whether quantization is allowed;
- accuracy threshold changes, if requested.

Default all risky choices to disabled. Quantization is not a mandatory rung and is never enabled by
autonomous mode without explicit user selection.

## Candidate Discipline

For each candidate:

```text
one principal hypothesis
-> source/export change
-> required accuracy gates
-> ATC + exact ABI
-> ais_bench --loop 50 for every role
-> weighted total
-> keep or rollback
```

Append to `reports/experiments.jsonl`: candidate ID, hypothesis, evidence, source diff/commit, commands,
artifact hashes, accuracy, latency, verdict, and rollback.

Do not create a permanent Manifest deployment for every candidate. Use `current/` for working artifacts,
retain reports and reproduction commands, delete rejected large ONNX/OM files when no longer needed,
and publish only the selected final generation.

## Optimization Ladder

The order is evidence-driven. Skip a rung when profiling shows it is irrelevant.

### 0. Freeze The Baseline

Require accepted Torch-vs-ONNX, accepted Torch-vs-OM, all OM loop-50 logs, weighted totals, input and
artifact hashes, toolchain versions, and invocation counts.

### 1. Remove Data-Path And Structural Waste

Inspect repeated dataset/buffer creation, avoidable H2D/D2H, host round-trips between roles, unused
outputs copied to host, missing buffer reuse, repeated invariant prefix/embedding/cache work, and
duplicated image encoding. Prefer reusable generic ACL changes over model-specific copies.

### 2. Profile And Rank ROI

Use `msprof` only as a diagnostic tool; final performance remains ais_bench loop 50. Check AICPU
fallback, host/runtime overhead, OP-type cumulative duration, normalized source-module names, and
`mac_ratio`, `mte1/mte2`, and vector ratios.

Rank by measured share, removability, expected benefit, implementation cost, accuracy risk, and risk of
being invalidated by later quantization/layout changes. Do not optimize a cheap non-bottleneck merely
because a fused operator exists.

### 3. Eliminate Exact Glue

Try semantics-preserving removal of unused outputs, Cast/Slice chains, materialized Expand/Tile,
redundant shape graphs, and avoidable layout conversion. For MQA, `num_kv_heads == 1` may allow MatMul
broadcast instead of `repeat_kv`; do not apply this to GQA where the head dimension cannot broadcast.

### 4. Use Exact NPU Fused Operators

A candidate requires all three:

1. target SoC has a kernel;
2. CANN ONNX plugin has a parser in a compatible opset/domain;
3. model subgraph has identical mathematics, layout, optional inputs, and output semantics.

Check CANN ops-info and `libops_all_onnx_plugin.so`; do not infer support from torch_npu API presence.
Reusable PI05 examples are exact RoPE/RMSNorm and, where weight order and formula match, GeGLU.

Beware hidden costs: ND-only fused operators can add TransData, alignment can add Pad/Slice, bool ops
can fall to AICPU, and an available parser may have no target kernel. Measure the complete role.

### 5. Remove Dtype Islands

Target measured fp16/fp32 Cast islands around attention, softmax, norm, reductions, masks, and MatMul.
Keep finite mask sentinels valid in the chosen dtype. A local FP16 rewrite is approximate unless proven
exact and therefore must pass the applicable accuracy gates.

### 6. Scan Static Shape/Tiling Choices

Only scan dimensions whose valid business range permits change, such as padding/sequence length, image
resolution, or chunk shape. Benchmark every point with loop 50. ATC tiling has discontinuous slow bands;
never predict winners from divisibility, powers of two, or prime factors. Changing valid input range or
truncating data is a hard approval gate.

### 7. Approximate Operators

Consider FastGELU or similar approximations only after user approval and hotspot evidence. Different
GELU approximations are not interchangeable merely because cosine is high at one layer. Validate final
action against the original Torch targets.

### 8. Optional Quantization

Run only if the user explicitly selected quantization. It is useful when GEMM/BatchMatMul and weight
movement dominate after glue removal.

Requirements:

- representative real LeRobot dataset observations;
- recorded calibration provenance;
- no random calibration fallback;
- start with selected linear layers;
- exclude sensitive norm, attention BMM, final action head, and empirically sensitive layers by default;
- validate each quantized role and the complete action;
- rerun loop 50 and weighted totals.

Quantization may rebuild layout and TransData decisions, so postpone fragile manual layout work until
the quantized graph is final.

### 9. Split, Parallelize, Or Cache

Attempt only when measured repeated computation or reusable state justifies additional roles. Include
launch, synchronization, device-link, and transfer overhead in the weighted result. Validate every role
handoff and final action. Do not copy PI05's split topology to unrelated policies.

### 10. Algorithmic Changes

Step count, schedule, tokenizer limits, action chunk semantics, or cache semantics are behavior changes,
not compiler optimizations. They always require explicit approval and separately identified reports.

## Completion

Select the fastest candidate that meets accepted accuracy and operational constraints. Publish only
that candidate, rerun final accuracy and loop-50 reports against final hashes, and provide `reproduce.sh`.
If no candidate improves weighted mean beyond noise, keep the conservative baseline and report that
optimization produced no accepted gain.
