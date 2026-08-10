# Ascend310P Optimization Patterns

Read this experience reference only during optional optimization for an exact `Ascend310P*` target,
after the conservative conversion baseline identifies a bottleneck. These patterns were measured on CANN
8.1 RC1/Ascend310P3 while porting SmolVLA and must be revalidated on the user's exact P1/P3 toolchain and
graph.

## Default Priorities

Use this order before custom operators, quantization, or algorithm changes:

1. export standard ONNX opset 17;
2. make the graph end-to-end FP16 except for accuracy-proven FP32 islands;
3. inspect ATC warnings and profile every role with `msprof`;
4. rewrite pathological rank-4 attention as rank-3 BMM by flattening batch and heads;
5. share ACL device buffers between serial OM roles;
6. optimize the invocation-weighted bottleneck, including repeatedly invoked Action Experts;
7. consider PFA, custom-domain operators, quantization, or algorithm changes only afterward.

## Opset 17 First

ONNX introduced standard `LayerNormalization` in opset 17. With the tested PyTorch exporter, opset 16
decomposed each LayerNorm into primitives such as:

```text
ReduceMean -> Sub -> Cast -> Pow -> ReduceMean -> Add -> Sqrt -> Div -> Cast -> Mul -> Add
```

On Ascend310P3/CANN 8.1 RC1, 50 decomposed vision LayerNorm sites placed expensive work on AI CPU.
Opset 17 emitted 50 standard `LayerNormalization` nodes, which ATC mapped to AI Core `LayerNorm`.

| Metric | Opset 16 | Opset 17 |
|---|---:|---:|
| ONNX nodes | 5213 | 4613 |
| `LayerNormalization` nodes | 0 | 50 |
| LayerNorm placement | AI CPU primitives | AI Core `LayerNorm` |
| LayerNorm-related time | about 514 ms | 2.720 ms |
| VLM mean | 992.047 ms | 479.944 ms |

Prefer opset 17, but count nodes, compile for the exact SoC, inspect ATC warnings, confirm placement with
`msprof`, and rerun final-action accuracy. Opset 17 does not improve a role with no eligible LayerNorm; a
measured Action Expert remained about 129-131 ms after an opset-only change.

## FP16 Is The Main 310P Eligibility Lever

Failure to use FP16 is a common Ascend310P performance problem. Start with FP16 inputs, weights,
projections, and MatMul/BMM. Do not retain broad FP32 islands merely because native Torch upcasts them.

Measured SmolVLA vision results:

- rank-3 FP32 attention: about 15.0 seconds;
- FP16 QK, FP32 Softmax, FP32 PV: about 7.5 seconds;
- FP32 QK, FP32 Softmax, FP16 PV: about 8.6 seconds;
- FP16 QK/PV with explicit FP32 Softmax: about 1.0 second;
- opset 17 then reduced VLM to about 0.48 seconds by fixing LayerNorm placement.

For the Action Expert, FP32 rank-4 QK dominated 98% of device time. Rank-3 alone became slower, but rank-3
plus FP16 QK reduced one invocation from about 129 ms to 4.2 ms. Its PV was already FP16 because
probabilities were cast to the value dtype.

Test shape and dtype changes together. An FP32 rank-3 candidate can regress even when rank-3 plus FP16 is
the winning combination. Retain FP32 only for accuracy-proven islands.

Compiler precision mode is a separate candidate axis from ONNX tensor dtype. First compile
`--precision_mode_v2=origin` to establish the conservative accuracy baseline. Once accepted, optionally
compile the same ONNX with ATC-default `--precision_mode_v2=fp16`; it may expose additional FP16 kernel
selection while preserving explicit graph-level FP32 islands. Accept it only when the same final-action
limits pass and loop-20 weighted latency improves. Do not pass the literal value `default`.

## FP32 Islands Can Poison Downstream Kernels

Treat an FP32 island as a connected region inside an otherwise FP16 graph where dtype promotion keeps
intermediate tensors in FP32 and propagates that dtype into downstream operators. The expensive operator
may not be the primitive that created the island. For example, a RoPE implementation can compute position
frequencies and trigonometric values in FP32 for valid numerical reasons, then accidentally perform:

```text
FP16 Q/K * FP32 cos/sin
-> FP32 rotated Q/K
-> FP32 attention QK BMM
```

On the measured SmolVLA/Ascend310P3 graph, the RoPE primitives themselves were cheap, but their FP32 outputs
made fifteen text QK operations use rank-4 FP32 `BatchMatMulV2` with ND layout and low parallelism. Each QK
took about 26-28 ms. The complete `BatchMatMulV2` category consumed 423.089 ms of a 479.944 ms VLM.

Replacing the exact half-rotation expression with `NPURotaryMul` preserved FP32 frequency/Sin/Cos
calculation, explicitly cast cos/sin to the Q/K dtype at the operator boundary, and produced FP16 rotated
Q/K. This made the downstream attention QK eligible for a high-performance FP16 kernel:

| Metric | Primitive RoPE | `NPURotaryMul` candidate |
|---|---:|---:|
| VLM mean | 479.944 ms | 66.761 ms |
| `BatchMatMulV2` total | 423.089 ms | 18.019 ms |
| Fused `RotaryMul` total | not applicable | 0.176 ms |

The 405.070 ms reduction in `BatchMatMulV2` explains 98.04% of the 413.184 ms VLM reduction. Do not report
this as the RotaryMul kernel itself saving 405 ms. Its direct execution is only about 0.176 ms; its main
effect is establishing a stable FP16 boundary that changes downstream kernel selection.

Use this diagnostic method for any normalization, position encoding, mask, activation, or glue rewrite:

1. inspect producer and downstream consumer dtype, rank, layout, and shape before and after the candidate;
2. compare the downstream high-cost operator's kernel, block dimension, and cumulative duration;
3. keep numerically sensitive calculation in FP32, but cast once at the intended consumer boundary;
4. use an exact fused operator when it gives ATC an unambiguous output dtype/layout contract;
5. measure the complete role, because the gain can be much larger than the fused operator's own time;
6. run single-variable candidates when exact per-rewrite attribution is required.

For SmolVLA, `NPURmsNorm` also replaced 31 primitive sites and executed in 0.530 ms total, but no
RMSNorm-only candidate was measured. The profile strongly attributes the large combined gain to the text QK
dtype/kernel change, while exact independent RMSNorm and RotaryMul contributions require separate ablations.

## Softmax And Accuracy Islands

End-to-end FP16 does not mean blindly forcing every operation to FP16. Keep finite masks and sensitive
reductions numerically valid. In the measured vision graph, changing an explicit FP32 Softmax request to
FP16 produced identical final-action metrics and no latency gain, so FP32 remained selected at no cost.

Another 310P text-attention graph produced invalid FP16 Softmax values. Inspect logits, mask Add,
probabilities, finiteness, and row sums when lowering precision. Final-action accuracy is the gate.

## Rank-3 Attention BMM

Ascend310P can select different kernels for rank-4 `[B,H,M,K] x [B,H,K,N]` and rank-3
`[B*H,M,K] x [B*H,K,N]`. Flattening batch and heads reduced an isolated FP32 vision PV from about
1633 ms to 271 ms.

Prove the Torch rewrite, inspect ONNX shape/dtype, confirm the graph hash and nodes changed, check that ATC
did not reconstruct rank 4, and measure the complete role. A source patch that does not change ONNX is not
optimization evidence. Patch the actual module execution path rather than assuming a framework registry
replacement is used.

## PFA Is Low Priority On 310P

Do not lead with PromptFlashAttention on Ascend310P. Warn the user before spending implementation time:

- P1/P3 deployments commonly cannot use PFA through a portable standard ONNX path;
- torch_npu API presence and a device kernel do not prove a usable parser path;
- CANN 8.1 RC1 exposes private `NPUPromptFlashAttention` parser entries rather than standard ONNX PFA;
- custom-domain use loses standard ORT validation;
- exact SoC kernel, parser/opset/domain, mask, layout, and accuracy still require proof.

Prefer standard ONNX rank-3 BMM plus FP16 QK/PV. Probe PFA only when the user explicitly accepts private
domain lock-in and exact P1/P3 deployment support is known.

## Serial OM Roles Must Share Device Buffers

For producer-consumer roles, use the PI05 Ascend pattern:

```text
one aclrtMalloc allocation
  -> producer output dataset slot
  -> consumer input dataset slot
```

Each model may own a separate `aclDataBuffer` wrapper, but both point to the same device pointer. The backend
owns the allocation; model datasets borrow it and must not free it. Validate dtype, shape, runtime index, and
byte size. Use Manifest `device_links` with matching internal semantics. Normal inference must not D2H the
producer output or H2D the consumer input; diagnostic capture is excluded from performance measurement.

## Do Not Unroll Denoising By Default

Do not normally merge all denoising steps into one exported OM. Static unrolling:

- fixes step count and schedule in the graph;
- makes schedule changes require re-export and recompilation;
- can duplicate Expert graph/weights and produce very large artifacts;
- increases compile time and memory pressure;
- mixes algorithm configuration with compiler optimization.

First optimize the reusable single-step Expert with opset 17, rank-3 attention, and FP16. Preserve a
configurable runtime schedule and share invariant buffers. If the user explicitly requests runtime work,
prefer device-resident recurrent state/velocity buffers and device Euler updates without changing the
single-step graph. Reducing denoising steps remains an algorithmic hard approval gate.

## No-Gain Results And Re-Profiling

No-gain and regression results lower a candidate's priority for a similar graph; they do not ban the
candidate in other policies, shapes, CANN releases, or exact SoCs. Retest when the mechanism could change.
Do not assume a plausible rewrite helps, but do not universalize one result:

- two-camera internal batching measured 479.238 ms versus 479.944 ms and provided no meaningful gain;
- query blocking alone did not improve vision PV;
- FP16 Softmax did not improve the selected graph;
- Expert opset 17 alone did not help;
- Expert rank-3 FP32 alone regressed;
- unchanged ONNX means the source patch did not reach export.

Multi-camera vision batching deserves explicit reconsideration. Its purpose is to increase AI Core
occupancy by running identical vision towers as one larger batch. It has produced meaningful gains in PI05,
even though the measured SmolVLA/Ascend310P3 graph gained only 0.15% after earlier attention optimization.
Ascend310P1 has more AI Cores than Ascend310P3, so a batch-1 camera branch may leave more capacity unused and
batch-2 may still improve utilization. Recommend a focused batch-1-versus-batch-N benchmark when:

- two or more cameras use the same vision weights and resolution;
- external camera ordering can be restored after an internal batch/split;
- profiling suggests low cube/core utilization or duplicated launch overhead;
- the target changes between P1 and P3;
- the policy family or vision token shape differs from the previous no-gain experiment.

Keep the external observation ABI and camera-major token semantics unchanged, and accept only measured
loop-20 weighted improvement with final-action accuracy.

Re-profile after every major gain. Calculate `sum(role_mean * invocations_per_action)`. The Expert initially
represented only about 2.3% of latency, but ten invocations became dominant after VLM optimization.
