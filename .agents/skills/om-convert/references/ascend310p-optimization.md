# Ascend310P Optimization Patterns

Read this reference when converting or optimizing for an exact `Ascend310P*` target. These patterns were
measured on CANN 8.1 RC1/Ascend310P3 while porting SmolVLA, and must be revalidated on the user's exact
P1/P3 toolchain and graph.

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

Do not assume a plausible rewrite helps:

- two-camera internal batching measured 479.238 ms versus 479.944 ms and provided no meaningful gain;
- query blocking alone did not improve vision PV;
- FP16 Softmax did not improve the selected graph;
- Expert opset 17 alone did not help;
- Expert rank-3 FP32 alone regressed;
- unchanged ONNX means the source patch did not reach export.

Re-profile after every major gain. Calculate `sum(role_mean * invocations_per_action)`. The Expert initially
represented only about 2.3% of latency, but ten invocations became dominant after VLM optimization.
