# Precision Drift Troubleshooting

Use this reference after any Torch-vs-ONNX or Torch-vs-OM failure, and before performance optimization of a
new policy. The rules below were proven while porting SmolVLA to Ascend310P3 and generalize to other VLA
policies.

## Required Comparison Ladder

Do not jump directly from native Torch to OM. Establish these boundaries in order with the same observation,
task, stochastic inputs, schedule, and postprocessing:

1. native Torch policy vs a manually decomposed native execution;
2. native Torch vs split Torch wrapper at the native dtype;
3. split native-dtype Torch vs split deployment-dtype Torch;
4. split deployment-dtype Torch vs portable ONNX;
5. native Torch target vs portable ONNX;
6. native Torch target vs OM;
7. if needed, VLM handoff, each velocity step, and each integrated state.

A pure wrapper/refactor must be bit-identical at the same dtype. If it is not, stop before ONNX export.

## Protect The Reference Process

PyTorch operations such as `module.to(dtype=...)`, `half()`, and monkey patches mutate objects or global
classes. Never derive native and converted references from a shared mutable model instance. Load independent
instances, or run each candidate in a clean process. Confirm the native reference is unchanged after every
candidate.

Test exporter patches independently against native Torch before combining them. A patch described as
"export-only" can still change eager semantics.

## Image Preprocessing Contract

Treat image preprocessing as model semantics:

- record layout, range, resize algorithm, `align_corners`, clamp range, padding side, and padding value;
- apply normalization in the same order as native Torch;
- share one helper between exporter, runtime, calibration, and diagnostics;
- prove the host helper against the native Torch helper with exact or agreed numerical equality.

Prefer a static VLM image ABI at the policy resolution and perform resize/padding on the host. Do not bake
Resize into ONNX/OM unless VLM KV and final action are revalidated. Small ATC Resize differences can be
amplified by attention, KV caching, and iterative action integration.

For `[0, 1]` inputs normalized later with `x * 2 - 1`, padding must be `0` before normalization so that it
becomes `-1`. Padding with `-1` before normalization incorrectly produces `-3`.

Do not create a resized validation batch by saving float images through a serializer that quantizes to uint8.
That changes more than the resize boundary. Compare resize-inside vs resize-outside in memory using the same
preprocessed tensors and noise.

## Static Vision Position IDs

When replacing boolean indexed assignment or dynamic position-ID construction, preserve operation order and
dtype exactly. The following are not necessarily equivalent in BF16/FP16:

```text
native: arange * reciprocal -> clamp -> cast to pixel dtype -> bucketize
wrong:  arange / count * (1 - epsilon) -> bucketize in FP32
```

Different rounding can select different position embedding indices. Validate every static vision patch in
eager Torch before export. In the proven SmolVLA case, the incorrect patch reduced action cosine to about
0.984 by itself; preserving native arithmetic made it bit-identical.

## Attention And Softmax

Inspect logits, masked logits, Softmax probabilities, value products, and output projections separately.
Softmax output must remain finite, in `[0, 1]`, and sum to one over the attention axis.

On Ascend310P3/CANN 8.1 RC1, a proven FP16 `SoftmaxV2` failure produced values up to 65504 for masked text
attention. Generic high-precision compiler flags did not fix it. The successful PI05-style contract was:

- build the additive prefix mask on the host in FP32;
- keep QK, mask Add, and Softmax in FP32;
- cast probabilities only before the value product;
- use eager attention on 310P when the fused attention path is unsupported.

Do not mechanically enable FlashAttention because another platform supports it. Prove parser, target kernel,
mask semantics, layout, and numerical equivalence on the exact SoC.

## Time Embedding And RoPE

Do not silently compute sinusoidal time embeddings in FP16. Use at least FP32 trigonometric computation and
cast only at the consuming layer. Compare every timestep because early-step errors accumulate.

In-place slice assignment in RoPE may export as ScatterND. An exact concat formulation can remove ScatterND,
but it must pass native-Torch and ORT equivalence. Removing ScatterND is not proof that it caused the observed
error.

## Task, Tokens, And Masks

Language policies must use the exact task stored in the versioned observation package unless the experiment
explicitly defines an override. Never let a comparison script silently replace a batch task with a CLI task.

Before blaming OM, compare:

- task string;
- token IDs;
- language attention mask;
- derived prefix mask;
- tokenizer asset path and revision.

A one-token or one-mask-bit difference invalidates the comparison.

## Noise And Other Stochastic Inputs

Persist a canonical deterministic sample plus metadata, not an ambiguous `.npy` alone. Record:

- generation algorithm and seed;
- shape;
- canonical storage dtype;
- Torch consumer dtype;
- ONNX/OM binding dtype;
- cast policy.

When consumers differ, cast the canonical values independently once for each backend. Do not round through
one backend dtype before converting to another. Inspect the actual consuming layer or ABI; dominant model
weights may be BF16 while the noise projection consumes FP32.

Run a noise-effectiveness test: identical noise must reproduce identical actions, and a materially different
noise must change the action.

## Eval Mode

Audit `eval()` when dropout or training branches are plausible, but prove it rather than assume it. Compare
direct native inference with the production Torch backend in one process, inspect `policy.training`, and compare
processed inputs. If actions and inputs are bit-identical, eval mode is excluded.

## Opset

Validate opset as an empirical toolchain variable. Portable ONNX can remain equivalent while ATC lowering
changes substantially. Record ONNX and OM results separately.

In the proven CANN 8.1 RC1/Ascend310P3 experiment:

- opset 14 and 16 produced equivalent OM accuracy;
- opset 18 passed ORT but failed badly after ATC;
- opset 17 was faster than <=16 in prior project experience, but it was not sufficient to ensure good kernels
  or correct exporter semantics.

Never infer OM accuracy from ORT success for a different opset.

## Invalid Evidence To Reject

Do not accept these as precision proof:

- `onnx.checker` without ORT metrics;
- ATC success without runtime output comparison;
- a mock backend;
- targets generated with unpersisted or ineffective noise;
- resized images serialized through an unaccounted quantization path;
- comparisons using a model instance mutated by another candidate;
- aggregate action metrics without checking split handoffs when localization is needed.

## Candidate Storage Discipline

Keep only one active ONNX/OM candidate. Preserve metrics, compiler commands, ABI summaries, and decisions, then
delete rejected ONNX, OM, ABI, and temporary bundles. Recreate candidates from recorded commands when needed.

## Proven Final Acceptance Example

The corrected SmolVLA opset-16 host-resize deployment used native BF16 Torch as the final reference and mixed
FP16/FP32 OM as the candidate. Across 20 real samples:

| Space | Mean L1 | Mean cosine | Minimum cosine |
|---|---:|---:|---:|
| Normalized action | 0.002975 | 0.999989 | 0.999944 |
| Final action | 0.094975 | 0.999996 | 0.999979 |

All samples passed the requested cosine threshold of 0.9999.
