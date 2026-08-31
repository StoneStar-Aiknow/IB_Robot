# Compiler Outputs And Export Requirements

## When to Read

- 执行 Required Workflow 步骤 1（Houmo 编译器导出）时
- 需要确认 PI0.5 或 SmolVLA 的 role 集合和 export 校验项时
- 需要了解 v1.3.0 仓库管理导出脚本的具体校验逻辑时

## PI0.5

Required compiler outputs:

| Role | Artifact pair |
|------|---------------|
| vision | vision `.hmm` + TCIM `model.json` |
| prefill | prefill `.hmm` + TCIM `model.json` |
| action_in_proj | projection `.hmm` + TCIM `model.json` |
| time_mlp | time MLP `.hmm` + TCIM `model.json` |
| decode | expert decode `.hmm` + TCIM `model.json` |
| action_out_proj | projection `.hmm` + TCIM `model.json` |
| embedding | `embedding.pt` |

The packager requires the exact PI0.5 role set. Prefill cache outputs and decode cache inputs must have identical names, dtypes, and shapes. Device links are output-sourced: prefill output -> decode input.

### PI0.5 v1.3.0 Export Requirements

The repository-managed conversion is:

```bash
source .shrc_local
MODEL_BUNDLE_ROOT=models/pi05 ./scripts/convert_hmm.sh pi05
```

The native exporter does not require the untracked Houmo vendor example or `xh_model_zoo`.
`MODEL_BUNDLE_ROOT` and `PI05_HMM_OUTPUT` must be workspace-relative. The output defaults to
`models/pi05_hmm_standard` and must not already exist, preventing stale HMONNX or TCIM products
from contaminating a rebuild. The workflow uses `transformers==5.3.0` and performs a strict
checkpoint-loading preflight.

Required checks after export:

- Prefill exposes `prefix_embs`, `attention_mask`, and `position_ids`, followed by 18 key-cache and
  18 value-cache outputs.
- Decode exposes action embeddings, attention mask, position IDs, condition, then the same 36 cache
  inputs with matching names, dtypes, and shapes.
- `embedding.pt["weight"]` exactly equals
  `model.safetensors["model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"]`.
- `provenance.json` records the checkpoint SHA-256, image ID, LeRobot HEAD, Transformers, xhquant,
  and TCIM versions.
- Run the strict packager before replacing an existing deployment; it validates the projection
  chain, action shapes, prefix capacity, and all cache links.

## SmolVLA

Required compiler outputs:

| Role | Artifact pair |
|------|---------------|
| vision | vision `.hmm` + TCIM `model.json` |
| prefill | prefill `.hmm` + TCIM `model.json` |
| action | action `.hmm` + TCIM `model.json` |
| embedding | `token_embedding.pt` |
| state_projection | `state_projection.pt` |

The current runtime graph is `vision -> embedding -> prefill -> action`; it does not execute a standalone decode role. Device links are output-sourced: prefill output -> action input.

### SmolVLA v1.3.0 Export Requirements

The repository-managed standard conversion is:

```bash
source .shrc_local
MODEL_BUNDLE_ROOT=models/smolvla ./scripts/convert_hmm.sh smolvla
```

The repository keeps one public dispatcher, `scripts/convert_hmm.sh <policy>`. Policy-specific
orchestration belongs beside its exporter under `model_utils/<policy>_export/`; do not add a new
root-level `scripts/convert_<policy>_hmm.sh` for each policy.

The SmolVLA workflow verifies the patched LeRobot v0.6.0 source, prefers `transformers==5.3.0` with `4.57.1` as a
loading fallback, exports all modules, quantizes them, compiles TCIM HMM files, writes provenance,
and packages a strict deployment under `models/smolvla_hmm_standard` by default.
`MODEL_BUNDLE_ROOT` is required, has no default, and must identify an existing workspace-relative
bundle directory. Keep this input parameter policy-neutral so every HMM converter uses the same
bundle-root contract.

When using `houmo-examples-xh2` v1.3.0, verify that the LLM KV-cache exporter loads the fine-tuned
`SmolVLAPolicy` before attempting to construct a standalone `SmolVLMWithExpertModel`. The upstream
script may load the base VLM successfully first and therefore never consume `model.safetensors` from
the policy bundle. That produces a valid-looking prefill HMM and token embedding from the base model,
not the requested policy checkpoint.

Required checks after export:

- `work_dirs/<llm>_kvcache/meta_info.json` must contain `"load_mode": "policy"`.
- `token_embedding.pt["weight"]` must equal
  `model.safetensors["model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"]` after the
  exporter's BF16-to-FP16 conversion.
- Extract `state_projection.pt` from the same checkpoint using `model.state_proj.weight` and
  `model.state_proj.bias`; do not reuse the projection from another policy bundle.
- Re-export action after changing the prefill export, then recompile both HMONNX files so the cache
  ABI and checkpoint provenance remain paired.

The verified repository workflow uses `transformers==5.3.0`, `tokenizers==0.22.2`, and
`diffusers==0.35.2`. The image's `hmquant-xh2` metadata still declares `transformers==4.51.0`, so
the script performs a real SmolVLA policy-loading preflight instead of trusting dependency metadata.
It also removes the image's incompatible `torchao==0.17.0` before importing LeRobot. The exporter
resolves `vlm_model_name` from the bundle's local `HuggingFaceTB/` directory without changing
`config.json`.

For the fixed 512x512 full-patch vision input, the managed exporter replaces SmolVLM's boolean
position-ID indexing with equivalent static position IDs and rejects any exported vision graph that
still contains `NonZero`. Vision and prefill export with CUDA float16; action exports with CUDA
float32 to avoid denoise dtype mixing. Calibration tensors are captured from the actual export
inputs, including action KV cache tensors produced by the same prefill forward pass.
