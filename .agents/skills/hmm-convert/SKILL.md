---
name: hmm-convert
description: "Package PI0.5 or SmolVLA models as Houmo HMM (.hmm) deployments for XH2 NPU (LQ50 / M50). Use when users mention 'convert to hmm', 'houmo', 'xh2', '后摩', 'tcim', 'xhquant', 'pi05 hmm', 'smolvla hmm', 'LQ50', 'M50', 'HMM 模型转换', 'w8a8', 'sefp', 'hmonnx', or package-hmm-deployment. Explicitly reject ACT HMM requests and route ACT to torch, ascend, hisilicon, or rknn."
---

# HMM Model Conversion Skill

Package Houmo compiler outputs for PI0.5 and SmolVLA into strict IB-Robot HMM deployments for XH2 NPU (LQ50 / M50).

## Supported Matrix

| Policy | HMM support | Graph |
|--------|-------------|-------|
| PI0.5 | Supported | vision + embedding + prefill + action projections + time MLP + decode |
| SmolVLA | Supported | vision + embedding/state projection + prefill + action |
| ACT | Unsupported | Reject; use `torch`, `ascend`, `hisilicon`, or `rknn` |

Never restore or recommend the removed ACT single-module HMM path, `config.hmm.json`, `HMM_MODEL_PATH`, directory scanning, `HMMRuntimeSession`, or `device:=hmm`.

## Required Workflow

1. Use Houmo's current `houmo-examples-xh2` workflow to export, quantize, and compile the supported policy modules.
2. Preserve the compiler-emitted TCIM `model.json` next to each `.hmm`; it is the source of runtime names, indices, dtypes, and shapes.
3. Keep the original LeRobot bundle metadata read-only: `config.json`, processor JSON/state, tokenizer assets, and policy metadata.
4. Create a path-only packaging spec for `package-hmm-deployment`.
5. Run the packager to write or update the bundle's only deployment manifest: `inference_manifest.json`.
6. Configure a named inference pipeline that selects the generated deployment.
7. Validate on the target or approved TCIM mocks.

The packager owns artifact generations, bindings, execution order, device links, UUID/revision updates, lightweight structural identities, and strict-loader verification. Do not hand-edit generated manifest identities or tensor bindings.

## Environment

The Houmo compiler toolchain may conflict with the IB-Robot Python environment. Use a dedicated venv or Houmo's current 24.04 toolchain image. Do not install `xhquant` or `tcim` into the main venv.

```bash
python3 -m venv .venv-hmm
source .venv-hmm/bin/activate
pip install xhquant tcim onnx onnxsim torch
```

Use `source .shrc_local` before running IB-Robot packaging, manifest validation, ROS, or project commands.

## Compiler Outputs

### PI0.5

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

#### PI0.5 v1.3.0 Export Requirements

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

### SmolVLA

Required compiler outputs:

| Role | Artifact pair |
|------|---------------|
| vision | vision `.hmm` + TCIM `model.json` |
| prefill | prefill `.hmm` + TCIM `model.json` |
| action | action `.hmm` + TCIM `model.json` |
| embedding | `token_embedding.pt` |
| state_projection | `state_projection.pt` |

The current runtime graph is `vision -> embedding -> prefill -> action`; it does not execute a standalone decode role. Device links are output-sourced: prefill output -> action input.

#### SmolVLA v1.3.0 Export Requirements

The repository-managed standard conversion is:

```bash
source .shrc_local
MODEL_BUNDLE_ROOT=models/smolvla ./scripts/convert_hmm.sh smolvla
```

The repository keeps one public dispatcher, `scripts/convert_hmm.sh <policy>`. Policy-specific
orchestration belongs beside its exporter under `model_utils/<policy>_export/`; do not add a new
root-level `scripts/convert_<policy>_hmm.sh` for each policy.

The SmolVLA workflow verifies the patched LeRobot v0.5.1 source, prefers `transformers==5.3.0` with `4.57.1` as a
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

## Packaging Spec

Use one JSON spec containing only source paths. Relative paths resolve from the spec directory.

PI0.5 example:

```json
{
  "vision": {
    "artifact": "/compiler/pi05/vision.hmm",
    "abi": "/compiler/pi05/vision/model.json"
  },
  "embedding": "/compiler/pi05/embedding.pt",
  "roles": {
    "prefill": {
      "artifact": "/compiler/pi05/prefill.hmm",
      "abi": "/compiler/pi05/prefill/model.json"
    },
    "action_in_proj": {
      "artifact": "/compiler/pi05/action_in_proj.hmm",
      "abi": "/compiler/pi05/action_in_proj/model.json"
    },
    "time_mlp": {
      "artifact": "/compiler/pi05/time_mlp.hmm",
      "abi": "/compiler/pi05/time_mlp/model.json"
    },
    "decode": {
      "artifact": "/compiler/pi05/decode.hmm",
      "abi": "/compiler/pi05/decode/model.json"
    },
    "action_out_proj": {
      "artifact": "/compiler/pi05/action_out_proj.hmm",
      "abi": "/compiler/pi05/action_out_proj/model.json"
    }
  },
  "vision_layout": "NCHW"
}
```

SmolVLA example:

```json
{
  "vision": {
    "artifact": "/compiler/smolvla/vision.hmm",
    "abi": "/compiler/smolvla/vision/model.json"
  },
  "embedding": "/compiler/smolvla/token_embedding.pt",
  "state_projection": "/compiler/smolvla/state_projection.pt",
  "roles": {
    "prefill": {
      "artifact": "/compiler/smolvla/prefill.hmm",
      "abi": "/compiler/smolvla/prefill/model.json"
    },
    "action": {
      "artifact": "/compiler/smolvla/action.hmm",
      "abi": "/compiler/smolvla/action/model.json"
    }
  },
  "vision_layout": "NCHW"
}
```

## Run The Packager

```bash
source .shrc_local
package-hmm-deployment \
    --bundle-root models/<policy_bundle> \
    --deployment hmm_lq50 \
    --target-soc lq50 \
    --target-runtime tcim-lite \
    --spec /path/to/hmm-package-spec.json
```

Source form:

```bash
source .shrc_local
python3 -m model_utils.hmm_export \
    --bundle-root models/<policy_bundle> \
    --deployment hmm_lq50 \
    --target-soc lq50 \
    --target-runtime tcim-lite \
    --spec /path/to/hmm-package-spec.json
```

Success means `<bundle-root>/inference_manifest.json` was written and accepted by the production strict loader.

## Named Pipeline Configuration

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      pipelines:
        policy:
          model_path: models/<policy_bundle>
          deployment: hmm_lq50
          execution_mode: monolithic
          request_timeout: 10.0
          default_task: "pick up the object"
    executor:
      type: topic
      mode: model_inference
      inference_pipeline: policy
```

Do not configure backend-valued `device`, a global model table, or a per-backend manifest filename.

## Validation

Run focused tests after changing HMM packaging or docs:

```bash
source .shrc_local
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH="$PWD/src/inference_manifest:$PWD/src/inference_service:$PWD/src/model_utils:$PYTHONPATH" \
pytest -q \
    src/model_utils/test/test_hmm_export.py \
    src/inference_service/tests/test_hmm_backend.py \
    src/inference_service/tests/test_backend_contract.py
```

## Troubleshooting

### ACT HMM request

Reject the request as unsupported. Offer ACT deployments on `torch`, `ascend`, `hisilicon`, or `rknn`; do not provide a compatibility workflow.

### TCIM ABI mismatch

Use the `model.json` emitted for the exact `.hmm` file. Do not infer tensor order from Python dictionaries or copy ABI metadata from another build.

### Manifest hash mismatch

Rerun `package-hmm-deployment`. Do not edit artifact identity, revision, or bundle digest manually.

### External tokenizer dependency

Vendor tokenizer/processor assets into the bundle and update the LeRobot metadata to reference the local directory before packaging.

### Board runtime initialization

Read `oh-constraints` first. Source the RoboFrame environment and `scripts/setup/houmo_hmm_env.sh`; native Houmo 1.3 uses `TCIM_BACKEND=xh2` and `HOUMO_TARGET=xh2`. `Xh2HalBackend` is only for the legacy runtime.

## References

- `docs/Houmo_HMM_Conversion.md`
- `src/model_utils/model_utils/hmm_export.py`
- `src/model_utils/model_utils/pi05_export/`
- `src/model_utils/model_utils/inference_manifest_export.py`
- `src/inference_service/inference_service/backends/hmm/backend.py`
- `src/model_utils/test/test_hmm_export.py`
- `src/inference_service/tests/test_hmm_backend.py`

## When To Use This Skill

Use for:

- PI0.5 or SmolVLA Houmo HMM conversion and packaging
- `xhquant`, `tcim`, `hmonnx`, LQ50, M50, or XH2 compiler troubleshooting
- `package-hmm-deployment` specs and unified manifest generation
- HMM execution roles, bindings, embedding artifacts, state projection, or device links

Do not use for:

- ACT HMM conversion; it is unsupported
- RKNN conversion; use `rknn-convert`
- Ascend OM conversion; use the Ascend exporter workflow
- General OpenHarmony board operations; read `oh-constraints` and use the relevant board skill
