---
name: hmm-convert
description: "Convert ACT / PI05 models to Houmo HMM (.hmm) format for XH2 NPU (LQ50 / M50) deployment. Use when user needs to 'convert to hmm', 'houmo', 'xh2', '后摩', 'tcim', 'xhquant', 'pi05 hmm', 'act hmm', 'LQ50', 'M50', 'HMM 模型转换', 'w8a8', 'sefp', 'hmonnx'. Triggers for Houmo NPU deployment, XH2 compilation, PI05 multi-module HMM pipeline."
---

# HMM Model Conversion Skill

Convert ACT / PI05 policy models to Houmo HMM (`.hmm`) format for XH2 NPU (LQ50 / M50) deployment.

## Scope

This skill covers:

- **ACT → HMM** (single-module, in-repo script `export_onnx_hmm.py`)
- **PI05 → HMM** (6-module split, external Houmo SDK `houmo-examples/pi05/demo.py`)

Do **not** use this skill for:

- RKNN conversion → use `rknn-convert`
- Ascend OM conversion → use `pi05_export/` package
- ONNX export for non-HMM targets → use `export_onnx_rknn.py` or `export_onnx_atc.py`

## Critical Workflow Split

| Policy | Modules | Conversion Tool | In-repo Script |
|--------|---------|-----------------|----------------|
| ACT | 1 (policy) | `export_onnx_hmm.py` | Yes |
| PI05 | 6 + embedding | `houmo-examples/pi05/demo.py` (external) | No (runtime + manifest contract only) |

### Architecture Difference

- **ACT HMM**: single `.hmm` module, loaded by `HMMRuntimeSession`, mirrors RKNN's single-model interface
- **PI05 HMM**: 6 `.hmm` modules + 1 `embedding.pt`, loaded by `PI05HMMRuntimeSession` → `PI05HMMModel`, which orchestrates the denoise loop on host CPU with KV-cache handoff by device-pointer sharing between prefill and decode modules

## Environment

### Houmo Toolchain (required for both ACT and PI05)

| Component | Source | Version |
|-----------|--------|---------|
| `xhquant` | `developer.houmoai.com/resources_v2` | ≥ xh2a_1.3.0 |
| `tcim` | bundled with `houmo-examples-xh2` | ≥ xh2a_1.3.0 |
| `houmo-examples-xh2` | `developer.houmoai.com/resources_v2` | v1.3.0 |
| Driver | `houmo-drv-xh2-1.3.0-1.aarch64.rpm` | V1.3.0 |
| Firmware | `M50_M2_fw-xh2_v1.3.0.tar.gz` | V1.3.0 |

Driver install guide: `docs/houmo_lq50_driver_install_oee.md`

### Python Environment

The Houmo toolchain (`xhquant` / `tcim`) has its own Python dependency tree that may conflict with LeRobot. Use a **dedicated venv** for HMM conversion:

```bash
cd <project_root>
python3 -m venv .venv-hmm
source .venv-hmm/bin/activate
pip install xhquant tcim onnx onnxsim torch  # torch version per xhquant requirements
```

**MUST NOT** install `xhquant` / `tcim` into the main venv.

## ACT → HMM Conversion

### Step 1: Export ONNX (main venv)

```bash
cd <project_root>
source .shrc_local
python3 src/model_utils/model_utils/export_onnx_hmm.py \
    --policy_path models/<your_model>/pretrained_model
```

Produces `act_ros2_hmm.onnx` in the policy directory.

If the ONNX already exists, pass `--onnx <path>` to skip export and only strip + simplify.

### Step 2: PTQ Quantize + Compile (.venv-hmm)

```bash
cd <project_root>
source .venv-hmm/bin/activate
python3 src/model_utils/model_utils/export_onnx_hmm.py \
    --onnx models/<your_model>/pretrained_model/act_ros2_hmm.onnx \
    --convert_hmm \
    --hmm_target xh2 \
    --hmm_quant_type w8a8h1_sefp \
    --hmm_ncore 2 \
    --hmm_opt_level O2
```

This runs the two-stage Houmo pipeline:

1. **PTQ** (`xhquant.api.convert_onnx_to_hmonnx`): quantizes the ONNX with `QuantScheme(DeviceType.XH2a, "w8a8h1_sefp")` using random calibration tensors → `<name>.hmonnx.onnx`
2. **Compile** (`tcim.build_from_hmonnx`): compiles to `.hmm` with `target="xh2", ncore=2, opt_level="O2"` → `model.hmm`

### Step 3: Verify Manifest

The script auto-generates `config.hmm.json`:

```json
{
  "schema_version": 1,
  "policy_type": "act",
  "backend": "hmm",
  "artifacts": {"policy": "model.hmm"},
  "execution": ["policy"]
}
```

Verify the output directory contains:

```
<policy_dir>/
├── model.hmm              # compiled HMM module
├── config.hmm.json        # manifest
├── tcim_work/             # compiler intermediates (can be deleted)
└── act_ros2_hmm.onnx      # source ONNX (keep for re-quantization)
```

### Quantization Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `w8a8h1_sefp` | Weight 8-bit, Activation 8-bit, hardware sefp (default) | General purpose, good accuracy/speed balance |
| `w16a16_sefp` | Weight 16-bit, Activation 16-bit | Higher accuracy, larger model |

## PI05 → HMM Conversion

PI05 is split into 6 modules by the Houmo toolchain. The conversion is performed **externally** using `houmo-examples/pi05/demo.py` — this repo provides only the runtime contract (`PI05HMMModel.py`) and manifest format.

### Step 1: Prepare Environment

```bash
# Install houmo-examples SDK
unzip houmo-examples-xh2_v1.3.0.zip -d /opt/houmo-examples
cd /opt/houmo-examples/pi05
# Follow the demo.py setup instructions for dependencies
```

### Step 2: Run External Compilation

The `houmo-examples/pi05/demo.py` script performs:

1. **Split PI05** into 6 subgraphs:

   | Module | Subgraph | Output File |
   |--------|----------|-------------|
   | vision | SigLIP image encoder | `siglip.hmm` |
   | prefill | Gemma-2B prefill (KV-cache init) | `gemma_2b_prefill.hmm` |
   | decode | Gemma-300M expert decode | `gemma_expert_300m_decode.hmm` |
   | time_mlp | Time embedding MLP | `time_mlp.hmm` |
   | action_in_proj | Action input projection | `action_in_proj.hmm` |
   | action_out_proj | Action output projection | `action_out_proj.hmm` |

2. **PTQ quantize** each module via `xhquant` (XH2a target)
3. **Compile** each via `tcim.build_from_hmonnx`
4. **Dump** Gemma token-embedding table to `embedding.pt`

### Step 3: Assemble Output Directory

Copy all 7 artifacts into a single directory:

```
<pi05_policy_dir>/
├── model/
│   ├── siglip.hmm
│   ├── gemma_2b_prefill.hmm
│   ├── gemma_expert_300m_decode.hmm
│   ├── time_mlp.hmm
│   ├── action_in_proj.hmm
│   ├── action_out_proj.hmm
│   └── embedding.pt
├── config.hmm.json        # manifest (see below)
└── config.json            # PI05 policy config (chunk_size, etc.)
```

### Step 4: Write Manifest

Create `config.hmm.json` manually (the external demo does not generate it):

```json
{
  "schema_version": 1,
  "policy_type": "pi05",
  "backend": "hmm",
  "artifacts": {
    "vision": "model/siglip.hmm",
    "prefill": "model/gemma_2b_prefill.hmm",
    "decode": "model/gemma_expert_300m_decode.hmm",
    "time_mlp": "model/time_mlp.hmm",
    "action_in_proj": "model/action_in_proj.hmm",
    "action_out_proj": "model/action_out_proj.hmm",
    "embedding": "model/embedding.pt"
  },
  "execution": ["vision", "prefill", "decode", "time_mlp", "action_in_proj", "action_out_proj"]
}
```

Key rules:
- `embedding` is in `artifacts` but **NOT** in `execution`
- `execution` lists exactly 6 modules in the order they appear in the pipeline
- Artifact paths are relative to the manifest directory

### Known Gap

The `PI05HMMModel` runtime initializes `action_proj` / `action_out_fc` CPU Linear layers with **random weights** (inherited from the upstream houmo-examples demo). Real deployment must load trained projection weights. Until then, action quality is undefined.

## Runtime Dispatch

HMM models are loaded based on `policy_type` + `backend`:

| Policy | Backend | Runtime Session | Manifest |
|--------|---------|-----------------|----------|
| ACT | `hmm` | `HMMRuntimeSession` | `config.hmm.json` (1 artifact) |
| PI05 | `hmm` | `PI05HMMRuntimeSession` → `PI05HMMModel` | `config.hmm.json` (7 artifacts) |

Launch with `device:=hmm`:

```bash
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:=<yaml> \
    policy_path:=<pi05_policy_dir> \
    device:=hmm
```

## Manifest Validation

The `load_compiled_manifest()` function in `compiled_policy.py` validates:
- `backend` field matches `"hmm"`
- `policy_type` matches `"act"` or `"pi05"`
- All artifacts in `execution` list exist and have `.hmm` suffix
- All required artifacts exist (PI05 also checks `embedding`)

If validation fails, the error message names the missing artifact.

## Troubleshooting

### Issue: `xhquant` or `tcim` ImportError
**Cause**: Houmo SDK not installed or wrong Python environment
**Fix**: Use a dedicated `.venv-hmm`, install `xhquant` + `tcim` per the Houmo SDK docs

### Issue: PTQ quantization accuracy degradation
**Cause**: Random calibration tensors may not represent real data distribution
**Fix**: For production, replace random calibration with real data samples; consider `w16a16_sefp` for higher accuracy

### Issue: `tcim.build_from_hmonnx` compilation fails
**Cause**: Unsupported ONNX operator or shape mismatch
**Fix**: Check the ONNX with `onnx.checker.check_model()`; simplify with `onnxsim` first; check `tcim_work/` for compiler error logs

### Issue: PI05 manifest validation fails
**Cause**: Missing artifact or wrong path in `config.hmm.json`
**Fix**: Ensure all 6 `.hmm` files + `embedding.pt` exist at the declared paths; paths are relative to the manifest directory

### Issue: PI05 actions are garbage
**Cause**: `action_proj` / `action_out_fc` using random weights (known gap)
**Fix**: Load trained projection weights into these CPU Linear layers (requires modifying `PI05HMMModel.__init__`)

## When to Use This Skill

Invoke this skill when:

- Converting ACT models to Houmo HMM format (`export_onnx_hmm.py`)
- Setting up PI05 HMM deployment (manifest, artifact layout)
- Troubleshooting Houmo xhquant / tcim compilation issues
- Understanding HMM manifest format (`config.hmm.json`)

Do **not** invoke for:

- RKNN conversion (use `rknn-convert`)
- Ascend OM conversion (use `pi05_export/`)
- HMM runtime debugging on the board (use `oh-constraints`)
