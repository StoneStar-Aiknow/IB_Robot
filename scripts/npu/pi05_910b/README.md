# PI0.5 Ascend 910B Selective-99 checkpoint tools

This directory owns the two offline tools for building the native Torch
Selective-99 INT8 checkpoint used by the PI0.5 Ascend 910B path and one
synchronized model-inference benchmark:

- `make_int8_rtn_perchannel_source.py` creates the intermediate RTN weight source from a BF16 checkpoint.
- `make_selective_positive_quant_pi05_checkpoint.py` combines that source with the BF16 checkpoint and writes the runtime checkpoint.
- `benchmark_pi05.py` reports only the synchronized `policy.predict_action_chunk()` latency.

## Provenance

These tools are ported from the public
[`Launch-pad-Infinity-Edge/lerobot_offical`](https://github.com/Launch-pad-Infinity-Edge/lerobot_offical)
repository, branch `main`, revision `f30268ab57f7cf26c496af58aa96892f21535600`.
The Selective-99 implementation originated in commit `a04ac0e`. Later
image-preprocessing changes are intentionally outside these offline tools.

Runtime quantization classes are target-owned by
`torch_models.pi05_ascend_910b.quantization`. This directory does not import or
modify `libs/lerobot`, the existing Ascend310P implementation, or the shared
policy registry.

## Checkpoint contract

The input BF16 checkpoint must contain `config.json` and a single
`model.safetensors`. The first stage selects exactly 99 Linear layers:

- 27 ViT `mlp.fc2` layers;
- 18 Prefix LLM `self_attn.o_proj` layers;
- 54 Prefix LLM MLP `gate_proj`, `up_proj`, and `down_proj` layers.

The action expert/DiT remains BF16. Selected weights use per-output-channel
symmetric RTN INT8:

```text
scale[o] = max(max(abs(W[o, :])) / 127, 1e-8)
qweight[o, i] = clamp(round(W[o, i] / scale[o]), -127, 127)
```

Runtime activations use dynamic symmetric per-token `absmax/127`. The generated
`config.json` must retain `quant_method=int8_w8a8`, integer W8A8 formats,
`smooth=false`, `group_size=0`, and the tool-generated include/exclude regexes.
The second stage verifies the first-stage SHA256, metadata, manifest, selected
layer count, and required `qweight`/`weight_scale` tensors before writing the
complete checkpoint. It also copies processor/tokenizer sidecars from the BF16
directory.

Run from the repository root:

```bash
export IBROBOT_CONDA_ENV=IB_Robot_ros

source .shrc_local && python3 scripts/npu/pi05_910b/make_int8_rtn_perchannel_source.py \
  --fp-model-path /path/to/pi05_bf16 \
  --output-dir /path/to/pi05_sel99_rtn_source

source .shrc_local && python3 scripts/npu/pi05_910b/make_selective_positive_quant_pi05_checkpoint.py \
  --fp-model-path /path/to/pi05_bf16 \
  --quant-model-path /path/to/pi05_sel99_rtn_source \
  --source-manifest /path/to/pi05_sel99_rtn_source/selection_provenance.csv \
  --output-dir /path/to/pi05_sel99_int8
```

## Model-inference latency benchmark

The benchmark loads the policy through IB_Robot's runtime provider so the
Ascend 910B INT8, TorchAir, and AB2 optimizations are prepared normally. It
preprocesses the synthetic observation once, outside the measured loop, and
then synchronizes the NPU immediately before and after
`policy.predict_action_chunk()`:

```bash
export IBROBOT_CONDA_ENV=IB_Robot_ros
source .shrc_local && python3 scripts/npu/pi05_910b/benchmark_pi05.py \
  --bundle /path/to/pi05_sel99_int8 \
  --deployment torch-npu \
  --warmup 3 \
  --iterations 50 \
  --output /path/to/pi05_model_latency.json
```

The JSON report has one timing metric, `model_inference_ms`. Model loading,
preprocessing, postprocessing, output materialization and validation, framework
orchestration, and ROS transport are intentionally outside its timing scope.
The synchronization is required because Torch-NPU graph execution is
asynchronous; the framework's unsynchronized `backend_latency_ms` is enqueue
time and must not be used as model execution latency.

## Verified 910B environment

The source guide records Ascend 910B3, Driver 25.5.0, CANN 9.2 package tree
(runtime/GE/OPP metadata 9.1), Python 3.12.13, PyTorch 2.10.0+cpu,
Torch-NPU 2.10.0, and TorchAir from `torch_npu.dynamo.torchair`. The tools
themselves are CPU-side and only require PyTorch, Safetensors, and the Python
standard library; loading and inference additionally require the target-owned
910B runtime package and compatible CANN/Torch-NPU installation.

## Deployment boundary

These tools produce a native Torch `model.safetensors` checkpoint for the
`torch_models.pi05_ascend_910b` implementation. They do not export ONNX, build
Ascend OM artifacts, or alter any existing ONNX/OM quantization profile. The
existing `torch_models.pi05_ascend_310p` native path and its CANN 8.1/310P
contract remain separate.
