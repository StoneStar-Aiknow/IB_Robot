# ACT 与 SmolVLA 详细 Workflow

## When to Read

- 执行 ACT 或 SmolVLA 的 ONNX-to-RKNN 转换时
- 需要参考 high-level exporter 或 low-level 编译命令时
- 需要了解 SmolVLA 三模块导出和 package-only 模式时

## ACT Workflow

The ACT exporter can orchestrate the complete split workflow. Run it from the main workspace and
point it at the isolated compiler interpreter:

```bash
source .shrc_local
python3 src/model_utils/model_utils/export_onnx_rknn.py \
    --policy_path models/<act_bundle> \
    --convert_rknn \
    --rknn_venv_python "$PWD/.venv-rknn/bin/python" \
    --rknn_mode float16 \
    --deployment rknn_rk3588
```

The exporter writes intermediate files below `<bundle>/model_utils_work/rknn`, requires the RKNN
compiler to emit `<model>.rknn.abi.json`, copies the final artifact under the bundle's managed
artifact directory, and calls `upsert_deployment()` to update `inference_manifest.json`.

For an existing ONNX graph, retain the original policy bundle for semantic validation:

```bash
source .shrc_local
python3 src/model_utils/model_utils/export_onnx_rknn.py \
    --onnx /path/to/model.onnx \
    --bundle_root models/<act_bundle> \
    --convert_rknn \
    --rknn_venv_python "$PWD/.venv-rknn/bin/python" \
    --rknn_mode float16 \
    --deployment rknn_rk3588
```

`--onnx --convert_rknn` without `--bundle_root` is invalid because the packager must compare the
compiler ABI against the policy's semantic inputs and output.

### Low-Level ACT Compilation

Use the low-level helper only when compilation must be run independently. Run it in a clean shell:

```bash
env -i HOME="$HOME" PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    PYTHONNOUSERSITE=1 \
    "$PWD/.venv-rknn/bin/python" .agents/skills/rknn-convert/convert_to_rknn.py \
    --onnx /path/to/model.onnx \
    --output /path/to/model.rknn \
    --abi-output /path/to/model.rknn.abi.json \
    --mode float16
```

The `.rknn` and matching ABI JSON are both required. Use the high-level exporter afterward or a
complete `package-compiled-deployment` spec; do not deploy the raw artifact alone.

## SmolVLA Workflow

SmolVLA uses three compiled modules plus host-side token embedding and state projection artifacts.

### Export ONNX Modules

```bash
source .shrc_local
python3 src/model_utils/model_utils/smolvla_export/export_rknn_modules.py \
    --model_path models/<smolvla_bundle> \
    --output_dir models/<smolvla_bundle>
```

Compile the generated vision, prefill, and action ONNX files with the isolated helper. Each RKNN
must retain its matching compiler ABI JSON.

### Package Existing Compiler Outputs

```bash
source .shrc_local
python3 src/model_utils/model_utils/smolvla_export/export_rknn_modules.py \
    --model_path models/<smolvla_bundle> \
    --output_dir models/<smolvla_bundle> \
    --package_only \
    --deployment rknn_rk3588 \
    --target_soc rk3588 \
    --target_runtime rknn-lite2
```

By default, package-only mode expects these files below `<output_dir>/onnx`:

- `smolvla_vision.rknn` and `smolvla_vision.rknn.abi.json`
- `smolvla_prefill.rknn` and `smolvla_prefill.rknn.abi.json`
- `smolvla_action.rknn` and `smolvla_action.rknn.abi.json`

Use `--vision_rknn`, `--vision_abi`, `--prefill_rknn`, `--prefill_abi`, `--action_rknn`, and
`--action_abi` when compiler outputs are elsewhere. The packager also requires `token_embedding.pt`,
`state_projection.pt`, and the original LeRobot metadata in the bundle.
