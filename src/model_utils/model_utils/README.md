# Model Utils

`model_utils` 提供 IB-Robot 模型导出、compiled artifact 打包、精度对比和数据集帧检查工具。
所有部署工具使用统一的 `inference_manifest.json`，并通过共享 writer 生成 artifact hashes、
bundle digest、execution roles 和 runtime tensor bindings。

## 统一 Bundle 规则

可部署策略目录必须包含 LeRobot 语义文件和唯一的 manifest：

```text
policy_bundle/
├── config.json
├── model.safetensors                         # Torch deployment 按需存在
├── policy_preprocessor.json
├── policy_preprocessor_step_*.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_*.safetensors
├── tokenizer/                                # PI0.5 / SmolVLA 按需存在
├── artifacts/<backend>/<deployment>/...
└── inference_manifest.json
```

`config.json` 和 processor 文件由 LeRobot 拥有，工具只读。不要向其中写入 backend flags 或
artifact paths。运行时只从命名 deployment 选择后端，不扫描目录、不猜测文件名、不用环境
变量覆盖 artifact。

Exporter / packager 负责：

1. 读取 compiler/runtime ABI。
2. 为每个 runtime tensor 生成 semantic、name/index、dtype、shape 和 image layout binding。
3. 将运行时 artifact 固化到 `artifacts/<backend>/<deployment>/`。
4. 计算 artifact 与 bundle 文件 SHA-256。
5. 计算 canonical bundle digest 并写入 `inference_manifest.json`。
6. 使用生产 strict loader 验证生成结果。

不要手工编辑 manifest hashes 或 bindings。

导出器默认把可重建的 ONNX、compiler output 和 ABI metadata 放在
`<bundle>/model_utils_work/<backend>/`。最终 manifest 只引用打包到
`artifacts/<backend>/<deployment>/` 的运行时 artifact。ABI JSON 用于生成 bindings，属于
构建输入或编译器输出，不是运行时 artifact；参数帮助中的 `ABI input` / `ABI output` 明确其方向。

## 环境

项目和 ROS 命令前加载：

```bash
source .shrc_local
```

RKNN Toolkit 与主环境依赖可能冲突，使用独立 `.venv-rknn`。Ascend 工具需要可用的 `atc`
和 CANN 环境。HMM 工具需要 TCIM 产物和对应 `model.json` ABI。

## LeRobot Torch CPU / GPU

标准 LeRobot `save_pretrained()` 策略目录可直接生成原生 Torch deployment：

```bash
source .shrc_local

ros2 run model_utils package-torch-deployment \
    --bundle-root /path/to/policy_bundle
```

默认在同一个 `inference_manifest.json` 中生成 `torch-cpu` 和 `torch-cuda`。Torch deployment
不需要 compiled artifact、runtime tensor bindings、execution roles 或 device links，只声明
`backend: torch` 和目标 device。工具仍会自动发现策略 processor/tokenizer/VLM 本地资产，
校验 `model.safetensors`，计算文件 SHA-256 与 bundle digest，并用生产 loader 验证最终结果。

只生成 CPU deployment 或使用自定义名称前缀：

```bash
ros2 run model_utils package-torch-deployment \
    --bundle-root /path/to/policy_bundle \
    --devices cpu \
    --deployment-prefix native
```

上述命令生成 `native-cpu`。`--devices` 也支持 `cuda`、`mps` 和 `npu`；生成 manifest
不要求当前主机具备对应设备，实际加载 deployment 时 runtime 才检查设备可用性。

## ACT Ascend Export

`export_onnx_atc.py` 导出 ACT ONNX、调用 ATC 生成 OM，并将 OM 打包为 `ascend`
deployment。最终打包必须提供 compiler/runtime introspected ABI JSON：

```bash
source .shrc_local

python3 src/model_utils/model_utils/export_onnx_atc.py \
    --pretrained_model /path/to/act_bundle \
    --soc_version Ascend310P3 \
    --om_abi_path /path/to/compiler-introspection/model.om.abi.json \
    --deployment ascend_310p3
```

未显式指定输出时，ONNX 和 ATC OM 工作产物写入
`<bundle>/model_utils_work/ascend/`；最终 OM 自动复制到
`artifacts/ascend/<deployment>/`。`--om_abi_path` 是已有的 compiler/runtime introspection
JSON 输入，ATC 命令本身不生成该文件。

若 ONNX 已存在：

```bash
python3 src/model_utils/model_utils/export_onnx_atc.py \
    --pretrained_model /path/to/act_bundle \
    --soc_version Ascend310P3 \
    --onnx_model_path /path/to/act.onnx \
    --om_model_path /tmp/act.om \
    --om_abi_path /tmp/act.om.abi.json \
    --deployment ascend_310p3 \
    --skip_onnx_export
```

ABI input names必须与 ACT 实际消费的 `observation.state` 和
`observation.images.*` 顺序一致，output 必须是唯一的 `action`。生成的 deployment backend
为 `ascend`，target runtime 为 `acl`。

## ACT Hisilicon SD3403

`export_onnx_hisilicon.py` 负责导出 vendor toolchain 使用的 ACT ONNX。其 `--device` 仅表示
执行 Torch export 的主机设备，不是运行时 backend selector：

```bash
source .shrc_local

python3 src/model_utils/model_utils/export_onnx_hisilicon.py \
    --policy_path /path/to/act_bundle \
    --policy_type act \
    --device cpu \
    --bundle_output /path/to/compiled_bundle
```

Vendor toolchain 生成 SD3403 OM、worker executable 和 ABI JSON 后，使用通用 packager 完成
deployment：

```bash
ros2 run model_utils package-compiled-deployment \
    --bundle-root /path/to/compiled_bundle \
    --deployment sd3403 \
    --backend hisilicon \
    --target-soc sd3403 \
    --target-runtime hisilicon-worker \
    --spec /path/to/hisilicon-package-spec.json
```

Hisilicon deployment 必须包含：

- `execution: ["policy"]`
- `policy` artifact，format `om`
- 可执行的 `worker` artifact，format `executable`
- `policy` role 的完整 inputs/outputs bindings

Hisilicon ONNX 默认写入 `<policy_bundle>/model_utils_work/hisilicon/`。

## ACT RKNN

### 创建 Toolkit 环境

```bash
python3 -m venv .venv-rknn
. .venv-rknn/bin/activate
python -m pip install rknn-toolkit2 onnx onnxruntime
```

### 从 Checkpoint 导出并转换

```bash
source .shrc_local

python3 src/model_utils/model_utils/export_onnx_rknn.py \
    --policy_path /path/to/act_bundle \
    --convert_rknn \
    --rknn_output /tmp/act.rknn \
    --rknn_abi_output /tmp/act.rknn.abi.json \
    --rknn_mode float16 \
    --rknn_venv_python "$WORKSPACE/.venv-rknn/bin/python" \
    --deployment rk3588
```

### 从已有 ONNX 转换

```bash
python3 src/model_utils/model_utils/export_onnx_rknn.py \
    --onnx /path/to/act.onnx \
    --bundle_root /path/to/act_bundle \
    --convert_rknn \
    --rknn_output /tmp/act.rknn \
    --rknn_abi_output /tmp/act.rknn.abi.json \
    --rknn_mode float16 \
    --rknn_venv_python "$WORKSPACE/.venv-rknn/bin/python" \
    --deployment rk3588
```

`--output` 是优化后 ONNX 路径；最终 RKNN 路径使用 `--rknn_output`。Compiler 必须生成
`--rknn_abi_output`，否则 exporter 不会写入 deployment。图像 layout 由 ABI 显式声明，
runtime 只对 `NHWC` image bindings 转换布局。

从 checkpoint 导出时，工作产物默认写入 `<bundle>/model_utils_work/rknn/`；最终 RKNN
自动复制到 `artifacts/rknn/<deployment>/`。`--rknn_abi_output` 是 converter 生成的输出。

完整板端流程见 `docs/OpenHarmony_EmbodiedAI_RKNN_Inference.md`。

## PI0.5 Ascend Split Export

`pi05-export` 将 PI0.5 拆分为 VLM 和 Action Expert，并可完成 ONNX、量化、OM 编译与等价性
验证。默认步骤为 `vlm_onnx,ae_onnx,vlm_om,ae_om`：

```bash
source .shrc_local

ros2 run model_utils pi05-export \
    --policy-path /path/to/pi05_bundle \
    --exp-dir /path/to/pi05_export_run \
    --soc-version Ascend310P3 \
    --device cpu \
    --dtype fp16
```

这里的 `--device` 仅是 Torch export/verification device。OM 完成后工具在 policy bundle 中
更新名为 `ascend` 的 unified deployment，并写入 VLM/Action Expert artifacts、bindings 和
device-pointer links。

选择步骤：

```bash
# 只导出 ONNX
ros2 run model_utils pi05-export \
    --policy-path /path/to/pi05_bundle \
    --exp-dir /path/to/run \
    --steps vlm_onnx,ae_onnx

# 导出、编译并验证
ros2 run model_utils pi05-export \
    --policy-path /path/to/pi05_bundle \
    --exp-dir /path/to/run \
    --soc-version Ascend310P3 \
    --task "pick up the cup" \
    --steps vlm_onnx,ae_onnx,vlm_om,ae_om,verify
```

需要直接调用 OM compiler wrapper 时：

```bash
python3 -m model_utils.pi05_export.convert_om \
    --pretrained-policy-path /path/to/pi05_bundle \
    --soc-version Ascend310P3 \
    --vlm-onnx /path/to/vlm.onnx \
    --ae-onnx /path/to/action_expert.onnx \
    --vlm-abi /path/to/vlm.om.abi.json \
    --ae-abi /path/to/action_expert.om.abi.json \
    --deployment ascend
```

运行 `python3 -m model_utils.pi05_export.convert_om --help` 查看当前参数名称和可选的
`--input-shape` / `--atc-arg` 配置。

## HMM Packaging

HMM 只支持 PI0.5 与 SmolVLA，不支持 ACT。TCIM 编译完成后准备一个只包含路径和 target
选择的 JSON spec，然后执行：

```bash
source .shrc_local

ros2 run model_utils package-hmm-deployment \
    --bundle-root /path/to/policy_bundle \
    --deployment lq50 \
    --target-soc lq50 \
    --target-runtime tcim-lite \
    --spec /path/to/hmm-package-spec.json
```

Packager 读取各 TCIM `model.json` ABI，生成 execution、bindings、device links、SHA-256 和
bundle digest。SmolVLA 还要求 `state_projection.pt`。详细 spec 和转换流程见
`docs/Houmo_HMM_Conversion.md`。

## 通用 Compiled Deployment Packager

Ascend、Hisilicon、RKNN 或 HMM 的 vendor compiler 已经产出 artifact 和 ABI 时，可使用：

```bash
ros2 run model_utils package-compiled-deployment \
    --bundle-root /path/to/policy_bundle \
    --deployment <name> \
    --backend <ascend|hisilicon|rknn|hmm> \
    --target-soc <soc> \
    --target-runtime <runtime> \
    --spec /path/to/package-spec.json
```

Spec 顶层字段：

```json
{
  "execution": ["policy"],
  "roles": {
    "policy": {
      "artifact": "./model.rknn",
      "format": "rknn",
      "abi": "./model.rknn.abi.json",
      "abi_format": "runtime",
      "input_semantics": {
        "state": "observation.state"
      },
      "output_semantics": {
        "actions": "action"
      },
      "image_layouts": {}
    }
  },
  "artifacts": {},
  "device_links": []
}
```

每个 execution role 必须有 artifact、ABI、完整 input semantic mapping 和 output semantic
mapping。`abi_format` 为 `runtime` 或 `tcim`。

## loss_compare

`loss_compare.py` 使用 `PureInferenceEngine` 和命名 deployment 比较同一 batch 在不同 runtime
上的输出。它不接受 runtime backend `--device`；选择 bundle deployment 使用
`--deployment`。

### 生成基准

```bash
source .shrc_local

python3 src/model_utils/model_utils/loss_compare.py \
    --policy_path /path/to/policy_bundle \
    --deployment cuda \
    --batch_path /path/to/batches.json \
    --exp-dir /path/to/experiment \
    --generate-target
```

`--exp-dir` 自动派生：

```text
target.json
target_raw.json
noises/
```

PI0.5/SmolVLA 的 external noise 会保存到 `noises/`，使不同机器和 deployment 使用相同
随机输入。

### 比较 Compiled Deployment

```bash
python3 src/model_utils/model_utils/loss_compare.py \
    --policy_path /path/to/policy_bundle \
    --deployment ascend \
    --batch_path /path/to/batches.json \
    --exp-dir /path/to/experiment
```

可用 `--model_dtype native|fp16|bf16|fp32` 请求 Torch model dtype；不支持该 runtime option
的 deployment 会拒绝启动。`loss_compare` 每个独立 sample 前重置 engine，并同时比较最终
postprocessed action 和可选 raw action。

配置文件默认位于 `~/.config/model_utils/loss_compare.yaml`。优先级为 CLI、profile、
defaults、last、builtin。查看全部参数：

```bash
python3 src/model_utils/model_utils/loss_compare.py --help
```

## frame_inspect

`frame_inspect` 从 LeRobot dataset 选择一个 frame 或 frame range，运行原生 policy，并导出
预测、label、delta、图像和汇总文件。

```bash
source .shrc_local

ros2 run model_utils frame_inspect \
    --policy-path /path/to/policy \
    --dataset-repo-id organization/dataset \
    --dataset-root /path/to/dataset_root \
    --output-dir /tmp/frame_inspect \
    --episode-index 0 \
    --frame-index 10 \
    --device cpu
```

这里的 `--device` 是离线 Torch policy device，不是 unified runtime backend selector。区间示例
使用 `--frame-index 10:30`；也可以使用 `--global-index` 选择全局 frame。

## Manifest 验证与排障

验证一个 deployment：

```bash
source .shrc_local
PYTHONPATH=src/inference_manifest \
python3 -c "from inference_manifest import load_inference_manifest; print(load_inference_manifest('/path/to/policy_bundle', 'cpu').fingerprint)"
```

常见错误：

| 错误 | 处理 |
| --- | --- |
| deployment 不存在 | 使用 manifest 中实际的 deployment 名称或重新运行 exporter |
| `SHA-256 mismatch` | artifact 或 semantic file 已改变；重新运行 owning exporter |
| `Bundle digest mismatch` | 重新生成完整 manifest，不要手改 digest |
| missing semantic dependency | 将 processor/tokenizer dependency vendored 到 bundle 后重新打包 |
| binding name/shape/layout mismatch | 用 compiler/runtime 实际 ABI 重新生成 bindings |
| unsupported policy/backend pair | 使用 registry 支持矩阵中的组合 |

初始支持矩阵：

| Policy family | `torch` | `ascend` | `hisilicon` | `rknn` | `hmm` |
| --- | --- | --- | --- | --- | --- |
| ACT | 支持 | 支持 | 支持 | 支持 | 不支持 |
| PI0.5 | 支持 | 支持 | 不支持 | 不支持 | 支持 |
| SmolVLA | 支持 | 不支持 | 不支持 | 支持 | 支持 |

更完整的运行时架构、digest 算法和 pipeline 配置见 `src/inference_service/README.md`。
