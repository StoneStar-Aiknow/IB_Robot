# Model Utils

model_utils 提供了一组用于 LeRobot 策略模型导出与验证的工具脚本，包含以下工具：

| 脚本 | 用途 |
| --- | --- |
| `export_onnx_atc.py` | 导出 ONNX 模型并通过 ATC 转换为 OM 格式（通用 Ascend 硬件） |
| `export_onnx_3403.py` | 专为 Ascend 3403 硬件导出 ONNX 模型 |
| `export_onnx_rknn.py` | 专为 RK3588 NPU 导出 ONNX 模型，并可一键转换为 RKNN 格式 |
| `loss_compare.py` | 跨平台模型推理精度对比验证 |
| `frame_inspect` | 脱机逐帧/区间策略推理检查；需要 `policy-path`、`dataset-root` 和帧选择参数 |
| `pi05_export/` | PI05 策略的 Ascend OM 拆分导出工具链（VLM + Action Expert 两段式导出与诊断），详见下文 |

---

## 模型文件说明

使用 LeRobot 训练出来的策略模型目录下应包含如下文件：

```
config.json
model.safetensors
policy_postprocessor.json
policy_postprocessor_step_0_unnormalizer_processor.safetensors
policy_preprocessor.json
policy_preprocessor_step_3_normalizer_processor.safetensors
train_config.json
```

其中 `model.safetensors` 是模型权重文件。例如模型文件位于 `path/to/pretrained_model/model.safetensors`，则传参时应使用 `path/to/pretrained_model`。

---

## export_onnx_atc.py

> **通用 Ascend 硬件的模型导出工具。**
>
> 该脚本会先将模型导出为 ONNX 格式，然后自动调用 ATC 工具将其转换为 OM 格式，适用于通用的 Ascend 硬件（如 310P3 等）。ATC 成功后会在策略模型目录写入 `config.om.json`，供 `device:=ascend_om` 运行时按 manifest 加载。

### 用法

```shell
python export_onnx_atc.py \
    --pretrained_model={策略模型目录路径} \
    --soc_version={Ascend 芯片版本号} \
    --onnx_model_path={ONNX 模型导出路径} \
    --om_model_path={OM 模型导出路径}
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--pretrained_model` | ✅ | — | LeRobot 训练出来的策略模型目录路径 |
| `--soc_version` | ✅ | — | 目标 Ascend 芯片版本号（如 `Ascend310P3`） |
| `--onnx_model_path` | ❌ | `{pretrained_model}/model.onnx` | ONNX 模型导出路径 |
| `--om_model_path` | ❌ | `{pretrained_model}/model.om` | OM 模型导出路径 |
| `--skip_onnx_export` | ❌ | `false` | 跳过 PyTorch -> ONNX 导出，直接将已有 ONNX 转为 OM |

### 输出文件

默认会生成：

```text
{pretrained_model}/model.onnx
{pretrained_model}/model.om
{pretrained_model}/config.om.json
```

`config.om.json` 是 compiled runtime 的 sidecar manifest。ACT 单 OM 模型的内容形如：

```json
{
  "schema_version": 1,
  "policy_type": "act",
  "backend": "ascend_om",
  "artifacts": {"policy": "model.om"},
  "execution": ["policy"]
}
```

运行推理时仍以策略模型目录作为 `policy_path`；`config.json` 保存 LeRobot 策略元数据，`config.om.json` 保存 compiled runtime artifact 信息。已有 ONNX 的输入名与尺寸必须与 `config.json` 匹配。

### 查看芯片版本号

可通过 `npu-smi info` 命令查看 Ascend 芯片型号：

```shell
$ npu-smi info
+--------------------------------------------------------------------------------------------------------+
| npu-smi 25.2.3                                   Version: 25.2.3                                       |
+-------------------------------+-----------------+------------------------------------------------------+
| NPU     Name                  | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
| Chip    Device                | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
+===============================+=================+======================================================+
| 224     310P3                 | OK              | NA           71                0     / 0             |
| 0       0                     | 0000:04:00.0    | 0            1263 / 44280                            |
+===============================+=================+======================================================+
```

如上所示芯片名称为 `310P3`，则对应参数为 `Ascend310P3`。

### 示例

```shell
python export_onnx_atc.py \
    --pretrained_model=path/to/pretrained_model \
    --soc_version=Ascend310P3
```

若 ONNX 已存在，可只执行 ATC 转换并生成 `config.om.json`：

```shell
python export_onnx_atc.py \
    --pretrained_model=path/to/pretrained_model \
    --soc_version=Ascend310P3 \
    --onnx_model_path=path/to/pretrained_model/act_ros2.onnx \
    --om_model_path=path/to/pretrained_model/model.om \
    --skip_onnx_export
```

---

## export_onnx_3403.py

> **专为 Ascend 3403 硬件保留的 ONNX 导出工具。**
>
> 由于 3403 的 ATC 转换流程需要单独处理，该脚本 **仅负责导出 ONNX 模型**，不包含 ATC/OM 转换步骤。

### 用法

```shell
python export_onnx_3403.py \
    --policy_path={策略模型目录路径} \
    --policy_type={策略类型} \
    --device={推理设备}
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--policy_path` | ✅ | — | LeRobot 训练出来的策略模型目录路径 |
| `--policy_type` | ❌ | `act` | 策略模型类型（目前支持 `act`） |
| `--device` | ❌ | `cpu` | 推理设备（如 `cpu`、`cuda`） |

### 示例

```shell
python export_onnx_3403.py \
    --policy_path=path/to/pretrained_model \
    --policy_type=act \
    --device=cpu
```

导出的 ONNX 文件将保存在 `policy_path` 目录下，包括原始模型 `act_ros2.onnx` 和简化后的 `act_ros2_simplified.onnx`。

---

## export_onnx_rknn.py

> **专为 RK3588 NPU 导出 ONNX 模型的工具。**
>
> 与 3403 导出相比，RKNN 版本只输出 `action`（去除中间 tensor），启用 constant folding，并可选一键转换为 `.rknn` 格式。

### RKNN 专用优化

- **仅输出 `action`**：去除 3403 导出中附带的 2 个中间输出，减小模型体积和推理开销
- **constant folding**：启用常量折叠优化计算图
- **onnxsim 简化**：进一步精简计算图
- **opset 13**：rknn-toolkit2 对 opset 13 兼容性最好

### 用法

```shell
# 仅导出 ONNX（需要 source .shrc_local 环境下运行，依赖 lerobot + torch）
python export_onnx_rknn.py \
    --policy_path={策略模型目录路径}

# 导出 ONNX 并一键转换为 RKNN
python export_onnx_rknn.py \
    --policy_path={策略模型目录路径} \
    --convert_rknn
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--policy_path` | ✅ | — | LeRobot 训练出来的策略模型目录路径 |
| `--policy_type` | ❌ | `act` | 策略模型类型（目前支持 `act`） |
| `--device` | ❌ | `cpu` | 导出时使用的设备（`cpu` 或 `cuda`） |
| `--convert_rknn` | ❌ | `false` | 导出后自动转换为 RKNN 格式 |
| `--rknn_mode` | ❌ | `float16` | RKNN 转换模式（`float16`/`int8`/`hybrid`） |
| `--rknn_output` | ❌ | `policy_path/model.rknn` | RKNN 输出路径 |
| `--rknn_venv_python` | ❌ | 自动检测 `.venv-rknn/bin/python` | RKNN 专用 Python 解释器路径 |

### 示例

```shell
# 仅导出 RKNN 专用 ONNX
python export_onnx_rknn.py \
    --policy_path=path/to/pretrained_model

# 导出 + float16 RKNN 转换（推荐用于 ACT 模型）
python export_onnx_rknn.py \
    --policy_path=path/to/pretrained_model \
    --convert_rknn \
    --rknn_mode=float16

# 导出 + int8 量化 RKNN 转换（适用于 CNN 模型）
python export_onnx_rknn.py \
    --policy_path=path/to/pretrained_model \
    --convert_rknn \
    --rknn_mode=int8
```

导出的 ONNX 文件为 `act_ros2_rknn.onnx`。若启用 `--convert_rknn`，默认还会在同一
`policy_path` 目录下生成 `model.rknn`，供 `device:=rknn` 直接按 `policy_path` 加载。

---

## loss_compare.py

> **跨平台模型推理精度对比工具。**
>
> 用于验证模型在不同平台（如 GPU PyTorch 推理 vs NPU OM 推理）上的输出一致性。支持生成基准推理结果和计算 L1 Loss。

### 工作流程

1. **生成基准数据**（`--generate-target`）：在 GPU/CPU 上使用 PyTorch 模型对输入 batch 进行推理，将输出保存为 JSON 文件作为基准。
2. **计算精度损失**：在目标平台上使用模型对相同 batch 进行推理，将结果与基准数据逐条对比，计算 L1 Loss。

### 用法

#### 生成基准数据

```shell
python loss_compare.py \
    --policy_path={策略模型目录路径} \
    --policy_type={策略类型} \
    --batch_path={输入 batch JSON 文件路径} \
    --target_path={基准输出 JSON 文件保存路径} \
    --generate-target
```

#### 计算精度损失

```shell
python loss_compare.py \
    --policy_path={策略模型目录路径} \
    --policy_type={策略类型} \
    --batch_path={输入 batch JSON 文件路径} \
    --target_path={基准输出 JSON 文件路径}
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--policy_path` | ✅ | — | LeRobot 训练出来的策略模型目录路径 |
| `--batch_path` | ✅ | — | 输入 batch 的 JSON 文件路径 |
| `--target_path` | ✅ | — | 基准推理输出的 JSON 文件路径（生成或读取） |
| `--policy_type` | ❌ | `act` | 策略模型类型（支持 `act`、`pi05`） |
| `--device` | ❌ | `cpu` | 推理设备（如 `cpu`、`cuda`） |
| `--generate-target` | ❌ | `false` | 指定后进入基准数据生成模式 |
| `--seed` | ❌ | `42` | 随机种子，用于固定扩散/flow-matching 噪声以保证可复现性 |
| `--noise-dir` | ❌ | `None` | 噪声文件目录，用于跨机器精度对比（Scheme C） |

### 噪声文件传递（Scheme C）

当使用 `--noise-dir` 参数时，可实现跨机器（如 GPU 与 NPU）的确定性推理对比：

- **生成基准时（GPU 端）**：自动生成噪声文件 `noise_NNNN.npy` 并保存到指定目录
- **计算损失时（NPU 端）**：从指定目录加载噪声文件，确保两端使用完全相同的噪声

### 示例

```shell
# 步骤 1：在 GPU 机器上生成基准数据和噪声文件
python loss_compare.py \
    --policy_path=path/to/pretrained_model \
    --policy_type=act \
    --batch_path=batches.json \
    --target_path=targets.json \
    --noise-dir=noise_files/ \
    --generate-target

# 步骤 2：在 NPU 机器上计算精度损失
python loss_compare.py \
    --policy_path=path/to/pretrained_model \
    --policy_type=act \
    --batch_path=batches.json \
    --target_path=targets.json \
    --noise-dir=noise_files/
```

---

## frame_inspect

> **脱机逐帧/区间策略推理检查工具。**
>
> 加载训练好的策略模型，对数据集中的单帧或帧区间进行离线推理，输出模型预测值与真实标签的逐维度对比，用于模型行为调试和精度分析。

### 工作模式

1. **单帧模式**：指定 `--global-index` 或 `--episode-index` + `--frame-index`，对单帧推理并输出对比 JSON、summary.txt 和帧图像。
2. **区间模式**：指定 `--episode-index` + `--frame-index start:end`，对连续帧区间逐帧推理，输出对比 CSV/JSON 和视频片段。

### 用法

#### 单帧推理（按全局索引）

```shell
frame_inspect \
    --policy-path path/to/pretrained_model \
    --dataset-repo-id my_dataset \
    --dataset-root path/to/dataset \
    --output-dir path/to/output \
    --global-index 42
```

#### 单帧推理（按 episode + frame）

```shell
frame_inspect \
    --policy-path path/to/pretrained_model \
    --dataset-repo-id my_dataset \
    --dataset-root path/to/dataset \
    --output-dir path/to/output \
    --episode-index 0 \
    --frame-index 15
```

#### 区间推理

```shell
frame_inspect \
    --policy-path path/to/pretrained_model \
    --dataset-repo-id my_dataset \
    --dataset-root path/to/dataset \
    --output-dir path/to/output \
    --episode-index 0 \
    --frame-index 10:30
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--policy-path` | ✅ | — | 策略模型目录路径 |
| `--dataset-repo-id` | ✅ | — | 数据集 repo_id |
| `--dataset-root` | ✅ | — | 数据集根目录路径 |
| `--output-dir` | ✅ | — | 输出目录 |
| `--global-index` | 单帧模式 | — | 数据集全局帧索引 |
| `--episode-index` | 单帧/区间模式 | — | Episode 索引 |
| `--frame-index` | 单帧/区间模式 | — | 帧索引（整数或 `start:end` 格式） |
| `--stats-dataset-repo-id` | ❌ | 同 `dataset-repo-id` | 训练时所用数据集的 repo_id（用于加载归一化统计） |
| `--stats-dataset-root` | ❌ | 同 `dataset-root` | 训练时所用数据集的路径 |
| `--device` | ❌ | 模型配置中的设备 | 推理设备（`cpu`、`cuda`） |
| `--use-imagenet-stats` / `--no-use-imagenet-stats` | ❌ | `--use-imagenet-stats` | 是否对图像使用 ImageNet 归一化统计 |
| `--reset-policy` / `--no-reset-policy` | ❌ | `--reset-policy` | 每帧推理前是否重置策略状态 |

### 输出文件

**单帧模式**：

```text
output_dir/
├── {camera_name}_frame.png     # 输入帧图像
├── comparison.json             # 预测值 vs 标签的逐维度对比
└── summary.txt                 # 制表符分隔的摘要
```

**区间模式**：

```text
output_dir/
├── {camera_name}_clip.mp4      # 输入帧视频片段
├── comparison.csv              # 所有帧的逐维度对比（CSV 格式）
└── comparison.json             # 区间汇总元数据
```

### 特性

- 兼容 dict 和 tuple 两种数据集样本格式
- 自动处理 `observation.current` / `observation.state` 键名差异，缺失时回退并发出警告
- 支持跨数据集归一化（通过 `--stats-dataset-*` 使用训练时的统计信息）

---

## pi05_export（PI05 Ascend OM 拆分导出工具链）

> **注意**：以下为简要说明，后续会补充更详细的端到端文档。

PI05 策略与单体 ACT 模型不同，导出时被拆分为 **VLM 预填充** 与 **Action Expert 去噪** 两个独立的
ONNX/OM artifact，分两步导出，并共同写入策略目录下的 `config.om.json`（供 `device:=ascend_om`
运行时按 `vlm -> action_expert` 顺序加载）。相关脚本位于 `model_utils/pi05_export/` 子包，统一以
`python -m model_utils.pi05_export.<脚本名>` 方式调用。

### 导出脚本

| 脚本 | 用途 |
| --- | --- |
| `convert_onnx_vlm` | 导出 VLM 段 ONNX，并将 `vlm` 条目写入 `config.om.json`；同时保存供 Action Expert 使用的运行期张量（`past_kv_tensor`、`prefix_pad_masks`） |
| `convert_onnx_action_expert` | 读取 VLM 导出保存的运行期张量，导出 Action Expert 段 ONNX，并将 `action_expert` 条目写入 `config.om.json` |

### 验证与诊断脚本

| 脚本 | 用途 |
| --- | --- |
| `verify_pi05_split_equivalence` | 校验拆分导出（VLM + Action Expert）与原始整体 PI05 策略的等价性；使用真实 batch 时需通过 `--task` 指定与部署一致的任务提示 |
| `dump_vlm_pt` | 在 PyTorch 侧 dump VLM 输入/输出张量，用于 PT/ORT/OM 三方逐张量对比 |
| `dump_vlm_ort` | 在 ONNX Runtime 侧 dump VLM 张量（当前固定使用 CPUExecutionProvider，仅适用于 CPU 兼容的 ONNX 图） |
| `dump_ae_pt` | 在 PyTorch 侧 dump Action Expert 输入/输出张量 |

### 推荐工作流

```shell
# 1. 导出 VLM 段（同时生成 Action Expert 所需运行期张量与 config.om.json 的 vlm 条目）
python -m model_utils.pi05_export.convert_onnx_vlm \
    --pretrained-policy-path path/to/pretrained_model

# 2. 导出 Action Expert 段（复用步骤 1 保存的运行期张量，补全 config.om.json）
python -m model_utils.pi05_export.convert_onnx_action_expert \
    --pretrained-policy-path path/to/pretrained_model

# 3.（可选）验证拆分导出与原始策略的等价性
python -m model_utils.pi05_export.verify_pi05_split_equivalence \
    --pretrained-policy-path path/to/pretrained_model \
    --task 'pick up the cup'
```

> 当 `--pretrained-policy-path` 传入的是 HuggingFace Hub repo id 而非本地目录时，需显式指定
> `--om-manifest-dir`，以确保 `config.om.json` 写入真实的本地策略目录而非当前工作目录。
