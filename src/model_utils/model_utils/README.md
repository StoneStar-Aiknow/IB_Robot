# Model Utils

model_utils 提供了一组用于 LeRobot 策略模型导出与验证的工具脚本，包含以下三个工具：

| 脚本 | 用途 |
| --- | --- |
| `export_onnx_atc.py` | 导出 ONNX 模型并通过 ATC 转换为 OM 格式（通用 Ascend 硬件） |
| `export_onnx_3403.py` | 专为 Ascend 3403 硬件导出 ONNX 模型 |
| `loss_compare.py` | 跨平台模型推理精度对比验证 |

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
> 该脚本会先将模型导出为 ONNX 格式，然后自动调用 ATC 工具将其转换为 OM 格式，适用于通用的 Ascend 硬件（如 310P3 等）。

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
