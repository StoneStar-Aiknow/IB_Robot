# Model Utils

model_utils 提供了一组用于 LeRobot 策略模型导出与验证的工具脚本，包含以下工具：

| 脚本 | 用途 |
| --- | --- |
| `export_onnx_atc.py` | 导出 ONNX 模型并通过 ATC 转换为 OM 格式（通用 Ascend 硬件） |
| `export_onnx_3403.py` | 专为 Ascend 3403 硬件导出 ONNX 模型 |
| `export_onnx_rknn.py` | 专为 RK3588 NPU 导出 ONNX 模型，并可一键转换为 RKNN 格式 |
| `export_onnx_hmm.py` | 专为后摩 HMM（LQ50 / M50 xh2）导出 ONNX 模型，并可一键 PTQ + 编译为 `.hmm` 格式 |
| `loss_compare.py` | 跨平台模型推理精度对比验证 |
| `frame_inspect` | 脱机逐帧/区间策略推理检查；需要 `policy-path`、`dataset-root` 和帧选择参数 |
| `pi05_export/` | PI05 策略的 Ascend OM 拆分导出工具链（VLM + Action Expert 两段式导出、ATC 转 OM、量化与验证），提供 `python -m model_utils.pi05_export` 一条命令端到端入口，详见下文 |

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

## export_onnx_hmm.py

> **专为后摩 HMM（LQ50 / M50 xh2）导出 ONNX 模型的工具。**
>
> 复用 RKNN 导出的 action-only ONNX 图（仅输出 `action`、constant folding、onnxsim 简化、opset 13），
> 并可选一键调用后摩大道工具链完成「PTQ 量化 + 编译」生成 `.hmm`，供 `device:=hmm` 加载。

### HMM 转换两阶段流程

`--convert_hmm` 会串联后摩大道工具链的两个阶段（与官方 `houmo-examples` 一致）：

1. **PTQ 量化（ONNX -> HMONNX）**：通过 `xhquant.api.convert_onnx_to_hmonnx`，使用
   `QuantScheme(target_device=DeviceType.XH2a, quant_type=w8a8h1_sefp)` 将 ONNX 量化为 HMONNX 中间格式。
2. **编译（HMONNX -> `.hmm`）**：通过 `tcim.build_from_hmonnx`，针对 `xh2` 目标编译为板端 `.hmm`。

> 转换阶段要求主机已安装后摩大道工具链（`xhquant`、`tcim`）。未安装时会跳过转换并给出提示，
> 此时仍会产出可复用的 `act_ros2_hmm.onnx`。

### 用法

```shell
# 仅导出 ONNX（需要 source .shrc_local 环境下运行，依赖 lerobot + torch）
python export_onnx_hmm.py \
    --policy_path={策略模型目录路径}

# 导出 ONNX 并一键 PTQ + 编译为 HMM（需主机已安装 xhquant + tcim）
python export_onnx_hmm.py \
    --policy_path={策略模型目录路径} \
    --convert_hmm
```

### 参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--policy_path` | ✅ | — | LeRobot 训练出来的策略模型目录路径（与 `--onnx` 二选一） |
| `--onnx` | ✅ | — | 已有 ONNX 模型路径，仅做 strip + simplify（与 `--policy_path` 二选一） |
| `--output` | ❌ | 自动 `_hmm` 后缀 | ONNX 输出路径 |
| `--device` | ❌ | `cpu` | 导出时使用的设备（`cpu` 或 `cuda`） |
| `--convert_hmm` | ❌ | `false` | 导出后自动 PTQ + 编译为 `.hmm` |
| `--hmm_output` | ❌ | `policy_path/model.hmm` | HMM 输出路径 |
| `--hmm_model_name` | ❌ | 目录名 | HMM 制品名称 |
| `--hmm_target` | ❌ | `xh2` | 后摩目标平台（LQ50 / M50 为 `xh2`） |
| `--hmm_ncore` | ❌ | `2` | 编译使用的核数 |
| `--hmm_opt_level` | ❌ | `O2` | 编译优化级别 |
| `--hmm_quant_type` | ❌ | `w8a8h1_sefp` | PTQ 量化类型（传给 `QuantScheme`） |

### 示例

```shell
# 仅导出 HMM 专用 ONNX
python export_onnx_hmm.py \
    --policy_path=path/to/pretrained_model

# 导出 + w8a8 量化 + 编译（推荐用于 ACT 模型）
python export_onnx_hmm.py \
    --policy_path=path/to/pretrained_model \
    --convert_hmm

# 使用已有 ONNX 直接 PTQ + 编译
python export_onnx_hmm.py \
    --onnx=path/to/act_ros2_hmm.onnx \
    --convert_hmm
```

导出的 ONNX 文件为 `act_ros2_hmm.onnx`。若启用 `--convert_hmm`，默认还会在同一
`policy_path` 目录下生成 `model.hmm` 与 `config.hmm.json`（编译制品清单），供 `device:=hmm`
直接按 `policy_path` 加载。

---

## loss_compare.py

> **跨平台模型推理精度对比工具。**
>
> 用于验证模型在不同平台（如 GPU PyTorch 推理 vs NPU OM 推理）上的输出一致性。支持生成基准推理结果和计算 L1 Loss。
>
> **注意**：该脚本现已统一通过 IB-Robot 的 `inference_service.InferenceCoordinator` 加载模型，因此既支持原生 LeRobot torch 模型，也支持 ib robot 中编译好的离线模型（昇腾 OM、3403、RKNN）。后端由 `--device` 参数自动选择，无需修改脚本。pi05 等 VLA 模型在 torch 与 OM 两种后端下都可直接对比。

### 快速开始（推荐）

为缓解「参数繁琐、路径长、含义易忘」三个痛点，loss_compare 提供 **交互向导 + profile 配置 + 派生路径 + 记住上次** 四件套。常用参数只需配置一次，之后一行命令即可复用。

> 旧的完整显式命令（见下方「高级用法（完整参数）」）**完全保持可用**，向后兼容；下面这套只是更省心的入口。

**第一次使用：交互向导**

直接不带参数运行（或加 `--init`），向导（英文）会逐项提示**含义 + 默认值**（目录类参数不显示样例路径，只描述内容/命名要求；非目录参数会给出样例），回车即用默认，结尾可把这组参数存成一个 profile。`policy_type` 会自动检测，不在向导中询问：

```shell
python loss_compare.py          # 无配置时自动进入向导
python loss_compare.py --init   # 任何时候强制重新进向导
```

**日常使用：profile + 实验目录**

```shell
# 引用 profile，只需再给一个实验目录；target/raw/noise 自动派生
python loss_compare.py --profile pi05-om --exp-dir /root/.../0612

# 临时覆盖某个参数（命令行优先级最高）
python loss_compare.py --profile pi05-om --exp-dir /root/.../0612 --device cuda
```

**派生路径约定**：`--exp-dir <DIR>` 会自动派生三条长路径，无需再分别手填：

| 派生项 | 路径 |
| --- | --- |
| `--target_path` | `<DIR>/target.json` |
| `--raw-target-path` | `<DIR>/target_raw.json` |
| `--noise-dir` | `<DIR>/noises/` |

显式传同名参数会覆盖对应派生值。`--generate-target` 时若派生/目标文件已存在，会**报错拒绝覆盖**，需更换 `--exp-dir` 或显式加 `--force`（防止误覆盖基准）。

**记住上次**：每次成功运行后，最终生效参数会自动写入配置文件的 `_last` 段。下次不指定 `--profile` 时即自动复用上次参数（启动时会打印每个参数的来源）：

```shell
python loss_compare.py --exp-dir /root/.../0613   # 其余参数沿用上次
```

**其他配置命令**：

```shell
python loss_compare.py --list-profiles            # 列出已有 profile
python loss_compare.py ... --save-as pi05-torch   # 把当前参数另存为 profile
python loss_compare.py --config /path/to.yaml ... # 指定配置文件
```

#### 配置文件

默认位置 `~/.config/model_utils/loss_compare.yaml`（可用 `--config` 或环境变量 `LOSS_COMPARE_CONFIG` 覆盖），三个段：

```yaml
# 所有 profile 共享的默认值
defaults:
  policy_type: pi05
  seed: 42
  batch_path: /root/.../batches_480_640_first_batch.json

# 命名 profile（常用参数组）
profiles:
  pi05-om:                       # 昇腾 OM 后端
    device: ascend_om
    policy_path: /root/.../019200/
  pi05-torch:                    # GPU torch 基准
    device: cuda
    policy_path: /root/.../019200/

# 由脚本自动回写，等价「记住上次」；不要手动维护
_last:
  profile: pi05-om
  device: ascend_om
  policy_path: /root/.../019200/
  exp_dir: /root/.../0612
```

参数优先级（高 → 低）：**命令行 > `--profile` > `defaults` > `_last`（仅未指定 profile 时）> 内置默认**。

> 提示：`_last` 与 profile 里**不会**保存派生出来的 `target/raw/noise` 绝对路径（只存 `exp_dir`），这样以后换 `--exp-dir` 才能正确重新派生。

---

## 高级用法（完整参数）

以下为不依赖 profile/向导的完整显式用法，适用于自动化/CI 或非标准目录布局。

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
| `--policy_path` | ✅ | — | LeRobot 训练出来的策略模型目录路径（torch 与 OM 共用同一目录，OM 需含 `config.om.json`）。可由 profile 提供 |
| `--batch_path` | ✅ | — | 输入 batch 的 JSON 文件路径。可由 profile/defaults 提供 |
| `--target_path` | ✅* | — | 基准推理输出的 JSON 文件路径（生成或读取）。*可由 `--exp-dir` 派生为 `<DIR>/target.json` |
| `--exp-dir` | ❌ | `None` | 实验目录：自动派生 `target.json`/`target_raw.json`/`noises/`，免去分别手填三条长路径 |
| `--policy_type` | ❌ | `act` | 策略类型提示（实际类型由加载的策略/manifest 自动检测，仅作回退） |
| `--device` | ❌ | `cpu` | 推理后端：原生 torch 用 `cpu`/`cuda`/`npu`；ib robot 离线模型用 `ascend_om`（含 pi05 OM）、`ascend_om_3403`、`rknn` |
| `--model_dtype` | ❌ | `native` | 仅对 torch 后端生效：将模型转为 `fp16`/`bf16`/`fp32`（编译后端使用其固定 dtype，忽略此参数） |
| `--generate-target` | ❌ | `false` | 指定后进入基准数据生成模式 |
| `--force` | ❌ | `false` | generate-target 时允许覆盖已存在的派生/目标文件（默认拒绝覆盖，防误删基准） |
| `--seed` | ❌ | `42` | 随机种子，用于固定扩散/flow-matching 噪声以保证可复现性 |
| `--task` | ❌ | `""` | VLA 策略（PI0/PI05/SmolVLA）的自然语言任务提示词，会被路由进 LeRobot 预处理器的 complementary_data 并 tokenize。生成基准与计算损失两端必须一致，否则对比无意义 |
| `--noise-dir` | ❌ | `None` | 噪声文件目录，用于跨机器精度对比（Scheme C）。可由 `--exp-dir` 派生为 `<DIR>/noises/` |
| `--raw-target-path` | ❌ | `None` | 归一化空间（后处理前）动作的导出/读取路径，用于区分模型漂移与反归一化放大。可由 `--exp-dir` 派生为 `<DIR>/target_raw.json` |

> 配置/向导相关：`--config`（配置文件路径）、`--profile`（引用 profile）、`--save-as`（另存为 profile）、`--init`（强制向导）、`--list-profiles`（列出 profile）。详见上方「快速开始」。

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

### pi05 OM 离线模型对比示例

pi05 在 GPU 上用 torch 生成基准，在昇腾上用 OM 离线模型对比（`--device=ascend_om`）。
由于 flow-matching ODE 对噪声敏感，跨平台对比务必配合 `--noise-dir` 传递相同噪声：

```shell
# 步骤 1：GPU torch 端生成基准 + 噪声 + 归一化空间基准
python loss_compare.py \
    --policy_path=path/to/pi05_model \
    --policy_type=pi05 \
    --device=cuda \
    --batch_path=batches.json \
    --target_path=targets.json \
    --raw-target-path=raw_targets.json \
    --noise-dir=noise_files/ \
    --generate-target

# 步骤 2：昇腾 OM 端计算精度损失（policy_path 目录需含 config.om.json 与 OM 文件）
python loss_compare.py \
    --policy_path=path/to/pi05_model \
    --policy_type=pi05 \
    --device=ascend_om \
    --batch_path=batches.json \
    --target_path=targets.json \
    --raw-target-path=raw_targets.json \
    --noise-dir=noise_files/
```

### OM 后端使用须知

`--device=ascend_om` 时，`policy_path` 目录除了 LeRobot 策略元数据（`config.json`、
`policy_preprocessor.json`、`policy_postprocessor.json` 及对应 safetensors）之外，还必须包含一个
**`config.om.json`** manifest，描述 OM 离线模型的 artifact。pi05（VLM + Action Expert 双 OM）的 manifest 形如：

```json
{
  "schema_version": 1,
  "policy_type": "pi05",
  "backend": "ascend_om",
  "artifacts": {
    "vlm": "vlm.om",
    "action_expert": "ae.om"
  },
  "execution": ["vlm", "action_expert"]
}
```

- `artifacts` 里的路径相对于 manifest 所在目录（可用 `artifact_dir` 指定子目录）；务必与目录下真实存在的 `.om` 文件名一致。
- ACT 单 OM 模型的 manifest 见 `export_onnx_atc.py` 一节（`"artifacts": {"policy": "model.om"}`）。
- pi05 等 VLA 模型需通过 `--task` 提供任务提示词（默认空串）；该提示词必须与生成基准时一致。
- OM 端会自动读取并 strip 掉 `config.json` 中 IB-Robot 特有的键
  （`is_ascend_om_enabled` / `om_vlm_model_path` / `om_action_expert_model_path` 等），无需手动清理。

> **自洽性自检**：在同一台板子上先用 `--generate-target` 生成 OM 基准（含 `--noise-dir`），再用相同噪声跑 compute-loss，应得到 `L1 = 0.000000`、`Cosine = 1.000000`（归一化空间同理）。这可用于在跨平台对比前确认整条 pre/infer/post 流水线确定且可复现。


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

PI05 策略与单体 ACT 模型不同，导出时被拆分为 **VLM 预填充** 与 **Action Expert（AE）去噪** 两个
独立的 ONNX/OM artifact，并共同写入策略目录下的 `config.om.json`（供 `device:=ascend_om` 运行时
按 `vlm -> action_expert` 顺序加载）。相关脚本位于 `model_utils/pi05_export/` 子包。

> **TL;DR**：日常只需记住**一条命令**。在装有 CANN（`atc`）的 Ascend 机器上执行：
>
> ```shell
> python -m model_utils.pi05_export \
>     --policy-path path/to/pretrained_model \
>     --soc-version Ascend310P3
> ```
>
> 它会按 `VLM 导出 → AE 导出 → ATC 转 OM` 自动串起整条链路，生成两个 `.om` 与 `config.om.json`。

### 一条命令的端到端流程（推荐）

`python -m model_utils.pi05_export` 是整个工具链的统一入口，自动编排各阶段并在阶段间正确
传递文件，你无需记忆多个模块路径，也无需手写 `atc` 命令。

```shell
# 仅导出 ONNX（不转 OM；适合在 GPU/CPU 机器上先把 ONNX 准备好）
python -m model_utils.pi05_export \
    --policy-path path/to/pretrained_model

# 导出 ONNX 并转 OM（在 Ascend 机器上一步到位）
python -m model_utils.pi05_export \
    --policy-path path/to/pretrained_model \
    --soc-version Ascend310P3

# 导出 + 转 OM + 等价性验证
python -m model_utils.pi05_export \
    --policy-path path/to/pretrained_model \
    --soc-version Ascend310P3 \
    --verify --task 'pick up the cup'
```

#### 参数

| 参数 | 必填 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--policy-path` | ✅ | — | 本地 PI05 策略目录（含 config + 权重） |
| `--dtype` | ❌ | `fp16` | 导出精度，**同时应用于 VLM 与 AE 两段**（`fp16` / `fp32` / `auto`） |
| `--soc-version` | ❌ | `None` | 给定时追加 ATC→OM 编译（如 `Ascend310P3`，见下文「查看芯片版本号」） |
| `--verify` | ❌ | `false` | 结尾运行拆分 vs 整体等价性验证（需同时给 `--task`） |
| `--task` | ❌ | `None` | `--verify` 所需的任务提示，须与部署 `default_task` 一致 |
| `--device` | ❌ | `cpu` | 导出/验证设备（`cpu` / `cuda:0` / `npu`），会体现在 ONNX 文件名中 |
| `--output-dir` | ❌ | `outputs/onnx` | 导出 ONNX 的目录 |
| `--runtime-save-dir` | ❌ | `runtime_save` | VLM→AE 中转张量目录（保留以便排查） |
| `--force` | ❌ | `false` | 即使产物已存在也强制重建每个阶段 |

#### 特性

- **可断点续跑**：每个阶段的产物路径会被提前预测；若文件已存在则跳过（`▷ skip`）。
  某阶段失败中断后，**重跑同一条命令即从断点继续**（已完成的自动跳过），需要重建则加 `--force`。
- **保留中间产物**：流程不删除任何中间文件，导出的 ONNX、`runtime_save/*.pth`、`config.om.json`
  都保留在盘上，便于检查或局部重跑。
- **实时反馈**：各阶段以子进程运行并透出 stdout/stderr，导出与 ATC 编译进度实时可见，每个阶段
  带 `▶ 开始 / ✓ 完成（耗时） / ✗ 失败` 横幅，不会让人误以为卡住。
- **统一日志风格**：全工具链统一为 `HH:MM:SS LEVEL message`，结尾打印结构化结果块。

#### 日志样例

成功跑通（`--policy-path ... --soc-version Ascend310P3`）的末尾结果块：

```text
HH:MM:SS INFO ────────────────────────────────────────
HH:MM:SS INFO PI05 export pipeline complete
HH:MM:SS INFO ────────────────────────────────────────
HH:MM:SS INFO   VLM ONNX           : outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx
HH:MM:SS INFO   Action Expert ONNX : outputs/onnx/pi05-action_expert_op17_nodyn_fp16_cpu.onnx
HH:MM:SS INFO   VLM OM             : path/to/pretrained_model/vlm.om
HH:MM:SS INFO   Action Expert OM   : path/to/pretrained_model/action_expert.om
HH:MM:SS INFO   OM manifest        : path/to/pretrained_model/config.om.json
HH:MM:SS INFO ────────────────────────────────────────
HH:MM:SS INFO   ✅ DONE
HH:MM:SS INFO ────────────────────────────────────────
```

断点续跑（前两步已完成）：

```text
HH:MM:SS INFO ▷ [1/3] VLM export — skip (exists: outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx)
HH:MM:SS INFO ▷ [2/3] Action Expert export — skip (exists: outputs/onnx/pi05-action_expert_op17_nodyn_fp16_cpu.onnx)
HH:MM:SS INFO ▶ [3/3] ATC → OM compile …
```

#### 导出 ONNX 的文件命名

自动生成的 ONNX 文件名编码了关键导出配置，便于区分不同配置的产物：

```text
pi05-vlm_op17_nodyn_fp16_cpu.onnx
         │    │     │    └─ 导出设备：cpu / cuda / npu
         │    │     └────── 精度：fp16 / fp32
         │    └──────────── 是否使用 dynamo 导出：nodyn / dyn
         └───────────────── ONNX opset 版本（默认 17）
```

> 说明：常量折叠（constant folding）默认开启，故不再编码进文件名。`dynamo` 模式会忽略 `--opset`
> 并固定使用 opset 18。

---

### 分步调用（高级 / 调试）

统一入口内部按顺序调用下列子脚本；需要单独运行某一步时也可直接调用。

#### 导出脚本

| 脚本 | 用途 |
| --- | --- |
| `convert_onnx_vlm` | 导出 VLM 段 ONNX，写入 `config.om.json` 的 `vlm` 条目；同时保存供 AE 使用的运行期张量（`past_kv_tensor`、`prefix_pad_masks`） |
| `convert_onnx_action_expert` | 读取 VLM 导出的运行期张量，导出 AE 段 ONNX，写入 `config.om.json` 的 `action_expert` 条目 |
| `convert_om` | 调用 `atc` 将已导出的 VLM / AE ONNX 编译为 `.om`，`--input_shape` 从 ONNX 静态形状**自动推导**，并补全 `config.om.json` |

```shell
# 1. 导出 VLM 段
python -m model_utils.pi05_export.convert_onnx_vlm \
    --pretrained-policy-path path/to/pretrained_model

# 2. 导出 AE 段（复用步骤 1 的运行期张量）
python -m model_utils.pi05_export.convert_onnx_action_expert \
    --pretrained-policy-path path/to/pretrained_model

# 3. ATC 转 OM（在 Ascend 机器上；可只传其中一个 --vlm-onnx / --ae-onnx 单独转换）
python -m model_utils.pi05_export.convert_om \
    --pretrained-policy-path path/to/pretrained_model \
    --soc-version Ascend310P3 \
    --vlm-onnx outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx \
    --ae-onnx  outputs/onnx/pi05-action_expert_op17_nodyn_fp16_cpu.onnx
```

> 当 `--pretrained-policy-path` 传入的是 HuggingFace Hub repo id 而非本地目录时，需在
> `convert_onnx_vlm` / `convert_onnx_action_expert` / `convert_om` 显式指定 `--om-manifest-dir`，
> 以确保 `config.om.json` 写入真实的本地策略目录而非当前工作目录。

#### 量化（W8A8 PTQ，可选）

将 ONNX 量化为 W8A8 以降低显存带宽压力。**必须提供真实标定数据**（用随机数据标定会得到不可用的模型）。

| 脚本 | 用途 |
| --- | --- |
| `quant.quantize_vlm` | 对 VLM（gemma_2b）ONNX 做 msModelSlim W8A8 量化 |
| `quant.quantize_ae` | 对 AE（gemma_300m）ONNX 做 W8A8 量化（10 步去噪，量化收益被放大约 10×） |
| `quant.inventory_quant_nodes` | 列出可量化节点清单，辅助决定哪些节点保留 fp16 |

```shell
# 先列出可量化节点（决定 fp16 豁免）
python -m model_utils.pi05_export.quant.quantize_vlm \
    --onnx-path outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx --list-nodes

# 量化 VLM（需真实 batch 标定）
python -m model_utils.pi05_export.quant.quantize_vlm \
    --onnx-path   outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx \
    --output-path outputs/onnx/pi05-vlm_w8a8.onnx \
    --policy-path path/to/pretrained_model \
    --batch-path  path/to/batches.json \
    --num-calib   16 \
    --task 'pick up the cup'

# 量化 AE（标定输入为 VLM 导出 / dump_ae_pt 产出的张量）
python -m model_utils.pi05_export.quant.quantize_ae \
    --onnx-path   outputs/onnx/pi05-action_expert_op17_nodyn_fp16_cpu.onnx \
    --output-path outputs/onnx/pi05-ae_w8a8.onnx \
    --calib-dir   path/to/ae_calib_dumps \
    --num-calib   16
```

> 量化得到的是 ONNX；随后仍需用 `convert_om` 将量化后的 ONNX 编译为 `.om`。

#### 验证与诊断脚本

| 脚本 | 用途 |
| --- | --- |
| `verify_pi05_split_equivalence` | 校验拆分导出（VLM + AE）与原始整体 PI05 策略的等价性；使用真实 batch 时需通过 `--task` 指定与部署一致的任务提示。给定 `--vlm-onnx-path` + `--ae-onnx-path` 时仅对比「整体 PyTorch vs ONNX 拆分」 |
| `dump_vlm_pt` | 在 PyTorch 侧 dump VLM 输入/输出张量，用于 PT/ORT/OM 三方逐张量对比 |
| `dump_vlm_ort` | 在 ONNX Runtime（CPU）侧 dump VLM 张量，定位 PT→ONNX 导出误差 |
| `dump_ae_pt` | 在 PyTorch 侧 dump AE 的完整 Euler 去噪轨迹，定位逐步发散点 |

```shell
# 验证（cosine ≥ 0.9999 → ✅ PASS；0.999 ≤ cosine < 0.9999 → ⚠️ MARGINAL；否则 ❌ FAIL）
python -m model_utils.pi05_export.verify_pi05_split_equivalence \
    --pretrained-policy-path path/to/pretrained_model \
    --vlm-onnx-path outputs/onnx/pi05-vlm_op17_nodyn_fp16_cpu.onnx \
    --ae-onnx-path  outputs/onnx/pi05-action_expert_op17_nodyn_fp16_cpu.onnx \
    --task 'pick up the cup'
```

