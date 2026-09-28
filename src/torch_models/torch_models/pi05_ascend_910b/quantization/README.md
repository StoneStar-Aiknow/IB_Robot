# PI0.5 Ascend 910B 权重量化说明（选择性量化 99 层）

本文说明 `torch_models.pi05_ascend_910b` 的原生 Torch 权重量化流程。它面向
Ascend 910B，采用**选择性量化 99 层、无平滑、W8A8**方案：离线只量化 99 个收益明确的
`Linear` 线性层，权重采用每输出通道舍入到最近值（RTN）的 INT8 量化，激活值在运行时
按词元动态量化。为便于与代码和产物对应，类名、字段名、文件名及命令行参数保留英文
原文，其余说明均使用中文。本文不改变量化代码，也不覆盖已有 ONNX/OM 流程。

## 1. 边界：这不是 ONNX/OM 量化

本流程生成原生 Torch 检查点（`model.safetensors` + `config.json`），由
`src/torch_models/torch_models/pi05_ascend_910b` 中的提供器和模型实现，在加载后替换
`Linear` 线性层、融合投影并执行 NPU 矩阵乘。它不会导出 ONNX、HMONNX/OM，也不会构建或修改
`models/_work` 中的 OM 候选产物；既有 Ascend310P/ONNX/OM 量化配置和 CANN 合约保持
独立。不要把本目录生成的 safetensors 文件当作 OM 文件，也不要用 OM 的校准参数或
逐张量参数填充此检查点。

运行时配置必须满足：

```json
{
  "quant_method": "int8_w8a8", "w_bits": 8, "a_bits": 8,
  "w_format": "int", "a_format": "int", "smooth": false,
  "group_size": 0
}
```

`QuantizationConfig` 会拒绝 SmoothQuant、非 8 位、浮点格式和分组权重。

## 2. 精确选择范围

第一阶段脚本从 BF16 `model.safetensors` 的键名计算选择，不依赖模块遍历顺序，且
命中数不正确会直接失败。99 个层为：

| 组件 | 模块后缀 | 数量 |
| --- | --- | ---: |
| SigLIP 视觉 Transformer | `paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.<i>.mlp.fc2` | 27 |
| 前缀大语言模型注意力层 | `...language_model.layers.<i>.self_attn.o_proj` | 18 |
| 前缀大语言模型多层感知机 | `...language_model.layers.<i>.mlp.gate_proj/up_proj/down_proj` | 54 |
| 合计 |  | **99** |

排除 `embeddings`、`embed_tokens`、`norm`/`layernorm`、`lm_head`；动作专家/DiT
全部保持 BF16。前缀多层感知机每层的 `gate_proj` 和 `up_proj` 在运行时沿输出维融合，两个
独立层变成一个 `gate_up`，因此 54 个多层感知机投影减去 18 个重复入口，再加 18 个
`o_proj` 与 27 个视觉 Transformer `fc2`，最终得到 **81 个实际启用的 INT8 投影**。
Q/K/V 融合仍会执行，但本选择表没有量化前缀 Q/K/V；融合不得改变选择计数。

## 3. 量化数学与数据格式

对每个选中权重矩阵 `W ∈ R[out,in]`，在 CPU 上转 FP32 并按输出行独立计算：

```text
scale[o] = max(max(abs(W[o, :])) / 127, 1e-8)
qweight[o,i] = clamp(round(W[o,i] / scale[o]), -127, 127)
W近似[o,i] = qweight[o,i] * scale[o]
```

代码实现为先计算 `max(abs(W[o,:])) / 127`，再执行 `clamp_min(1e-8)`。这是对称、
每输出通道、舍入到最近值（RTN）的量化，不是 SmoothQuant、GPTQ 或校准量化；
误差检查要求每行最大误差不超过
`scale[o]/2`（浮点容差除外）。权重码为 `int8`，缩放因子固定为 `float32`。

激活值不在离线阶段校准。每次前向计算将输入行（即词元）`x` 动态量化：

```text
a_scale[row] = max(max(abs(x[row, :])) / 127, 1e-8)
qactivation = clamp(round(x / a_scale), -127, 127).int8
Y = (qactivation @ qweight.T).int32 * a_scale[:,None] * scale[None,:]
```

NPU 使用 `torch_npu.npu_dynamic_quant(..., quant_mode="pertoken")` 和
`torch_npu.npu_quant_matmul`；CPU 使用同等语义的 PyTorch int32 参考路径，输出再转
为调用方数据类型（通常为 BF16）。偏置仍为浮点，并在反量化后相加。

### 状态字典结构

量化 `Int8W8A8Linear` 不再保存 `weight`，而是：

| 键 | 数据类型/形状 | 含义 |
| --- | --- | --- |
| `<module>.qweight` | `int8 [out,in]` | 权重码（范围 `[-127,127]`） |
| `<module>.weight_scale` | `float32 [out]` | 每输出通道反量化缩放因子 |
| `<module>.bias` | 原数据类型 `[out]`，可选 | 原始偏置 |

`smooth_scale` 不存在；NPU 的 `_npu_qweight_prepacked` 是非持久化运行时缓存，
不会写回状态字典。上层 LeRobot 采用 `strict=False` 加载检查点；随后由
`validate_quantized` 补充缺失张量检查并验证内容，缺失缩放因子（以 NaN 作为哨兵值）
或全零 `qweight` 均会报错。

## 4. 两阶段生成命令

在仓库根目录执行。输入 BF16 目录必须含 `config.json` 和单个
`model.safetensors`；输出目录必须不存在。

```bash
source .shrc_local && python3 \
  scripts/npu/pi05_910b/make_int8_rtn_perchannel_source.py \
  --fp-model-path /path/to/pi05_bf16 \
  --output-dir /path/to/pi05_sel99_rtn_source

source .shrc_local && python3 \
  scripts/npu/pi05_910b/make_selective_positive_quant_pi05_checkpoint.py \
  --fp-model-path /path/to/pi05_bf16 \
  --quant-model-path /path/to/pi05_sel99_rtn_source \
  --source-manifest /path/to/pi05_sel99_rtn_source/selection_provenance.csv \
  --output-dir /path/to/pi05_sel99_int8
```

第一阶段产物：

* `model.safetensors`：仅含 99 层的 `qweight`/`weight_scale`；
* `selection_provenance.csv`：模块、分组、权重量化和运行时激活量化说明；
* `rtn_source_metadata.json`：方法、正则、99/分组计数、误差报告、BF16 输入和
  输出文件 SHA256。

第二阶段复制 BF16 的处理器和分词器配套文件，保留 `tokenizer_max_length`，
将选中层替换成第一阶段张量，其余模型权重（包括动作专家/DiT）原样以 BF16
复制。它写出：

* `model.safetensors`、`config.json`；
* `selective_quant_metadata.json`：两阶段方法、99/81 计数、加载张量计数、
  三个输入来源 SHA256、RTN 元数据 SHA256 和最终输出 SHA256。

第二阶段在写文件前会验证 RTN 源元数据的 `output_sha256`、方法包含 RTN、计数、
每个选中模块的两个量化张量和来源清单。重新生成时请使用新目录，避免
覆盖既有产物。

## 5. 加载与运行时流程

1. 提供器读取 `config.json`，把 `quantization` 对象解析为本地
   `QuantizationConfig`；模型在加载前按包含/排除正则表达式将 99 个 `nn.Linear`
   替换为 `Int8W8A8Linear`，然后加载上述状态字典并进行完整性检查。
2. `prepare_inference_optimizations` 在权重加载之后执行 QKV 融合和前缀 `gate/up`
   融合；融合只拼接输出维，因为共享输入的逐词元激活缩放因子相同，所以结果与
   分开执行保持一致。动作专家不进行多层感知机融合。
3. 所有融合完成后，枚举去掉 `gate/up` 与 `q/k/v` 别名的实际启用 INT8 模块，调用
   `prepare_npu_qweight_layout("nz")`。它把 `qweight.T` 转成 Ascend
   `FRACTAL_NZ`，并断言 Torch-NPU 格式标识为 **29**（Torch-NPU 2.10 返回符号名
   `FRACTAL_NZ`）；代码会先显式启用内部格式。这是运行时非持久化的预排布缓存，
   不是新的检查点格式。
4. 前向计算由动态逐词元激活量化、INT8 通用矩阵乘和 FP32 缩放后处理完成，
   再转为 BF16 或调用方数据类型；NPU 图编译捕获的是预排布后的模块。普通 CPU 参考路径
   不需要 `torch_npu` 或 FRACTAL_NZ。

## 6. 环境与版本合约

代码记录并检查的提供器合约是：物理设备名包含 `Ascend910B`，或者采用 CANN/
Torch-NPU 对 93 系列 910B 产品返回的 `Ascend910_93*` 名称（本机为
`Ascend910_9362`）；PyTorch 和
Torch-NPU 的基础版本都是 **2.10.0**；Transformers 为 **>=5.4,<5.6**；
模型数据类型为 `native` 或 `bf16`，实际计算为 BF16。已记录的验证环境为 Ascend
910B3、驱动 25.5.0、CANN 9.2（运行时/GE/OPP 软件包树显示 9.1）、Python 3.12.13、
PyTorch 2.10.0+cpu、Torch-NPU 2.10.0、TorchAir（`torch_npu.dynamo.torchair`）。
当前验证使用项目脚本创建的仓库 `venv`，Python 3.12.14，并安装 PyTorch 2.10.0、
Torch-NPU 2.10.0、TorchVision 0.25.0、Transformers 5.5.4、LeRobot 0.6.0 和 pytest 8.4.2。
910B 发布版本只在仓库 `venv` 中安装、验证和运行，不使用 Conda；provider 不额外探测或拒绝
调用方环境。这里列出的
源环境与仓库环境是版本记录和“失败即终止”条件，不代表任意 910B 主机都已验证；
部署前必须由目标机自行确认驱动/CANN/Torch-NPU ABI 兼容，不能仅凭型号宣称性能。

## 7. CPU 合约验证

所有项目命令都先加载完整 `.shrc_local`，不要手工拼接 `PYTHONPATH` 或只激活虚拟环境：

```bash
cd /path/to/IB_Robot

source .shrc_local && python3 -m pytest -q \
  src/torch_models/test/test_pi05_ascend_910b_quantization.py

source .shrc_local && python3 -m pytest -q \
  src/torch_models/test/test_pi05_ascend_910b_quantization.py \
  src/torch_models/test/test_pi05_ascend_910b_provider.py
```

前一命令覆盖配置限制、替换、状态字典加载哨兵、CPU 动态量化、QKV/`gate-up`
融合和实际启用模块的预排布选择；后一命令还覆盖提供器解析和版本/设备的“失败即终止”
逻辑。没有 NPU 时，不要把 CPU 测试结果解释为 910B 算子或时延验证结果。

## 8. 真实 910B 验证清单

在目标机执行 `./scripts/setup.sh --yes --profile inference`，再执行 `source .shrc_local` 后，
至少留存以下证据：

* `npu-smi info` 输出的硬件型号和驱动信息；使用 `python3 -c` 打印 `torch`、`torch_npu`、
  `transformers`、TorchAir 版本；确认满足上一节合约。
* 读取 `selective_quant_metadata.json`，核对三项输入 SHA256、RTN 元数据哈希、
  `selected_linear_count=99`、`expected_active_projection_count_after_gate_up_fusion=81`，
  并核对来源清单行数和分组 `27/18/54`。
* 加载模型包后确认优化元数据：实际启用的量化层为 81、
  `int8_prepack.enabled=true`、格式为 `29`/FRACTAL_NZ；确认动作专家/DiT
  仍是 BF16，不能只看到模型加载成功就视为验证通过。
* 用固定图配置运行代表性图像/文本和完整去噪，记录首轮预热后的多轮 p50/p95
  时延、峰值显存/内存、输出有限性，并与同版本 BF16 模型包对齐输入、词元长度、
  去噪步数和批量大小。
* 进行任务级准确率、动作成功率回归和数值差异检查；分别抽样视觉、前缀大语言模型、
  动作专家路径。只通过算子冒烟测试不足以证明模型质量。

### 当前仓库 venv 验证记录（2026-09-28）

仓库 `venv` 中 `python3 -m pip check` 返回“未发现损坏的依赖关系”（原始输出为
`No broken requirements found`）。环境可通过 Torch-NPU 识别 `Ascend910_9362` 并导入
`torch_npu.dynamo.torchair`；完整 Torch 模型测试集共 **65 项通过**，覆盖 910B provider、量化、
图像预处理和设备路由。

同日在仓库 `venv` 中完成一次真实 910B 模型端到端功能验证，进程内不存在 Conda 环境变量。
Selective-99 权重成功加载，81 个实际 INT8 投影完成约 2.02 GB 的 `FRACTAL_NZ` 预排布，
TorchAir prefix/denoise 双图、AB2/6 和 NPU 融合算子均成功启用。首次图编译预热为
55.844 s；随后 3 次 preprocessor → policy → postprocessor 同步调用分别为
51.742 ms、49.782 ms 和 49.793 ms，平均 50.439 ms；动作输出形状为 `[50, 7]` 且全部为有限值。

测试时设备上另有评测任务，因此这 3 次数据只证明 venv-only 优化链路可运行，不作为正式性能基线，
也不包含统一框架或 ROS 传输。正式时延仍需在空闲 NPU 上按基准脚本的预热和迭代口径复测。

## 9. 失败模式、限制与性能/精度注意事项

* 选中层不是 99、正则表达式与模型不匹配、缺少 `config.json` 或单个 safetensors 文件、
  输出目录已存在，都会“失败即终止”；不要通过删掉校验或改计数“修复”。
* RTN 是无校准的每通道近似；缩放因子下限、舍入/截断、BF16 输出转换都会
  影响边界值。`scale`/`qweight` 缺失、NaN 或全零会在加载后报错。
* 平台若没有 Torch-NPU，NPU 前向计算和预排布不可用；CPU 只用于语义和回归测试。
  `FRACTAL_NZ` 只在 `qweight` 已位于 NPU 且 Torch-NPU 支持格式 29 时生成。
* 融合减少通用矩阵乘和访存调用，但实际收益取决于词元长度、张量形状、CANN 算子和
  图编译缓存；81 是实际启用的投影数，不是 81 个原始状态字典层。
* 无平滑方案避免额外平滑状态和校准流程，但不等于精度无损。必须以同版本 BF16
  基线做任务级精度与性能对照；本文不承诺固定吞吐、时延或准确率提升。

## 10. 来源与维护边界

离线工具来源为公开的
[`Launch-pad-Infinity-Edge/lerobot_offical`](https://github.com/Launch-pad-Infinity-Edge/lerobot_offical)
仓库 `main` 分支，基线提交为 `f30268ab57f7cf26c496af58aa96892f21535600`，选择性量化
99 层的实现起源于 `a04ac0e`。本目录的运行时类属于 `torch_models.pi05_ascend_910b`，不导入或修改
`libs/lerobot`、Ascend310P 实现或共享策略注册表。若选择规则、版本合约或
状态字典结构改变，应同步更新本说明、脚本说明、元数据校验和 CPU 测试。
