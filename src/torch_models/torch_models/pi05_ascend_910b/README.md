# PI0.5 Ascend 910B 原生 PyTorch NPU 推理说明

本目录提供 IB_Robot 中 PI0.5 面向昇腾 Ascend 910B 系列 NPU 的原生 PyTorch 推理实现。该实现通过
`torch_npu` 和 TorchAir 执行模型，不经过 ONNX/OM 转换，同时支持 BF16 权重与 Selective-99 INT8
权重。原有 Ascend 310P 推理实现保持独立，不受本目录代码影响。

模型权重不随源码仓库分发。当前实现使用的公开基线为
[LeRobot PI0.5 LIBERO finetuned v0.4.4](https://huggingface.co/lerobot/pi05_libero_finetuned_v044)，
使用者需要自行下载模型，并确认、接受模型仓库标注的许可证和使用条件。

## 1. 实现定位

本实现从公开项目
[`Launch-pad-Infinity-Edge/lerobot_offical`](https://github.com/Launch-pad-Infinity-Edge/lerobot_offical)
的 PI0.5 NPU 实现适配而来，参考 revision 为
`f30268ab57f7cf26c496af58aa96892f21535600`。移植的源文件保留 Apache-2.0 许可证声明。

代码以独立包的形式接入 IB_Robot，不修改 `libs/lerobot`。本目录自行维护 PI Gemma、SigLIP NPU、
910B policy 和 INT8 量化模块，仅复用 LeRobot v0.6 的公开配置、预处理器和后处理器接口。

主要文件如下：

| 文件/目录 | 作用 |
|---|---|
| `provider.py` | 校验运行环境和模型 bundle，加载 PI0.5 配置并创建统一 provider |
| `modeling_pi05_ascend_910b.py` | PI0.5 policy、TorchAir 图执行、AB2/Euler 去噪及 NPU 融合优化 |
| `pi_gemma.py` | PI Gemma 模型适配 |
| `vision_siglip_npu.py` | SigLIP 视觉编码器的 NPU 实现 |
| `quantization/` | Selective-99 W8A8 量化配置、算子替换、融合和 INT8 kernel |

## 2. 推理调用链

统一推理框架中的调用顺序为：

```text
schema-v3 inference manifest
  -> 根据 policy=pi05、deployment=torch-npu 和物理 NPU 型号选择 provider
  -> 读取并规范化 config.json
  -> 严格加载本地 model.safetensors、processor 和 tokenizer
  -> 初始化 PI05Ascend910BPolicy
  -> 启用 NPU 融合、INT8 权重预打包和 TorchAir 图
  -> LeRobot preprocessor
  -> policy.predict_action_chunk()
  -> LeRobot postprocessor
  -> 统一动作结果
```

硬件路由规则如下：

- 物理设备名包含 `Ascend910B`，或者使用 CANN/Torch-NPU 的 `Ascend910_93*` 命名时，进入本目录的
  910B provider；
- `Ascend310P` 继续使用原有 310P provider；
- 设备型号、运行库版本或 bundle 不符合约束时直接报错，不自动回退到 CUDA、CPU 或 310P 路径。

配置解析阶段会暂时使用 CPU 设备值读取 LeRobot 配置，统一 session 随后再写入 manifest 指定的 NPU
设备。该处理只用于绕开 LeRobot v0.6 不识别 `npu` 配置的问题，不代表模型会在 CPU 上推理。

## 3. 推理优化

910B 路径启用以下优化：

- TorchAir prefix/denoise 双图编译，固定 shape 场景使用 `fullgraph=true`；
- NPU fused ops；
- QKV 融合；
- Prefix MLP 融合；
- shared-prefix FIAS；
- AdaRMS bias 融合；
- INT8 权重在融合完成后预打包为 NPU `FRACTAL_NZ` 布局；
- BF16 模型执行，并保持 LeRobot 动作输出契约。

provider 的图编译默认配置为：

```text
compile_inference_graph=true
compile_inference_backend=torchair
compile_inference_fullgraph=true
compile_inference_dynamic=false
compile_frozen_parameter=true
compile_tiling_schedule_optimize=true
```

### 3.1 AB2 与推理步数

实现同时支持 Euler 和 AB2 去噪。未在模型配置中声明时，provider 保持 `euler` 默认值。当前 910B
优化时延测试使用量化后模型配置中的：

```json
{
  "denoise_solver": "ab2",
  "num_inference_steps": 6
}
```

因此，“AB2、6 steps”是当前优化 bundle 的配置，不是对所有 PI0.5 检查点强制覆盖的全局默认值。
修改求解器或步数会改变图缓存、精度和时延，修改 `config.json` 后还必须同步更新并校验 manifest 摘要。

## 4. 权重与 bundle 要求

### 4.1 BF16 bundle

BF16 输入检查点至少需要包含：

- `config.json`；
- 单个 `model.safetensors`；
- LeRobot preprocessor 和 postprocessor 配置；
- 完整 tokenizer 文件，包括 tokenizer 子目录中的资源。

910B 优化路径只接受 `model_dtype=native` 或 `model_dtype=bf16`，模型实际以 BF16 物化。

### 4.2 Selective-99 INT8 bundle

Selective-99 使用 W8A8 方案：

- 离线权重采用逐输出通道对称 RTN INT8；
- 运行时激活采用逐 token 动态对称 INT8；
- 共选择 99 个 Linear 层；
- `gate/up` 融合后实际启用 81 个 INT8 投影；
- 动作专家/DiT 保持 BF16；
- `smooth=false`，`group_size=0`。

量化采用两阶段转换：第一阶段生成 RTN 中间权重与选择清单，第二阶段合并 INT8/BF16 张量以及
processor、postprocessor 和 tokenizer 资源。完整命令、张量契约和校验方法见
[`quantization/README.md`](quantization/README.md) 与
[`scripts/npu/pi05_910b/README.md`](../../../../scripts/npu/pi05_910b/README.md)。

最终部署目录还必须包含 schema-v3 `inference_manifest.json`。manifest 中应使用：

```text
policy: pi05
deployment: torch-npu
model interface: policy/pi05/predict
```

权重、配置、tokenizer 和 manifest 摘要必须相互一致，不允许把不同转换批次的文件混合使用。

## 5. 环境要求

已验证的核心软件组合为：

| 组件 | 版本/约束 |
|---|---|
| NPU | Ascend 910B 系列；设备名包含 `Ascend910B` 或 `Ascend910_93` |
| CANN | 已验证 CANN 9.2；实际部署需同时满足驱动与 Torch-NPU 兼容矩阵 |
| PyTorch | 基础版本必须为 `2.10.0` |
| Torch-NPU | 基础版本必须为 `2.10.0` |
| Transformers | `>=5.4,<5.6` |
| 模型 dtype | `native` 或 `bf16`，运行时使用 BF16 |

推荐使用项目 Conda 环境初始化依赖和 ROS 工作区：

```bash
export IBROBOT_CONDA_ENV=IB_Robot_ros
source .shrc_local
```

provider 会在加载模型前检查设备型号、PyTorch、Torch-NPU、Transformers、tokenizer 和
`model.safetensors`。任何关键条件不满足都会中止加载，避免使用未经验证的降级路径产生错误结果。

## 6. 使用方法

### 6.1 生成 Selective-99 INT8 权重

从仓库根目录执行：

```bash
export IBROBOT_CONDA_ENV=IB_Robot_ros
export PI05_FP_MODEL_DIR="<BF16 模型目录>"
export PI05_RTN_SOURCE_DIR="<RTN 中间产物目录>"
export PI05_INT8_MODEL_DIR="<最终 INT8 模型目录>"

source .shrc_local && python3 \
  scripts/npu/pi05_910b/make_int8_rtn_perchannel_source.py \
  --fp-model-path "${PI05_FP_MODEL_DIR}" \
  --output-dir "${PI05_RTN_SOURCE_DIR}"

source .shrc_local && python3 \
  scripts/npu/pi05_910b/make_selective_positive_quant_pi05_checkpoint.py \
  --fp-model-path "${PI05_FP_MODEL_DIR}" \
  --quant-model-path "${PI05_RTN_SOURCE_DIR}" \
  --source-manifest "${PI05_RTN_SOURCE_DIR}/selection_provenance.csv" \
  --output-dir "${PI05_INT8_MODEL_DIR}"
```

工具会校验源权重摘要、量化元数据、99 层选择清单以及必需的 `qweight`/`weight_scale` 张量。最终
bundle 的 manifest 仍需按实际部署信息生成，并保证其摘要与量化产物一致。

### 6.2 运行纯模型同步时延测试

```bash
export IBROBOT_CONDA_ENV=IB_Robot_ros
source .shrc_local && python3 scripts/npu/pi05_910b/benchmark_pi05.py \
  --bundle "${PI05_INT8_MODEL_DIR}" \
  --deployment torch-npu \
  --warmup 3 \
  --iterations 50 \
  --output "<测试结果 JSON>"
```

该脚本只统计以下同步区间：

```text
torch.npu.synchronize()
  -> policy.predict_action_chunk()
  -> torch.npu.synchronize()
```

模型加载、预处理、后处理、输出校验、统一框架调度和 ROS 传输均不计入该指标。NPU 图执行是异步的，
没有前后同步的 `backend_latency_ms` 只表示下发时间，不能视为完整模型推理时延。

### 6.3 通过机器人配置启用

项目提供 `franka_libero_pi05_910b_latency.yaml` 示例配置。该配置继承 LIBERO 评估基线，只覆盖额外
相机语义、模型 bundle、`torch-npu` deployment 和 monolithic 执行模式。部署时应把其中的相对模型
路径替换为实际 bundle 位置，不要在可提交配置中写入服务器私有绝对路径。

## 7. 时延口径

当前实机历史结果的典型范围如下，仅用于说明不同测试层级，不构成跨环境性能承诺：

| 测试层级 | 典型时延 | 是否包含框架/通信 |
|---|---:|---|
| 同步 `policy.predict_action_chunk()` | 约 48～49 ms | 不包含统一框架和 ROS |
| 统一推理框架 | 约 53 ms | 包含 pre/postprocess、校验和框架调度 |
| ROS 客户端端到端 | 约 59～62 ms | 进一步包含请求处理、编码、DDS 和 executor 调度 |

完整测试环境、统计方法、顺序分段和差异分析见
[`docs/pi05_npu_910b_inference_latency_zh.md`](../../../../docs/pi05_npu_910b_inference_latency_zh.md)。

## 8. 已知限制

- 首次推理会触发 TorchAir 图编译，不能计入稳定时延均值；
- token 长度、去噪步数、求解器、CANN/TorchAir 版本和系统负载都会影响时延；
- INT8 kernel 和预打包布局针对 Ascend 910B 优化，不保证可直接用于其他 NPU 型号；
- 本目录只负责模型 provider，完整机器人链路仍依赖 ROS 消息包、`xacro`、仿真或硬件插件；
- 当前路径输出动作块形状由模型契约决定，LIBERO 检查点的预期结果为 `[50, 7]`；
- 修改模型或量化配置后必须重新验证精度、manifest 摘要和 NPU 图编译结果。
