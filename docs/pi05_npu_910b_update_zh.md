# PI0.5 Ascend 910B NPU 推理更新说明

## 1. 更新概述

本次更新为 IB_Robot 增加 PI0.5 在昇腾 Ascend 910B 上的原生 PyTorch NPU 推理路径，覆盖模型加载、
硬件路由、Selective-99 INT8 权重量化、TorchAir 图执行、AB2 去噪、统一框架接入、单入口性能测试及
回归测试。原有 Ascend 310P 推理路径保持不变。

模型权重不随源码仓库分发。默认参考开源模型
[LeRobot PI0.5 LIBERO finetuned v0.4.4](https://huggingface.co/lerobot/pi05_libero_finetuned_v044)，
使用者需要自行确认并遵守模型仓库的许可证和使用条件。

## 2. 主要更新内容

### 2.1 新增 Ascend 910B 推理后端

新增独立的 PI0.5 910B provider 和模型实现，主要能力包括：

- 使用原生 PyTorch 与 `torch_npu` 加载 PI0.5，不依赖 ONNX/OM 转换链路；
- 支持 TorchAir `fullgraph` 编译，并分别缓存 prefix 与 denoise 图；
- 支持 AB2 去噪求解器，默认采用 6 个推理步；
- 支持 BF16 权重和 Selective-99 INT8 权重；
- 对量化投影执行运行时替换、`gate/up` 融合和 NPU 权重预打包；
- 保留 LeRobot preprocessor、policy、postprocessor 的完整推理语义。

统一策略路由会读取运行设备的物理型号：

- `Ascend310P` 继续使用已有 310P provider；
- `Ascend910B` 和 `Ascend910_93` 使用新增 910B provider；
- 非法或不受支持的设备组合会直接报错，避免静默落入错误后端。

### 2.2 接入统一推理框架

LeRobot Torch session 现在会把 manifest 中的物理设备信息传递给 provider，并允许 provider 在加载
模型前规范化配置。910B provider 会先以 CPU 配置解析 LeRobot 元数据，再按 manifest 指定的 NPU
设备完成模型部署，避免配置解析阶段误触 CUDA 回退逻辑。

服务调用链保持为：

```text
请求输入
  -> LeRobot preprocessor
  -> PI0.5 Ascend 910B policy
  -> LeRobot postprocessor
  -> 统一动作结果
```

### 2.3 新增 Selective-99 INT8 量化工具

新增两阶段离线量化流程：

1. 从公开 BF16 检查点生成逐输出通道对称 RTN INT8 中间权重和选择清单；
2. 将量化张量、保留的 BF16 张量、处理器和 tokenizer 资源合并为可部署检查点。

Selective-99 选择 99 个线性层；运行时完成 `gate/up` 融合后，对应 81 个实际启用的 INT8 投影。
动作专家/DiT 保持 BF16。第二阶段会递归复制 tokenizer 等模型侧资源，并排除缓存和版本控制目录，
避免生成的 bundle 缺少嵌套 tokenizer 文件。

量化方法、命令和产物约束详见
`src/torch_models/torch_models/pi05_ascend_910b/quantization/README.md`。

### 2.4 收敛性能测试入口

NPU 性能测试统一使用：

```bash
source .shrc_local
python3 scripts/npu/pi05_910b/benchmark_pi05.py \
  --bundle "<量化后模型目录>" \
  --deployment torch-npu \
  --warmup 3 \
  --iterations 50 \
  --output "<结果 JSON>"
```

脚本只保留关键的同步 `policy.predict_action_chunk` 墙钟时延，避免把异步下发时间误认为完整模型计算
时间。统一框架和 ROS 端到端链路的历史测试口径及分段结果见
`docs/pi05_npu_910b_inference_latency_zh.md`。

### 2.5 环境初始化更新

910B 与现有 310P NPU 路径统一使用仓库 `venv`。`scripts/setup.sh` 负责创建环境和安装与平台/CANN
匹配的依赖。运行 910B 路径时不要设置 `IBROBOT_CONDA_ENV`；`.shrc_local` 使用默认分支激活仓库
`venv`，并统一加载 ROS 2、workspace overlay、CANN 和项目源码：

```bash
./scripts/setup.sh --yes --profile inference
source .shrc_local
```

910B 优化路径要求兼容的 PyTorch、`torch_npu`、CANN、TorchAir 和 Transformers 版本。发布版本只在
仓库 `venv` 中安装、验证和运行，不使用 Conda；provider 不探测或拒绝调用方环境，但仍会显式校验
关键版本与设备类型，环境不满足时会给出错误，而不是继续运行不可复现的降级路径。

### 2.6 新增示例配置和回归测试

新增 `franka_libero_pi05_910b_latency.yaml`，以已有 LIBERO 评估配置为基础，只覆盖 PI0.5 910B 模型、
`torch-npu` 部署方式和单体执行模式，减少重复配置漂移。

新增测试覆盖：

- 物理 NPU 型号与 provider 路由；
- 910B provider 的配置加载和运行前置校验；
- 图像预处理语义；
- Selective-99 层选择、量化替换、融合及检查点装载；
- 两阶段量化工具对嵌套 tokenizer 资源的复制；
- session 与自定义架构接入契约。

## 3. 兼容性与行为变化

- Ascend 310P 的原有实现和路由不变；
- 910B 路径不会自动回退到 CUDA、CPU 或 310P 实现；
- 量化 bundle 必须包含完整模型侧资源和与内容一致的 schema-v3 manifest；
- 修改 `config.json` 后必须重新生成或校验 manifest 摘要；
- 首次调用可能包含图编译开销，稳定时延统计必须先完成 warmup；
- 权重和性能结果不作为源码发布物，仓库只提供转换工具、配置、说明与测试。

## 4. 本次验证结果

提交前已完成以下验证：

- 完整 Torch 模型测试集（含 910B provider、量化、图像预处理和路由）：`65 passed`；
- 仓库 `venv` 下真实 910B 模型端到端功能验证通过，3 次稳定调用平均 `50.439 ms`，输出为
  `[50, 7]` 有限值；测试时设备存在其他负载，该结果不作为正式性能基线；
- Selective-99 全量模拟转换：选择 99 层，生成 198 个量化张量，并保留 BF16 张量；
- 嵌套 tokenizer 目录复制验证通过；
- 示例配置继承、空相机输入补充、`torch-npu` 部署和单体执行模式解析通过；
- `.shrc_local` 的 Bash/Zsh 语法、默认仓库 `venv` 激活与 `pip check` 通过；
- 代码差异空白检查通过，`libs/lerobot` 子模块无修改。

相关 NPU 实机历史结果为：纯 PI0.5 优化路径约 48～49 ms，统一框架约 53 ms，ROS 客户端端到端约
59～62 ms。各阶段定义、测试顺序和差异原因以专项时延文档为准。

## 5. 已知限制

- 完整机器人启动链路还依赖 ROS 工作区中的 `xacro`、消息包、仿真和硬件插件；仅安装模型依赖不足以
  启动全部机器人节点；
- INT8 路径针对 Ascend 910B 的算子和内存布局优化，不保证可直接用于其他 NPU 型号；
- 图编译缓存、token 长度、去噪步数、CANN/TorchAir 版本和系统负载都会影响时延，跨环境比较时必须
  保持这些变量一致。
