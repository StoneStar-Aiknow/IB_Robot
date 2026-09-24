# PI0.5 NPU 910B 推理支线与端到端时延说明

本文记录 IB_Robot 中 PI0.5 在 Ascend 910B 上的 NPU 优化推理支线、测试口径和端到端时延拆分。
本文数据用于定位性能差异，不作为跨设备、跨模型或不同输入配置之间的直接性能承诺。

## 1. 测试结论

当前 PI0.5 NPU 支线已经使用 Selective-99 INT8 权重、TorchAir fullgraph 和 AB2/6。约 53 ms 的
统一框架时延不是量化权重、AB2、模型移植代码、noise 生成或 Conda 环境退化造成的：同一已加载
模型实例绕过统一框架执行完整 preprocessor → policy → postprocessor 时为 49.043 ms，通过
`manager.infer` 时为 52.895 ms。

统一框架增加的 3.852 ms 中，backend 和 postprocessor 后的两次输出校验会分别执行
`detach().cpu().numpy()`。交错消融显示，这两次 NPU→CPU 校验带来 2.914 ms 的端到端增量，约占
框架增量的 75.7%。剩余约 0.9～1.3 ms 来自 unified runtime stage 调度、请求/结果构造、
metadata/codec 等封装及运行波动。

ROS 2 客户端从发送 goal 到收到结果平均需要 61.859 ms，其中服务端框架推理为 53.305 ms，
其余 8.554 ms 包含服务端观测采样与输入转换、manager 外层封装、结果编码，以及 ROS Action/DDS
往返和 executor 调度。这里的 8.554 ms 不能全部视为网络时延。

## 2. 测试环境与配置

| 项目 | 配置 |
|---|---|
| NPU | Ascend 910B，设备名 `Ascend910_9362` |
| Conda 环境 | `IB_Robot_ros` |
| 开源模型 | [LeRobot PI0.5 LIBERO finetuned v0.4.4](https://huggingface.co/lerobot/pi05_libero_finetuned_v044) |
| 推理 bundle | 使用本项目脚本从开源 BF16 权重量化并按 schema-v3 manifest 打包，不随仓库分发权重文件 |
| 量化 | Selective-99 INT8，81 个投影层，qweight 预打包为 `FRACTAL_NZ` |
| 图执行 | TorchAir，`fullgraph=true`，prefix/denoise 双图 |
| 去噪 | AB2，6 steps |
| batch size | 1 |
| action 输出 | `[50, 7]` |
| 当前 token length | 200 |
| 统计方式 | 3 次 warmup 后统计 50 次；首次图编译不计入均值 |

本次官方实现与 IB_Robot 对比使用同一份量化后 `model.safetensors`，其 SHA-256 为：

```text
36264552e19a1c4b50d83a7b592c2c3470cb8945ecef44d3e3f4eaf1219878fb
```

两侧使用的 `tokenizer.json` SHA-256 均为：

```text
ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563
```

因此本轮对比不存在权重文件或 tokenizer 内容不一致。

### 2.1 开源模型获取

本项目不提交或随源码分发模型权重。使用者应从公开模型仓库下载 BF16 检查点，并自行确认、接受
模型仓库标注的许可证及使用条件：

```text
https://huggingface.co/lerobot/pi05_libero_finetuned_v044
```

可使用 Hugging Face CLI 下载到使用者自行选择的位置：

```bash
export PI05_FP_MODEL_DIR="<本地 BF16 模型目录>"
hf download lerobot/pi05_libero_finetuned_v044 \
  --local-dir "${PI05_FP_MODEL_DIR}"
```

下载目录应至少包含 `config.json`、单个 `model.safetensors`、preprocessor、postprocessor
以及 tokenizer 配套文件。公开仓库若更新文件，建议固定 revision，并保存下载文件摘要，保证测试可复现。

### 2.2 Selective-99 INT8 量化与打包

量化采用两阶段流程。第一阶段从 BF16 权重生成每输出通道对称 RTN INT8 中间产物；第二阶段把量化张量
与未量化 BF16 张量、处理器和 tokenizer 合并成完整原生 Torch 检查点：

```bash
export IBROBOT_CONDA_ENV=IB_Robot_ros
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

该方案离线选择 99 个 `Linear` 层：27 个视觉 MLP `fc2`、18 个前缀语言模型
`self_attn.o_proj`、54 个前缀语言模型 MLP 投影。运行时完成 `gate/up` 融合后，对应 81 个实际启用的
INT8 投影。权重使用每输出通道对称 RTN INT8，激活在运行时执行逐 token 动态 INT8 量化；动作专家/DiT
保持 BF16。量化配置为 W8A8、`smooth=false`、`group_size=0`，不是 ONNX/OM 或 SmoothQuant 流程。

最终部署目录还需包含 schema-v3 `inference_manifest.json`，其中 deployment 使用 `torch-npu`，模型接口为
`policy/pi05/predict`。本次时延数据使用 `denoise_solver=ab2`、`num_inference_steps=6`；如果修改生成后的
`config.json`，必须同步重新生成或校验 manifest digest，不能继续使用与文件内容不一致的摘要。

纯模型同步时延测试可使用：

```bash
export IBROBOT_CONDA_ENV=IB_Robot_ros
source .shrc_local && python3 scripts/npu/pi05_910b/benchmark_pi05.py \
  --bundle "${PI05_INT8_MODEL_DIR}" \
  --deployment torch-npu \
  --warmup 3 \
  --iterations 50 \
  --output "<测试结果 JSON>"
```

## 3. 推理调用链与顺序时延

一次 ROS 客户端请求按以下顺序执行：

1. 客户端调用 `send_goal_async`，ROS Action/DDS 将 goal 发送到服务端。
2. 服务端接收 goal，完成请求准入、观测缓存采样、消息解码、状态转换和 NumPy→Torch 转换。
3. 服务端进入 `manager.infer`，执行统一框架 stage 和 PI0.5 NPU 推理。
4. 服务端把动作转换为连续 `float32`，编码为 TensorMsg，并发布动作消息、完成 goal。
5. ROS Action/DDS 将 result 返回客户端，客户端 executor 唤醒并完成 future。
6. 客户端停止端到端计时，随后才解码 action chunk 并检查 shape/有限值。

### 3.1 客户端端到端分解

数据来自本次专项测试记录中的客户端计时和服务端 monotonic 时间戳。服务端与客户端运行在同一主机，
因此可使用同一 monotonic clock domain 做差；原始测试文件不作为项目运行依赖。

| 顺序 | 阶段 | 平均时延 (ms) | 说明 |
|---:|---|---:|---|
| 1/5 | ROS Action/DDS 两端合计 | 6.426 | goal 上行、goal acceptance、result 下行、序列化/反序列化、executor 调度和服务端响应尾部；现有埋点不能继续拆分上下行 |
| 2 | 服务端请求准入与观测处理 | 1.110 | operation lock、观测采样/解码、状态转换、NumPy→Torch |
| 3a | manager 外层封装 | 0.556 | `manager.infer` 外层与内部 `PipelineResult.total_latency_ms` 的差值 |
| 3b | 统一框架内部推理 | 53.305 | 服务端返回的 `result.total_latency_ms` |
| 4 | 动作转换与结果构建 | 0.463 | 动作 CPU 转换、TensorMsg 编码、动作发布、goal 完成和响应字段构建 |
|  | **客户端总计** | **61.859** | 从调用 `send_goal_async` 前到收到 action result |

计算关系：

```text
61.859 = 6.426 + 1.110 + 0.556 + 53.305 + 0.463 ms
```

ROS benchmark 中原有的 `ros_overhead_ms` 定义为：

```text
client_wall_ms - server_inference_ms = 61.859 - 53.305 = 8.554 ms
```

该值同时包含 2.128 ms 的服务端非框架处理和 6.426 ms 的 ROS Action/DDS/调度残差，不能直接标注为
“网络耗时”。

### 3.2 客户端时延分布

| 指标 | mean (ms) | median (ms) | P90 (ms) | P95 (ms) | min (ms) | max (ms) | std (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 客户端总时延 | 61.859 | 61.601 | 66.259 | 66.625 | 56.909 | 66.800 | 3.668 |
| 服务端框架推理 | 53.305 | 53.216 | 53.810 | 54.726 | 52.197 | 55.132 | 0.601 |
| ROS overhead（混合口径） | 8.554 | 7.266 | 12.917 | 13.470 | 4.530 | 14.189 | 3.618 |
| ROS Action/DDS/调度残差 | 6.426 | 4.790 | 11.027 | — | 2.679 | 11.836 | 3.674 |

服务端推理相对稳定，客户端 P90 上升主要来自 ROS Action/DDS 和 executor 调度残差。要继续拆分该
6.426 ms，需要增加客户端 goal 发送、goal accepted、result future 完成以及服务端 response return 等
时间戳。

## 4. 各推理口径对比

| 测试口径 | mean (ms) | median (ms) | P90 (ms) | 说明 |
|---|---:|---:|---:|---|
| 官方 E2E，token=183 | 48.492 | 48.817 | 48.944 | 历史约 48 ms 基线 |
| 官方 E2E，token=200 | 49.052 | 48.929 | 49.767 | 与当前 bundle 的 token length 对齐 |
| 官方 E2E，token=200，策略内部生成 noise | 49.003 | 48.928 | 49.027 | noise 生成不是时延上升原因 |
| 官方 E2E，token=200，不替换 FastGELU | 48.976 | 48.906 | 49.022 | FastGELU 差异不是时延上升原因 |
| 官方代码 + `IB_Robot_ros` Conda | 48.892 | 48.928 | 49.800 | Conda 环境不是时延上升原因 |
| IB_Robot 同实例 direct E2E | 49.043 | 48.954 | 49.742 | 完整 preprocessor → policy → postprocessor |
| IB_Robot 同实例 manager E2E | 52.895 | 52.842 | 52.879 | 相比 direct 增加 3.852 ms |
| IB_Robot 独立 framework benchmark | 53.248 | 53.608 | 53.676 | 包含统一框架完整调用与外层同步 |
| ROS 服务端框架推理 | 53.305 | 53.216 | 53.810 | ROS 服务端返回的内部框架时延 |
| ROS 客户端端到端 | 61.859 | 61.601 | 66.259 | goal 发送至 result 到达 |

官方历史结果使用 token length 183，当前 bundle 使用 200。单变量对照显示，183→200 增加约
0.560 ms。因此比较“官方 48 ms”和“IB_Robot 53 ms”时，既包含输入长度差异，也包含统一框架开销。

## 5. 统一框架内部时延

独立 framework benchmark 的 50 次统计按执行顺序如下：

| 顺序 | 阶段/指标 | mean (ms) | 解释 |
|---:|---|---:|---|
| 1 | `preprocess_ms` | 0.935 | preprocessor 的 Python/异步下发阶段 |
| 2 | `backend_ms` | 3.176 | NPU 异步图下发和 Python 侧处理，不是模型真实计算时延 |
| 3 | backend 输出校验 | 未单独埋点 | 第一次 `detach().cpu().numpy()`，会等待此前 NPU 工作完成 |
| 4 | `postprocess_ms` | 0.112 | postprocessor 的 Python/异步下发阶段 |
| 5 | postprocess 输出校验 | 未单独埋点 | 第二次 `detach().cpu().numpy()`；两次校验的联合墙钟增量实测为 2.914 ms |
| 6 | `framework_total_ms` | 52.645 | 从框架 preprocess 开始到 completion 的累计内部时延 |
| 7 | 框架外层调用 | 0.543 | `runtime_wall_ms - framework_total_ms` |
| 8 | `output_materialize_ms` | 0.060 | 外层同步完成后的结果 materialize |
|  | `runtime_wall_ms` | 53.187 | `manager.infer` 加调用前后 NPU 同步 |
|  | **`wall_ms`** | **53.248** | 完整 framework benchmark 外层时延 |

`preprocess_ms + backend_ms + postprocess_ms` 不能与 `framework_total_ms` 直接相加比较。NPU 是异步执行，
`backend_ms` 在设备完成计算前已经结束；真正等待 NPU 完成的时间落在后续同步和 CPU 输出校验中。因此
不得用约 3 ms 的 `backend_ms` 宣称 PI0.5 模型推理只需要 3 ms。

## 6. 双重输出校验的影响

统一框架当前在两个位置调用 `validate_action_output()`：

1. backend decode 后检查模型原始动作。
2. postprocess 后检查最终动作。

校验函数会对 NPU Tensor 执行 `detach()`、`cpu()`、`numpy()`，再检查 shape、chunk size、action
dimension 和 `NaN/Inf`。这会在链路中间引入两次 NPU→CPU materialization 和设备同步。

同一进程、同一模型实例、strict/no-validation 逐次交错消融结果：

| 模式 | mean (ms) | median (ms) | P90 (ms) |
|---|---:|---:|---:|
| 保留两次严格校验 | 53.217 | 53.285 | 54.291 |
| 临时禁用两次校验 | 50.303 | 50.301 | 50.358 |
| **校验增量** | **2.914** | — | — |

完全禁用校验只用于诊断，不能直接作为生产修改。否则会失去以下故障检测：

- 输出 rank/shape 和 action dimension 检查；
- `actual_chunk_size` 一致性检查；
- 模型原始输出和最终输出的 `NaN/Inf` 检查；
- postprocessor 异常的及时发现。

推荐保持 Tensor 元数据校验常驻，避免为了 shape/rank 检查执行 D2H；有限值检查可根据安全要求改为一次
终态检查、按帧抽样或严格诊断模式。真实机器人控制入口仍应保留动作范围和有限值安全兜底。

## 7. ROS 计时范围说明

本报告所用历史客户端 benchmark 的计时起点位于每轮 `publish_observations()` 和一次 `spin_once()` 之后，终点位于
`get_result_async()` future 完成之后。因此：

计入客户端 61.859 ms 的内容：

- goal 发送、接受和 result 返回；
- 服务端观测缓存采样与解码；
- 统一框架和 NPU 推理；
- 服务端 action 转换、编码与发布；
- ROS Action/DDS 和 executor 调度。

未计入客户端 61.859 ms 的内容：

- 本轮观测图像和 state 的发布；
- 客户端收到 result 后的 `TensorMsgConverter.from_variant()`；
- 客户端 action 转 NumPy、shape 检查和 `np.isfinite()`。

若要测量“传感器发布到客户端可用动作”的完整闭环时延，需要把计时起点前移到观测发布前，并把终点后移到
客户端解码与校验完成后。

## 8. 优化优先级

1. 优化两次输出校验，避免重复 D2H，同进程消融收益为 2.914 ms。
2. 在 ROS 客户端和服务端补齐分段时间戳，拆分 6.426 ms 的 goal 上行、goal acceptance、result 下行和
   executor 调度。
3. 根据安全要求决定最终有限值检查采用每帧、抽样还是诊断模式，不能为降低时延直接删除全部校验。
4. token length 是否从 200 调整为 183 必须依据训练配置和任务文本长度决定，不能只依据时延修改。
5. 常驻 benchmark 收敛为一个脚本和一个 `model_inference_ms` 指标；framework、server、client 等细分口径
   仅在专项端到端诊断时临时启用，避免把框架或 ROS 开销归因于模型。

若仅按消融结果去除重复同步，而其他 ROS 开销保持不变，客户端均值理论上可由约 61.86 ms 降至约
58.94 ms；这是基于独立增量的估算，不是修改后的实测结果。

## 9. 测试产物与代码位置

专项诊断的原始结果属于测试环境产物，不随源码仓库发布，也不在文档中绑定服务器目录。使用者可通过
`benchmark_pi05.py --output <测试结果 JSON>` 将结果写入自行选择的位置。当前常驻脚本只保留 NPU
同步包围的模型 callback 时延，不再生成 preprocess、postprocess、framework 或 ROS 分段指标；本文前述
框架和 ROS 分段数据用于解释本次验证结论。

主要代码位置：

```text
scripts/npu/pi05_910b/benchmark_pi05.py
src/inference_service/inference_service/pipeline/runtime.py
src/inference_service/inference_service/pipeline/validation.py
src/inference_service/inference_service/pipeline_policy_node.py
src/inference_service/inference_service/model_sessions/lerobot_torch.py
src/torch_models/torch_models/pi05_ascend_910b/
```
