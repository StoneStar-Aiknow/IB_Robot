# inference_service

`inference_service` 是 IB-Robot 的统一推理运行时。它以一个模型 bundle 和一个命名
deployment 为输入，统一承载 Torch、Ascend、Hisilicon、RKNN 与 HMM 后端。策略模型额外使用稳定的
pipeline ID，并支持单体和边云分布式执行。

运行时从 bundle 中唯一的 `inference_manifest.json` 读取模型类型、语义 tensor、命名 deployment、artifact、
执行顺序和 runtime ABI bindings。Launch 和 robot YAML 通过 deployment 名称选择运行配置。

非 policy bundle 不需要 LeRobot metadata 或 `action` 输出。公共本地执行通过
`ModelRequest`、`ExecutionContext`、`ModelRuntimeFactory` 和 `ModelRuntimeHandle` 进入，成功结果统一为
`ModelResult`；`NamedTensorRequest` 只作为尚未完成迁移的 session 内部适配值保留。
生命周期、准入、健康状态、deployment fingerprint 与资源回收由 handle 负责。
模型家族的预处理、后处理和 ROS service 形状不属于 backend runtime，由调用方 adapter/plugin 持有。

manifest fingerprint 是经过验证的 bundle 结构身份，deployment fingerprint 标识所选运行部署。常规 loader
不会为了生成诊断身份扫描大型权重文件；启用推理 scheduler 时，本地 compiled artifact 还必须声明内容
`sha256`，`robot_config` 会在构造 scheduled launch 前流式校验文件内容。

## 核心概念

### Bundle

每个可部署目录必须包含唯一的 `inference_manifest.json`。Policy bundle 还必须包含 LeRobot 语义文件：

```text
policy_bundle/
├── config.json
├── model.safetensors                         # Torch deployment 需要时存在
├── policy_preprocessor.json
├── policy_preprocessor_step_*.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_*.safetensors
├── tokenizer/                                # PI0.5 / SmolVLA 按需存在
├── artifacts/
│   ├── ascend/<deployment>/...
│   ├── hisilicon/<deployment>/...
│   ├── rknn/<deployment>/...
│   └── hmm/<deployment>/...
└── inference_manifest.json
```

LeRobot 拥有 `config.json`、processor JSON、processor state、tokenizer 和原生权重。
IB-Robot 只读这些文件，不会添加字段、删字段、重写 device 或创建临时策略目录。
部署信息只属于 `inference_manifest.json`。

### Deployment

一个 manifest 可以为同一策略声明多个命名 deployment，例如 `cpu`、`cuda`、
`rk3588`、`ascend_310p3` 或 `lq50`。Pipeline 选择的是 deployment 名称，而不是后端名。

部署通过显式的 v3 runtime profile 声明后端、target 和运行实例字段：

```json
{
  "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
  "revision": 1,
  "execution_contract": {
    "state_scope": "request",
    "execution_structure": "direct",
    "cancellation_granularity": "request_boundary"
  },
  "runtime_profile": {
    "backend": "torch",
    "target": {"runtime": "torch"},
    "profile": {"device": "cpu"}
  }
}
```

编译 deployment 声明 target、artifact、执行顺序和完整 runtime ABI bindings：

```json
{
  "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
  "revision": 3,
  "execution_contract": {
    "state_scope": "request",
    "execution_structure": "direct",
    "cancellation_granularity": "request_boundary"
  },
  "runtime_profile": {
    "backend": "rknn",
    "target": {
      "soc": "rk3588",
      "runtime": "rknn-lite2"
    },
    "profile": {
      "target_name": "rk3588",
      "core_mask": 7,
      "device_id": 0
    }
  },
  "artifacts": {
    "policy": {
      "path": "artifacts/rknn/rk3588/generations/<uuid>/policy.rknn",
      "format": "rknn"
    }
  },
  "execution": ["policy"],
  "bindings": {
    "policy": {
      "inputs": [
        {
          "semantic": "observation.state",
          "runtime_name": "observation.state",
          "index": 0,
          "dtype": "float32",
          "shape": [1, 6]
        },
        {
          "semantic": "observation.images.top",
          "runtime_name": "observation.images.top",
          "index": 1,
          "dtype": "float32",
          "shape": [1, 480, 640, 3],
          "layout": "NHWC"
        }
      ],
      "outputs": [
        {
          "semantic": "action",
          "runtime_name": "action",
          "index": 0,
          "dtype": "float32",
          "shape": [1, 100, 6]
        }
      ]
    }
  }
}
```

`execution` 中的每个角色都必须同时存在 artifact 和非空 binding group。图像 binding
必须显式声明 `NCHW` 或 `NHWC`；非图像 tensor 不得仅根据 rank 自动转置。多模块模型
使用匹配的 `internal.*` semantic，或使用 `device_links` 声明 producer、consumer、
device-pointer ownership 和 inference lifetime。

### Pipeline

Pipeline ID 是模型实例和 ROS 路由的稳定标识，必须匹配
`^[a-z][a-z0-9_]{0,62}$`。每个 pipeline 独立拥有：

- 策略 bundle 和命名 deployment
- LeRobot preprocessor / postprocessor
- policy codec 与 binding execution plan
- 后端实例、准入状态和生命周期
- Action、reset、health、action output 和分布式 transport endpoints

`GenericModelPipeline` 是策略、感知、Echo 等模型共用的公开运行时入口，统一生命周期、准入、
deadline、cancellation 和 health。`PipelineRuntimeCore` 是其内部可复用的状态机与并发控制实现。
`InferencePipeline` 是策略模型 facade，在通用运行时之上增加 LeRobot processor、policy codec 和
action 结果适配。编译模型由 `SequentialModelExecutor` 按 `InferenceStage` 序列执行；循环 family
通过 `IterativeStage` 驱动多个 role，并在一个 `ModelSessionExecution` scope 中共享设备资源。

默认 endpoint：

| 接口 | 默认值 |
| --- | --- |
| 本地节点 | `inference_<pipeline_id>` |
| 云端节点 | `inference_<pipeline_id>_cloud` |
| Action server | `/inference/<pipeline_id>/dispatch` |
| Reset service | `/inference/<pipeline_id>/reset` |
| Health topic | `/inference/<pipeline_id>/health` |
| Action output | `/actions/<pipeline_id>` |
| 分布式 request | `/inference/<pipeline_id>/request` |
| 分布式 result | `/inference/<pipeline_id>/result` |
| 分布式 heartbeat | `/inference/<pipeline_id>/heartbeat` |
| 视频 descriptor | `/inference/<pipeline_id>/video/descriptors` |
| 视频 status | `/inference/<pipeline_id>/video/status` |

## 推理调度控制面

唯一开关是 `control_modes.<mode>.inference.scheduler.enable`，默认 `false`。

- 开关缺失或为 `false`：保持 legacy 路径
  `executor.inference_pipeline -> pipeline /dispatch + /reset`。Launch 只创建
  `pipeline_policy_node` 和 `action_dispatcher_node`；pipeline 不注册 scheduled action server，
  不发布 serving status，也不创建 product session 或 Global Scheduler。
- `scheduler` 块存在且 `enable: false` 时，可以原样保留完整调度配置；这些字段处于 dormant 状态，不生成
  runtime policy，也不改变 legacy 节点参数、接口、线程数或 backend 执行方式。因此启停调度只需修改
  `scheduler.enable`。若整个 `scheduler` 块缺失，scheduled 字段仍按 unknown field 拒绝，以保留拼写检查。
- 开关为 `true`：Launch 创建 pipeline、`global_inference_scheduler_node` 和
  `scheduled_action_dispatcher_node`。产品调用只经过 Global 的 Open、Dispatch、Close endpoint，
  Global 负责逻辑 session、多 pipeline binding、逐请求路由、deadline precheck 和 quarantine；pipeline 负责
  generation fence、实际 admission、执行和 drain。Launch 只为 `required: true` pipeline 注册
  进程退出后的全局 Shutdown；optional pipeline 退出后由 readiness 和路由逻辑将其排除。该路径当前只接受
  `execution_mode: monolithic`。

生产执行路径只支持 schema v3 whole-graph plan。公开 Open 只建立 route-independent 的逻辑 session 和
logical generation，不选择模型、不检查 fallback，也不访问 pipeline。每个 Dispatch 都携带自己的 target
pipeline、fallback chain、priority 和 deadline；候选首次被该 session 选中时，Global 才向对应 pipeline
下发私有 Open/reset 并记录其 pipeline generation。同一个逻辑 session 可按需建立多个 pipeline binding。
Close 停止新 admission，并逐一 drain/reset 实际使用过的全部 binding；从未 Dispatch 的 session 可直接关闭。

priority-0 请求按 `[target, *fallback_chain]` 顺序逐个检查。`profile_path` 可选，但实际参与本次准入的候选
必须有匹配身份、覆盖当前输入契约和 prompt 大小、未过期的离线测量：
已绑定候选使用 `full_infer` closure；尚未绑定的候选还必须包含 `session_open` closure，确保 Open/reset 与推理
其 p99 admission SLA 估计加 safety margin 不超过该请求的绝对 deadline。同一个 `hardware_resource_id` 上已准入的 priority-0 工作按同优先级
FIFO 串行下发：reservation 未成为资源队首前不会 Open/Dispatch；真正取得下发权时会按当前剩余时间再次检查
deadline。已知未开始或已知完成时释放，执行结果不确定时保持隔离直到对应 pipeline 重启。
这不是 pipeline 数量上限，也不限制 priority 大于 0 的请求。缺失或无效 profile 的候选按不可准入处理并继续
fallback；没有候选可完成时返回
`no_feasible_deadline/NOT_STARTED`，不会下发；
`NOT_STARTED` 才继续下一个 fallback，`UNKNOWN` 立即 quarantine，绝不 fallback。Global ingress 仍保持有界：
Open/Close 各两个执行 context；Dispatch 共四个 context，其中 lower-priority 最多占两个，不能耗尽为 priority-0
保留的进入能力。priority-0 可使用任意空闲 Dispatch context。

profile 文件顶层只包含 `closure_profiles`。Global 消费的 entry scope 为 `global_proxy`。action-generation entry
必须携带 pipeline 发布的 `input_contract_fingerprint` 和标定覆盖的 `prompt_bytes_max`；session-control entry 使用
空 input fingerprint 和 `prompt_bytes_max: 0`。每个 entry 还必须携带匹配的 deployment、hardware、
`profile_compatibility_fingerprint`、work class、closure key、priority、采样时间、样本数、goal acceptance p99.9
和 closure latency p99。Global 选择能够覆盖当前 prompt 的最小 profile bucket；超出覆盖范围时 fail closed。
这些统计量定义 p99 admission SLA，不是绝对完成保证；最终估计仍叠加配置的 safety margin。

priority 大于 0 的请求只路由到 target，不查询 profile，也不尝试 fallback；调用方 deadline 不参与业务准入。
Global 会生成独立的内部 request timeout 并传给 pipeline，用于约束 RPC、取消和故障恢复，避免调用永久挂起。

通用 priority 通过 scheduled request metadata 传给 monolithic backend；`0` 为最高优先级，数值越大优先级越低。
通用 wire 取值为非负 int32，不施加 backend-specific 上限。只有支持多优先级的 backend 才暴露显式的
generic-to-native priority mapping capability；没有该 capability 的 backend 为单优先级，只接受 priority 0。
当前只有 Ascend 暴露多优先级 mapping。
AscendBackend 在 scheduled 模式加载时查询 `acl.rt.device_get_stream_priority_range()`，并通过
`acl.rt.create_stream_with_config()` 为硬件支持的每一级优先级创建可复用 stream。Ascend 接受 `[0, 7]`，并将
通用 priority 一一映射到同编号 ACL priority；大于 `7` 的请求在模型执行前直接拒绝。模型通过
`acl.mdl.execute_async()` 下发，并在读取输出或复用 dataset/buffer 前调用
`acl.rt.synchronize_stream()`；同一 backend 实例仍保持单 in-flight。调度模式不设置进程内跨 context 数量上限，
从而允许 ACL 在不同 context 的 stream 之间执行硬件优先级抢占；实际并发能力由 Ascend runtime 和硬件决定。
disabled/legacy 模式不创建 stream，继续调用同步 `acl.mdl.execute()`。
pipeline 的 `action_generation.max_in_flight` 不得超过 backend 声明的 `max_in_flight_per_instance`；没有可用
execution slot 时立即返回 `pipeline_busy/NOT_STARTED`，不在 pipeline 内排队。该容量必须是正整数；pipeline
Dispatch action server 的有界 ingress 容量至少保留两个 context 以处理重复请求，并随更高执行容量扩展。
serving status 仅在 session state 允许且
`current_in_flight < max_in_flight` 时报告 `accepting_requests=true`。后端报告的
serving status 中的 `hardware_priority_levels` 是该 capability 的只读观测值；请求 priority 超出 backend mapping
声明范围时，pipeline 在模型执行前返回不可恢复的 `unsupported_priority/NOT_STARTED`。Global readiness 至少要求
一个通用 priority level；当 launch 配置的默认 priority 大于 0 时，只额外检查默认 target pipeline 在线且支持该
priority，不要求 fallback 或其他 pipeline 支持多优先级。

`hardware_resource_id` 是 backend 实际运行设备的身份，用于 serving readiness 和 Global deadline reservation；
`resource_domain` 只用于进程内并发准入。两者独立：Ascend priority mode 不设置进程内 resource domain gate，
但仍上报 `ascend:<device_id>` 作为硬件身份。
`hardware_profile_fingerprint` 则是离线标定环境的 SHA-256 身份，必须覆盖 SoC、CANN/驱动/固件版本、频率/功耗
模式和其他会影响时延的配置。profile entry 使用独立的 `hardware_fingerprint` 与其匹配，不能再用 `ascend:0`
这类资源互斥 ID 代替标定环境身份。

分布式推理继续使用原 protocol v2 和原 topic，不参与 scheduler、product session 或优先级抢占。
`scheduler.enable=true` 与 `execution_mode: distributed` 的组合会在 `robot_config` 校验阶段被拒绝。

## Robot 配置

推理直接配置在 `control_modes.<mode>.inference.pipelines` 下：

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      pipelines:
        policy:
          model_path: models/so101_act
          deployment: rk3588
          execution_mode: monolithic
          request_timeout: 5.0
          default_task: pick up the banana
          runtime_options: {}
```

相对 `model_path` 只相对于环境变量 `WORKSPACE` 解析。`WORKSPACE` 未设置时会失败，
不会回退到当前目录、YAML 目录或源码目录。执行项目命令前应先运行：

```bash
source .shrc_local
```

多模型通过多个独立 pipeline 配置：

```yaml
pipelines:
  action_policy:
    model_path: models/so101_act
    deployment: rk3588
    execution_mode: monolithic
  auxiliary_policy:
    model_path: models/auxiliary_smolvla
    deployment: cpu
    execution_mode: monolithic
```

YAML 是默认配置来源。开发调试时可只覆盖一个明确指定的 pipeline：

```bash
ros2 launch robot_config robot.launch.py \
    config_path:=/absolute/path/to/robot.yaml \
    control_mode:=model_inference \
    inference_pipeline:=policy \
    inference_execution_mode:=distributed
```

`inference_execution_mode` 为空时完全使用 YAML；非空时必须同时提供
`inference_pipeline`，避免多 pipeline 配置被全局误覆盖。

可以在每个 pipeline 的 `transport` mapping 中覆盖 endpoint。不同 pipeline 的节点名、
Action、service 和 topic 必须唯一。Monolithic pipeline 不允许配置 cloud node、request、
result 或 heartbeat overrides。

## 执行模式

### Monolithic

`pipeline_policy_node` 在一个进程内执行：

```text
ROS observations
  -> contract adapter
  -> LeRobot preprocessor
  -> semantic batch
  -> native policy, or policy codec + shared stage executor
  -> Backend.infer, or ModelSession role execution
  -> semantic action
  -> LeRobot postprocessor
  -> DispatchInfer result and /actions/<pipeline_id>
```

进入 processor 前，节点会对策略 `input_features` 所需的每个 observation 执行 readiness
检查：缓存中必须已有时间戳不晚于请求时间、且满足 `align.strategy`（`hold` / `asof` / `drop`）
的样本；当 contract 为该 observation 配置了
`max_age_ms > 0` 时，该值作为在线推理额外的最大样本年龄，与 `asof` 对齐使用的 `tol_ms` 分离。
在线年龄使用节点不可被请求方回拨的本地接收时钟计算，request timestamp 仅用于历史样本对齐选择。
缺失、晚于请求时间或过期的样本会返回可恢复的 `observation_not_ready`，不会静默补零后执行模型。调用 pipeline
reset service 会重置 policy、LeRobot preprocessor/postprocessor 并清空 observation 缓存，下一次推理
必须等待新 episode 的输入。推理、reset 和分布式 cancel 都使用 pipeline `request_timeout`
作为协作式 deadline；锁和 admission 等待会按时退出，backend/processor hook 超时会在其返回时被检测。
reset 或 cancel 结果不确定时会使 edge fail closed，避免 cloud 与 edge episode 状态不一致。

完整机器人启动通常由 `robot_config` launch builder 根据 YAML 创建 pipeline。仅评估一个
pipeline 时可直接使用：

```bash
source .shrc_local
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:="$WORKSPACE/src/robot_config/config/robots/so101_single_arm.yaml" \
    model_path:="$WORKSPACE/models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515" \
    deployment:=cpu \
    pipeline_id:=policy \
    action_server:=/inference/policy/dispatch \
    reset_service:=/inference/policy/reset
```

触发一次推理：

```bash
ros2 action send_goal /inference/policy/dispatch \
    ibrobot_msgs/action/DispatchInfer \
    "{obs_timestamp: {sec: 0, nanosec: 0}, prompt: '', inference_id: 'test-001'}"
```

### Distributed

分布式 pipeline 在 edge 的 `pipeline_policy_node` 中保留 observation 采样、机器人状态单位换算、
非图像 tensor 序列化、action `TemporalSmoother` 和最终机器人单位换算。Cloud 的
`pure_inference_node` 组装完整 raw observation，在同一进程中依次运行 LeRobot preprocessor、
selected backend 和 postprocessor，避免 processor state 跨机器分裂。两端在接受请求前必须匹配：

```text
Robot / Edge host                              Compute / Cloud host
┌──────────────────────────────────┐           ┌──────────────────────────────────┐
│ action_dispatch                  │           │ inference_<pipeline_id>_cloud    │
│       │ DispatchInfer            │           │ pure_inference_node              │
│       v                          │           │                                  │
│ inference_<pipeline_id>          │           │ selected deployment and backend  │
│ pipeline_policy_node             │           │                                  │
│ ├─ observation sampling          │           │ ├─ raw observation assembly      │
│ ├─ state/action unit conversion  │           │ ├─ LeRobot pre/postprocessors    │
│ └─ action TemporalSmoother       │           │ └─ selected deployment/backend   │
│       │ request      ▲ result     │           │       ▲ request      │ result    │
└───────┼──────────────┼────────────┘           └───────┼──────────────┼───────────┘
        │              │                                │              │
        └──────────────┴──────── ROS 2 transport ───────┴──────────────┘
          /inference/<pipeline_id>/request
          /inference/<pipeline_id>/result
          /inference/<pipeline_id>/heartbeat
```

每个 pipeline 使用独立的 request、result 和 heartbeat endpoint；两端使用相同的 pipeline
ID、bundle 和 deployment 身份信息。

图像 observation 可继续使用显式 `mode: dds`，或使用 H.264 RTP/UDP 数据面。RTP 模式下，DDS
只承载 descriptor、status、timestamp mapping、request/result 和 heartbeat；H.264 payload 不进入
`VariantsList`。每路相机使用唯一 stream ID、SSRC 和偶数 UDP port，`port + 1` 保留用于端点冲突检查。
Cloud 必须收到匹配 protocol/session generation、contract fingerprint 和 deployment fingerprint 的
全部 descriptor，并等到 keyframe 与 RTP-to-capture timestamp mapping 就绪后才接受请求。

`encoder_backend` 可设为 `software`、`nvidia`、`ascend` 或 `auto`；`nvidia` 当前仅提供 NVENC 编码，
不能用作 decoder。Software backend 使用 PyAV 15 和其 FFmpeg `libx264`/H.264 decoder；启动 probe
缺少 codec 时会 fail closed。NVIDIA backend 会实际打开一个 `h264_nvenc` session，使用 ultra-low-latency、
zero-delay、无 B-frame 和重复 SPS/PPS 的 H.264 配置；RGB/BGR 到 NV12 的转换仍由 FFmpeg 完成，不是
CUDA zero-copy。Ascend backend
仅在选择或自动探测时查找私有 FFmpeg `h264_ascend`，可用
`IBROBOT_ASCEND_FFMPEG` 或 `IBROBOT_ASCEND_FFMPEG_PREFIX` 指定；标准 RPM 安装的
`/usr/bin/ffmpeg-ascend` 也会自动探测，不替换系统 FFmpeg，也不引入
ACL/DVPP Python 依赖。启动日志（"Video stream startup" 行）报告 configured/selected
backend、endpoint、fingerprint、lifecycle 和 readiness。

RPM 安装的 ffmpeg-ascend（`/usr/bin/ffmpeg-ascend`、`/usr/local/bin/ffmpeg-ascend`
及其 dispatch 的 `/usr/local/ffmpeg-ascend-*/bin/ffmpeg` payload）默认以隔离环境启动：
子进程会剔除继承自其他 CANN toolkit 的安装路径变量
（`ASCEND_TOOLKIT_HOME`、`ASCEND_HOME_PATH`、`ASCEND_AICPU_PATH`、`ASCEND_OPP_PATH`、
`ASCEND_NNRT_HOME`、`ASCEND_NNAE_HOME`、`TOOLCHAIN_HOME`），但保留设备运行时变量
（如 `ASCEND_RT_VISIBLE_DEVICES`、`ASCEND_DEVICE_ID`），以保留多 NPU 配置。私有构建默认不隔离，
可用 `IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV=1` 强制开启、`=0` 关闭 RPM 默认隔离；
不带隔离探测 RPM payload 已知会在首帧挂起。

Ascend DVPP VENC 通道是 per-device 硬件资源，`DeviceVideoStreamManager` 按
观察值 key 排序对 Ascend 流紧凑编号 1..N（≤128），`device_id` 当前固定为 0
（单 NPU 场景）；多 NPU 需在 pipeline/资源层引入 device 分配契约，不在
manager 硬编码。

`auto` 按 `ascend`、`nvidia`、`software` 顺序探测。310B 优先使用 Ascend，本机 NVIDIA GPU 可用时
自动选择 NVENC，其他 Linux 主机回退 software；显式 backend 失败时不会回退。

RTP/UDP 不提供认证、加密或完整性保护，只允许可信机器人网络。链路中断、descriptor 不一致、显式
backend 不可用、时间映射过期或跨相机 skew 超限都会拒绝推理，不会自动回退 DDS。回滚必须在两端
部署匹配的 `mode: dds` contract。当前 rosbag/MCAP 录制仍采集 DDS image topic；RTP-aware recording
和不可信网络安全属于后续工作。

- pipeline ID
- manifest schema version
- bundle digest
- deployment name
- selected deployment fingerprint
- policy input/output summary
- cloud backend `READY` state

Cloud 示例：

```bash
source .shrc_local
ros2 launch inference_service cloud_inference.launch.py \
    pipeline_id:=policy \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda
```

单独调试 distributed Edge 时也可直接使用：

```bash
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:=/absolute/path/to/robot.yaml \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda \
    pipeline_id:=policy \
    inference_execution_mode:=distributed
```

本机同时启动 Edge 和 Cloud：

```bash
ros2 launch inference_service local_distributed_inference.launch.py \
    robot_config_path:=/absolute/path/to/robot.yaml \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda \
    pipeline_id:=policy
```

Edge 可由 robot YAML 中 `execution_mode: distributed` 创建，也可由上述显式 launch override
创建。成功握手后会绑定唯一 session ID 和 generation。Heartbeat 超时、cloud 重启、
fingerprint 改变或 backend 离开 `READY` 都会立即撤销 readiness、拒绝新请求，并使
in-flight request 返回结构化 unavailable 错误。不属于当前 session ID 和 generation 的
response 会被丢弃，连接恢复后必须重新握手。

重新握手不是 stateful backend 的完整恢复条件。替换已有 session 时，Cloud 会先停止旧
session 准入并等待其 runtime 操作结束；stateless backend 随后可直接创建新 generation，
stateful backend 则必须先成功 reset，并确认 backend 回到 `READY`，才能发布新 generation。
Reset 失败时 Cloud 会持续 fail-closed，后续 heartbeat 不能绕过该恢复屏障。若 stateful
backend 声明 `resettable: false`，session rollover 无法通过重新握手恢复，必须重启或重建
Cloud runtime 后才能重新提供服务。

## 后端与支持矩阵

Canonical backend 只有以下五个：

| Backend | 用途 |
| --- | --- |
| `torch` | 原生 LeRobot，device 为 `cpu`、`cuda`、`mps` 或 `npu` |
| `ascend` | Ascend ACL 执行 OM artifact |
| `hisilicon` | 面向 `sd3403` target SoC 的 Hisilicon worker runtime |
| `rknn` | RKNNLite 执行 RKNN artifact |
| `hmm` | Houmo TCIM 执行 HMM 多模块 artifact |

Ascend compiled deployment 可通过 manifest `device_links` 把 producer output buffer 直接绑定到 consumer input。
`AscendOmModelSession` 按 `execution` 顺序调度这些 role，只把公开输出以及未声明 device link 的 host-routed 中间
tensor 复制回主机；设备生命周期、buffer ownership 和串行准入仍由 shared runtime 统一管理。

编译 PI0.5 与 SmolVLA 不再由旧 backend 类持有 family 循环。它们通过
`GenericModelPipeline -> SequentialModelExecutor -> InferenceStage -> ModelSession` 执行；对应
`AscendBackend`、`HMMBackend` 和 `RKNNBackend` 对已迁移 family fail-closed。以下矩阵在启动时
强制校验，不在表中的组合会被拒绝：

| Policy family | `torch` | `ascend` | `hisilicon` | `rknn` | `hmm` |
| --- | --- | --- | --- | --- | --- |
| ACT | Backend | Backend | Backend | Backend | 不支持 |
| Diffusion Policy | Backend | 不支持 | 不支持 | 不支持 | 不支持 |
| PI0.5 | Backend | ModelSession | 不支持 | 不支持 | ModelSession |
| SmolVLA | Backend | 不支持 | 不支持 | ModelSession | ModelSession |

`Backend` 表示直接调用 `*Backend.infer()`；`ModelSession` 表示共享 family executor 按 role 调用
`*ModelSession`。感知 family 的 registry 支持矩阵如下：

| Perception family | `torch` | `ascend` |
| --- | --- | --- |
| RAM++ | 支持 | 支持 |
| SAM2 | 支持（automatic） | 支持（prompt） |
| SigLIP2 | 支持 | 支持 |
| Grounding DINO | 支持（combined） | 支持（raw） |
| GraspGen | 支持（CUDA） | 支持 |
| Dummy Echo | 支持 | 不支持 |

Registry 还要求每个 family/backend 声明 `ConformanceEvidence`；仅列出 family 而没有一致的验证证据
同样会 fail-closed。

### PI0.5 Ascend 行为

优化后的 PI0.5 VLM 在模型内部把多相机图像合并为一个临时 vision batch，并在 handoff 前恢复
camera-major prefix。该优化不改变外部 VLM ABI：runtime bindings、逐相机 observation semantic、
raw image shape 和 ROS camera topic 契约保持不变。

NPU 导出默认对 Gemma text MLP 使用精度保持的 `NPUGeglu`。显式参数
`--fast-gelu-scope vision|vlm-text|ae|all` 可将近似 `NPUFastGelu` 限制到指定模型区域；旧参数
`--fast-gelu` 等价于全局 `all`。近似路径可能降低动作精度，必须针对既有 baseline 验证。

新 Action Expert OM 的 runtime output 名为 `velocity` 或 `v_t`，Manifest 仍将其映射为策略
semantic `action`。Ascend backend 从选中 deployment 的 `denoising_schedule` artifact 读取严格
递减的 timesteps，并在 host 侧执行 `x_next = x_t + (next_t - t) * velocity` Euler integration，
最终才返回 action。Exporter 未提供 `--schedule-file` 时，根据 `config.num_inference_steps` 打包
uniform schedule；提供时只接受严格的 `pi05-denoising-schedule-v1` JSON。

`denoising_schedule` 是 versioned、non-execution artifact：不加入 `execution` 或 `bindings`，但
其 artifact path 和 deployment revision 属于 deployment identity，schedule 变化会改变 selected
deployment fingerprint。生产 Runtime 不扫描 bundle 根目录的 `schedule.json`，也不接受 schedule
override。`loss_compare`/tuner 通过隔离的 diagnostic backend factory 注入临时 schedule；
`curvature_log_path` 仅记录诊断数据，最终 schedule 必须安装回 Manifest。

兼容性由 Action Expert runtime output 明确决定。已有 output 为 `action` 且没有 schedule artifact
的 legacy PI0.5 deployment 保留旧的逐步 action-output 行为；velocity deployment 缺少 schedule
则拒绝加载，不会猜测默认值。`hardware_mock` 仍只验证 raw image/topic、joint 和 action 契约，
不需要 PI0.5 或 schedule 专用修改。

原生 Torch Diffusion Policy 按模型 `n_obs_steps` 在 contract 控制频率上采样历史观测，
`predict_action_chunk()` 返回的 nominal chunk 长度取自 `n_action_steps`。历史不足时使用每个
观测流的首帧进行左侧填充；不同频率的传感器仍按相同时间栅格应用各自的 `hold`、`asof`
或 `drop` 对齐策略。

可选 SDK 延迟导入。仅导入 `inference_service` 不要求 ACL、RKNNLite、TCIM、torch NPU 或
Hisilicon worker dependency；只有选择相应 deployment 时才检查依赖。

## 生命周期、健康与能力

Backend 状态包括 `CREATED`、`LOADING`、`READY`、`DEGRADED`、`RECOVERING`、
`FAILED`、`CLOSING` 和 `CLOSED`。只有 `READY` 接受请求。`close()` 必须幂等，部分加载
失败也必须释放已经创建的 context、model handle、device buffer 或 worker。

Pipeline 状态包括 `CREATED`、`LOADING`、`HANDSHAKING`、`READY`、`RESETTING`、
`DEGRADED`、`FAILED`、`CLOSING` 和 `CLOSED`。Reset 期间阻止新请求；`CLOSING` 和
`CLOSED` 是终态。

Backend capabilities 决定：

- 是否 stateful、resettable、thread-safe
- 单实例最大 in-flight 请求数
- 是否支持多个实例
- 是否存在共享 resource domain 及其总限制
- 是否支持 attention 和 cancellation

默认采用保守串行限制。只有 conformance tests 证明重叠调用、输出隔离、故障隔离和确定性
清理后，backend 才能声明更高并发。不同 pipeline 有独立准入状态，但共享 accelerator
resource domain 时仍可能被后端串行化。

## Manifest Identity

启动顺序是：严格 JSON/schema 校验，deployment 选择，UUID/revision 与轻量 bundle digest 校验，
路径安全和普通文件校验，LeRobot metadata，binding compatibility，最后创建 backend runtime。
常规 Runtime 不读取 OM、RKNN、HMM 或 safetensors 计算内容 SHA-256；scheduled 本地 compiled deployment
由 `robot_config` 在 launch 构造前额外验证 manifest 中声明的 artifact SHA-256。

`bundle.digest` 算法：

1. 将每个 `bundle.files` path 规范化、去重并排序。
2. 加入 bundle UUID、revision、name 和结构格式域。
3. 使用 canonical UTF-8 JSON 序列化这个小型声明。
4. 对声明 bytes 计算 SHA-256，不读取声明指向的文件。

UUID/revision/digest/fingerprint 用于版本标识和分布式一致性，不提供文件防篡改。正式 artifact 更新
必须经过 packager 并发布新 revision；生产防篡改应使用只读镜像、签名或 verity。

Selected deployment fingerprint 对以下 canonical object 计算 SHA-256：

```json
{
  "format": "ibrobot.deployment-structure-v3",
  "schema_version": 3,
  "bundle_digest": "...",
  "deployment_name": "rk3588",
  "deployment": {}
}
```

路径不能是绝对路径、不能包含 parent traversal、不能通过 symlink 逃逸 bundle root，规范化
后也不能重复。

### Identity 错误处理

遇到以下错误时不要手工改 identity：

- `Bundle digest mismatch`
- unsupported schema v1（必须重新导出或重新打包）
- missing or unexpected LeRobot semantic files
- execution role 缺少 artifact 或 bindings
- runtime ABI 与 LeRobot feature shape 不匹配

应重新运行拥有该 artifact 的 exporter 或 packaging workflow。Exporter 负责复制 artifact、
读取 compiler/runtime ABI、生成 bindings、更新 UUID/revision 和轻量结构摘要，并通过生产 loader
重新验证 manifest。Schema v1 和旧版 artifact 不受支持，必须使用当前 exporter 或 packager
重新生成完整 schema-v3 whole-graph bundle。

## Exporter 入口

通用 compiled artifact packaging：

```bash
ros2 run model_utils package-compiled-deployment \
    --bundle-root /path/to/policy_bundle \
    --deployment rk3588 \
    --backend rknn \
    --target-soc rk3588 \
    --target-runtime rknn-lite2 \
    --spec /path/to/compiler-package-spec.json
```

PI0.5 / SmolVLA HMM 使用：

```bash
ros2 run model_utils package-hmm-deployment --help
```

ACT Ascend、ACT RKNN、Hisilicon 和 policy-specific multi-module exporters 位于
`model_utils`。所有工具最终必须调用 shared `inference_manifest` writer；不要手工维护
artifact path、binding 或 digest。

## 验证

从源码运行测试时优先使用 source package path，并禁用外部 pytest plugins：

```bash
source .shrc_local
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src/inference_manifest:src/inference_service \
pytest -q src/inference_service/tests
```

只对本次修改的 Python 文件执行 Ruff。项目或 ROS 命令前必须先加载 `.shrc_local`。
## Typed Model Services

`model_service_node` is the family-neutral host for strongly typed model services. Each process loads one schema-v3
bundle, one named deployment, and one `ModelServicePlugin`; the plugin owns domain request/response mapping while the
shared `ModelSession` owns admission, health, accelerator lifetime, and cleanup.

`robot_config` launches the same host for RAM++, SigLIP2, GraspGen, and ZipVoice TTS. Model packages provide plugins;
they must not implement parallel ROS nodes or duplicate `ModelRuntimeInfo` projection.

When `required=false`, an initialization failure leaves the typed endpoint online so callers receive a not-ready
response and diagnostics. The host does not retry plugin initialization on later requests. After repairing a missing
bundle, dependency, or device, restart the node to recover. Request validation and other request-scoped failures do
not change runtime health; only plugin/session health determines whether the runtime is failed.
