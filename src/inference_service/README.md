# inference_service

`inference_service` 是 IB-Robot 的统一推理运行时。它以一个策略 bundle、一个命名
deployment 和一个稳定的 pipeline ID 为输入，统一承载 Torch、Ascend、Hisilicon、
RKNN 与 HMM 后端，并支持单体和边云分布式执行。

运行时从策略 bundle 中唯一的 `inference_manifest.json` 读取命名 deployment、artifact、
执行顺序和 runtime ABI bindings。Launch 和 robot YAML 通过 deployment 名称选择运行配置。

## 核心概念

### 策略 Bundle

每个可部署策略目录必须同时包含 LeRobot 语义文件和唯一的
`inference_manifest.json`：

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

Torch deployment 直接声明运行设备：

```json
{
  "backend": "torch",
  "device": "cpu"
}
```

编译 deployment 声明 target、artifact、执行顺序和完整 runtime ABI bindings：

```json
{
  "backend": "rknn",
  "target": {
    "soc": "rk3588",
    "runtime": "rknn-lite2"
  },
  "artifacts": {
    "policy": {
      "path": "artifacts/rknn/rk3588/policy.rknn",
      "format": "rknn",
      "sha256": "<64 lowercase hex>"
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
  -> native policy or policy codec + bindings
  -> selected deployment backend
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

分布式 pipeline 在 edge 的 `pipeline_policy_node` 中保留 observation adapter、processor 和
postprocessor，在 cloud 的 `pure_inference_node` 中加载 selected backend。两端在发送 tensor
前必须匹配：

```text
Robot / Edge host                              Compute / Cloud host
┌──────────────────────────────────┐           ┌──────────────────────────────────┐
│ action_dispatch                  │           │ inference_<pipeline_id>_cloud    │
│       │ DispatchInfer            │           │ pure_inference_node              │
│       v                          │           │                                  │
│ inference_<pipeline_id>          │           │ selected deployment and backend  │
│ pipeline_policy_node             │           │                                  │
│ ├─ observation contract adapter  │           │                                  │
│ ├─ LeRobot preprocessor          │           │                                  │
│ └─ LeRobot postprocessor         │           │                                  │
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

以下支持矩阵在启动时强制校验，不在表中的组合会被拒绝：

| Policy family | `torch` | `ascend` | `hisilicon` | `rknn` | `hmm` |
| --- | --- | --- | --- | --- | --- |
| ACT | 支持 | 支持 | 支持 | 支持 | 不支持 |
| PI0.5 | 支持 | 支持 | 不支持 | 不支持 | 支持 |
| SmolVLA | 支持 | 不支持 | 不支持 | 支持 | 支持 |

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

## Manifest 完整性

启动顺序是：严格 JSON/schema 校验，deployment 选择，路径安全校验，bundle 文件 SHA-256，
bundle digest，LeRobot metadata，compiled artifact SHA-256，binding compatibility，最后才创建
backend runtime。

`bundle.digest` 算法：

1. 将每个 `bundle.files` path 规范化为唯一的 bundle-relative POSIX path。
2. 按 path 排序 `{\"path\": ..., \"sha256\": ...}` entries。
3. 使用 UTF-8 JSON、无多余空白、key 顺序为 `path`、`sha256` 序列化整个数组。
4. 对序列化 bytes 计算 SHA-256。

Selected deployment fingerprint 对以下 canonical object 计算 SHA-256：

```json
{
  "schema_version": 1,
  "bundle_digest": "...",
  "deployment_name": "rk3588",
  "deployment": {}
}
```

路径不能是绝对路径、不能包含 parent traversal、不能通过 symlink 逃逸 bundle root，规范化
后也不能重复。

### 完整性错误处理

遇到以下错误时不要手工改 hash：

- `SHA-256 mismatch`
- `Bundle digest mismatch`
- missing or unexpected LeRobot semantic files
- execution role 缺少 artifact 或 bindings
- runtime ABI 与 LeRobot feature shape 不匹配

应重新运行拥有该 artifact 的 exporter 或 packaging workflow。Exporter 负责复制 artifact、
读取 compiler/runtime ABI、生成 bindings、计算所有 SHA-256、更新 bundle digest，并通过生产
loader 重新验证 manifest。

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
