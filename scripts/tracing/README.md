# IB-Robot 链路追踪（ros2_tracing + LTTng）

基于 ROS 2 最佳实践的低开销推理链路追踪方案。

## 环境准备

```bash
./scripts/setup.sh
```

`./scripts/setup.sh` 现在会一并安装 tracing 依赖（LTTng、ros2_tracing、
babeltrace2、`tracetools-analysis`）。

如果工作区已经初始化完成，只是后补 tracing 工具，仍可单独执行：

```bash
bash scripts/tracing/setup_tracing.sh
```

## 使用方式

### 方式 A：在 launch 中集成开启（推荐）

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm use_sim:=true \
    control_mode:=model_inference \
    enable_tracing:=true
```

这会通过 `robot_config.launch_builders.tracing` 在 launch 期间创建一个 LTTng
session，同时采集 ROS 2 UST 事件（`ros2:*`）和 Python 业务追踪点
（`ib_trace.*`）。

如果默认 session 名 `ib_robot_trace` 已经被占用，launch 会自动追加时间戳后缀，
避免覆盖已有 trace；如果你显式传了 `trace_session_name:=...`，则不会帮你覆盖同名会话。

### 方式 B：单独控制 trace 会话

```bash
# 终端 1
bash scripts/tracing/start_trace.sh

# 终端 2
ros2 launch robot_config robot.launch.py ...

# 结束后
bash scripts/tracing/stop_trace.sh
```

手动模式下如果默认 session 名已存在，`start_trace.sh` 也会自动换成带时间戳的新名字；
如果你显式传入了同名 session，则脚本会直接报错，避免踩掉旧数据。

### 结果分析

```bash
python3 scripts/tracing/analyze_trace.py --trace-dir ~/.ros/tracing/ib_robot_trace
```

分析脚本现在会输出两部分：
- **请求级阶段时延**：`Obs sampling → Dispatch→Infer → Inference → Dispatch decode → Refill→Execute ...`
- **Observation 汇总**：每个 observation 的接收传输时延（`transport_ms`）和采样时新鲜度（`age_ms`）

## 工作原理

不需要额外单独维护一个 tracing 包，整体方案遵循 ROS 2 标准做法：

1. **`robot_config.launch_builders.tracing` 管理 LTTng session**
   `robot.launch.py` 只负责声明 launch 参数并组合 builder 输出；tracing builder
   启动时启用 `ros2:*` UST 事件和 Python tracing domain `ib_trace.*`，
   退出时自动 stop/destroy session。

2. **节点中使用 `logging.getLogger('ib_trace.*')` + `lttngust`**
   节点为 `ib_trace.*` logger 显式绑定 LTTng Python handler，因此业务事件会
   作为 Python domain 事件写入 trace；当 LTTng 未开启时，额外开销很低。

3. **`ros2 trace` CLI / `lttng` CLI**  
   用于手动管理 trace session 的标准工具。

4. **`babeltrace2` / Trace Compass / `tracetools_analysis`**  
   用于后处理和分析的标准工具链。

## 业务追踪点

业务事件通过 `ib_trace.*` logger 以
`logging.info("[event_name] key=value ...")` 形式发出：

| 事件 | Logger | 节点 |
|------|--------|------|
| `dispatch_request` | `ib_trace.dispatch` | action_dispatcher_node |
| `dispatch_result` | `ib_trace.dispatch` | action_dispatcher_node |
| `dispatch_decode` | `ib_trace.dispatch` | action_dispatcher_node |
| `queue_refill` | `ib_trace.dispatch` | action_dispatcher_node |
| `action_execute` | `ib_trace.dispatch` | action_dispatcher_node |
| `action_topic_publish` | `ib_trace.execute` | topic_executor |
| `obs_receive` | `ib_trace.policy` | lerobot_policy_node |
| `obs_sample` | `ib_trace.policy` | lerobot_policy_node |
| `obs_frame` | `ib_trace.policy` | lerobot_policy_node |
| `preprocess_begin/end` | `ib_trace.policy` | lerobot_policy_node |
| `inference_begin/end` | `ib_trace.policy` / `ib_trace.inference` | 两者都有 |
| `postprocess_begin/end` | `ib_trace.policy` | lerobot_policy_node |
| `action_chunk_publish` | `ib_trace.policy` | lerobot_policy_node |
| `edge_publish` | `ib_trace.policy` | lerobot_policy_node |
| `edge_receive` | `ib_trace.policy` | lerobot_policy_node |

## 文件列表

```
robot.launch.py          ← launch 编排入口（enable_tracing:=true 时接入 tracing builder）
src/robot_config/robot_config/launch_builders/tracing.py
                        ← 启动/停止 LTTng session
scripts/tracing/
├── setup_tracing.sh     ← 给已初始化工作区补装 tracing 依赖（常规 setup 已包含）
├── start_trace.sh       ← 手动启动 tracing session
├── stop_trace.sh        ← 手动停止 tracing session
├── analyze_trace.py     ← 时延分析 CLI
├── README.md            ← 中文文档
└── README.en.md         ← 英文文档
```
