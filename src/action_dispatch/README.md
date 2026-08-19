# Action Dispatch

动作分发层，位于推理模型与 ros2_control 之间的拉取式动作分发系统。

## 概述

本包提供了一个高效的动作分发机制，用于将具身智能模型输出的动作分发给机器人控制器。支持 Action Chunking 模型（如 ACT、Diffusion Policy）的跨帧平滑功能，确保连续推理输出之间的平滑过渡。

本包包含两个互斥 executable，具体选择只来自
`control_modes.<mode>.inference.scheduler.enable`：

- 缺失或 `false`：启动 `action_dispatcher_node`，沿
  `executor.inference_pipeline -> DispatchInfer /dispatch + /reset` 工作，行为与原路径一致。
- `true`：启动 `scheduled_action_dispatcher_node`，等待 Global readiness 后依次使用
  `OpenInferenceSession`、`ScheduledDispatchInfer` 和 `CloseInferenceSession`。Open 只建立逻辑 session，
  不携带模型或 fallback；dispatcher 拥有单个 product session，
  校验所有结果 identity；terminal failure 或 `UNKNOWN` 会先清空队列/平滑器并发布 safe-stop，再 Close 并进入
  `FAILED`。每次 Dispatch 都显式携带 target、priority 和新的绝对 deadline；priority-0 还携带 fallback chain。
  只有适合重试的 recoverable `NOT_STARTED` 才使用新 request UUID 且复用原 deadline 做有界重试；deadline
  不可行、容量已满和 ingress 拒绝会直接 safe-stop/Close；Global 为 priority-0 保留独立进入容量，低优先级
  请求不能将其耗尽。Stop/Restart 会等待未决 Open 后用实际
  generation Close；SIGINT/SIGTERM 退出时保持 executor 和 ROS context 存活到 safe-stop/Close 完成或超时。
  新 action chunk 会校验 tensor step 数与 result `chunk_size` 一致，再按推理期间已执行的动作数对齐；队列暂时
  耗尽时持续发布最后动作保持控制输出。safe-stop 的 joint snapshot freshness 使用本进程 monotonic receive
  time，不依赖 ROS/sim time 或消息 header stamp。

两个节点都使用 `/action_dispatcher` 名称并保留 start/stop/status/toggle-smoothing 接口，launch graph 不会让它们
共存。Scheduled dispatcher 消费 Global 返回的 whole-graph action chunk。

## 系统架构

### 组件架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              IB Robot System                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         ┌─────────────┐ │
│  │  Inference       │         │  Action          │         │  ros2_      │ │
│  │  Service         │         │  Dispatch        │         │  control    │ │
│  │                  │         │                  │         │             │ │
│  │ ┌──────────────┐ │         │ ┌──────────────┐ │         │ ┌─────────┐ │ │
│  │ │ Model        │ │         │ │ Action       │ │         │ │ Joint   │ │ │
│  │ │ (ACT/Diff)   │ │         │ │ Dispatcher   │ │         │ │ State   │ │ │
│  │ └──────────────┘ │         │ │   Node       │ │         │ │ Pub/Sub │ │ │
│  │                  │         │ └──────────────┘ │         │ └─────────┘ │ │
│  │                  │         │        │         │         │             │ │
│  │                  │         │        ▼         │         │             │ │
│  │                  │         │ ┌──────────────┐ │         │             │ │
│  │                  │         │ │ Temporal     │ │         │             │ │
│  │                  │         │ │ Smoother     │ │         │             │ │
│  │                  │         │ └──────────────┘ │         │             │ │
│  │                  │         │        │         │         │             │ │
│  │                  │         │        ▼         │         │             │ │
│  │                  │         │ ┌──────────────┐ │         │             │ │
│  │                  │         │ │ Topic        │───────────▶│ Controllers│ │
│  │                  │         │ │ Executor     │ │         │             │ │
│  │                  │         │ └──────────────┘ │         │             │ │
│  └──────────────────┘         └──────────────────┘         └─────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 通信架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ROS2 Communication                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                              ┌──────────────────┐     │
│  │ Inference Service │                              │ Action Dispatch  │     │
│  │                   │                              │                  │     │
│  │                   │    DispatchInfer Action      │                  │     │
│  │                   │◀─────────────────────────────│                  │     │
│  │                   │    (ibrobot_msgs/action)     │                  │     │
│  │                   │                              │                  │     │
│  │                   │    VariantsList (Result)     │                  │     │
│  │                   │─────────────────────────────▶│                  │     │
│  │                   │    (action chunk tensor)     │                  │     │
│  └──────────────────┘                              └────────┬─────────┘     │
│                                                             │                │
│                                                             │                │
│  ┌──────────────────┐                              ┌────────▼─────────┐     │
│  │ ros2_control     │                              │ TopicExecutor    │     │
│  │                  │◀─────────────────────────────│                  │     │
│  │ /joint_commands  │   Float64MultiArray /        │                  │     │
│  │ /arm_commands    │   JointTrajectory            │                  │     │
│  └──────────────────┘                              └──────────────────┘     │
│                                                                              │
│  ┌──────────────────┐                              ┌──────────────────┐     │
│  │ Sensor Layer     │                              │ Action Dispatch  │     │
│  │                  │                              │                  │     │
│  │ /joint_states    │─────────────────────────────▶│ (subscription)   │     │
│  │ (JointState)     │   optional                   │                  │     │
│  └──────────────────┘                              └──────────────────┘     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 内部数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ActionDispatcherNode 内部流程                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌─────────────┐                                                          │
│    │ 推理请求    │                                                          │
│    │ (水位线触发)│                                                          │
│    └──────┬──────┘                                                          │
│           │                                                                  │
│           ▼                                                                  │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│    │ 记录当前    │      │ 发送        │      │ 等待        │               │
│    │ 队列长度    │─────▶│ DispatchInfer─────▶│ 推理结果    │               │
│    │             │      │ Goal        │      │             │               │
│    └─────────────┘      └─────────────┘      └──────┬──────┘               │
│                                                      │                      │
│                                                      ▼                      │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│    │ 计算推理    │      │ 时间对齐    │      │ 解码        │               │
│    │ 期间执行    │◀─────│ (跳过已执行 │◀─────│ VariantsList│               │
│    │ 的动作数    │      │  的动作)    │      │ to Tensor   │               │
│    └──────┬──────┘      └─────────────┘      └─────────────┘               │
│           │                                                                  │
│           ▼                                                                  │
│    ┌─────────────────────────────────────────────────────────┐             │
│    │                    TemporalSmoother                      │             │
│    │  ┌─────────────────────────────────────────────────┐    │             │
│    │  │  启用平滑:                                        │    │             │
│    │  │    old_actions + new_actions → blended_actions   │    │             │
│    │  │    (指数加权平滑重叠区域)                         │    │             │
│    │  └─────────────────────────────────────────────────┘    │             │
│    │  ┌─────────────────────────────────────────────────┐    │             │
│    │  │  禁用平滑:                                        │    │             │
│    │  │    new_actions → 直接替换队列                     │    │             │
│    │  └─────────────────────────────────────────────────┘    │             │
│    └───────────────────────────┬─────────────────────────────┘             │
│                                │                                            │
│                                ▼                                            │
│    ┌─────────────┐      ┌─────────────┐      ┌─────────────┐               │
│    │ 控制循环    │      │ 取出下一个  │      │ TopicExecutor│               │
│    │ (100Hz)     │─────▶│ 动作        │─────▶│ 发布到话题   │               │
│    │             │      │             │      │             │               │
│    └─────────────┘      └─────────────┘      └─────────────┘               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 核心组件

### 1. ActionDispatcherNode

主要的 ROS2 节点，负责：
- 维护动作队列
- 基于水位线触发推理请求
- 以固定频率向 ros2_control 发布动作
- 可选的跨帧时间平滑

### 2. TemporalSmoother

跨帧指数平滑器，用于处理 Action Chunking 模型的输出：
- 维护平滑后的动作规划
- 新推理结果到达时进行时间对齐
- 指数加权平滑重叠区域

### 3. TopicExecutor

基于话题的动作执行器：
- 根据 Contract 规范路由动作到正确的话题
- 支持 `Float64MultiArray` 和 `JointTrajectory` 消息类型
- 高频率位置控制

## 安装

```bash
cd ~/ibrobot_ws
colcon build --packages-select action_dispatch
source install/setup.bash
```

## 使用方法

### 启动节点

```bash
ros2 run action_dispatch action_dispatcher_node
```

### 参数配置

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `queue_size` | int | 100 | 动作队列最大长度 |
| `watermark_threshold` | int | 20 | 触发推理的水位线阈值 |
| `control_frequency` | double | 100.0 | 控制频率 (Hz) |
| `inference_action_server` | string | `/inference/policy/dispatch` | 命名 pipeline 的推理 Action；robot_config 会按 `executor.inference_pipeline` 覆盖 |
| `inference_reset_service` | string | `/inference/policy/reset` | 命名 pipeline 的 reset 服务；reset 时 best-effort 调用 |
| `policy_reset_timeout_sec` | double | 2.0 | 等待推理侧 policy 重置完成的最长时间 |
| `contract_path` | string | `''` | Contract 文件路径 |
| `joint_state_topic` | string | `/joint_states` | 关节状态话题 |
| `navigation_mode` | bool | false | 导航模式（启动时停止，等待外部触发） |
| `temporal_smoothing_enabled` | bool | false | 是否启用跨帧平滑 |
| `temporal_ensemble_coeff` | double | 0.01 | 平滑系数 |
| `chunk_size` | int | 100 | Action Chunk 大小 |
| `smoothing_device` | string | `''` | 平滑计算设备 (空=自动检测) |

### Launch 文件示例

```python
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='action_dispatch',
            executable='action_dispatcher_node',
            name='action_dispatcher',
            parameters=[{
                'queue_size': 100,
                'watermark_threshold': 20,
                'control_frequency': 100.0,
                'temporal_smoothing_enabled': True,
                'temporal_ensemble_coeff': 0.01,
                'chunk_size': 100,
                'contract_path': '/path/to/contract.yaml',
            }]
        )
    ])
```

## 跨帧平滑功能

### 原理说明

具身模型通常以 Action Chunk 形式输出，一次推理输出 n 个动作。跨帧平滑解决的问题：

```
第一次推理: 产生 n 个 action chunk
执行了 l 个后 (l < n)，第二次推理完成
新推理结果需要与剩余的 n-l 个动作做平滑对齐
```

### 跨帧平滑示意图

```
时间轴 (Timeline) ──────────────────────────────────────────────────────────────────────────────▶

【T1: 初始状态】     第一次推理输出 (n=10)
                    [ a1, a2, a3, a4, a5, a6, a7, a8, a9, a10 ]
                    └──┬───┘      └──────────┬───────────┘
                       │                     │
                (推理期间耗时执行)        (待平滑的旧动作)
                       │                     │
【T2: 推理结束】     执行中/已过时          剩余队列 (7个)
                    [ a1, a2, a3 ] ──────▶ [ a4, a5, a6, a7, a8, a9, a10 ]
                    (已执行 3 个)                   │
                                                   │
【T3: 新旧对齐】     第二次推理输出 (n=10)           │ (寻找对应索引)
                    [ b1, b2, b3, b4, b5, b6, b7, b8, b9, b10 ]
                    └─────┬─────┘ └──────────┬───────────┘
                       (跳过)            (对应的新动作)
                                             │
【T4: 执行平滑】                             ▼
                    ┌──────────────────────────────────────────┐
                    │  Blend(a4,b4) ... Blend(a10,b10) + [b新] │
                    └──────────────────────────────────────────┘
                                        │
【T5: 最终输出】     [ m4, m5, m6, m7, m8, m9, m10 ] + [b11...] 
                    (平滑后的混合动作序列)
```

### 平滑过程详解

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          跨帧平滑计算过程                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  原始动作队列 (第一次推理结果):                                                 │
│  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─┐ ┌────┬────┬────┬────┬────┬────┬────┐                  │
│  │  a1   a2   a3   │ │ a4 │ a5 │ a6 │ a7 │ a8 │ a9 │a10 │                   │
│  └ ─ ─ ─ ─ ─ ─ ─ ─ ─┘ └────┴────┴────┴────┴────┴────┴────┘                  │
│  ╎                     │    │    │    │    │    │    │                      │
│  ╎ 已执行 (跳过)        │    │    │    │    │    │    │   剩余队列             │
│  ╎ (推理期间执行了3个)   │    │    │    │    │    │    │   count: [1,1,1,1,1,1,1]│
│  ╎                    ▼    ▼    ▼    ▼    ▼    ▼    ▼                       │
│  ╎                                                                          │
│  ╎  新推理结果 (完整):                                                        │
│  ╎  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─┐ ┌────┬────┬────┬────┬────┬────┬────┐               │
│  ╎  │  b1   b2   b3    │ │ b4 │ b5 │ b6 │ b7 │ b8 │ b9 │b10 │               │
│  ╎  └ ─ ─ ─ ─ ─ ─ ─ ─ ─┘ └────┴────┴────┴────┴────┴────┴────┘               │
│  ╎  已过时 (跳过)          │    │    │    │    │    │    │                     │
│  ╎                       │    │    │    │    └────┴────┴──▶ 新动作尾部        │
│  ╎                       │    │    │    │          (直接追加)                │
│  ╎                       └────┴────┴────┴──▶ 重叠区域 (需要平滑)              │
│  ╎                                                                          │
│  ╎  权重计算: w = exp(-0.01 * k),  cumsum = [1.00, 1.99, 2.97, ...]          │
│                                                                              │
│  平滑计算 (重叠区域):                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  blended[i] = (old[i] * cumsum[count-1] + new[i] * weight[count])   │   │
│  │                         cumsum[count]                                 │   │
│  │                                                                       │   │
│  │  示例 (i=0, count=1):                                                 │   │
│  │    blended = (a4 * 1.00 + b4 * 0.99) / 1.99                          │   │
│  │           = 0.502 * a4 + 0.498 * b4                                  │   │
│  │                                                                       │   │
│  │  多次平滑后 (count=k):                                                │   │
│  │    旧动作权重逐渐累积，新动作权重递减                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  最终平滑结果:                                                               │
│  ┌────────┬────────┬────────┬────────┬────┬────┬────┬────┬────┬────┐      │
│  │blend(4)│blend(5)│blend(6)│blend(7)│ b8 │ b9 │b10 │    │    │    │      │
│  └────────┴────────┴────────┴────────┴────┴────┴────┴────┴────┴────┘      │
│    └──────────┬──────────┘   └──┬──┘                                        │
│         平滑区域              新动作尾部                                     │
│                                                                              │
│  图例: ╎ ╎ ╎ = 虚线表示已执行/已过时的动作，不参与平滑计算                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 平滑公式

```python
blended[i] = (old[i] * cumsum[count[i]-1] + new[i] * weight[count[i]]) / cumsum[count[i]]
```

其中：
- `old[i]`: 旧动作规划中的第 i 个动作
- `new[i]`: 新推理结果中的第 i 个动作
- `weight[k]`: 第 k 次贡献的权重 = exp(-coeff * k)
- `cumsum[k]`: 累积权重和

### 平滑系数说明

| 系数值 | 效果 |
|--------|------|
| `0.0` | 均匀权重，无新旧偏好 |
| `正数` | 更倾向于旧动作（稳定、保守） |
| `负数` | 更倾向于新动作（响应快，可能抖动） |

默认值 `0.01` 来自 ACT 原论文。

### 运行时切换

```bash
# 切换平滑开关
ros2 service call /action_dispatcher/toggle_smoothing std_srvs/srv/Empty

# 重置状态
ros2 service call /action_dispatcher/reset std_srvs/srv/Empty "{}"
```

## 导航模式 (Navigation Mode)

当 `navigation_mode=true` 时，系统启动后处于停止状态，等待外部触发开始执行。此模式用于 nav2 导航到达目的地后，触发 ACT 模型执行抓取任务。

### 工作流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Navigation Mode 工作流程                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. 系统启动                                                                  │
│     ┌─────────────┐                                                          │
│     │ Dispatcher  │  启动时 _is_running = False                              │
│     │ [NAV] 模式  │  系统就绪，等待触发                                        │
│     └─────────────┘                                                          │
│                                                                              │
│  2. Nav2 导航                                                                 │
│     ┌─────────────┐                                                          │
│     │   Nav2      │  导航到目标位置                                            │
│     │  导航中...  │  Dispatcher 不执行动作                                     │
│     └─────────────┘                                                          │
│                                                                              │
│  3. 到达目的地                                                                │
│     ┌─────────────┐                                                          │
│     │  Nav2 完成  │  调用 /action_dispatcher/start_evaluate                  │
│     └─────────────┘                                                          │
│                                                                              │
│  4. ACT 执行                                                                  │
│     ┌─────────────┐                                                          │
│     │ Dispatcher  │  _is_running = True                                      │
│     │ 开始执行    │  触发推理，执行 ACT 动作序列                                │
│     └─────────────┘                                                          │
│                                                                              │
│  5. 任务完成                                                                  │
│     ┌─────────────┐                                                          │
│     │  调用       │  调用 /action_dispatcher/stop_evaluate                   │
│     │ stop_evaluate│  停止执行，底盘速度置零                                   │
│     └─────────────┘                                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 使用方法

```bash
# 启动系统（navigation_mode=true 时）
ros2 launch robot_config robot.launch.py robot_config:=lekiwi_realsense_navigation control_mode:=navi

# Nav2 到达目的地后，开始执行
ros2 service call /action_dispatcher/start_evaluate std_srvs/srv/Trigger

# 任务完成后，停止执行
ros2 service call /action_dispatcher/stop_evaluate std_srvs/srv/Trigger

# 查询当前状态
ros2 service call /action_dispatcher/get_status std_srvs/srv/Trigger
```

### 配置示例

在机器人配置 YAML 中启用：

```yaml
control_modes:
  navi:
    executor:
      navigation_mode: true    # 启用导航模式
      watermark_threshold: 20
      control_frequency: 30.0
```

## 话题和服务

### 与 Inference Service 通信

| 方向 | 话题/Action | 消息类型 | 说明 |
|------|-------------|----------|------|
| legacy 请求 | `/inference/policy/dispatch` | `ibrobot_msgs/action/DispatchInfer` | scheduler 缺失/false 时向 `executor.inference_pipeline` 发送推理请求 |
| scheduled session | `/inference/session/open`、`/inference/session/close` | `OpenInferenceSession`、`CloseInferenceSession` | scheduler true 时只访问 Global endpoint；Open 不参与模型路由 |
| scheduled 请求 | `/inference/dispatch` | `ScheduledDispatchInfer` | scheduler true 时携带 session/generation/request/target/priority/deadline；priority-0 另带 fallback chain |
| 响应 | `result.action_chunk` | `ibrobot_msgs/msg/VariantsList` | 接收动作块 (Tensor) |

### 发布话题

| 话题 | 消息类型 | 说明 |
|------|----------|------|
| `~/queue_size` | `std_msgs/Int32` | 当前队列长度 |
| `~/smoothing_enabled` | `std_msgs/Bool` | 平滑是否启用 |

### 订阅话题

| 话题 | 消息类型 | 说明 |
|------|----------|------|
| `/joint_states` | `sensor_msgs/JointState` | 关节状态（可选） |

### 服务

| 服务 | 类型 | 说明 |
|------|------|------|
| `~/reset` | `std_srvs/Empty` | （仅 legacy/关闭分支）重置队列和状态；同时 best-effort 调用 `inference_reset_service` 重置推理侧 policy 状态。调度启用路径不使用本服务，改用 `~/restart_session` |
| `~/toggle_smoothing` | `std_srvs/Empty` | 切换平滑开关 |
| `~/start_evaluate` | `std_srvs/Trigger` | 恢复 dispatcher 执行 |
| `~/stop_evaluate` | `std_srvs/Trigger` | 暂停 dispatcher 执行；`navigation_mode=true` 时额外停止底盘 |
| `~/get_status` | `std_srvs/Trigger` | 获取运行状态；scheduled 路径返回 session state machine 状态 |
| `~/restart_session` | `std_srvs/Trigger` | 仅 scheduled executable：safe-stop、Close、清理本地状态并用新 UUID Open |

### 与 ros2_control 通信

| 方向 | 话题 | 消息类型 | 说明 |
|------|------|----------|------|
| 发布 | `/joint_commands` | `std_msgs/Float64MultiArray` | 关节位置命令 |
| 发布 | `/arm_controller/joint_trajectory` | `trajectory_msgs/JointTrajectory` | 轨迹命令 |

## API 使用

### 直接使用 TemporalSmoother

```python
from action_dispatch import TemporalSmoother, TemporalSmootherConfig

# 创建配置
config = TemporalSmootherConfig(
    enabled=True,
    chunk_size=100,
    temporal_ensemble_coeff=0.01,
)

# 创建平滑器
smoother = TemporalSmoother(config)

# 第一次推理
actions1 = model.inference(obs)  # shape: (100, action_dim)
smoother.update(actions1, actions_executed=0)

# 逐个获取动作
for _ in range(30):
    action = smoother.get_next_action()
    robot.execute(action)

# 第二次推理（期间执行了30个动作）
actions2 = model.inference(obs)
smoother.update(actions2, actions_executed=30)

# 继续执行平滑后的动作
while smoother.plan_length > 0:
    action = smoother.get_next_action()
    robot.execute(action)
```

### 使用 TemporalSmootherManager

```python
from action_dispatch import TemporalSmootherManager

manager = TemporalSmootherManager(
    enabled=True,
    chunk_size=100,
    temporal_ensemble_coeff=0.01,
)

# 运行时切换
manager.set_enabled(False)  # 禁用平滑
manager.set_enabled(True)   # 启用平滑

# 查看状态
print(f"Plan length: {manager.plan_length}")
print(f"Smoothing enabled: {manager.is_enabled}")
```

## 依赖

- ROS2 Humble
- Python 3.10+
- PyTorch
- NumPy
- ibrobot_msgs
- tensormsg

## 许可证

Apache License 2.0
