# embodied_agent 节点说明

`embodied_agent` 是当前最小具身闭环中的**任务入口与任务编排包**。
它不直接控制机械臂，而是把文本命令转换成结构化任务，再规划为技能序列，最后把技能逐个交给 `skill_library` 执行。

当前包内包含 3 个 ROS 2 节点：

| 节点 | 控制台入口 | 主要职责 |
| --- | --- | --- |
| `task_entry_node` | `task_entry_node = embodied_agent.task_entry_node:main` | 把 `/voice_command` 文本优先做规则直达，命中则直接产出 `planned_task`，否则封装成 `TaskCommand` 交给 planner |
| `task_planner_node` | `task_planner_node = embodied_agent.task_planner_node:main` | 按规则把文本任务规划为技能序列 |
| `task_executor_node` | `task_executor_node = embodied_agent.task_executor_node:main` | 顺序调用技能 action，并发布任务状态 |

## 1. 在整体架构中的位置

当前最小闭环链路是：

```text
/voice_command
  -> task_entry_node
  -> 规则可直达: /embodied/planned_task -> task_executor_node
  -> 规则未命中: /embodied/task_command -> task_planner_node / vlm_task_planner_node
  -> /embodied/planned_task
  -> task_executor_node
  -> /embodied/execute_skill
  -> skill_library
  -> MoveIt / gripper
```

其中：

- `robot_config` 仍然是配置和 launch 的单一事实来源。
- `embodied_agent` 只负责**任务理解、规则规划、执行编排**。
- 安全校验由 `safety_guard` 负责。
- 技能和原子动作执行由 `skill_library` 负责。

## 2. 推荐启动方式

建议通过统一 launch 启动，而不是手工分别起节点：

```bash
cd ~/IB_Robot
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.sh && \
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=moveit_planning \
  use_sim:=true \
  with_embodied:=true \
  moveit_display:=false
```

其中 `with_embodied:=true` 会临时覆盖：

- `robot.embodied.enabled=true`

默认 YAML 中该能力仍是关闭的。

## 3. task_entry_node

`task_entry_node` 是最前面的文本任务入口。

### 作用

1. 订阅文本命令。
2. 为每条命令生成唯一 `task_id`。
3. 先用现有规则解析器判断是否能直接映射到技能序列。
4. 能直接映射时，直接发布到 `/embodied/planned_task`。
5. 规则未命中时，再封装成 `ibrobot_msgs/msg/TaskCommand` 发布到规划阶段。

说明：

- `task_entry_node` 现在会为每个任务写入统一的**任务总超时预算**。
- deadline 会通过 `TaskCommand.context_json.timeout_context` 贯穿 planner / executor。
- 规划、执行都会共同消耗这一预算，而不是只在执行阶段单独计时。

### 当前接口

| 方向 | 话题 | 类型 | 说明 |
| --- | --- | --- | --- |
| 订阅 | `/voice_command` | `std_msgs/msg/String` | 当前默认文本输入入口 |
| 发布 | `/embodied/planned_task` | `ibrobot_msgs/msg/TaskCommand` | 规则直达命中后的已规划任务 |
| 发布 | `/embodied/task_command` | `ibrobot_msgs/msg/TaskCommand` | 规则未命中的任务封装，交给 planner/VLM |
| 发布 | `/embodied/task_status` | `ibrobot_msgs/msg/TaskStatus` | 规则直达命中时补发 `planned` 状态 |

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input_topic` | `/voice_command` | 文本命令来源 |
| `output_topic` | `/embodied/task_command` | 任务封装输出 |
| `planned_output_topic` | `/embodied/planned_task` | 规则直达输出 |
| `status_topic` | `/embodied/task_status` | 规则直达时的任务状态输出 |
| `default_target_name` | `demo_object` | 规则直达时使用的默认命名目标 |
| `default_place_name` | `tray_right` | 规则直达时使用的默认放置位姿 |
| `default_relative_motion_step_m` | `0.03` | “一点”默认映射步长（米） |
| `default_task_timeout_sec` | `180.0` | 单个任务的端到端总超时预算 |
| `debug_tracing` | `false` | 是否打印调试日志 |

## 4. task_planner_node

`task_planner_node` 把 `TaskCommand.raw_command` 转成一个**确定性的技能序列**。

### 当前支持的任务意图

| 输入意图 | 规划结果 |
| --- | --- |
| `观察桌面` / `看看桌面` / `观察场景` | `inspect_scene` |
| `回到home` / `回原位` / `回安全位` | `recover_safe_pose` |
| `抓取目标物并放到右侧托盘` | `pick_named_target` -> `place_named_pose` |
| `夹爪往前/后/左/右/上/下一点` | `move_relative_ee` |
| 仅包含抓取类词汇（抓 / 拿 / 取） | `pick_named_target` |
| 仅包含放置类词汇（放） | `place_named_pose` |

### 当前约束

- 这是**规则规划器**，不是通用大模型 Planner。
- 目标物和放置位当前会映射到 YAML 中的默认命名目标与命名位姿。
- 不支持的文本会直接拒绝，并在 `/embodied/task_status` 发布 `rejected`。

### 当前接口

| 方向 | 话题 | 类型 | 说明 |
| --- | --- | --- | --- |
| 订阅 | `/embodied/task_command` | `ibrobot_msgs/msg/TaskCommand` | 原始任务 |
| 发布 | `/embodied/planned_task` | `ibrobot_msgs/msg/TaskCommand` | 已规划任务 |
| 发布 | `/embodied/task_status` | `ibrobot_msgs/msg/TaskStatus` | 规划结果与拒绝原因 |

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input_topic` | `/embodied/task_command` | 规划输入 |
| `output_topic` | `/embodied/planned_task` | 规划输出 |
| `status_topic` | `/embodied/task_status` | 任务状态输出 |
| `default_target_name` | `demo_object` | 默认命名目标 |
| `default_place_name` | `tray_right` | 默认放置位姿 |
| `default_relative_motion_step_m` | `0.03` | “一点”默认映射步长（米） |
| `debug_tracing` | `false` | 是否打印规划调试日志 |

## 5. task_executor_node

`task_executor_node` 是任务级执行编排器。

### 作用

1. 读取 `planned_task` 中的 `skill_sequence`。
2. 逐个调用 `/embodied/execute_skill` action。
3. 按阶段发布 `TaskStatus`。
4. 在超时、拒绝、服务缺失时明确失败并带错误码退出。
5. 超时后会向下游 skill action 发送 cancel，而不是只在上层报错。

### 当前接口

| 方向 | 名称 | 类型 | 说明 |
| --- | --- | --- | --- |
| 订阅 | `/embodied/planned_task` | `ibrobot_msgs/msg/TaskCommand` | 已规划任务 |
| 发布 | `/embodied/task_status` | `ibrobot_msgs/msg/TaskStatus` | 执行中、失败、完成状态 |
| 调用 | `/embodied/execute_skill` | `ibrobot_msgs/action/SkillCommand` | 技能执行入口 |

### 当前状态机语义

常见状态包括：

- `planned`
- `executing`
- `completed`
- `failed`
- `rejected`

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `input_topic` | `/embodied/planned_task` | 执行输入 |
| `status_topic` | `/embodied/task_status` | 状态输出 |
| `skill_action_name` | `/embodied/execute_skill` | 技能 action 名 |
| `default_task_timeout_sec` | `180.0` | 单个任务的端到端总超时预算 |
| `rpc_timeout_sec` | `5.0` | 等待 action/server/service 响应的统一 RPC 超时 |
| `debug_tracing` | `false` | 是否打印执行调试日志 |

## 6. 任务与状态接口

### `ibrobot_msgs/msg/TaskCommand`

当前主要使用这些字段：

| 字段 | 说明 |
| --- | --- |
| `task_id` | 任务唯一 ID |
| `raw_command` | 原始中文命令 |
| `task_type` | 规则规划后的任务类型 |
| `target_name` | 规划出的命名目标 |
| `place_name` | 规划出的命名放置位 |
| `timeout_sec` | 任务总超时预算（秒），由入口统一设置并向后透传 |
| `context_json` | 传递 `skill_sequence` 与 `timeout_context` 等上下文 |

## 6.1 当前归一后的超时类型

当前具身主链路只保留 5 类 timeout / freshness 配置：

1. `task_budget_sec`：任务端到端总预算
2. `scene_freshness_sec`：图像/深度/状态的新鲜度门槛
3. `model_idle_timeout_sec`：大模型输出空闲超时
4. `rpc_timeout_sec`：等待 action/server/service 的统一 RPC 超时
5. `gripper_settle_sec`：夹爪命令发出后的稳定等待时间

### `ibrobot_msgs/msg/TaskStatus`

当前主要使用这些字段：

| 字段 | 说明 |
| --- | --- |
| `state` | 当前状态 |
| `success` | 当前阶段是否成功 |
| `current_skill` | 正在执行的技能 |
| `completed_skills` | 已完成技能列表 |
| `error_code` | 明确错误码 |
| `message` | 详细说明 |
| `recoverable` | 是否可恢复 |
| `replan_requested` | 是否建议重规划 |

## 7. 当前验证通过的最小闭环

当前已经验证通过的仿真命令路径：

```bash
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '抓取目标物并放到右侧托盘'}"
```

可以观测到：

1. `/embodied/task_command`
2. `/embodied/planned_task`
3. `/embodied/task_status`
4. 技能 action 被依次调用
5. 最终 `TaskStatus.state=completed`

当前也支持相对位移类命令，例如：

```bash
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '夹爪往前一点'}"
```

这类命令会被规划成单技能：

- `move_relative_ee`

方向语义由执行层按 `robot.embodied.execution.relative_motion_reference_frame=base`
和 `relative_motion_direction_mapping` 解释，规划层只保留
`forward/backward/left/right/up/down` 语义标签。

## 8. 当前限制

- 目前只支持**最小规则闭环**，不是开放式具身智能 Agent。
- 当前文本解析只覆盖少量中文模式。
- 目标物识别仍是命名目标映射，不是视觉 grounding。
- 执行依赖 `moveit_planning` 控制模式和 `skill_library`/`safety_guard` 正常工作。
