# embodied_agent 节点说明

`embodied_agent` 是当前最小具身闭环中的**任务入口与任务编排包**。
它不直接控制机械臂，而是把文本命令转换成结构化任务，再规划为技能序列，最后把技能逐个交给 `skill_library` 执行。

当前包内包含 3 个 ROS 2 节点：

| 节点 | 控制台入口 | 主要职责 |
| --- | --- | --- |
| `task_entry_node` | `task_entry_node = embodied_agent.task_entry_node:main` | 把 `/voice_command` 文本优先做规则直达，命中则直接产出 `planned_task`，否则封装成 `TaskCommand` 交给 planner |
| `task_planner_node` | `task_planner_node = embodied_agent.task_planner_node:main` | 按规则把文本任务规划为技能序列 |
| `task_executor_node` | `task_executor_node = embodied_agent.task_executor_node:main` | 顺序调用技能 action，并发布任务状态 |

包内还提供一个**非节点的库接口** `embodied_agent.llm_client_service.LLMClientService`：封装云端对话大模型的一行式调用，供上层按需 import，不随 launch 起节点。详见第 6 节。

## 1. 在整体架构中的位置

当前最小闭环链路是：

```text
/voice_command
  -> task_entry_node
  -> 命中视觉互动触发词(如"分院帽"): /embodied/perception_request -> perception_service_node -> /embodied/perception_result (立即返回，不进 planner/executor)
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
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=moveit_planning \
  use_sim:=true \
  moveit_display:=false
```

`embodied_bringup` 会读取同一份 `robot_config` YAML，并临时覆盖：

- `robot.embodied.enabled=true`

默认 YAML 中该能力仍是关闭的。

## 3. task_entry_node

`task_entry_node` 是最前面的文本任务入口。

### 作用

1. 订阅文本命令。
2. 为每条命令生成唯一 `task_id`。
3. **先匹配视觉趣味游戏触发词**（如"分院帽"，别名与开关来自 `embodied.entry.visual_games`）。命中即构造 `SceneAnalysisRequest` 发到 `/embodied/perception_request` 交给 `perception_service_node`，并**立即返回**——一句语音只属于一个业务域，不会同时触发趣味 VLM 和机器人任务规划。
4. 未命中游戏时，用现有规则解析器判断是否能直接映射到技能序列。
5. 能直接映射时，直接发布到 `/embodied/planned_task`。
6. 规则未命中时，再封装成 `ibrobot_msgs/msg/TaskCommand` 发布到规划阶段。

说明：

- 视觉游戏结果以 `/embodied/perception_result` 上的 `SceneAnalysisResult` 为准，通过 `source=game.<name>`（如 `game.sorting_hat`）识别业务类型，`scene_summary` 保存最终结果（分院帽为四学院之一），失败时带 `error_code`/`message`。
- 启用某游戏需**同时**置 `embodied.perception.enabled: true` 与 `embodied.entry.visual_games.<name>.enabled: true`；否则请求会打到无人消费的 topic，`task_entry_node` 会记 ERROR 并丢弃请求（配置层 `validate_config` 也会拒绝该不一致配置）。
- **输入前置条件（通用 `required_inputs`）**：请求在 `context_json.required_inputs` 里声明它真正需要的输入，perception 据此判定哪些缺失才阻塞。分院帽只声明 `primary_image`，因此即便 EE pose / joint state 离线（MoveIt 未起、控制器重启、独立跑 perception）也能成功，只要主相机图像可用。未声明 `required_inputs` 的普通感知请求维持严格默认：要求主图 + EE pose + joint state 全部在线，否则返回 `SCENE_ANALYSIS_FAILED`。

说明：

- `task_entry_node` 现在会为每个任务写入统一的**任务总超时预算**。
- deadline 会通过 `TaskCommand.context_json.timeout_context` 贯穿 planner / executor。
- 规划、执行都会共同消耗这一预算，而不是只在执行阶段单独计时。

### 当前接口

| 方向 | 话题 | 类型 | 说明 |
| --- | --- | --- | --- |
| 订阅 | `/voice_command` | `std_msgs/msg/String` | 当前默认文本输入入口 |
| 发布 | `/embodied/perception_request` | `ibrobot_msgs/msg/SceneAnalysisRequest` | 命中视觉游戏触发词后发给 perception |
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
| `default_place_name` | `home` | 保留参数；当前规则规划不会生成放置技能 |
| `default_relative_motion_step_m` | `0.03` | “一点”默认映射步长（米） |
| `default_task_timeout_sec` | `180.0` | 单个任务的端到端总超时预算 |
| `perception_request_topic` | `/embodied/perception_request` | 视觉互动请求输出（复用 perception 请求 topic） |
| `perception_enabled` | `false` | perception 是否启用；用于互动启用一致性校验 |
| `entry_visual_games_json` | `{}` | 入口视觉趣味游戏策略（开关 + 触发别名），来自 `embodied.entry.visual_games` |
| `skill_aliases_json` | `{}` | 从启用且 `rule_entry: true` 的 YAML skill 注入的规则入口别名 |
| `debug_tracing` | `false` | 是否打印调试日志 |

## 4. task_planner_node

`task_planner_node` 把 `TaskCommand.raw_command` 转成一个**确定性的技能序列**。

### 当前支持的任务意图

| 输入意图 | 规划结果 |
| --- | --- |
| `观察点` / `观察位置` / `观察桌面` / `看看桌面` / `观察场景` | `inspect_scene` |
| `原位` / `原点` / `回到home` / `回原位` / `回安全位` | `recover_safe_pose` |
| `零点` / `零位` / `回零点` / `到零点` | `recover_zero_pose` |
| `夹爪往前/后/左/右/上/下一点` | `move_relative_ee` |
| `打开夹爪` / `开爪` | `open_gripper_skill` |
| `关闭夹爪` / `夹紧` | `close_gripper_skill` |
| `顺时针旋转 45 度` | `rotate_gripper_cw` |
| `逆时针旋转 45 度` | `rotate_gripper_ccw` |

### Rule-entry alias 与 `task_type` 契约

`skill_aliases_json` 由 `robot_config` 中启用且 `description.rule_entry: true` 的 skill
生成，只用于让无参数社交动作进入确定性规则解析。设置 `disabled: true` 的 skill 不会进入
别名集合。社交动作命中后，其 skill 名同时作为 `task_type` 和 `skill_sequence` 的唯一项。

既有观察、回位、夹爪开合、相对移动和带角度旋转命令仍优先走专用规则分支，不由 alias
改写其公开 `task_type`。例如观察保持 `observe_scene`、打开夹爪保持 `open_gripper`、
相对移动保持 `relative_motion`。

### 当前约束

- 这是**规则规划器**，不是通用大模型 Planner。
- 规则规划器仍不直接生成抓取；VLM 或 Hermes 可在机器人配置允许时显式选择 `pick_object`。
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
| `default_place_name` | `home` | 保留参数；当前规则规划不会生成放置技能 |
| `default_relative_motion_step_m` | `0.03` | “一点”默认映射步长（米） |
| `skill_aliases_json` | `{}` | 从启用且 `rule_entry: true` 的 YAML skill 注入的规则入口别名 |
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

## 6. LLMClientService

`LLMClientService` 是一个**库接口**（非 ROS 节点，无控制台入口、无话题/参数），封装云端对话大模型的一行式调用。它接收由业务方提供的预设 prompt 与用户文字，调用云端对话模型生成回复文本。

### 作用

1. 可选地从文件读入预设 system prompt，作为每轮请求都携带的系统指令；不提供时退化为无预设的裸对话。
2. 接收用户文字，调用云端对话模型生成回复。
3. 复用 `embodied_common` 的 `VLMClient` 管理多轮上下文，业务层不自行维护对话历史。
4. 返回底层结构化 dict，云端错误（网络、配额、缺 API key 等）如实透传，不吞异常、不返回空串。

说明：

- 本接口**不内置任何默认 prompt**：system prompt 的内容与来源属于业务设计，由业务方通过 `system_prompt_path` 传入。
- 预设 system prompt（若提供）独立保存，每轮请求都置于最前，不写入对话历史；因此对话变长也不会把它挤出上下文，`reset()` 也不会丢失它。
- 上下文管理完全由 `embodied_common.vlm_api_client.VLMClient` 负责，本接口只做拼装调用。
- 云端模型路由（provider / endpoint / API key 环境变量名）由 `embodied_common` 的 `vlm_models.yaml` 单一管理；API key 通过环境变量注入，不写入任何配置文件。

### 当前接口

| 方法 | 说明 |
| --- | --- |
| `reply(user_text) -> dict` | 发送一轮用户文字，返回 `status`/`content`/`error`/`usage`/`timing_ms` 等结构化字段；空或非字符串输入抛 `ValueError` |
| `reset() -> None` | 清空多轮对话历史；预设 system prompt 不受影响，后续仍会携带 |

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `system_prompt_path` | `None` | 预设 system prompt 文件路径（由业务方提供）；`None` 时不设 system prompt，退化为无预设的裸对话 |
| `model` | `None` | 指定 `vlm_models.yaml` 中的模型名；`None` 时使用 `defaults.model` |
| `vlm` | `None` | 注入自定义 `VLMClient`（用于测试或依赖注入）；给定时忽略 `system_prompt_path` |

### 使用示例

```python
from embodied_agent.llm_client_service import LLMClientService

svc = LLMClientService(system_prompt_path="/path/to/your_system_prompt.txt")  # 业务方提供 prompt
# svc = LLMClientService()                # 也可不传，退化为无预设裸对话
result = svc.reply("你好呀，你是谁？")       # 返回结构化 dict
if result["status"] == "ok":
    print(result["content"])
svc.reset()                              # 开启新话题（system prompt 仍保留）
```

调用前需按 `vlm_models.yaml` 中对应模型的 `api_key_env` 注入 API key，例如：

```bash
export ALIYUN_API_KEY=sk-xxxxxx
```

## 7. 任务与状态接口

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

## 7.1 当前归一后的超时类型

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

## 8. 当前验证通过的最小闭环

当前已经验证通过的仿真命令路径：

```bash
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '夹爪往前一点'}"
```

可以观测到：

1. `/embodied/task_command`
2. `/embodied/planned_task`
3. `/embodied/task_status`
4. 技能 action 被依次调用
5. 最终 `TaskStatus.state=completed`

当前也支持夹爪开合类命令，例如：

```bash
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '打开夹爪'}"
```

这类命令会被规划成单技能：

- `open_gripper_skill`

当前还支持直接移动到配置好的 named pose：

```bash
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '原位'}"
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '观察点'}"
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '零点'}"
```

这三类命令分别会被规划成：

- `recover_safe_pose`
- `inspect_scene`
- `recover_zero_pose`

方向语义由执行层按 `robot.embodied.execution.relative_motion_reference_frame=base`
和 `relative_motion_direction_mapping` 解释，规划层只保留
`forward/backward/left/right/up/down` 语义标签。

## 9. 当前限制

- 目前只支持**最小规则闭环**，不是开放式具身智能 Agent。
- 当前文本解析只覆盖少量中文模式。
- 目标物识别仍是命名目标映射，不是视觉 grounding。
- 执行依赖 `moveit_planning` 控制模式和 `skill_library`/`safety_guard` 正常工作。
