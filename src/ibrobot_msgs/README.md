# ibrobot_msgs

`ibrobot_msgs` 是 IB-Robot 系统的**统一接口定义包**，包含所有 ROS 2 消息（msg）、动作（action）和服务（srv）的定义。

## 1. 消息定义（msg/）

### `TaskCommand.msg`

具身 AI 链路的主要任务数据载体，贯穿任务入口、规划和执行全流程。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `task_id` | `string` | 任务唯一 ID |
| `raw_command` | `string` | 原始自然语言命令 |
| `task_type` | `string` | 规则规划后的任务类型 |
| `target_name` | `string` | 命名目标（如 `demo_object`） |
| `place_name` | `string` | 命名放置位（如 `tray_right`） |
| `timeout_sec` | `float64` | 任务总超时预算（秒） |
| `context_json` | `string` | JSON 字符串，携带 `skill_sequence`、`timeout_context` 等上下文 |

### `TaskStatus.msg`

任务执行状态报告，由 `task_entry_node`、`task_executor_node` 等节点发布到 `/embodied/task_status`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `state` | `string` | 当前状态（`planned` / `executing` / `completed` / `failed` / `rejected`） |
| `success` | `bool` | 当前阶段是否成功 |
| `current_skill` | `string` | 正在执行的技能名 |
| `completed_skills` | `string[]` | 已完成技能列表 |
| `error_code` | `string` | 明确错误码 |
| `message` | `string` | 详细说明 |
| `recoverable` | `bool` | 是否可恢复 |
| `replan_requested` | `bool` | 是否建议重规划 |

### `SceneAnalysisRequest.msg`

结构化场景理解请求，发布到 `perception_service_node`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | `string` | 请求唯一 ID |
| `source` | `string` | 请求来源（如 `cli`、`vlm_planner`） |
| `session_id` | `string` | 会话 ID，用于多轮上下文 |
| `user_text` | `string` | 用户输入文本 |
| `context_json` | `string` | 附加上下文（关注目标、安全限制等） |
| `timeout_sec` | `float64` | 本次请求的模型超时（覆盖默认值） |

### `SceneAnalysisResult.msg`

结构化场景理解结果，由 `perception_service_node` 发布到 `/embodied/perception_result`。

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `request_id` | `string` | 对应请求 ID |
| `success` | `bool` | 理解是否成功 |
| `summary` | `string` | 场景文本摘要 |
| `objects_json` | `string` | 检测到的物体列表（JSON） |
| `error_message` | `string` | 失败时的错误信息 |

### `RobotStatus.msg`

机器人当前状态汇报，包含末端位姿、关节状态和控制模式信息。

### `TaskStep.msg`

单个任务步骤描述，用于 `ExecuteTaskPlan.action` 中的步骤序列。

### `Variant.msg` / `VariantsList.msg`

通用变体类型，用于传递不同类型的配置或参数值。

---

## 2. 动作定义（action/）

### `SkillCommand.action`

技能执行动作接口，由 `skill_executor_node` 提供，路径 `/embodied/execute_skill`。

**Goal 字段**

| 字段 | 说明 |
| --- | --- |
| `task_id` | 任务 ID |
| `skill_name` | 技能名（如 `pick_named_target`、`move_relative_ee`） |
| `target_name` | 命名目标 |
| `place_name` | 命名放置位 |
| `motion_direction` | 相对运动方向（`forward` / `backward` / `left` / `right` / `up` / `down`） |
| `motion_distance` | 相对运动距离（米） |
| `timeout_sec` | 技能超时时间 |

**Result 字段**

| 字段 | 说明 |
| --- | --- |
| `success` | 是否成功 |
| `error_code` | 错误码 |
| `message` | 详细说明 |

### `PrimitiveCommand.action`

原子动作接口，由 `skill_executor_node` 提供，路径 `/embodied/execute_primitive`。

**Goal 字段**

| 字段 | 说明 |
| --- | --- |
| `task_id` | 任务 ID |
| `primitive_name` | 原子动作名（`move_to_named_pose` / `move_relative_ee` / `move_to_joint_positions` / `move_through_joint_positions` / `open_gripper` / `close_gripper`） |
| `pose_name` | 命名位姿（`move_to_named_pose` 使用） |
| `relative_dx/dy/dz` | 相对增量（`move_relative_ee` 使用，单位米） |
| `gripper_position` | 夹爪目标开合量（`[0.0, 1.0]`） |
| `joint_names` | 关节名列表，joint primitive 使用 |
| `joint_positions` | 单个关节目标位置，`move_to_joint_positions` 使用 |
| `primitive_duration_sec` | 单点关节轨迹持续时间 |
| `joint_waypoints` | 扁平化关节路点序列，按 `joint_names` 顺序展开 |
| `joint_waypoint_count` | `joint_waypoints` 中包含的路点数量 |
| `waypoint_duration_sec` | 相邻关节路点的时间间隔 |
| `timeout_sec` | primitive 超时时间 |

### `ExecuteTaskPlan.action`

MoveIt 任务步骤执行接口，由 `task_executor_node` 提供，路径 `/task_executor/execute_task_plan`。

### `DispatchInfer.action`

推理派发动作接口，兼容 OpenClaw 社交控制链路。

### `RecordEpisode.action`

Episode 录制控制接口，由 `dataset_tools` 的录制服务提供。

### `RunPolicy.action`

策略执行动作接口，用于模型推理控制链路。

---

## 3. 服务定义（srv/）

### `ValidateSkill.srv`

技能安全校验服务，路径 `/embodied/validate_skill`，由 `safety_guard_node` 提供。

**请求**

| 字段 | 说明 |
| --- | --- |
| `skill_name` | 待执行技能名 |
| `target_name` | 命名目标 |
| `place_name` | 命名放置位 |

**响应**

| 字段 | 说明 |
| --- | --- |
| `allowed` | 是否允许执行 |
| `reason` | 不允许时的原因 |

### `ValidatePrimitive.srv`

原子动作安全校验服务，路径 `/embodied/validate_primitive`，由 `safety_guard_node` 提供。

**请求**

| 字段 | 说明 |
| --- | --- |
| `primitive_name` | 原子动作名 |
| `pose_name` | 命名位姿名 |
| `relative_dx/dy/dz` | 相对位移增量 |
| `target_x/y/z` | 执行层解析出的目标末端位置 |
| `gripper_position` | 夹爪目标开合量 |
| `joint_names` | 关节名列表，joint primitive 使用 |
| `joint_positions` | 单个关节目标位置 |
| `primitive_duration_sec` | 单点关节轨迹持续时间 |
| `joint_waypoints` | 扁平化关节路点序列 |
| `joint_waypoint_count` | 关节路点数量 |
| `waypoint_duration_sec` | 相邻关节路点的时间间隔 |

**响应**

| 字段 | 说明 |
| --- | --- |
| `allowed` | 是否允许执行 |
| `reason` | 不允许时的原因 |

### `MoveToPose.srv`

MoveIt 位姿移动服务，路径由 `moveit_gateway` 提供。

### `RecognizeFile.srv`

语音文件识别服务，由 `voice_asr_service` 提供。

### `SetHotwords.srv`

热词设置服务，由 `voice_asr_service` 提供。

---

## 4. 许可证

Apache-2.0
