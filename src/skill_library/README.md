# skill_library 节点说明

`skill_library` 是当前具身执行链路里的**技能执行层**。  
它不直接做任务理解，而是负责把上层给出的技能请求拆成有限 primitive，或委托给配置声明的受保护执行器，
再桥接到底层机械臂和夹爪控制接口。

当前包内包含 1 个 ROS 2 节点：

| 节点 | 控制台入口 | 主要职责 |
| --- | --- | --- |
| `skill_executor_node` | `skill_executor_node = skill_library.skill_executor_node:main` | 提供技能/primitive action；位姿与夹爪动作交给 `task_dispatch`，关节轨迹发送到控制器 action |

## 1. 现在可以如何控制机械臂

当前有 3 种主要控制方式，都是通过 `skill_library` 最终落到真实执行：

| 控制方式 | 入口 | 适合场景 |
| --- | --- | --- |
| 自然语言任务 | `/voice_command` | 当前规则入口支持观察、回位、夹爪开合、相对移动、夹爪旋转以及社交手势（挥手、点头、庆祝等，需 launch 注入 SSOT 别名） |
| 技能级控制 | `/embodied/execute_skill` | 明确指定技能名，做稳定、可控的动作编排 |
| primitive 级控制 | `/embodied/execute_primitive` | 直接控制命名位姿、相对位移、关节轨迹、夹爪开合 |

整体链路如下：

```text
/voice_command
  -> task_entry_node
  -> task_planner_node / vlm_task_planner_node
  -> task_executor_node
  -> /embodied/execute_skill
  -> skill_executor_node
  -> /embodied/execute_primitive
  -> pose/rotation/gripper: /task_executor/execute_task_plan
  -> joint trajectory: /arm_trajectory_controller/follow_joint_trajectory
```

## 2. 技能模板来源

当前技能由 `robot_config` 中的 `skill_templates` 统一配置。普通技能展开为有限 primitive；
`executor: grasp_pipeline` 类型的技能委托给受保护的抓取闭环 action，但仍从同一个
`/embodied/execute_skill` 入口进入。

技能 action 层只执行当前模板集合中存在的技能。`get_skill_templates()` 的选择规则是：

- `skill_templates_json` 非空时，使用当前机器人 YAML 注入并过滤 `disabled: true` 后的启用
  `embodied.skill_templates` 集合。
- 未提供、仅空白或为 `{}` 时，模板集合和 Gateway capability catalog 都为空；不会回退到
  `embodied_common.skill_templates.DEFAULT_SKILL_TEMPLATES`。

两套模板不会自动合并。当前机器人的实际技能集合必须以
`robot_config/config/robots/<robot>.yaml` 为准，不应在本 README 另建固定白名单。

SO101 当前配置示例：

| 类别 | 技能 |
| --- | --- |
| 观察与恢复 | `inspect_scene`、`recover_safe_pose`、`recover_zero_pose` |
| 参数化动作 | `move_relative_ee`、`rotate_gripper_cw`、`rotate_gripper_ccw` |
| 夹爪 | `open_gripper_skill`、`close_gripper_skill` |
| 社交与娱乐 | `dance_basic`、`wave_hello`、`nod_yes`、`shake_no`、`celebrate`、`greet_observe_raise`、`act_cute`、`happy_spin_upright` |
| 抓取 | `pick_object`（仅抓取配置，要求显式传入 `target_name`） |

## 3. 当前支持的 primitive

`skill_library` 只允许有限 primitive，避免上层直接下发任意危险动作：

| primitive | 作用 |
| --- | --- |
| `move_to_named_pose` | 移动到命名位姿 |
| `move_to_pose` | 移动到经过安全校验的动态 base-frame 位姿 |
| `move_to_configuration` | 经 MoveIt 移动到经过校验的完整机械臂关节配置 |
| `move_relative_ee` | 相对当前末端位姿做笛卡尔增量移动 |
| `move_to_joint_positions` | 按完整手臂关节顺序移动到单个关节目标 |
| `move_through_joint_positions` | 按完整手臂关节顺序执行多路点关节轨迹 |
| `open_gripper` | 张开夹爪 |
| `close_gripper` | 闭合夹爪 |
| `rotate_gripper_cw` | 绕当前末端局部 Z 轴顺时针旋转 |
| `rotate_gripper_ccw` | 绕当前末端局部 Z 轴逆时针旋转 |

## 4. 技能到 primitive 的映射方式

当前不是硬编码大分支，而是**模板驱动**：

- 模板来源：`robot_config.config.robots.<robot>.yaml`
- 运行时参数：`skill_templates_json`
- 目标相关位姿来自：`named_targets_json`
- 全局命名位姿来自：`named_poses_json`

### 4.1 夹爪归一化（initial_gripper_state）

模板可声明 `initial_gripper_state`（`open` / `closed` / `hold` / `none`），使技能
执行前显式归一化夹爪状态。例如 `wave_hello` 配置 `initial_gripper_state: closed`
后，实际 primitive 序列会自动在最前面插入一条 `close_gripper`。

- 纯夹爪技能（`open_gripper_skill` / `close_gripper_skill`）不应声明该字段，避免冗余。
- 非法值（如 `half_open`）会在 resolve 阶段抛出 `ValueError`。

例如：

| 技能 | primitive 序列 |
| --- | --- |
| `wave_hello` | `close_gripper` -> `move_to_joint_positions(entry)` -> `move_through_joint_positions(joint_waypoints)` -> `move_to_joint_positions(return)` |
| `recover_zero_pose` | `move_to_named_pose(zero)` |
| `move_relative_ee` | `move_relative_ee(direction, distance)` |
| `dance_basic` | `close_gripper` -> `move_to_joint_positions(entry)` -> `move_through_joint_positions(joint_waypoints)` |
| `celebrate` | `close_gripper` -> `move_to_named_pose(observe_table)` -> 多个 `move_relative_ee` -> `move_to_named_pose(observe_table)` |

`pick_object` 不展开静态 `named_targets` 位姿，而是委托给
`/manipulation/execute_pick`。GraspGen 在运行时生成动态 6-DOF 候选，执行器再通过安全 primitive
完成 approach、补偿下降、夹爪闭合和 lift。

## 5. 直接控制机械臂的几种用法

### 5.1 用自然语言控制

自然语言控制走的是：

- topic：`/voice_command`
- 类型：`std_msgs/msg/String`

标准发送方式如下：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '夹爪往前一点'}"
```

把 `data` 里的中文自然语言替换成不同指令即可。

当前 README 推荐使用下面这些自然语言输入：

| 自然语言输入 | 典型效果 | 对应技能 |
| --- | --- | --- |
| `观察桌面` | 移动到桌面观察位 | `inspect_scene` |
| `看看桌面` | 移动到桌面观察位 | `inspect_scene` |
| `查看桌面` | 移动到桌面观察位 | `inspect_scene` |
| `观察场景` | 移动到桌面观察位 | `inspect_scene` |
| `夹爪往前一点` | 末端向前相对移动一步 | `move_relative_ee` |
| `夹爪往左一点` | 末端向左相对移动一步 | `move_relative_ee` |
| `夹爪往上一点` | 末端向上相对移动一步 | `move_relative_ee` |
| `回原位` | 回到安全位 / home 位 | `recover_safe_pose` |
| `挥挥手` | 执行腕部挥手动作 | `wave_hello` |
| `庆祝一下` | 在观察位执行庆祝动作 | `celebrate` |
| `开心转圈` | 执行直立旋转动作 | `happy_spin_upright` |

观察、回位、夹爪和参数化移动保留基础规则；机器人 YAML 中满足
`description.rule_entry: true` 且 `requires_motion_params: false` 的技能，还会通过
`skill_aliases_json` 把 `aliases_zh` 注入规则解析器。实际可用同义词以当前机器人 YAML 为准。

例如：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '夹爪往前一点'}"

source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '回原位'}"
```

### 5.2 直接发技能 action

适合调试 skill 级执行，不经过自然语言解析。

动作接口：

- `/embodied/execute_skill`
- 类型：`ibrobot_msgs/action/SkillCommand`

常用字段：

| 字段 | 说明 |
| --- | --- |
| `task_id` | 任务 ID |
| `skill_name` | 技能名，如 `pick_object` |
| `target_name` | 目标引用；对 `pick_object` 表示运行时视觉文本查询，如 `banana` |
| `place_name` | 命名放置位，如 `tray_right` |
| `motion_direction` | 相对运动方向 |
| `motion_distance` | 相对运动距离 |
| `timeout_sec` | 技能超时 |

例如挥手：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_skill ibrobot_msgs/action/SkillCommand \
'{
  task_id: "demo-wave",
  skill_name: "wave_hello",
  target_name: "",
  place_name: "",
  motion_direction: "",
  motion_distance: 0.0,
  timeout_sec: 15.0
}'
```

例如让末端向前移动 3 cm：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_skill ibrobot_msgs/action/SkillCommand \
'{
  task_id: "demo-move-forward",
  skill_name: "move_relative_ee",
  target_name: "",
  place_name: "",
  motion_direction: "forward",
  motion_distance: 0.03,
  timeout_sec: 15.0
}'
```

### 5.3 直接发 primitive action

适合最低层调试。

动作接口：

- `/embodied/execute_primitive`
- 类型：`ibrobot_msgs/action/PrimitiveCommand`

例如直接去某个命名位姿：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_primitive ibrobot_msgs/action/PrimitiveCommand \
'{
  task_id: "demo-home",
  primitive_name: "move_to_named_pose",
  pose_name: "home",
  relative_dx: 0.0,
  relative_dy: 0.0,
  relative_dz: 0.0,
  gripper_position: 0.0,
  timeout_sec: 10.0
}'
```

例如让末端向前移动 3 cm：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_primitive ibrobot_msgs/action/PrimitiveCommand \
'{
  task_id: "demo-forward",
  primitive_name: "move_relative_ee",
  pose_name: "",
  relative_dx: 0.03,
  relative_dy: 0.0,
  relative_dz: 0.0,
  gripper_position: 0.0,
  timeout_sec: 10.0
}'
```

例如直接张开夹爪：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_primitive ibrobot_msgs/action/PrimitiveCommand \
'{
  task_id: "demo-open",
  primitive_name: "open_gripper",
  pose_name: "",
  relative_dx: 0.0,
  relative_dy: 0.0,
  relative_dz: 0.0,
  gripper_position: 1.0,
  timeout_sec: 5.0
}'
```

例如直接执行一段关节路点轨迹：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_primitive ibrobot_msgs/action/PrimitiveCommand \
'{
  task_id: "demo-joint-waypoints",
  primitive_name: "move_through_joint_positions",
  pose_name: "",
  relative_dx: 0.0,
  relative_dy: 0.0,
  relative_dz: 0.0,
  gripper_position: 0.0,
  joint_names: ["1", "2", "3", "4", "5"],
  joint_waypoints: [0.02, 0.54, -0.82, -0.18, 0.02],
  joint_waypoint_count: 1,
  waypoint_duration_sec: 0.5,
  timeout_sec: 10.0
}'
```

## 6. primitive 如何桥接到底层

### 位姿、旋转与夹爪控制

以下 primitive 会被转换为单步 `ibrobot_msgs/action/ExecuteTaskPlan` goal，并发送到
`/task_executor/execute_task_plan`：

- `move_to_named_pose`
- `move_relative_ee`
- `rotate_gripper_cw`
- `rotate_gripper_ccw`
- `open_gripper`
- `close_gripper`

`task_dispatch` 负责继续路由到底层执行接口。`SkillExecutorNode` 当前不直接向
`/cmd_pose` 或夹爪控制 topic 发布这些动作。

### 手臂关节轨迹控制

`move_to_joint_positions` / `move_through_joint_positions` 最终都会发送 action 到：

- `/arm_trajectory_controller/follow_joint_trajectory`

消息类型：

- `control_msgs/action/FollowJointTrajectory`

关节名顺序来自 `robot_config` 注入的 `arm_joint_names_json`，关节限位由
`joint_limits_json` 传给安全层校验。

## 7. 相对移动语义

`move_relative_ee` 的方向语义由 `robot_config` 提供，当前默认是 base 坐标系：

| 中文语义 | 方向 | 默认增量方向 |
| --- | --- | --- |
| 前 | `forward` | `+x` |
| 后 | `backward` | `-x` |
| 左 | `left` | `+y` |
| 右 | `right` | `-y` |
| 上 | `up` | `+z` |
| 下 | `down` | `-z` |

规则解析默认步长来自：

- `embodied.execution.relative_motion_step_m`

当前默认值通常为：

- `0.03 m`

## 8. 安全联动

`skill_library` 不会绕过安全层，但两个 action 入口的校验路径不同：

- 高层 `SkillCommand` 先经过 Gateway 准入，再调用 `/embodied/validate_skill` 校验整个 skill；在父 admission
  有效时，解析出的每个内部 primitive 复用父 `SkillCommand` 的 borrowed lease，经
  `/embodied/execute_primitive`，并在下游 dispatch 前各自调用 `/embodied/validate_primitive`。
- 直接/外部 `PrimitiveCommand` 的 Gateway primitive 准入只检查运动授权、所需控制模式，以及 execution
  lease/busy 状态。对相对移动和夹爪旋转，随后才取得并检查新鲜的 EE state；
  `/embodied/validate_primitive` 再校验该 primitive 的最终 target/request。下游 action/server readiness 和
  已捕获 EE pose 的最终新鲜度在各自实际 send 边界前检查，而不是在校验前使用单一、通用的 readiness gate。
  直接/外部 primitive 不会额外调用 `/embodied/validate_skill`。

如果任一步被拒绝：

1. 当前 skill / primitive 立即停止
2. 不向下发送动作
3. 返回明确的 `error_code` 和 `message`

在调用安全校验之前，Gateway 还要求启动时只读参数 `motion_authorized=true`，并要求实际
`active_control_mode` 等于 SSOT 的 `skill_required_control_mode`。运动授权默认关闭，且只能由
`embodied_pipeline.launch.py` 的操作员 launch argument 注入；机器人 YAML、Agent、CLI 和动态参数
都不能开启或修改授权。未授权请求返回 `MOTION_NOT_AUTHORIZED`，控制模式不匹配返回
`CONTROL_MODE_MISMATCH`，两种情况都不会向 primitive 层发送动作。

### 8.1 Gateway 准入、readiness 与新鲜度

Gateway 先规范化 task ID 与公开 skill payload，再按固定顺序评估：操作员授权、控制模式、现有 root
执行 busy 状态、task budget 和该技能的 runtime readiness。只有通过这些检查后，才会在本 Gateway
进程内原子地查询 ledger、取得 root execution lease，并创建 active ledger 记录；相同 task ID/hash 返回
`DUPLICATE_TASK_ID`，同一 task ID 的不同 hash 返回 `TASK_ID_CONFLICT`。lease 只协调本进程内的
root skill 或外部 primitive，不是 ROS graph 范围的锁。

`SkillCommand` 被拒绝时保留 admission 的原始 `error_code` 和 `message`，不会改写成笼统的安全错误。
典型返回为：`MOTION_NOT_AUTHORIZED: operator authorization is disabled`、
`CONTROL_MODE_MISMATCH: requires <required>, active mode is <active>`、
`SKILL_BUSY: another root execution is active`，以及 `CAPABILITY_NOT_READY` 加以下首个缺失原因：
`validate skill service unavailable`、`task executor action unavailable`、
`arm trajectory action unavailable`、`ee pose unavailable or stale`。

末端位姿回调在状态锁内同时记录消息和 monotonic receipt 时间；相对移动与夹爪旋转会从同一锁内深拷贝
一个原子 snapshot。若该 snapshot 在 primitive 安全校验前已经过期，会以
`CAPABILITY_NOT_READY: ee pose unavailable or stale` 返回，既不调用 `ValidatePrimitive`，也不向下游
发送动作。若它在校验或 task executor readiness 等待期间过期，发送前会再次检查，仍在下游 dispatch 前
失败。Gateway 不会对此类失败或任何动作失败自动重试。

### 8.2 UUID 维度

Gateway 涉及两类 UUID，分工不同，不要混淆：

- **task identifier（UUIDv5）**：`embodied_common.skill_request.skill_goal_uuid(task_id)` 派生
  `ibrobot:{task_id}` 的 UUIDv5，用于 ledger key、payload hash 关联，以及 `cancel --task-id` 的
  `CancelGoal.goal_id`。CLI 的 `execute` 与 `cancel` 共用同一 task ID 的 UUIDv5。
- **internal primitive ROS action goal_id（UUIDv4）**：`skill_executor_node` 在派发内部 primitive 时
  用 `uuid.uuid4()` 随机生成 ROS action `goal_id.uuid`，仅用于 rclpy 跟踪 `goal_handle`，不暴露给
  Agent，也不参与 ledger 关联。随机 UUID 避免 goal ID 冲突，是 rclpy 的实现细节。

因此 PR 自述的「确定性 goal UUID」专指 task identifier 维度；internal primitive 的 ROS action
goal_id 不在此约定范围内。

## 9. 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `skill_action_name` | `/embodied/execute_skill` | 技能 action 名 |
| `primitive_action_name` | `/embodied/execute_primitive` | primitive action 名 |
| `validate_skill_service` | `/embodied/validate_skill` | 技能校验服务 |
| `validate_primitive_service` | `/embodied/validate_primitive` | primitive 校验服务 |
| `skill_gateway_status_service` | `/embodied/get_skill_gateway_status` | 启动时固定；Gateway 状态服务名 |
| `robot_name` | `unknown` | 启动时固定；capability view 中的机器人名称 |
| `motion_authorized` | `false` | 启动时固定；唯一运动授权值，默认拒绝运动 |
| `active_control_mode` | 空字符串 | 启动时固定；launch override 生效后的实际控制模式 |
| `skill_required_control_mode` | 空字符串 | 启动时固定；SSOT 要求的技能控制模式 |
| `named_poses_json` | `{}` | 命名位姿字典 |
| `named_targets_json` | `{}` | 命名目标字典 |
| `skill_templates_json` | 空字符串 | 省略、空字符串、仅空白字符或显式 `{}` 均表示空模板/能力目录；Gateway 不启用 legacy fallback 模板；非空 JSON 为显式 SSOT |
| `relative_motion_reference_frame` | `base` | 相对运动参考系 |
| `relative_motion_direction_mapping_json` | `{}` | 相对运动方向映射 |
| `rpc_timeout_sec` | `5.0` | 启动时固定；等待校验服务 / primitive action 的统一 RPC 超时，并参与 capability digest |
| `gripper_settle_sec` | `1.5` | 启动时固定；夹爪 goal 接受等待预算的组成部分，并参与 capability digest |
| `default_skill_timeout_sec` | `30.0` | 启动时固定；Gateway skill 默认超时，并参与 capability digest |
| `task_budget_sec` | `180.0` | 启动时固定；Gateway task 预算，并参与 capability digest |
| `robot_state_freshness_sec` | `0.5` | 启动时固定；机器人状态新鲜度阈值，并参与 capability digest |
| `scene_freshness_sec` | `0.5` | 启动时固定；场景新鲜度阈值，并参与 capability digest |
| `model_idle_timeout_sec` | `120.0` | 启动时固定；模型空闲超时，并参与 capability digest |
| `gripper_open_position` | `1.0` | 张开值 |
| `gripper_closed_position` | `0.0` | 闭合值 |
| `arm_joint_names_json` | `[]` | 手臂关节名顺序 |
| `joint_limits_json` | `{}` | 关节限位配置 |
| `arm_trajectory_action_name` | `/arm_trajectory_controller/follow_joint_trajectory` | 手臂轨迹 action 名 |
| `task_executor_action_name` | `/task_executor/execute_task_plan` | task_dispatch 执行动作名 |
| `pick_action_name` | `/manipulation/execute_pick` | 委托型抓取技能 action 名 |
| `move_configuration_service` | `/moveit_gateway/move_to_configuration` | 精确 IK 配置 MoveIt 服务 |
| `ee_pose_topic` | `/robot_status/ee_pose` | 末端位姿反馈 topic |
| `joint_state_topic` | `/joint_states` | 关节状态反馈 topic |
| `debug_tracing` | `false` | 是否输出调试日志 |

### 取消终态契约

父 `SkillCommand` 请求取消后，执行器会取消其直接 child primitive，并分别在最多
`rpc_timeout_sec` 内确认 child cancel response 和 child result 终态；即使 goal 在取消请求后才被接受，
也会继续取消并 drain。

- 仅当两项确认都完成时，SkillCommand 才返回 `SKILL_CANCELLED` 并进入 canceled 终态。
- send goal/result 状态未知、cancel response 或 terminal 超时、以及下游 primitive 清理状态未知时，
  SkillCommand 会 abort 并返回 `SKILL_CANCEL_TIMEOUT`。对应 Gateway ledger 终态和 audit terminal
  `error_code` 也保持为 `SKILL_CANCEL_TIMEOUT`。
- PrimitiveCommand 的底层清理信号仍可使用 `CANCEL_CLEANUP_TIMEOUT`，但不会作为 SkillCommand 的
  公共错误码暴露。

### SkillCommand feedback

SkillCommand 每个 primitive 步骤都发布 `state="executing"`，并使用 `step <current> of <total>` 作为
`detail`。反馈不包含 primitive 名称、pose、关节或夹爪目标；PrimitiveCommand 自己的内部 feedback 不受此限制。

## 10. 当前限制

- `move_to_named_pose` / `move_relative_ee` 现在会等待 `/robot_status/ee_pose` 收敛到目标附近，而不是只做固定 sleep
- gripper primitive 的 goal 接受等待预算使用 `gripper_settle_sec + 3.0`，执行结果等待上限至少为 15 秒
- 关节 primitive 要求命令完整手臂关节列表，并依赖安全层做关节限位检查
- 技能集合仍是有限白名单，不支持任意自由组合动作
- 使用 `target_pose_key` 的模板依赖 `named_targets_json` 预先配置对应位姿键
- `pick_object` 已接入 GraspGen、IK/FK 接触补偿、候选重试和抓后验证，但仍不包含连续视觉伺服或力控。
- `move_to_configuration` 底层仍是同步服务，不能提供 action 级运动硬取消。
