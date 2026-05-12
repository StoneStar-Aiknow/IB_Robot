# skill_library 节点说明

`skill_library` 是当前具身执行链路里的**技能执行层**。  
它不直接做任务理解，而是负责把上层给出的技能请求，拆成有限的 primitive，并桥接到底层机械臂和夹爪控制接口。

当前包内包含 1 个 ROS 2 节点：

| 节点 | 控制台入口 | 主要职责 |
| --- | --- | --- |
| `skill_executor_node` | `skill_executor_node = skill_library.skill_executor_node:main` | 提供技能 action、primitive action，并把技能执行到 `/cmd_pose` 和夹爪控制接口 |

## 1. 现在可以如何控制机械臂

当前有 3 种主要控制方式，都是通过 `skill_library` 最终落到真实执行：

| 控制方式 | 入口 | 适合场景 |
| --- | --- | --- |
| 自然语言任务 | `/voice_command` | 直接说“把香蕉夹起来”“把夹爪移动到香蕉上方” |
| 技能级控制 | `/embodied/execute_skill` | 明确指定技能名，做稳定、可控的动作编排 |
| primitive 级控制 | `/embodied/execute_primitive` | 直接控制命名位姿、相对位移、夹爪开合 |

整体链路如下：

```text
/voice_command
  -> task_entry_node
  -> task_planner_node / vlm_task_planner_node
  -> task_executor_node
  -> /embodied/execute_skill
  -> skill_executor_node
  -> /embodied/execute_primitive
  -> /cmd_pose or /gripper_position_controller/commands
```

## 2. 当前支持的技能

当前技能由 `robot_config` 中的 `skill_templates` 统一配置，默认支持：

| 技能名 | 作用 | 典型输入 |
| --- | --- | --- |
| `inspect_scene` | 移动到观察位看桌面 | “观察桌面” |
| `observe_target_area` | 移动到目标观察位 | “观察香蕉” |
| `approach_named_target` | 接近目标预抓取位 | “靠近香蕉” |
| `hover_named_target` | 移动到目标上方悬停位 | “移动到香蕉上方” |
| `pick_named_target` | 预抓取 -> 抓取 -> 闭合夹爪 -> 抬起 | “把香蕉夹起来” |
| `lift_named_target` | 把已抓取目标抬起 | “把香蕉抬起来” |
| `retreat_from_target` | 从目标位置后撤 | “从香蕉旁边后撤” |
| `place_named_pose` | 移动到放置位并张开夹爪 | “放到右侧托盘” |
| `release_at_named_pose` | 移动到指定放置位并释放 | “在右侧托盘松开夹爪” |
| `recover_safe_pose` | 回到安全位 / home 位 | “回原位” |
| `move_relative_ee` | 末端沿 base 坐标系相对移动 | “夹爪往前一点” |
| `open_gripper_skill` | 单独张开夹爪 | 上层显式下发 |
| `close_gripper_skill` | 单独闭合夹爪 | 上层显式下发 |

## 3. 当前支持的 primitive

`skill_library` 只允许有限 primitive，避免上层直接下发任意危险动作：

| primitive | 作用 |
| --- | --- |
| `move_to_named_pose` | 移动到命名位姿 |
| `move_relative_ee` | 相对当前末端位姿做笛卡尔增量移动 |
| `open_gripper` | 张开夹爪 |
| `close_gripper` | 闭合夹爪 |

## 4. 技能到 primitive 的映射方式

当前不是硬编码大分支，而是**模板驱动**：

- 模板来源：`robot_config.config.robots.<robot>.yaml`
- 运行时参数：`skill_templates_json`
- 目标相关位姿来自：`named_targets_json`
- 全局命名位姿来自：`named_poses_json`

例如：

| 技能 | primitive 序列 |
| --- | --- |
| `inspect_scene` | `move_to_named_pose(observe_table)` |
| `observe_target_area` | `move_to_named_pose(target.observe_pose)` |
| `approach_named_target` | `move_to_named_pose(target.pregrasp_pose)` |
| `hover_named_target` | `move_to_named_pose(target.hover_pose)` |
| `pick_named_target` | `move_to_named_pose(target.pregrasp_pose)` -> `move_to_named_pose(target.grasp_pose)` -> `close_gripper` -> `move_to_named_pose(target.lift_pose)` |
| `retreat_from_target` | `move_to_named_pose(target.retreat_pose)` |
| `place_named_pose` | `move_to_named_pose(place_name)` -> `open_gripper` |
| `move_relative_ee` | `move_relative_ee(direction, distance)` |

## 5. 直接控制机械臂的几种用法

### 5.1 用自然语言控制

自然语言控制走的是：

- topic：`/voice_command`
- 类型：`std_msgs/msg/String`

标准发送方式如下：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '把夹爪往靠近香蕉的方向移动'}"
```

把 `data` 里的中文自然语言替换成不同指令即可。

当前 README 推荐使用下面这些自然语言输入：

| 自然语言输入 | 典型效果 | 对应技能 |
| --- | --- | --- |
| `观察桌面` | 移动到桌面观察位 | `inspect_scene` |
| `看看桌面` | 移动到桌面观察位 | `inspect_scene` |
| `查看桌面` | 移动到桌面观察位 | `inspect_scene` |
| `观察场景` | 移动到桌面观察位 | `inspect_scene` |
| `观察香蕉` | 移动到香蕉观察位 | `observe_target_area` |
| `看看香蕉` | 移动到香蕉观察位 | `observe_target_area` |
| `查看香蕉` | 移动到香蕉观察位 | `observe_target_area` |
| `把夹爪移动到香蕉的上面` | 移动到香蕉上方悬停位 | `hover_named_target` |
| `把夹爪往靠近香蕉的方向移动` | 接近香蕉预抓取位 | `approach_named_target` |
| `把香蕉夹起来` | 执行抓取并抬起 | `pick_named_target` |
| `把香蕉抬起来` | 抬起目标 | `lift_named_target` |
| `从香蕉旁边后撤` | 从目标区域撤回 | `retreat_from_target` |
| `把香蕉放到右侧托盘` | 移动到托盘并释放 | `place_named_pose` |
| `在右侧托盘松开夹爪` | 在托盘位置松开夹爪 | `release_at_named_pose` |
| `夹爪往前一点` | 末端向前相对移动一步 | `move_relative_ee` |
| `夹爪往左一点` | 末端向左相对移动一步 | `move_relative_ee` |
| `夹爪往上一点` | 末端向上相对移动一步 | `move_relative_ee` |
| `回原位` | 回到安全位 / home 位 | `recover_safe_pose` |

当前规则解析里已经明确支持的同义表达包括：

```text
观察桌面
看看桌面
查看桌面
扫描桌面
观察场景
看看场景
观察香蕉
看看香蕉
查看香蕉
观察目标区域
看看目标区域
把夹爪移动到香蕉的上面
把夹爪往靠近香蕉的方向移动
把香蕉抬起来
从香蕉旁边后撤
把香蕉放到右侧托盘
在右侧托盘松开夹爪
夹爪往前一点
夹爪往左一点
夹爪往上一点
回原位
```

例如：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '观察香蕉'}"

source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '把夹爪移动到香蕉的上面'}"

source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '把香蕉放到右侧托盘'}"
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
| `skill_name` | 技能名 |
| `target_name` | 命名目标，如 `banana` |
| `place_name` | 命名放置位，如 `tray_right` |
| `motion_direction` | 相对运动方向 |
| `motion_distance` | 相对运动距离 |
| `timeout_sec` | 技能超时 |

例如悬停到香蕉上方：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_skill ibrobot_msgs/action/SkillCommand \
'{
  task_id: "demo-hover-banana",
  skill_name: "hover_named_target",
  target_name: "banana",
  place_name: "",
  motion_direction: "",
  motion_distance: 0.0,
  timeout_sec: 15.0
}'
```

例如后撤：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
ros2 action send_goal /embodied/execute_skill ibrobot_msgs/action/SkillCommand \
'{
  task_id: "demo-retreat-banana",
  skill_name: "retreat_from_target",
  target_name: "banana",
  place_name: "",
  motion_direction: "",
  motion_distance: 0.0,
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

## 6. primitive 如何桥接到底层

### 位姿控制

`move_to_named_pose` 和 `move_relative_ee` 最终都会发布到：

- `/cmd_pose`

该接口由 `robot_moveit` 中的 `moveit_gateway` 消费。

### 夹爪控制

`open_gripper` / `close_gripper` 最终都会发布到：

- `/gripper_position_controller/commands`

消息类型：

- `std_msgs/msg/Float64MultiArray`

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

`skill_library` 不会绕过安全层。  
每次执行前都会调用：

- `/embodied/validate_skill`
- `/embodied/validate_primitive`

如果任一步被拒绝：

1. 当前 skill / primitive 立即停止
2. 不向下发送动作
3. 返回明确的 `error_code` 和 `message`

## 9. 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `skill_action_name` | `/embodied/execute_skill` | 技能 action 名 |
| `primitive_action_name` | `/embodied/execute_primitive` | primitive action 名 |
| `validate_skill_service` | `/embodied/validate_skill` | 技能校验服务 |
| `validate_primitive_service` | `/embodied/validate_primitive` | primitive 校验服务 |
| `named_poses_json` | `{}` | 命名位姿字典 |
| `named_targets_json` | `{}` | 命名目标字典 |
| `skill_templates_json` | `{}` | 技能模板字典 |
| `relative_motion_reference_frame` | `base` | 相对运动参考系 |
| `relative_motion_direction_mapping_json` | `{}` | 相对运动方向映射 |
| `rpc_timeout_sec` | `5.0` | 等待校验服务 / primitive action 的统一 RPC 超时 |
| `gripper_settle_sec` | `1.5` | gripper primitive 的稳定等待时间 |
| `gripper_open_position` | `1.0` | 张开值 |
| `gripper_closed_position` | `0.0` | 闭合值 |
| `debug_tracing` | `true` | 是否输出调试日志 |

## 10. 当前限制

- `move_to_named_pose` / `move_relative_ee` 现在会等待 `/robot_status/ee_pose` 收敛到目标附近，而不是只做固定 sleep
- gripper primitive 仍使用 `gripper_settle_sec` 作为稳定等待时间
- 技能集合仍是有限白名单，不支持任意自由组合动作
- 命名目标依赖 `robot_config` 里预先配置好的 `observe_pose / pregrasp_pose / hover_pose / grasp_pose / lift_pose / retreat_pose`
- 复杂抓取仍未加入视觉伺服、力反馈、碰撞后自恢复或在线重规划
