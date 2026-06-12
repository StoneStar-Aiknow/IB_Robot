# safety_guard 节点说明

`safety_guard` 是当前最小具身闭环中的**显式安全校验包**。
它不负责规划和执行，而是把“当前这一步是否允许执行”独立成可调用服务，避免把安全判断散落在 planner 或 executor 里。

当前包内包含 1 个 ROS 2 节点：

| 节点 | 控制台入口 | 主要职责 |
| --- | --- | --- |
| `safety_guard_node` | `safety_guard_node = safety_guard.safety_guard_node:main` | 对技能请求和原子动作请求做白名单与工作空间校验 |

## 1. 在最小闭环中的位置

当前闭环中的安全位置如下：

```text
Task / Skill request
  -> safety_guard
  -> allow / reject
  -> skill_library 才继续执行
```

更具体地说：

- `skill_library` 在执行 `SkillCommand` 前，会先调用 `/embodied/validate_skill`
- `skill_library` 在执行 `PrimitiveCommand` 前，会先调用 `/embodied/validate_primitive`
- 校验失败会直接中止当前 skill 或 primitive，而不是静默跳过

## 2. 配置来源

`safety_guard` 不自己维护独立 YAML。
它读取的是 `robot_config` 注入的 `robot.embodied` 相关配置，主要包括：

- `named_poses`
- `named_targets`
- `safety.workspace`

运行时这些内容以 JSON 参数形式传入：

| 参数 | 说明 |
| --- | --- |
| `named_poses_json` | 命名位姿字典 |
| `named_targets_json` | 命名目标字典 |
| `workspace_json` | 工作空间边界 |

这保证了安全规则仍然受 `robot_config` 统一管理。

## 3. safety_guard_node

### 作用

1. 暴露技能级校验服务。
2. 暴露原子动作级校验服务。
3. 对当前最小闭环中的有限技能和原子动作做白名单限制。
4. 对位姿目标执行工作空间边界检查。
5. 在 `debug_tracing=true` 时打印每次校验的决策日志。

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `validate_skill_service` | `/embodied/validate_skill` | 技能校验服务名 |
| `validate_primitive_service` | `/embodied/validate_primitive` | 原子动作校验服务名 |
| `named_poses_json` | `{}` | 命名位姿 |
| `named_targets_json` | `{}` | 命名目标 |
| `workspace_json` | `{}` | 工作空间边界 |
| `debug_tracing` | `false` | 是否输出调试日志 |

## 4. 当前 ROS 服务接口

### `/embodied/validate_skill`

类型：`ibrobot_msgs/srv/ValidateSkill`

**请求字段**

| 字段 | 说明 |
| --- | --- |
| `skill_name` | 待执行技能名 |
| `target_name` | 命名目标 |
| `place_name` | 命名放置位 |

**响应字段**

| 字段 | 说明 |
| --- | --- |
| `allowed` | 是否允许 |
| `reason` | 不允许时的原因 |

### `/embodied/validate_primitive`

类型：`ibrobot_msgs/srv/ValidatePrimitive`

**请求字段**

| 字段 | 说明 |
| --- | --- |
| `primitive_name` | 原子动作名 |
| `pose_name` | 命名位姿名 |
| `gripper_position` | 夹爪目标开合量 |

**响应字段**

| 字段 | 说明 |
| --- | --- |
| `allowed` | 是否允许 |
| `reason` | 不允许时的原因 |

## 5. 当前内置规则

### 技能白名单

当前只允许以下技能：

- `inspect_scene`
- `pick_named_target`
- `place_named_pose`
- `recover_safe_pose`
- `move_relative_ee`

除此之外一律拒绝。

### 技能依赖检查

当前会做这些检查：

| 技能 | 检查项 |
| --- | --- |
| `inspect_scene` | 必须存在 `observe_table` 命名位姿 |
| `recover_safe_pose` | 必须存在 `home` 命名位姿 |
| `pick_named_target` | `target_name` 必须存在于 `named_targets` |
| `place_named_pose` | `place_name` 必须存在于 `named_poses` |

### 原子动作白名单

当前只允许以下 primitive：

- `move_to_named_pose`
- `move_relative_ee`
- `open_gripper`
- `close_gripper`

### 原子动作边界检查

#### `move_to_named_pose`

会检查：

1. `pose_name` 是否存在
2. 位姿是否包含 `position.x/y/z`
3. 位姿是否落在 `workspace` 指定边界内

#### `move_relative_ee`

会检查：

1. 相对位移增量不能全为 0
2. 由执行层计算得到的目标位姿是否仍落在 `workspace` 内
3. 技能层传下来的方向和步长是否合法

如果机械臂当前已经贴近边界，则部分方向会被拒绝；`workspace` 需要覆盖机器人默认启动姿态，否则第一条相对移动也会被直接拦截。

#### `open_gripper` / `close_gripper`

会检查：

- `gripper_position` 是否位于 `[0.0, 1.0]`

## 6. 工作空间定义

当前 `workspace_json` 使用简单的轴对齐范围：

```yaml
workspace:
  x: [min, max]
  y: [min, max]
  z: [min, max]
```

这是当前最小闭环阶段的保守实现，目的是：

- 避免规划/执行请求引用明显越界的命名位姿
- 让错误在安全层被显式拒绝
- 保持规则足够简单，便于后续升级

## 7. 调试与观察

统一 launch 启动后，可以直接查看安全层日志：

```bash
cd /home/lwh/code/IB_Robot
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.sh
ros2 service list | grep embodied
```

也可以手工调用服务验证一条技能：

```bash
ros2 service call /embodied/validate_skill ibrobot_msgs/srv/ValidateSkill \
  "{skill_name: pick_named_target, target_name: demo_object, place_name: ''}"
```

## 8. 当前限制

- 目前只做**白名单 + 工作空间**级别的静态校验。
- 还没有碰撞检测、动力学约束、实时状态联动等更复杂安全能力。
- 也没有引入独立安全状态机；当前是同步请求-响应式安全判断。
