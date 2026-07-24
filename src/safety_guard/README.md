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
- `joints.arm`
- `teleoperation.safety.joint_limits`

运行时这些内容以 JSON 参数形式传入：

| 参数 | 说明 |
| --- | --- |
| `named_poses_json` | 命名位姿字典 |
| `named_targets_json` | 命名目标字典 |
| `workspace_json` | 工作空间边界 |
| `arm_joint_names_json` | 手臂关节名顺序 |
| `joint_limits_json` | 关节限位字典 |

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
| `arm_joint_names_json` | `[]` | 手臂关节名顺序 |
| `joint_limits_json` | `{}` | 关节限位字典 |
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
| `relative_dx/dy/dz` | 相对位移增量 |
| `target_x/y/z` | 执行层解析出的目标末端位置 |
| `gripper_position` | 夹爪目标开合量 |
| `joint_names` | 关节名列表 |
| `joint_positions` | 单个关节目标位置 |
| `primitive_duration_sec` | 单点关节轨迹持续时间 |
| `joint_waypoints` | 扁平化关节路点序列 |
| `joint_waypoint_count` | 关节路点数量 |
| `waypoint_duration_sec` | 相邻关节路点的时间间隔 |

**响应字段**

| 字段 | 说明 |
| --- | --- |
| `allowed` | 是否允许 |
| `reason` | 不允许时的原因 |

## 5. 当前内置规则

### 技能白名单

技能白名单由 `get_skill_templates()` 选择：

- `skill_templates_json` 非空时，使用当前机器人 YAML 注入并过滤 `disabled: true` 后的启用
  `embodied.skill_templates` 集合
- 未提供机器人模板时，回退到 `embodied_common.skill_templates` 的默认模板

机器人模板非空时替代默认模板，两套模板不会自动合并。请求技能必须存在于当前选中的模板集合中。
安全守卫随后按顺序校验每个 primitive，任何未知技能、未知 primitive 或参数越界都会被拒绝。

### 技能依赖检查

当前会做这些检查：

| 技能 | 检查项 |
| --- | --- |
| `inspect_scene` | 必须存在 `observe_table` 命名位姿 |
| `recover_safe_pose` | 必须存在 `home` 命名位姿 |
| `recover_zero_pose` | 必须存在 `zero` 命名位姿 |
| YAML 派生技能 | 必须存在于 `skill_templates_json`，并通过其全部 primitive 校验 |

### 原子动作白名单

当前只允许以下 primitive：

- `move_to_named_pose`
- `move_relative_ee`
- `move_to_joint_positions`
- `move_through_joint_positions`
- `open_gripper`
- `close_gripper`
- `rotate_gripper_cw`
- `rotate_gripper_ccw`

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

#### `move_to_joint_positions`

会检查：

1. `joint_names` 必须存在，且与 `joint_positions` 数量一致
2. 如果配置了 `arm_joint_names_json`，请求必须按完整手臂关节顺序下发
3. `primitive_duration_sec` 必须大于 0
4. 每个关节目标必须落在 `joint_limits_json` 范围内

#### `move_through_joint_positions`

会检查：

1. `joint_names` 必须存在
2. `joint_waypoint_count` 必须大于 0
3. `joint_waypoints` 长度必须等于 `len(joint_names) * joint_waypoint_count`
4. `waypoint_duration_sec` 必须大于 0
5. 每个路点都必须符合手臂关节顺序和关节限位

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
cd ~/IB_Robot
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.sh
ros2 service list | grep embodied
```

也可以手工调用服务验证一条技能：

```bash
ros2 service call /embodied/validate_skill ibrobot_msgs/srv/ValidateSkill \
  "{skill_name: inspect_scene, target_name: '', place_name: ''}"
```

## 8. 当前限制

- 目前只做**白名单 + 工作空间 + 关节限位**级别的静态校验。
- 还没有碰撞检测、动力学约束、实时状态联动等更复杂安全能力。
- 也没有引入独立安全状态机；当前是同步请求-响应式安全判断。
