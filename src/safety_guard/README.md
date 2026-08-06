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

- `skill_library` 在执行 `SkillCommand` 前，会先调用 `/embodied/validate_skill`（请求携带 `dispatch_binding`）
- `skill_library` 在执行 `PrimitiveCommand` 前，会先调用 `/embodied/validate_primitive`（请求携带 `dispatch_binding` 和 `dispatch_nonce`）
- 校验失败会直接中止当前 skill 或 primitive，而不是静默跳过
- validate 是只读 preflight，不查询 coordinator；exact snapshot、nonce、digest、step payload 的权威校验
  在后续 Gateway action admission 完成

## 2. 配置来源：exact snapshot synchronization

`safety_guard` 不再从 `robot_config` 注入的 `skill_templates_json` 读取模板，也不维护独立 YAML。
它通过 **exact registry snapshot synchronization** 从 `skill_library` Gateway 拉取与当前 runtime 完全
一致的 verified snapshot，校验在本地内存中完成，**不执行任何 ROS I/O**。

同步机制（均在节点内异步完成，不阻塞 validate 调用）：

1. **`SkillRegistryEvent` 订阅**（`/embodied/skill_registry_events`，QoS 固定为
   RELIABLE / TRANSIENT_LOCAL / KEEP_LAST / depth 1）：成功 reload 后由 Gateway 发布，告知晚加入者
   应查询哪个 `(epoch, new_generation, registry_digest)`。收到事件后立即请求对应 snapshot。
2. **`GetSkillGatewayStatus` 周期轮询**（`/embodied/get_skill_gateway_status`，周期
   `snapshot_sync_period_sec`，默认 `1.0`）：获取当前 identity 和 `retained_generations`，对每个 retained
   generation 请求缺失的 snapshot，并 `reconcile` 本地缓存。缓存只保留 Gateway 明确声明的
   `retained_generations` 和 current generation，不自行延长已释放 generation 的安全可用期。
3. **`GetSkillSnapshot` 拉取**（`/embodied/get_skill_snapshot`）：按 `(epoch, generation)` 精确查询，
   `generation>0` 必须精确匹配，不静默升级到更新版本；已被回收返回 `SKILL_SNAPSHOT_NOT_RETAINED`。

拉取到的 `snapshot_json` 进入 `SafetySnapshotCache` 前必须通过完整校验：必须是 canonical JSON、
`schema_version=1`、字段集严格等于 `snapshot_payload_v1`；本地重算 `registry_digest`、`capability_digest`、
`provenance_digest` 必须与响应一致；`primitive_contract_digest` 必须等于本地 SSOT
`PRIMITIVE_CONTRACT_DIGEST`；skill 名称必须唯一非空。任一校验失败以 `SKILL_SNAPSHOT_DIGEST_MISMATCH`
或 `SKILL_SCHEMA_INVALID` 拒绝并丢弃，不更新 current 指针。

校验时 robot context（`named_poses`、`named_targets`、`workspace_limits`、`arm_joint_names`、`joint_limits`）
和 `skill_templates` 全部来自 verified snapshot，不再来自节点启动参数。启动参数 `named_poses_json` 等
仍声明并加载到实例属性，但 validate 处理器不再读取它们。`skill_templates_json` 参数已移除。

## 3. safety_guard_node

### 作用

1. 暴露技能级只读 preflight 校验服务。
2. 暴露原子动作级只读 preflight 校验服务。
3. 在已验证的 exact snapshot 上做白名单 + 工作空间 + 关节限位静态校验。
4. 维护 `SafetySnapshotCache`，按 `(epoch, generation)` 索引 verified snapshot，并随 Gateway 同步。
5. 在 `debug_tracing=true` 时打印每次校验的决策日志。

### 主要参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `validate_skill_service` | `/embodied/validate_skill` | 技能校验服务名 |
| `validate_primitive_service` | `/embodied/validate_primitive` | 原子动作校验服务名 |
| `skill_gateway_status_service` | `/embodied/get_skill_gateway_status` | Gateway 状态服务名（轮询同步） |
| `skill_catalog_snapshot_service` | `/embodied/get_skill_snapshot` | exact snapshot 查询服务名 |
| `skill_registry_event_topic` | `/embodied/skill_registry_events` | registry 事件 topic |
| `snapshot_sync_period_sec` | `1.0` | 状态轮询周期，必须为正 |
| `named_poses_json` | `{}` | 启动时加载，不再驱动 validate（snapshot `robot_context` 为准） |
| `named_targets_json` | `{}` | 启动时加载，不再驱动 validate |
| `workspace_json` | `{}` | 启动时加载，不再驱动 validate |
| `arm_joint_names_json` | `[]` | 启动时加载，不再驱动 validate |
| `joint_limits_json` | `{}` | 启动时加载，不再驱动 validate |
| `debug_tracing` | `false` | 是否输出调试日志 |

## 4. 当前 ROS 服务接口

### `/embodied/validate_skill`

类型：`ibrobot_msgs/srv/ValidateSkill`。只读 preflight：校验 exact snapshot、entry 参数和当前机器人状态，
但不查询或修改 coordinator，也不得把存在 `root_lease_nonce` 解释为执行授权。Workflow root、nonce、
digest、期望 index 和完整 step payload 的权威校验只在后续 Gateway action admission 发生，因此 validation
通过不保证 admission 成功。

**请求字段**

| 字段 | 说明 |
| --- | --- |
| `dispatch_binding` | `DispatchBinding`，必须携带完整期望 registry identity；`root_lease_nonce` 仅用于关联，不视为授权；`dispatch_nonce` 必须为空 |
| `skill_name` | 待执行技能名 |
| `target_name` | 命名目标 |
| `place_name` | 命名放置位 |
| `motion_direction` | 相对运动方向 |
| `motion_distance` | 相对运动距离 |

**响应字段**

| 字段 | 说明 |
| --- | --- |
| `allowed` | 是否允许 |
| `reason` | 不允许时的原因 |
| `error_code` | 稳定错误码 |
| `actual_registry_epoch` | 实际使用的 registry epoch |
| `actual_registry_generation` | 实际使用的 registry generation |
| `actual_registry_digest` | 实际使用的 registry digest |
| `diagnostics` | `SkillDiagnostic[]` 结构化诊断 |

### `/embodied/validate_primitive`

类型：`ibrobot_msgs/srv/ValidatePrimitive`。请求必须携带 exact registry identity 和非空 `dispatch_nonce`。

**请求字段**

| 字段 | 说明 |
| --- | --- |
| `dispatch_binding` | `DispatchBinding`，必须携带完整期望 registry identity 和非空 `dispatch_nonce` |
| `primitive_name` | 原子动作名 |
| `pose_name` | 命名位姿名 |
| `relative_dx/dy/dz` | 相对位移增量 |
| `target_x/y/z` | 执行层解析出的目标末端位置 |
| `target_qx/qy/qz/qw` | 执行层解析出的目标末端姿态四元数 |
| `velocity_scaling` | MoveIt 速度比例 |
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
| `error_code` | 稳定错误码 |
| `actual_registry_epoch` | 实际使用的 registry epoch |
| `actual_registry_generation` | 实际使用的 registry generation |
| `actual_registry_digest` | 实际使用的 registry digest |
| `diagnostics` | `SkillDiagnostic[]` 结构化诊断 |

### Fail-closed 行为

两个 validate 服务均 fail closed：

- `dispatch_binding.schema_version != 1` 或期望 identity 不完整（epoch 空、generation<=0、digest 空）：
  `allowed=false`、`error_code=SKILL_SCHEMA_INVALID`，`actual_*` 取当前缓存 identity（可能为空）。
  `ValidatePrimitive` 还要求 `dispatch_nonce` 非空。
- exact snapshot 未缓存（已被回收）：`error_code=SKILL_SNAPSHOT_NOT_RETAINED`。
- 缓存的 snapshot digest 与请求期望不一致：`error_code=SKILL_REGISTRY_VERSION_MISMATCH`。
- snapshot payload 校验失败：`error_code=SKILL_SCHEMA_INVALID` 或 `SKILL_SNAPSHOT_DIGEST_MISMATCH`。
- 校验通过且白名单/工作空间/关节限位全部满足：`allowed=true`、`error_code=""`。
- 校验通过但规则不满足：`allowed=false`，skill 返回 `SKILL_LIMIT_VIOLATION`，primitive 返回
  `PRIMITIVE_LIMIT_VIOLATION`。
- 任一异常被捕获后返回 `allowed=false` 和内部错误消息，绝不静默放行。

## 5. 当前内置规则

### 技能白名单

技能白名单来自 verified snapshot 的 `templates`（由 `SkillCatalogCompiler` 编译）。请求技能必须存在于
当前选中的 snapshot 模板集合中。安全守卫随后按顺序校验每个 primitive，任何未知技能、未知 primitive 或
参数越界都会被拒绝。不存在「默认模板回退」路径——snapshot 未就绪时 validate 直接 fail closed。

### 技能依赖检查

当前会做这些检查：

| 技能 | 检查项 |
| --- | --- |
| `inspect_scene` | 必须存在 `observe_table` 命名位姿 |
| `recover_safe_pose` | 必须存在 `home` 命名位姿 |
| `recover_zero_pose` | 必须存在 `zero` 命名位姿 |
| catalog 派生技能 | 必须存在于当前 verified snapshot `templates`，并通过其全部 primitive 校验 |

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

1. `pose_name` 是否存在（snapshot `robot_context.named_poses`）
2. 位姿是否包含 `position.x/y/z`
3. 位姿是否落在 `workspace_limits` 指定边界内

#### `move_relative_ee`

会检查：

1. 相对位移增量不能全为 0
2. 由执行层计算得到的目标位姿是否仍落在 `workspace_limits` 内
3. 技能层传下来的方向和步长是否合法

如果机械臂当前已经贴近边界，则部分方向会被拒绝；`workspace_limits` 需要覆盖机器人默认启动姿态，否则第一条相对移动也会被直接拦截。

#### `open_gripper` / `close_gripper`

会检查：

- `gripper_position` 是否位于 `[0.0, 1.0]`

#### `move_to_joint_positions`

会检查：

1. `joint_names` 必须存在，且与 `joint_positions` 数量一致
2. 如果 snapshot `robot_context.arm_joint_names` 非空，请求必须按完整手臂关节顺序下发
3. `primitive_duration_sec` 必须大于 0
4. 每个关节目标必须落在 `robot_context.joint_limits` 范围内

#### `move_through_joint_positions`

会检查：

1. `joint_names` 必须存在
2. `joint_waypoint_count` 必须大于 0
3. `joint_waypoints` 长度必须等于 `len(joint_names) * joint_waypoint_count`
4. `waypoint_duration_sec` 必须大于 0
5. 每个路点都必须符合手臂关节顺序和关节限位

## 6. 工作空间定义

`workspace_limits`（snapshot `robot_context` 中）使用简单的轴对齐范围：

```yaml
workspace_limits:
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

`validate_skill` / `validate_primitive` 请求必须携带完整 `dispatch_binding`（exact registry identity），
手工构造较繁琐。推荐使用 `robot-skill validate`，CLI 会查询 `GetSkillGatewayStatus` 取回当前 identity
并构造只读 preflight 请求：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && \
robot-skill validate inspect_scene
```

也可以监听 registry 事件观察 snapshot 同步：

```bash
ros2 topic echo /embodied/skill_registry_events
```

## 8. 当前限制

- 目前只做**白名单 + 工作空间 + 关节限位**级别的静态校验。
- 还没有碰撞检测、动力学约束、实时状态联动等更复杂安全能力。
- 也没有引入独立安全状态机；当前是同步请求-响应式安全判断。
- validate 是只读 preflight，不查询或修改 coordinator；通过 validate 不保证后续 Gateway action admission
  成功。
