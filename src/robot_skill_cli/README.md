# robot_skill_cli

`robot_skill_cli` 提供稳定的 `robot-skill` 命令行接口，将 Agent 的技能发现、校验、执行和取消请求限定在
ROS Capability Gateway 的公开边界内。默认 Agent 控制链路为：

```text
Hermes -> ibrobot-control Agent Skill -> robot-skill -> ROS Capability Gateway
```

CLI 不直接调用 primitive、MoveIt、controller 或裸 ROS 运动接口，也不修改 Gateway 的授权和安全策略。

## 环境与配置

在工作区根目录加载环境：

```bash
source .shrc_local
source install/local_setup.sh
```

每个命令可用 `--config-name NAME` 选择配置，或用 `--config-path PATH` 指向 YAML；两个 flag 在 CLI 中互斥。
配置解析完全复用 `robot_config.resolve_robot_config_path()`，CLI 不维护第二套路径优先级：底层选择顺序是
显式 path、显式 name、`ROBOT_CONFIG`、`ROBOT_NAME`、默认 `so101_single_arm`。按名称先查安装目录，再查
源码 `config/robots/`；显式 path 必须存在。

## 命令

| 命令 | ROS | 用途 |
|---|---:|---|
| `list-skills` | 否 | 列出所有启用的高层技能及公开描述 |
| `describe SKILL` | 否 | 查看参数 schema、单位、语义描述和 timeout policy |
| `list-poses` | 否 | 列出公开命名位姿 |
| `status` | 是 | 读取 Gateway 授权、控制模式、readiness、busy 和 ledger 状态 |
| `validate SKILL` | 是 | 本地 schema 检查后调用 Gateway 安全校验，不执行动作 |
| `execute SKILL --task-id ID` | 是 | 通过 `SkillCommand` 执行高层技能 |
| `cancel --task-id ID` | 是 | 以同一 deterministic goal UUID 请求取消并轮询 terminal |
| `reload-catalog --request-id ID --force` | 是 | 重新编译并原子激活 Gateway 已配置的 Skill catalog source |
| `plan-workflow --text TEXT --workflow-json JSON --request-id ID` | 是 | 提交一份短时 typed Agent plan |
| `validate-plan --plan-token TOKEN` | 是 | 对 exact snapshot 计划做只读逐步预检 |
| `confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID [--timeout-sec SEC]` | 是 | 校验身份/摘要/task_id 并冻结 `task_budget_sec`，转入 `CONFIRMED` |
| `execute-plan --plan-token TOKEN --confirmation-token TOKEN --task-id ID [--timeout-sec SEC]` | 是 | 执行已确认的 Agent plan，必须复用 confirm 冻结的同一预算 |
| `cancel-plan --task-id ID` | 是 | 通过 Agent plan action 的标准 CancelGoal 请求取消 |

catalog-only 命令不初始化 `rclpy`，只读取本地归一化配置。runtime 命令只访问 Gateway status、
`ValidateSkill`、`SkillCommand`、Agent plan services/actions 和标准 `CancelGoal` 接口。

```bash
robot-skill --config-name so101_single_arm list-skills
robot-skill --config-name so101_single_arm describe move_relative_ee
robot-skill --config-name so101_single_arm list-poses
robot-skill --config-name so101_single_arm status
robot-skill --config-name so101_single_arm reload-catalog \
  --request-id reload-001 --force

robot-skill --config-name so101_single_arm validate move_relative_ee \
  --motion-direction forward --motion-distance 0.03

robot-skill --config-name so101_single_arm execute move_relative_ee \
  --task-id task-20260725-001 \
  --motion-direction forward --motion-distance 0.03

robot-skill --config-name so101_single_arm cancel \
  --task-id task-20260725-001

robot-skill --config-name so101_single_arm plan-workflow \
  --request-id plan-request-001 --text "打开夹爪" \
  --workflow-json '[{"skill_name":"open_gripper_skill"}]'
robot-skill --config-name so101_single_arm validate-plan \
  --plan-token PLAN_TOKEN
robot-skill --config-name so101_single_arm confirm-plan \
  --plan-token PLAN_TOKEN --plan-digest PLAN_DIGEST --task-id agent-task-001
# 可选 --timeout-sec 30 冻结一份小于等于 Gateway task budget 的预算
robot-skill --config-name so101_single_arm execute-plan \
  --plan-token PLAN_TOKEN --confirmation-token CONFIRMATION_TOKEN --task-id agent-task-001
# execute-plan 必须传 confirm 时使用的同一 --timeout-sec；省略时两端都默认 Gateway task budget
robot-skill --config-name so101_single_arm cancel-plan \
  --task-id agent-task-001
```

可用 typed flags 为 `--target-name`、`--place-name`、`--motion-direction`、`--motion-distance` 和
`--timeout-sec`。CLI 根据当前技能显式 `capability.parameters` schema 拒绝缺失参数和不属于该技能的参数。

### 公开 catalog 字段

`list-skills` 输出 `robot_name`、`config_digest` 和技能数组；每个技能只含 `name`、`summary`、`domain`、
`moves_robot`、`required_control_mode`。`describe SKILL` 在这些字段之外输出该技能的 `parameters`、
`recovery_policy`、完整 `timeout_policy` 和 `config_digest`。`list-poses` 只输出命名位姿名称，不输出坐标。
primitive sequence、目标绑定、关节值和 ROS transport 名称不属于 CLI catalog。

## 调用顺序

传统单技能 Agent 必须按以下顺序工作：

1. `status`
2. `list-skills`
3. `describe SKILL`
4. `validate SKILL`
5. 获取用户明确的运动确认
6. `execute SKILL --task-id ID`

自然语言计划必须按以下顺序工作：

1. `plan-workflow --text TEXT --workflow-json JSON --request-id ID`
2. `validate-plan --plan-token TOKEN`
3. 向用户展示步骤、参数、plan digest 和 registry identity，获取明确确认并生成 task ID
4. `confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID [--timeout-sec SEC]`
5. `execute-plan --plan-token TOKEN --confirmation-token TOKEN --task-id ID [--timeout-sec SEC]`

`confirm-plan` 与 `execute-plan` 共享可选的 `--timeout-sec`：省略时两端都默认使用 Gateway 当前
`task_budget_sec`；显式给出时必须为有限正数且不超过 Gateway task budget。`confirm-plan` 把该值以 float32
冻结进计划，`execute-plan` 必须传入**同一**值（float32 严格相等），否则协调器以 `SKILL_REQUEST_ID_CONFLICT`
拒绝——这是为了防止执行阶段悄悄放大或缩小用户确认过的任务预算。

当前 `execute` 可用 SIGINT/SIGTERM 请求取消；另一个进程可使用 `cancel --task-id ID`。失败、timeout 或
停止状态未知时不得自动重试。`SKILL_CANCEL_TIMEOUT` 表示机器人停止状态未知，不能表述为“已停止”。

`validate` 与 `execute` 先以规范 payload 请求 Gateway：字符串去空白、方向小写，未提供 timeout 时使用
Gateway 默认值；timeout 必须为有限正数且不超过 Gateway task budget。payload hash 是规范 JSON 的 SHA-256。
`execute` 的 ROS goal UUID，以及 `cancel --task-id` 使用的 CancelGoal UUID，都是同一 task ID 的
UUIDv5（`ibrobot:{task_id}`）。`cancel` 只对 ledger 中 `active` task 发取消并轮询 `terminal`；已终态 task
返回 `already_terminal`，未知 task 返回 `GOAL_NOT_FOUND`。

## 输出契约

除 `execute` 和 `execute-plan` 外，命令向 stdout 输出单行 JSON envelope：

```json
{"command":"status","data":{},"error":null,"ok":true,"schema_version":1}
```

`execute` 与 `execute-plan` 输出 JSONL：零到多条 `feedback`，最后恰好一条 `result`。每行都包含 `task_id` 和
`payload_hash`；公开结果只提供 `executed_step_count`，不暴露 primitive、pose 或 joint 名称。

```json
{"data":{"detail":"step 1 of 1","state":"executing"},"event":"feedback","payload_hash":"...","schema_version":1,"task_id":"task-20260725-001"}
{"data":{"error_code":"","executed_step_count":1,"message":"skill completed","success":true},"event":"result","payload_hash":"...","schema_version":1,"task_id":"task-20260725-001"}
```

普通命令错误也使用同一 JSON envelope 的 `error.code` 与 `error.message`。当 Gateway capability reason
为 `CODE: detailed message` 时，CLI 用第一个冒号前的文本作为 `error.code`；`error.message` 保留原始
reason（没有冒号时两者都是该 code）。因此调用方应按 `error.code` 做稳定分类，不要解析详细消息。

| 退出码 | 含义 |
|---:|---|
| `0` | 成功 |
| `2` | 参数、配置或 schema 错误 |
| `3` | Gateway、readiness 或 safety 拒绝 |
| `4` | ROS/Gateway 不可用 |
| `124` | timeout，可能包含停止状态未知 |
| `130` | SIGINT 取消已收敛到 terminal |
| `143` | SIGTERM 取消已收敛到 terminal |

## 授权边界

`authorize_motion` 是唯一运行时运动授权来源，launch 默认值为 `false`。CLI 和 Agent 不能启动或重启
pipeline、开启授权、修改 ROS 参数，或从 YAML 推导授权。只有操作员完成现场安全检查后，才能在启动
pipeline 时显式授权。未授权状态下仍可使用 catalog、`status` 和不执行动作的本地检查；Gateway 会拒绝运动。

`robot_mcp` 兼容层已移除。Hermes 推荐配置和示例统一通过 `robot-skill` 访问 Capability Gateway。

## Hermes 启动器

完整构建后可运行：

```bash
hermes-robot --config-name so101_single_arm
```

启动器要求 Hermes Agent `0.16.0` 或更新版本，并在启动前验证 `hermes`、`robot-skill`、安装空间中的
`ibrobot-control`、目标 robot config、Gateway control-plane status 以及全部 Agent plan service/action。
`motion_authorized=false` 不阻止 Hermes 启动，只会继续由 Gateway 拒绝运动。启动器仅设置精确
`ROBOT_CONFIG` 并预加载 `ibrobot-control`；它不会启动/重启 pipeline、修改 ROS 参数或开启运动授权。
