# embodied_agent 架构契约

`embodied_agent` 是 Hermes-only 具身运行时中的 Agent plan 生命周期与 Workflow 编排包。
它不拥有 Skill catalog、运动授权或物理执行权。

## 当前 ROS 节点

| 节点 | 主要职责 |
| --- | --- |
| `agent_plan_node` | plan / validate / confirm / execute 生命周期，按顺序调用 Skill Gateway |
| `task_entry_node` | 遗留 voice adapter，当前不由 bringup 启动 |
| `task_executor_node` | 遗留 planned-task executor，当前不由 bringup 启动 |

规则 `task_planner_node` 已删除；`vlm_task_planner` 也已删除。运行时唯一公开入口模式为
`embodied.entry_mode: hermes`。

## 调用链

```text
Hermes / robot-skill
  -> /embodied/plan_agent_command
  -> /embodied/validate_agent_plan
  -> 用户确认后 /embodied/confirm_agent_plan
  -> /embodied/execute_agent_plan
  -> skill_library Gateway
```

## Agent plan 状态机

```text
PLANNED -> VALIDATED -> CONFIRMED -> ACCEPTED -> TERMINAL
```

- plan 捕获 exact catalog identity，并保存短时不可变 `AgentPlan`。
- validate 对每个步骤执行只读 Safety 预检。
- confirm 绑定 plan digest、task ID、registry identity 和绝对 task budget。
- execute 复用确认时冻结的预算，通过 Gateway 执行 Skill 或 Workflow。
- child 接受、取消或终态未知时保持 plan 为 `ACCEPTED`，不得自动重试或释放可能仍有效的 root lease。

Agent 必须通过 `robot-skill plan-workflow` 提交结构化步骤；机器人运行时不解析自然语言，
`raw_command` 只作为审计文本和幂等请求摘要的一部分。

## 边界

- 本包不得调用 primitive、MoveIt 或 controller。
- motion authorization 只能来自操作员 launch 参数。
- 所有执行必须经过 `skill_library` 和 `safety_guard`。
- `perception_service` 是独立服务，不由已停用的 voice adapter 自动路由。

## 启动

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.sh && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  entry_mode:=hermes \
  control_mode:=moveit_planning \
  authorize_motion:=false
```
