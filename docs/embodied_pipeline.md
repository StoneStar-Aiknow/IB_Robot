# Hermes-only 具身执行流水线

## 当前拓扑

```text
Hermes / robot-skill
    -> PlanAgentCommand
    -> ValidateAgentPlan
    -> ConfirmAgentPlan
    -> ExecuteAgentPlan
    -> skill_library Gateway
    -> task_dispatch / manipulation_execution / controller
```

`agent_plan_node` 只负责任务生命周期和 Workflow 编排，不直接执行 primitive。
`skill_executor_node` 是唯一拥有 motion admission、root lease、ledger 和物理下发权的节点。

## 已移除路径

`/voice_command` 不再是机器人运动入口。`perception_service` 仍可作为独立场景感知
服务启动，但不再由 task entry 自动路由。

## Agent 计划协议

`PlanAgentCommand` 只接受调用方构造的 typed `WorkflowStep[]`；`raw_command` 仅用于审计和
幂等请求摘要，机器人运行时不解析自然语言。

所有执行仍必须经过：

1. exact catalog identity 捕获；
2. `ValidateAgentPlan` 逐步预检；
3. 用户确认和不可变 task budget；
4. `ExecuteAgentPlan`；
5. Gateway admission 与 Safety 校验。
