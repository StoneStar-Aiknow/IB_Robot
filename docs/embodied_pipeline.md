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

`robot-skill` 的运动命令只暴露高层技能，不调用 primitive、MoveIt、controller 或裸 ROS 运动接口。
视觉游戏使用独立的 `list-games`、`describe-game`、`start-game`、`game-result` 控制面，不进入运动 Gateway。
start 使用调用方 request ID 和独立 game config digest；在 advertised retention 窗口内相同 ID/请求可幂等
恢复且不会重复发布感知请求或事件。记录过期后 Gateway 不再保证检测重复，调用方应始终生成全局唯一 ID，
也不能用换新 ID 的方式自动重试同一次业务请求。

启用 `robot.grasp_execution` 后，`pick_object` 由 Gateway 构造版本化 delegated `PickObject` goal，
再进入 `manipulation_execution/pick_executor_node`。该执行器统一承载 execute、plan-only 和
observe-only 三种模式，并通过隔离的并行 IK/FK worker 准备候选；所有实际运动仍逐步回到
`skill_library` 的 `PrimitiveCommand` 安全边界。

## 已移除路径

`/voice_command` 不再是机器人运动入口。`perception_service` 仍可作为独立场景感知
服务启动，但不再由 task entry 自动路由。

## 视觉游戏控制平面

`visual_game_gateway_node` 为非运动视觉游戏（分院帽等）提供独立控制平面，与上述 Hermes
运动链路并存但不交互。游戏请求不进入 `agent_plan_node` / `skill_executor_node`，运动 Agent plan
也不触发游戏。

```text
robot-skill start-game -> StartVisualGame -> visual_game_gateway_node
     -> /embodied/perception_request -> perception_service_node
     -> /embodied/perception_result -> gateway 校验 + 有界 ledger
  -> GetVisualGameResult 查询 pending/terminal
  -> /embodied/visual_game_events 供可选 TTS、UI、日志等旁路消费者
```

- 视觉游戏只允许 Agent 通过 `robot-skill` 发起；`/voice_command` 和 `task_entry_node` 不参与游戏路由。
- gateway 用 `embodied_common.visual_game_contracts` 校验业务终态（分院帽四学院 / `NO_PERSON`），
  并以调用方 request ID 在 advertised retention 窗口内幂等恢复，不重复发布感知请求。
- 游戏定义、handler、timeout、retention、ledger capacity 全部配置化，由
  `visual_game_contracts` 归一化并计算 game capability digest；加新游戏可不改节点代码。

## Agent 计划协议

`PlanAgentCommand` 只接受调用方构造的 typed `WorkflowStep[]`；`raw_command` 仅用于审计和
幂等请求摘要，机器人运行时不解析自然语言。

所有执行仍必须经过：

1. exact catalog identity 捕获；
2. `ValidateAgentPlan` 逐步预检；
3. exact plan 展示后的内部 confirm 和不可变 task budget；
4. `ExecuteAgentPlan`；
5. Gateway admission 与 Safety 校验。
