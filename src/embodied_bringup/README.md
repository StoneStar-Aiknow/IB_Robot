# embodied_bringup 架构契约

`embodied_bringup` 是 Hermes-only 具身运行时的启动编排包。它消费 `robot_config`
SSOT YAML，启动 Agent plan、安全校验、Skill Gateway 以及可选感知和抓取执行服务。

## 职责边界

本包负责：

- 提供 `embodied_pipeline.launch.py` 公开入口。
- 从 `robot_config` 加载机器人配置并向下游注入参数。
- 启动 `agent_plan_node`、`safety_guard_node` 和 `skill_executor_node`。
- 按配置启动独立 `perception_service` 与抓取执行依赖。

本包不负责：

- 自行维护机器人配置或 Skill catalog。
- 实现文本规划、VLM 规划、安全规则或物理执行。
- 绕过 `skill_library` / `safety_guard` 发布运动命令。

规则 Planner ROS 节点和 `vlm_task_planner` 包已经移除。普通 `/voice_command`
流水线不再启动；当前唯一合法入口模式为 `embodied.entry_mode: hermes`。

## 依赖方向

```text
embodied_bringup
    -> robot_config
    -> embodied_agent        # Agent plan 生命周期
    -> safety_guard          # 只读校验
    -> skill_library         # 唯一物理执行 Gateway
    -> perception_service    # 可选独立感知服务
    -> manipulation_service  # 可选抓取感知/规划
    -> manipulation_execution
```

## 启动

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.sh && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=moveit_planning \
  entry_mode:=hermes \
  authorize_motion:=false
```

`authorize_motion` 默认关闭。只有操作员完成现场安全检查后才能在 launch 时显式开启；
Agent、CLI、YAML 和动态参数都不能代替操作员授权。

生产部署还应启用 SROS 2 caller policy，限制 Agent、plan coordinator、Gateway 和 operator 各自可调用的
endpoint。先按 `sros2/README.md` 生成部署 keystore 并设置 `ROS_SECURITY_ENABLE=true`、
`ROS_SECURITY_STRATEGY=Enforce`、`ROS_SECURITY_KEYSTORE`，再追加 `enable_caller_policy:=true`。Hermes/CLI
使用 `/hermes_cli` enclave，catalog reload 使用 `/operator` enclave。没有 keystore 时不得启用该参数。

`with_perception` 只控制独立感知服务，不恢复已删除的 voice/VLM Planner 流水线。
