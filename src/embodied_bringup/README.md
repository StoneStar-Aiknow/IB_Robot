# embodied_bringup 架构契约

`embodied_bringup` 是 Hermes-only 具身运行时的启动编排包。它消费 `robot_config`
SSOT YAML，启动 Agent plan、安全校验、Skill Gateway 以及可选感知和抓取执行服务。

## 职责边界

本包负责：

- 提供 `embodied_pipeline.launch.py` 公开入口。
- 从 `robot_config` 加载机器人配置并向下游注入参数。
- 启动 `agent_plan_node`、`safety_guard_node` 和 `skill_executor_node`。
- 按配置启动独立 `perception_service` 与抓取执行依赖。
- 当 `robot.grasp_execution.enabled=true` 时，编排 Grounded-SAM2、GraspGen、抓取验证器和
  `manipulation_execution/pick_executor_node`。
- 将 `grasp_execution.perception_node/planner_node.host_runtime` 转换为对应节点的进程环境；该块不作为
  ROS 参数传给业务节点。
- 当 `robot.grasp_execution.ik.worker_count>0` 时，自动包含 `robot_moveit/so101_ik_workers.launch.py`，
  为 Hermes 启动与监督式抓取脚本相同的并行候选 IK/FK 池。
- 保持具身业务运行时依赖集中在 bringup 层，避免 `robot_config` 反向依赖业务包。

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

SO-101 真机手动验证必须使用 `ROS_DOMAIN_ID=52`、显式设置 `moveit_display:=true`，并通过 Hermes 完成
plan/validate/confirm/execute。完整启动、回原位和关停流程见
[`docs/hermes_so101_real_robot_manual_validation_zh.md`](../../docs/hermes_so101_real_robot_manual_validation_zh.md)。

`with_perception` 只控制独立感知服务，不恢复已删除的 voice/VLM Planner 流水线。

主要参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `robot_config` | `so101_single_arm` | robot_config 中的机器人配置名 |
| `config_path` | 空 | 可选的 YAML 绝对路径覆盖 |
| `control_mode` | `moveit_planning` | 具身闭环当前要求 MoveIt 兼容控制模式 |
| `use_sim` | `false` | 是否启动仿真路径 |
| `with_moveit` | 空 | 传递给基础 robot launch 的 MoveIt 覆盖参数 |
| `moveit_display` | `false` | 是否启动 MoveIt RViz |
| `with_embodied` | `true` | 是否启动具身运行时节点 |
| `with_perception` | 空 | 覆盖 `robot.embodied.perception.enabled` |
| `authorize_motion` | `false` | 操作员运动授权；唯一运行时授权来源 |

## 已知限制

- 当前具身闭环要求 `control_mode:=moveit_planning` 或名称中包含 `moveit` 的兼容控制模式。
- `so101_handeye_realsense_grasp` 可通过显式 `pick_object` 技能从 Hermes 调用完整抓取闭环。
- 真机端口、相机和手眼标定直接维护在该 robot YAML 中；本 launch 与 `robot-skill` 应使用同一个
  `robot_config` 名称，workspace 外部完整 YAML 才需要显式传 `config_path`。
