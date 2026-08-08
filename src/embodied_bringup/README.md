# embodied_bringup 架构契约

`embodied_bringup` 是具身 AI 运行时的启动编排包。它消费 `robot_config` 的
SSOT YAML 配置，负责把任务入口、规则/VLM planner、任务执行器、技能执行器、
安全校验和可选感知服务组合成一个运行时闭环。

## 职责边界

本包负责：

- 提供具身运行时的公开 launch 入口。
- 从 `robot_config` 加载机器人 YAML，并把配置以 ROS 参数注入下游节点。
- 编排 `embodied_agent`、`skill_library`、`safety_guard`、`vlm_task_planner`
  和 `perception_service` 的启动顺序与参数。
- 当 `robot.grasp_execution.enabled=true` 时，编排 Grounded-SAM2、GraspGen、抓取验证器和
  `manipulation_execution/pick_executor_node`。
- 将 `grasp_execution.perception_node/planner_node.host_runtime` 转换为对应节点的进程环境；该块不作为
  ROS 参数传给业务节点。
- 当 `robot.grasp_execution.ik.worker_count>0` 时，自动包含 `robot_moveit/so101_ik_workers.launch.py`，
  为 Hermes 启动与监督式抓取脚本相同的并行候选 IK/FK 池。
- 保持具身业务运行时依赖集中在 bringup 层，避免 `robot_config` 反向依赖业务包。

本包不负责：

- 定义新的机器人配置源。机器人型号、命名位姿、技能模板、workspace、相机 topic
  仍以 `robot_config` YAML 为单一事实来源。
- 实现任务理解、技能展开、安全规则、VLM 调用或感知算法。
- 绕过 `skill_library` / `safety_guard` 直接发布 MoveIt、关节或夹爪命令。

## 依赖方向

```text
embodied_bringup
    -> robot_config          # 读取 SSOT 配置并包含基础机器人 launch
    -> embodied_agent        # 任务入口、规划和任务执行编排
    -> skill_library         # 技能/primitive 执行
    -> safety_guard          # 技能和 primitive 校验
    -> vlm_task_planner      # 可选 VLM 规划
    -> perception_service    # 可选连续场景理解
    -> manipulation_service  # 可选 GraspGen 规划与抓后验证
    -> manipulation_execution # 可选抓取闭环编排
    -> robot_moveit          # 可选隔离 IK/FK worker 进程
```

`robot_config` 不应 import 或依赖 `embodied_bringup`。如果需要启动完整具身链路，
应从本包的 launch 入口启动，而不是让 SSOT 配置包反向知道运行时业务包。

## 公开入口

推荐启动完整具身闭环：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.sh && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=moveit_planning \
  use_sim:=true \
  moveit_display:=false
```

`embodied_pipeline.launch.py` 会：

1. 包含基础 `robot_config robot.launch.py`，并显式传入 `with_embodied:=false`，避免
   `robot_config` 反向启动具身业务节点。
2. 再由本包读取同一份机器人 YAML，并启动具身运行时节点。

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

`authorize_motion` 默认关闭，因此标准启动只提供可查询的 Gateway 状态，不允许技能下发运动。
只有操作员在完成现场安全检查后，才可以在启动时显式开启该参数。Agent、CLI 和运行中的节点
不得代替操作员执行授权启动，也不得通过 YAML 或动态参数开启授权。实际 `control_mode` launch
override 会原样注入 Gateway；`skill_required_control_mode`、status service、机器人名称和 timeout
策略继续来自 `robot_config` SSOT。

## Rule-entry alias 注入

`generate_embodied_nodes()` 从同一份 YAML `skill_templates` 中提取未禁用、
`description.rule_entry: true` 且不需要运动参数的中文别名，生成唯一一份
`skill_aliases_json`。该值同时注入 `task_entry_node`、规则 `task_planner_node` 和
`vlm_task_planner_node` 的确定性 fallback，三条入口因此共享同一规则 alias 契约。

未标记 `rule_entry` 的观察、回位、夹爪和参数化运动 alias 不会进入该 JSON；这些命令继续
由 `embodied_agent` 的专用规则分支维持既有 `task_type`。

## 已知限制

- 当前具身闭环要求 `control_mode:=moveit_planning` 或名称中包含 `moveit` 的兼容控制模式。
- 自然语言规则入口只支持观察、回位、夹爪开合、相对移动和夹爪旋转等最小闭环动作。
- 抓取、放置、目标物操作当前不由规则入口直接生成；应通过后续 VLM/显式技能链路完善。
- 视觉趣味游戏（分院帽等）由 `task_entry_node` 路由到 `perception_service`：触发别名与开关来自
  `embodied.entry.visual_games`，camera/VLM/timeout 仍由 `embodied.perception` 唯一管理。
  启用某游戏需**同时**置 `embodied.perception.enabled: true` 与该游戏的 `enabled: true`；若 launch
  override（如 `with_perception:=false`）造成不一致，`embodied_pipeline.launch.py` 会在生成节点前 fail-fast
  拒绝启动，避免生成不一致的运行时节点图。
- `so101_handeye_realsense_grasp` 可通过显式 `pick_object` 技能从 Hermes 调用完整抓取闭环。
- 真机端口、相机和手眼标定直接维护在该 robot YAML 中；本 launch 与 `robot-skill` 应使用同一个
  `robot_config` 名称，workspace 外部完整 YAML 才需要显式传 `config_path`。
