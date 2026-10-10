# Program Gateway 与 Kimi Agent 验证

本次移植将 PR 456 的 typed Program 和持久化执行账本接入现有 Skill Gateway，
Agent 可提交完整 workflow；原有 Skill 和 Agent Plan 接口仍保留。

## 本机 SO-101 入口

在已初始化的工作区根目录执行；worktree 的共享 venv 初始化遵循
`.agents/skills/ibrobot-worktree-env/SKILL.md`，不要加载其他工作区的 overlay。

```bash
source .shrc_local
export ROS_DOMAIN_ID=52
export ROS2CLI_DISABLE_DAEMON=1
# KIMICODE_API_KEY 由本地环境提供，不写入配置或日志。
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_agent_manual_hardware \
  use_sim:=false control_mode:=moveit_planning authorize_motion:=true \
  with_embodied:=true with_perception:=false moveit_display:=false
```

该入口自动启动 `ibrobot_agent_node`，不需要重复启动 Agent 节点。
另一个使用相同工作区、ROS domain 的交互终端运行：

```bash
ros2 run ibrobot_agent ibrobot_agent_chat \
  --channel-id agent_cli --principal-id local_operator \
  --robot-scope so101_single_arm
```

客户端可随时连接；typed Program 采用直接执行、异步展示，不再需要展示回执，
机器人可能先于界面显示开始运动（见 `src/ibrobot_agent/README.md`「展示协议」）。
不要通过立即 EOF 或紧接 `/quit` 的管道模拟交互客户端。
直接使用 `robot-skill` 时，provider 配置需要本次 launch 生成的
`/tmp/ibrobot_interfaces_*/robot.yaml` 绑定快照。

## 2026-10-10 实测：异步展示

同一入口与 provider（`kimicode` / `kimi-for-coding`），探针不发送任何展示回执：

```bash
cd src/ibrobot_agent/test
python3 agent_node_kimi_hardware_probe.py --steps nod_yes
python3 agent_node_kimi_hardware_probe.py \
  --text "先点头一次，然后回到安全位置，只执行这两个动作。" --steps nod_yes recover_safe_pose
```

| 请求 | 规划步骤 | 终态 | presentation 到达 | terminal 到达 |
|---|---|---|---|---|
| 只执行一次点头动作，不要执行其他动作 | `nod_yes` | `SUCCEEDED` | 2924 ms | 10243 ms |
| 先点头一次，然后回到安全位置，只执行这两个动作 | `nod_yes` → `recover_safe_pose` | `SUCCEEDED` | 5039 ms | 14340 ms |

两次事件序列均为 `proposal_ready` → `presentation`（`async_display: true`）→ `terminal`，
没有 `presentation_rendered`；控制器日志共 9 次 `Goal reached, success!`。
时间从发布请求起算，包含 Kimi 规划与物理运动；与下方旧样本的任务相同但未做受控对比，
不能据此计算展示门控节省的时间。

## 2026-10-08 实测（旧展示回执流程）

真实 provider 为 `kimicode`，模型为 `kimi-for-coding`。
持续在线客户端通过 `/agent/request` 提交自然语言，核对并展示 Kimi 生成的
准确步骤，发送展示回执，然后等待 `/agent/event` 的 terminal 事件。

| 请求 | 规划步骤 | 客户端终态 | 端到端时间 |
|---|---|---|---|
| 只执行一次点头动作，不要执行其他动作 | `nod_yes` | `SUCCEEDED` | 11307.958 ms |
| 先点头一次，然后回到安全位置，只执行这两个动作 | `nod_yes` → `recover_safe_pose` | `SUCCEEDED` | 13625.802 ms |

单步采用仓库的 `agent_node_kimi_hardware_probe.py`；多步探针复用
`AgentChatNode` 和展示渲染器。流水线日志包含真实夹爪和机械臂控制器的
`Goal reached, success!`。时间包含规划、展示和物理运动；两次样本不能证明
相对旧版本的性能无回退，也不代表取消、并发、所有技能或视觉抓取已验收。

修复了空成功消息写入 conversation store 抛错、导致终态已持久化但客户端
收不到 terminal 事件的问题。回归测试使用真实 SQLite conversation store。

正式测试入口：

```bash
colcon test --merge-install --packages-select \
  embodied_common skill_library robot_skill_cli ibrobot_agent embodied_bringup
colcon test-result --all --verbose
```

模型 bundle 依赖测试在本地缺失模型时按现有 marker 跳过。
取消收敛、并发执行和异常清理仍需独立扩展验证，不能将账本接受取消等同于物理停止。
