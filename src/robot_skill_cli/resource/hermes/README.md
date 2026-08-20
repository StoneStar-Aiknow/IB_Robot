# Hermes 感知策略与 raw-ROS 拦截

此目录打包 IB-Robot Hermes 集成中与感知读取和裸 ROS 拦截相关的资源：策略文档
(`POLICY.md`) 和 `pre_tool_call` 拦截 hook (`hooks/ibrobot-block-raw-ros`)。完整的 Hermes
persona 见仓库 [`docs/hermes_soul/SOUL.md`](../../../docs/hermes_soul/SOUL.md)，其中已包含
`ibrobot-perceive` 使用指引。

## 资源

| 文件 | 说明 |
|---|---|
| `POLICY.md` | 强制策略：感知读取只能走 `ibrobot-perceive`，运动只能走 `robot-skill` + Gateway；裸 `ros2` 子命令被禁止 |
| `hooks/ibrobot-block-raw-ros` | Hermes `pre_tool_call` hook，用 `shlex` 分词拦截裸 `ros2`/`rclpy`/`roslaunch` 调用 |

## `ibrobot-perceive`（感知读取唯一入口）

`ibrobot-perceive` 是 `robot_skill_cli` 提供的独立 console script，不经过 Gateway。它以硬编码
source/field allowlist 限制可读取的感知量，通过 `ros2 topic echo --once` 读取 YAML 输出并打印请求
字段的裸字面量值，供 LLM 直接读取并注入 `workflow_json`。当前 allowlist 包含
`voice_direction`（topic `/voice/speech_direction`，字段 `azimuth_rad`、`seq_id`）和
`arm_joint_position`（字段 `position`），扩展必须修改源码，不接受 config.yaml 覆盖。

`--source` 是语义别名而非 ROS topic 名：`voice_direction` 是 `voice_asr_service` 的固定契约，topic 写死
不读 robot_config；`arm_joint_position` 的 topic 在运行时从 `robot_config` 的 `moveit.joint_state_topic`
解析（so101 -> `/joint_states`，lekiwi_handeye -> `/arm_joint_state_broadcaster/joint_states`），复用
`resolve_robot_config_path()`，不维护第二套路径优先级。安全边界是硬编码的 *field* 集合，不是 topic 名。
`arm_joint_position.position` 直接返回原始弧度数组，不提供关节名映射。

`ros2 topic echo --once` 返回的是下一条已发布消息的单次点时值，不是持久快照。对
`/voice/speech_direction` 这类事件型 topic，发布方不活跃时会在超时内取不到值；取到的值在消费时
可能已经过期。该值在 `plan-workflow` 时冻结进 plan digest，后续按 frozen plan 语义审计；执行结果
以真实运动为准，不自动重试或修正。

## `ibrobot-block-raw-ros`（pre_tool_call 拦截 hook）

此 hook 是 `pre_tool_call` 防御层。它从 stdin 读取 Hermes tool payload，提取 `tool_input.command`，
用 `shlex` 分词后阻断任何裸 `ros2` 子命令和 `rclpy`/`roslaunch` 间接调用，强制 LLM 走
`ibrobot-perceive`（感知读取）或 `robot-skill`（运动控制）。

**它是 defense-in-depth，不是沙箱。** 权威边界仍然是操作员的 `authorize_motion` 授权门禁和
Gateway 的 plan validation。hook 输出 `{"action":"block","message":"..."}` 表示拦截，空输出表示放行；
exit code 不用于拦截判定。

协议：

- stdin：JSON，含 `tool_input.command`（字符串或列表）。
- stdout：拦截时输出一行 `{"action":"block","message":"..."}`；放行时不输出。
- 解析失败或缺少 `command` 字段时放行（fail-open），因为 hook 无法判断意图。

## 前置条件

- 已构建 `robot_skill_cli`、`robot_config`、`ibrobot_msgs`。
- 当前终端已 source 目标 IB-Robot workspace（`source .shrc_local`）。
- 真机运动仍必须由操作员在启动 pipeline 时显式设置 `authorize_motion:=true`。

Hermes persona 和完整具身任务工作流见
[`docs/hermes_soul/SOUL.md`](../../../docs/hermes_soul/SOUL.md)。
