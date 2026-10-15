# Hermes 集成发布源

此目录是 IB-Robot Hermes 集成的唯一发布源，包含推荐 `SOUL.md`、强制策略 (`POLICY.md`)、
`RESPONSE_PROTOCOL.md` 回复契约、结构化输出插件和手动同步入口。同步会安装当前 `ibrobot-control` Skill，
绑定当前 workspace、robot config 和 ROS Domain，并清理已退役的 `ibrobot-robot-control` Plugin
遗留副本（即时执行改由 `robot-skill run-workflow` 的显式 `immediate_after_presentation` 模式承担）。

## 资源

| 文件 | 说明 |
|---|---|
| `SOUL.md` | Hermes persona 与完整具身任务工作流（含 `ibrobot-perceive` 使用指引） |
| `POLICY.md` | 强制控制与感知策略；引用系统注入的结构化回复协议 |
| `RESPONSE_PROTOCOL.md` | v1 最终回复与 v2 中间回复契约；v2 增加 `completion` 收尾策略 |
| `hooks/ibrobot-block-raw-ros` | Hermes `pre_tool_call` hook，用 `shlex` 分词拦截裸 `ros2`/`rclpy`/`roslaunch` 调用 |
| `hooks/ibrobot-speak` | 结构化语音事件的 ROS 投递端，source `.shrc_local` 后运行 `hermes_tts_hook`；不再直接挂载到 `post_llm_call` |
| `hooks/ibrobot-lifecycle-speech` | 旧生命周期兼容入口；同步后不再挂载，避免重复提示与额外文案模型调用 |
| `hooks/ibrobot-interim-speech` | 旧安装兼容入口，只接受结构化回复；新安装由原生插件监听 interim，不重复挂载 |
| `sync_hermes.sh` | 手动同步入口（等价于 `hermes-robot-configure`） |

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

## 同步

从仓库根目录执行：

```bash
source .shrc_local
export ROS_DOMAIN_ID=56

hermes-robot-configure \
  --config-name so101_single_arm \
  --soul-mode replace \
  --accept-hooks \
  --install-runtime-adapter \
  --restart-gateway
```

`.shrc_local` 是 ROS+venv+install overlay 的 SSOT 入口（见 `AGENTS.md` 环境初始化章节），
无需重复 `source install/setup.bash`。

也可以使用此目录中的 shell 入口：

```bash
bash src/robot_skill_cli/resource/hermes/sync_hermes.sh \
  --config-name so101_single_arm \
  --soul-mode replace \
  --accept-hooks \
  --install-runtime-adapter \
  --restart-gateway
```

先预览而不写文件：

```bash
hermes-robot-configure --config-name so101_single_arm --dry-run
```

`--soul-mode replace` 会先备份现有 `SOUL.md`，再安装仓库中的完整推荐版本。使用
`--soul-mode merge` 会在现有文件中维护一个 `IBROBOT-MANAGED` 区块；`--soul-mode skip` 不修改 SOUL。

## 同步产物

同步在当前 Hermes profile 下生成以下受管文件（带 managed 标记，重跑同步即覆盖，不要手改）：

- `skills/ibrobot-control/SKILL.md`：当前 `ibrobot-control` Agent Skill 副本。
- `ibrobot/bin/robot-skill`：绑定 `ROBOT_CONFIG`、`ROS_DOMAIN_ID` 并 source workspace
  `.shrc_local` 的 wrapper；进入 Hermes 会话后用它而非裸 `robot-skill`，且不得再传
  `--config-name` / `--config-path`。
- `ibrobot/ibrobot-env.sh`：写入 `terminal.shell_init_files` 的环境文件，并把
  `auto_source_bashrc` 置 `false`，使受管环境优先于用户 bashrc；它同时把 `ibrobot/bin`
  加入 `PATH`。排查「Hermes 不再 source 我的 bashrc」时先看此文件与该开关。
- `plugins/ibrobot-response/`：原生输出适配器、纯 Python 回复解析器和协议文档。Hermes 使用自己的
  Python 加载这些文件，不需要导入 ROS 包。由 `hermes-robot-configure` 管理，遇到同名用户插件会报冲突。
- `hooks/ibrobot-speak`：结构化语音事件投递端，source `.shrc_local` 后
  `exec python3 -m robot_skill_cli.hermes_tts_hook`；服务名与超时来自 `robot_config` SSOT。
- `hooks/ibrobot-lifecycle-speech`：保留兼容文件，同步时移除其受管 hook 注册；`robot-skill`
  wrapper 关闭旧的 `plan_confirmed` 固定播报。过程文案由主模型的同一次工具响应生成。
- `hooks/ibrobot-interim-speech`：兼容入口，新安装不挂载此 shell hook，避免重复播报。
- `display.streaming=false` 和 `streaming.enabled=false`：先校验完整 JSON 再显示渲染结果，避免
  原始协议碎片泄露到 CLI/消息平台。普通 interim 展示关闭，完整的结构化 interim 仍可由插件播报。

受管语音使用原生插件回调，不再需要 shell observer 授权。`--accept-hooks` 保留为兼容参数，
已有启动命令仍可使用且不会执行授权操作；用户自管 hook 保留原有配置与授权方式。
runtime adapter 需要显式传入 `--install-runtime-adapter`；不传时 profile/语音同步仍可完成，
但 `ibrobot-motion` 和优先停止不会启用。

## 前置条件

- Hermes 必须提供原生 `register_system_prompt_section`、`pre_llm_call`、`transform_llm_output`、
  `post_llm_call` 和 `on_interim_message` 接口；当前验证版本为 0.20.2。
- 已构建 `robot_skill_cli`、`robot_config`、`ibrobot_msgs` 和 TTS 相关包。
- 当前终端已 source 目标 IB-Robot workspace（`source .shrc_local`）。
- 启用语音时，目标 robot config 的 `voice_tts.enabled` 必须为 `true`。
- 真机运动仍必须由操作员在启动 pipeline 时显式设置 `authorize_motion:=true`。

## 执行期间输入“别动”

使用 `--install-runtime-adapter` 时，同步器安装 `ibrobot-motion` 插件和 Hermes 原生控制 hook 适配器。
当前适配的 Hermes commit 为
`cf64ca20c5ab99ebf7e8ca272c69edc7ea0636ed`（0.20.2）；同步前校验版本、源码锚点和已安装文件哈希，
未知版本或被修改的受管文件会报错。Hermes runtime 修改前生成 `.ibrobot-backup-*` 文件；
`.ibrobot-control-adapter.json` 记录版本和文件哈希。更新 Hermes 后须重新验证适配器，不能只比较版本字符串。

- 在经典 Hermes CLI（`hermes --cli`）输入 `别动`、`别动了`、`停止`、`停下`、`停`、`stop` 或 `halt`，
  立即锁存当前会话的停止要求；Gateway 文本消息在权限校验后走同一入口。大小写和首尾空白被规范化。
- `run-workflow` 由受管理的执行器启动；停止通过 profile 的 `ibrobot/control-state/` 传给所属 CLI，
  不等待模型、普通消息队列或会话租约。该目录只对 profile 所属用户开放，不从记录中的 PID 发信号。
- CLI 继续取消根 Agent Plan，并等待完整终态。停止与新目标提交在同一状态锁下排序；展示失败或展示期间
  收到停止时不会提交目标。机器人 CLI 的正常 SIGINT/SIGTERM 处理仍可用于直接命令诊断。
- 受管理的机器人任务不经过 Hermes 通用终端的 1 秒 SIGKILL。退出等待预算从机器人 RPC 配置推导；
  超过预算、客户端异常退出或终态身份不一致时报告状态未知并阻止后续运动。
- owner 仍存活时只由其 CLI 取消；owner 和 CLI 都退出后，停止入口才可用保留的完整身份发 `cancel-plan`。
  不能手工删除状态目录来解除未知状态，也不能用 `cancel` 代替根计划取消。
- 停止后本回合不能自动提交另一动作；终态确定后，由用户输入新的独立请求开始下一回合。

同步后需要退出并重新启动已有 Hermes CLI；`--restart-gateway` 只重启消息 Gateway。
当前适配覆盖经典 CLI 和 Gateway 文本入口，未实现 TypeScript TUI、ACP 或流式语音的输入回调。
成功验收应同时看到取消终态和当前子动作停止；单独出现“正在停止”、工具退出或 `Redirected` 不能确认停稳。

### 中间回复与本地收尾

主模型在同一次响应中输出完整 v2 JSON 正文和正常的工具调用：

```json
{"schema_version":2,"speech":"好的，我来关闭夹爪。","details":"","completion":"after_workflow"}
```

`on_interim_message` 在该响应完成、工具执行前异步投递正文，TTS 与工具准备并行。
它不是 token 流式播报，也不保证短动作开始前已经发声；原始 JSON 的流式显示仍关闭。

`completion=after_workflow` 只适用于用户要求一个简单动作的回合。当前支持单步骤的
`open_gripper_skill`、`close_gripper_skill`、`recover_safe_pose`、`nod_yes`、`shake_no`、`wave_hello`。
受管执行器校验完整计划身份、结果状态与步数，工具循环核对本批次的 tool call ID 和最终记录的
工具结果，再生成本地 v1 回复，经原有 SessionDB、显示与语音 finalizer 结束回合，省去收尾模型请求。

`completion=continue`、v1 回复、无正文、多工具、多步骤、未支持技能、结果未知、用户新指令、
中断或被更改的工具结果继续走正常 Hermes 对话。需要分析、解释或后续动作时必须使用 `continue`。
不解析任意终端输出作为完成证明，也不使用模型预先写好的成功声明。

安装器为受支持的 Hermes revision 更新可逆的通用 `finalize_tool_batch` hook，业务规则在
`ibrobot-motion` 插件内；适配器 manifest 必须是当前版本，文件哈希不匹配时拒绝覆盖用户修改。
独立进程的测试见 `test/test_hermes_runtime_e2e.py` 文件说明，需在隔离 Hermes checkout 中运行。

## 生效与验证

本地 CLI：

```bash
hermes-robot --config-name so101_single_arm -- --cli
```

飞书使用长期运行的 Hermes Gateway。同步配置或 Skill 后必须重启，并在飞书中发送 `/new` 创建新会话：

```bash
hermes gateway restart
hermes gateway status
hermes hooks list
hermes hooks doctor
```

动作请求的预期流程为：展示并 flush 冻结计划，内部调用 `confirm-plan` 绑定 exact tuple，然后立即
`execute-plan`，不等待用户再次回复“确认”。这不会绕过 `authorize_motion`、Gateway 校验或停止能力。

最终回复必须是 `{"schema_version":1,"speech":"自然语言","details":"技术详情"}`。
原生插件在 `transform_llm_output` 中校验并渲染正文与详情，在 `post_llm_call` 中只投递已解析的
`speech`。数据按当前线程的 session/turn 绑定，并核对渲染结果，不能从显示文本重新猜测语音字段。
协议在新会话的系统提示中注入，`pre_llm_call` 同时向旧会话补充相同协议。
显示层使用 HTML details/pre 保留完整技术证据；CLI 会将标签显示为文本，是否支持折叠由消息客户端决定。

最终回复兼容 v1 语音事件：

```json
{"schema_version":1,"session_id":"session","turn_id":"turn","phase":"final","response":{"schema_version":1,"speech":"夹爪已打开。","details":"goal_status=4"}}
```

interim 使用 v2 事件，增加整数 `sequence`（来自 Hermes iteration），按 session/turn/phase/sequence
原子去重。同回合不同中间段可各播一次，重放同一事件不会重复。失败或未知播放结果不自动重播。
仅 `speech` 进入 `/voice_tts/synthesize` 和 `/voice_tts/play`；格式错误、缺字段、重复字段或截断 JSON
记录错误并跳过播报，原始文本可在显示详情中查看，绝不退回整段朗读。文本语义仍需依据工具结果评估，
结构校验不会证明机器人成功或停止。技术详情只供查看，不能影响控制流程。

TTS 不再维护路径、错误码或特定场景措辞替换。只进行 Unicode 规范化和前端兼容检查；当前 ZipVoice
不支持 ASCII 英文字母，遇到该情况报错并跳过语音，不删除英文字母改变含义。日志位于
`/tmp/hermes-speak.log`，记录实际 speech 与事件标识；去重记录位于 `/tmp/ibrobot-speech-events/`。

收到当前会话的停止词、最终回复生成或回合结束时写入语音终态标记；尚未处理、等待服务、合成中或等待播放锁的旧 interim
会在投递前被淘汰。已送入播放设备的音频不撤回。标记与去重文件使用同一
`IBROBOT_SPEECH_STATE` 目录，因此原生插件与 speaker 必须继承相同配置。

内存中的活动回合身份在 `post_llm_call` 后回收；中断、失败或空回复由 Hermes 标准的
`on_session_end` 回调清理，不依赖可选 runtime adapter。停止回调只处理尚未结束的活动回合。

## 升级

每次更新并重新构建 IB-Robot 后再次执行同一同步命令。同步是幂等的，不会重复添加 hook、Skill 或
SOUL 托管区块。配置变化后必须重启 Gateway；仅使用本地 CLI 时重新启动 CLI 会话即可。
