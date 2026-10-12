# ibrobot_agent 架构契约

`ibrobot_agent` 是 IB-Robot 自然语言 Agent 的孵化运行时包。它把用户的自然语言请求转成
严格校验的 canonical Program，并把全部机器人执行委托给既有 Capability Gateway 链路
（Agent ExecutionPort → `ExecuteProgram` → ExecutionStore → Safety Guard → Skill Executor）。
本包不拥有 Skill catalog、运动授权或物理执行权，也不实现感知。

本包与 `embodied_agent` 互补而非重复：`embodied_agent` 提供既有 Agent Plan 兼容入口；
生产 Agent 路径通过 `ExecuteProgram` 一次提交完整 workflow，`ibrobot_agent` 负责自然语言理解、
Planner 适配、请求状态机与持久化 ledger。Planner 是本包进程内的库适配器
（rule / vlm 两种模式），不是独立 ROS 节点，也不恢复历史上已删除的
`vlm_task_planner` 包路线。

## 当前 ROS 节点

| 节点 / 入口 | 主要职责 |
| --- | --- |
| `ibrobot_agent_node` | Agent 组合根：transport 与生命周期；ROS-free 的 `AgentService` 通过注入的 port 工作 |
| `agent_voice_bridge`（`ibrobot_agent_voice_bridge`） | 语音通道客户端：`/voice_command` 最终文本 → `/agent/request`；停止词 → `/agent/control stop`；展示 exact plan 后回执；可选 TTS 播报回复 |
| `chat_tui.py`（`ros2 run ibrobot_agent ibrobot_agent_chat` / `scripts/run_agent_chat.sh`） | 交互式终端 UI：输入历史、异步事件渲染、`/stop` 与停止词旁路；`--channel-id`/`--principal-id`/`--robot-scope` 覆盖会话身份以匹配 profile 的 `embodied.agent` 配置 |

## 调用链

```text
用户 / chat_tui
  -> /agent/request (std_msgs/String, JSON)
  -> ibrobot_agent_node
     -> Planner（rule / vlm，严格 JSON 契约）
     -> Agent ExecutionPort 复用 robot_skill_cli.RosBridge
         -> /embodied/execute_program
         -> ExecutionStore（幂等、事件、终态、取消）
         -> Safety Guard -> Skill Executor
  <- /agent/response（同步应答） /agent/event（有序事件）
   -> /agent/control（request 粒度 stop / 旧 Agent Plan 的 exact plan 展示完成回执）
```

## 请求状态机

```text
RECEIVED -> PLANNING -> PROPOSAL_READY -> PREPARING -> MAY_EXECUTE -> RUNNING
    -> SUCCEEDED / FAILED / CANCELLED / CANCELLED_BEFORE_EXECUTION
非运动应答: PLANNING -> ANSWERED
终态未知:   -> UNKNOWN（触发 robot-scope quarantine）
```

- 单 robot-scope 同一时间只接受一个活动请求，重复 request_id 幂等去重；BUSY 是终态，重试必须生成新的 request_id。
- stop 以 generation 防竞态；执行前后取消分别落到 `CANCELLED_BEFORE_EXECUTION` / `CANCELLED`。
- 请求、事件序列与会话记忆持久化在 SQLite（WAL + FULL sync）ledger 中。

## Topic / 服务契约（孵化期临时契约）

`/agent/request`、`/agent/response`、`/agent/event`、`/agent/control` 当前使用
`std_msgs/String` 承载 JSON。入站请求与控制消息由节点做构造器级 schema 校验
（`schema_version`、字段类型与非空检查）；重复 key、NaN/Infinity、围栏文本、
tool call 的严格拒绝只作用于 Planner 响应（`ibrobot_agent/planner.py`）。这是
**孵化期的有意决策**：契约稳定前不把不成熟接口固化进 `ibrobot_msgs`。计划在孵化期
结束后迁移为 `ibrobot_msgs` 的类型化消息；在此之前，外部集成方不应把这些 topic
当作稳定 API。Gateway 服务侧调用保持既有类型化契约不变。

节点另提供 `~/health` 与 `~/ready`（`std_srvs/Trigger`）。

生产 Program 路径使用 `registry_epoch`、`registry_generation` 和
`registry_digest` 绑定一次精确 Skill catalog snapshot；ExecutionStore 负责 root
execution 的幂等、查询、事件和终态。旧 Agent Plan 服务只作为兼容/诊断入口，不能作为
新的 Agent 调用路径。

`/agent/request` 的 JSON 在解析前限制为 64 KiB UTF-8 字节；`AgentRequest.text` 去除首尾空白后最多
4096 个字符（中文字符按一个字符计数）。超限返回 `REQUEST_SCHEMA_INVALID`，不会截断指令或进入规划队列。
限制由 `ibrobot_agent.contracts` 统一定义。状态、技能列表与姿态列表查询复用 Planner 层的
确定性只读解析，否定或假设请求不走该快捷路径；普通只读问句直接读取 Gateway 事实，不依赖模型调用。

## 安全设计

- 运动默认关闭：`execution_enabled=false` 时零 goal 提交，仅规划与应答。
- 执行必须同时满足：非空 `test_allowlist`、Gateway readiness（live catalog 身份一致）、
  操作员 `authorize_motion` launch 参数。
- 内置 RulePlanner 只允许在仿真模式执行；真机执行必须配置 vlm Planner。
- Planner 输出只接受单一严格 JSON 对象；未知字段、未知 skill、编造参数一律拒绝。
- 终态未知（UNKNOWN）进入 robot-scope quarantine，需人工介入。
- 启动时在接受请求前执行单事务恢复：未完成且可能已提交的请求转为持久化 UNKNOWN，保留 TaskRef
  供人工通过 Gateway 核实；确定未提交的中断请求转为 FAILED 或 CANCELLED_BEFORE_EXECUTION。
  不重放旧请求，不以 Gateway 空闲推断成功；已有终态和其他 robot scope 的记录保持原样。
- 部署锁（`deployment_lock_path`）防止双 Agent 实例并存。
- 提交前持久化回调异常由 `AgentService` 根据 ledger 判定：未提交时失败；可能已提交时
  UNKNOWN/quarantine；无法读取或写入 ledger 时保持隔离，不由 transport adapter 猜测终态。
- Planner 凭据统一按 Kimi 模式提供：`base_url` + `api_key_env`（环境变量名）；
  `embodied.agent.planner.api_key` 字面密钥会被 robot_config 校验直接拒绝，
  也不会进入 `config_digest` 的 preimage。

## 依赖方向

```text
ibrobot_agent
    -> robot_skill_cli   # RosBridge / catalog view（复用，不复制）
    -> embodied_common   # canon / workflow_contracts / VLMAPIClient
    -> robot_config      # SSOT 配置与 launch 校验（经 embodied_bringup 注入参数）
    -> rclpy / std_msgs / std_srvs
```

本包不得：直接发布运动命令或调用 controller/MoveIt、绕过 `skill_library` 与
`safety_guard`、维护机器人本体配置、感知模型或 Skill catalog。

## 配置入口

全部配置来自 `robot_config` SSOT YAML 的 `embodied.agent` 块（字段表与校验规则见
`robot_config` README 的具身 AI 流水线一节），由 `embodied_bringup` 在
`embodied.entry_mode: agent` 时注入为节点参数。仓库自带 SO-101 孵化 profile：
`so101_agent_manual`（真机手动无运动）、`so101_single_arm_agent_test`（rule Planner 测试）、
`so101_single_arm_agent_gazebo`（Gazebo 执行）、`so101_single_arm_agent_hardware*`（真机执行/停止）。

## 展示协议

### typed Program：直接执行、异步展示（生产默认）

Agent 通过 `/embodied/execute_program` 提交的 typed Program 使用
`execution_mode=direct_execute_async_presentation`：**机器人可能先运动，界面随后显示**。
无客户端、展示输出失败、事件发布慢或展示事件被丢弃，均不阻止合法 Program 提交，也不改变 Gateway 终态。

- 发送前仍同步持久化：先 `bind_task`（task/proposal 关联），再写入 submission intent（`may_have_submitted`），
  发送前再检查 stop。关键持久化失败时零下发；可能已提交时按 ledger 保守进入 UNKNOWN。
- `presentation` 事件 `detail` 带 `async_display: true` 与空 `receipt_token`，不再产生 `presentation_rendered`。
  TUI 校验身份与 digest 后只显示，不发送回执；带空 token 的回执为 no-op，既不解锁也不停止执行。
- `/agent/event` 由有界后台线程发布。队列满时优先淘汰展示等普通事件；terminal、stop、quarantine 等关键通知
  只在超过硬上限时丢弃。所有事件先写入 Agent ledger，物理执行事实以 Gateway 账本为准。
- TUI 渲染异步展示失败只告警，不会被当作 stop；人工 `/stop`、停止词与 Ctrl-C 语义不变。
- 迟到或重复事件不会让已终态请求重新显示为活动状态。

由于不再等待展示，“先看计划再动”的保证随之取消。真机需要预先审批时，应在发请求前通过 dry-run
（`execution_enabled=false`）或人工核对完成，不能依赖执行中的展示拦截错误计划。

### 旧 Agent Plan：展示完成回执（兼容）

`immediate_after_presentation` 仍要求完整 exact plan 展示并 flush 后才 technical confirm。
`presentation` 事件的 `detail` 包含完整 `presentation`（task/plan/registry tuple、有序步骤、预算和摘要）、
canonical `presentation_digest` 与一次性 `receipt_token`。TUI 在自己的事件循环中用底层 terminal output
打印并 flush 整个计划，再发送：

```json
{"operation":"presentation_rendered","request_id":"...","request_key":{"robot_scope":"...","channel_id":"...","principal_id":"...","request_id":"..."},"presentation_digest":"...","receipt_token":"..."}
```

服务端仅接受当前待展示计划的精确身份、摘要、token 和有效期；回执不代表用户二次批准，不授予运动权限。
`embodied.agent.presentation_timeout_sec` 默认 30 秒，必须为有限正数，仅作用于该兼容路径；无客户端、输出失败、
事件传输失败、超时、错误回执或停止请求均不能进入 confirm/goal 发送。除 typed Program 模式外的任何
`execution_mode` 都按此路径 fail closed。
`/agent/control` 是受信本地客户端边界，回执是协议证据而非对任意恶意 ROS peer 的认证。

## 语音入口（voice bridge）

`embodied.agent.voice_bridge.enabled: true` 时，`embodied_bringup` 在 Agent 节点旁启动
`agent_voice_bridge`。它与 `chat_tui` 一样只是 `/agent/*` 契约的客户端，不绕过 Agent 的规划、白名单
与 Gateway；事件身份按 profile 的 `channel_id`/`principal_id`/`robot_scope` 过滤，语音 profile 应使用
独立的 `channel_id`（如 `voice_asr`）。

```text
voice_asr_node --/voice_command(String)--> agent_voice_bridge --/agent/request(JSON)--> ibrobot_agent_node
               --/voice_partial(String)-->    |  停止关键词 ----------/agent/control stop
                                              |  presentation --(播报/日志)--> /agent/control presentation_rendered
                                              '<-- /agent/event: 终态与回答 --(TTS 可选)--> 用户
```

- `text_corrections` 先把 ASR 偏好的同音写法纠正为领域词（如“家长”→“夹爪”），再做停止词/确认词匹配与请求封装；
  纠正前后的文本都会写入日志。替换是**整句范围的字面替换**：配置的误识写法若在日常对话中也会出现
  （例如真的谈到“家长”），该句同样会被改写。只配置在本机器人语境下不会歧义的写法。
- 启用 TTS 时，播报期间（以及恢复后 `resume_settle_sec`，默认 1 秒内）收到的识别结果会被丢弃，
  避免 ASR 把机器人自己的声音或被打断的半句话当成新指令、进而反复触发播报。
- 同一通道同时只跟踪一个请求：执行中的新话语被丢弃并提示“请稍等”，停止词始终生效。
- 停止词是**关键词匹配**：句中出现 `stop_phrases` 任一词（默认 停止/别动/停/stop/halt）即中断，例如
  “帮我停一下”“别动了谢谢”。代价是“不停地挥手”这类句子在执行中也会触发停止；没有可中断的请求时，
  只含关键词的普通句子仍作为指令发送。
- 实时中断：桥接同时订阅 ASR 的 `/voice_partial`（`partial_topic`，置空关闭），部分识别结果一出现停止
  关键词就发送 stop，不必等 VAD 静音与尾部补零；同一句话只由 partial 发送一次 stop，这句话的最终文本归入
  这次停止，不会再作为新指令发给 Agent（否则会占住通道，紧随其后的「继续」被当作忙碌丢弃）。
- 请求超过 `request_timeout_sec`（默认 180 秒）仍无终态事件时，桥接先发送 stop 再释放通道并播报
  “任务超时，已停止”；在它的终态到达前，停止关键词仍指向该请求，迟到的 presentation 不再回执。
- `confirm_mode: auto`（默认，仿真与真机相同）：说完自然语言、Agent 规划完成后直接执行，无需人工二次确认。
  exact plan 渲染（TTS 播放完成；未启用 TTS 时写日志）后即回执；TTS 失败不回执，Agent 按
  `presentation_timeout_sec` 失败关闭。执行中说停止词随时中断。
- `confirm_mode: voice_confirm`（可选）：播报计划后等待确认词才回执；取消词、停止词或 `confirm_timeout_sec`
  超时发送 stop。
- 澄清回答（`needs_clarification`）后的下一句自动携带 `reply_to_request_id`；`session_idle_sec` 后开启新会话。
- 启用 TTS 时，播报期间向 `/voice_control` 发送 stop/start 暂停 ASR，避免机器人识别自己的声音。

仿真端到端验证（ASR 独立运行，经 `/voice_file_input` 注入录音；真机换用 `so101_agent_voice_hardware`）：

```bash
source .shrc_local && export ROS_DOMAIN_ID=87 && \
ros2 launch embodied_bringup embodied_pipeline.launch.py robot_config:=so101_single_arm_agent_voice_gazebo \
  use_sim:=true control_mode:=moveit_planning entry_mode:=agent authorize_motion:=true
# 另一个终端：使用默认 ASR 模型（X-ASR），手动模式只处理注入的文件
ros2 run voice_asr_service voice_asr_node --ros-args -p active_mode:=manual
ros2 topic echo /agent/event &
ros2 topic pub --once /voice_file_input std_msgs/String "{data: /abs/path/nod.wav}"
```

`/agent/event` 依次出现 `proposal_ready` → `presentation` → `presentation_rendered` → `terminal`（`SUCCEEDED`）即链路打通。

## 已知限制

- 孵化期包（`incubation: true` 强制标记），String JSON topic 契约尚未类型化。
- 会话记忆为有界滚动窗口，不做长期记忆检索。
- `chat_tui` 为单操作员本地终端，不提供多通道并发接入；Agent 节点一次只绑定一个 `channel_id`，
  语音与终端不能同时作为同一 Agent 的入口。
- UNKNOWN 恢复采用保守隔离，不自动重放或推断旧任务终态；需操作员依据保留的 TaskRef 完成核实。
