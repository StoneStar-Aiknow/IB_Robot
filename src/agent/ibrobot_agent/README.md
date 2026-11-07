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
| `agent_voice_bridge`（`ibrobot_agent_voice_bridge`） | 语音通道客户端：`/voice_command` 最终文本 → `/agent/request`；停止词 → `/agent/control stop`；播报 exact plan（仅旧 Agent Plan 路径回执）；可选 TTS 播报回复 |
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

Planner 配置 `stream: true` 时，模型回答里的 `user_message` 一闭合，Agent 就在
`/agent/event` 上发布一条 `reply_preview` 事件（`state=PLANNING`，正文在
`user_message`），每个请求至多一次，请求已停止时不发；要等 `kind` 到达才发，
`read_only` 不发（其回复由 Gateway 状态重建）。它只是**预告**：Agent 不渲染、
不播报，计划仍在完整回答到达后按原样严格校验；事件同样写入 ledger。由通道决定是否
呈现——`agent_voice_bridge` 在启用 TTS 时播报，并在最终回答与预告相同时不再重复；
预告之后计划被拒，终态照常报失败。

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
- 内置 RulePlanner 只允许在仿真模式执行；真机执行必须配置 vlm Planner。唯一例外是部署显式列出的
  `planner.direct_commands`：整句精确匹配（否定/假设句不触发）时本地生成单步计划而不调用模型，
  技能必须在 live catalog 中且 planner 可选、启用执行时还必须在 allowlist 内；之后的 Gateway
  校验与执行路径与模型规划完全相同。提案的 `planner_identity` 记为 `route=direct_command`，
  ledger 可区分这一步不是模型的决定。没有内置短语。
- Planner 输出只接受单一严格 JSON 对象；未知字段、未知 skill、编造参数一律拒绝。
- 唯一的例外是不会触发执行的回复（对话、只读查询、澄清、拒绝）在修复不产生歧义时会被修正，而不是变成
  FAILED 终态：本类型用不到的字段若是空值（`""`、`null`、`[]`）直接去掉；澄清缺 `missing_fields` 时补
  `["request"]`；拒绝的 `reason_code` 先去空白并转大写，仍不合法则回退为 `UNSUPPORTED_TASK`。修正记录
  （字段、动作、原值）写入 ledger 的 `planner_outcome` 事件 `planner_repairs` 并记 warning 日志，不进入
  digest。workflow 回复从不修正；字段与类型矛盾（如澄清带 `steps`、对话带 `summary`）仍然拒绝。
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
                                              |  presentation --(播报/日志)--> [仅旧 Agent Plan] /agent/control presentation_rendered
                                              '<-- /agent/event: 终态与回答 --(TTS 可选)--> 用户
```

- `text_corrections` 先把 ASR 偏好的同音写法纠正为领域词（如“家长”→“夹爪”），再做停止词/确认词匹配与请求封装；
  纠正前后的文本都会写入日志。替换是**整句范围的字面替换**：配置的误识写法若在日常对话中也会出现
  （例如真的谈到“家长”），该句同样会被改写。只配置在本机器人语境下不会歧义的写法。
- 启用 TTS 且没有可中断的请求时，播报期间向 `/voice_control` 发送 stop/start 暂停 ASR，播报期间（以及恢复后
  `resume_settle_sec`，默认 1 秒内）收到的识别结果会被丢弃，避免 ASR 把机器人自己的声音或被打断的半句话当成
  新指令、进而反复触发播报。
- 有可中断的请求时（执行中、等待确认、超时后等待终态）播报**不暂停 ASR**：播报期间及其后
  `resume_settle_sec` 内只放行含停止关键词的识别结果（partial 实时中断照常），其余当作回声丢弃。因此桥接在
  这段时间说的话（忙碌提示、超时提示、「继续」播报）都不含停止词，`busy_message` 含停止词会被配置校验拒绝；
  万一某句播报含停止词，该句仍按旧方式暂停 ASR 并打 warning，以免机器人听到自己说“停”而中断请求。
- **寻址门控（`wake_phrases`，默认关闭）**：平台自身的唤醒词只在**首次**武装 VAD，`only_voice` 模式此后
  保持常开，因此房间里的每一句话都会到达桥接，无论是否对机器人说的。配置 `wake_phrases` 后，一句话必须以
  其中之一**开头**才被转发；命中后唤醒词从文本中剥离（ASR 会丢掉逗号，「灵犀灵犀，挥个手」实际到达的是
  「灵犀灵犀挥个手」，planner 不应看到前缀）。命中即开 `wake_window_sec` 窗口，窗口内的后续话无需再点名，
  但**只有被真正处理的话才续期窗口**——否则旁边的持续交谈会把它一直撑开，门控退化为常开。
  句中提及不算点名（「昨天我跟灵犀说过这个事」被忽略），这是用前缀匹配而非包含匹配的原因。
  **停止词不受此门控**：安全优先于寻址。ASR 把专名读成同音词是常态（X-ASR 实测「灵犀」→「林夕」/「林溪」），
  用现成的 `text_corrections` 吸收即可，门控在纠正之后判定。
  门控实现在 `ibrobot_agent/addressee.py`（`AddresseeGate`，不依赖 ROS）：它持有窗口或会话、待用的唤醒事件、
  锚点与尚未判定的测量；`voice_bridge` 只保留停止词、确认、请求这些协议分支，每句转写先调 `open_if_new_session`
  （任何分支返回之前），走到寻址这一步再调 `judge`。
- **按段配对，不按到达时间**：每句转写、每条声纹、每个唤醒事件都带上它所属的语音段
  `(source_id, segment_sequence)`（开启该段的 BEGIN 块，见 `ibrobot_msgs` 的 `SpeechTranscript`、
  `SpeakerEmbedding`、`SpeechWakeEvent`）。声纹只判定同一段的转写；没有转写的段（ASR 丢弃了它）的声纹不判定
  任何一句，过 `speaker_max_age_sec` 后丢弃；某段的声纹丢了，只有这一句在等满 `measurement_grace_sec` 后按
  fail-open 放行，下一句照样用它自己的声纹判定。启用声纹校验或 `vendor_session` 时（`pairs_by_segment`），
  桥接改订阅 `voice_asr.transcript_topic` 上的 `SpeechTranscript`（launch 从 voice_asr 推导，`input_topic`
  此时不可配置），因此要求 `voice_asr.input_mode: external_vad`——只有自带分段的音源能说出转写属于哪一段。
  此前按到达时间配对：停止词、确认回答不消费声纹，旁人转写若先于自己的声纹到达就拿用户的声纹判定；
  停止词的声纹先到会把暂存的指令放出去；上一句的转写在新唤醒之后才到会抢走这次唤醒。按段配对后这些都不再成立。
- **两种触发方式（`addressee_trigger`）**：门控需要先认定"哪一句自证是对机器人说的"，判据取决于音源。
  - `wake_phrase`（默认）：话语必须以 `wake_phrases` 之一**开头**，适用于转写里仍带唤醒词的音源。
  - `vendor_session`：音源**先吃掉自己的唤醒词**再放行语音，唤醒词到不了文本层（X2 实测「灵犀灵犀，挥个手」
    到达时是「挥个手」），音源改以唤醒事件报告它。此模式下**不得**再配 `wake_phrases`（两套判据冲突，且前缀匹配
    会拒绝每一句话），配置校验会拒绝；必须绑定 `wake_event`，否则启动失败。
  - **只有唤醒事件能开会话**：唤醒事件指明的那一段的转写自证是对机器人说的、设置锚点；会话内的后续话逐项与
    锚点比对。静默超过 `session_gap_sec` 后会话结束，**下一句不再自动成为会话首句**，没有新的唤醒事件就一律拒绝
    （「no wake event opened a session」）。原因：X2 在没有唤醒词时也会放出房间里的话，"静默后的第一句"不构成
    任何寻址证据——旁人在安静片刻后随口一句指令，原先会被放行并设成锚点。代价：唤醒应答的 token 格式一变、
    收不到唤醒事件时，机器人对所有人失聪（失效方向是拒绝而不是放行），见 `aimdk_robot` README。
  - 唤醒事件匹配**同一音源、从被唤醒段起的第一条转写**：X2 有时先为唤醒词残留单开一段再丢弃，确认因此配在
    那一段上，真正的指令在下一段。序号更早的段（例如新唤醒之前开始、转写却晚到的一句）不是这次唤醒的首句，
    按它所在的会话判定。唤醒事件等待它的首句最多 `wake_window_sec`，过期作废，以免唤醒词之后没有可用转写时
    把会话交给很久以后开口的人。
  - **唤醒事件与转写谁先到都行**：两者走不同话题，没有先后保证。没有会话时到达的转写会暂存，等它那一段的唤醒事件
    （上限 `measurement_grace_sec`）；更晚一段的唤醒先到、或同一音源更晚的段已经路由过时不再等。唤醒事件指向的段
    （或同源更晚的段）的转写**已经路由过**时，这个唤醒来得太迟，直接作废并记日志，不会留着把会话交给下一个开口的人。
    会话内有人接管时，他的转写若先于他的唤醒到达，会按当前会话判定（通常被挡），迟到的唤醒随之作废——失效方向是
    这一句要重说，而不是放进旁人。
  - 会话存活由会话内**每一条**转写续期，**包括被校验链拒绝的那些**：音源的会话由房间里任何语音维持。
    间隔在**最终转写之间**测量，因此包含上一句的时长；`session_gap_sec` 太小，用户停顿后的追问就要重说唤醒词。
  - **会话首句无论走哪个分支都会重设锚点、消耗唤醒**：首句是停止词、已由 partial 发过停止的那句 final、规范化后为空的
    文本、或恰好回答了一个待确认的旧路径计划时，也先开会话、用它的测量重设锚点再处理（`open_if_new_session` 在
    `on_transcript` 的任何返回之前调用），否则上一位用户的锚点会继续判定新会话，没被消耗的唤醒还会落到下一个开口的人身上。
  - **唤醒事件（`wake_event`）**：配置 `wake_event: {interface: audio.wake_event, type: ibrobot_msgs/msg/SpeechWakeEvent}`，
    桥接订阅 runtime 声明的唤醒事件话题（X2 由 `aimdk_robot` 从厂商助手的应答推导，见其 README）。会话内有人说
    唤醒词时，他的那一段成为新的会话首句，于是换人只需说一次唤醒词。话题名来自 runtime 接口描述，不在本配置里重复。
- **锚点校验链（`speaker_check_enabled`，默认关）**：唤醒词开窗之后，
  窗口属于「接下来说话的人」还是「开窗的那个人」，仅凭文本无法区分。校验链解决这一层：说出唤醒词的那句
  **自证**是对机器人说的，它不参与校验，而是**设置锚点**（声纹向量）；窗口内的后续话逐项与
  锚点比对，全部通过才转给 Agent。下一次唤醒词会用说话人自己的测量**重设**锚点，所以换人只需点一次名。
  目前只有声纹一项；方向校验需要声源方位数据，X2 没有可用来源，未实现（`AnchorCheck` 协议留有扩展位）。
  - **声纹锚点随会话累积**：窗口内每一句**全部检查都通过**的话，其声纹以等权平均并入锚点；被拒绝的话
    不改动锚点。原因是单句声纹噪声大：X2 实测厂商把锚点句截到 3 秒时，说话人自己的后半句只得 0.59 而被
    误拒，同期另一人得 0.56；用真机数据回放，累积后同一句升到 0.67、插话者最高 0.44。
    **只有明显像本人的话才并入**：相似度须至少高出阈值 `speaker_merge_margin`（默认 0.10），刚过阈值的话
    照常放行但不改动锚点。2026-10-25 真机实测，两段扩音演讲录音中一段 0.66 的追问并入后，把另一位演讲者从
    0.59 抬到 0.66 而放行；按当天 37 条真机声纹回放，余量 0.10 把误放从 1 降到 0、误拒不变（2）。
    新会话从第一句重新锚定。
  - **确认/取消回答不受校验链约束**（与停止词同理）：只发生在旧 Agent Plan 的 `voice_confirm` 路径上，
    typed Program 不等回执、不进入待确认状态。
  - **测量缺失即放行（fail-open）**：传感器坏掉不能让机器人变聋；误拒（忽略自己的用户）比误放严重
    得多。两个向量不可比较（`SpeakerEmbedding.model_id` 不同，或维度不同）也算「不是证据」而非不匹配：
    放行，且不并入锚点。同维度换模型时数值看起来照样合理，只有 `model_id` 能看出来。
  - **太短而没有声纹的后续话可配置为挡住（`speaker_reject_short`，默认 false）**：`speaker_id_node` 对短于
    `min_speech_seconds` 的段发空向量。默认按上一条放行；开启后，**已有锚点时**这类话被挡。原因是 2026-10-26
    X2 真人实测：两个人实际说的指令（「挥手」「握手握手」）只有 1 s 左右人声，12 句里 9 句没有声纹，第二个人的
    指令全部被放行。代价是用户本人的短指令也要多说几个字。会话首句、停止词、确认回答不受影响；声纹消息根本没到
    （节点挂了）仍按 fail-open 放行。
  - **首句没有声纹时整窗放行**：会话首句太短（不足 `min_speech_seconds`）或只是唤醒词残留时没有声纹，
    本会话不设锚点，窗口对任何声音放行，直到下次唤醒。**不**拿之后某一句来补设锚点：首句之后先开口的可能是
    旁人，锚点落到他身上会让机器人整段会话不理自己的用户——那比放进一句插话严重。
  - **只在必要时等待声纹，且声纹一到就判定**：只有「要和锚点比对、而本段声纹还没到」的转写才暂存（锚点本身
    还在等会话首句的声纹时也暂存）；停止词、确认回答、会话首句（自证，不比对）和没有锚点时都不等。首句先放行，
    它那一段的声纹到达即设为锚点。暂存的转写在本段声纹到达时立刻判定，`measurement_grace_sec` 只是等待上限。
    `speaker_id_node` 对每段人声发一条消息，过短的段发空向量，表示「这段没有声纹」，桥接收到即放行，不必等满上限。
    X2 实测（2026-10-25，93 句）：声纹中位数比转写早 0.14 s 到达，此前一律等满 1.5 s 时转写到判定中位数 1.6 s。
    暂存队列在 `VoiceBridgeCore` 里（`on_heard` / `release_held`），按到达顺序放行。**停止词（final 或 partial）
    取消所有仍在暂存的转写**：它们是停止之前说的，停止之后再发出去会让机器人刚被叫停就动；取消了暂存句的
    停止词即使当时没有在执行的请求，也按停止处理，不会变成一条新指令。停止词自己的声纹只属于停止词那一段，
    不会把暂存的指令放出去。停止词是关键词匹配，「不停地挥手」这类句子同样会取消暂存句，与下文的取舍一致。
  - **声纹（`speaker_check_enabled`）**：由 `voice_asr_service` 的 `speaker_id_node` 在**同一条音频流**上
    并行提取，每段人声发一条 `SpeakerEmbedding`。它是独立节点而非 ASR 内的一步，是时延决定的：向量在人声结束时
    发布，转写还要等 VAD 静音加尾部补静音（默认 1.5 s），所以通常不必等。模型在某一段上推理出错时节点不发布
    这一段（该句按缺失放行）、连续出错时重新加载部署，节点本身不退出。
    向量只在进程内存与消息里存在，**不落盘、不跨会话保留**（生物特征数据）。
    `speaker_min_similarity` 默认 0.60，即 X2 真机测试所用值（旁人与锚点 0.28–0.56，用户本人累积后
    0.67–0.84），但它**必须用真人录音现场标定**：取决于模型、麦克风、房间以及房间里的人有多像；
    照搬 benchmark 数字的结果是「忽略用户」或「放进整个房间」。X2 上的完整配置见 `robot_config` 的
    `aimdk_x2_voice_addressee.yaml`。2026-10-27 X2 复测：厂商切出的段自带余量，「挥手」「鼓掌」整段 1.66 s，
    有声纹，对本人锚点 0.63–0.70；旁人 1.5 s 片段 0.38–0.49、长句 0.34–0.48。
    **标定方法**：在部署现场，让每个人经机器人自己的麦克风说一句开场和几句后续，记录 `/voice/speaker_embedding`
    的向量，比较「同一人跨句」与「不同人」两组余弦。两组重叠就说明没有可用阈值，应保持
    `speaker_check_enabled: false`——这是真实结论，不是调参问题；否则把阈值放在间隙里、靠近同一人的下限
    （误拒会让机器人不理自己的用户，代价更大）。标定前先在干净音频上确认每个标签组内确实是同一个人：
    标签错误看起来和「模型分不开声音」一模一样。录下的向量是生物特征数据，定完阈值即删除。
- **语义闸门（`semantic_gate_mode`，默认 off）**：唤醒词与声纹回答的是「谁在说、是不是刚被点名」，回答不了
  「这句话是不是对机器人说的」。会议室里没人叫它，任何一句闲聊都会进 planner——实测「感谢各位领导莅临」
  会被规划成动作。语义闸门在转写文本上补这一层：字符 1–3gram + 逻辑回归，进程内纯 Python 打分，
  **45 µs/句、无新依赖**，模型是 200 KB 的 JSON。
  - **它是否决器，不是判官**：只在「分数 < `semantic_gate_threshold`（默认 0.10）**且** 句中没有任何
    catalog 技能词 **且** 不以（被 ASR 打烂的）唤醒词开头」时才拦。停止词、确认/取消、bypass 短语在它之前
    处理，永不被拦——安全路径不经过任何模型。技能词表由
    `python3 tools/addressee/emit_lexicon.py --yaml` 从 `skill_catalog` 的 `aliases_zh` 生成，新技能不必重训模型。
  - **`shadow` 先跑一段时间**：判定照做、日志照写（`semantic_gate_log_path`，JSONL），但一句都不拦。
    板上部署 profile 现在就是这个态。日志里的每一行都是带模型的判词、分数与上下文（`after_wake`、
    `active_request`），是下一版模型的标注队列，而不是单边抽样：被放过的错误（该拦没拦）同样在里面。
  - **`enforce`** 才真正丢弃，用一个 `Log` 说明「not addressed to me (semantic …)」，与声纹拒绝同一措辞。
  - **模型加载失败默认 fail-open**（`semantic_gate_fail_open: true`）：闸门自己坏掉不能让机器人变聋；
    要让它成为启动期的硬约束，设 `false`，launch 会失败在加载阶段而不是静默降级。
  - 训练/评测/词表在 `tools/addressee/`（README 里有完整命令与实测数字）；设计与实测报告见
    `docs/aimdk_x2_addressee_semantic_gate_design_zh.md`。语料上 639 句分组 5 折：阈值 0.10 时准确率
    0.930、指令召回 0.891、闲聊拦下 0.949；加上技能词表与唤醒前缀后指令召回升到 0.962。留一技能测试
    （新技能从没训过）召回只有 0.45–0.67，所以它是否决器——放行是默认，拦是例外。
  - **已知边界**：词袋模型按词汇重叠泛化。语料外的闲聊如果用了机器人常用词（“今天天气不错”“你好”）
    分数反而很高，会被放行；这与语料的负样本多是会议长句有关——所以第一阶段是 `shadow`，
    用现场日志（尤其唤醒之后跟同事聊的那部分）重训，再决定是否 `enforce`。
- 同一通道同时只跟踪一个请求：执行中的新话语被丢弃并提示“我正在执行上一个任务，请稍等。”。
- 停止词是**关键词匹配**：句中出现 `stop_phrases` 任一词（默认 停止/别动/停/stop/halt）即中断，例如
  “帮我停一下”“别动了谢谢”。代价是“不停地挥手”这类句子在执行中也会触发停止；没有可中断的请求时，
  只含关键词的普通句子仍作为指令发送。
- 实时中断：桥接同时订阅 ASR 的 `/voice_partial`，部分识别结果一出现停止
  关键词就发送 stop，不必等 VAD 静音与尾部补零；同一句话只由 partial 发送一次 stop，这句话的最终文本归入
  这次停止，不会再作为新指令发给 Agent（否则会占住通道，紧随其后的「继续」被当作忙碌丢弃）。
  `/voice_partial`、`/voice_status`、`/voice_control` 由 `voice_asr_node` 固定，不在 YAML 中配置
  （`partial_topic` 只作为 ROS 参数保留，置空可关闭实时中断）。
- 请求超过 `request_timeout_sec`（默认 180 秒）仍无终态事件时，桥接先发送 stop 再释放通道并播报
  “任务超时，正在中止。”；在它的终态到达前，停止关键词仍指向该请求，迟到的 presentation 不再回执。
- 「继续」由 Agent 处理（见下节），桥接只把它当普通话语发送，并播报 Agent 的 `continuation` 事件。
- `reply_preview` 事件（流式 Planner 的提前回复）只对当前活动请求播报一次；终态回复与它相同则不再重复。
  桥接已对该请求发出 stop 后到达的预告不播报（Agent 可能尚未收到 stop）。typed Program 的
  “正在执行：…”在已有预告时不再播报（预告已经说过要做什么，这一句没有回执在等）；旧回执路径仍播报
  exact plan，因为它的回执表示用户听到的是计划本身，而预告只是模型的文字。
- **typed Program（真机与仿真经 Gateway 的默认路径）不等待任何回执**（见上文「typed Program：异步展示」）：
  `presentation` 事件带 `async_display: true` 时，桥接只播报（无 TTS 时写日志）“正在执行：…”，不回执、
  不询问确认——机器人此时可能已经在动，再问“请说确认或取消”只会误导用户。语音通道对运动的控制手段是
  **停止词**；需要预先审批的真机场景按上文用 dry-run 或人工核对，不能依赖语音确认拦截计划。
- `confirm_mode` 只对旧 Agent Plan 的 `immediate_after_presentation` 回执路径生效；typed Program 下配置
  `voice_confirm` 时，桥接在每个计划处记录“voice_confirm cannot gate …”日志。未配置时随 TTS 取默认值：
  启用 TTS 为 `auto`，未启用 TTS 为 `voice_confirm`。
- `confirm_mode: auto`（旧路径）：exact plan 渲染完成（TTS 播放完成）后即回执；TTS 失败不回执，Agent 按
  `presentation_timeout_sec` 失败关闭。**未启用 TTS 时唯一的“渲染”是写日志**，节点启动时会打印 warning。
- `confirm_mode: voice_confirm`（旧路径）：播报（无 TTS 时写日志）计划后等待确认词才回执；取消词、停止词或
  `confirm_timeout_sec` 超时发送 stop。含否定词（不/别）的回答（如“不执行”“我不确定”“别执行了”）即使包含
  确认词也**不回执**，而是再次询问；取消词优先于确认词（“不要执行”即取消）。
- 仓库自带的两个语音 profile（`so101_agent_voice_hardware`、`so101_single_arm_agent_voice_gazebo`）走
  typed Program 路径且未启用 TTS：计划只写入桥接日志，机器人规划完即执行，用于操作员盯着终端日志的联调场景。
- 停止词、「继续」词表与文本闭环客户端、Hermes 共用 `embodied_common.control_phrases`，各通道只决定匹配
  方式（本桥接为关键词，文本闭环为整句）。
- 澄清回答（`needs_clarification`）后的下一句自动携带 `reply_to_request_id`；`session_idle_sec` 后开启新会话。

### 「继续」：按账本恢复被中断的任务

Agent 在规划前识别 `继续/继续吧/接着来/go on/continue`（与 `robot_skill_cli` 的 closed grammar 同一词表，
仅对整句生效，“继续挥手”是普通指令），然后查两本账：

1. Agent 账本：同一 `session_id`（及 channel/principal/robot_scope）下最近一个非问答、非准入拒绝的请求，
   从其绑定的 proposal 取原始指令文本。
2. Gateway 账本：对 `CANCELLED` 的请求按同一 `request_id` 调 `GetExecution`，要求 `CANCELED`、execution_id
   与 Agent 记录一致，并取 `completed_step_count` 与 `canonical_steps`。

只有确认安全取消（或尚未提交就停止）才继续：Agent 用原始文本和“已完整完成 N/M 步、第 N+1 步需重做”的提示
**重新规划**，以「继续」这条请求自己的 request_id 提交一个全新 Program；不切片、不复用旧计划（被打断步可续做时
例外，见下文「步内续做」），Gateway 照常校验。播报“上次任务已完整完成 N/M 步，现在继续剩余部分。”（播报时请求可被中断，故不含停止词）。`SUCCEEDED` 回答已完成；`FAILED`、`UNKNOWN`、
规划阶段取消、Gateway 查询失败或两本账不一致一律拒绝（`CONTINUE_REFUSED`），不产生任何运动。

这是已确认的语义。`robot_skill_cli` README 与 `ibrobot-control` skill 中“不得切片旧步骤、拒绝 `--resume`”约束的是
闭环客户端；Agent 侧的「继续」提交的是经过完整校验的新 Program，不在其禁止范围内（两处文档已注明）。

两种规划器的做法不同：VLM Planner 依据 continuation hint 自行规划剩余部分；离线 RulePlanner（仅仿真/测试用）
对原文重新规划后，按累计 `completed_step_count` 去掉已完整完成的前 N 步（`steps[N:]`）。后者是对**新规划结果**
按计数截取，不复用旧 plan，生成的 Program 同样经过 presentation、白名单与 Gateway 校验。

**步内续做**：Gateway 报告被打断的那一步可续做（`step_resumable`，目前只有 X2 在 `stream` 模式下回放的手势）
时，「继续」**不调用规划器**：Agent 直接把 Gateway 账本里从被打断步开始的规范步骤
（`canonical_steps[completed_step_count:]`）作为新 Program 提交，带 `continues_from_execution_id`，该步从停下处
接着做（Gateway 与运行时会再次校验，失效则从头做）。这些步骤仍经过白名单、catalog 与 schema 版本校验；proposal
的 `planner_identity.route` 为 `continuation_ledger_replay`，请求账本记 `continuation_replayed` 事件。播报“上次任务被
打断的动作已做到 40%，现在从中断处接着做。”。X2 实测「继续」到手臂开始动由约 5.7 s（规划）降到亚秒级。
不可续做的「继续」（步间停下、断点失效、尚未提交即停止）仍按原话重新规划。停止后若
先下了别的指令，「继续」指向的是那条新指令，被打断的任务即作废。

限制：其他技能的 `completed_step_count` 是下界——Gateway 不记录它们的步内进度，被打断的那一步会从头重做，
非幂等技能（如已抓起再抓）需要用户自行判断；规划器看不到世界状态。「继续」只在同一会话内有效，语音通道
`session_idle_sec`（默认 120 秒）无活动后换新会话即无可继续的任务。连续“停—继续—停—继续”时，
已完成步骤跨请求累计。

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

`/agent/event` 依次出现 `proposal_ready` → `presentation`（`async_display: true`）→ `terminal`（`SUCCEEDED`）即链路打通；
typed Program 路径不产生 `presentation_rendered`。

## 已知限制

- 孵化期包（`incubation: true` 强制标记），String JSON topic 契约尚未类型化。
- 会话记忆为有界滚动窗口，不做长期记忆检索。
- `chat_tui` 为单操作员本地终端，不提供多通道并发接入；Agent 节点一次只绑定一个 `channel_id`，
  语音与终端不能同时作为同一 Agent 的入口。
- UNKNOWN 恢复采用保守隔离，不自动重放或推断旧任务终态；需操作员依据保留的 TaskRef 完成核实。
