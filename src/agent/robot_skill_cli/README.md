# robot_skill_cli

`robot_skill_cli` 提供约束 LLM/Agent 对 ROS 访问面的命令行工具集，包含三条受控入口：

```text
运动控制:   Hermes -> ibrobot-control Skill -> robot-skill [-> robot-skill-daemon] -> ROS Capability Gateway
感知读取:   Hermes -> ibrobot-perceive (硬编码 allowlist)              -> ros2 topic echo --once
裸 ROS 拦截: Hermes pre_tool_call -> ibrobot-block-raw-ros hook        -> block ros2/rclpy/roslaunch
```

- `robot-skill`：运动技能请求的唯一受控入口，限定在 Capability Gateway 公开边界内，不直接调用
  primitive、MoveIt、controller 或裸 ROS 运动接口，也不修改 Gateway 的授权和安全策略。
- `ibrobot-perceive`：只读感知读取的唯一受信入口，硬编码 source/field allowlist + 审计日志，
  有界超时；不经 Gateway，不初始化 `rclpy`。
- `ibrobot-block-raw-ros`：Hermes `pre_tool_call` 防御 hook，拦截裸 `ros2`/`rclpy`/`roslaunch`
  调用，是 defense-in-depth，权威边界仍是 `authorize_motion` 与 Gateway plan validation。

运动控制始终走 Gateway；感知读取走受控 allowlist；裸 `ros2` 调用被 hook 拦截。

## 环境与配置

在工作区根目录加载环境：

```bash
source .shrc_local
```

`.shrc_local` 是 ROS+venv+install overlay 的 SSOT 入口（见 `AGENTS.md` 环境初始化章节），无需重复 `source install/setup.bash`。

`run-workflow` 会先完成 exact plan 的展示回调并 flush 输出，再执行内部技术 confirm；展示失败或收到停止请求时不会
进入运动执行。`immediate_after_presentation` 只取消二次用户确认等待，不绕过 Gateway validation、operator
`authorize_motion` 或 action admission。可选 `--request-id` 是调用方幂等键；timeout 或结果未知时不得自动重放。

`run-workflow` 底层的 `InteractiveController` 也可作为库级集成入口。展示回调收到的
presentation dict 除步骤、plan digest、registry identity 与 task ID 外，还包含 `plan_kind` 和
`proposed_task_budget_sec`（fresh status 的任务预算快照）。库调用方可传入构造参数 `submission_callback`，在
goal 提交前收到展示 tuple（task/plan ID、plan digest、registry identity、expected step count）；回调抛错会以
`SUBMISSION_PERSISTENCE_FAILED` 终态阻断 goal 提交。`run(..., expected_registry_identity=...)` 会在 planning
前校验 live catalog 身份，与快照不一致时按 `SKILL_SNAPSHOT_DIGEST_MISMATCH` fail closed，不发送 goal。

当前 `ibrobot_agent` 生产路径直接通过 `RosBridge` 使用 `/embodied/execute_program`，由
Program Gateway 负责一次性 admission、ExecutionStore 幂等账本和 child Skill 派发。
CLI 仍保留 `InteractiveController`，用于展示、停止、Program 终态收敛以及旧 runtime/test
bridge 的兼容路径；旧 Agent Plan 接口不再是生产 Agent 主调用链。

`ib_trace` 的阶段字段与 `[bind]` 标识符经过有界清洗；日志用的清洗不会改写请求幂等键或
提交给 Gateway 的 plan/task identity。

每个命令可用 `--config-name NAME` 选择配置，或用 `--config-path PATH` 指向 YAML；两个 flag 在 CLI 中互斥。
配置解析完全复用 `robot_config.resolve_robot_config_path()`，CLI 不维护第二套路径优先级：底层选择顺序是
显式 path、显式 name、`ROBOT_CONFIG`、`ROBOT_NAME`、默认 `so101_single_arm`。按名称先查安装目录，再查
源码 `config/robots/`；显式 path 必须存在。

### 常驻 ROS bridge

每次调用 `robot-skill` 都是新进程。原先每个进程都要新建 ROS 节点，并等待已运行的 Gateway 发现它（Fast DDS 下
约 1 s，偶发 2–3 s），然后才能发出第一个请求。现在需要访问 Gateway 的命令会把 ROS 传输交给常驻进程
`robot-skill-daemon`，复用它已建立并完成发现的连接。daemon 的生命周期由 embodied pipeline launch 管理，
`robot-skill` 默认只连接、不自行拉起后台进程：

- **随机器人启动**：`embodied_pipeline.launch.py` 在启动 Skill Gateway 的同时启动
  `robot-skill-daemon --config-path <本次启动的配置快照> --replace --idle-timeout 0`。它按与 `robot-skill`
  相同的方式解析该配置的 Gateway 接口，提前建立连接，并发送只读 status 探测直到 Gateway 应答，因此启动后的第一次
  调用也无需等待发现。它随 launch 退出；启动时若已有按需启动的旧 daemon 占用同一 socket，会先让其退出再接管。
  `embodied.resident_cli_bridge: false` 可关闭这一启动项。
- **默认只连接**：没有正在运行的 daemon 时（例如未用 embodied pipeline、关闭了启动项、在另一台机器上执行 CLI），
  `robot-skill` 直接使用进程内 bridge，行为与原先相同，不会在后台留下任何进程。
- **按需启动（显式开启）**：设置 `IBROBOT_ROBOT_SKILL_DAEMON=spawn` 后，没有 daemon 时第一次调用会在后台启动
  daemon，连续空闲 15 分钟（`IBROBOT_ROBOT_SKILL_DAEMON_IDLE_SEC`）后自动退出；适合不经 embodied pipeline、又需要
  连续调用的会话。
- 只有 ROS 传输在 daemon 中执行；参数解析、catalog 加载、展示、停止状态和 stdout 输出仍在 CLI 进程内，
  命令行为与输出契约不变。CLI 进程退出不会取消已提交的 goal，与原先一致。
- 每个 Gateway 传输配置共享一个 DDS participant；调用方 token（`IBROBOT_EXECUTION_TOKEN_ROBOT_SKILL_CLI`）按每个
  客户端会话传入，只用于该会话的请求。daemon 启动时清除继承来的 token 等私有环境变量（包括 launch 环境）。
- 停止（`--stop`）或被新 daemon 接管（`--replace`）时，daemon 立即释放 socket、不再接受新连接，但继续服务已连接的
  命令直到它们结束（最长 10 分钟），正在执行的命令仍能观察并取消自己的动作。launch 退出（SIGINT/SIGTERM）时
  Gateway 也随之退出，daemon 最多再等 3 秒。
- 如果 daemon 仍意外退出，CLI 中未完成的 goal 立即以 `ROS_UNAVAILABLE` 失败，命令走原有的传输失败路径：按任务 id
  的状态查询、取消和执行记录查询会改由进程内新建的 bridge 发出，因此仍能取消并确认动作终态；新的 goal 不会改走
  进程内 bridge。
- 每个客户端有独立的发送队列，不读输出的客户端不会阻塞其他客户端的回调；Gateway 未应答时，建立连接最多等 5 秒
  预热，不会卡住其他配置或后续命令。
- daemon 按用户隔离：socket 位于权限 `0700` 的 `/run/user/<uid>/ibrobot/`（该目录不存在时为
  `/tmp/ibrobot-<uid>/`，可用 `IBROBOT_ROBOT_SKILL_DAEMON_DIR` 覆盖），不依赖 `XDG_RUNTIME_DIR`，并校验对端 uid。
  socket 名由决定 ROS 图与安全身份的变量（`ROS_DOMAIN_ID`、`RMW_IMPLEMENTATION`、`ROS_LOCALHOST_ONLY`、发现范围
  与 discovery server、`CYCLONEDDS_URI`、Fast DDS profile 文件、`ROS_SECURITY_*`）以及已安装代码共同决定：不同 domain、不同 RMW
  或重新构建后会使用新的 daemon。只影响本进程传输细节的变量（如 launch 统一设置的 `FASTDDS_BUILTIN_TRANSPORTS`）
  和 `PYTHONPATH` 的写法不参与，因此 launch 启动的 daemon 与用户 shell、Hermes 包装脚本中的 CLI 使用同一个 socket。
- daemon 无法启动或连接时自动回退到进程内 bridge，行为与原先相同，只是没有时延收益。
- `IBROBOT_ROBOT_SKILL_DAEMON` 取值：未设置或 `1` 只连接已运行的 daemon；`spawn` 允许按需启动；`0` 完全关闭
  （launch 启动项也会直接退出）。`robot-skill-daemon --stop` 停止
  当前环境对应的 daemon；按需启动的 daemon 日志写在 socket 同目录的 `.log` 文件（超过 1 MiB 时轮转为 `.log.1`），launch 启动的 daemon 输出在
  launch 日志中。pytest 下默认关闭（根目录 `conftest.py`）。
- `ibrobot_agent` 的自然语言 Agent 路径不经过 `robot-skill`，它在节点内一直持有自己的 `RosBridge`，不受影响。

## 命令

| 命令 | ROS | 用途 |
|---|---:|---|
| `list-skills` | 否 | 列出所有启用的高层技能及公开描述 |
| `list-games` | 否 | 列出配置中已启用且允许 Agent 触发的视觉游戏 |
| `describe-game GAME` | 否 | 查看视觉游戏输入、结果 schema、timeout 和配置摘要 |
| `describe SKILL` | 否 | 查看参数 schema、单位、语义描述和 timeout policy |
| `list-poses` | 否 | 列出公开命名位姿 |
| `status` | 是 | 读取 Gateway 授权、控制模式、readiness、busy 和 ledger 状态 |
| `validate SKILL` | 是 | 本地 schema 检查后调用 Gateway 安全校验，不执行动作 |
| `execute SKILL --task-id ID` | 是 | 通过 `SkillCommand` 执行高层技能 |
| `cancel --task-id ID` | 是 | 以同一 deterministic goal UUID 请求取消并轮询 terminal |
| `reload-catalog --request-id ID --force` | 是 | 重新编译并原子激活 Gateway 已配置的 Skill catalog source |
| `plan-workflow --text TEXT --workflow-json JSON --request-id ID` | 是 | 旧协议诊断入口：提交 typed AgentPlan |
| `run-workflow [--request-id ID] --text TEXT --workflow-json JSON` | 是 | 展示后通过 ExecuteProgram 自动执行 |
| `validate-plan --plan-token TOKEN` | 是 | 对 exact snapshot 计划做只读逐步预检 |
| `confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID [--timeout-sec SEC]` | 是 | 校验身份/摘要/task_id 并冻结 `task_budget_sec`，转入 `CONFIRMED` |
| `execute-plan ... --plan-id ID --plan-digest DIGEST --registry-* ... --expected-step-count N` | 是 | 执行已确认的 Agent plan，并以展示过的 tuple 校验终态 |
| `cancel-plan --task-id ID --plan-id ID --plan-digest DIGEST --registry-* ... --expected-step-count N` | 是 | 请求取消并以展示过的 tuple 校验终态 |
| `robot-skill-closed-loop ...` | 是 | 展示 Workflow 后立即执行，并验证「别动」和安全 continuation 门禁 |
| `start-game GAME --request-id ID` | 是 | 以调用方 ID 幂等发起视觉游戏 |
| `game-result --request-id ID` | 是 | 查询视觉游戏的 pending/terminal 结果 |
| `ibrobot-perceive --source SOURCE --field FIELD [--config-name NAME]` | 否 | 只读感知 source 读取（硬编码 allowlist + audit log），LLM 唯一受信入口 |

`ibrobot-perceive` 是独立的 console script，不经过 Gateway，也不初始化 `rclpy`。它以硬编码
source/field allowlist 限制可读取的感知量，通过 `ros2 topic echo --once` 读取 YAML 输出并打印请求
字段的裸字面量值（供 LLM 直接读取并注入 `workflow_json`），不遵循下文的 JSON envelope 输出契约；
任何非 allowlist 的 source/field 都会被拒绝并写入 `/tmp/hermes-perceive.log`。当前 allowlist 包含
`voice_direction`（topic `/voice/speech_direction`，字段 `azimuth_rad`、`seq_id`）和
`arm_joint_position`（字段 `position`），扩展必须修改源码，不接受 config.yaml 覆盖。

`--source` 是语义别名，不是 ROS topic 名；config-backed source 的实际 topic 在运行时从 `robot_config`
解析，复用 `resolve_robot_config_path()`，CLI 不维护第二套路径优先级：

- `voice_direction`：`voice_asr_service` 固定契约，topic 写死为 `/voice/speech_direction`，不读 robot_config。
- `arm_joint_position`：topic 取自 `robot_config` 的 `moveit.joint_state_topic`，缺失时回退 `/joint_states`。
  so101_single_arm 解析为 `/joint_states`；lekiwi_handeye_realsense_grasp 解析为
  `/arm_joint_state_broadcaster/joint_states`（手臂关节，不含底盘轮子）。`--config-name`/`--config-path`
  语义与 `robot-skill` 一致且互斥；省略时按 `ROBOT_CONFIG`/`ROBOT_NAME`/默认 `so101_single_arm` 解析。
  安全边界是硬编码的 *field* 集合，不是 topic 名——topic 名是机器人级配置，不是安全敏感面。

`arm_joint_position.position` 返回原始弧度数组，不包含关节名映射，不得编造关节名称、重排数值或把
数组索引解释为具体关节。

`ros2 topic echo --once` 返回下一条已发布消息的单次点时值，不是持久快照。对
`/voice/speech_direction` 这类事件型 topic，发布方不活跃时会在超时（5s）内取不到值；取到的值在
消费时可能已经过期。该字面量在 `plan-workflow` 时冻结进 plan digest，后续按 frozen plan 语义审计；
执行结果以真实运动为准，不自动重试或修正。

catalog-only 命令不初始化 `rclpy`，只读取本地归一化配置。runtime 命令只访问 Gateway status、
`ValidateSkill`、`SkillCommand`、`ValidatePrimitive`、`PrimitiveCommand`、Agent plan services/actions 和标准
`CancelGoal` 接口；视觉游戏 runtime 命令只访问 start/result 服务。Hermes 启动前会先验证四个版本化
Skill/Primitive 公共接口，再检查 Gateway/Agent 接口。

```bash
robot-skill --config-name so101_single_arm list-skills
robot-skill --config-name so101_single_arm describe move_relative_ee
robot-skill --config-name so101_single_arm list-poses
robot-skill --config-name so101_single_arm status
robot-skill --config-name so101_single_arm reload-catalog \
  --request-id reload-001 --force

robot-skill --config-name so101_single_arm validate move_relative_ee \
  --motion-direction forward --motion-distance 0.03

robot-skill --config-name so101_single_arm execute move_relative_ee \
  --task-id task-20260725-001 \
  --motion-direction forward --motion-distance 0.03

robot-skill --config-name so101_single_arm cancel \
  --task-id task-20260725-001

robot-skill --config-name so101_single_arm plan-workflow \
  --request-id plan-request-001 --text "打开夹爪" \
  --workflow-json '[{"schema_version":1,"skill_name":"open_gripper_skill"}]'
robot-skill --config-name so101_single_arm validate-plan \
  --plan-token PLAN_TOKEN
robot-skill --config-name so101_single_arm confirm-plan \
  --plan-token PLAN_TOKEN --plan-digest PLAN_DIGEST --task-id agent-task-001
# 可选 --timeout-sec 30 冻结一份小于等于 Gateway task budget 的预算
robot-skill --config-name so101_single_arm execute-plan \
  --plan-token PLAN_TOKEN --confirmation-token CONFIRMATION_TOKEN --task-id agent-task-001 \
  --plan-id PLAN_ID --plan-digest PLAN_DIGEST \
  --registry-epoch REGISTRY_EPOCH --registry-generation REGISTRY_GENERATION \
  --registry-digest REGISTRY_DIGEST --expected-step-count 1
# execute-plan 必须传 confirm 时使用的同一 --timeout-sec；省略时两端都默认 Gateway task budget
robot-skill --config-name so101_single_arm cancel-plan \
  --task-id agent-task-001 --plan-id PLAN_ID --plan-digest PLAN_DIGEST \
  --registry-epoch REGISTRY_EPOCH --registry-generation REGISTRY_GENERATION \
  --registry-digest REGISTRY_DIGEST --expected-step-count 1

robot-skill --config-name so101_single_arm list-games
robot-skill --config-name so101_single_arm describe-game sorting_hat
robot-skill --config-name so101_single_arm start-game sorting_hat --request-id game-20260803-001
robot-skill --config-name so101_single_arm game-result --request-id game-20260803-001
```

### Wire schema 与原子部署

`SkillCommand.Goal`、`PrimitiveCommand.Goal`、`ValidateSkill.Request` 和 `ValidatePrimitive.Request` 的首字段都是
`uint32 schema_version`，当前 v1 公共 wire 合同必须显式发送 `1`。`WorkflowStep` 也必须携带显式版本：旧的非导航
步骤使用 v1，导航步骤使用 v2；CLI 会拒绝缺少版本的导航 typed step，并且不会根据 `domain` 推导或改写版本。

IDL、生成的 ROS 接口、`embodied_common` wire preflight、执行器、CLI 和 Agent skill 文档必须作为一个版本化发布单元
原子部署。启动时发现生成接口不是同一版本会在创建 ROS client、subscription、action 或 readiness 之前失败；不要把
旧生成接口与新节点或新 CLI 混装。

可用 typed flags 包括 `--target-name`、`--place-name`、`--motion-direction`、`--motion-distance`、
`--arm-side`、`--imitation-duration-sec` 和 `--timeout-sec`。CLI 根据当前技能显式
`capability.parameters` schema 拒绝缺失参数和不属于该技能的参数。

`imitate_human_motion` 在 `lekiwi_handeye_realsense_grasp_lidar` profile 中公开，调用示例：

```bash
robot-skill --config-name lekiwi_nav_grasp validate imitate_human_motion \
  --arm-side auto --imitation-duration-sec 5
```

该命令只进入现有 Gateway；内部 HRI Action 不是 CLI 公共入口。

### 公开 catalog 字段

`list-skills` 输出 `robot_name`、`config_digest` 和技能数组；每个技能只含 `name`、`summary`、`domain`、
`moves_robot`、`required_control_mode`。`describe SKILL` 在这些字段之外输出该技能的 `parameters`、
`recovery_policy`、运动能力 `timeout_policy` 和 `config_digest`。`list-poses` 只输出命名位姿名称，不输出坐标。
primitive sequence、目标绑定、关节值和 ROS transport 名称不属于 CLI catalog。

`list-games` 只公开已启用游戏的 `name`、`summary`、`result_field` 和视觉游戏 `config_digest`；
四个视觉游戏命令使用独立的轻量配置上下文，不编译运动 Skill catalog，也不要求运动服务或
机器人 description 包才能完成发现、启动和查询。
`describe-game` 进一步公开 required inputs、结果 schema、timeout、retention 与 ledger capacity。视觉游戏不属于运动 capability，因此不进入
`list-skills`、`ValidateSkill` 或 `SkillCommand`。

### 视觉游戏与 TTS

`start-game` 使用调用方提供的 request ID，通过视觉游戏控制服务幂等发起请求，不等待 VLM 结果；
在 advertised retention 窗口内，同 ID、同游戏的重复 start 返回原请求且不重复执行，同 ID、不同游戏被拒绝。
记录过期后 Gateway 不再保留该 ID，也不再保证检测重复；调用方应始终生成全局唯一 ID。`game-result` 按 ID
查询 Gateway 保存的 pending/terminal 结果；
pending 超过配置 deadline 会收敛为 `GAME_RESULT_TIMEOUT`。`sorting_hat` 成功终态中的 `scene_summary` 已由
perception response contract 约束为四学院之一；没有清晰可见的人时 Gateway 返回 `NO_PERSON` 失败终态，
不会向调用方暴露可播报的 `scene_summary`。Agent 仍应轮询至 `terminal=true` 获取结构化结果，但不得把结果
再次交给自身 TTS；运行时统一由
`VisualGameEvent -> visual_game_announcer_node -> /voice_tts/synthesize -> /voice_tts/play`
完成本机合成和播放。TTS 或播放服务不可用时跳过，CLI 不负责声卡播放。
`start-game` 的 accepted 仅表示 Gateway 已记账并把请求交给一个在线 subscriber；相机或 VLM 的运行时错误会在
后续 `game-result` 中作为失败终态返回。

CLI 与运行中 Gateway 必须使用相同的视觉游戏 `config_digest`。start service 响应丢失时，可以查询同一 ID，
或在 retention 窗口内用完全相同的 game/ID 重发 start 进行幂等恢复；不得换新 ID 自动重试。

## 调用顺序

传统单技能低层 CLI 流程按以下顺序工作：

1. `status`
2. `list-skills`
3. `describe SKILL`
4. `validate SKILL`
5. `execute SKILL --task-id ID`

以下分阶段顺序仅用于诊断、协议测试和明确的低层集成调试。正常 Hermes 自然语言运动请求不得逐个调用这些
命令，必须使用上文的 `run-workflow` 单一复合入口。

在明确进行分阶段协议测试时，顺序为：

1. `plan-workflow --text TEXT --workflow-json JSON --request-id ID`
2. `validate-plan --plan-token TOKEN`
3. 向用户展示步骤、参数、plan digest、registry identity 和 fresh task ID，并 flush 输出
4. 不等待二次确认，立即调用 `confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID [--timeout-sec SEC]`
5. 立即调用 `execute-plan`，除 token、task ID 和相同 timeout 外，传入刚展示的 plan ID/digest、registry
   epoch/generation/digest 和 expected step count

这里的 `confirm-plan` 是 Gateway 对 exact plan/task tuple 的内部技术绑定，不是用户二次确认。用户在展示后
说「别动」时，控制器必须把停止意图锁存到 goal 发送、goal acceptance 和执行阶段；goal 已可能提交时由唯一
执行线程发起一次 action cancellation 并等待权威终态；不要同时从另一个进程调用 `cancel-plan`。pre-send stop
必须抑制新 goal；只有能证明 fresh task 从未提交的 in-process controller 才不发送 CancelGoal，独立 CLI 必须
对可能存在的幂等 retry goal 执行 convergence。`GoalStatus` 本身不足以证明安全结果：result 的 success/error、plan
ID/digest、registry identity 和 completed step count 也必须与命令携带的展示 tuple 一致，否则按
`SKILL_CANCEL_TIMEOUT`/未知状态 fail closed。已知的 uncertain-motion error 保留原始 code 并退出 15；只有身份
或结构证明无效时才合成为 `SKILL_CANCEL_TIMEOUT`。

`robot-skill-closed-loop --resume` 在当前基线明确拒绝。闭环客户端拿到的 `completed_step_count` 只是遥测，
不是续跑授权；客户端不得切片旧步骤并用新 task ID 绕过 Gateway。确定的
`CANCELED + SKILL_CANCELLED` 终态后，用户可以提出一个独立的新 continuation 请求；成功、失败和未知状态
都不授权自动继续。

以上约束的对象是本包的闭环客户端。孵化期 Agent（`ibrobot_agent`）对「继续」的处理是已确认的语义，不属于
被禁止的切片续跑：Agent 在自己的账本和 Gateway 账本（`GetExecution`）都确认上一条请求为确定的 `CANCELED`
之后，把原始指令和已完整完成的步数交给规划器，只规划未完成部分，再以新的 request ID 提交一个全新 Program，
照常经过 presentation 与 Gateway 校验，旧 plan/task 不复用。唯一的例外是**步内续做**：被打断的那一步由执行器
留下了断点（`GetExecution.step_resumable`）时，Agent 不调用规划器，直接提交 Gateway 账本中从该步开始的规范步骤，
并以 `ExecuteProgram.continues_from_execution_id` 引用被打断的执行。这是由 Gateway 自己准入的续做（同一
caller/robot、该机器人最近一次执行、`CANCELED` 且有断点、第 0 步与被打断步规范化一致，否则
`RESUME_CHECKPOINT_INVALID`），不是客户端绕过 Gateway；本包只透传这两个字段，闭环客户端的 `--resume` 仍然拒绝。详见
[`ibrobot_agent` README](../ibrobot_agent/README.md)「继续」一节。

`confirm-plan` 与 `execute-plan` 共享可选的 `--timeout-sec`：省略时两端都默认使用 Gateway 当前
`task_budget_sec`；显式给出时必须为有限正数且不超过 Gateway task budget。`confirm-plan` 把该值以 float32
冻结进计划，`execute-plan` 必须传入**同一**值（float32 严格相等），否则协调器以 `SKILL_REQUEST_ID_CONFLICT`
拒绝——这是为了防止执行阶段悄悄放大或缩小展示并绑定过的任务预算。

当前 `execute` 和 `execute-plan` 可用 SIGINT/SIGTERM 请求取消；另一个进程分别使用 `cancel` 或携带完整
展示 tuple 的 `cancel-plan`。同一 Agent plan 不得同时使用 signal 和外部 `cancel-plan`。失败、timeout 或
停止状态未知时不得自动重试。`SKILL_CANCEL_TIMEOUT` 表示机器人停止状态未知，不能表述为“已停止”。

`validate` 与 `execute` 先以规范 payload 请求 Gateway：字符串去空白、方向小写，未提供 timeout 时使用
Gateway 默认值；timeout 必须为有限正数且不超过 Gateway task budget。payload hash 是规范 JSON 的 SHA-256。
`execute` 的 ROS goal UUID，以及 `cancel --task-id` 使用的 CancelGoal UUID，都是同一 task ID 的
UUIDv5（`ibrobot:{task_id}`）。`cancel` 只对 ledger 中 `active` task 发取消并轮询 `terminal`；已终态 task
返回 `already_terminal`，未知 task 返回 `GOAL_NOT_FOUND`。

## 输出契约

除 `execute`、`execute-plan`、`run-workflow` 和 `ibrobot-perceive` 外，命令向 stdout 输出单行 JSON envelope：

```json
{"command":"status","data":{},"error":null,"ok":true,"schema_version":1}
```

`ibrobot-perceive` 是有意例外：它直接打印请求字段的裸字面量值（如 `0.5236`）供 LLM 读取并
注入 `workflow_json`，错误信息走 stderr；详见上文「命令」表格。

`execute` 与 `execute-plan` 输出 JSONL：零到多条 `feedback`，最后恰好一条 `result`。每行都包含 `task_id` 和
`payload_hash`；公开结果只提供 `executed_step_count`，不暴露 primitive、pose 或 joint 名称。

```json
{"data":{"detail":"step 1 of 1","state":"executing"},"event":"feedback","payload_hash":"...","schema_version":1,"task_id":"task-20260725-001"}
{"data":{"error_code":"","executed_step_count":1,"message":"skill completed","success":true},"event":"result","payload_hash":"...","schema_version":1,"task_id":"task-20260725-001"}
```

`run-workflow` 也使用版本化 JSONL：先输出一条 `workflow_presentation`，最后输出一条
`workflow_terminal`。两条记录都包含 `schema_version`、`command` 和 `task_id`；展示记录必须包含完整
步骤、plan digest、registry identity 和 execution mode。

普通命令错误也使用同一 JSON envelope 的 `error.code` 与 `error.message`。当 Gateway capability reason
为 `CODE: detailed message` 时，CLI 用第一个冒号前的文本作为 `error.code`；`error.message` 保留原始
reason（没有冒号时两者都是该 code）。因此调用方应按 `error.code` 做稳定分类，不要解析详细消息。

| 退出码 | 含义 |
|---:|---|
| `0` | 成功 |
| `2` | 参数、配置或 schema 错误 |
| `3` | Gateway、readiness 或 safety 拒绝 |
| `4` | ROS/Gateway 不可用 |
| `124` | timeout，可能包含停止状态未知 |
| `130` | direct `execute` 的 SIGINT 取消已收敛到 terminal |
| `143` | direct `execute` 的 SIGTERM 取消已收敛到 terminal |

上述 `3/4/124/130/143` 是 direct `validate/execute/cancel` 的兼容契约。Agent plan
命令（`plan-workflow`、`validate-plan`、`confirm-plan`、`execute-plan`、`cancel-plan`）使用
以下稳定分组：`10` 表示 package/profile/implementation 不存在，`11` 表示 schema/reference/limit
校验失败，`13` 表示 registry/workflow/admission/finalization 状态错误，`14` 表示 ROS transport
不可达，`15` 表示 timeout/budget/deadline 或停止状态未知。

## 授权边界

`authorize_motion` 是唯一运行时运动授权来源，launch 默认值为 `false`。CLI 和 Agent 不能启动或重启
pipeline、开启授权、修改 ROS 参数，或从 YAML 推导授权。只有操作员完成现场安全检查后，才能在启动
pipeline 时显式授权。未授权状态下仍可使用 catalog、`status` 和不执行动作的本地检查；Gateway 会拒绝运动。

常驻 bridge `robot-skill-daemon` 是 `robot-skill` 的 ROS 传输进程，不是新的授权入口：

- 它只向 Capability Gateway 发出与 `robot-skill` 相同的 typed 请求，Gateway 仍逐请求校验 `caller_scope`、
  调用方 token 和 `authorize_motion`；daemon 不持有授权，也不绕过 Gateway。
- 它的 Unix socket 位于权限 `0700` 的用户目录，只接受同一 uid 的对端（无法确认对端 uid 时拒绝）。能连上 socket
  的进程本来就能以同一用户直接运行 `robot-skill` 或 rclpy，因此不扩大访问面；但 CLI 侧的展示、停止状态和
  `ibrobot-block-raw-ros` hook 只约束经过 `robot-skill` 的调用，不约束直接连接 socket 的进程。
- daemon 只接受已知的 Gateway 接口参数（不接受 `caller_token_env` 等身份字段）、只允许调用固定的 Gateway 方法，
  并在启动时清除继承来的 `IBROBOT_EXECUTION_TOKEN_*` 等调用方私有环境变量。

`robot_mcp` 兼容层已移除。Hermes 推荐配置和示例统一通过 `robot-skill` 访问 Capability Gateway。

## Hermes 启动器

完整构建后可运行：

```bash
hermes-robot --config-name so101_single_arm
hermes-robot --config-name lekiwi_handeye_realsense_grasp -- --cli
hermes-robot --config-name lekiwi_handeye_realsense_grasp_pc -- --cli
hermes-robot --config-name so101_single_arm --mode visual-games
hermes-robot --config-name so101_single_arm --mode motion
hermes-robot --config-name so101_single_arm --mode both
```

启动器要求 Hermes Agent `0.16.0` 或更新版本，并在启动前验证 `hermes`、`robot-skill`、安装空间中的
`ibrobot-control`、目标 robot config 以及所选控制面的 ROS 接口。`--mode visual-games` 只预检独立的
`StartVisualGame` / `GetVisualGameResult` service，不要求 MoveIt、Skill Gateway 或 Agent plan interfaces；
`--mode motion` 预检 Gateway status 和全部 Agent plan service/action；`--mode both` 同时预检两套接口。
默认 `--mode auto` 按配置能力选择：仅有已启用视觉游戏时预检视觉游戏接口，仅有运动 catalog 时维持运动
预检，两者都有时预检两套接口。显式 `--mode` 可用于只检查其中一个控制面。启动时会将安装空间中的
`ibrobot-control` 幂等注册到当前 Hermes profile 的 `skills/` 目录；仅更新带有
`robot_skill_cli` 所有权标记的副本，遇到同名的用户自管 skill 时会以 `AGENT_SKILL_CONFLICT` 退出。
`motion_authorized=false` 不阻止 Hermes 启动，只会继续由 Gateway 拒绝运动。启动器仅设置精确
`ROBOT_CONFIG` 并预加载 `ibrobot-control`；它不会启动/重启 pipeline、修改 ROS 参数或开启运动授权。
进入 Hermes 会话后只调用启动器注入到 `PATH` 的 `robot-skill`，不得再传 `--config-name` 或
`--config-path`。自然语言抓取与其他 motion Skill 使用统一的 `run-workflow` 复合入口；该入口内部完成
`status -> catalog -> plan -> validate -> present/flush -> confirm -> execute` 生命周期；
抓取计划使用 `pick_object` 和必填的 `target_name`，Gateway 再将其委派给配置绑定的 `grasp_pipeline`。

运动 catalog 的 `reload-catalog` 是显式、受控的运行时 snapshot 切换，不是自动监听文件。视觉游戏当前
不支持热加载：YAML 配置变更需要重启 pipeline，Python handler 变更要重新构建并重启；
`reload-catalog` 不会更新视觉游戏。

## raw-ROS 拦截 hook

`resource/hermes/hooks/ibrobot-block-raw-ros` 是 Hermes `pre_tool_call` hook，从 stdin 解析 tool
payload 并用 `shlex` 分词，阻断任何裸 `ros2` 子命令和 `rclpy`/`roslaunch` 间接调用，强制 LLM 走
`ibrobot-perceive`（感知读取）或 `robot-skill`（运动控制）。它是 defense-in-depth，权威边界仍是
`authorize_motion` 和 Gateway plan validation。策略全文见
[`resource/hermes/POLICY.md`](resource/hermes/POLICY.md)。

## Hermes 集成同步

`hermes-robot-configure` 是 IB-Robot Hermes 集成的幂等 profile 同步命令，把 Skill、SOUL 策略和
speech hook 一次性写入当前 Hermes profile。它取代了手工维护的 `SOUL.md`、`POLICY.md`、
`ibrobot-control` Plugin 和 TTS hook 副本，作为 `robot_skill_cli/resource/hermes/` 唯一发布源的
落地入口。

```bash
hermes-robot-configure \
  --config-name so101_single_arm \
  --soul-mode replace \
  --accept-hooks \
  --install-runtime-adapter \
  --restart-gateway
```

也可以使用安装到 `share/robot_skill_cli/hermes/sync_hermes.sh` 的 shell 入口（等价于
`hermes-robot-configure`），适合源码场景或无 `install/setup.bash` 时使用：

```bash
bash src/agent/robot_skill_cli/resource/hermes/sync_hermes.sh \
  --config-name so101_single_arm --dry-run
```

同步会执行以下动作（`--dry-run` 仅预览不写文件）：

- 调用 `resolve_robot_config_path()` / `load_robot_config()` 从 `robot_config` SSOT 读取
  `voice_tts` 服务名、playback 服务名、synthesis 超时和 playback 超时；严禁硬编码 ROS 服务名。
  其中 `synthesis_timeout_sec` 是 hook 等待 `/voice_tts/synthesize` 返回合成结果的超时，
  `playback_timeout_sec` 是等待 `/voice_tts/play` 完成播放的超时；`tts_timeout_sec` 是
  `embodied_agent` 视觉游戏播报节点的内部 RPC 超时，三者语义不同，互不替代。
- 验证当前 `ROS_DOMAIN_ID`（0-232）和 workspace `.shrc_local`（ROS+venv+install overlay 的
  SSOT 入口，见 `AGENTS.md`）。
- 启用 `ibrobot-motion` 时，把当前 shell 已设置的 `RMW_IMPLEMENTATION`、`CYCLONEDDS_URI`、
  `ROS_LOCALHOST_ONLY` 写入 profile，受管运动始终使用这组值。之后如果更换 RMW（例如在 Fast DDS 与
  CycloneDDS 之间切换），必须在新的 shell 环境中重新执行 `hermes-robot-configure`；`hermes-robot`
  启动时会比较 profile 中的绑定值与当前 shell，不一致时以 `ROBOT_TRANSPORT_MISMATCH` 退出。
- 把 `resource/hermes/SOUL.md`（`--soul-mode replace`）或 `POLICY.md` 托管区块（`--soul-mode merge`）
  写入 profile/SOUL.md，`skip` 不修改 SOUL；`replace` 会先备份。
- 把 `ibrobot-control` Skill 幂等注册到 `profile/skills/`，仅替换带 `robot_skill_cli` 所有权标记
  的副本，遇到同名用户自管 skill 时以 `AGENT_SKILL_CONFLICT` 退出。
- 生成 `robot-skill`、最终回复 speech、生命周期 speech 和 interim speech wrapper，以及一个环境文件；
  `robot-skill`、最终回复 speech 和 interim wrapper source workspace `.shrc_local`；lifecycle wrapper
  依赖 Hermes 已初始化的 workspace 环境，不硬编码 `/opt/ros/humble/setup.bash` 或 venv 路径。
  interim wrapper 还绑定当前 Hermes profile 的 `ibrobot-speak` 路径。
- 清理旧的 `post_llm_call`、生命周期和 `on_interim_message` speech hook managed 副本；
  `--disable-speech` 移除全部受管 speech hook。
- 原生 Hermes response/motion plugin 是最终回复、中间回复和停止控制的唯一事件生产者；保留的
  `ibrobot-speak` wrapper 只负责在 ROS 环境中投递结构化语音事件，不再作为独立的 `post_llm_call`
  或生命周期文案生产者。同步完成后必须重启已有 Hermes CLI。
- 移除已退役的 `ibrobot-robot-control` Plugin 副本（即时执行改由 `robot-skill run-workflow` 的
  显式 `immediate_after_presentation` 模式承担），并清理 `external_dirs` 缓存目录下同名的 stale skill 副本。
  Plugin 目录只在带 `hermes-robot-configure` managed 标记时才会被 quarantine（重命名），
  不删除无标记或用户自管内容。
- `--accept-hooks` 是兼容参数，不会更改 Hermes hook 授权；受管语音使用原生插件回调。
  runtime adapter 默认不安装。只有显式传入 `--install-runtime-adapter` 时，才会校验并修改受支持的
  Hermes checkout，同时启用 `ibrobot-motion`；未安装时受管运动和优先停止保持禁用。
  `--restart-gateway` 调用 `hermes gateway restart`。

`hermes-robot-speak` 是由原生 Hermes response plugin 调用的 ROS 语音投递 Python 入口（由 `ibrobot-speak`
shell wrapper 调用），
从 stdin 读取 Hermes hook payload，提取最终 assistant 回复，经 `sanitize_for_tts` 移除 ASCII
字母 run 和连续 7 位以上数字（避免播报技术标识符；ZipVoice 中文前端不发音英文），调用
`ibrobot_msgs.srv.SynthesizeSpeech` 合成 WAV
并通过 `PlayAudioFile` 服务播放。TTS 是系统自动功能，不是机器人 Skill，不进入 `workflow_json`。
诊断日志位于 `/tmp/hermes-speak.log`（受 `IBROBOT_HERMES_TTS_LOG` 覆盖）。

### Hermes runtime adapter 生命周期

runtime adapter 默认不安装。需要受管运动与优先停止时，显式使用 `--install-runtime-adapter`；此操作仅支持
指定 Hermes Git revision，文件哈希和适配器版本记录在 checkout 的 `.ibrobot-control-adapter.json`。
未知 revision、文件漂移或不支持的既有 manifest 会 fail closed，不会部分改写外部运行时。同步前会生成
`.ibrobot-backup-*` 备份。普通 profile 同步（包括 `--disable-speech`）不检查或修改 Hermes 安装目录。
长期应优先迁移到 Hermes 正式 plugin lifecycle API，避免依赖源码锚点。

详细同步步骤、前置条件和升级流程见
[`resource/hermes/README.md`](resource/hermes/README.md)。

SO-101 真机的完整手动验证步骤见
[`docs/hermes_so101_real_robot_manual_validation_zh.md`](../../../docs/hermes_so101_real_robot_manual_validation_zh.md)。

## 排障：DDS 发现时延与 RMW 选择

ROS 2 中间件沿用发行版默认的 Fast DDS，`.shrc_local` 不修改 `RMW_IMPLEMENTATION`，正常使用不需要切换。

- **现象**：Fast DDS 下每个新进程（每次 `robot-skill` 调用都是新进程）需要先与已运行的 Gateway 完成 DDS 发现，
  才能发出第一个请求。在 SO-101 单机上实测约 1 s，偶发 2–3 s；耗时随网络、节点数量和发现配置变化。
  通过 `embodied_pipeline.launch.py` 启动时，[常驻 ROS bridge](#常驻-ros-bridge) 已提前完成发现，`robot-skill`
  调用不再承担这段时延；只有 daemon 未运行（未用 pipeline、`embodied.resident_cli_bridge: false` 或
  `IBROBOT_ROBOT_SKILL_DAEMON=0`）时才会出现。
- **可选：改用 CycloneDDS**：安装 `ros-humble-rmw-cyclonedds-cpp`，并在 `source .shrc_local` 之前执行
  `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`。目前只在未接相机的 SO-101 单机上验证过（敲命令到机械臂开始
  运动约 0.8 s）；相机图像传输、LeKiwi、openEuler 与跨机器部署尚未验证。
- **影响整个 ROS 图**：同一 ROS 图内所有进程必须使用同一 RMW。切换后要重启所有已在运行的节点（包括 launch），
  并执行 `ros2 daemon stop`，只改 `robot-skill` 所在 shell 不会生效。
- **Hermes**：受管运动使用 `hermes-robot-configure` 时绑定的 RMW，切换后必须在新的 shell 中重新执行
  `hermes-robot-configure`，否则 `hermes-robot` 以 `ROBOT_TRANSPORT_MISMATCH` 拒绝启动（见
  [Hermes 集成同步](#hermes-集成同步)）。
