# RoboDojo 接入设计说明

本文说明 RoboDojo 双臂 ARX X5 如何接入 IB-Robot 正式执行链，以及 Agent 评测怎样作为可选的外挂层接到运行时上、怎样与 Agent 隔离。只描述已经实现的结构。

## 1. 分层与包归属

| 层 | 包 | 职责 | 禁止依赖 |
|---|---|---|---|
| 仿真平台注册 | `robot_runtime`（`sim_platforms`，master 已有） | `ibrobot.sim_platforms` 平台发现、`SimEmbodiment` 协议、物理选择下的时钟策略 | 任何仿真器 |
| 没有控制器的平台的运行时 | `robot_runtime_sim.direct_control` | 引擎（会话与 reset 代次、唯一写入者、命令幂等、空闲不推进、结果未知停写）、平台适配器端口、ROS 节点外壳、`/clock` | RoboDojo/Isaac 导入、X5 名称、固定臂数、任务名、评测概念 |
| 平台 | `robodojo_robot` | `robodojo` 平台注册与 embodiment、RoboDojo 观测键与动作键编解码、托管 RoboDojo ws 策略服务端、Isaac 侧客户端、头相机标定叠加图 | 复制调度核心、绕过 Gateway 执行产品动作 |
| 本体与部署配置 | `robot_config`（`robodojo_x5.yaml`）、`robodojo_robot/profiles/arx_x5_dual.yaml` | 运动组、关节、TCP、相机、拍长的唯一来源 | 实验 seed、判分规则、模型密钥 |
| 正式执行链 | `skill_catalog`、`skill_library`（Gateway）、`safety_guard`、`task_dispatch`、`embodied_common`、`ibrobot_msgs` | 多臂 `target_group` / `approach`、安全预检 | 平台名分支 |

评测相关的层（都在 `src/benchmark/` 下，和 `benchmark_runtime` 并列；运行时和平台包都不依赖它们）：

| 层 | 包 | 职责 | 禁止依赖 |
|---|---|---|---|
| 评测面 | `agent_eval` | TaskSpec / Recipe、逐步请求（`episode_context`：白名单观测 → 机器人状态文字 + 配方，不跨步保留任何状态）、EpisodeRunner、单命令 suite、报告（逐局写 `benchmark_runtime` 的 canonical 记录）、平台插件接口 | 持有 Agent 工作记忆、读取平台结果拼提示、导入 Isaac |
| 运行时评测扩展 | `agent_eval.runtime_extension:attach` | 挂到 direct-control 运行时上：准入服务 `/evaluation/admit_episode`、运动闸门、判分通道 `/evaluation/native_result` | 导入 `robot_runtime_sim`（只用传进来的节点）、在引擎锁内阻塞或回调引擎 |
| 评测平台插件 | `agent_eval_robodojo`（入口 `agent_eval.platforms: robodojo`） | 为 suite 组装两个进程：IB-Robot 栈与 RoboDojo 评测客户端；随包带 RoboDojo 套件 | 复制调度核心、绕过 Gateway 执行产品动作 |

`robodojo` 是注册的物理仿真平台：`robot_config` 选 `simulation.platform: robodojo`，launch preparation 按 master 的统一规则让所有消费者使用仿真时间；`robodojo_robot/runtime.launch.py` 通过 `sim_embodiment_actions` 分派到平台的 embodiment，embodiment 启动 `direct_control_runtime` 并指定 RoboDojo 适配器。适配器只由 embodiment 用 `module:attr` 指定，没有单独的适配器注册表。

## 2. 对正式执行链的修改

本实现修改了 Gateway、`safety_guard` 和 `task_dispatch`。修改都是可选扩展：机器人配置不声明对应字段时，契约摘要与行为不变。

1. 多臂：`MoveToPose`、`ComputeIk`、`TaskStep`、`PrimitiveCommand`、`ValidatePrimitive` 增加 `target_group`；技能参数 `arm_side` 由 `embodied.execution.arm_groups` 映射到运动组；`safety_guard` 校验 `target_group` 是本机的运动组（工作空间和速度上限仍是整机一套，不分组）；`task_dispatch` 按组绑定夹爪动作。
2. 夹爪姿态：`approach` 字段与 `set_gripper_approach` 技能，预设来自 `embodied.execution.gripper_approach`。
3. Agent 规划器可把配置的相机最新帧附到提示词（`planner.vision_topics`）。

## 3. 进程与时钟

| 进程 | Python | 职责 |
|---|---|---|
| ROS 进程 | 3.10 | `robodojo_runtime`（`direct_control_runtime` + RoboDojo 适配器，评测时另挂评测扩展）、Agent、Gateway、`safety_guard`、`task_dispatch`；评测时另有 `agent_eval` runner |
| RoboDojo 评测客户端 | 3.11（conda） | Isaac Sim 物理、cuRobo 逆解、RoboDojo 自己的判分 |

1. RoboDojo 持有主循环（platform pull）。客户端每拍调用 `update_obs` 与 `get_action`。
2. 没有命令时运行时回答 idle，客户端不调用 `take_action`，仿真时间不前进。
3. 三种时间各管一件事：
   1. 步号：会话内记账、平台的步数预算。
   2. ROS 时间 `/clock`：本进程所有会话累计的仿真时间（拍数 × 40 ms），reset 不回退，发布序列也不回跳；只在 RoboDojo 推进时前进，Agent 思考时整个世界停住。运行时发布的状态、关节、图像的时间戳都是仿真时间，订阅方用 `use_sim_time`，`now()` 和时间戳在同一条时间轴上。
   3. 墙钟：任务和动作的期限都按墙钟算，包括 task_dispatch 单步运动 60 s、技能执行器等待动作结果、运行时单条命令 55 s、平台长时间无调用 120 s。这些期限用来发现链路卡死，取值远大于正常动作所需的墙钟时间；Isaac 极慢时可能先到期，此时命令以超时或 UNKNOWN 结束，不会当成成功。
4. 运行时节点自己用墙钟（它是时钟源）：状态、关节、图像按墙钟定时发布，仿真暂停时照常发布；时间戳是对应观测的仿真时间。
5. 取消命令有 2 种方式：
   1. 按 `command_id` 撤销一条命令：`/simulation/cancel_command`（`CancelSimulationCommand`）。如果取消比命令先到达运行时，运行时在本代次记下该 `command_id`，之后到达的同 ID 请求直接以 CANCELLED 结束，不推进任何拍。task_dispatch 给每个 MOVE_TO_POSE 步骤的 `MoveToPose` 请求带上自己生成的 `command_id`；任务被取消或单步超时时，经 `task_dispatch.cancel_command_interface` 绑定的这个服务撤销该命令，等运行时确认、`MoveToPose` 也返回后，这一步才结束。夹爪动作的 action 取消也只撤销该命令；在命令入队前就被取消的，不会入队、不推进任何拍。
   2. 全局停止：`StopRuntime` 或切到 idle 模式。
6. 已交给平台的一拍不能撤回，取消从下一拍生效。
7. 一次运动按请求速度需要的拍数超过 `max_motion_ticks` 时直接拒绝（REJECTED，不推进任何拍），不压缩轨迹，所以不会比请求的速度快。
8. 客户端启动前清除父 shell 的 ROS 与 venv 环境变量（`run_eval_client.sh` 白名单重启）。下游会再次激活 Conda；激活钩子、环境配置和 `.pth` 仍可能引入其他路径。激活前的 `print-env` 不能证明最终 Isaac 解释器已隔离，检查方法见[操作手册第 4 节](../src/robots/robodojo/robodojo_robot/README.md#4-检查仿真侧环境)。
9. 运行时回答 reset 后，客户端把 RoboDojo ws 客户端的重连预算收紧到 30 s（`IBROBOT_RUNTIME_RECONNECT_S`，在白名单里，可从启动环境传入；只改本次会话的配置对象，不改 RoboDojo 仓）。
10. 客户端向运行时的调用被拒绝或断联时，客户端立即停止调用 `take_action`，写 `interrupted` 工件（无判分），把该局标为 unstable，RoboDojo 不计分也不保留视频。
11. 局结束前，客户端上传最后一拍后的观测，所以会话结束事件的 `ticks` 等于平台的 `take_action_cnt`（另记 `platform_ticks`）。

## 4. 会话生命周期

一个 RoboDojo episode 是引擎的一个会话（session）。状态：`CREATED → RESETTING → READY → RUNNING → STOPPING → FINALIZED`，异常进入 `FAILED` 或 `UNKNOWN`。

1. `reset` 取消并排空上一代命令，代次加 1；旧代次的观测、IK 回包和调用被拒绝。
2. 新代次的第一帧观测到达后才进入 `READY`，之前的运动请求被拒绝。
3. 同一 `command_id` 再次提交时返回第一次的结果；内容不同则拒绝。
4. 命令已交给平台但墙钟超时，或平台连接中断：结果记为 `UNKNOWN`，运行时停止接受运动，直到下一次 `reset` 或 `/simulation/finish_session`；不自动重试。
5. 只有一个写入者（`claim_writer`）能驱动拍循环。
6. `/simulation/finish_session`：保持当前位置直到平台结束本会话；`/simulation/abort_session`：停止写目标，会话结束且没有平台结果。
7. 引擎没有准入规则；评测层可以通过 `extensions` 安装运动闸门（`set_motion_gate`），见 §5。
8. 状态和图像消息的时间戳是观测的仿真时间，不是发布时刻；新会话即使拍号相同也重新发布图像。
9. 启动时检查描述符（dt 有限且为正、pose_frame 非空、TCP 为 3 个有限数），检查 profile 公开的能力与适配器支持的命令、时钟模式一致，不一致则拒绝启动。
10. `ComputeIk` 拒绝运行时坐标系以外的 frame、带 seed 的请求和比平台保证更严的姿态容差，不做近似。

## 5. 评测与信息隔离

1. 公共话题 `/simulation/session`、`/simulation/clock` 不含分数、seed 或布局。
2. RoboDojo 的判分只作为会话的平台结果交给监听者；没有监听者时只写日志，不发布到任何话题。
3. 评测层怎么挂上：运行时节点启动时读 profile 的 `simulation.<平台>.extensions`（并上节点参数 `extensions`），逐个以节点为参数调用。随包的 profile 不列任何扩展；`agent_eval run_suite` 把 robot_config 的 runtime profile 复制到本次运行目录，在副本里列上 `agent_eval.runtime_extension:attach`，再让本次运行派生的 robot_config 的 `runtime.profile` 指向副本，启动链路不改。副本的路径和源文件哈希记在报告里。
4. 评测扩展只做三件事：
   1. 运动闸门：没准入的代次，所有运动请求都被拒绝（REJECTED）。上一局 Agent 迟到的请求（包括取消后才到的）因此不会在新一局执行。
   2. `/evaluation/admit_episode`（`AdmitSimulationEpisode`）：只接受运行时当前的会话和代次。
   3. 把每个会话的平台结果发到 `/evaluation/native_result`（latched），只有 runner 订阅。中断的会话（平台调用失败、断联、客户端或 runner 放弃）也发，带 `interrupted: true`、没有判分。

   闸门和结果监听在引擎持锁时被调用，只查表和发布，不阻塞、不回调引擎。
5. 准入：runner 确认本会话的位姿、关节和规划器相机都已到达本会话的数据，且上一局的 Agent 请求都已结束，才调用 `/evaluation/admit_episode`。缺任何一项就调用 `/simulation/abort_session`，该局不运行、不判分。时间戳是仿真时间，reset 时不回退但可能不前进（新会话第一帧可能和上一会话最后一帧同一时刻），所以“本会话的数据”按两条判断：
   1. 在 `/simulation/clock` 第一次报告本会话已有观测（状态不再是 RESETTING）之后到达。运行时在 reset 和新会话第一帧之间不发任何状态和图像，并且先发这条时钟再发这一帧。
   2. 时间戳不早于本会话的起点（该时钟的 `sim_time_total_s - sim_time_s`）。
6. 逐步请求只由白名单观测拼成：每臂 TCP 位置与四元数、夹爪开度、腕关节角、拍数、拍预算、任务指令，再附配方。评测侧不跨步保留任何状态，每次决策另开一个 Agent 会话，规划器看不到本局之前的请求和回复；记忆由 Agent 自己提供。
7. Agent 说 DONE 只是结束请求，评测侧不复核；成功只取平台结果。
8. TaskSpec 与 Recipe 的加载器拒绝物体位姿、目标坐标、判分阈值、seed 等字段。
9. Agent 规划器订阅 `/simulation/clock`（`planner.vision_epoch_topic`）：会话变化时清空图像缓存，拒收时间戳早于本会话起点的帧。
10. 报告：成功只认平台判分。没有判分的局（中断、放弃、结果没到）不算成功也不算失败，记为基础设施错误；报告不设单一头条成功率，并列“成功 / 有判分的局”（`native_model_success_rate`）和“成功 / 计划局数”（`planned_success_rate`），另列中断局数；有中断或少跑的运行标为 `partial`（不完整）。

## 6. 运行方法

安装、资产下载、构建和操作步骤只维护在 [RoboDojo 首次使用手册](../src/robots/robodojo/robodojo_robot/README.md)。
先阅读依赖与许可，再选择无模型运行时冒烟或 Agent 单局评测。
手册分别说明 ROS/Conda 环境、三个手动终端、端口与 ROS 域、结果判读、停止和排障。
不要把架构说明当成安装完成的证明。

`robodojo_general_pickup_smoke` 使用 `9001` 布局集合的 1 局。
`robodojo_general_pickup_e2` 使用 `9001`、`9002`、`9012` 三个集合，各 10 局。
仓库不提供这些集合的分发或生成步骤，也未证明上游标准资产包含它们；缺失时停止评测。
每次 suite 输出包括 `report.json`、`canonical/`、逐局记录、进程日志、派生 robot_config 和 runtime profile 副本。
原生视频和桥接工件的准确路径见报告，不靠目录时间推测。

## 7. 限制

1. 第二平台只有 CPU 替身（`robot_runtime_sim.testing.direct_control_fakes`：2×6 自由度 pull、1×7 自由度 host step），没有接入第二个真实平台。
2. 已执行的物理拍不能撤回；取消从下一拍生效。
3. IK 只能来自平台（`ik: platform`）；运行时侧求解器、仿真自由运行时按 `/clock` 采样下发目标、移动底盘运动组都还是扩展点，没有实现。
4. 配方 `general_pickup_x5` 的规则文字是 X5 抓取专用的，每次请求都附上；换任务或本体要另写配方。
5. 规划器 token 用量未在 `/agent/event` 上发布，报告中 tokens 为 null 并写明原因。报告的 `recipe_task_hash` 只覆盖配方与任务卡；完整提示词哈希为 null 并写明原因，逐步请求文本哈希写在每局记录中。
6. 评测时 profile 是复制到运行目录再用的，profile 里不能有相对路径引用（`arx_x5_dual.yaml` 没有）。
7. runner 放弃的局，平台结果里没有 seed 和布局；报告用该批次的客户端 seed 和布局补上，canonical 记录里标 `seed_source: client_pass`。
