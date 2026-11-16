# RoboDojo 接入设计说明

本文说明 RoboDojo 双臂 ARX X5 如何接入 IB-Robot 正式执行链，只描述已经实现的结构。评测（准入、判分通道、任务说明）不在这里，见评测 MR。

## 1. 分层与包归属

| 层 | 包 | 职责 | 禁止依赖 |
|---|---|---|---|
| 仿真平台注册 | `robot_runtime`（`sim_platforms`，master 已有） | `ibrobot.sim_platforms` 平台发现、`SimEmbodiment` 协议、物理选择下的时钟策略 | 任何仿真器 |
| 没有控制器的平台的运行时 | `robot_runtime_sim.direct_control` | 引擎（会话与 reset 代次、唯一写入者、命令幂等、空闲不推进、结果未知停写）、平台适配器端口、ROS 节点外壳、`/clock` | RoboDojo/Isaac 导入、X5 名称、固定臂数、任务名、评测概念 |
| 平台 | `robodojo_robot` | `robodojo` 平台注册与 embodiment、RoboDojo 观测键与动作键编解码、托管 RoboDojo ws 策略服务端、Isaac 侧客户端、头相机标定叠加图 | 复制调度核心、绕过 Gateway 执行产品动作 |
| 本体与部署配置 | `robot_config`（`robodojo_x5.yaml`）、`robodojo_robot/profiles/arx_x5_dual.yaml` | 运动组、关节、TCP、相机、拍长的唯一来源 | 实验 seed、判分规则、模型密钥 |
| 正式执行链 | `skill_catalog`、`skill_library`（Gateway）、`safety_guard`、`task_dispatch`、`embodied_common`、`ibrobot_msgs` | 多臂 `target_group` / `approach`、安全预检 | 平台名分支 |

`robodojo` 是注册的物理仿真平台：`robot_config` 选 `simulation.platform: robodojo`，launch preparation 按 master 的统一规则让所有消费者使用仿真时间；`robodojo_robot/runtime.launch.py` 通过 `sim_embodiment_actions` 分派到平台的 embodiment，embodiment 启动 `direct_control_runtime` 并指定 RoboDojo 适配器。适配器只由 embodiment 用 `module:attr` 指定，没有单独的适配器注册表。

## 2. 对正式执行链的修改

本实现修改了 Gateway、`safety_guard` 和 `task_dispatch`。修改都是可选扩展：机器人配置不声明对应字段时，契约摘要与行为不变。

1. 多臂：`MoveToPose`、`ComputeIk`、`TaskStep`、`PrimitiveCommand`、`ValidatePrimitive` 增加 `target_group`；技能参数 `arm_side` 由 `embodied.execution.arm_groups` 映射到运动组；`safety_guard` 校验 `target_group` 是本机的运动组（工作空间和速度上限仍是整机一套，不分组）；`task_dispatch` 按组绑定夹爪动作。
2. 夹爪姿态：`approach` 字段与 `set_gripper_approach` 技能，预设来自 `embodied.execution.gripper_approach`。
3. Agent 规划器可把配置的相机最新帧附到提示词（`planner.vision_topics`）。

## 3. 进程与时钟

| 进程 | Python | 职责 |
|---|---|---|
| ROS 进程 | 3.10 | `robodojo_runtime`（`direct_control_runtime` + RoboDojo 适配器）、Agent、Gateway、`safety_guard`、`task_dispatch` |
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
8. 客户端启动前清除 ROS 与 venv 环境变量（`run_eval_client.sh` 白名单重启），避免 3.10 包进入 conda 解释器。
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
7. 引擎没有准入规则；评测层可以通过节点 `extensions` 安装运动闸门（`set_motion_gate`）。
8. 状态和图像消息的时间戳是观测的仿真时间，不是发布时刻；新会话即使拍号相同也重新发布图像。
9. 启动时检查描述符（dt 有限且为正、pose_frame 非空、TCP 为 3 个有限数），检查 profile 公开的能力与适配器支持的命令、时钟模式一致，不一致则拒绝启动。
10. `ComputeIk` 拒绝运行时坐标系以外的 frame、带 seed 的请求和比平台保证更严的姿态容差，不做近似。

## 5. 信息隔离

1. 公共话题 `/simulation/session`、`/simulation/clock` 不含分数、seed 或布局。
2. RoboDojo 的判分只作为会话的平台结果交给监听者；没有监听者时只写日志，不发布到任何话题。

## 6. 运行方法

```bash
export ROBODOJO_ROOT=<RoboDojo checkout>
./scripts/build.sh --agent
./scripts/build.sh --packages-select task_dispatch robot_runtime_sim robodojo_robot
# 运行时单独冒烟（另一个终端运行 RoboDojo 客户端）
ros2 launch robodojo_robot runtime.launch.py profile:=arx_x5_dual
TASK=general_pickup EVAL_NUM=1 bash install/share/robodojo_robot/robodojo_client/run_eval_client.sh
python3 src/robot_runtime_sim/scripts/direct_control_smoke.py --group right_arm --out smoke.json --finish
```

## 7. 限制

1. 第二平台只有 CPU 替身（`robot_runtime_sim.testing.direct_control_fakes`：2×6 自由度 pull、1×7 自由度 host step），没有接入第二个真实平台。
2. 已执行的物理拍不能撤回；取消从下一拍生效。
3. IK 只能来自平台（`ik: platform`）；运行时侧求解器、仿真自由运行时按 `/clock` 采样下发目标、移动底盘运动组都还是扩展点，没有实现。
