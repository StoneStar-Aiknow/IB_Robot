# X2 命名准备姿态（第一版）

自由姿态可以生成动作；任务准备姿态使用确定的关节目标，不调用语言模型，也不进入动作记忆库。
URS 是厂商控制模式，准备姿态是具体的双臂关节位置。进入 URS 不代表已到准备姿态。

## 配置入口

参考配置：`src/robot_config/config/robots/aimdk_x2_grasp_task_poses_310p.yaml`。
这是独立的 opt-in 配置，不改变原有抓取或自由姿态配置的默认行为。
该配置仍需要原抓取部署的模型、运行时、MuJoCo 和官方 X2 MJCF 资源；不是免资源的 Mock 启动入口。

组合配置 `src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml` 也已接入任务准备姿态：
保留语音、手势、照镜子及抓取目录，同时公开 `return_to_task_pose(target_name="grasp_ready")`。
可以向 Agent 说“回到抓取准备位置”；Agent 选择技能后仍须经 Gateway 校验与执行。
`pick_object` 复用同一父任务中的准备阶段，总技能预算为 150 秒；准备失败不会开始抓取。
照镜子持续占用手臂时，回位和抓取均被阻止，必须先停止照镜子并确认释放。
该配置默认引用下述未批准示例，真机需另行提供现场校准且批准的姿态文件；仅加载配置不代表可上真机。
启用确定性准备位不需要姿态生成 LLM，但默认生成后端仍需 MuJoCo 与官方 X2 MJCF；
原有 Agent planner、感知模型、Orin 抓取头和音频资源要求不因此消失。

本地位姿文件：`src/robot_config/config/task_poses/x2_example.yaml`。
其 `grasp_ready` 来自测试参考姿态，**不是已校准的真机抓取准备位**，`hardware_approved` 为 false。
Mock 测试不改变该字段，真实部署会拒绝执行未批准的姿态。

可以另存到 `~/.config/ibrobot/x2/task_poses.yaml`，再在机器人 YAML 引用：

```yaml
robot:
  task_poses:
    file: ~/.config/ibrobot/x2/task_poses.yaml
```

文件包含：

- `schema_version: 1`、匹配机器人配置的 `robot_type`。
- `joint_names`：运行时定义顺序的全部 14 个手臂关节。
- `poses.<name>.joint_positions`：完整的双臂关节角，单位 rad。
- `position_tolerance_rad`：最大单关节到位误差，默认 0.03 rad。
- `stable_sec`：实测到位保持时间，默认 0.3 秒。
- `timeout_sec`：本阶段超时，默认 30 秒，同时受父任务剩余预算限制。
- `max_state_age_sec`：每个关节的反馈有效期，默认 0.5 秒。
- `hardware_approved`：示例默认为 false；真机目标应经示教和现场验证后再批准。
- `hands: {left: open, right: open}`：可选手型，只支持 `open`，按左手、右手顺序执行；省略则只控制双臂。
- `unloaded_operation_approved`：含手型时默认为 false；用于现场确认的空载操作，不代表能自动检测持物。
- `skill_entry_poses.pick_object: grasp_ready`：抓取前置姿态映射，第一版只支持 `pick_object`。

角度非有限、关节缺失/重复、未知引用、非法容差和机器人型号不匹配会被拒绝。
加载后建立规范化快照和 digest，纳入执行上下文及姿态执行器身份。任务使用启动时的快照，
不在执行中重读文件。修改后需要重新加载/重启部署，CLI 与运行时快照不一致时不能执行旧计划。

## 执行流程

独立技能 `return_to_task_pose(target_name="grasp_ready")` 与抓取前置阶段共用同一处理器：

1. 经现有 Gateway 验证与模式/能力准入。
2. 检查目标名称、快照 digest、硬件批准状态和实时反馈。
3. 配置手型时，预检左右映射、碰撞覆盖、反馈、命令板增益，以及 Gateway 的活动任务是否为当前父任务。
   必要时通过 runtime 手部通道保持当前双手，确认接管增益为 40/2。不会自行切换模式。
4. 从接管后的实时起点规划双臂路径，沿用 X2 限位、自碰撞、重力负载、速度和调试位移检查。
   用左右手各自的关节模型补查最终下发路径中的实际手指形状，包含双手之间、手与身体及固定下肢的间隙。
   插值余量按连杆对的相对运动计算，共同祖先的刚体运动抵消；手指允许的 0.025 rad 与头部允许的 0.005 rad 漂移分别计入。
   双臂运动中监测该漂移，超过界限会请求取消，不再开始后续张手或抓取。
   已到位则跳过运动，但仍等待新反馈证明持续稳定；轨迹 action 返回成功本身不视为到位。
5. 配置手型时，重新检查手部路径，先开左手，再开右手。已张开的手跳过输出，仍检查新鲜稳定反馈。
   双臂和手指不同时运动，另一只手保持原位；右手单独张开时，必须先核验左手已张开。
6. 释放手部输出并观察 2 秒，联合确认双臂、手型、身体保持、命令板目标、控制模式和任务归属。
7. 全部成功后才发送抓取 action，由抓取后端开始获取当前观测、生成抓取计划。

准备阶段属于同一次抓取 admission 内声明的执行阶段，不递归调用 Gateway，不提前把父任务标为成功。
阶段名和进度通过 feedback 输出。失败、停止、反馈过期、未到位或预算耗尽均阻断抓取；不自动重试。
没有 `hands` 的姿态仍只控制双臂。左手完成后右手失败会保留部分完成记录，不闭手回滚。
action 结果包含 `preparation` JSON，记录 `phase_sec`、`total_sec`、`completed_hands` 和 `skipped_hands`；
阶段包括预检、接管、规划、双臂运动/稳定、左右手张开和释放后确认。

X2 的 `stream` 与 `policy_stream` 均对应 URS。本版依据公开模式的 `allows_trajectory` 允许在
`policy_stream` 执行调用方规划的关节轨迹，准备与抓取可以在同一模式内顺序完成。
厂商预设动作及记录手势的准入、取消规则保持独立，不因开放准备轨迹而扩大。
功能不会自动执行开机后的关节使能/站立序列，也不会进入零力矩。

## Mock 验证

在工作区根目录加载完整环境后：

```bash
source .shrc_local && colcon test --merge-install \
  --packages-select pose_composer aimdk_grasp robot_config skill_catalog skill_library \
  --event-handlers console_direct+ --return-code-on-test-failure
```

新增验收 `pose_composer/test/test_task_pose_gateway_mock.py` 使用真实 Gateway 子步骤路由、姿态 action
处理器、X2 模型/规划与快照校验，替换传输、硬件、准入状态、时钟和抓取为 Mock。
它验证已到位跳过、回位后重新开始感知/抓取、旧反馈/未到位/版本不一致/停止时阻断，以及独立回位。
组合姿态用例进一步覆盖先左后右、部分完成、父任务预算、阶段失败及取消后不开始感知/抓取。
`aimdk_grasp/test/test_hand_restore.py` 使用真实手部状态机与 Mock 输出端，覆盖接管、逐步插值、
执行权冲突、模式变化、反馈过期、身体/命令漂移、跟踪失败与释放后回弹。
`test_hand_restore_ros.py` 在隔离 ROS 域内验证公开话题、Gateway 服务、左右手数据顺序和输出释放。
运行时测试另通过 ROS 图和厂商 Mock 验证 policy_stream 中的轨迹执行、取消和模式保持。
这些测试不连接或控制真机，不依赖 LLM，不报告真机时延。

`test_fixed_poses.py` 使用真实固定执行器和左右手几何，验证举臂/单臂种子后回到 HOME，
覆盖 `return_to_task_pose("grasp_ready")` 与“双臂放下”入口以及左右手依次实际张开。
技能子步骤路由由 `test_task_pose_gateway_mock.py` 覆盖，生成姿态的求解、限位及路径由
`test_core.py`、`test_spec.py` 覆盖。传输、执行反馈与时钟使用 Mock，不属于真机验收。
手部初始食指关节设为 0.15 rad，确保没有将“两手已张开、跳过输出”当作张手成功。

2026-11-04 固定模式实现之前的正式验收：`pose_composer` 159 项、`skill_library` 343 项，
`aimdk_grasp` 手部准备/几何/ROS 通道三个测试文件 43 项通过，共 545 项。
三个包均经 `scripts/build.sh` 构建。手部定向验收命令：

```bash
source .shrc_local && colcon test --merge-install --packages-select aimdk_grasp \
  --event-handlers console_direct+ --return-code-on-test-failure \
  --pytest-args test/test_hand_restore.py test/test_hand_restore_geometry.py test/test_hand_restore_ros.py
```

## 第一版边界

只接受完整双臂关节目标；不支持世界坐标中的末端位置/姿态目标。手型必须显式配置。
已有 X2 CPU 路径检查覆盖自身几何和模型中的固定身体，不包含桌面、物体或人员的实时环境避障。
`hardware_approved` 不是环境安全证明。真实桌旁自动准备必须验证实际路径和环境避障，必要时接入
cuRobo 场景规划；本版示例不能直接批准为任意环境下的安全回位点。

硬件部署需要明确的合法双臂目标、空载约定、与实际装配一致的左右手模型及关节映射，
并核对固定下肢包围盒使用的站立姿态与现场路径。各手模型的 `R_*` 是模型内部别名。
Mock 使用真实双手几何和模拟执行反馈，不能替代现场路径与执行验证。
仓库示例中的两个批准字段均保持 false；具体现场确认记录在独立部署配置中。

本功能扩展了 `ExecuteArmPose` 消息定义。将来部署时需要同时重建 `ibrobot_msgs` 和相关调用方，
不能只把新增 Python 模块复制进仍使用旧消息类型的运行目录。姿态快照及仿真许可参数为启动只读参数。
准备完成后还会重新核验运行时健康、停止锁存与模式；模式变化不会被准备阶段自动改回。

## 不依赖 MuJoCo 的固定动作模式

板端设置 `pose_composer.mode: fixed`、`fixed_library` 指向离线编译的 JSON，
并关闭 `memory.enabled`，不配置 `llm`。保留 `task_poses` 和左右手模型配置。
`free_pose` 仅接受库中的种子 ID 或确切别名；未列出的描述、相对调整均拒绝，不触发在线生成。
`return_to_task_pose` 使用同一固定后端执行配置目标，并继续调用共享的双手恢复状态机。
工作站上的原动作生成模式保持可用。

离线编译器输出固定关节目标、凸体几何和质量数据，并同时使用原 MuJoCo 后端与板端算法
验证 HOME 与种子目标之间的双向路径。运行时采用 NumPy/SciPy，按实时起点生成单段关节空间
最小加加速度插值，最大速度 0.5 rad/s；危险直线路径直接拒绝，不生成绕行路径。
没有删除 SweepCheck：仍检查实测手指形状、头部、身体、双手间隙、站立腿部包络和静态重力。
腿部包围盒只用于快速排除远距离情况；接近时按已验证的站立腿部凸包计算实体净空，
保留 10 mm 模型误差余量。手—腿和双手之间按实际参与碰撞的手部链节计算位移余量，
不再把整只手的最大值用于所有碰撞球。拒绝日志包含净空、所需余量、链节和轨迹采样位置。
启动配置绑定动作文件哈希；文件变化、模型/关节限位不匹配会拒绝启动。
下发前必须再次取得 0.5 秒内的有限关节反馈，起点偏移不得超过 0.005 rad，
首段几何检查包含该偏移余量。物理后端位于 `aimdk_grasp`，Agent 层负责选择后端和协调任务。

本次只编译并启用 `arms_up`；其他种子尚未启用，含 keep-current 语义的种子暂不支持编译。
现场选择的 HOME 为双臂各 `[0.4, 0, 0, -1.2, 0, 0, 0]` rad，左右手使用本页的张开数组。
本次固定举臂动作和 HOME 的到位容差为 0.1 rad，仍要求连续稳定到位。
`pose_composer.lookup.enabled` 默认开启：Agent 先查询已编译种子，命中后跳过模型规划。
该开关独立于 `pose_composer.memory.enabled`；固定模式仍关闭可写记忆，不增加动作生成或保存能力。
该容差不改变轨迹下发前的 0.005 rad 起点偏移限制，也不改变碰撞检查余量。
编译方式与配置详见 [pose_composer README](../src/agent/pose_composer/README.md#fixed-seed-library-on-devices-without-mujoco)。

固定模式新增验收：工作站 `pose_composer` 171 项通过，配置文件/快照定向测试 23 项通过，
手部恢复定向测试 43 项通过，均经 `colcon test --merge-install`。
其中 `test_fixed_poses.py` 使用真实 CPU 几何和模拟传输贯通举臂、回位与双手实际张开，
并覆盖未知指令、模型/文件篡改、限位/速度、取消、起点漂移、反馈过期和非有限反馈的拒绝。
`test_compiled_geometry_reference.py` 在工作站对照 MuJoCo 验证 FK、静态重力与碰撞距离。
板端不运行依赖 MuJoCo 的工作站测试，使用：

```bash
source .shrc_local
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=87
export IBROBOT_FIXED_LIBRARY=/absolute/path/fixed_poses.json
colcon test --merge-install --base-paths src --packages-ignore sim_models \
  --packages-select pose_composer --pytest-args test/test_fixed_poses.py
colcon test-result --test-result-base build/pose_composer --verbose
```

独立测试部署的构建、原生 Mock、耗时及执行准备状态由部署目录中的证据文件记录；
这些结果与真机动作验收分开记录，不能仅因离线检查通过而宣称机器人已完成动作。
