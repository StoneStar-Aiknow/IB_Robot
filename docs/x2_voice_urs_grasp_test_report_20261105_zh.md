# X2 语音、URS 手势与抓取规划测试报告

## 1. 测试结论

测试日期为 2026-11-04 至 2026-11-05。
本文时间均采用北京时间。

已通过的验证包括：

- 真人语音进入自动语音识别（ASR）与 Agent。
- Agent 回复进入语音合成（TTS），现场操作员确认听到回复。
- 录制苹果图片的检测与分割。
- 合成点云的抓取候选生成。
- URS 修复的 26 项定向 `colcon test` 测试。
- 真实规划模型的 4 项纯规划回归。
- 修复后的构建、启动、服务就绪和只读技能校验。
- 两端流水线停机和残留模型进程清理。

URS 指上半身遥控分离模式 `UPPERBODY_REMOTE_SPLIT`。
修复后的手势与抓取共用运行时模式 `policy_stream`。

本次未完成修复后实机挥手或实物抓取验收。
软件测试通过不代表物体已经抓起。
全文的“通过”仅适用于表格列出的验证范围。

报告编写时，两端流水线均已关闭。
编写报告没有再次启动流水线或提交动作。

## 2. 版本与环境

| 项目 | 本次使用值 |
| --- | --- |
| 部署来源分支 | `1028_agi_lyx_all` |
| 部署源码基准 | `b712892ca6772ae68506e1a5312573ce89076c5a` |
| 本地提交分支 | `deploy/310p-20261104` |
| 修复源码 | 与本报告位于同一提交 |
| 本地部署副本 | `/mnt/data/lwh/IB_Robot_hz_deploy_20261104` |
| 310P 工作区 | `/root/lwh/IB_Robot_hz_20261104` |
| Orin 工作区 | `/home/agi/IB_Robot_hz_20261104` |
| 310P 环境 | openEuler Embedded、ROS 2 Humble、Python 3.11.6 |
| 模型执行 | 310P 复用昇腾环境，Orin 复用 CUDA/cuRobo 环境 |
| 机器人运行时 profile | `x2_ultra_omnihand` |
| 真机通信 | `ROS_DOMAIN_ID=0`，跨主机 DDS 通信 |
| 隔离测试通信 | 域 96、97 或 98，`ROS_LOCALHOST_ONLY=1` |

DDS 指数据分发服务（Data Distribution Service）。
模型、虚拟环境和厂商软件开发工具包（SDK）均复用现场部署。
测试没有重新安装整套环境。

前序部署补齐了既有环境中的 `lxml` 和 `zstandard`。
这些是部署环境处理，不属于本提交的源码改动。

验证结果覆盖本节列出的部署基准。
本次未在 `oee_summit_2026` 的独立源码树上重跑测试。

## 3. 流水线组成与配置

310P 主管线承载 ASR、Agent、TTS 和能力网关（Gateway）。
主管线还承载机器人运行时与抓取模型服务。
Orin 抓取头提供相机采集和抓取规划后端。

现场数据路径为：

```text
真人语音
  -> 厂商 audio.speech
  -> ASR /voice_command
  -> Agent /agent/request
  -> Gateway 技能准入
  -> 手势回放或抓取执行器

Agent 回复
  -> ZipVoice TTS
  -> runtime audio.play
  -> 机器人扬声器

抓取执行器
  -> Orin 相机与规划后端
  -> 310P 检测、分割、候选生成服务
  -> 运行时 policy_stream
```

现场最终使用的两份配置为：

- 310P：`/root/lwh/IB_Robot_hz_20261104/commissioning/robots/hz_site_live.yaml`。
- Orin：`/home/agi/IB_Robot_hz_20261104/commissioning/robots/hz_site_live.yaml`。

配置生成器继承 `aimdk_x2_voice_mirror_grasp.yaml`。
配置保留 ASR、Agent、ZipVoice TTS 和 3 个抓取模型。
配置关闭照镜子会话、人体重定向和视觉交互。

目录保留 7 项技能：

| 技能 | 用途 |
| --- | --- |
| `wave_hand` | 挥手 |
| `handshake` | 握手 |
| `raise_hand` | 举手 |
| `blow_kiss` | 飞吻 |
| `clap_hands` | 鼓掌 |
| `self_introduction` | 播放自我介绍录音 |
| `pick_object` | 根据目标名称抓取物体 |

`pick_object` 接收通用 `target_name`。
本次纯规划回归使用 `an apple`。
通用参数能力不代表所有物体均已完成实机验收。
抓取配置保留参考配置的 `arm_side: auto` 和重试参数。

## 4. 实际启动方法

本节记录现场已经执行的命令与参数。
现场启动器 `site_runner.sh` 加载各端的环境脚本。
环境脚本以本工作区的 `.shrc_local` 为入口。
环境脚本还选择厂商 SDK 和临时 DDS 配置。

启动器、DDS 文件和日志属于现场部署材料。
这些材料未全部包含在本提交中。
本节不是干净检出后的完整安装说明。

### 4.1 构建与配置生成

前期准备完成了 310P 和 Orin 的部署构建。
URS 修复后，310P 增量构建了 3 个包。
Orin 抓取算法未变，修复时保留了抓取头进程。

310P 的修复构建命令为：

```bash
cd /root/lwh/IB_Robot_hz_20261104
source commissioning/env.sh
./scripts/build.sh --mixin release test -- \
  --base-paths src/robots/aimdk/aimdk_robot \
               src/agent/ibrobot_agent \
               src/agent/skill_library \
  --packages-select aimdk_robot ibrobot_agent skill_library \
  --parallel-workers 2
```

两端分别运行 `commissioning/prepare_voice.py`。
生成器将本机路径写入现场 YAML。
310P 随后运行 `commissioning/preflight.py`。

### 4.2 首轮真机 dryrun

首轮使用真实麦克风和扬声器。
310P 配置为 `hz_site_dryrun.yaml`。
启动参数为 `authorize_motion:=false`。
Agent 配置为 `execution_enabled: false`。

首轮验证语音理解与回复，不提交抓取动作。
Orin 抓取头同步启动，提供相机和规划服务。

### 4.3 最终 Orin 抓取头

最终 Orin 抓取头使用下面的启动参数。
进程启动于 2026-11-04 23:47:25。

```bash
cd /home/agi/IB_Robot_hz_20261104
export HZ_TARGET=robot HZ_ROS_DOMAIN_ID=0
nohup setsid bash "$PWD/commissioning/site_runner.sh" orin \
  python3 /opt/ros/humble/bin/ros2 launch aimdk_grasp x2_grasp_head.launch.py \
  "config_path:=$PWD/commissioning/robots/hz_site_live.yaml" \
  runtime_timeout:=180 \
  > "$PWD/commissioning/logs/no_mirror_orin_20261104_234725.log" 2>&1 < /dev/null &
```

URS 修复期间，Orin 抓取头保持运行。
修复只同步了配置和源码，没有重启 Orin 抓取头。

### 4.4 URS 修复后的 310P 主管线

用户在现场测试阶段明确授权实际运动。
最终主管线沿用 `authorize_motion:=true`。
这个参数允许真实运动。
本次修复验证没有自动发送运动请求。

最终启动时间为 2026-11-05 00:41:20。

```bash
cd /root/lwh/IB_Robot_hz_20261104
export HZ_TARGET=robot HZ_ROS_DOMAIN_ID=0
nohup setsid bash "$PWD/commissioning/site_runner.sh" 310p \
  python3 /opt/ros/humble/bin/ros2 launch embodied_bringup embodied_pipeline.launch.py \
  "config_path:=$PWD/commissioning/robots/hz_site_live.yaml" \
  control_mode:=moveit_planning \
  with_moveit:=false use_sim:=false authorize_motion:=true \
  > "$PWD/commissioning/logs/urs_fixed_310p_20261105_004120.log" 2>&1 < /dev/null &
```

`moveit_planning` 在这里是技能控制模式名称。
X2 抓取使用 cuRobo 和流式执行，不启动 MoveIt。
现场配置将 `moveit_planning` 映射到 `policy_stream`。
修复后，`named_motion` 也映射到 `policy_stream`。

后台启动采用独立进程组，并记录进程号与启动标识。
停机只处理本部署拥有的进程组。

本分支没有 `/embodied/authorize_motion` 动态授权服务。
运动授权取自启动参数，改变参数需重启主管线。
`voice_confirm` 不能阻止本分支的异步 typed Program 提交。
运动限制依赖 dryrun 配置和 Gateway 授权检查。

## 5. 成功的功能验证

### 5.1 语音与离线感知

下表汇总前期无运动验证和现场 dryrun。
这些结果不是 URS 修复后的实机动作结果。

| 功能 | 实际结果 | 验证范围 |
| --- | --- | --- |
| 音频文件到 ASR | 识别“帮我抓取苹果” | 合成音频输入 |
| ASR 到 Agent | 生成 `pick_object(target_name="an apple")` | dryrun，未提交动作 |
| Agent 到 TTS 播放话题 | 收到 72,320 字节音频 | 隔离播放话题 |
| 真人语音到 Agent | 请求 `voice-7835754d564d` 完成 dryrun | 真实麦克风输入 |
| 现场语音回复 | 回复“好的，我来帮你拿起苹果。” | 操作员确认听到 |
| 苹果图片检测 | 置信度 0.77487，检测耗时 411.66 ms | 2026-11-01 录制图片 |
| 苹果图片分割 | 生成 1 个掩膜，耗时 217.92 ms | 同一录制图片 |
| GraspGenX 候选生成 | 生成 8 个候选，耗时 4,596.20 ms | 合成球形点云 |

真人口令曾识别为“帮我发去苹果”。
Agent 仍理解为拿取苹果，并生成回复。
这证明语音接入和意图理解在该案例中成功。
本次未统计 ASR 准确率。

### 5.2 修复后的真实模型纯规划回归

测试使用部署配置中的真实规划模型。
测试脚本未创建执行客户端，也未提交动作。

| 输入与上下文 | 模型输出 | 结果 |
| --- | --- | --- |
| 手势模式下输入“抓取苹果” | `pick_object`，目标为 `an apple` | 通过 |
| 带旧拒绝对话，再输入“抓取苹果” | `pick_object`，目标为 `an apple` | 通过 |
| 输入“先挥手，然后抓取苹果” | `wave_hand` 后接 `pick_object` | 通过 |
| 输入“不要抓取苹果” | 对话回复，动作列表为空 | 通过 |

### 5.3 修复后的部署与准入验证

| 检查项 | 结果 | 验证范围 |
| --- | --- | --- |
| 5 套配置加载、绑定与目录编译 | 通过 | 配置静态校验 |
| 5 条录制轨迹加载与限速检查 | 通过 | 轨迹文件校验 |
| 两类控制模式映射 | 均为 `policy_stream` | 运行中的配置快照 |
| 安装后的回放模式 | `stream`、`policy_stream` | 安装产物核对 |
| 规划器提示版本 | `agent-planner-v4` | 实际导入代码核对 |
| 310P 修复增量构建 | 3 个包成功 | 构建日志 |
| Agent readiness | `true` | 只读就绪服务 |
| 3 个抓取模型服务 | 全部可发现 | 服务发现 |
| Orin 相机采集服务 | 可发现 | 服务发现 |
| TTS 合成服务 | 可发现 | 服务发现 |
| 抓取后端预热 | `grasp backends prepared` | 启动日志 |
| Gateway 状态 | `READY`、`busy=false` | 只读状态查询 |
| 7 项技能目录 | 均可见 | Gateway 目录 |
| `wave_hand` 校验 | `allowed=true` | 只读校验，未执行 |
| `pick_object` 校验 | `allowed=true`，目标为 `an apple` | 只读校验，未执行 |

服务发现不等于模型预热完成。
Gateway 的 `allowed=true` 不等于实机动作成功。

最终启动日志包含：

```text
voice bridge ready: /voice_command -> /agent/request
VoiceASRNode initialized
grasp backends prepared; accepting /manipulation/execute_pick goals
```

## 6. 定向 colcon 测试

### 6.1 结果

| 包 | 通过数 | 未选中数 | 检查内容 |
| --- | --- | --- | --- |
| `aimdk_robot` | 6 | 249 | 回放、手臂仲裁、轨迹、目标侧检查 |
| `ibrobot_agent` | 18 | 398 | 规划提示、响应解析与调用方回归 |
| `skill_library` | 2 | 336 | 共享模式与 idle 准入检查 |
| 合计 | 26 | 983 | 定向验证 |

两个测试命令的退出码均为 0。
“未选中”表示本次筛选没有执行这些用例。
它不同于执行后通过，也不同于运行时跳过。

`policy_stream` 回放测试未调用厂商预置动作。
测试确认手臂拒绝其他来源的插入指令。
测试还确认模拟平台保持 URS。
这些结果来自厂商接口模拟节点，不来自真实机械臂。

Gateway 测试覆盖挥手、抓取、握手的连续准入。
测试确认这 3 次准入没有请求运行时模式切换。

### 6.2 执行命令

运行时测试使用隔离域 98。
310P 上的最终命令为：

```bash
cd /root/lwh/IB_Robot_hz_20261104
export HZ_ROS_DOMAIN_ID=98
source commissioning/env.sh
timeout --signal=TERM --kill-after=15s 180s \
  python3 -m colcon test --merge-install \
  --base-paths src/robots/aimdk/aimdk_robot \
  --packages-select aimdk_robot --executor sequential \
  --return-code-on-test-failure --event-handlers console_cohesion+ \
  --pytest-args \
  -k 'stream_mode_replays_the_recording_and_owns_the_arm_channel or shipped_trajectories_load_and_replay_advertised_motions or replay_refuses_a_side' \
  -vv
```

Agent 与 Gateway 定向测试使用隔离域 96。
310P 上的命令为：

```bash
cd /root/lwh/IB_Robot_hz_20261104
export HZ_ROS_DOMAIN_ID=96
source commissioning/env.sh
timeout --signal=TERM --kill-after=10s 150s \
  python3 -m colcon --log-base commissioning/logs/urs_unit_colcon test \
  --merge-install \
  --base-paths src/agent/skill_library src/agent/ibrobot_agent \
  --packages-select skill_library ibrobot_agent --executor sequential \
  --return-code-on-test-failure --event-handlers console_cohesion+ \
  --pytest-args \
  -k 'gesture_pick_gesture_share_policy_stream_without_a_mode_switch or idle_mapped_skill_mode_never_requests_a_runtime_mode_switch or prompt_ or parser_rejects'
```

测试环境将 `ROS_LOCALHOST_ONLY` 设置为 `1`。
测试没有接入真机域 0。

纯规划回归在隔离环境运行：

```bash
cd /root/lwh/IB_Robot_hz_20261104
export HZ_ROS_DOMAIN_ID=96
source commissioning/env.sh
timeout --signal=TERM --kill-after=10s 180s \
  python3 commissioning/probe_urs_planner.py
```

### 6.3 首轮失败与处理

首轮运行时测试出现夹具冷启动超时。
夹具的 15 s 等待时间不足以覆盖冷启动。
夹具初始化失败后未清理节点，干扰了后续测试。

修复后的夹具采用 30 s 有界等待。
夹具在初始化失败时清理模拟节点与数据平面。
生产运行时的反馈阈值和安全检查保持原值。

首轮结果为 4 项失败、2 项通过。
清理夹具后，复跑曾出现 1 项失败、5 项通过。
补齐冷启动等待后，最终 6 项全部通过。
原始失败日志保留在现场证据目录。

## 7. 故障定位与源码改动

### 7.1 手势错误选择厂商预置动作

修复前，挥手和握手返回：

```text
NAMED_MOTION_PLATFORM_NOT_READY
platform is in UPPERBODY_REMOTE_SPLIT;
preset motions require STAND_DEFAULT
```

现场 `named_motion` 控制模式原先映射到 `idle`。
运行时因此选择厂商预置动作路径。
平台实际处于 URS，预置动作要求稳定站立。
运行时拒绝了动作请求。

机器人 profile 已有 5 条录制轨迹。
修复将手势映射到抓取共用的 `policy_stream`。
修复还允许运行时在这个模式下回放轨迹。
预置动作的稳定站立检查继续保留。
轨迹限速、反馈跟随和手臂仲裁检查继续保留。

### 7.2 规划器误判抓取能力

请求 `voice-502ac8fecdc6` 输入为“抓取苹果”。
规划器直接返回 `UNSUPPORTED_TASK`。
模型声称只支持手势，需要先切换控制模式。
这次请求尚未进入抓取执行器。

当前目录实际包含 `pick_object`。
修复明确了目录与当前模式的不同含义。
当前目录决定支持哪些技能。
Gateway 检查技能的模式准入和运动授权。
旧对话中的错误回复不能覆盖当前目录。

### 7.3 本提交的 8 个源码文件

| 文件 | 改动 |
| --- | --- |
| [OmniHand profile](../src/robots/aimdk/aimdk_robot/profiles/x2_ultra_omnihand.yaml) | 增加 `policy_stream` 回放模式 |
| [planner.py](../src/agent/ibrobot_agent/ibrobot_agent/planner.py) | 区分当前模式和目录能力 |
| [node.py](../src/agent/ibrobot_agent/ibrobot_agent/node.py) | 记录提示版本 `agent-planner-v4` |
| [运行时回放测试](../src/robots/aimdk/aimdk_robot/test/test_runtime_bridge.py) | 覆盖共享模式，修复夹具清理和等待 |
| [Gateway 准入测试](../src/agent/skill_library/test/test_runtime_named_motion_executor.py) | 验证手势与抓取之间不切模式 |
| [配置生成器](../commissioning/prepare_voice.py) | 生成非照镜子配置和共享模式映射 |
| [配置预检](../commissioning/preflight.py) | 检查轨迹、模式和抓取技能可见性 |
| [纯规划回归](../commissioning/probe_urs_planner.py) | 覆盖 4 个真实模型案例 |

修改的 Python 文件通过 Ruff 检查与格式化。
部署脚本格式化前后的 Python 语法树一致。
本报告随同这 8 个文件提交。

## 8. 未通过与尚未验收的内容

### 8.1 前期全包测试仍有失败

前期隔离测试记录如下：

| 包 | 前期结果 |
| --- | --- |
| `aimdk_dataplane` | 30 个测试通过，包含 2 个 CTest 目标 |
| `voice_asr_service` | 209 passed，15 skipped |
| `ibrobot_agent` | 396 passed，20 failed |
| `voice_tts_service` | 164 passed，8 failed |

Agent 失败包括缺少 `json` 导入和行为断言失败。
TTS 失败包括测试导入错误和模型音频契约断言失败。
本次 26 项定向通过不能消除这 28 项历史失败。
本次未重新执行全部包测试。

部署工作区另有前序测试去重改动。
文件为 `src/agent/ibrobot_agent/test/test_planner.py`。
这项旧改动未纳入本提交。
测试选择计数对应测试时的部署工作区。

### 8.2 实机与模型限制

- 修复后的实机挥手和握手尚未验收。
- 实物抓起、抬升、保持、放回、退回尚未验收。
- 真实苹果点云的可执行抓取候选尚未验收。
- Orin 碰撞规划与机械臂执行的完整任务尚未验收。
- ASR 准确率和语音自激抑制尚未完成专项测试。
- 自我介绍录音 `self_introduction.wav` 仍缺失。
- 照镜子功能按测试配置关闭。

GraspGenX 仍提示缺少 3 项夹爪资产：

```text
points.json
proc_gripper_only_pointnet_vae_repr.json
tsdf.npy
```

服务使用 dummy values 的警告仍需单独分析。
本报告不据此宣告真实抓取安全。

一次历史重启还出现过支撑平面检查失败：

```text
NoSupportError: No near-field support plane beneath the target
```

这个错误发生在抓取后端准备阶段。
它不同于规划器的 `UNSUPPORTED_TASK`。
最终修复启动已完成预热，但不能豁免场景检查。

## 9. 停机结果

测试结束后，用户要求关闭流水线。
两端分别执行了本部署的停止入口：

```bash
# 在 310P 工作区执行
bash commissioning/site.sh stop 310p

# 在 Orin 工作区执行
bash commissioning/site.sh stop orin
```

Orin 抓取头正常退出。
310P 的运行时、Agent 和抓取执行器均已退出。
310P 的 GraspGen 模型进程在 INT 后仍挂起。
核对进程组身份后，清理流程发送 TERM。
等待 10 s 后，清理流程对残留进程组发送 KILL。

停机后再次检查了两端进程列表。
检查未发现本部署的主管线或抓取头进程。
检查也未发现本次部署的活动模型残留进程。
测试结束后没有自动重启流水线。

## 10. 证据位置

原始日志与 JSON 结果保留在现场部署目录。
这些原始证据未全部纳入 Git 提交。
本文保留了结果、命令和关键错误内容。

| 证据 | 部署目录中的位置 |
| --- | --- |
| 最终 310P 启动日志 | `commissioning/logs/urs_fixed_310p_20261105_004120.log` |
| 最终 Orin 启动日志 | `commissioning/logs/no_mirror_orin_20261104_234725.log` |
| 运行时定向测试 | `commissioning/logs/urs_colcon_runtime_final_20261105.log` |
| Agent/Gateway 定向测试 | `commissioning/logs/urs_colcon_unit_20261105.log` |
| 真实模型纯规划结果 | `commissioning/logs/urs_planner_probe_20261105.json` |
| 配置预检 | `commissioning/logs/urs_preflight_20261105.log` |
| 修复构建日志 | `commissioning/logs/urs_build_310p_20261105.log` |
| Agent 与服务就绪 | `commissioning/logs/urs_site_ready_20261105.json` |
| Gateway 状态 | `commissioning/logs/urs_gateway_status_20261105.json` |
| 挥手只读校验 | `commissioning/logs/urs_validate_wave_20261105.json` |
| 抓取只读校验 | `commissioning/logs/urs_validate_pick_20261105.json` |
| 真人语音 dryrun 日志 | `commissioning/logs/operator_dryrun_310p_20261104_221237.log` |
| 前期全包测试 | `commissioning/logs/site_colcon_test_retry.log` |

本地证据副本位于部署副本的两个目录：

- `commissioning/evidence_resume_20261104/`。
- `commissioning/evidence_urs_20261105/`。

前者保存离线语音、检测、分割和候选生成结果。
后者保存 URS 修复后的定向测试与部署结果。
现场耳听结论来自原会话中的操作员确认。
