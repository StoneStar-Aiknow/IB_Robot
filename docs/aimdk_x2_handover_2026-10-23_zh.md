# X2 语音 + 镜子栈 交接清单（2026-10-23）

> 面向接手人：这份是**操作手册**，先看「1. 怎么启动全链路」和「2. 怎么停」，再按需翻后面的坑。
> 设计来龙去脉、逐日排查过程、每个补丁为什么这么写，在
> [`aimdk_x2_voice_mirror_handover_zh.md`](aimdk_x2_voice_mirror_handover_zh.md)（500+ 行，第 0–17 节）。

## 0. 一句话现状

310P 板子（`10.42.0.178`，机器人网段 `10.0.1.15`）上的 `/root/workspace/lyx/IB_Robot_hz` 跑一整套 ROS 2 栈：
**厂商增强流 → 自研 ASR → 自研 agent（planner 是 Windows 上的 qwen3.8-27B）→ 5 个预设手势 / 照镜子 / 自我介绍 → ZipVoice TTS**。
2026-10-23 晚真机全链路已验证可用；遗留问题集中在**镜子跟随只有 ~5 Hz**和**一个 `runtime_lost` 误判**（见第 6 节）。

---

## 1. 怎么启动全链路（重点）

顺序不能反：**先大模型 → 再对时 → 最后主栈**。三段各自失败的表现见 1.5。

### 1.1 第一段：Windows 侧的大模型（planner）

agent 的规划器走本机的 OpenAI 兼容接口，两个进程串联：

| 步骤 | 命令 / 操作 | 说明 |
|---|---|---|
| ① 起引擎 | 双击 `D:\ninfer86\start-pq2-dflash-lan.bat` | ninfer-serve，监听 `127.0.0.1:8094`，模型 `Ternary-Bonsai-2-27B`，model-id `qwen3.8-27b` |
| ② 起中继 | WSL 里 `systemctl --user start ninfer-relay.service` | `/home/oee/ninfer_strip_relay.py`，把 `0.0.0.0:18094` 转到 `127.0.0.1:8094` |
| ③ 自检 | `curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:18094/v1/models` | 期望 `200` |

板子侧必须也能访问：在板子上 `curl -s -o /dev/null -w "%{http_code}\n" http://10.42.0.177:18094/v1/models` 期望 `200`。
agent 的 `base_url` 写死为 `http://10.42.0.177:18094/v1`（见第 7 节板端改动）。

### 1.2 第二段：板子对时（必须一直跑）

```bash
ssh root@10.42.0.178
systemd-run --unit=x2-clock --collect /root/workspace/lyx/clock_sync_loop.sh
```

- `clock_sync_loop.sh`：每 12 s 调 `tools/sync_platform_clock.py`，用厂商消息时间戳校准本机时钟。
- **为什么必须**：runtime 校验时钟偏差 > 100 ms 会直接 FAULTED，`wait_for_runtime` 拿不到就绪接口，整个 launch 零残留退出。机器人没有 RTC，重启后时钟必然偏。
- 验证：`systemctl is-active x2-clock` → `active`。

### 1.3 第三段：板子主栈

```bash
systemd-run --unit=x2-all --collect bash -c "/root/workspace/lyx/run_x2_all.sh > /tmp/x2_all.log 2>&1"
```

`run_x2_all.sh` 内部做三件事：

1. `source /root/workspace/lyx/ibrobot_env.sh`（唯一环境入口：cd 工作区 → `source .shrc_local` → source 厂商 AimDK overlay）
2. `unset http_proxy/https_proxy/...`（**否则 Clash 会劫持 VLM 调用**，表现为 `VLM API HTTP 502`）
3. `ros2 launch embodied_bringup embodied_pipeline.launch.py robot_config:=aimdk_x2_voice_mirror_all control_mode:=named_motion authorize_motion:=true`

要点：

- `robot_config:=aimdk_x2_voice_mirror_all` 是**语音 + 5 手势 + 照镜子 + 自我介绍**的合并部署 profile（继承链：`aimdk_x2_all → voice_mirror → magicmirror`）。
- `authorize_motion:=true` 是动作授权开关，**不带它手势不会真动**。
- 一定要用 `systemd-run` 起：SSH 会话被 cgroup `/CtrlCPU` 钉在 CPU0（`nproc`=1），只有 systemd 起的服务才拿得到 16 核。

### 1.4 启动后怎么判断“真的起来了”

按这个顺序看，一步不通就别往下走：

| 检查 | 命令 | 期望 |
|---|---|---|
| 1. 栈在跑 | `systemctl is-active x2-all` | `active` |
| 2. runtime | `ros2 topic echo /runtime_status --once \| head -3` | `lifecycle: ACTIVE`；再看 `faults: []` |
| 3. ASR | `ros2 topic echo /voice_status --once` | `data: listening` |
| 4. 语音链路 | 说「挥挥手」 | 日志 `[say] 正在执行：挥手。` → 约 7 s → `SUCCEEDED` → `[say] 执行完成。` |
| 5. 镜子 | 说「照镜子」 | 四句语音（启动中/我准备好咯/开始模仿啦）+ `MIRRORING/running` |
| 6. 停镜子 | 说「停止照镜子」 | `下次再见` + `set_session stop: success=True … platform confirmed STAND_DEFAULT`（约 1 s） |

日志统一在 `/tmp/x2_all.log`（launch 输出）和 `/tmp/roslog/`（节点日志）。
时间线参考（实测冷启动）：15 s runtime ACTIVE → 25 s voice bridge ready → 60 s ASR 模型加载完 → 60–90 s 感知模型就绪（`output_hz` 从 0 变正）→ 语音四句预热完成。

### 1.5 起不来时的三种典型表现

| 表现 | 原因 | 处置 |
|---|---|---|
| 日志 `runtime lifecycle is 'FAULTED'`，`faults: clock disagrees` | 没跑对时，或对时被停 | 起 `x2-clock`，等 12 s 再看 |
| 语音识别正常、机器人不回话；日志 `VLM API network error: [Errno 111] Connection refused` | Windows 引擎或 18094 中继没起 | 回到 1.1，先起 bat 再起 relay |
| 识别对了、动作被拒：`NAMED_MOTION_PLATFORM_NOT_READY … platform is in UPPERBODY_REMOTE_SPLIT` | 有人用厂商工具把运控切到了 JD/URS，预设动作要求 `STAND_DEFAULT` | 用厂商工具按 JD → SD 切回；确认 `action_desc: STAND_DEFAULT` |

---

## 2. 怎么停

```bash
systemctl stop x2-all x2-clock
```

Windows 侧的引擎/中继可以留着（不占板子资源）。彻底停：`systemctl --user stop ninfer-relay.service` + 关闭引擎窗口。

---

## 3. 语音怎么用（现场提示）

- **只说短句**：`挥挥手` / `握个手` / `举手` / `比个心` / `鼓个掌` / `照镜子` / `停止照镜子` / `自我介绍`。长句和快语速识别率明显下降（480 ms 流式 int8 模型 + 厂商 VAD 分段的固有上限）。
- 一句说完等它回完再说下一句，`confirm_mode: auto` 下并发请求会被丢掉（会说“我正在执行上一个任务，请稍等”）。
- 手势指令在镜子轮次进行中会被拒：镜子独占 `policy_stream`，日志 `another runtime producer is active`。

---

## 4. 当前栈里有什么

| 项 | 值 |
|---|---|
| 部署 profile | `aimdk_x2_voice_mirror_all.yaml`（控制模式 `named_motion` 为默认，镜子用 `model_inference`） |
| 节点数 | 约 105（含厂商栈） |
| ASR | `models/voice_asr/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05`，外部 VAD（厂商分段），`modified_beam_search` + 热词 |
| 热词表 | `robot_config/config/hotwords/x2_commands.txt`（profile 里 `voice_asr.hotwords_file`） |
| TTS | ZipVoice，`ascend_310p`，bundle `models/zipvoice`（schema v3） |
| 自我介绍 | 录音回放，`assets/audio/self_introduction.wav`（34.5 s），走 `/voice_tts/play`，播放期间 ASR 静音 |
| 镜子 | `mirror_session.publish_enabled: true`（真发 `/aimdk/arm/commands`，实测 43–50 Hz）；`voice.enabled: true`（四句语音）；`operator.send_enabled: true`（物理触发生效） |
| 感知 | 前中相机 30 Hz → YOLOX + PEAR（ROS 服务），姿态流 `published` 正常 |
| 日志 | `/tmp/x2_all.log` |

---

## 5. 已验证的能力（2026-10-23 真机数据）

| 能力 | 证据 |
|---|---|
| 启动 | 105 节点，`runtime ACTIVE`，`faults: []` |
| 语音→LLM→TTS | 注入「你好」→ 中文回复并播报 |
| 手势 | 「挥挥手」→ `wave_hand` 7.18 s `SUCCEEDED`（safety allowed=True） |
| 照镜子 | 接管 `policy_stream`、`commands_sent` 数千条、`/mc/upper_body_command` 43.5 Hz |
| 停镜子 | `set_session stop` 0.98–1.06 s，平台确认 `STAND_DEFAULT` |
| 镜子语音 | 启动中 / 我准备好咯 / 开始模仿啦 / 下次再见 四句全部播报 |
| 物理触发 | `mirror_operator: send_enabled=true: gestures and touches WILL call the Gateway and the session` |
| hand 反馈 | 全日志 0 次 `joint feedback stale`、0 次 DEGRADED（hand 流实测 150 Hz） |
| 自我介绍 | 录音回放 34.5 s，前后无 TTS 报幕 |

---

## 6. 已知问题（按接手优先级）

### 6.1 镜子跟随慢（体验问题，未解）

- 实测：姿态流 `output_hz 5.2–13 Hz` 波动，retarget `age_ms` 峰值 1.9 s。
- 根因：**一块 310P 只有一个 device**，PEAR（1.3G）、YOLOX（191M）、ZipVoice（502M）共用，TTS 一次合成约 2 s NPU，感知被挤。
- 可行的三条路：① 输出端 50 Hz 线性插值（治本，改 retarget/会话输出，约 30 行）；② 降 YOLOX 刷新（1 s→2 s）抢 1–2 Hz；③ 镜子轮次内把 TTS 限成短应答。

### 6.2 `runtime_lost` 误判（会让轮次提前结束）

- 现象：镜子轮次进行中被结束，日志 `MIRRORING/running -> WAIT/none (runtime_lost)`。
- 原因：会话要求 2 s 内收到 `/runtime_status`，而 runtime 只发 **1 Hz**；语音预热/推理让板子变忙时会话自己晚了。外部探针实测 `max_gap 1.016 s`，runtime 本身正常。
- 修法（未做）：`src/manipulation_execution/manipulation_execution/mirror_session/runtime.py` 的 `STATUS_MAX_AGE_S: 2.0 → 4.0`（板端补丁，属 grangerxsp 的包）。

### 6.3 `upper-body command withheld`

- 日志：`upper-body command withheld: no fresh measurement to hold the uncommanded joints`（一轮出现约 6 次）。
- 含义：某一拍没拿到 head 两关节的新鲜测量（0.5 s 窗口），runtime 按设计**拒绝发半成品指令**（宁可不动也不发零填充）。这是安全行为，不是崩溃；但会表现为动作顿一下。

### 6.4 长句 ASR 差

- 现状：短指令准（`做个挥手动作`、`停止照镜子`），自由长句常错（`等级太低了`、`你先按摩站后那啥`）。
- 根治：把“厂商 VAD 分段完成后的最终识别”换成非流式 paraformer（节点已有 offline 分支，需下载 ~230 MB 并重新打包 bundle）。

### 6.5 物理触发的语义（容易被误解）

operator 现在看的是**后脑相机**，所以：

- 双遮 → `mirror_start`；
- 长遮（**包括摸后脑勺**）→ `mirror_stop`；
- 摸头传感器（`/aima/hal/sensor/touch_head`）→ 开始/取消轮次。

想要“摸头=开始”的手感，得走 touch_head，或者把 `mirror_session.operator.camera_topic` 覆盖回前中相机。

---

## 7. 板端本地改动（**没入库，换板/重装要带**）

| 文件 | 改了什么 | 归属/处置 |
|---|---|---|
| `aimdk_robot/runtime_node.py` | ①`player_state()` 访问器（T6 固件回 `task_id 0`，只能靠 player_state 判起停）；②**已包含上游 `9e31a250` 的 hand callback group 修法** | XiaoqiangWu 的包，上游化需与其协调；合并推送后拉取会覆盖 ①，需重新补 |
| `aimdk_robot/named_motion.py` | `task_id ≤ 0` → `_await_preset_without_task()`，用 player_state 判完成 | 同上 |
| `robot_config/robot_config/loader.py` | `robot_supported_control_modes()` 支持显式 `skill_supported_control_modes`（X2 没有导航服务，`nav_stage: hybrid` 走不通，否则手势+镜子不能共存） | 属框架改动，建议上游化 |
| `robot_config/config/robots/aimdk_x2_voice_agent.yaml` | planner `base_url` → `http://10.42.0.177:18094/v1` | 现场部署配置，留板端 |
| `libs/lerobot` | 检到 `a2be6a8a`（`30da8e68` 要求 Python ≥3.12） | 环境适配 |
| `assets/audio/self_introduction.wav` | 自我介绍录音（2.2 MB，未跟踪） | 二进制，留板端 |
| `robot_config/config/mirror_session/demo_relaxed.yaml` | 放宽新鲜度/确认时间的 demo 规则副本（当前 profile 用的是 `demo.yaml`，此文件未启用） | 未跟踪 |
| `robot_config/config/robots/aimdk_x2_voice_raw.yaml` | 原始流（无 AEC）部署 profile | 未跟踪 |

> 板端编译产物与源码是 **硬链接**（`--symlink-install`）：改 `src/` 里的 `.py`/`.yaml` 后 **只需重启 `x2-all`**，不用 `colcon build`。只有改 `setup.py`/入口点/新增文件才需要重编。

---

## 8. 仓库与提交

- 仓库：`gitcode.com/grangerxsp/IB_Robot_hz`，分支 **`1028_agi_lyx`**（板端工作区同一分支）。
- 本线相对 `f4219dd9` 新增：`d13fb421`（合入 `origin/1028_agi` 的 hand callback group + operator 相机）、`cb588911`（交接文档第 17 节）、以及 `feat(robot_config)` 的镜子语音/物理触发开关、本文件的提交。
- 2026-10-23 合入的上游内容：`9e31a250`（hand 独立 callback group，修 `joint feedback stale: hand`）、`62121c77`（镜子 operator 相机改由 `operator.camera_topic` 指定，magicmirror 用后脑相机）。

---

## 9. 接手后的建议顺序

1. 先把 1.1–1.4 走一遍，确认现场可用（30 分钟）。
2. 修 6.2 的 `STATUS_MAX_AGE_S`（一行，风险最低、立刻减少“轮次莫名结束”）。
3. 再攻 6.1 的镜子平滑度（插值或减 NPU 争用），这是体验最大的一块。
4. 长期：6.4 的离线 ASR、把 7 节里框架类补丁（loader、player_state）推上游。

## 10. 常用命令速查

```bash
# 起 / 停
systemd-run --unit=x2-clock --collect /root/workspace/lyx/clock_sync_loop.sh
systemd-run --unit=x2-all --collect bash -c "/root/workspace/lyx/run_x2_all.sh > /tmp/x2_all.log 2>&1"
systemctl stop x2-all x2-clock

# 状态
ros2 topic echo /runtime_status --once | head -5     # lifecycle + faults
ros2 topic echo /voice_status --once                  # listening?
ros2 topic echo /mirror_session/status --once         # state / phase
ros2 node list | wc -l                                # ~105

# 平台动作（预设动作要求 STAND_DEFAULT）
ros2 topic echo /aima/mc/common/state --once | grep action_desc

# 手工触发一条指令（绕过 ASR，验证 agent/技能）
ros2 topic pub --once /voice_command std_msgs/msg/String "{data: '挥挥手'}"

# 日志
tail -f /tmp/x2_all.log
```
