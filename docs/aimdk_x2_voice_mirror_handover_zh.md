# X2 语音 + 照镜子 交接文档（2026-10-23）

> 交接人：lyx。接手人按第 9 节“第一小时检查清单”走一遍即可确认环境可用。
> 本文只写"现在这台机器上真实存在的东西"，所有结论都有日志/命令出处；未完成的写在
> 第 7 节，不要当成已完成来汇报。

## 0. 一句话现状

310P 算力包（`10.42.0.178` / 机器人网段 `10.0.1.15`）上跑着**自研语音全链路**：
`原始采集 → silero VAD → X-ASR → /voice_command → ibrobot_agent（本机 27B 大模型）→ ZipVoice TTS → 厂商扬声器`，
并已验证两条技能分支：**手势**（挥手等 5 个）与**照镜子**（mirror_start/stop）。
厂商侧当前是 `only_voice`（POST 值 `run_mode=2`），系统状态是 `Develop_Audio_ROS`。

当前只有一个服务在跑：`x2-voice.service`（语音 + 手势）。语音 + 照镜子的 `x2-mirror.service`
处于 stopped（两者 catalog 互斥，见 7.1）。

## 1. 机器与网络拓扑

| 机器 | 地址 | 角色 | 备注 |
|---|---|---|---|
| 310P 算力包（本机） | `10.42.0.178` / `10.0.1.15` | 跑 IB_Robot 全栈 | openEuler Embedded aarch64，16 核，CANN 8.1.RC1，NPU `npu-smi 24.1.rc3` |
| PC1 运控 | `10.0.1.40` | `mc` / `rc` / `task_manager` | 不要在这台跑我们的栈 |
| PC2 开发单元 | `10.0.1.41` | Jetson 开发环境 | 有 `/agibot/software/aimdk` 与 457 个 msg 的 aimdk_msgs |
| PC3 交互单元 | `10.0.1.42` | `agent` / `hal_audio` / `interaction_slave` | `aima em ...` 只能在这台执行 |
| 开发 PC（Windows + WSL） | WSL `10.42.0.177` | 本地大模型 + 代理出口 | Clash 混合端口 `7897`；WSL 与 Windows 是 mirrored 网络 |

- 板子出网：**没有直连**，只能走开发机的 Clash：`http_proxy=http://10.42.0.177:7897`
  （已写进 `/etc/pip.conf`、`/etc/dnf/dnf.conf`、`/etc/profile.d/x2_proxy.sh`）。
  Clash 关了，板子的 pip/git 就断。
- 板子 DNS：`/etc/resolv-conf.systemd` 依次 `10.0.200.41`、`223.5.5.5`。
- 板子**没有 RTC**，重启后时钟回到 1970；`x2-clock.service` 每 12s 用平台消息时间戳校准
  （偏差 >100ms 会让 runtime 直接 FAULTED，见 8.1）。
- 登录凭据不在文档里，向 lyx 要。

## 2. 代码：哪些进了仓库，哪些只在板上

仓库：`https://gitcode.com/grangerxsp/IB_Robot_hz`，分支 `1028_agi_lyx`。

**已入库并推送的提交（板子和 WSL 两侧都对齐到 `87c6be40`）**

| 提交 | 内容 |
|---|---|
| `22fcd5c3` | `aimdk_robot: report audio.capture readiness for shared_audio consumers` — `runtime_node.py` 的 `OBSERVED_INTERFACES` 增加 `audio.capture`（否则 `wait_for_runtime` 永远等不到该接口就绪，启动 60s 后退出），配套改 `test_runtime_speech_readiness.py` |
| `87c6be40` | `robot_config: add the X2 voice agent on the magic mirror app` — 新增 `config/robots/aimdk_x2_voice_mirror.yaml` 与 `skill_catalog/config/profiles/aimdk_x2_mirror_voice.yaml` |

推送方法（板子没有 gitcode 凭据，必须在 WSL 侧推）：

```bash
# WSL 侧，代理必须开着
cd ~/IB_Robot_hz
https_proxy=http://127.0.0.1:7897 git push origin 1028_agi_lyx
```

**只在板子上、没有入库的改动（重要，`git reset --hard` 会全部丢掉）**

| 位置 | 改动 | 为什么不能直接入库 |
|---|---|---|
| `src/robots/aimdk/aimdk_robot/aimdk_robot/named_motion.py` | T6 固件回 `task_id 0` → 改成 `_await_preset_without_task()`，用 `player_state` 判起停完成 | 属于 XiaoqiangWu 的包，语义要和他对齐（旧 310P 上同样打的是临时补丁） |
| `src/robots/aimdk/aimdk_robot/aimdk_robot/runtime_node.py` | 新增 `player_state()` 访问器 | 同上，配套上一个补丁 |
| `src/robot_config/config/robots/aimdk_x2_voice_raw.yaml` | 新 profile：语音 + 手势，ASR 走 `shared_audio`/`audio.capture` | 需要先在 `aimdk_x2_voice_agent` 之上确认部署口径；目录名（robot name）必须沿用 `aimdk_x2_skills` |
| `src/robot_config/config/robots/aimdk_x2_voice_agent.yaml` | `planner.base_url` 改成 `http://10.42.0.177:18094/v1` | 现场地址，不入库 |
| `libs/lerobot`（子模块） | checkout `a2be6a8a`（`requires-python >=3.11`） | 分支钉的 `30da8e68` 要 Python 3.12，板子是 3.11 |
| `src/agent/ibrobot_agent/ibrobot_agent/voice_bridge.py` | 自言自语两处补丁：① 订阅厂商 `/aima/hal/audio/play_state`，PLAYING 时 `/voice_control stop`、STOPED 后 `start` + settle；② 丢掉“复述自己刚说的话”的 transcript（20s 窗口、归一化后 60% 重合） | liuweihong 的包；X2 专用（`aimdk_msgs` 不存在时自动跳过），需要与他对齐再上游 |

## 3. 板端工作区与关键路径

```
/root/workspace/lyx/
├── IB_Robot_hz/                 # 工作区（git clone，HEAD=87c6be40）
│   ├── venv/                    # 专用 venv（见 3.1）
│   ├── install/                 # merge-install（57 个包）
│   ├── models/                  # 见 4
│   └── src/robot_config/config/robots/aimdk_x2_voice_raw.yaml
├── ibrobot_env.sh               # 唯一环境入口：进工作区 + .shrc_local + aimdk SDK overlay
├── run_x2_voice_raw.sh          # 当前在用：语音 + 手势（authorize_motion:=true）
├── run_x2_mirror.sh             # 语音 + 照镜子（authorize_motion:=true）
├── run_x2_stack.sh              # 老口径：语音 + 手势但 ASR 走厂商增强流（external_vad），现环境下喂不到数据
├── clock_sync_loop.sh           # 对时循环
└── tools/
    ├── sync_platform_clock.py   # 用 /aima/mc/common/state 的 header.stamp 校时
    └── rate_probe.py            # 任意话题测频率：python3 rate_probe.py <topic> [秒]
```

### 3.1 环境事实（踩过的坑都在这）

- `source /root/workspace/lyx/ibrobot_env.sh` 是**唯一入口**，它会：进工作区 → `source .shrc_local`
  （ROS humble + venv + PYTHONPATH）→ 追加 aimdk SDK overlay
  `/root/workspace/agi_x2/drf/aimdk/install`（aarch64 + py3.11 预编译，163 个 msg）。
- venv 是 `--system-site-packages` 建的，并且有一个 `.pth` 指到 `drf` 的 venv
  （借用 torch/torch_npu/sherpa_onnx/onnxruntime，省 7GB）；缺 `lark` 就是因为忘了这个开关。
- venv 里的 `colcon`、`cmake`（3.27.9）、`ros2`（shebang 指向 venv python）都是**必须**的：
  系统的 `/usr/local/bin/{colcon,cmake}` 是坏的 python wrapper，系统的 pydantic 是 v1 会被 `ros2 launch` 先抢到。
- `empy` 必须 `3.3.4`（4.x 报 `AttributeError processing template 'msg.idl.em'`）。
- 构建必须用 `./scripts/build.sh`（它处理 `--merge-install`、跳过 openEuler 上没有的 `sim_models`/`fast_calib`），
  直接 `colcon build` 会在 `install/share/sim_models/package.sh` 上失败。
- **ssh 会话被 cgroup `/CtrlCPU` 钉在 CPU0**（`nproc` = 1，`/sys/fs/cgroup` 只读改不了）。
  构建/长任务要用 systemd 临时单元才拿得到 16 核：

  ```bash
  systemd-run --unit=lyx-build --collect bash -c 'cd /root/workspace/lyx/IB_Robot_hz && source /root/workspace/lyx/ibrobot_env.sh && ./scripts/build.sh --clean > /tmp/build.log 2>&1'
  ```
  同一个包单核 2 分 19 秒 → 16 核 12.9 秒；全量 60 个包 19 分钟。

### 3.2 systemd 单元

| 单元 | 作用 | 现状 |
|---|---|---|
| `x2-voice.service` | `/root/workspace/lyx/run_x2_voice_raw.sh`（语音 + 手势） | **active** |
| `x2-mirror.service` | `/root/workspace/lyx/run_x2_mirror.sh`（语音 + 照镜子） | stopped |
| `x2-clock.service` | `clock_sync_loop.sh`，12s 一次对时 | active |
| `lyx-build.service` | 一次性全量构建 | 已结束 |

日志：`/tmp/x2_voice_raw.log`、`/tmp/x2_mirror.log`；ROS 日志 `/tmp/roslog/`；
agent 台账 `/tmp/ibrobot-agent-x2-voice/{requests,conversation}.sqlite3`。

## 4. 模型清单（都在 `models/`，均已在板上）

| 目录 | 大小 | 用途 | 来源 |
|---|---|---|---|
| `voice_asr/sherpa-onnx-x-asr-480ms-...-2026-06-05` | 162M | X-ASR，deployment `torch_cpu`（CPU 跑） | GitHub `k2-fsa/sherpa-onnx` releases 下载后用 `python3 -m voice_asr_service.package_sherpa_asr_bundle` 打包 |
| `silero-vad` | 2.3M | VAD | `./scripts/download_speech_direction_models.sh --silero-only` |
| `zipvoice` → `zipvoice_v3` | 545M | ZipVoice TTS，deployment `ascend_310p` | schema **v3** bundle（drf 那份是 v2，`unsupported schema_version 2`），om 硬链到 `/root/workspace/drf/worktrees/test_om/models/zipvoice_upload_bundle/artifacts` |
| `yolox_x_640` | 191M | 照镜子的行人检测 | `/root/ibrobot-video-verify-7bc44687/models/yolox_x_640`（带 `assets/adapter.json`，缺它 config 校验会失败） |
| `pear_parameter_network` | 1.3G | PEAR 姿态参数 | 同上 |

Language 前端依赖（venv 内）：`cn2an`、`jieba`、`pypinyin`、`piper_phonemize`（英文；有 aarch64 wheel）。

## 5. 起停与验证

```bash
# 环境（每条新 shell 都要）
source /root/workspace/lyx/ibrobot_env.sh

# 看当前栈
systemctl is-active x2-voice x2-clock
ros2 topic echo --once /runtime_status | grep -E 'lifecycle|faults'   # 期望 ACTIVE / faults: []
ros2 topic echo --once /voice_status                                  # 期望 listening

# 只语音 + 手势（当前）
systemctl stop x2-mirror 2>/dev/null; systemd-run --unit=x2-clock --collect /root/workspace/lyx/clock_sync_loop.sh
systemd-run --unit=x2-voice --collect bash -c '/root/workspace/lyx/run_x2_voice_raw.sh > /tmp/x2_voice_raw.log 2>&1'

# 语音 + 照镜子（互斥，先停 x2-voice）
systemctl stop x2-voice
systemd-run --unit=x2-mirror --collect bash -c '/root/workspace/lyx/run_x2_mirror.sh > /tmp/x2_mirror.log 2>&1'

# 不带嘴的冒烟：直接注入文本，验证 规划 → 执行 → 播报
ros2 topic pub --once /voice_command std_msgs/String "{data: '挥挥手'}"
ros2 topic pub --once /voice_command std_msgs/String "{data: '照镜子'}"
ros2 topic pub --once /voice_command std_msgs/String "{data: '停止照镜子'}"
```

照镜子额外看两处：`ros2 topic echo --once /mirror_session/status`（`commands_sent`、
`commands_malformed`）与 `ros2 topic hz /aimdk/arm/commands`（镜像是 50Hz，
`commands_malformed` 必须是 0）。

## 6. 已验证的事实（2026-10-22/23 真机）

| 项 | 证据 |
|---|---|
| 全量构建 | 55 个包 / 0 失败，`Build complete`，install/share 57 个包 |
| ASR | 日志 `Command: 林夕挥挥手`、`Command: 可是要动`；`/voice_status = listening` |
| 自研 agent | `ibrobot_agent_node` + `agent_voice_bridge`，planner 调 `10.42.0.177:18094`（Ternary-Bonsai-2-27B，model-id `qwen3.8-27b`） |
| TTS 出声 | PC3 `hal_audio.log`：`AudioPlayer::pub_play_state ibrobot.aimdk_robot state 1 → 2`；现场听得到 |
| 手势 | “挥挥手” → `wave_hand` → 7.29s → `SUCCEEDED` → “执行完成。” |
| 照镜子 | “照镜子” → `mirror_start` → `set_session` 接管 `policy_stream` → `MIRRORING/running` → `/aimdk/arm/commands` 49.94Hz，`commands_sent 3212`、`commands_malformed 0`；“停止照镜子” → `wrap_up` → `WAIT`，厂商确认 `STAND_DEFAULT`，末条指令距复位姿态 0.00° |
| 本机大模型 | 板子直连 `http://10.42.0.177:18094/v1/models` 200；chat 返回正常 JSON |
| 厂商侧模式 | `only_voice`：agent.log `Set RunMode to OnlyVoice`，`omnis_sdk_config: "run_mode":2`，持久化在 `/agibot/data/var/agent/agent_config.yaml` |

## 7. 未完成 / 请同事继续

1. **手势与照镜子互斥**：catalog 一个 profile 只能有一种 control mode（手势 `named_motion`、镜子
   `model_inference`）。要共存得把 X2 profile 改成 hybrid：`robot_context_schema_version: 3` +
   `motion_mode.manipulation_control_mode / navigation_control_mode`，让 `supported_control_modes`
   同时包含两者。改动面比现在这两个 profile 大，建议先出设计再动。
2. **`task_id=0` 补丁上游化**：现在只在本机 `named_motion.py`。需要和 XiaoqiangWu 确认
   “固件接受但无 task” 时的语义（是否等同 fire-and-forget），再决定正式实现。
3. **ASR 自激**：TTS 播完，ASR 会把上一句（“正在执行”）再转成一条指令。`voice_bridge`
   已有 `mute_asr_during_tts`，但 settle 时间偏短；可调大或改为播报期间彻底闭麦。
4. **镜像语音关闭**：`mirror_session.voice.enabled: false`（避免和 agent TTS 抢音频焦点）。
   要听“我准备好咯”那句就打开，但要先解决焦点竞争。
5. **照镜子的 demo 钩子需要镜头里有人**：`actor_reason: no_person` 时不会开轮次（这是设计）。
6. **only_voice 的副作用**：厂商 `network_detection` 会因为“音频没上报到云”累积，可能播
   “网络状态不佳，请重新检查网络”。出现时先看 PC3 `agent.log` 的 `audio commited alarm` /
   `skip send audio data to cloud`。
7. **系统状态仍是 `Develop_Audio_ROS`**（不是 Business）。这是上一轮为了拿原声音频迁移的，
   没动。要回 Business 需要能调 `MigrateSystemState` 的机器再迁回去。

## 8. 排障对照表

| 现象 | 根因 | 处理 |
|---|---|---|
| 启动即 `runtime lifecycle is 'FAULTED'`，`faults: clock disagrees ... -0.3s` | 板子时钟漂移超 100ms | `systemd-run --unit=x2-clock --collect /root/workspace/lyx/clock_sync_loop.sh`，然后重启栈 |
| `wait_for_runtime: interface 'audio.capture' is unknown: no sample` | 老代码 `OBSERVED_INTERFACES` 只有 `audio.speech` | 已由 `22fcd5c3` 修；若是旧树，改 `runtime_node.py` 后重编 `aimdk_robot` |
| 动作报 `NAMED_MOTION_REJECTED: ... task_id 0` | T6 固件回 task_id 0 | 打第 2 节的 `named_motion.py` / `runtime_node.py` 板端补丁 |
| `Model service initialization failed: unsupported schema_version 2` | zipvoice bundle 是 v2 | 换 `models/zipvoice_v3`（schema 3） |
| `ZipVoice Chinese frontend dependency is unavailable: No module named 'cn2an'/'jieba'` | venv 缺依赖 | `venv/bin/pip install cn2an jieba pypinyin piper_phonemize` |
| `planner model call failed: VLM API HTTP 502` | 启动栈的 shell 带了 `http_proxy`，请求被 Clash 劫持 | `run_*.sh` 里 `unset http_proxy`（已内置） |
| `No module named 'lark'` | venv 不是 `--system-site-packages` | 改 `venv/pyvenv.cfg` 的 `include-system-site-packages = true` |
| `cannot import name 'model_validator'` | 系统 pydantic v1 先被 import | 用 venv 里的 `ros2`（她的 shebang 指向 venv python） |
| `colcon: ModuleNotFoundError: colcon_core` / `cmake: No module named cmake` | 用了 `/usr/local/bin/{colcon,cmake}` | 一律 `source ibrobot_env.sh`，让 venv/bin 排在前面 |
| `colcon test` 报 `install/share/sim_models/package.sh` 不存在 | openEuler 跳过了 sim_models | `--packages-ignore sim_models fast_calib`（build.sh 内部已这么做） |
| 镜像 bundle 校验失败 `assets/adapter.json 不存在` | 只拷了 `.om` | 用 `/root/ibrobot-video-verify-7bc44687/models/*` 整目录（带 assets） |
| `/agent/process_audio_output` 一条不发 | `only_voice` 静默 / 没人说话 / 厂商没在流 | 看一下 `audio_manager.cpp:739 skip send audio data to cloud`；我们的 ASR 走 `audio.capture`，不受影响 |

## 9. 接手人第一小时检查清单

1. `source /root/workspace/lyx/ibrobot_env.sh`，确认 `ros2 pkg list | wc -l` ≈ 430、`python3 -c "import aimdk_msgs, ibrobot_msgs"` 通过。
2. `systemctl is-active x2-voice x2-clock`，`/runtime_status` 是 ACTIVE 且 `faults: []`。
3. 板子 → 大模型：`curl --noproxy '*' http://10.42.0.177:18094/v1/models` 返回 200（这台 Windows 的 Clash 与 ninfer 都得开着）。
4. 不说话，先注入文本冒烟：`ros2 topic pub --once /voice_command std_msgs/String "{data: '你好'}"`，应听到 TTS 回答。
5. 现场对着机器人说“挥挥手”，应看到抬手并听到“执行完成”。
6. 要测照镜子：停 `x2-voice` → 起 `x2-mirror`，站在镜头前说“照镜子”，
   `ros2 topic hz /aimdk/arm/commands` 应到 ~50Hz；说“停止照镜子”应回到 `WAIT`。
   注意 `mirror_session.publish_enabled` 现在是 `true`（会真的动上半身），测完想让它不动就改 `false` 重编。

## 10. 回滚 / 恢复

- 停全部：`systemctl stop x2-voice x2-mirror x2-clock`（机器人回到厂商空闲状态）。
- 恢复厂商对话模式：`SetAgentPropertiesRequest` 写 `RUN_MODE=2 → 'normal'`，再 `aima em restart-app agent`（PC3）。
- 代码回到上游：`cd /root/workspace/lyx/IB_Robot_hz && git reset --hard origin/1028_agi_lyx`
  ——**会丢掉第 2 节“只在板上”的 5 项改动**，回滚前先 `git diff > /tmp/board-local.patch` 备份。
- 模型目录留着即可，不占用回滚。

## 11. 附：关键命令速查

```bash
# 测话题频率（任意话题，RELIABLE 订阅）
python3 /root/workspace/lyx/tools/rate_probe.py /aima/hal/audio/capture 10
python3 /root/workspace/lyx/tools/rate_probe.py /audio/capture_stamped 10

# 强制对时一次
python3 /root/workspace/lyx/tools/sync_platform_clock.py 20

# PC3 上的厂商命令（必须先 source）
source /home/agi/.aima/env/bashrc && aima em get-app agent
source /home/agi/.aima/env/bashrc && aima em restart-app agent

# 写 only_voice / normal（在 310P 上，DDS 直连）
ros2 service call /aimdk_5Fmsgs/srv/SetAgentPropertiesRequest aimdk_msgs/srv/SetAgentPropertiesRequest \
  "{contents: {properties: [{key: {value: 2}, value: only_voice}]}}"
```

## 12. 延迟实测与调优（2026-10-23 晚补充）

### 12.1 "慢"花在哪（说话结束 → 听到回复，合计约 5.5–6 s）

| 环节 | 实测 | 说明 |
|---|---|---|
| ASR 收尾 | ~1.5 s | `voice_asr.final_tail_padding_seconds = 1.5`；README 要求 ≥1.5，低于它“确认/招呼”这类句尾字会丢，别随便降 |
| planner | 1.0–1.2 s | 本机 27B（`10.42.0.177:18094`），含目录 prompt 的 prefill |
| ZipVoice 合成 | 1.9–2.1 s | **近似固定成本**：`"好的。"` 1.88 s（音频 0.46 s，RTF 4.09）、`"挥挥手，我正在挥手。"` 2.07 s（音频 1.95 s，RTF 1.06）→ 短回复也不便宜 |
| 播放预缓冲 | 1.0 s | 厂商 speaker 契约 `write_chunk_ms: 200 / prebuffer_ms: 1000`（`profiles/x2_ultra.yaml`），属接口声明，不是本地参数，不建议改 |
| 起播后分包 | 200 ms/包 | 只影响连续播放的粒度 |

### 12.2 本次已调

- `voice_bridge.resume_settle_sec`：`1.0 → 0.6`（写在 `aimdk_x2_voice_raw.yaml`）。
  作用：回复播完后 ASR 保持静音的窗口，缩短轮次间隔。代价：尾音回灌风险上升，
  若出现“它把自己的话当指令”，先调回 0.8–1.0。

### 12.3 想真正“连续对话”，需要改代码（本次没做）

1. **中间结果提前触发**：`/voice_partial` 命中已知技能别名时先规划执行，省掉 1.5 s 收尾。
2. **按句合成、边合成边播**：现在整段回复一次合成（1.9 s 固定成本），改成第一句先合成先播。
3. **对比 `external_vad`（厂商分段）**：厂商 VAD 的分段边界可能比 silero+1.5 s tail 更快，值得 A/B。
4. **固定ack 缓存**：把“好的/收到”这类高频短句预合成成文件直接播，省掉合成那 1.9 s。

### 12.4 复测命令

```bash
# 合成固定成本（看 inference_time_ms / total_duration_sec）
ros2 service call /voice_tts/synthesize ibrobot_msgs/srv/SynthesizeSpeech "{text: '好的。'}"
# 轮次间隔（改 resume_settle_sec 后重启生效）
ros2 param get /agent_voice_bridge resume_settle_sec
```

### 12.5 自言自语（原始流无 AEC）的处置——2026-10-23 夜

原始流没有回声消除，麦克风会把机器人自己的播报再收一遍，于是它“回答自己”。两处板端补丁（都在
`voice_bridge.py`，未入库）：

1. **按厂商播放状态闭麦**：订阅 `/aima/hal/audio/play_state`（`aimdk_msgs/msg/PlayStateChange`），
   `state=1(PLAYING)` 时向 `/voice_control` 发 `stop` 并置 `_deaf_until=inf`；`state=0/2` 时发 `start`
   并留 `resume_settle_sec`。比“按 TTS 服务返回时刻”准，因为播放服务返回只代表队列交给了平台。
2. **文本级回声过滤**：记下最近一次 `[say]` 的文本，20 秒内若 transcript 归一化后与它相同/包含/60%
   重合，直接丢弃并打印 `ignored transcript that repeats the reply`。这一层是兜底：即使 ASR 缓冲区
   里还有回声，也不会变成指令。

代价与边界：**不能打断播报**（播报期间 ASR 是 stop 的）；用户若在机器人刚说完时原样重复它的话，会被当成回声丢掉。
验证状态：代码已编译、ruff 通过、栈已重启，但在复测时机器人整机离线（见 12.6），**回声抑制尚未完成真机确认**。

### 12.6 2026-10-23 01:54 机器人离线（交接时请先确认）

- 现象：板子 `10.0.1.40/41/42` 全部 ping 不通、ARP `FAILED`/`PROBE`；`10.42.0.1` 也不通；
  板子自身 `eth0` 地址正常（`10.0.1.15 / 10.42.0.178`），所以是**机器人侧或机器人网段**掉了，
  不是板子或我们的栈。
- 连带：`/aima/mc/common/state`、`/aima/sm/system_state`、`/aima/hal/audio/capture` 全部无 publisher，
  runtime 在 `wait_for_runtime` 处 FAULTED（预期行为），`x2-voice` 起不来。
- 恢复步骤：机器人上电/网络恢复 → 确认 `ping 10.0.1.42` 通 → `ros2 topic list | grep aima | wc -l`
  能看到话题 → `systemd-run --unit=x2-voice --collect bash -c '/root/workspace/lyx/run_x2_voice_raw.sh > /tmp/x2_voice_raw.log 2>&1'`
  → 看 `/runtime_status` 是否 ACTIVE、`/voice_status` 是否 listening。

## 13. 2026-10-23 白天：手势 + 照镜子合并栈（状态与坑）

### 13.1 合并栈怎么来的

一个栈里同时跑「5 个预设手势 + mirror_start/stop + 增强流 ASR + agent + ZipVoice」，
新增两个文件 + 两个框架小补丁（都在板子工作区，**未入库**）：

| 改动 | 作用 |
|---|---|
| `robot_config/config/robots/aimdk_x2_voice_mirror_all.yaml` | 继承 `aimdk_x2_voice_mirror`，重新打开 `runtime_named_motion`、`named_motion` 控制模式、7 项 `test_allowlist`、外部 VAD 语音链；`skill_required_control_mode: model_inference`（mirror session 要求与全局技能模式相同），`skill_supported_control_modes: [named_motion, model_inference]` |
| `skill_catalog/config/profiles/aimdk_x2_all.yaml` | 5 手势 + mirror_start/stop 的目录 |
| `robot_config/loader.py` | 支持显式 `skill_supported_control_modes`（没有它就只能靠 `nav_stage: hybrid`，而 X2 没有导航服务）；digest 也带上该字段 |
| `skill_catalog/validator.py` | 放宽「supported_control_modes 必须 schema v3」，改为「必须是本 schema 合法模式且至少两个」 |
| `manipulation_execution/mirror_session/runtime.py` | 释放失败的 message 截断到 240 字（Agent 的 message 字段上限 300，超了会让请求变 UNKNOWN 并隔离机器人） |

启动：`systemd-run --unit=x2-all --collect bash -c '/root/workspace/lyx/run_x2_all.sh > /tmp/x2_all.log 2>&1'`
（`run_x2_all.sh` 用 `robot_config:=aimdk_x2_voice_mirror_all control_mode:=named_motion authorize_motion:=true`）

### 13.2 已验证

- `挥挥手` → `wave_hand` 真机执行 7.2–7.4 s → SUCCEEDED（runtime 自己从 idle/policy_stream 之间切换正常）。
- `照镜子` → `mirror_start` → `set_session success` 接管 `policy_stream` → STANDBY/approach→ready → SUCCEEDED → 有人的话进 MIRRORING。
- 增强流在**有人说话时**确实出帧（5 分钟窗口 212 帧，0.71 Hz），静默期 0 条是正常的。

### 13.3 已知问题：`mirror_stop` 释放失败 → 机器人被隔离

- 现象：`set_session stop` 报 `idle request refused 3 times ... MOTION_BUSY（当前动作正在播放，无法切换到 STAND_DEFAULT）`，随后 bridge 回「执行结果未知，请人工检查机器人状态」，后续所有技能被 `ROBOT_QUARANTINED` 拒绝。
- 根因链：① 平台在释放那一刻仍“动作播放中”，`SetMcAction STAND_DEFAULT` 被拒；② 失败 message 超过 300 字 → Agent 契约校验失败 → 请求状态 UNKNOWN；③ Agent 的 `is_robot_quarantined`（`request_store.py:201`）看到该 scope 存在 UNKNOWN 就隔离整个机器人。
- 已修（2026-10-23 下午）：
  1. **message 截断到 240 字**，失败不再因为超过 Agent 的 300 字上限变成 UNKNOWN；
  2. **MOTION_BUSY 自愈**：`release_to_idle` 遇到平台拒绝（消息含 `MOTION_BUSY`/`busy`/“正在播放”）时，
     先调 `/runtime/stop` 的 `HOLD` 取消正在播放的动作，再重发 `set_mode("idle")`（HOLD 会留下 stop latch，
     紧随其后的 idle 请求正是官方文档里清 latch 的动作）。实现三处：
     `mirror_session/runtime.py`（`_busy_refusal` + 重试分支 + `RuntimePort.stop`）、
     `mirror_session/endpoints.py`（新增 `runtime.stop` 接口与 `RELEASE_INTERFACES`）、
     `mirror_session_node.py`（`RosRuntimePort.stop` 客户端，`ibrobot_msgs/srv/StopRuntime`）。
- 回归验证：`manipulation_execution` 426 个测试只有 1 个失败
  （`test_magicmirror_plan_reaches_mirror_session_through_the_entry_point`），把本次改动 `git stash`
  后仍然失败，属该分支既有的 launch-baseline 漂移，不是本次引入。起停实测：start→MIRRORING→stop 1.02 s
  回 idle，平台确认 STAND_DEFAULT；MOTION_BUSY 分支因当时平台不忙未被触发，触发时会打印
  `stop(HOLD) accepted/refused`。
- 手工解除隔离（本轮用过，有效）：停栈 → 在 `/tmp/ibrobot-agent-x2-voice/requests.sqlite3` 里把该 `state='UNKNOWN'` 的行改成 `FAILED` 并写入一条 `reconciled` 事件 → 重启栈。
- 另一条正路：`skill_library.execution_maintenance`（`--store /root/.ros/ibrobot/executions.sqlite3 --execution-id ... --evidence ... --confirm-hardware-stopped --confirm-controller-queues-reset`），但它只管执行器 ledger，Agent 的 requests ledger 要上面那步。

### 13.4 今天还遇到的两件现场事

1. **MC 卡在 UPPERBODY_REMOTE_SPLIT**：早上 06:42 有人用厂商 `py_examples set_mc_action` 切过去没切回，`status=100(busy)` 导致所有预设动作被拒。恢复顺序：`JD` → `SD`（切回 STAND_DEFAULT），必要时 `aima em restart-app mc`。
2. **310P 板子约 50 分钟 SSH/ICMP 不可达但进程没死**（uptime 未归零、dmesg 无 OOM）。同期 load 21+，怀疑重载把网络/sshd 饿死；恢复后一切正常。重负载实验（感知三模型 + TTS 同卡）要小心。

### 13.5 “进得了状态却照不了镜子”的两个原因（2026-10-23 下午，已修）

**原因一：`mirror_session.publish_enabled` 是部署开关，默认关。**
合并 profile 最初写的是 `false`（安全默认）。此时会话一切正常——`set_session success`、接管
`policy_stream`、进入 `MIRRORING/running` 跑了 42 秒、内部算出 819 条指令——但按契约**一条都不发给
机械臂**，全部落到 `/mirror_session/arm_command_preview`。日志里这句就是判据：

```
[mirror_session]: publish_enabled=false: nothing is sent to /aimdk/arm/commands
                  or /aimdk/hand/joint_commands; commands go to ~/arm_command_preview ...
```

所以“技能执行成功、状态也进去了，但机器人不动”不是 bug，是配置。要真跟随必须
`aimdk_x2_voice_mirror_all.yaml: mirror_session.publish_enabled: true`。

**原因二：手型口径不匹配 → runtime DEGRADED → 会话把轮次当故障取消。**
平台报的手型是 `nimble_hands`（有时 `none`），而本部署用的 runtime profile
`x2_ultra_omnihand` 声明 `dexterous_joint`。于是：

1. runtime 启动时 `_verify_hand_type()` 报 “declared 'dexterous_joint' is unverified”；
2. 手部反馈 `/aima/hal/joint/hand/state` 一旦断 0.5 秒，`_check_feedback()` 判
   `joint feedback stale: hand` → runtime DEGRADED；
3. mirror session 的契约是“runtime DEGRADED 即故障”，于是轮次在 0.16 秒内被取消
   （`last_transition: reason=fault`，`detail: runtime DEGRADED: joint feedback stale: hand`）。

表现就是“刚说照镜子、它应了一声、然后什么都没发生”。修法（板端补丁，
`aimdk_robot/runtime_node.py::_check_feedback`）：手型未经平台验证时，把 `hand` 从必需反馈组里剔除——
声明之外的手型数据是加成，不是契约，它瞬断不该降级 runtime。

**修好后实测（同一会话，2026-10-23 15:0x–15:2x）**：

| 指标 | 值 |
|---|---|
| MIRRORING 轮次 | 4 次 |
| 轮次结束原因 | 2 次 `no_actor`（人走出画面，设计行为）、2 次 `mirror_stop`（口述停止） |
| `/aimdk/arm/commands` | 50.045 Hz（406 帧窗口） |
| mirror_stop 释放 | 1.02 s / 1.05 s，`platform confirmed STAND_DEFAULT` |
| 控制模式 | start 接管 `policy_stream`，stop 回 `idle` |

**排障提示**：轮次没在跑时 `/aimdk/arm/commands` 没有发布者是正常的（不是卡死）；判活看
`/mirror_session/status` 的 `state/phase`（`WAIT/none` = 空闲，`STANDBY/ready` = 等待人进入画面，
`MIRRORING/running` = 正在跟随）。轮次靠“人在头部相机画面里且距离够近”，侧身或太远会以
`no_actor` 收尾。
