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

表现就是“刚说照镜子、它应了一声、然后什么都没发生”。

**当时的修法是板端绕行**：在 `aimdk_robot/runtime_node.py::_check_feedback` 里，手型未经平台验证时把
`hand` 从必需反馈组剔除。**这个绕行补丁已于 2026-10-23 晚删除**——上级分支从根因上修好了（hand 订阅
单独一个 callback group，不再排在 4 路 500 Hz 关节流之后被饿死），细节见第 17 节。

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

## 14. 自我介绍技能（录音回放，2026-10-23 落地）

团队 10-21 的决定：自我介绍稿不再走 ZipVoice 合成（长文本会丢结尾、分块拼接有接缝），
改为**预先录音 + 按固定音频回放**，并且要**固化成 embodied 技能**，不在语音桥里硬编码。

### 14.1 组成

| 环节 | 实现 |
|---|---|
| 技能定义 | `skill_catalog/config/skills/self_introduction/{manifest.yaml, SKILL.md, implementations/x2_runtime_v1.yaml}` |
| 执行器 | 新增 delegated executor `runtime_play_audio`（`embodied_common/skill_templates.py` 注册；`skill_executor_node.py` 的 `_execute_play_audio_skill`） |
| 绑定 | catalog 只写 `binding.clip: self_intro`；文件路径由部署 profile 的 `embodied.runtime_play_audio.clips` 解析 |
| 播放服务 | `/voice_tts/play`（`ibrobot_msgs/srv/PlayAudioFile`，路径必须是板子上存在的 wav） |
| ASR 静音 | 播放前发 `/voice_control` `stop`，播完发 `start`（`asr_control_topic` 可配） |
| 不报幕 | 语音桥新增 `silent_skills`：计划里每一步都是它时，不念“正在执行：X。”也不念“执行完成。”（失败仍报） |
| CLI 侧 | `robot_skill_cli/catalog.py` 同步注册该执行器，否则 daemon 会报 `delegated executor is not registered` |

### 14.2 换音频 / 加技能

- 换音频：直接替换板子上 `/root/workspace/lyx/IB_Robot_hz/assets/audio/self_introduction.wav`
  （单声道、当前 32 kHz / 34.5 s / 2.2 MB），重启栈即可；clip 名不用改。
- 加第二段录音：在 profile 的 `embodied.runtime_play_audio.clips` 里加 `另一名字: /绝对路径`，
  复制一份 skill 目录并把 `binding.clip` 改成该名字。

### 14.3 实测与已知

- 真机实测：`自我介绍` → 规划 → 安全放行 → 播放 **35.2–35.4 s** → SUCCEEDED；
  第三次起日志里没有任何 `[say]`（前后都不再念话）。
- **音频文件不入库**（2.2 MB 二进制），只保留路径约定；部署时从 `assets/audio/` 放置。
- 技能装在 `aimdk_x2_all` 目录（不是 `aimdk_x2_skills`）：后者没有启用 `voice_tts`，
  放进去会让纯手势部署因缺少播放服务而启动失败。

### 14.4 自我介绍同时挥手（2026-11-06）

`self_introduction` 的实现绑定增加 `binding.motion: wave`。技能名和别名不变。

- 执行器在发送前完成全部检查：音频文件、播放服务、动作能力、`wave` 名称和动作 action。
- 执行器先异步发送播放请求，然后立即发送 `wave`。两个请求都不等待对方受理。
- 播放请求发送失败时，执行器不发送 `wave`。
- 挥手进行中播放失败时，执行器取消 `wave`，并等待动作结束。
- 两个分支都成功后，技能才成功。根目标只结束一次。
- ASR 在两个分支都结束后恢复。
- 技能声明改为 `moves_robot: true`，并要求 `motion.named`。
- 播放服务没有取消接口。技能取消或超时后，录音可能继续播放，结果消息会说明这一点。
- 挥手复用 `wave_hand` 的 URS 录制轨迹，不含灵巧手动作。

### 14.5 展会配置（`aimdk_x2_summit`，2026-11-08）

`aimdk_x2_summit` 叠加在 `aimdk_x2_voice_mirror_grasp` 上，提供问候、录音问答、预设手势和水果礼物。

- 本版不启用照镜子，也不启动 YOLOX 和 PEAR。说“动作模仿”时播放 `mimic_not_ready`。
- 水果请求在同一 workflow 内执行相邻的 `pick_object → place_in_container`。
  两步使用相同目标。固定领取容器为橙色的碗，配置位于 `planner.fruit_delivery`。
- 放置保留释放后的视觉验证。只有释放、撤离、验证和清理全部成功，才播报放置成功。
  已释放但验证失败时，提示工作人员检查，不宣布可以领取。
- 单独抓取仍按原策略放回。持久 session 不会使两个独立请求获得持物交接。
- 固定问答用录音技能。抓取进度、结果和追问用 ZipVoice 实时合成。
- 录音不入库。在每个检出目录执行 `python3 commissioning/make_summit_clips.py <录音 zip>`，然后把
  `commissioning/audio/summit/` 复制到板上同一路径。
- 介绍录音支持“别说了”打断，配套手势等待安全返回。最长录音为 21.9 s。
- 抓取和放置期间，“别说了”只取消普通播报。“不玩了”停止任务并保留 session。
  停止或清理未确认时，系统继续阻止新运动。
- 现场配置要叠加到 `aimdk_x2_summit.yaml` 上。覆盖 `test_allowlist` 或 `silent_skills` 时，要保留
  该文件列出的技能。
- 独立抓取播报使用 `voice_bridge.speech_messages`。成对礼物播报使用 `GiftWorkflowSpeech`。
  容器识别描述和中文标签均来自 `planner.fruit_delivery`。
- 2026-11-08 在 310P 真机验证：说“我想要一个橘子”，机器人抓起橘子并放进橙色的碗。
- 遗留问题：在接近阶段说“不玩了”，任务取消，但恢复状态为失败并报 E06。

## 15. 镜子“卡顿”的实测结论（2026-10-23 下午）

现象：照镜子过程中机械臂会“僵住—猛地跟上”，说话时更明显。排查方法与会话内工具：
`/root/workspace/lyx/tools/mirror_monitor.py`（每 2 s 采样 preview 增量、`age_ms`、每进程 CPU、`core_busy`）。

关键数据（同一块板、同一轮次）：

| 指标 | 实测 |
|---|---|
| preview 每 2 s 发布增量 | 中位 **0**，80 个窗口里 **47 个为 0**，峰值 250 |
| `age_ms`（retarget 看到的姿态年龄） | 中位 365 ms，峰值 **1959–3121 ms** |
| retarget 自身耗时 | p50 **32.8 ms**（不是瓶颈） |
| PEAR 服务单次往返 | **110–190 ms**（`inference_time_ms` 只 32–39 ms） |
| YOLOX 服务单次往返 | **275–386 ms**（推理 185–218 ms），每秒插一次 |
| 相机 | 2688×1944 @21.7 Hz（已是“半分辨率”） |
| 把图缩到 1344×972 / 672×486 | 往返时间几乎不变 → 开销在**服务往返与排队**，不在图像大小 |
| 整机 CPU | 16 核平均 **41%**（中位 36%、峰值 77%），远未饱和 |

结论：
1. 感知链路的上限就是 **5–7 Hz 且抖动**，`mock_playback.input_staleness_sec=0.5` 会在超过 0.5 s 的
   间隔里让 retarget 直接停发，于是“停—爆发”。
2. 回退三处配置（镜子自己的 voice 关、`rpc_timeout_sec` 2.0、`demo.yaml`）**没有改善**
   零窗口比例（62% → 59%），所以不是配置问题。
3. 可行方向（未实施）：在 50 Hz 输出端对姿态做**线性插值**（5–7 Hz 关键帧 → 连续轨迹）；
   把 YOLOX 刷新从 1 s 放宽到 2 s 以减少插队；缩小相机分辨率只降低 CPU，不能改善往返。

## 16. ASR 准确率：从贪婪解码到 beam + 热词（2026-10-23 晚）

症状：短指令偶尔对、长句基本乱；`停止照镜子` 被听成 `银子/名字/瓶子/影子照镜子`，于是规划器匹配到尾巴的
`照镜子`，用户说“停止”反而开了镜子。

### 16.1 三个根因

1. **解码方式**：`_create_streaming_recognizer()` 调 `OnlineRecognizer.from_transducer()` 时没传
   `decoding_method`，sherpa 默认 greedy——int8 量化的小模型上 greedy 错误率明显高。
2. **热词是假的**：节点有 `~/set_hotwords`（`ibrobot_msgs/srv/SetHotwords`），但实现里
   `if hasattr(self._recognizer, "set_hotwords")` 对 `sherpa_onnx.OnlineRecognizer` 恒为 False，
   所以调用永远“成功但无效果”，热词从来没生效过。
3. **桥接器抢停**：`on_partial()` 用“包含停止词”判定，`停止照镜子` 的部分结果 `停止` 一出现就
   把当前请求取消（`CANCELLED_BEFORE_EXECUTION`），随后最终文本又被当作“已停过”丢弃 →
   镜子继续跑，用户听到“已停止”。

### 16.2 改了什么

| 位置 | 改动 |
|---|---|
| `asr_inference_module.py` | `decoding_method="modified_beam_search"`、`max_active_paths=4`；有热词文件时加 `hotwords_file` + `hotwords_score=1.5`；`set_hotwords()` 改为写盘 + 重建识别器（真的生效） |
| `voice_asr_node.py` | 新增 `hotwords_file` 参数并在启动日志里标出 `beam_search + hotwords` |
| `launch_builders/voice_asr.py` | 把 `voice_asr.hotwords_file` 传给节点，并用 `resolve_ros_path()` 支持 `$(find robot_config)` |
| `config/hotwords/x2_commands.txt` | 指令热词表（`停止照镜子 :5`、`停止/停下/别照了 :4`、手势与自我介绍词 :2–3） |
| `ibrobot_agent/voice_bridge.py` | ① 部分结果只在“本身是停止词、或是停止词前缀”时才抢停；② 网关因镜子占着 `policy_stream` 拒绝第二个生产者时，提示“照镜子正占着动作通道，请先说停止照镜子。” |
| 部署 profile | `voice_asr.hotwords_file` + `agent.voice_bridge.text_corrections`（把 `银子/名字/瓶子/影子/林子照镜子` 等变体映射回规范词；**注意 YAML 重复键后写覆盖先写**，本次踩过一次） |

### 16.3 实测（2026-10-23 19:00–19:20）

- `停止照镜子` → 识别正确 → 请求 → `mirror_stop` → `set_session stop: success=True … 0.99–1.02 s` → 回 idle。
- `开始照镜子` → 识别正确 → `mirror_start` → `MIRRORING/running`。
- 自我介绍录音回放：35.2–35.4 s，SUCCEEDED，前后无 TTS。
- 仍然存在的问题：自由长句依旧会错（例如 `等级太低了`、`只要不说话他就很顺畅`），那是厂商 VAD 分段 +
  480 ms 流式 int8 模型的能力上限；要彻底解决需把“分段完成后的最终识别”换成非流式 paraformer（未实施）。

### 16.4 调参备忘

- 热词格式：每行 `词 :权重`，权重 2–5；**长词权重高于短词**（`停止照镜子 :5` vs `照镜子 :2`），
  否则 beam 会偏向短词。
- `max_active_paths` 现在写死 4（模块内 `self._beam_size`）；识别变慢可降到 2。
- 错词映射是纯配置：加一行 `规范词: [变体…]` 再重编 `robot_config` 重启即可，不必改代码。

## 17. 合入上级分支 1028_agi（2026-10-23 晚，本地已合、未推送）

上游 `origin/1028_agi`（`ef0d9df8`）比我们的分叉点 `d7b6600b` 多 4 个提交（2 个真实改动 + 2 个 PR 合并），
只碰 4 个文件、+43/-9。这 4 个文件本分支也都改过，`git merge` 全部自动合并、无冲突。合并提交 `d13fb421`
在分支 `1028_agi_lyx` 上，**尚未 push**。

### 17.1 带进来什么

| 上游提交 | 内容 | 对我们的影响 |
|---|---|---|
| `9e31a250`（PR !3） | `aimdk_robot`：hand 状态订阅单独一个 `MutuallyExclusiveCallbackGroup`，配套新增测试 | 就是 13.5“原因二”的根因修法；板端绕行补丁已删（见 17.3），效果待重编重启后验证 |
| `62121c77`（PR !2） | `robot_config`：镜子 operator 的相机改由 `operator.camera_topic` 指定，magicmirror 指向后脑相机 | 我们的 profile 继承了该 operator 块，**operator 从此看后脑相机**（见 17.2） |

`mirror_session.py` 在本分支已从 `robot_config/launch_builders/` 搬到
`manipulation_execution/launch_builders/`，git 靠重命名识别把上游那段 operator 相机补丁落到了新路径。
合完核对过：`_operator_parameters` 里 `camera_topic` 优先取 `operator.camera_topic`，缺省才回落到模仿相机。

### 17.2 operator 相机的变化（继承来的，不需要改配置）

继承链是 `aimdk_x2_voice_mirror_all → aimdk_x2_voice_mirror → aimdk_x2_magicmirror`，我们两个 voice 镜子
profile 都没写 `operator` 块，所以合并后解析结果是：

```yaml
mirror_session:
  operator:
    enabled: true
    send_enabled: false
    camera_topic: /aima/hal/sensor/rgb_head_rear/rgb_image/compressed
```

即遮挡判断 / 摸头 / LED 用的是后脑相机，不再是模仿用的前中相机。**这不影响姿态流**——
`motion_imitation_stream` 的帧源没有变。要退回前中相机，在部署 profile 里覆盖一行
`mirror_session.operator.camera_topic: <原相机>` 即可。

### 17.3 板端现状

- 板子 HEAD 仍是 `f4219dd9`（合并未推送，拉不到）。
- `aimdk_robot/runtime_node.py` 已就地改成“上游修法 + 保留 `player_state()`”：与合并后的分支**逐字节一致**，
  只多一个 T6 用的 `player_state()` 访问器。原绕行补丁已删除，备份在
  `/tmp/runtime_node.py.bak-20261023-193751`。
- 改的是源码。这个工作区是 `--symlink-install`，`build/` 与 `install/` 里的 `.py` / `.yaml` 和 `src/` 是**硬链接/软链接（inode 相同）**，
  所以**重启 `x2-all` 就生效，不需要重编**——只有改 `setup.py`、入口点或新增文件才需要 `colcon build`。

### 17.4 待办

- 推送合并后，板子按第 3 节流程同步（`git checkout -- <已提交路径>` → 删同名未跟踪文件 → `merge --ff-only`）。
  届时板端 `runtime_node.py` 的本地改动会被上游版本覆盖，`player_state()` 要重新补，或推动上游接受它。
- 真机验两件事：① 长时照镜子并同时说话时，`joint feedback stale: hand` 是否彻底不再出现；
  ② operator 换后脑相机后，遮挡判断与摸头是否仍正常触发。

### 17.5 2026-10-23 晚：真机验证 + 镜子语音/物理触发打开

**已做的验证（重启后跑的是新代码）**：

| 项 | 结果 |
|---|---|
| 启动 | 105 节点，`runtime ACTIVE`，`faults: []` |
| ASR | `beam_search + hotwords` 加载完成；`/voice_status: listening` |
| 语音→LLM→TTS | 注入「你好」→ 中文回复并播报 |
| 手势 | 「挥挥手」→ `wave_hand` 7.18 s，`SUCCEEDED` |
| 照镜子 | 接管 `policy_stream`，`/mc/upper_body_command` 43.5 Hz |
| 停镜子 | `set_session stop` 0.98–1.06 s，平台确认 `STAND_DEFAULT` |
| hand 反馈 | 全日志 0 次 `joint feedback stale`、0 次 DEGRADED；hand 流实测 150 Hz |

**打开的两个开关**（`aimdk_x2_voice_mirror_all.yaml`，已随合并一起入库）：

- `mirror_session.voice.enabled: true` —— 四句语音恢复（启动中 / 我准备好咯 / 开始模仿啦 / 下次再见），实测全部播报。
  代价：每句约 2 s NPU 推理，轮次开头几秒会和姿态流抢设备。
- `mirror_session.operator.send_enabled: true` —— 物理触发真的生效（启动日志：`gestures and touches WILL call the Gateway and the session`）。
  **注意**：operator 看后脑相机，所以“摸后脑勺”= 遮挡 = `long_cover` = **停止**轮次；双遮才是启动。

**当晚发现但未修的**：个别轮次以 `MIRRORING/running -> WAIT/none (runtime_lost)` 结束——会话要求 2 s 内收到
`/runtime_status`，而 runtime 只发 1 Hz，语音预热时板子变忙，会话自己晚到就判“runtime 丢了”。外部探针实测
`max_gap 1.016 s`，runtime 本身正常。修法见 `aimdk_x2_handover_2026-10-23_zh.md` 第 6.2 节（`STATUS_MAX_AGE_S` 2.0 → 4.0）。

**接手入口**：日常操作手册见 [`aimdk_x2_handover_2026-10-23_zh.md`](aimdk_x2_handover_2026-10-23_zh.md)（启动全链路 / 停止 / 验证 / 已知问题 / 板端补丁清单）。

---

## 18. 2026-11-04：换板、追平与运维加固

本节记录一次完整的“板子掉了 → 换板 → 追平主线”的过程，以及期间确认的几条容易误判的事实。

### 18.1 拓扑与地址

| 设备 | 地址 | 说明 |
|---|---|---|
| 310P 板（当前在用） | `10.42.0.178` / `10.0.1.15` / `10.0.200.148`（同一 eth0 三个网段） | openEuler Embedded，无 RTC |
| Windows（planner 主机） | `10.42.0.100`（实验室网口） | 旧的 `10.42.0.177`、`192.168.3.100` 都已不存在 |
| 运控 PC | `10.0.1.40` | 不开 SSH |
| Jetson（感知） | `10.0.1.41` | 不开 SSH |
| 交互 PC（agent） | `10.0.1.42` | 不开 SSH |

机器人机内三台都不开 SSH，访问它们必须拿板子当跳板。**不要再让板子连 WiFi**：2026-11-04 一天之内 WiFi 地址（`192.168.3.10`）掉了三次，有线全天零中断。

### 18.2 掉线排查：三个容易误判的点

1. **“load average 14” 不是 CPU 过载**。`top` 里 CPU 98.5% idle，负载来自 14 个厂商 D 状态线程
   （`bbox_main`、`bbox_up_0`/`bbox_down_0`、`ts_proxy`、`lpm_proxy`、`hsm_proxy`、`dms_poll_task`、
   `hi_venc_*`/`hi_vpc_*`/`hi_vdec_*`/`hi_jpegd_*`）。它们长期不可中断，恒定贡献 load ≈ 14。
   **别再用 load 判断这块板是否过载**，要看 `%Cpu(s)` 的 idle 与 `ps` 的 pcpu。
2. **掉线是整机级复位，不是网络问题**。判据：`10.0.1.15` 与 `10.42.0.178` 同时消失。
   若只有一个网段消失，才怀疑网线/交换机/网段配置。
3. **该镜像的 journal 不可用**。`journalctl` 读 journald 自己写的文件会报
   `uses an unsupported feature, ignoring file`，即使建了 `/var/log/journal` 也读不出内容。
   所以排查掉线**不能依赖 journal**。

已落地的可用证据链（都开机自启）：

| 机制 | 位置 | 说明 |
|---|---|---|
| `rsyslog` | `/var/log/messages` | journald `ForwardToSyslog=yes`，文本可读，logrotate 每日轮转 |
| `x2-kmsg-tail.service` | `/var/log/kmsg-tail.log` | 每 15 s 追加 `dmesg -T`，保留最后 4000 行；断电/看门狗复位前的最后几分钟都在这里 |

下次掉线后的两条命令：

```bash
tail -200 /var/log/kmsg-tail.log     # 内核：panic / watchdog / 供电 / 网口
tail -200 /var/log/messages          # 服务层最后动作
```

**镜像自带 bug（已修）**：`/etc/logrotate.d/rsyslog` 与 `/etc/logrotate.d/logrotate.rsyslog`
重复声明 `/var/log/messages`，logrotate 报 `duplicate log entry` 后跳过整个文件 →
`messages` 永不轮转，会一直涨到写满闪存。已把后者改名为 `.disabled`，`logrotate.service` 现在 `status=0`。

### 18.3 planner 地址参数化（`d3114171`）

planner 主机地址会随“板子插在哪个网”变化（一天内变过三次：`10.42.0.177` → `192.168.3.100` → `10.42.0.100`），
把它写死在 profile 里意味着每次换网都要改文件 + 重编。现在改为启动时传参：

```bash
# embodied_pipeline.launch.py 新增参数
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=aimdk_x2_voice_mirror_all \
  planner_base_url:=http://10.42.0.100:18094/v1
```

- 覆盖逻辑在 `embodied_bringup/launch_builders/embodied.py` 的 `apply_planner_base_url()`：
  **空值保持 profile 原值**；没有 `agent.planner` 的部署完全不碰。
- 现场启动脚本 `/root/workspace/lyx/run_x2_all.sh` 里有
  `PLANNER_BASE_URL=${PLANNER_BASE_URL:-http://10.42.0.100:18094/v1}`，换机器只改这一行。
- 验证：`ros2 param get /ibrobot_agent_node planner_config_json` 里应出现该地址。
- **不要用环境变量直接写 URL**：仓库的 `$(env VAR)` 替换只作用于路径类字段，且变量未设置时直接
  `ValueError` 让配置加载失败。

### 18.4 板端补丁归属（追平后的现状）

| 文件 | 状态 | 处置 |
|---|---|---|
| `robot_config/robot_config/loader.py` | **已上游**（`a9af2c07`，含 8 个单测） | 板端不再有本地改动 |
| `aimdk_robot/named_motion.py` + `runtime_node.py` | 仍留板端（T6 `task_id=0` 适配） | 待与 XiaoqiangWu 协调，材料：`/root/workspace/lyx/aimdk_robot_t6_taskid_workaround.{md,patch}` |
| `libs/lerobot` 指针 | 留板端 | 环境适配，按仓库规则不得普通提交 |
| 两个 profile 的 `base_url` | 留板端 | 站点值；主栈走 launch 参数 |
| `assets/audio/self_introduction.wav`、`demo_relaxed.yaml`、`aimdk_x2_voice_raw.yaml` | 未跟踪 | 板端资源与实验 profile |

### 18.5 追平操作与四个坑

```bash
cd /root/workspace/lyx/IB_Robot_hz
git stash push -u -m "board-local before fast-forward"
git -c http.proxy=http://10.42.0.100:7897 fetch origin
git merge --ff-only origin/1028_agi_lyx
git apply /root/workspace/lyx/aimdk_robot_t6_taskid_workaround.patch
# 重放 profile 的 planner 默认值 + 站点地址，恢复未跟踪资源
./scripts/build.sh -- --packages-ignore feetech_sdk so101_sdk
```

1. **`git config http.proxy` 会盖掉环境变量**。板子上它长期指向旧 IP（`10.42.0.177:7897`），
   `git fetch` 因此一直失败，报错却像是“网络不通”。要么改掉配置，要么用 `git -c http.proxy=...`。
2. **`install/share` 是拷贝，不是硬链**。改 `src/` 里的 `.yaml` 必须重编对应包才生效
   （`colcon build --packages-select robot_config`）；只有 Python 入口是符号链接。
3. **`colcon test` 需要 `--merge-install`**：这块板的 `install/` 是 merged layout，
   不带该参数会报 `The install directory 'install' was created with the layout 'merged'`。
4. **`colcon test` 会因缺 `install/share/sim_models/package.sh` 失败**：
   先 `colcon build --merge-install --packages-select sim_models`，否则测试在收集阶段就失败，
   看起来像“被测包的测试挂了”。

### 18.6 本次结束时的状态

- 板端 `HEAD = a9af2c07`，与主线一致；本地剩余改动仅 18.4 表中那几项。
- 全量构建 54 个包、0 失败（跳过 `feetech_sdk`/`so101_sdk`，SO-101 与 X2 无关且需要联网 clone）。
- 重启后验收：runtime `ACTIVE`、“挥挥手” `SUCCEEDED`、ASR `streaming + beam_search + hotwords`、
  109 个节点、0 traceback、`planner_config_json` 指向 `10.42.0.100`。
- 备份：`/root/workspace/lyx/backup-20261104/`（`pre-rebase.diff`、`untracked.tgz`、`status.txt`、`HEAD.txt`）
  以及 `git stash@{0}`。
