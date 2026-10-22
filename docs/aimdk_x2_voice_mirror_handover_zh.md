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
