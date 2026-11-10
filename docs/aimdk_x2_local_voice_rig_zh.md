# X2 本机语音交互验证环境（只模拟机器人动作）

本文说明 `tools/x2_local_voice/` 的实现逻辑和用法。这套环境在 x86 工作站上运行展会配置
`aimdk_x2_summit` 的语音交互。本机麦克风和耳机代替 X2 的麦克风和扬声器。机器人动作由厂商 Mock
执行，抓取和放置的感知与运动由假后端执行。其余节点和展会部署相同。

任何 worktree 都可以使用这套环境。脚本从自己的位置推导检出目录，不写死路径。

## 1. 哪些是真实的，哪些被替换

| 部分 | 本机环境 | 说明 |
|------|----------|------|
| Agent、planner、语音桥、Gateway | 真实 | planner 访问配置中的地址（展会为 `10.42.0.233`） |
| ASR、声纹、意图识别 | 真实 | 模型与展会相同，在 CPU 上运行 |
| TTS | 真实 | ZipVoice 使用 `ubuntu_onnx` 部署（310P 使用 OM） |
| 录音片段 | 真实 | 来自 `commissioning/audio/summit/` |
| X2 runtime、data plane | 真实 | 以 `use_sim:=true` 启动 |
| X2 平台（HAL、运动控制） | 替换 | 仓库自带的 `aimdk_vendor_mock` |
| X2 麦克风与扬声器 | 替换 | `local_audio_hal.py`，使用本机声卡 |
| 抓取、放置的感知与运动 | 替换 | `mock_manipulation.py`；流水线、托管和 Gateway 契约仍为真实 |
| 固定姿态库、`free_pose`、`return_to_task_pose` | 启用 | 执行器使用本机编译的姿态库和模拟运动接口 |
| 照镜子、视觉互动、握手 | 关闭 | 本机缺少视觉输入，握手默认禁用 |

ROS 只在本机回环上通信（`ROS_LOCALHOST_ONLY=1`），默认域号为 91。局域网中的机器人使用域 0，
本机环境不会向机器人发送话题。

## 2. 数据流

### 2.1 采音

1. `parecord` 从本机麦克风读取 16 kHz 单声道 PCM。
2. `local_audio_hal.py` 用项目的 Silero VAD 模型和 `VADModule` 状态机切分语音段。
3. 语音段开始时，系统发布 BEGIN 块。BEGIN 块包含 650 ms 的预录音频，与厂商的回填行为相同。
4. 语音段进行中，系统每 96 ms 发布一个 PROCESSING 块。静音 0.6 s 后，系统发布 END 块。
5. 这些块的类型是 `aimdk_msgs/ProcessedAudioOutput`，话题是 `/agent/process_audio_output`。
   真机上，厂商在同一话题上发布同样的消息。
6. X2 runtime 把这些块转成 `/audio/speech_chunk`。ASR 和声纹节点从这里读取，后面的链路与真机相同。

Silero 会话只使用 1 个线程，并关闭空转等待。项目的 manifest 会话默认占用全部 CPU 核。

### 2.2 放音

1. TTS 合成的语音和录音片段经 `/audio/play` 进入 X2 runtime。
2. runtime 向厂商 Mock 申请音频焦点。
3. runtime 把音频发到 `/aima/hal/audio/playback`。
4. `local_audio_hal.py` 订阅这个话题，用 `pacat` 播放到本机扬声器。
5. 如果 `pkg_name` 变化，系统丢弃缓冲中的音频。厂商 `hal_audio` 的行为相同。

### 2.3 抓取与放置

`mock_manipulation.py` 提供 `grasp_backends` 和 `place_backends` 两个后端工厂。
真实的 `grasp_pipeline_node`、`place_pipeline_node` 和物体托管调用这两个工厂。

- 假后端复用 `src/manipulation_execution/test/` 中的单元测试替身，因此遵守与流水线测试相同的契约。
- 每个步骤都有延时：抓取约 20 s，放置约 12 s。系统因此有时间播报进度。
- 两个假执行器运行在不同进程中，真机上只有一条手臂。交接姿态设为放置执行器的起点，
  使放置开始时的持握姿态检查通过。
- 托管交给放置之后，抓取执行器报告手臂已回到起点。真机上，放置结束时手臂回到起点。
- `mock_model_bundle/` 提供模型身份和模拟手部预设，不含权重。
  Gateway 与抓取执行器使用模型身份。姿态执行器使用手部预设检查模拟双手。

## 3. 配置如何生成

`start.sh` 调用 `make_config.py` 生成两个文件，`stop.sh` 删除这两个文件：

- `src/robot_config/config/robots/aimdk_x2_summit_local_voice.yaml`：机器人配置。
- `src/agent/skill_catalog/config/profiles/aimdk_x2_summit_local_voice.yaml`：技能目录。

`make_config.py` 先加载 `aimdk_x2_summit.yaml` 及其整条继承链，再改动 `deviate()` 中列出的项，
最后写出一个不带 `base_config` 的独立文件。使用独立文件有 2 个原因：

- 展会配置以后的改动会自动进入本机环境。
- 叠加层只能增加或覆盖，不能删除技能、直达命令或整段配置。独立文件可以删除这些内容。

`deviate()` 中的改动全部来自本机缺少的条件：

| 改动 | 原因 |
|------|------|
| 关闭照镜子、动作模仿流和视觉互动 | 没有头部相机和 Ascend YOLOX/PEAR |
| 控制模式改为 `moveit_planning` | `model_inference` 只用于照镜子 |
| 清空 `perception_services` | Grounding DINO、SAM2、GraspGen 都是 Ascend 部署 |
| 保留 `task_poses` 并启用固定姿态执行器 | 本机验证自由摆姿和归位流程 |
| 清空本机的 `blocked_task_poses` | 模拟验证不使用真机的待标定禁用列表 |
| 抓取、放置改用假后端和身份包 | 没有相机和手臂 |
| 删除 `runtime_named_motion.hand_preparation` | 预设手势直接使用厂商 Mock，固定姿态保留手部检查 |
| ASR、声纹、TTS、意图模型指向本机模型目录 | 模型不在检出目录中 |
| TTS 改用 `ubuntu_onnx` | 本机没有 NPU |
| 账本、Gateway 记录、意图日志写入本机状态目录 | 与其他本机测试隔离 |
| 删除技能 `mirror_start`、`mirror_stop`、`handshake`，并删除指向这些技能的直达命令 | 视觉输入缺失，握手默认禁用 |
| `tts_keep_english` 取决于 `piper_phonemize` 能否导入 | 缺少英文前端时，含英文的句子会整句合成失败 |

生成文件必须放在源码树中，原因是 2 个：

- 技能目录只按名称在 `profiles/` 中查找。
- 机器人配置中的相对路径按配置文件所在目录解析。

两个生成文件都不提交。如果脚本异常退出，执行 `stop.sh` 删除这两个文件。

启动脚本还会检查固定姿态库与当前动作轨迹是否匹配。
如果编译产物缺失或失效，系统使用已有种子重新编译。
编译产物位于 `<状态目录>/fixed_poses.json`。
这一步需要已下载的 X2 MuJoCo 模型。

语音桥读取展会配置中的自定义文案、录音匹配和读音替换。
语音桥还会暂存播报期间的追问，暂存时长为 10 秒。

## 4. 前置条件

1. 在检出目录中完成构建，`voice_asr_service` 和 `aimdk_robot` 都要构建：

   ```bash
   source .shrc_local && source ~/aimdk/overlay_x86/install/local_setup.bash && \
     ./scripts/build.sh -- --packages-up-to embodied_bringup voice_asr_service voice_tts_service aimdk_robot pose_composer \
     --packages-ignore livox_ros_driver2 fast_lio usb_cam fast_calib pymoveit2
   ```

2. 安装 AimDK x86 overlay。默认路径为 `~/aimdk/overlay_x86/install/local_setup.bash`。
   如果 overlay 在其他位置，设置 `AIMDK_OVERLAY`。
3. 按下面的布局准备模型目录。默认目录为 `<检出目录>/models`，也可以用 `X2_LOCAL_VOICE_MODELS` 指定。
   目录中的条目可以是符号链接。

   ```text
   <models>/voice_asr/sherpa-onnx-x-asr-480ms-streaming-zipformer-transducer-zh-en-int8-2026-06-05/
   <models>/zipvoice/                      # 含 ubuntu_onnx 部署
   <models>/campplus-speaker/              # 含 torch_cpu 部署
   <models>/addressee/addressee_zh_v2.json
   <models>/silero-vad/assets/silero_vad.onnx
   ```

4. 确认 PulseAudio 会话可用：执行 `pactl info`。
5. 准备 planner 密钥。设置 `X2_PLANNER_API_KEY`，或者设置 `X2_PLANNER_API_KEY_FILE` 指向密钥文件。
6. （可选）要播报英文单词，在 TTS 使用的 venv 中安装 `piper_phonemize==1.1.0`。

## 5. 使用步骤

1. 启动环境：

   ```bash
   tools/x2_local_voice/start.sh
   ```

   如果要指定声卡，加 `--source <麦克风> --sink <耳机>`。用 `LC_ALL=C pactl list short sources` 和
   `LC_ALL=C pactl list short sinks` 查看名称。加 `--show-level` 时，系统每 2 s 记录一次麦克风峰值。
2. 在另一个终端中观察交互：

   ```bash
   tools/x2_local_voice/watch.sh
   ```

3. 对麦克风说“欧拉欧拉”加请求，例如“欧拉欧拉，我想要一个橘子”。
4. 停止环境：

   ```bash
   tools/x2_local_voice/stop.sh
   ```

如果要模拟抓取或放置失败，在 `start.sh` 前设置 `LOCAL_VOICE_PICK=empty` 或 `LOCAL_VOICE_PLACE=unseen`。

## 6. 不用麦克风的测试方法

| 工具 | 用途 |
|------|------|
| `say.sh "文本"` | 把文本当作 ASR 结果送入语音桥，并等待终态。文本没有声纹，声纹检查放行这类语句 |
| `tools/synth_wav.py 文本 out.wav` | 用运行中的 TTS 合成一个测试说话人的语音 |
| `tools/inject_wav.py a.wav b.wav` | 把 WAV 按厂商语音段格式注入，经过 ASR、声纹和意图识别 |
| `tools/gate_probe.py 配置 模型 句子…` | 不启动 ROS，用语音桥相同的规则给句子打分 |

Python 工具通过 `rig_env.sh` 运行，例如：

```bash
tools/x2_local_voice/rig_env.sh python3 tools/x2_local_voice/tools/inject_wav.py owner.wav
```

用 `inject_wav.py` 测试声纹时，用 `synth_wav.py` 合成“主人”的语音，用另一个人的录音当作旁人。
注入期间不要对着麦克风说话。两个来源的语音段交叠时，ASR 会丢弃其中一段。

## 7. 环境变量

| 变量 | 默认值 | 作用 |
|------|--------|------|
| `X2_LOCAL_VOICE_MODELS` | `<检出目录>/models` | 模型目录 |
| `X2_LOCAL_VOICE_STATE` | `<检出目录>/log/x2_local_voice` | 日志、账本和进程组文件 |
| `X2_LOCAL_VOICE_DOMAIN` | `91` | ROS 域号，只用于本机回环 |
| `AIMDK_OVERLAY` | `~/aimdk/overlay_x86/install/local_setup.bash` | AimDK overlay |
| `X2_PLANNER_API_KEY` / `X2_PLANNER_API_KEY_FILE` | 无 | planner 密钥 |
| `LOCAL_VOICE_PICK` / `LOCAL_VOICE_PLACE` | `success` | 假抓取、假放置的结果 |

`rig_env.sh` 用 `env -i` 建立干净环境，然后依次加载 `.shrc_local`、AimDK overlay 和本检出目录的
`install`。系统保留 `XDG_RUNTIME_DIR` 和 `DBUS_SESSION_BUS_ADDRESS`，否则 `pacat` 无法连接
PulseAudio。

## 8. 日志与状态

- `<状态目录>/logs/pipeline_latest.log`：流水线日志。
- `<状态目录>/logs/audio_hal_latest.log`：音频桥日志。
- `<状态目录>/agent/`：本次运行的 Agent 账本、Gateway 记录和意图日志。
- 每次启动时，系统把上一次的 `agent/` 移到 `logs/agent_state_before_<时间>/`。
- `<状态目录>/gift_state.sqlite3`：Agent 管理的领取状态。该文件保留，不随 `agent/` 归档。

有效唤醒“欧拉欧拉”开始新的领取轮次。聊天会话和聊天上下文保留。
只说唤醒词，或连说“欧拉欧拉，给我一个苹果”，都可以开始新轮次。
同一轮内送出礼物后，再次索要会收到重新唤醒的提示。
执行中、持物或执行结果待确认时，唤醒不会清除现有任务和保护状态。
状态库中仍在执行的记录，在重启后转为待确认，不会自动恢复领取资格。

同一次运行中，如果请求结果为 UNKNOWN，机器人进入隔离（`ROBOT_QUARANTINED`），与真机相同。
重启本机环境后，新的请求账本解除该隔离。领取状态库中的待确认记录继续保留。

## 9. 已知限制

- 本机 CPU 合成一句需要 4 s 至 13 s。进度播报在合成完成前过期，系统丢弃这些播报。
  310P 使用 NPU 合成，不受这个限制。
- 本机 CPU 上 ASR 的准确率和判停时延与 310P 不同。
- 本机没有回声消除，回采声道填零。语音桥在播报期间让 ASR 静音。建议使用耳机。
- 厂商 Mock 立即执行动作指令。动作的效果和时长只能在真机上验证。
- 讲解期间说“停”或“别动”，手势沿原路径退回约 10 s。Agent 只等约 5 s 的取消确认，
  请求记为 UNKNOWN，机器人进入隔离。真机上也有这个问题，见交接文档 14.6。

## 10. 排障

| 现象 | 处理 |
|------|------|
| `missing model file` | 按第 4 节布局准备模型目录，或设置 `X2_LOCAL_VOICE_MODELS` |
| `already running` | 执行 `stop.sh` |
| 所有请求回“执行结果未知” | 本次运行中出现过 UNKNOWN。执行 `stop.sh` 和 `start.sh` |
| 语音桥启动失败，报意图模型 | 确认 `<models>/addressee/addressee_zh_v2.json` 存在 |
| 耳机没有声音 | 用 `--sink` 指定设备，查看 `audio_hal_latest.log` 中的 `playback start` |
| `say.sh` 无响应 | 查看 `pipeline_latest.log` 中语音桥是否 `ready`，以及会话窗口是否已过期 |
| 注入的语音没有识别结果 | 确认注入期间麦克风没有语音段；`inject_wav.py` 会等待订阅端匹配后再发送 |
