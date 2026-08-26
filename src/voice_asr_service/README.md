# VoiceASRNode 节点说明

`voice_asr_node.py` 是 `voice_asr_service` 包中的运行时 ROS 2 语音识别节点。
它负责把麦克风音频或音频文件转换成文本，并统一管理音频采集、VAD、sherpa-onnx 模型加载、ROS 接口以及内部状态机。

这份 README 同时说明两个独立节点：本文第 1～14 节描述 `VoiceASRNode`；下方“Speech Direction 独立节点”一节描述 `SpeechDirectionNode`。两者当前没有共享音频采集链路。

## Speech Direction 独立节点

`speech_direction_node` 属于 `voice_asr_service`，与上文所述的 `VoiceASRNode` 是两个独立节点。它负责多通道人声音频增强、语音门控和方向估计，不执行 ASR，也不控制底盘。

麦克风、阵列、输入源、模型和运行时效参数统一由以下包内配置管理：

```text
src/voice_asr_service/config/speech_direction.yaml
```

可独立启动方向节点：

```bash
ros2 launch voice_asr_service speech_direction.launch.py
```

该 launch 默认读取上述 YAML；也可通过 `config_file` 指向同结构的配置文件。模型相对路径在 launch 边界相对 `models_root`（默认 `<标准工作区>/models`）解析，绝对路径保持原样。自定义 colcon install base 时应显式指定模型根，例如：

```bash
ros2 launch voice_asr_service speech_direction.launch.py models_root:=/path/to/models
```

YAML 的模型相对路径以 models 目录为根，不再包含额外的 `models/` 前缀。`speech_direction` 不属于 `robot_config` 的机器人级配置，也不由 `sound_follow` 管理。

### 平台 profile

公共算法、阵列、门控、时序和诊断参数只维护在 `speech_direction.yaml`，平台差异由独立的 `speech_direction_profiles.yaml` 覆盖，launch 边界通过 `profile` 参数选择：

| profile | 默认 | 适用平台 | 后端组合 |
| --- | --- | --- | --- |
| `ascend_310p` | 是 | Atlas 310P 推理盒 | Silero `raw_acl` + FullSubNet `stateful_raw_acl` |
| `ubuntu_cuda` | 否 | x86 + NVIDIA CUDA | Silero `onnx` + FullSubNet `stateful_torch_cuda` |
| `custom` | 否 | 自定义 | 用户自行填写，仍受半组合校验约束 |

```bash
ros2 launch voice_asr_service speech_direction.launch.py profile:=ubuntu_cuda
```

profile 只允许覆盖 `silero_vad_backend`、`silero_vad_model_path`、`fullsubnet_backend`、`fullsubnet_device` 四个平台字段；写入公共算法字段会被 launch 拒绝。`ascend_310p` 与 `ubuntu_cuda` 还会被校验后端组合一致性，避免后端与模型路径错配的半切换配置。仅 `raw_acl` 相关后端会在节点环境注入 CANN 库路径，`ubuntu_cuda` 不注入无关环境。

### 模型资产下载

执行 `python3 scripts/verify_speech_direction_assets.py` 时，脚本走两条校验：先用 `load_inference_manifest_metadata` 标准入口校验 `config/inference_manifest.json` 的 bundle 结构、deployment bindings/execution 与 semantic_identity（不校验文件存在），再读 `config/assets/adapter.json` 逐资产校验 310P 资产的 `algorithm_contract`（family、norm_type、time_steps、input_samples 等）、文件大小与 SHA-256。标准 manifest schema 不收留 sha256/size/algorithm_contract，这些字段由 `assets/adapter.json` 承载，仍由 Python 的 `STATEFUL_FULLSUBNET_CONTRACT` 做 SSOT 校验。脚本只校验不下载，资产不在本仓库管理，需从 NAS 手动获取后放入 `models/` 对应路径；缺失的资产会打印来源提示并跳过，已存在的资产校验不通过则报错。FullSubNet 两平台共用同一 cumulative 218epochs checkpoint 权重：310P 预导出为 stateful FB/SB 拆分 OM，Ubuntu 由 Torch 直接加载同一 checkpoint。

### 配置所有权

| 类别 | YAML 参数 | 当前配置 / 含义 |
| --- | --- | --- |
| 麦克风与音频 | `device_name_contains` | `ReSpeaker`，实时设备名匹配条件 |
| 麦克风与音频 | `sample_rate` | `16000` Hz；当前 speech-direction 完整算法链仅支持此采样率，其他值会在参数校验阶段被拒绝 |
| 麦克风与音频 | `channel_indices` | `[1, 2, 3, 4]`，参与处理的输入通道 |
| 麦克风与音频 | `arecord_device` | `hw:0,0`，arecord 直采的 ALSA 硬件设备；310P/Ubuntu 均用 arecord 子进程直采，不再依赖 PyAudio，系统需安装 `alsa-utils` |
| 阵列 | `mount_yaw_deg` | `0.0`，阵列安装偏角（度），逆时针为正。把阵列坐标系角度对齐到小车坐标系，详见下方[坐标系与安装偏角](#坐标系与安装偏角) |
| 阵列 | `angle_step_degree` | `5`，SRP-PHAT 扫描角度步长（度），DOA 输出只能为该步长的整数倍；必须为 360 的正整数约数。详见下方[SRP 角度精度](#srp-角度精度) |
| 阵列 | `mic_positions` | 四麦二维坐标的一维展开数组，长度必须为通道数的两倍 |
| 输入源 | `input_source` | `device`；也支持 `wav` |
| 输入源 | `wav_path` | WAV 输入路径；`input_source=wav` 时必填。离线输入与实时 ReSpeaker 原始流使用同一入口契约，文件必须是 `16 kHz / 6 通道` WAV，并保留设备输入的 ch0～ch5；pipeline 随后按 `channel_indices=[1,2,3,4]` 选取四个麦克风通道。文中的“4 通道增强/DOA”是内部算法通道数，不表示接受 4 通道 WAV |
| 输入源 | `wav_replay_rate` | `1.0`，WAV 回放倍率，必须大于 0。有效音频播放到 EOF 后，回放器会根据增强窗口尾部、hop 大小和段末静音门限追加若干个内部 6 通道零帧，使现有 VAD/状态机自然结算最后一个语音段；无需在测试文件末尾手工添加静音 |
| 模型 | `silero_vad_model_path` | Silero VAD 模型路径 |
| 模型 | `silero_vad_backend` | `raw_acl`（310P OM）；Ubuntu profile 覆盖为 `onnx` |
| 模型 | `fullsubnet_ckpt` | FullSubNet cumulative 218epochs checkpoint，两平台共用同一权重 |
| 模型 | `fullsubnet_backend` | `stateful_raw_acl`（310P 拆分 OM）；Ubuntu profile 覆盖为 `stateful_torch_cuda` |
| 模型 | `fullsubnet_device` | `cuda`；Ubuntu Torch 后端固定 CUDA，禁止静默回退 CPU |
| 模型 | `fullsubnet_device_id` | `0`；310P ACL 设备 ID，多卡场景下指定目标卡 |
| 模型 | `fullsubnet_acl_config_path` | `""`；ACL 配置文件路径，空表示用默认配置 |
| 模型 | `fullsubnet_stateful_fb_om_path` | stateful raw_acl 的 FullBand 拆分 OM 路径，相对 `models/` |
| 模型 | `fullsubnet_stateful_sb_om_path` | stateful raw_acl 的 SubBand 拆分 OM 路径，相对 `models/` |
| 模型 | `fullsubnet_stateful_manifest_path` | stateful cumulative manifest 路径（声明 norm/checkpoint 契约），相对 `models/` |
| 模型 | `fullsubnet_om_path` | legacy 单体 OM 回退路径，相对 `models/`；仅 legacy backend 使用 |
| 运行时效 | `speech_direction_max_age_ms` | `1300` ms，方向结果最大保鲜时间 |

这些参数均由 `voice_asr_service` 独占管理。`sound_follow` 只维护底盘最小集和跟随行为参数，其完整 launch 通过无参数 include 复用本包的 `speech_direction.launch.py`，不读取或转发任何音频参数。


### 高通量离线维测

`speech_direction.yaml` 还包含以下 9 个高通量维测参数：

| YAML 参数 | 当前值 | 说明 |
| --- | --- | --- |
| `diagnostics_high_throughput_enabled` | `false` | 总开关；默认关闭，现场诊断/测试时显式开启 |
| `diagnostics_rollover_seconds` | `300` | raw、enhanced 和 metrics 的固定分卷时长（秒） |
| `diagnostics_save_raw6ch` | `true` | 总开关开启时保存完整原始 6 通道 PCM WAV；总开关关闭时不写盘 |
| `diagnostics_save_enh4ch` | `true` | 总开关开启时保存 FullSubNet 输出的 4 通道 PCM WAV |
| `diagnostics_save_frame_metrics` | `true` | 总开关开启时保存逐 hop 的 VAD、RMS、DOA 和耗时 JSONL |
| `diagnostics_save_gray_events` | `true` | 总开关开启时保存灰区事件摘要；运行时不截取灰区 WAV |
| `diagnostics_queue_size` | `128` | 后台 writer 有界队列容量 |
| `diagnostics_drop_when_full` | `true` | 接口兼容项；实时链路始终非阻塞，队列满时丢弃并计数 |
| `fullsubnet_timing_enabled` | `false` | FullSubNet STFT/FB/SB/postprocess 分阶段计时；仅性能分析时开启，生产默认关闭 |

默认关闭高通量维测（总开关 `false`），避免部署即写盘到磁盘满。需要现场诊断时，
将 `diagnostics_high_throughput_enabled` 改为 `true`；四个 `save_*` 子开关在总开关
开启时表示默认保存哪些流，可按定位需求独立关闭。启用后，每次启动会创建独立会话：

```text
~/.ros/speech_direction/runs/run_<YYYYmmdd-HHMMSS>/
├── manifest.json
├── audio/full/raw6ch_*.wav
├── audio/full/enh4ch_*.wav
├── metrics/frames_*.jsonl
└── events/gray_events.jsonl
```

节点停止后 `manifest.json` 才进入 `completed` 终态。writer 写盘失败只会永久停用本次
维测旁路，方向 pipeline 继续运行，并在基础 `/diagnostics` 中报告 `WARN`；不会在线生成
报告，也不会加载 Plotly 或任何绘图 fallback。

离线报告依赖 Plotly 与 Matplotlib，仅 `speech_direction_report` CLI 需要，`speech_direction_node`
运行时不加载。首次使用前通过可选开关安装（不会污染板端/CI 的 base 安装链）：

```bash
./scripts/setup.sh --with-diagnostics      # 新装工作区时一并带上
# 或:
python3 -m pip install -r requirements/diagnostics.txt
```

停止节点后可离线生成报告：

```bash
source .shrc_local && speech_direction_report \
  ~/.ros/speech_direction/runs/run_<YYYYmmdd-HHMMSS>
```

默认严格同时生成 `reports/doa_curves.html` 和 `reports/doa_curves.png`，不会生成灰区音频。
需要导出灰区时显式增加 `--extract-gray-audio`；事件跨分卷时会按统一 sample 轴拼接
`raw6ch` 和 `enh4ch`：

```bash
source .shrc_local && speech_direction_report \
  ~/.ros/speech_direction/runs/run_<YYYYmmdd-HHMMSS> \
  --extract-gray-audio
```

输出已存在时命令默认拒绝覆盖；确认重跑可加 `--overwrite`。HTML 与 PNG 是严格双依赖：
必须同时安装 Plotly 和 Matplotlib，缺任一项即失败，不提供降级或 fallback。同一会话不要
并发生成报告。输出提交边界只是先生成临时 HTML/PNG，再按 HTML、PNG 顺序分别替换；
它不是事务锁或 journal，第二次替换失败时可能留下半套新输出，请检查后带
`--overwrite` 顺序重跑。

### 链路分叉：stateful 生产链路与 legacy 显式对照链路

本 PR 同时保留两套门控 + 两套 pipeline，`node.py` 按 `fullsubnet_backend.startswith("stateful_")` 选择哪一套，二者不静默回退：

| 链路 | 后端 | 门控 | pipeline | 时序参数 |
| --- | --- | --- | --- | --- |
| **生产链路** | `stateful_raw_acl` / `stateful_torch_*` | `TemporalSpeechGate`（状态机帧级） | `StreamingSpeechDirectionPipeline` | tick=256、model_batch=512、SRP frame=4096/hop=512 |
| 显式对照 | `legacy_om` / `legacy_torch` | `SpeechGate`（Top-2 hop 级） | `SpeechDirectionPipeline` | hop=2048、enh_block=8192 |

- 生产部署用 `stateful_*`，走 `StreamingSpeechDirectionPipeline` + `TemporalSpeechGate`；`legacy_*` 仅作显式对照保留，启动时按后端名严格选择，**不会从 stateful 静默回退到 legacy**，后端与路径错配会在 `_build_and_start` 抛 `ValueError`。
- 两套 pipeline 都消费同一组 `VadState` / `DoaState`，但调用节奏不同：streaming 每 256 样本 tick 一次 `vad_state.update`（两个 tick 合并为一次 T=2 模型推理），legacy 每 2048 样本 hop 一次。两者都向 `DoaState.update` 写段级 DOA，`meta.type` 取值相同（`mid_long_seg` / `seg_end`），但 `mid_long_seg` 的触发条件不同——streaming 按 `max_accum_samples`（样本计数），legacy 按 `max_accum_dur_s`（墙钟时长）。`node._poll_and_publish` 按 `result["type"]` 区分段末与中间方向，上游消费者无需感知链路分叉。
- `SpeechGate` 的 Top-2 选择对 hop_size 敏感（旧值 2048 vs 新值 512 会改变 Top-2 候选），故 legacy 链路固定用 2048 hop，不沿用 stateful 的 512；两套链路的时序参数各自独立，不共享。

### 发布契约、坐标与故障语义

- 方向发布到 `/voice/speech_direction`，消息类型为 `ibrobot_msgs/msg/SpeechDirection`，QoS 为 `RELIABLE + KEEP_LAST(1)`。
- `header.frame_id` 为 `base_link`；`azimuth_rad` 遵循 REP-103：`0` 为前、`+π/2` 为左、`-π/2` 为右，左转为正。
- `header.stamp` 按方向类型分流构造，承载方向的"真年龄"信息，而非固定取发布时刻：
  - 段末方向（`seg_end`）：`stamp = 发布时刻的 ROS 时钟 − age`，还原段结束时刻，使消费者按 `now − stamp` 判过期时得到真实年龄，executor 积压/DDS 延迟不会被盖掉；
  - 中间方向（`mid_long_seg`）：`stamp = 发布时刻的 ROS 时钟`，`age≈0`，符合"立即响应正在说话"的低延迟设计，由 QoS `KEEP_LAST(1)` 与 `seq_id` 去重兜底，不额外设过期上限。
  - 方向的 `age` 在 `runtime` 内部用墙钟（`time.time()`）计算；`stamp` 在 `node` 用 ROS 时钟（`get_clock().now()`）构造。**当前部署 `use_sim_time=false`，ROS 时钟与墙钟同属系统时钟域，二者差值有效**，上述分流在实车上正确。`use_sim_time=true`（仿真/Bag 回放）下 ROS 时钟与墙钟不同步，本 PR 不解决该混合时钟域；如需在仿真或回放场景消费方向，应使用 `use_sim_time=false`，或后续单独统一为单一时钟域。
- 长语音累计达到 `max_accum_dur_s` 时发布一次中间方向并清理本轮累积，避免持续讲话时等待整段结束才响应；语音段结束时再发布当前累积窗口的段末方向。两类输出都是有效方向事件。
- 每次中间方向或段末方向都有独立递增的 `seq_id`，消费者按输出事件去重；`seq_id` 不表示“一段语音只对应一个序号”。
- 节点只在取得新的有效方向事件时发布；无人声、结果过期或降级时不发布方向。
- 参数缺失、参数非法或配置的模型资产不存在时，节点启动失败。其中模型资产缺失会提示运行 `python3 scripts/verify_speech_direction_assets.py` 校验资产清单（脚本只校验不下载，资产需从 NAS 手动获取）。
- 模型资产已存在但模型加载、音频设备打开或运行时推理失败时，节点保持运行、不发布方向，并通过 `/diagnostics` 持续报告降级状态。

#### 坐标系与安装偏角

阵列坐标系（SRP-PHAT 算法内部约定）：`0°=右(+x)`，`90°=前(+y)`，`180°=左`，`270°=后`，逆时针为正。发布到 `/voice/speech_direction` 的 `azimuth_rad` 遵循 REP-103：`0=前`，`+π/2=左`，`-π/2=右`，左转为正。

`mount_yaw_deg` 就是把阵列坐标系对齐到小车坐标系的安装偏角（度），转换式为：

```
ros_azimuth = radians(阵列角度) - π/2 + radians(mount_yaw_deg)
```

- `mount_yaw_deg=0`：阵列正前方（90°）对齐小车正前方（0 rad）。
- 阵列相对小车正前方**逆时针**偏 α°：`mount_yaw_deg=α`（偏角为正）。
- 阵列相对小车正前方**顺时针**偏 α°：`mount_yaw_deg=-α`（偏角为负）。

按阵列实际安装姿态填入该夹角，输出的 `azimuth_rad` 即在车体坐标系下。默认 `0.0`，通过 `speech_direction.yaml` 配置。

**当前默认情况**：默认值 `0.0` 表示假设阵列正前方（90°）对齐小车正前方、无安装偏角——这是默认假设，**并非物理标定结果**。离线回归基线（`test/speech_direction/audio/` 下 6 文件 12 段，均 ≤15° 通过）即在此假设下录制与校验，因此基线对偏角敏感的标定数据不反映实车安装姿态。实车上若阵列安装存在夹角，必须按实际测量据实填入，否则输出方位会带一个等于该夹角的系统偏差；保持默认 `0.0` 等价于"忽略安装夹角，按无偏角处理"。

#### SRP 角度精度

`angle_step_degree` 是 SRP-PHAT 扫描角度步长（度），节点启动时生成 `0, step, 2·step, …, 360-step` 的候选角度序列并预计算导向相位矩阵；最终 DOA 只能为该步长的整数倍。必须为 360 的正整数约数（如 `1、2、3、4、5、6、8、9、10`），否则候选角度无法均匀覆盖整圈，节点启动时校验失败。

默认 `5`（对应回归基线：5 文件 8 段 max err ~15°）。

**关于调成更小步长（如 1°）**：程序能正常计算并按 1° 步长输出角度值，**但 4 麦小阵列的物理角度分辨能力有限**——空间谱主瓣较宽，相邻多个角度的 score 接近，`argmax` 在主瓣顶部易被噪声和相位误差在相邻角度间随机推动。步长小于阵列物理分辨能力后，输出精度不再随之提升，仅增加计算量（候选角度数线性增长，导向相位矩阵与 einsum 投票代价随之上升）与噪声敏感度。

真正限制角度精度的是阵列孔径与频段（参见 `doa/srp_phat.py` 的几何与频段参数），而非扫描步长。要追求更高角度精度，需换更大孔径或更多麦克风的阵列，不是单靠调小 `angle_step_degree`。默认 `5` 是与当前基线匹配的务实值。

### 实时音频采集兼容性

`speech_direction_node` 通过受控 `arecord` 子进程直接采集 ALSA `hw` 设备，
并把原始交错多通道 int16 PCM 写入内部 RingBuffer。该路径不再依赖 PyAudio，
用于规避 Ubuntu 22.04 apt 默认 PyAudio 0.2.11 的 callback ABI 路径；音频后端异常通过
受控 fatal 通道上报，不会从 callback C bridge 异步注入 ROS executor。采集线程只负责
读取和缓冲，VAD、增强与 DOA 仍在独立 worker 中执行。当前已完成软件单元测试、构建
与并发生命周期验证，尚未使用真实 ReSpeaker 硬件完成采集验证。

> 部署依赖变更：音频采集后端已由 PyAudio 切换为 `arecord`（`alsa-utils` 提供）。
> 310P 与 Ubuntu 平台均使用 arecord 子进程直采 ALSA，系统需安装 `alsa-utils`；
> 采集设备由 `arecord_device` 参数（如 `hw:0,0`）指定，选择方式见下表。

### 麦克风资源互斥

`VoiceASRNode` 与 `SpeechDirectionNode` 当前各自建立音频采集链路，并未共享采集。两者不能同时占用同一个 ReSpeaker 实时输入设备；启动方向节点前，应停止正在使用该设备进行实时识别的 `VoiceASRNode`，反之亦然。WAV / 文件输入不等同于占用实时麦克风设备。

## 1. VoiceASRNode 节点职责

`VoiceASRNode` 支持两类输入路径：

| 输入路径 | 作用 | 模型要求 |
| --- | --- | --- |
| 麦克风实时识别 | 从音频输入设备持续监听并输出识别文本 | **必须使用流式模型** |
| 音频文件识别 | 解码文件并返回/发布识别结果 | 可使用流式或离线模型 |

核心职责包括：

1. 读取 ROS 参数并初始化各个运行模块。
2. 在模型文件缺失时自动解析并下载默认 ASR bundle。
3. 从麦克风采集音频或从文件加载音频。
4. 使用 VAD 判断语音起止边界。
5. 调用 sherpa-onnx 执行解码，并发布中间/最终结果。
6. 通过 ROS topic 和 service 暴露控制与文件识别能力。

## 2. 文件位置与启动入口

| 项目 | 路径 |
| --- | --- |
| 节点实现 | `src/voice_asr_service/voice_asr_service/voice_asr_node.py` |
| 控制台入口 | `voice_asr_node = voice_asr_service.voice_asr_node:main` |
| 包级 README | `src/voice_asr_service/README.md` |

直接调试节点时可这样运行：

```bash
cd /path/to/IB_Robot
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 run voice_asr_service voice_asr_node --ros-args \
  -p model_path:=models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 \
  -p model_type:=streaming
```

生产或完整系统场景仍建议通过 `robot_config` 启动，因为机器人级参数的单一事实来源仍然是 `robot_config`：

```bash
cd /path/to/IB_Robot
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm
```

如果希望在统一 launch 下**临时启用并自动开始监听**，可直接增加：

```bash
cd /path/to/IB_Robot
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  voice_asr_auto_start:=true
```

这里的 `voice_asr_auto_start` 是 **launch 参数**，不是 YAML 字段。它会在启动时临时覆盖为：

- `voice_asr.enabled=true`

`active_mode` 默认已经是 `continuous`，因此启用后会自动开始监听。

## 3. 运行时结构

`VoiceASRNode` 本身更像一个编排节点，具体功能主要分发给内部模块：

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| `AudioCaptureModule` | `audio_capture_module.py` | 麦克风设备选择、缓冲、pre-roll、分块采集 |
| `FileInputModule` | `file_input_module.py` | 文件加载、解码、重采样、进度回调 |
| `VADModule` | `vad_module.py` | 语音活动检测与语音/静音分段 |
| `ASRInferenceModule` | `asr_inference_module.py` | sherpa-onnx 模型初始化与解码 |
| `StateMachine` | `state_machine.py` | 节点模式与状态切换 |
| `model_manager` | `model_manager.py` | 在配置模型缺失时解析/下载默认 ASR bundle |

整体数据流：

```text
麦克风或音频文件
  -> 音频归一化 / 缓冲
  -> VAD 分段
  -> sherpa-onnx 解码
  -> 中间 / 最终文本
  -> ROS topic / service 响应
```

## 4. 识别模式

节点内部通过 `StateMachine` 维护 `active_mode`，当前支持：

| 值 | 含义 |
| --- | --- |
| `manual` | 默认空闲，由 service 或 `/voice_control` 触发识别 |
| `continuous` | 节点启动后自动进入监听 |
| `wake_word` | 状态机预留值；当前节点里还没有独立的唤醒词流水线 |

关键行为约束：

- **麦克风实时识别必须使用流式模型。**
- **离线模型仍可用于 `~/recognize_file` 和 `/voice_file_input`。**
- 如果当前加载的是离线模型，而外部请求实时识别，节点会明确拒绝并记录错误，而不是崩溃。

## 5. 模型加载与自动下载

节点主要读取这些参数：

- `model_path`
- `tokens_path`
- `model_type`
- `language`
- `provider`
- `auto_download_model`

初始化流程如下：

1. `resolve_model_assets()` 先检查 `model_path` 是否为空或已存在。
2. 如果 `model_path` 为空，或配置的模型缺失，且 `auto_download_model=true`，
   节点会在启动时按当前意图选择默认 bundle 并在缺失时自动下载。
3. 下载后的 bundle 路径会回填到节点实际使用的运行参数里。
4. `ASRInferenceModule.initialize()` 根据模型类型创建流式或离线 recognizer。

当前默认 bundle：

| Profile | Bundle | 用途 |
| --- | --- | --- |
| `streaming_zh` | `sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23` | 默认中文实时 ASR |
| `offline_zh` | `sherpa-onnx-paraformer-zh-int8-2025-10-07` | 默认中文离线文件识别 |

模型目录：

```text
models/voice_asr/
```

### 流式与离线模型的判定

运行时的区分方式是：

- 流式模型：目录中存在 `encoder*.onnx`、`decoder*.onnx`、`joiner*.onnx`
- 离线模型：通常是 paraformer 这种单模型 ONNX 布局，例如 `model.int8.onnx`

## 6. 实时麦克风识别流程

实时识别由控制循环定时器和 `_process_audio()` 驱动：

1. 从 `AudioCaptureModule` 读取一个音频块。
2. 调用 `VADModule.process()` 判断当前音频状态。
3. 检测到开始讲话后，创建一个流式 ASR 会话。
4. 先补喂一小段 pre-roll，避免句首被截断；默认通过 `realtime_pre_roll_seconds=0.5` 保留实时缓存，实际一次性喂给流式 ASR 的音频会被限制在最近 0.5 秒内，避免启动识别时阻塞控制循环。
5. 在语音活动期间持续向 ASR 喂入音频块。
6. 如果 `publish_partial=true`，就发布中间结果。
7. 在静音或超时后结束识别，并发布最终结果。

实时链路的几个细节：

- VAD 进入 `STARTING`、`SPEAKING` 或 `ENDING` 都会被视为语音活动并喂给 ASR，避免截断句首或句尾。
- `realtime_pre_roll_seconds` 会保留 VAD 判定前的实时音频，减少句首丢失；当前帧会从 pre-roll 中裁掉，避免重复喂入。为保证实时性，流式 ASR 启动时最多一次性补喂最近 0.5 秒。
- 如果检测到讲话时当前模型是离线模型，节点会停止采集并记录明确错误。

## 7. 文件识别流程

即使麦克风实时识别不可用，文件识别仍然可以工作。

当前有两个入口：

| 入口 | 类型 | 行为 |
| --- | --- | --- |
| `~/recognize_file` | Service | 同步请求 / 响应 |
| `/voice_file_input` | Topic | 异步后台线程处理 |

### `ibrobot_msgs/srv/RecognizeFile`

**请求字段**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `file_path` | `string` | 待识别文件路径 |
| `enable_vad` | `bool` | 是否先做 VAD 分段 |

**响应字段**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `success` | `bool` | 是否识别成功 |
| `error_message` | `string` | 失败原因 |
| `results` | `string[]` | 每段识别文本 |
| `timestamps` | `float32[]` | 每段起始时间 |
| `durations` | `float32[]` | 每段时长 |

## 8. ROS 接口

### 发布的话题

| 话题 | 类型 | 含义 |
| --- | --- | --- |
| `output_topic`（默认 `/voice_command`） | `std_msgs/String` | 最终识别文本 |
| `/voice_partial` | `std_msgs/String` | 中间识别结果 |
| `/voice_status` | `std_msgs/String` | 当前节点状态 |
| `/voice_confidence` | `std_msgs/Float32` | 最终结果置信度 |
| `/voice_file_progress` | `std_msgs/Float32` | 文件处理进度 |

### 订阅的话题

| 话题 | 类型 | 含义 |
| --- | --- | --- |
| `/voice_control` | `std_msgs/String` | 通过文本命令控制开始/停止识别 |
| `/voice_file_input` | `std_msgs/String` | 提交待异步识别的文件路径 |

当前可识别的 `/voice_control` 命令包括：

- `start`
- `开始`
- `开始监听`
- `stop`
- `停止`
- `停止监听`

### 服务

| 服务 | 类型 | 含义 |
| --- | --- | --- |
| `~/start_recognition` | `std_srvs/srv/Empty` | 开始一次实时监听 |
| `~/stop_recognition` | `std_srvs/srv/Empty` | 停止当前实时监听 |
| `~/set_hotwords` | `ibrobot_msgs/srv/SetHotwords` | 更新热词增强配置 |
| `~/recognize_file` | `ibrobot_msgs/srv/RecognizeFile` | 识别一个音频文件 |

### `ibrobot_msgs/srv/SetHotwords`

**请求字段**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `hotwords` | `string[]` | 需要增强的热词 |
| `boost_scores` | `float32[]` | 每个热词对应的增强分数 |

**响应字段**

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `success` | `bool` | 是否设置成功 |
| `error_message` | `string` | 失败原因 |

## 9. 参数说明

### ASR 行为参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `active_mode` | `continuous` | 节点激活模式 |
| `language` | `zh` | 传给 ASR 初始化的语言提示 |
| `model_path` | `""` | 模型文件或目录路径；具体机器人配置可在 `robot_config` YAML 中覆盖 |
| `tokens_path` | `""` | 可选的显式 tokens 路径 |
| `provider` | `cpu` | sherpa-onnx 推理 provider |
| `model_type` | `auto` | `auto`、`streaming` 或 `offline` |
| `auto_download_model` | `true` | 配置模型缺失时是否自动下载默认 bundle |
| `max_recording_duration` | `10.0` | 实时识别最长录音时长，超时后强制收尾 |
| `publish_partial` | `true` | 是否发布中间解码结果 |
| `output_topic` | `/voice_command` | 最终命令输出 topic |
| `exit_on_init_failure` | `true` | 初始化失败时是否直接抛错退出 |

### 音频 / VAD 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `vad_sensitivity` | `0.6` | VAD 灵敏度 |
| `realtime_pre_roll_seconds` | `0.5` | 识别启动时补回的实时缓存时长，用于减少句首丢字 |
| `sample_rate` | `16000` | 当前完整 Voice ASR 链路仅支持 16000 Hz；其他值属于无效配置，节点会拒绝初始化 |
| `chunk_size` | `512` | 当前完整 Voice ASR 链路仅支持 512 样本帧；其他值属于无效配置，节点会拒绝初始化 |
| `buffer_seconds` | `5.0` | 音频环形缓冲区时长 |
| `device_index` | `-1` | 显式音频设备索引；`-1` 表示默认设备 |
| `device_name` | `""` | 优先按设备名匹配，失败后回退到索引 |

当前 16kHz/512 是实时麦克风、文件识别、VAD 后端和 ASR 模型共同遵守的系统级硬限制；
Silero ONNX 后端会在 512 样本音频帧前额外拼接 64 个内部 context 样本，该 context 由 VAD 内部跨帧维护，不应配置到 `chunk_size` 中。

## 10. 状态机

节点状态包括：

| 状态 | 含义 |
| --- | --- |
| `idle` | 空闲，等待触发 |
| `listening` | 正在监听并等待语音开始 |
| `recognizing` | 已检测到语音，ASR 流正在运行 |
| `hold` | 预留中间状态 |
| `error` | 运行时错误状态 |

典型的实时路径如下：

```text
idle -> listening -> recognizing -> listening -> idle
```

节点会把状态变化发布到 `/voice_status`。

## 11. 失败处理

节点已经对以下常见失败情况做了显式保护：

- `model_path` 缺失
- ASR 初始化失败
- 使用离线模型请求实时识别
- 文件解码失败
- 初始化失败后继续收到识别请求

需要注意：

- `VoiceASRNode initialized` **并不代表** ASR 已经可用。
- 真正的成功信号通常是后续日志里的 `ASR model loaded: ...`。
- 如果 `exit_on_init_failure=true`，初始化失败会直接导致启动失败。
- 如果 `exit_on_init_failure=false`，节点会继续存活，但在 ASR 初始化成功之前会拒绝相关请求。

## 12. 推荐配置方式

机器人级别的 SSOT 位于：

```text
src/robot_config/config/robots/so101_single_arm.yaml
```

典型的 ASR 配置片段如下：

```yaml
robot:
  voice_asr:
    enabled: false
    active_mode: continuous
    language: zh
    model_path: models/voice_asr/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23
    tokens_path: ""
    provider: cpu
    model_type: streaming
    auto_download_model: true
    max_recording_duration: 10.0
    vad_sensitivity: 0.6
    realtime_pre_roll_seconds: 0.5
    publish_partial: true
    output_topic: /voice_command
    sample_rate: 16000
    chunk_size: 512
    buffer_seconds: 5.0
    device_index: -1
    device_name: ""
    exit_on_init_failure: true
```

默认建议把 `enabled` 保持为 `false`，只在需要时通过 `voice_asr_auto_start:=true` 临时启用；如果你的机器人就是要长期带语音入口，再把 YAML 改成 `enabled: true` 即可。

如果只想做离线文件识别，可以切换到离线 bundle，并继续使用 `~/recognize_file`。

## 13. 排障

| 现象 | 常见原因 | 检查点 |
| --- | --- | --- |
| 节点能启动，但实时识别始终不可用 | 加载的是离线模型 | 查看日志里是否出现 `Offline ASR model loaded` |
| `start_recognition` 被拒绝 | ASR 未就绪，或当前模型是离线模型 | 查看 `_asr_init_error` 相关日志和模型类型 |
| 文件识别立即失败 | 文件路径错误或解码失败 | 确认文件存在且格式受支持 |
| 麦克风没有音频输入 | 设备选择不对 | 检查启动时的设备日志，使用 `device_name` 或 `device_index` 指定 |
| 模型路径缺失 | bundle 尚未下载完成 | 开启 `auto_download_model`，并在首次启动 ASR 节点时等待自动下载完成 |

## 14. 当前已验证行为

当前实现已经验证过以下能力：

- 流式模型初始化
- 离线模型下的实时识别保护逻辑
- 配置模型缺失时的自动解析与下载
- 使用自带 streaming 样例音频进行真实解码
- 保持离线文件识别可用

这意味着节点当前支持的预期分工是：

- **流式模型负责麦克风实时识别**
- **离线或流式模型都可以用于文件识别**
