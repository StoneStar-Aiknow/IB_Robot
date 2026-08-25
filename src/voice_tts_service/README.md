# Voice TTS 语音合成服务

[English](README.en.md)

`voice_tts_service` 是 IB-Robot 的 ZipVoice 模型服务 plugin。它由 `inference_service` 的通用
`model_service_node` 承载，接收文本和可选参考音色，调用显式选择的
ZipVoice deployment，返回一个或多个可独立播放的单声道 WAV PCM16 音频段，并提供播放服务所在主机的扬声器
播放接口。

本包负责“文本转音频”和播放端本机 WAV 文件，不负责 ASR、业务编排或通过 SSH 调用远端推理。

## 1. 包职责

核心职责包括：

1. 提供统一的 ROS 2 TTS 服务接口。
2. 校验 ZipVoice 模型 bundle、manifest 和 named deployment。
3. 通过通用模型服务宿主管理模型加载、常驻复用和关闭。
4. 对长文本进行有界分段，并把模型输出统一封装为 WAV PCM16。
5. 在 Ascend 310P 上编排 Text Encoder OM、Flow Decoder OM 和 CPU Vocos。
6. 对请求大小、分段数量和响应音频大小设置明确上限。
7. 通过 ALSA 和系统默认音频输出同步播放本机 WAV 文件，并返回稳定的成功或失败状态。

不属于本包的职责：

- 麦克风采集和语音识别。
- 麦克风或扬声器设备的发现、热插拔和混音管理。
- 机器人业务流程或对话状态管理。
- 根据操作系统猜测模型后端，或在加载失败时静默切换后端。
- 在运行时通过 SSH 执行模型推理。

## 2. 文件位置与启动入口

| 项目 | 路径或入口 |
| --- | --- |
| 通用 ROS 宿主 | `inference_service/inference_service/model_service_node.py` |
| TTS plugin | `voice_tts_service/voice_tts_service/model_service_plugin.py` |
| 音频播放节点 | `voice_tts_service/voice_tts_service/audio_playback_node.py` |
| 310P adapter | `voice_tts_service/zipvoice_310p_adapter.py` |
| ONNX adapter (Ubuntu) | `voice_tts_service/zipvoice_onnx_adapter.py` |
| 模型 bundle 工具 | `voice_tts_service/package_zipvoice_310p.py` |
| 调试 launch | `launch/voice_tts.launch.py` |
| 控制台入口 | `model_service_node = inference_service.model_service_node:main` |
| 模型打包入口 | `package_zipvoice_310p = voice_tts_service.package_zipvoice_310p:main` |

完整机器人系统应通过 `robot_config` 启动，机器人 YAML 是 TTS 配置的单一事实来源：

```bash
source .shrc_local
export ROS_DOMAIN_ID=42
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm
```

包级 launch 仅用于独立调试：

```bash
source .shrc_local
export ROS_DOMAIN_ID=42
ros2 launch voice_tts_service voice_tts.launch.py \
  bundle_path:=/path/to/zipvoice-bundle \
  deployment:=ascend_310p
```

Ubuntu 主机可使用 ONNX 后端（onnxruntime + CPU Vocos），无需 Ascend 硬件：

```bash
source .shrc_local
ros2 launch voice_tts_service voice_tts.launch.py \
  bundle_path:=/path/to/zipvoice-bundle \
  deployment:=ubuntu_onnx
```

## 3. 数据流

```text
SynthesizeSpeech 请求
  -> 请求校验与文本分段
  -> ModelRequest + ExecutionContext
  -> ModelRuntimeHandle.execute
  -> 启动时加载并常驻的 ZipVoice ModelSession resource
  -> ZipVoice tokenizer
  -> Text Encoder OM
  -> Flow Decoder OM（主机侧 4-step Euler）
  -> CPU Vocos
  -> 单声道 float PCM
  -> WAV PCM16 封装
  -> SynthesizedAudio[] 响应
```

播放链路与模型推理解耦：

```text
PlayAudioFile 请求（播放端本机绝对路径）
  -> WAV 文件校验
  -> ALSA aplay
  -> success / error_code / message
```

`ZipVoiceSynthesizePlugin` 把 `ModelSession` resource 放入 `RuntimeAssembly`，再将所有权转移给
`ModelRuntimeHandle`。Handle 串行准入请求并管理公开生命周期、健康状态、取消和关闭等待；session 只持有
并释放 ZipVoice 的 vendor 模型与设备资源。

## 4. ROS 接口

### 4.1 语音合成

| 项目 | 值 |
| --- | --- |
| 服务名 | `/voice_tts/synthesize` |
| 服务类型 | `ibrobot_msgs/srv/SynthesizeSpeech` |
| 音频段类型 | `ibrobot_msgs/msg/SynthesizedAudio` |

请求字段：

| 字段 | 含义 |
| --- | --- |
| `text` | 待合成的完整文本 |
| `prompt_audio` | 可选的内存 WAV 文件字节 |
| `prompt_audio_format` | 提供参考音频时填写 `wav` |
| `prompt_text` | 参考音频中实际说出的文本 |

响应包含成功状态、稳定错误码、模型运行信息、总耗时，以及一个或多个完整 WAV 音频段。接口不接受服务端
输入或输出文件路径，因此可以跨主机通过 ROS 2 调用。

只传文字时使用 bundle 中配置的默认音色：

```bash
source .shrc_local
export ROS_DOMAIN_ID=42
ros2 service call /voice_tts/synthesize ibrobot_msgs/srv/SynthesizeSpeech \
  "{text: '机器人语音服务已准备就绪。', prompt_audio: [], prompt_audio_format: '', prompt_text: ''}"
```

当前验证的 `ascend_310p` deployment 使用固定默认音色，不支持请求级音色克隆。传入参考音频时会返回
`UNSUPPORTED_PROMPT`，不会静默忽略请求参数。

### 4.2 音频播放

| 项目 | 值 |
| --- | --- |
| 服务名 | `/voice_tts/play` |
| 服务类型 | `ibrobot_msgs/srv/PlayAudioFile` |
| 输入 | 播放服务所在机器上的 WAV 文件绝对路径 |

服务同步等待整段音频播放完成，再返回 `success=true`。路径不存在、WAV 无效、ALSA 设备不可用、播放超时或
`aplay` 返回非零状态时返回 `success=false`，并填写稳定的 `error_code` 和 `message`。服务不会通过 SSH 拉取
文件，也不会解释调用端本机路径。

```bash
ros2 service call /voice_tts/play ibrobot_msgs/srv/PlayAudioFile \
  "{file_path: '/tmp/voice_tts/output.wav'}"
```

### 4.3 模型生命周期

| 阶段 | 类型 | 行为 |
| --- | --- | --- |
| 节点启动 | 通用宿主 + TTS plugin | 校验 bundle，构造 session、`RuntimeAssembly` 和 handle，并调用 `handle.load()` |
| 节点退出 | plugin 的 `ModelRuntimeHandle` | 停止准入、等待 active inference，并关闭 assembly-owned session resource |

生命周期如下：

```text
节点启动 -> 通用宿主校验 bundle -> plugin 构造 RuntimeAssembly/ModelRuntimeHandle -> handle.load()
synthesize -> handle.execute(ModelRequest, ExecutionContext) -> 复用常驻 ModelSession resource
节点退出 -> handle.close() -> session 释放 OM、ACL lease、Vocos、tokenizer 和 prompt
```

模型准入、公开生命周期、健康、取消和关闭 drain 由 `ModelRuntimeHandle` 管理。`ModelSession` 是
handle-owned resource，负责 ZipVoice vendor 资源的加载、执行与释放；宿主关闭 plugin 时由 handle 等待推理结束。

`exit_on_init_failure=false`（对应通用宿主的 `required=false`）只保证初始化失败后 typed endpoint 继续在线并
返回 `MODEL_NOT_READY`。当前宿主不会在后续请求中自动重试初始化；修复 bundle、依赖或设备后必须重启节点。
`INVALID_TEXT`、`UNSUPPORTED_PROMPT` 等请求级错误不会改变 handle 健康状态，也不会阻塞后续有效请求。

## 5. 模型 bundle 与 deployment

模型路径不写死在代码中，由 `robot_config` 的 `bundle_path` 指定。相对路径以 `.shrc_local` 设置的绝对
`WORKSPACE` 为根目录解析。例如：

```yaml
bundle_path: models/zipvoice
deployment: ascend_310p
```

对应模型目录为：

```text
$WORKSPACE/models/zipvoice/
├── inference_manifest.json
├── assets/
│   └── ...
└── artifacts/
    └── ...
```

bundle 必须满足：

- manifest schema 为 v3。
- 模型身份为 `interface=tensor_model`、`model_type=zipvoice`、`operation=synthesize`。
- `deployment` 必须是 manifest 中存在的命名 deployment。

### 5.1 已验证的 deployment

| deployment | backend | 后端运行时 | 说明 |
| --- | --- | --- | --- |
| `ascend_310p` | ascend | ACL + OM | 310P 上编排 Text Encoder OM、Flow Decoder OM 和 CPU Vocos |
| `ubuntu_onnx` | torch | onnxruntime + CPU | Ubuntu 主机用 onnxruntime 加载上游 ONNX 模型，复用 310P bundle 的 tokens/Vocos/prompt 资产 |

`ubuntu_onnx` deployment 的 ONNX 模型从 [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) 上游获取（ModelScope 镜像），通过 `scripts/download_voice_tts_models.sh` 下载。该脚本同时生成 `zipvoice_onnx.json` 运行时配置和 `inference_manifest.json`（含 `ubuntu_onnx` deployment），使 bundle 可直接用于 `deployment:=ubuntu_onnx`。

manifest 的 bundle digest 和 deployment fingerprint 用于结构身份与部署一致性，不读取模型文件内容，
也不提供运行时防篡改。310P 打包工具会在复制前校验已知来源的 OM、Vocos checkpoint、token table
和 golden fixture 的 SHA-256；运行时负责校验 manifest、路径安全、文件存在性和模型 ABI。正式部署如需
内容防篡改，应使用只读镜像、签名或 verity。

运行时不会根据操作系统推断后端，也不会在 deployment 加载失败时回退到另一个后端。

### 5.2 打包 Ascend 310P 模型

模型文件体积较大，不提交到 Git。使用打包工具从已准备好的 ZipVoice 交付目录生成标准 bundle：

```bash
source .shrc_local
ZIPVOICE_SOURCE_DIR=/path/to/zipvoice-delivery
ros2 run voice_tts_service package_zipvoice_310p \
  --source "$ZIPVOICE_SOURCE_DIR" \
  --destination "$WORKSPACE/models/zipvoice"
```

打包工具会校验 Text Encoder OM、Flow Decoder OM、Vocos checkpoint、token table 和默认 prompt，并生成
`ascend_310p` deployment。中文前端依赖由项目 `requirements/voice-tts.txt` 固定安装，Vocos 推理实现由
`voice_tts_service.vocos_backend` 版本控制；模型 bundle 只包含权重和配置，不携带可执行 Python 源码。
设备地址和模型原始源码目录都不会成为运行时配置。

当前受控 Vocos 实现记录为 `0.1.0-zipvoice-310p-delivery`，其来源文件 SHA-256 写在模块常量中。
该交付实现使用项目现有 Torch、NumPy 和 SciPy，不安装 PyPI `vocos`，避免其传递依赖替换 ROS/Ascend ABI。

## 6. `robot_config` 配置

生产配置位于机器人 YAML 的 `robot.voice_tts`：

```yaml
voice_tts:
  enabled: true
  bundle_path: models/zipvoice
  deployment: ascend_310p

  service_name: /voice_tts/synthesize
  playback_service_name: /voice_tts/play
  playback_timeout_sec: 300.0

  prompt_profile: default
  segment_max_chars: 200
  segment_pause_ms: 150
  max_request_chars: 4000
  max_prompt_audio_bytes: 10485760
  max_prompt_duration_sec: 30.0
  max_segments: 32
  max_response_audio_bytes: 67108864

  device_id: 0
  exit_on_init_failure: true
```

关键参数：

| 参数 | 含义 |
| --- | --- |
| `enabled` | 是否由统一 `robot.launch.py` 启动 TTS 节点 |
| `bundle_path` | ZipVoice bundle 路径 |
| `deployment` | manifest 中的命名部署 |
| `playback_service_name` | 本机 WAV 播放服务名 |
| `playback_timeout_sec` | 单次同步播放超时 |
| `prompt_profile` | 默认音色 profile |
| `segment_max_chars` | 单个公共文本段的最大字符数 |
| `max_request_chars` | 单次请求的最大字符数 |
| `max_segments` | 单次响应允许的最大音频段数量 |
| `device_id` | Ascend 设备编号 |
| `exit_on_init_failure` | bundle 初始化失败时是否退出节点 |

代表性配置默认 `enabled=false`，在模型 bundle 和 deployment 准备完成前不会改变现有机器人启动行为。

## 7. Ascend 310P 支持范围

当前验证的 `ascend_310p` deployment：

- 目标为 Ascend 310P1。
- Text Encoder 和 Flow Decoder 使用 OM，并跨请求常驻。
- Flow Decoder 由主机执行 4-step Euler 调度。
- Vocos 当前在 CPU PyTorch 上执行。
- 输出为 24 kHz、单声道 WAV PCM16。
- 支持中文、阿拉伯数字和常用中英文标点。
- 输入文本会移除 Markdown 格式符号和控制字符；表情或其他不可朗读符号若紧邻已有标点则直接移除，
  否则替换为句号，以保留必要停顿；换行同样规范化为句号。
- ASCII 英文单词会明确失败。
- 使用 bundle 中的固定默认 prompt，不支持请求级音色克隆。

`ZipVoiceAscendSession` 继承公共 `AscendOmModelSession`，作为 `RuntimeAssembly` 中的 `ModelSession`
resource 持有 ACL lease、OM、Vocos、tokenizer 和 prompt。Plugin 调用
`ModelRuntimeHandle.execute(ModelRequest, ExecutionContext)`；handle 负责准入、生命周期、健康、取消和关闭等待，
session 负责 vendor 资源，不会在同一进程中重复初始化全局 ACL。

## 8. 稳定错误码

| 错误码 | 含义 |
| --- | --- |
| `INVALID_TEXT` | 文本为空或规范化后为空 |
| `INVALID_PROMPT_PAIR` | 参考音频、格式和文本没有成组提供 |
| `INVALID_PROMPT_AUDIO` | 参考 WAV 无法解码或格式不支持 |
| `PROMPT_TOO_LARGE` | 参考音频超过字节数或时长上限 |
| `REQUEST_TOO_LARGE` | 文本或分段数量超过请求上限 |
| `RESPONSE_TOO_LARGE` | 合成音频超过响应字节上限 |
| `MODEL_NOT_READY` | bundle、deployment 或模型 runtime 加载失败 |
| `INFERENCE_FAILED` | 模型推理失败 |
| `INVALID_AUDIO_OUTPUT` | 模型输出为空或包含 NaN/Inf |
| `UNSUPPORTED_PROMPT` | 所选 deployment 不支持请求级音色克隆 |
| `INTERNAL_ERROR` | 未分类的服务内部错误 |

播放接口还会返回 `INVALID_PATH`、`FILE_NOT_FOUND`、`NOT_A_FILE`、`UNSUPPORTED_FORMAT`、
`INVALID_AUDIO_FILE`、`PLAYER_NOT_FOUND`、`PLAYBACK_TIMEOUT` 或 `PLAYBACK_FAILED`。

## 9. 测试与构建

执行项目命令前必须加载环境：

```bash
source .shrc_local
```

运行包测试：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q src/voice_tts_service/test
```

构建单包：

```bash
colcon build --symlink-install --merge-install --packages-select voice_tts_service
```
