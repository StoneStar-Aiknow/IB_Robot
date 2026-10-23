# X2 语音增强流与外部分段 ASR

## 决策与边界

当前实现把增强流桥接放在 `aimdk_runtime`（`runtime_node.py`）内部，与原始采集流、播放和音频焦点并列：runtime 读取厂商 `ProcessedAudioOutput`，发布项目中立的 `SpeechAudioChunk`。X2 仍然只有一个厂商边界，没有第二个进程或第二个 `aimdk_msgs` 入口；`voice_asr_service` 不导入 `aimdk_msgs`。`input_mode` 只有 `shared_audio` 与 `external_vad` 两个取值。

选择厂商增强流的依据是现场对照：厂商 AEC/波束成形流能抑制本机 TTS 回声，而原始麦克风会将播报混入识别。本地再次运行 Silero VAD 会重新判断句首，可能截断厂商已经回填的音频。保留外部分段语义，复用既有 `ASRInferenceModule`、manifest 模型装配和统一推理运行时。

真机验证结论：唤醒后给出语音指令，识别经 `/voice_command` 下发并执行 SUCCEEDED。厂商语音会话窗口由厂商策略控制（唤醒后静默一段时间关闭），其确切时长厂商无文档、未测到边界；`audio.speech` 的 ready 只证明运行时的该链路在运行，不证明厂商已唤醒。

## 原子传输契约

主链为：厂商增强流 → `aimdk_runtime`（`_setup_processed_audio`）→ `SpeechAudioChunk` → `ExternalVadInput` → ASR → `/voice_command`。

同一条链上的 `MessageHeader.sequence` 在厂商 IDL 中定义为连续递增 ID，但 X2 固件当前实际发送的是**常量**（2026-10-15 真机实测），因此**在 X2 上上游丢帧不可观测**。runtime 只把严格向前跳号当作丢帧证据，此时让项目 `sequence` 跳一个号，消费端据此丢弃当前段；常量、重复与回退值都不判缺号——早期版本用「不等于上次 +1」判定，结果每条消息都被判缺号、语音被全部丢弃。`stream_id` 标识厂商流，其变化会重置该基线。向前跳号检测保留给真正实现该 IDL 的固件；在 X2 上，段超时与不完整段丢弃是唯一兜底。

`SpeechAudioChunk` 的 PCM 格式固定为单声道、16000 Hz、S16LE。每条消息包含 `source_id`、`sequence`、`state` 和 `audio_data`。`source_id` 每次 runtime 启动重新生成，`sequence` 在该源内单调递增。消费者发现缺号时丢弃当前段，重复或倒序消息不再识别。

BEGIN 先打开新流，再喂入该消息携带的完整回填 PCM；PROCESSING 持续喂入；END 先喂入尾部 PCM，再做最终识别。音频与边界在一条 ROS 消息中，协议不依赖两个 topic 的接收顺序，也不存在仅含 state 的独立分段消息。

runtime 还发布 `/audio/enhanced_capture_stamped`，仅用于录音或诊断；该端点像其他公共话题一样在 profile 的 `interfaces` 中以 `audio.enhanced_capture` 声明（含自己的 QoS），因此在 runtime description 中可见，未被声明时不创建发布者；ASR 不用它与另一 topic 拼接。空 PCM、state=0 的在线心跳只表明 runtime 侧该链路在运行，让接口状态在唤醒之前就是 ready；心跳不代表厂商已唤醒、不延长 ASR 音频停滞期限，也不生成文字。

`audio.speech` 的在位状态由 runtime 复用既有 `robot_runtime.InterfaceMonitor` 上报，与 facade/mock runtime 的判据完全一致：只看话题上有没有数据，不因为「自己发布了」就宣称就绪。该 monitor 通过 `OBSERVED_INTERFACES = ("audio.speech",)` 过滤，只订阅这一个端点；厂商原生传感器（相机、雷达、IMU）刻意排除，否则会在 runtime 进程内把那些流再订一遍、与运动控制争资源。监视订阅挂在 runtime 的独立 callback group 上，避免排在高频关节回调之后被误判为过期。持续 2 秒既无 chunk 也无心跳即转 `stale`，`wait_for_runtime` 便不会放行消费者。

厂商输入默认请求 BEST_EFFORT + VOLATILE。它可匹配 TRANSIENT_LOCAL 发布者，但不申请历史语音回放：runtime 启动前锁存的旧语音不得被当成新说话。项目 chunk 输出是 RELIABLE + VOLATILE。厂商 `header.sequence` 只在固件真正实现递增时才是丢帧证据（X2 当前是常量，见上），因此段超时和不完整段丢弃始终是必要兜底，而不是可选优化。

## 识别生命周期

`input_mode` 选择数据来源，`active_mode` 选择激活方式；两者独立。

- `manual`：初始 IDLE，必须通过 start 服务或控制 topic 启用。
- `continuous`：流式模型就绪后自动进入 LISTENING。
- 外部模式不支持 `wake_word`；厂商唤醒属于音源侧策略，不用它静默覆盖激活配置。

只有 BEGIN 能打开流。stop 丢弃当前流、释放 ASR stream slot 并回到 IDLE；start 等待新的 BEGIN。关闭期间的 PROCESSING/END 不会在恢复后继续旧指令。

开流、喂入或 finalize 失败时，整段丢弃并释放资源；该段剩余 PCM 不成为下一段的前缀。奇数字节等无效 PCM 在回调边界拒绝。重复 BEGIN 丢弃旧段，再处理新 BEGIN 自己的音频。

`max_recording_duration` 同时限制段的墙钟持续时间和累计样本时长。`external_idle_timeout` 限制活动段没有真实 PCM 的时间，默认 2 秒。取消、丢包、超时和错误路径不发布部分指令。少于 `external_min_samples`（默认 3200）的段不发布最终文字。

## 配置所有权

厂商输入 topic 和订阅 QoS 归 X2 runtime profile 的 `vendor.topics.processed_audio`、`vendor.qos.processed_audio`。公共原子音频端点归 profile 的 `interfaces.audio.speech`，其 QoS 与 endpoint 由 `runtime_node` 直接读取；profile 未声明 `audio.speech` 而声明了 `processed_audio` 时 runtime 启动失败，而不是退回硬编码 topic。能力门控与原始采集一致：`interaction.audio_capture` 且 `processed_audio` 存在才接线。

语音链位于**独立的叠加 profile** `aimdk_x2_voice.yaml`（`base_config: aimdk_x2_skills.yaml`），而不是在 `aimdk_x2_skills.yaml` 里打开。原因有两条：X-ASR bundle 是仓库外提供物，语音的 launch 校验在 bundle 缺失时会直接抛错；`audio.speech` 绑定又会进入 `required_interface_ids()`，其就绪参与 launch 准入。把语音拆出去后，只做手势的部署用 `aimdk_x2_skills` 就能直接启动，既不依赖 bundle 也不受语音就绪超时影响。

因此部署命令随之变化：

```bash
# 只手势（无语音依赖）
ros2 launch embodied_bringup embodied_pipeline.launch.py robot_config:=aimdk_x2_skills ...
# 完整语音链
ros2 launch embodied_bringup embodied_pipeline.launch.py robot_config:=aimdk_x2_voice ...
```

`aimdk_x2_voice` 使用 `audio_io.source: runtime`，不再额外打开 ALSA capture；通过 `voice_asr.audio_input` 的逻辑 ID `audio.speech` 绑定 ASR 的输入，并刻意不写 `bundle_path` / `deployment`，让两者跟随 `VOICE_ASR_DEFAULTS`，避免主线换模型时 X2 被钉在旧值上。更改 runtime profile 中的 endpoint 会同时改变生产者与消费端。绑定缺失、接口类型不符、显式 topic 冲突或 runtime 根本不发布该接口时，都在 launch 前失败——没有 `audio_topic` 之类可静默兜底的旧字段。

独立 ASR 节点可显式使用 `chunk_topic`（这是 external 模式唯一的输入参数）；绑定式机器人配置优先使用 runtime description。模型以 `bundle_path` 和命名 `deployment` 选择；X-ASR bundle 为外部资产，需按 voice_asr_service README 打包。external 模式不装配 Silero；文件识别若启用本地 VAD，仍需提供对应 bundle。

## 实验路径和非目标

FullSubNet 增强 PCM 输出及 `x2_enhanced_asr.launch.py` 保留为默认关闭的对照工具，复用原有 speech-direction pipeline。它们不参与厂商 external_vad 主链。启用输出会关闭 raw activity gate，以保留静音供下游本地 VAD 收尾；因此会增加持续推理负载。

五个 named-motion 的 `duration_sec_estimate` 是作者实测的规划估计，不改变运行时动作边界。本次修复不执行动作，不改变机器人授权或厂商仲裁行为。

## 验证要求

纯逻辑回归覆盖原子 BEGIN/END 音频、连续语句、重复/缺号/源重启、manual 门控、stop/start、坏 PCM、短段、三类解码异常及后续段恢复。节点测试覆盖服务/控制 topic 与适配器生命周期的一致性。配置测试覆盖 profile 的非默认端点绑定、类型冲突，以及「runtime 不发布该接口时语音 profile 拒绝启动、手势 profile 照常可用」。就绪契约由 `aimdk_robot/test/test_runtime_speech_readiness.py` 覆盖：以真实 `RuntimeState` 输出为输入，无样本必须拒绝、注入心跳后放行、静默 2 秒后转 stale 再拒绝，并另有真实 ROS 订阅验证心跳确实到达订阅者。ROS 消息构建和包验收通过项目构建入口与 `colcon test` 进行。

测试以原子消息的行为为对象：BEGIN/PROCESSING/END 的喂入与收尾、缺号与源重启、坏包与短段、回调异常后的恢复；不断言两个 publisher 的调用次序等价于网络接收顺序。实际执行结果单独记录，不以本设计文档声称全部验证通过。
