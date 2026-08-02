# 观测视频流传输

## 概述

IB-Robot 支持将相机观测从 DDS tensor 传输切换为 H.264 RTP/UDP 视频流，降低分布式推理的网络带宽占用，同时保持时间戳同步和多相机对齐语义。

**架构**：
- 机器人侧（本机/传感器侧）：采样相机 topic，硬件或软件编码为 H.264，通过 RTP/UDP 推送到云端
- 云端（推理侧）：硬件或软件解码 H.264 流，按推理请求的时间戳从环形缓冲取帧，组装为 LeRobot observation
- DDS 仍承载：session 协商、timestamp mapping、heartbeat、request/result、非图像观测（如 `observation.state`）

**适用场景**：
- 分布式推理（`execution_mode: distributed`）
- 多相机高分辨率观测（如三相机 640×480@30fps）
- 云端或板端推理节点带宽受限

## 快速开始

### 1. 在 robot YAML 的 `contract.observations` 中配置 `transport`

**单臂双相机示例**（SO-101）：

```yaml
contract:
  observations:
    - key: observation.state
      topic: /joint_states
      type: sensor_msgs/msg/JointState
      # ... state observation 保持 DDS 传输

    - key: observation.images.top
      topic: /camera/top/image_raw
      type: sensor_msgs/msg/Image
      image: {resize: [480, 640], encoding: rgb8}
      align: {strategy: hold, stamp: header, tol_ms: 1500, max_age_ms: 500}
      transport:
        mode: rtp                          # 启用 RTP 视频流
        stream_id: top                     # 流标识，必须唯一
        endpoint: {host: 192.168.136.127, port: 55004}  # 云端接收地址
        codec: h264
        encoder_backend: nvidia            # 机器人侧编码器：nvidia | software
        decoder_backend: ascend            # 云端解码器：ascend | software
        h264: {profile: main, bitrate_bps: 4000000, gop_frames: 15}
        media: {width: 640, height: 480, frame_rate_hz: 20, pixel_format: nv12, color_space: bt709, color_range: limited}
        buffer: {sender_queue_frames: 2, receiver_queue_packets: 256, decoded_frame_capacity: 32, retention_ms: 1000}
        readiness: {keyframe_timeout_ms: 5000, timestamp_mapping_max_age_ms: 1000, max_inter_camera_skew_ms: 100}
        security: none

    - key: observation.images.wrist
      topic: /camera/wrist/image_raw
      type: sensor_msgs/msg/Image
      image: {resize: [480, 640], encoding: rgb8}
      align: {strategy: hold, stamp: header, tol_ms: 1500, max_age_ms: 500}
      transport:
        mode: rtp
        stream_id: wrist                   # 不同流使用不同 stream_id
        endpoint: {host: 192.168.136.127, port: 55006}  # 每路流独立端口
        codec: h264
        encoder_backend: nvidia
        decoder_backend: ascend
        h264: {profile: main, bitrate_bps: 4000000, gop_frames: 15}
        media: {width: 640, height: 480, frame_rate_hz: 20, pixel_format: nv12, color_space: bt709, color_range: limited}
        buffer: {sender_queue_frames: 2, receiver_queue_packets: 256, decoded_frame_capacity: 32, retention_ms: 1000}
        readiness: {keyframe_timeout_ms: 5000, timestamp_mapping_max_age_ms: 1000, max_inter_camera_skew_ms: 100}
        security: none
```

**关键字段说明**：

| 字段 | 说明 |
|------|------|
| `mode: rtp` | 切换到 RTP 视频流传输；`mode: dds` 或省略 `transport` 字段则继续用 DDS tensor |
| `stream_id` | 流标识符，同一 session 内必须唯一，用于 DDS descriptor/status 关联 |
| `endpoint.host` | 云端可绑定的 IP 地址；不能是 `127.0.0.1`（仅本机有效） |
| `endpoint.port` | 云端 UDP 接收端口；每路流使用不同偶数端口，系统同时保留 `port+1` |
| `encoder_backend` | 机器人侧编码器：`nvidia`（NVIDIA GPU NVENC）或 `software`（libx264，CPU） |
| `decoder_backend` | 云端解码器：`ascend`（Ascend NPU DVPP）或 `software`（FFmpeg CPU） |
| `h264.bitrate_bps` | H.264 码率（bps），建议 2-6 Mbps；过低影响推理精度，过高占用带宽 |
| `h264.gop_frames` | GOP 长度（I 帧间隔），建议 10-20；过短增加码率，过长延长冷启动 |
| `readiness.max_inter_camera_skew_ms` | 多相机同步容差（默认 100ms）；超过则拒绝推理请求 |

### 2. 机器人侧和云端必须使用相同配置

**Contract fingerprint 校验**：机器人侧和云端在启动时根据 `contract` 和 `deployment` 计算 fingerprint，两者必须完全一致才能协商成功。修改任何 transport 参数后，需同步更新两端配置并重启。

### 3. 启动分布式推理

**云端**（openEuler Embedded aarch64 板端）：

```bash
cd /IB_Robot
source .shrc_local
source /path/to/video_workspace/install/setup.zsh  # 包含视频流代码的 workspace

# Ascend FFmpeg 环境（若使用 ascend decoder）
export IBROBOT_ASCEND_FFMPEG=/path/to/ffmpeg-ascend/bin/ffmpeg
export IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV=1
export LD_LIBRARY_PATH=/path/to/ffmpeg-ascend/lib:$LD_LIBRARY_PATH

# 启动云端推理节点
python3 -m inference_service.pure_inference_node --ros-args \
  -p pipeline_id:=policy \
  -p model_path:=/path/to/model_bundle \
  -p deployment:=ascend-310b1 \
  -p request_timeout:=120.0 \
  -p robot_config_path:=/path/to/cloud_robot.yaml
```

**机器人侧**（Ubuntu 22.04 主机）：

```bash
source .shrc_local

# 启动机器人硬件 + 相机 + 分布式推理
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  config_path:=/path/to/edge_robot.yaml \
  control_mode:=model_inference
```

### 4. 验证视频流状态

**检查 descriptor 协商**：

```bash
ros2 topic echo /inference/policy/video/descriptors --once
```

预期输出包含每路流的 descriptor，`session_id` 和 `session_generation` 必须匹配。

**检查运行时状态**：

```bash
ros2 topic hz /inference/policy/video/status
```

预期输出每路流的实时指标：

```yaml
stream_id: top
lifecycle_state: ready
ready: true
timestamp_mapping_valid: true
keyframe_ready: true
encoded_frames: 1234      # 机器人侧编码帧数
decoded_frames: 1200      # 云端解码帧数
sent_packets: 28000       # 机器人侧发送 RTP 包数
received_packets: 27800   # 云端接收 RTP 包数
lost_packets: 200         # 丢包数（UDP 不保证可靠）
dropped_packets: 0        # 当前缓冲区溢出丢弃数
```

**触发推理**：

```bash
ros2 action send_goal /inference/policy/dispatch ibrobot_msgs/action/DispatchInfer "{prompt: 'pick up the banana'}"
```

成功返回 `chunk_size: 100` 且 DDS `tensors` 中只包含 `observation.state`（不含图像），证明图像通过 RTP 传输并成功组装。

## 编码器和解码器后端

### NVIDIA 编码器（`nvidia`）

**依赖**：
- NVIDIA GPU（支持 NVENC，如 RTX 3090 / RTX 4090 / T4）
- NVIDIA 驱动（提供 `libnvidia-encode.so`，如 nvidia-driver-550+）
- PyAV 15.1.0+（PyPI wheel 自带 NVENC 支持，无需编译 FFmpeg）

**特点**：
- 硬件加速 H.264 编码，CPU 占用低
- 适合机器人侧主机有 NVIDIA GPU 的场景
- 编码质量和码率控制优于 software（libx264）

**验证**：

```bash
python3 -c "import av; print(av.codec.codecs_available); print('h264_nvenc' in av.codec.codecs_available)"
```

**注意**：PyAV wheel 自带的 libavcodec 与系统 `ffmpeg` 命令独立。`ffmpeg -encoders | grep nvenc` 无结果不影响 PyAV 使用 NVENC。

### Software 编码器（`software`）

**依赖**：
- PyAV 15.1.0+（PyPI wheel 自带 libx264 支持）

**特点**：
- CPU 软件编码，无需 GPU
- 编码质量和压缩率优于 NVENC（相同码率下）
- CPU 占用较高，适合机器人侧主机 CPU 充裕的场景

### Ascend 解码器（`ascend`）

**依赖**：
- Ascend 310P / 310B NPU
- 隔离的 FFmpeg（自带 CANN `h264_ascend` 插件，不要替换系统 FFmpeg）；实测 4.4.2 与 6.1.1 均可用
- openEuler Embedded aarch64 或 Ubuntu aarch64

**环境变量**：

```bash
# Orange Pi AI Pro 20T (openEuler Embedded) 示例
export IBROBOT_ASCEND_FFMPEG=/home/HwHiAiUser/ffmpeg-ascend-cann83/install/bin/ffmpeg
export IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV=1  # 隔离 LD_LIBRARY_PATH，防止污染系统 FFmpeg
export LD_LIBRARY_PATH=/home/HwHiAiUser/ffmpeg-ascend-cann83/install/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:$LD_LIBRARY_PATH
```

**特点**：
- 硬件加速 H.264 解码，CPU 占用极低
- 多路流需分配不同 `channel_id`（自动按 `observation_key` 排序分配，从 1 开始）
- 适合云端推理节点在 Ascend 板端的场景

**实现方式**：与 `nvidia` / `software` 后端不同，Ascend 后端不通过 PyAV 调用。`h264_ascend` 是 CANN 私有编解码器，不在上游 FFmpeg 中，PyAV wheel 自带的 libavcodec 没有该 codec，也无法承载其 `device_id` / `channel_id` 设备语义。因此 Ascend 后端以子进程方式驱动隔离的 Ascend FFmpeg 二进制，通过管道和本地 UDP 交换裸帧与 H.264 码流，把 CANN 运行时依赖限制在子进程内，不影响 Python 进程和系统 FFmpeg。

**验证**：

```bash
$IBROBOT_ASCEND_FFMPEG -codecs 2>/dev/null | grep h264_ascend
```

### Software 解码器（`software`）

**依赖**：
- PyAV 15.1.0+
- FFmpeg 标准 H.264 解码器

**特点**：
- CPU 软件解码，无需专用硬件
- 适合云端推理节点 CPU 充裕或无硬件加速的场景

## 时间戳同步机制

视频流传输后，推理请求仍需指定目标时间戳，云端根据时间戳从缓冲区取帧。同步机制如下：

### 1. 机器人侧建立 RTP ↔ ROS 时间映射

机器人侧每秒通过 DDS 发布 `VideoStreamStatus`，包含：

```yaml
mapping_rtp_timestamp: 2194729234          # 当前 RTP 时间戳（90kHz 时钟）
mapping_capture_time:                      # 对应的 ROS 采集时间
  sec: 1785490012
  nanosec: 686427000
```

这个映射关系建立了 RTP 时间域和 ROS 时间域之间的锚点。

### 2. 推理请求带目标 ROS 时间

机器人侧在发送推理请求时，从 DDS 获取 `/joint_states` 的时间戳，作为 `observation_timestamp` 发送给云端：

```yaml
DistributedInferenceRequest:
  observation_timestamp:
    sec: 1785490015
    nanosec: 123456789
  stream_observation_keys: ["observation.images.top", "observation.images.wrist"]
  stream_ids: ["top", "wrist"]
  tensors: [observation.state]  # 只有 joint state 走 DDS
```

### 3. 云端逆向计算并取帧

云端收到推理请求后：

1. **提取 RTP 时间戳**：从环形缓冲区的每个解码帧提取 RTP 时间戳（来自 RTP 包头）
2. **转换回 ROS 时间**：用最新的 mapping 关系将 RTP 时间戳转回 ROS 时间：
   ```python
   ros_time_ns = mapping_capture_time_ns + (rtp_timestamp - mapping_rtp_timestamp) * 1e9 / 90000
   ```
3. **选择最接近帧**：找到 ROS 时间最接近 `observation_timestamp` 的帧（允许容差 `max_age_ms`，默认 500ms）
4. **多相机同步校验**：检查所有相机的帧时间戳偏差是否 < `max_inter_camera_skew_ms`（默认 100ms）
5. **组装观测**：若同步成功，返回多相机对齐的观测；否则拒绝推理请求并报错 `observation_not_ready`

### 4. 失败模式

| 失败原因 | 错误码 | 说明 |
|----------|--------|------|
| `unmapped` | `observation_not_ready` | 云端尚未收到机器人侧的 timestamp mapping |
| `stale` | `observation_not_ready` | Mapping 超过 `max_mapping_age_ms`（默认 1000ms），可能机器人侧断连 |
| `pre_keyframe` | `observation_not_ready` | 云端尚未收到首个 I 帧（GOP 起始），等待中 |
| `missing` | `observation_not_ready` | 缓冲区中没有足够接近目标时间戳的帧（可能丢包或帧率不足） |
| `skewed` | `observation_not_ready` | 多相机时间偏差超过 `max_inter_camera_skew_ms`，无法对齐 |

所有 `observation_not_ready` 错误都是 **recoverable**，机器人侧会自动重试。若持续失败，检查：
- 机器人侧和云端的 `ROS_DOMAIN_ID` 是否一致
- UDP 端口是否被占用或防火墙拦截
- 网络丢包率是否过高（`lost_packets` 持续增长）
- Camera topic 发布频率是否低于 `media.frame_rate_hz`

## 真机验证结果

### 测试配置

- **Robot**：SO-101 单臂（6 关节）+ 双 DECXIN 相机（640×480@30fps MJPG）
- **机器人侧**：Ubuntu 22.04，RTX 3090，ROS 2 Humble
- **云端**：openEuler Embedded aarch64，Ascend 310B1，ROS 2 Humble
- **Model**：ACT 1-arm 2-cam banana pick（distilled，160k steps）
- **Network**：192.168.136.0/24 千兆有线直连

### OM (ascend-310b1) 部署验证

**配置**：
- Encoder: `nvidia`（机器人侧 RTX 3090 NVENC）
- Decoder: `ascend`（云端 310B1 DVPP，channel_id=1/2）
- Bitrate: 4 Mbps，GOP 15 frames

**结果**：
- Session 协商：2 路流 descriptor 匹配，fingerprint 一致
- 视频传输：`top` 发送 6221 包 / 接收 5980 包，`wrist` 发送 5781 包 / 接收 5467 包
- 解码稳定：`top` 解码 144 帧，`wrist` 解码 120 帧
- 丢包恢复：启动时 `lost_packets=138/104`，触发 `pre_keyframe` 等待下一个 I 帧，成功恢复后 `dropped_packets=0`
- 推理成功：返回 `chunk_size=100`，DDS tensor 中只有 `observation.state`（0.024 KB），图像不在 DDS（节省 ~1.8 MB/request）

### Torch-NPU 部署验证

**配置**：
- Encoder: `nvidia`
- Decoder: `ascend`
- Bitrate: 4 Mbps，GOP 15 frames

**结果**：
- 模型加载：首次加载耗时约 5 分钟（含 NPU graph 编译和算子初始化）
- 视频流：双路流协商和解码正常
- 首次推理：超过 120 秒超时（NPU 算子首次执行需 JIT 编译），后续推理需测试更长超时

### 发现并修复的 Bug

在真机验证中发现并修复 5 个关键 bug：

1. **QoS depth 冲突**：机器人侧和云端的 descriptor 订阅共享了 heartbeat 的 `depth=1` QoS，导致多路流的 descriptor 互相覆盖。修复：分离 descriptor QoS（`depth=流数量`）。

2. **并发 descriptor 竞态**：云端收到多路流 descriptor 时，多个线程同时调用 receiver 创建，导致重复创建和端口冲突。修复：加 `_receiver_start_lock` 和已创建检查。

3. **Ascend decoder channel 冲突**：两路流都使用 `channel_id=0`，争用同一硬件解码通道。修复：按 `observation_key` 排序分配唯一 channel_id（从 1 开始）。

4. **RTP 丢包后 reset 循环**：丢包后调用 `decoder.reset()` 重启 FFmpeg，期间队列溢出触发新 gap，再次 reset，形成不可恢复循环。修复：移除 reset 调用，等待下一组 SPS/PPS/IDR 自然恢复。

5. **Stale descriptor 污染**：DDS `TRANSIENT_LOCAL` 缓存的旧 session descriptor 在新 session 启动时被错误接受，导致协商失败。修复：增加 session envelope 前置校验，直接拒绝 stale descriptor。

所有修复已通过单元测试（36/36 passed）和真机 E2E 验证。

## 安全和限制

### 当前限制

- **无认证/加密**：RTP/UDP 明文传输，无认证、完整性保护或机密性保障，仅适用于可信机器人网络（如内网直连、VPN 或物理隔离网络）
- **丢包不重传**：UDP 不保证可靠传输，网络丢包会导致 `lost_packets` 增加；解码器在收到下一个 I 帧前可能无法恢复
- **录制不完整**：当前 rosbag/MCAP 录制仍基于原始 DDS image topic，不录制 RTP payload；需要视频流时，必须同时录制 DDS 模式或使用专用 RTP 录制工具

### 安全建议

- 仅在可信网络中使用（内网、VPN、物理隔离）
- 不要在公网或不可信 WiFi 上使用 `security: none`
- 规划独立的机器人控制网段，与办公网络隔离
- 定期检查 cloud 的 UDP 端口访问策略（如 iptables / firewall）

### 回滚到 DDS 传输

需要回滚时：

1. 在机器人侧和云端的 robot YAML 中将 `transport.mode: rtp` 改为 `transport.mode: dds`（或完全删除 `transport` 字段）
2. 同步重启机器人侧和云端节点
3. 验证 contract fingerprint 重新匹配

**不要通过删除 `transport` 字段临时触发隐式回退**；回滚应显式配置 `mode: dds` 并确保两端 contract fingerprint 一致。

## 故障排查

### Descriptor 协商失败

**症状**：云端日志报 `invalid video stream descriptor: does not match the active session` 或 `contract fingerprint mismatch`

**原因**：
- 机器人侧和云端的 `contract` 配置不一致（如分辨率、码率、GOP 不同）
- 机器人侧和云端的 `deployment` 不一致
- DDS `TRANSIENT_LOCAL` 缓存了旧 session 的 descriptor

**解决**：
1. 确认机器人侧和云端使用相同的 robot YAML 或 contract 配置
2. 同时停止机器人侧和云端，等待 5 秒后再启动（清空 DDS 缓存）
3. 检查启动日志中的 `contract_fingerprint` 和 `deployment_fingerprint` 是否一致

### 持续 `pre_keyframe` 或 `missing`

**症状**：推理请求持续失败，错误为 `observation_not_ready: pre_keyframe` 或 `missing`

**原因**：
- 云端未收到 I 帧（可能 GOP 过长或网络丢包严重）
- 机器人侧的相机发布频率低于 `media.frame_rate_hz`
- UDP 端口被占用或防火墙拦截

**解决**：
1. 检查云端日志中的 `received_packets` 是否持续增长（若为 0，说明 UDP 不通）
2. 检查 `lost_packets` 增长速度（若过快，降低码率或缩短 GOP）
3. 在云端主机上验证 UDP 端口可用：
   ```bash
   sudo netstat -ulnp | grep 55004
   # 或
   sudo lsof -i UDP:55004
   ```
4. 检查机器人侧相机 topic 的实际发布频率：
   ```bash
   ros2 topic hz /camera/top/image_raw
   ```

### 多相机 `skewed`

**症状**：推理请求失败，错误为 `observation_not_ready: skewed`

**原因**：多相机采集时间偏差超过 `max_inter_camera_skew_ms`（默认 100ms）

**解决**：
1. 检查相机硬件时间戳是否同步（如使用外部触发或 PTP 时钟同步）
2. 增大 `readiness.max_inter_camera_skew_ms`（如改为 200ms），但过大会影响动作精度
3. 检查相机驱动的时间戳来源（`image_raw.header.stamp` 应为采集时刻，不是发布时刻）

### 解码器启动失败

**症状**：云端日志报 `backend 'ascend' failed to load` 或 `FFmpeg process exited`

**原因**：
- `IBROBOT_ASCEND_FFMPEG` 未设置或路径错误
- FFmpeg 缺少 `h264_ascend` 插件
- Ascend NPU 驱动未加载

**解决**：
1. 验证 Ascend FFmpeg：
   ```bash
   $IBROBOT_ASCEND_FFMPEG -codecs 2>/dev/null | grep h264_ascend
   ```
2. 检查 CANN 驱动：
   ```bash
   npu-smi info
   ```
3. 确认环境变量：
   ```bash
   echo $IBROBOT_ASCEND_FFMPEG
   echo $LD_LIBRARY_PATH | grep ffmpeg-ascend
   ```

## 性能和带宽

### 带宽占用（实测）

**DDS tensor 模式**（baseline）：
- 单相机 640×480 RGB：~0.9 MB/frame = 18 MB/s @ 20Hz
- 双相机：~36 MB/s
- 三相机：~54 MB/s

**RTP 视频流模式**（4 Mbps H.264）：
- 单相机：~0.5 MB/s（压缩比 36:1）
- 双相机：~1.0 MB/s（节省 97%）
- 三相机：~1.5 MB/s（节省 97%）

**控制面开销**（DDS）：
- Descriptor：~1 KB/session
- Status：~0.5 KB/s/stream
- Request：~0.2 KB/request（不含图像 tensor）
- Result：~2.4 KB/response（100-step action chunk）

### 延迟（实测）

**编码延迟**（机器人侧）：
- NVIDIA NVENC：~5 ms/frame @ 640×480
- Software (libx264)：未在本次真机验证中测量（主机为 Intel i7-14700）

**解码延迟**（云端）：
- Ascend DVPP：~3 ms/frame @ 640×480
- Software (FFmpeg)：未在本次真机验证中测量（板端为 Orange Pi AI Pro 20T，Cortex-A73 4 核）

**网络延迟**（千兆有线直连）：
- RTT：~0.5 ms
- Jitter：~0.1 ms

**端到端延迟**（采集 → 推理 → 动作）：
- DDS tensor：~50 ms（baseline）
- RTP 视频流：~60 ms（+10 ms 编解码开销）

对于 20 Hz 控制频率（50 ms/cycle），RTP 增加的延迟在可接受范围内。

## 参考配置

完整的 SO-101 单臂双相机配置示例见：
- 机器人侧 YAML: `src/robot_config/config/robots/dev_rtp_single_camera.yaml`
- 云端 YAML: `src/robot_config/config/robots/dev_rtp_multi_camera.yaml`（示例中为多相机，可简化为双相机）

多相机配置示例（三相机）见：
- `src/robot_config/config/robots/dev_rtp_multi_camera.yaml`

## 相关文档

- [分布式推理架构](../src/inference_service/README.md)
- [Robot Config 契约系统](../src/robot_config/README.md)
- [视频编解码后端开发](../src/inference_service/inference_service/video_codec.py)
- [RTP 协议实现](../src/inference_service/inference_service/video_rtp.py)
- [时间戳同步](../src/inference_service/inference_service/observation_sync.py)
