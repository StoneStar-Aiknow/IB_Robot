# perception_service 节点说明

`perception_service` 是当前具身链路中的**连续场景理解包**。

它负责：

1. 持续监听相机、关节状态、末端位姿
2. 接收用户的连续文本提问或结构化请求
3. 将图像、机器人状态、用户补充信息一起发给大模型理解
4. 发布结构化理解结果和简短文本摘要
5. 可选运行 Grounding-DINO + SAM2 开放词汇检测与分割节点

## 1. 典型链路

```text
/camera/top/image_raw + /joint_states + /robot_status/ee_pose
  + 用户文本 / 用户 context_json
  -> perception_service_node
  -> /embodied/perception_result
  -> /embodied/perception_summary
```

如果启用了 RealSense / RGB-D，则当前链路也支持：

```text
/camera/front_camera/color/image_raw
  + /camera/front_camera/color/camera_info
  + /camera/front_camera/aligned_depth_to_color/image_raw
  + /camera/front_camera/depth/color/points
  + /joint_states
  + /robot_status/ee_pose
  -> perception_service_node
  -> 本地多模态模型
  -> /embodied/perception_result
```

## 2. 主要输入

### 2.1 当前支持的场景输入类型

现在 `perception_service_node` 不再只支持单张主相机图像，还支持：

| 输入 | 说明 |
| --- | --- |
| `primary_camera_topic` | 主视角 RGB 图像 |
| `wrist_camera_topic` | 末端 wrist 视角 RGB 图像，可选 |
| `primary_camera_info_topic` | 主视角内参，可选 |
| `primary_aligned_depth_topic` | 主视角对齐深度图，可选 |
| `primary_pointcloud_topic` | 主视角点云，可选 |
| `wrist_camera_info_topic` | wrist 视角内参，可选 |
| `wrist_aligned_depth_topic` | wrist 视角对齐深度图，可选 |
| `wrist_pointcloud_topic` | wrist 视角点云，可选 |

当只配置 `primary_camera_topic` 时，行为与旧版本保持一致。  
当额外配置 wrist / depth / pointcloud 时，节点会自动进入**多视角 + RGB-D**分析模式。

### 简单连续交互

```bash
ros2 topic pub --once /embodied/perception_text std_msgs/msg/String "{data: '看看桌面上有什么'}"
```

### 结构化请求

```bash
ros2 topic pub --once /embodied/perception_request ibrobot_msgs/msg/SceneAnalysisRequest \
  "{request_id: 'req-1', source: 'cli', session_id: 'demo', user_text: '判断红色物体是否适合抓取', context_json: '{\"focus_object\":\"red_block\",\"goal\":\"graspability\"}', timeout_sec: 120.0}"
```

### 视觉趣味游戏请求（分院帽等）

`perception_service_node` 作为通用视觉分析运行时，不感知具体游戏业务。入口层
（`embodied_agent/task_entry_node`）命中"分院帽"等触发词后，会构造一条带角色 prompt 的
`SceneAnalysisRequest`（`source=game.<name>`，如 `game.sorting_hat`）发到本节点。
本节点按普通 scene-analysis 请求处理：抓场景 → 调 VLM → 发 `SceneAnalysisResult`。结果消费方
通过 `source` 识别业务类型，`scene_summary` 保存最终结果，失败时以 `error_code`/`message` 表达。
请求 `context_json.required_inputs` 声明该请求真正需要的输入（`primary_image` / `ee_pose` /
`joint_state`），本节点据此判定哪些缺失才阻塞：分院帽只声明 `primary_image`，故 EE pose / joint
state 离线时仍可成功；未声明或畸形的 `required_inputs` 维持严格默认（三者全需在线）。
`context_json.response_contract` 声明输出契约；当前支持 `kind=enum`，在发布成功结果前校验指定字段
（如 `scene_summary`）严格属于 `allowed_values`。契约声明畸形、`kind` 缺失/不受支持或字段值越界时，
本节点发布 `success=false`、`error_code=INVALID_RESPONSE_CONTRACT`，并保留 `raw_response` 便于诊断。

## 3. 主要输出

| topic | 类型 | 说明 |
| --- | --- | --- |
| `/embodied/perception_result` | `ibrobot_msgs/msg/SceneAnalysisResult` | 结构化场景理解结果 |
| `/embodied/perception_summary` | `std_msgs/msg/String` | 面向人类快速查看的摘要 |

说明：

- `SceneAnalysisResult` 的 ROS 消息字段本身没有因为 RGB-D 接入而破坏兼容。
- RGB-D / 多视角的中间上下文会进入节点内部 `scene_snapshot` / prompt，不会影响原有调用方。

## 4. 连续交互设计

- `session_id` 用于区分会话
- 节点会保留最近若干轮问答摘要，作为下一轮理解的上下文
- `context_json` 可携带用户附加信息，例如：
  - 关注目标
  - 关注区域
  - 安全限制
  - 任务目标

## 5. 当前默认模型配置

- provider: `openai_compatible`
- base URL: `http://localhost:8000/v1`
- model: `Qwen3.5-9B`
- API key env: 可留空；若本地服务需要鉴权，再额外配置环境变量
- 大模型输出超时：按**输出空闲超时**计算，默认来自 `embodied.timeouts.model_idle_timeout_sec`
- `SceneAnalysisRequest.timeout_sec` 现在会直接覆盖本次请求的模型 idle timeout，不再被强制抬高到 `120s`

仍然保留原远端 Kimicode 调用能力；只需把配置改回：

- provider: `kimicode`
- base URL: `https://api.kimi.com/coding/v1`
- api_key_env: `KIMICODE_API_KEY`
- model: `kimi-for-coding`

## 6. RealSense / RGB-D 接入说明

### 6.1 推荐的 `scene_sources` 配置

若要让 `perception_service_node` 直接消费 RealSense 数据，推荐在 `robot_config` 中配置：

```yaml
embodied:
  timeouts:
    task_budget_sec: 180.0
    scene_freshness_sec: 0.5
    model_idle_timeout_sec: 120.0
    rpc_timeout_sec: 5.0
    gripper_settle_sec: 1.5
  perception:
    scene_sources:
      primary_camera_topic: /camera/front_camera/color/image_raw
      primary_camera_info_topic: /camera/front_camera/color/camera_info
      primary_aligned_depth_topic: /camera/front_camera/aligned_depth_to_color/image_raw
      primary_pointcloud_topic: /camera/front_camera/depth/color/points
      ee_pose_topic: /robot_status/ee_pose
      joint_state_topic: /joint_states
      require_depth: true
      require_pointcloud: false
```

### 6.2 当前已落地的能力

当前 RealSense 改造已经支持：

1. 主相机 RGB 图像分析
2. wrist 图像与主图**联合分析**
3. `camera_info` 摘要注入
4. aligned depth 的结构化深度摘要注入
5. pointcloud 元信息摘要注入
6. 基于 RGB-D 的距离 / 遮挡 / 近距离障碍风险判断

注意：当前不是把原始深度数组或点云直接发给大模型，而是先在本地做摘要，再注入 prompt。

### 6.3 运行时注意事项

1. RealSense 图像/深度/点云订阅已改为**传感器 QoS**，否则容易因 QoS 不匹配而收不到数据。
2. 若 `require_depth=true`，但没有收到有效 depth topic，请求会直接失败，而不是静默退化。
3. 实际单独跑 `realsense2_camera_node` 时，设备默认常见原始 topic 是：

```text
/camera/camera/color/image_raw
/camera/camera/color/camera_info
/camera/camera/aligned_depth_to_color/image_raw
/camera/camera/depth/color/points
```

如果希望统一走仓内推荐的 `/camera/front/...` 命名，应通过 `robot_config` launch 或显式 remap 做标准化。

## 7. Grounded-SAM2 检测分割

`grounded_sam2_node` 和 `grounded_sam2_snapshot` 已并入 `perception_service`，作为感知包中的开放词汇检测/分割能力。
抓取流水线默认使用腕部相机 topic：

- RGB：`/camera/wrist/image_raw`
- 对齐深度：`/camera/wrist/aligned_depth_to_color/image_raw`
- CameraInfo：`/camera/wrist/aligned_depth_to_color/camera_info`

首次使用前安装可选依赖并下载模型：

```bash
./scripts/setup.sh --with-perception
./scripts/download_perception_models.sh
```

构建并运行在线节点：

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select ibrobot_msgs perception_service
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 run perception_service grounded_sam2_node
```

该节点提供 `ibrobot_msgs/srv/DetectSegment`，并发布 `ibrobot_msgs/msg/DetectionArray`，供下游 `manipulation_service` 抓取规划消费。

## 8. Generic Model Services

`model_service_node` 是强类型模型服务的通用宿主。每个实例从 robot-config SSOT 读取 bundle、命名 deployment、
plugin class、具体 ROS service type、endpoint 和 runtime options。宿主不包含模型家族到 executable 的映射，也
不通过一个匿名 tensor service 暴露推理。

Plugin 必须实现 `ModelServicePlugin`：拥有具体 service request/response 映射，调用 adapter 和 shared
`ModelSession`，并投影统一的 runtime health。Adapter 拥有模型语义预处理和后处理；`inference_service` 拥有
设备生命周期、准入、tensor ABI、健康状态和资源回收。

每个响应中的 `ModelRuntimeInfo` 报告 instance ID、manifest fingerprint、deployment name/fingerprint、
backend、runtime state、readiness 和 failure reason。诊断身份来自节点配置和已验证 manifest，不允许 plugin
metadata 覆写，也不在启动时重新计算权重 SHA-256。

`perception_service.echo_adapter:EchoServicePlugin` 是唯一的通用层参考集成。它使用真实的
`ibrobot_msgs/srv/EchoModel` 契约以及无权重、无硬件 SDK 的
identity tensor session 验证 bundle -> runtime -> adapter -> typed response 链路，仅证明框架契约，不表示任何
生产模型或 accelerator deployment 已通过语义/硬件 conformance。生产服务类型、adapter、wrapper 和配置由各自
消费功能负责。

## 9. Semantic Mapping Model Services

新语义建图流水线使用五个独立的 generic `model_service_node` 进程：SAM2 mask、RAM++ tagging、SigLIP2
masked-image、SigLIP2 text 和可选 Grounding DINO confirmation。每个进程只承载一个 typed endpoint 和一个
named deployment；SigLIP2 image/text 不共享进程，但必须声明兼容的 embedding space。

每个 generic host 通过 response `ModelRuntimeInfo` 报告 readiness、semantic identity 和 deployment
provenance。Ascend OM 只能通过 schema-v2 manifest named deployment 进入 shared `AscendOmModelSession`；raw
wrapper 的 `backend=ascend_om` 路径不再提供。CUDA 不可用时必须在 SSOT 中选择另一个已经验证的 named
deployment，节点不会自动切换 backend。

感知模型与 ACT/PI0.5 使用相同的 bundle-first 结构。下载脚本直接生成四个 bundle 根目录：
`models/sam2.1_hiera_tiny/`、`models/ram_plus_swin_large_14m/`、
`models/siglip2_so400m_patch14_384/` 和 `models/grounded_sam2_swint_ogc/`。每个目录包含
`inference_manifest.json`、`assets/adapter.json`、模型资产和 `torch_cpu`/`torch_cuda` named deployment。
Grounding DINO Swin-T OGC 的网络结构配置由 `perception_service.grounding_dino_config` 常量绑定到软件版本，
不作为模型资产复制进 bundle；bundle 仅保存 checkpoint、文本编码器和 SAM2 checkpoint 等运行资产。

SigLIP2 image/text 服务共享同一 bundle 与 embedding identity，但仍由独立进程加载独立 session。RAM++ 仍从
`ram_models/recognize-anything/` 导入上游 `ram` 源码；源码不是模型资产。ONNX、OM 等未验证转换结果只能放在
bundle 的 `model_utils_work/`，通过 conformance 与 promotion 后才可复制到不可变
`artifacts/<backend>/<deployment>/generations/<uuid>/` 并注册为 named deployment。这些 service-backed mapping
资产不应与 legacy `grounded_sam2_node` 的 `DetectSegment` contract 混用。

经 ABI 审核的 compiled-only 资产由 `perception_service.package_ascend_perception_bundles` 从
`model_utils_work/candidates/ascend_*` 打包。SAM2 和 SigLIP2 提供 `ascend_310p`、`ascend_310b` deployment；
Grounding DINO 当前只提供固定 720x1280、文本长度 8 的 `ascend_310p` deployment。多 OM pipeline 的中间 tensor
通过 manifest `device_links` 保持在设备侧，service adapter 只负责模型语义预处理与最终输出后处理。

### 检测结果质心

每个 `Detection2D` 携带两种 3D 质心（相机光学系，单位 m）：

| 字段 | 计算方式 | 适用场景 |
|------|---------|---------|
| `centroid_xyz` | mask 内可见表面点云的**算术平均** | 快速估计、小扁平物体 |
| `volume_centroid_xyz` | 点云**凸包体积质心**（四面体分解 + 体积加权） | 凸形物体的物理中心近似 |
| `volume_m3` | 凸包体积 | 物体尺寸估计 |

**差异**：`centroid_xyz` 只反映相机能看到的正面，深度方向偏前；`volume_centroid_xyz` 把凸包"虚构背面"补回，质心沿夹爪 −Z 方向后移约 7~12 mm（取决于物体厚度），更接近真实物理中心。

**已知局限**：
- 弯曲物体（banana）：凸包质心可能侧向偏出物体轮廓（投影法 ~47% 落在 mask 外）
- 小圆物体（strawberry）：凸包背面空腔大，质心离实际表面 11~15 mm
- 凸形物体（marker、cucumber）：两种质心差异小，体积质心可靠

若需更准确的体积质心，应使用多视角点云融合补全背面，再做网格体积积分。
