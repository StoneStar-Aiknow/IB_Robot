# perception_service 节点说明

`perception_service` 是当前具身链路中的**连续场景理解包**。

它负责：

1. 持续监听相机、关节状态、末端位姿
2. 接收用户的连续文本提问或结构化请求
3. 将图像、机器人状态、用户补充信息一起发给大模型理解
4. 发布结构化理解结果和简短文本摘要

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
