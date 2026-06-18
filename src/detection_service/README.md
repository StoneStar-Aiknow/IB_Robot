# detection_service

`detection_service` 提供开放词汇目标检测与分割能力，当前实现为
Grounding-DINO + SAM2。该包只负责感知模型的 ROS 2 封装、消息转换、
RGB-D 输入读取、检测结果发布和调试快照导出，不负责机器人控制、抓取规划、
策略推理或数据集转换。

## 入口与接口

节点/工具：

- `grounded_sam2_node`：在线节点。订阅 RGB、对齐深度和 CameraInfo，
  提供 `~/detect_and_segment` 服务，并发布 `~/detections`。
- `grounded_sam2_snapshot`：调试工具。调用在线检测服务，保存输入图、
  overlay、mask、点云、`result.json` 和 `index.html`。

ROS 接口：

- 服务：`ibrobot_msgs/srv/DetectSegment`
- 话题：`ibrobot_msgs/msg/DetectionArray`
- 单个检测结果：`ibrobot_msgs/msg/Detection2D`

默认在线话题遵循 `robot_config` 的 front RGB-D camera 约定：

- RGB：`/camera/front/image_raw`
- 对齐深度：`/camera/front/aligned_depth_to_color/image_raw`
- CameraInfo：`/camera/front/camera_info`

RealSense 顶置相机调试时常用的话题为：

- RGB：`/camera/camera/color/image_raw`
- 对齐深度：`/camera/camera/aligned_depth_to_color/image_raw`
- CameraInfo：`/camera/camera/color/camera_info`

## 环境与依赖

首次使用感知模型时安装依赖和权重：

```bash
./scripts/setup.sh --with-detection
```

已有 Python 环境时可单独安装：

```bash
python3 -m pip install -r requirements/detection.txt
./scripts/download_detection_models.sh
```

模型文件默认从 `models/perception/` 解析。Grounding-DINO 的 BERT 文本编码器
优先使用 `models/perception/bert-base-uncased`，不存在时才使用参数中的模型名。

所有 ROS 调试命令都应在仓库根目录运行，并在同一条命令里完成环境初始化：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && <ros2 command>
```

修改代码或首次运行前，先构建接口和本包：

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select ibrobot_msgs detection_service
```

## 调试 grounded_sam2_node

### 1. 单独运行 Grounding-DINO + SAM2 在线节点

使用默认 front camera 话题：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run detection_service grounded_sam2_node
```

使用 RealSense 顶置相机话题：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run detection_service grounded_sam2_node --ros-args \
  -p rgb_topic:=/camera/camera/color/image_raw \
  -p depth_topic:=/camera/camera/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/camera/color/camera_info
```

常用参数：

- `rgb_topic`：RGB 图像输入。检测和分割基于该图像。
- `depth_topic`：与 RGB 对齐的深度图。用于 3D centroid 和点云调试信息。
- `camera_info_topic`：与 RGB/深度对应的内参。点云重建和 3D 坐标依赖它。
- `device`：模型推理设备，默认 `cuda`；无 GPU 时可改为 `cpu`，速度会明显下降。
- `model_dir`：模型根目录，默认空字符串，表示使用 `models/perception/`。
- `sam_checkpoint`：SAM2 权重文件名，默认 `sam2.1_hiera_tiny.pt`。
- `sam_config`：SAM2 配置路径，默认 `configs/sam2.1/sam2.1_hiera_t.yaml`。
- `gdino_config`：Grounding-DINO 配置文件，默认 `GroundingDINO_SwinT_OGC.py`。
- `gdino_checkpoint`：Grounding-DINO 权重，默认 `groundingdino_swint_ogc.pth`。
- `gdino_text_encoder`：文本编码器模型名或本地目录，默认 `bert-base-uncased`。
- `box_threshold`：Grounding-DINO box 置信度阈值，默认 `0.35`。
- `text_threshold`：Grounding-DINO 文本匹配阈值，默认 `0.25`。

### 2. 直接调用检测服务

节点启动后，可用 service call 验证 prompt 是否能检测到目标：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 service call /grounded_sam2/detect_and_segment ibrobot_msgs/srv/DetectSegment "{text_prompt: 'banana', confidence_threshold: 0.25}"
```

请求字段：

- `text_prompt`：自然语言目标名，例如 `banana`、`cup`、`person`。
- `confidence_threshold`：本次请求临时覆盖 `box_threshold`；设为 `0` 时使用节点参数默认值。

响应重点看：

- `success` / `message`：是否检测成功以及失败原因。
- `detections.detections[*].confidence`：每个检测框的置信度。
- `detections.detections[*].bbox_xyxy`：像素坐标框。
- `detections.detections[*].mask`：二值分割 mask，header 使用源 RGB 时间戳和 frame。

### 3. 订阅检测结果

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 topic echo /grounded_sam2/detections
```

该话题发布最近一次服务调用的检测结果，便于检查 bbox、mask header、frame_id 和置信度。

## 调试 grounded_sam2_snapshot

`grounded_sam2_snapshot` 适合保存一次完整感知结果，确认 RGB、mask、深度点云和
CameraInfo 是否对齐。运行前必须先启动 `grounded_sam2_node`。

RealSense 顶置相机示例：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && ros2 run detection_service grounded_sam2_snapshot \
  --prompt "banana" \
  --confidence-threshold 0.10 \
  --rgb-topic /camera/camera/color/image_raw \
  --depth-topic /camera/camera/aligned_depth_to_color/image_raw \
  --camera-info-topic /camera/camera/color/camera_info \
  --out-dir outputs/grounded_sam2
```

常用参数：

- `--prompt`：检测目标，必填。
- `--confidence-threshold`：本次 service call 的 box 阈值；`0` 表示使用在线节点默认值。
- `--service`：检测服务名，默认 `/grounded_sam2/detect_and_segment`。
- `--rgb-topic`：保存为 `input.png` 的 RGB 图像来源。
- `--depth-topic`：点云导出使用的对齐深度图。
- `--camera-info-topic`：与 RGB/深度匹配的内参。
- `--out-dir`：快照输出根目录；每次运行会新建时间戳子目录。
- `--image-timeout`：等待 RGB 图像的秒数，默认 `10.0`。
- `--service-timeout`：等待 Grounded-SAM2 推理结果的秒数，默认 `240.0`。
- `--alpha`：overlay 中 mask 透明度，默认 `0.45`。
- `--show`：保存后打开 OpenCV 窗口；无桌面环境时不要加。
- `--no-pointcloud`：跳过深度图和 PLY 点云导出。
- `--depth-timeout`：等待深度图和 CameraInfo 的秒数，默认 `10.0`。
- `--depth-scale`：整数深度转米的比例，RealSense `Z16` 通常为 `1000.0`。
- `--depth-trunc`：丢弃超过该距离的深度点，单位米，默认 `3.0`。
- `--max-full-cloud-points`：`cloud_full.ply` 最大点数；`0` 表示保存全部有效点。
- `--max-object-cloud-points`：每个目标点云最大点数；`0` 表示保存全部目标点。

输出目录包含：

- `input.png`：本次调试使用的 RGB 图。
- `overlay.png`：bbox 和 mask 叠加图。
- `comparison.png`：输入图与 overlay 对比图。
- `mask_*.png`：每个检测目标的二值 mask。
- `depth.png` / `depth_raw.npy`：深度预览和原始深度数组。
- `cloud_full.ply`：由对齐深度重建的整帧彩色点云。
- `cloud_*.ply`：每个检测 mask 对应的目标点云。
- `result.json`：prompt、置信度、bbox、mask、CameraInfo、点云和 3D centroid 元数据。
- `index.html`：浏览器查看用的摘要页面。

## 常见调试路径

### 只验证 Grounding-DINO + SAM2 是否能识别目标

1. 启动 `grounded_sam2_node`。
2. 调用 `/grounded_sam2/detect_and_segment`。
3. 如果没有结果，先降低 `confidence_threshold` 到 `0.10`，再检查 prompt 是否具体。

### 验证 RGB、深度和内参是否对齐

1. 启动 `grounded_sam2_node`。
2. 运行 `grounded_sam2_snapshot`，不要加 `--no-pointcloud`。
3. 查看 `overlay.png`、`cloud_*.ply` 和 `result.json` 中的 `camera_info`。

### 给 grasp_service 准备离线输入

1. 使用 `grounded_sam2_snapshot --prompt <object>` 保存一次结果。
2. 记录输出目录，例如 `outputs/grounded_sam2/<timestamp>_banana/`。
3. 将该目录作为 `grasp_service/test_graspgen.py --data-dir` 输入。

## 排障

- `DetectSegment service is not available`：`grounded_sam2_node` 未启动，或
  `ROS_DOMAIN_ID` 不一致。
- `No RGB frame received`：RGB topic 名不对，或相机驱动未发布图像。
- 有 bbox 但 mask 不准：提高 prompt 具体性，或调整 `box_threshold` /
  `confidence_threshold`。
- 点云为空或偏移：检查 `depth_topic` 是否为 aligned depth，且
  `camera_info_topic` 是否来自同一 RGB 相机。
- `ModuleNotFoundError`：未执行 `source .shrc_local`，或没有安装 perception 依赖。

## 架构边界

- 机器人拓扑、相机挂载和默认话题应保留在 `robot_config` YAML 中。
- 本包不拥有机器人控制、抓取规划、策略推理、动作执行或数据集转换逻辑。
- 第三方模型源码应作为安装依赖使用；若后续引入本地 patch stack，需要在
  `third_party/` 下单独记录。
