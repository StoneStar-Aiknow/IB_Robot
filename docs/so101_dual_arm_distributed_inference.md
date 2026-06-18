# SO-101 双臂工作流与分布式推理配置

本文说明 `so101_dual_arm` 的使用路径：双臂校准、遥操作、数据采集、数据转换，以及双机/板端分布式推理时 robot YAML 应如何配置、本机和板端命令如何下发。

典型分布式推理部署如下：

- 本机/edge host：连接双臂硬件和相机，启动 `robot_config robot.launch.py`，负责采集观测、前处理、动作后处理和控制器下发。
- 板端/cloud host：启动 `inference_service cloud_inference.launch.py`，负责加载模型并执行纯推理。

两端通过 ROS 2 topic 通信：

```text
本机 act_inference_node -> /preprocessed/batch -> 板端 pure_inference_cloud
本机 act_inference_node <- /inference/action     <- 板端 pure_inference_cloud
```

## 机械臂校准

先分别校准双臂 follower 和 leader。以下端口为示例，应按现场实际 `/dev/ttyACM*` 枚举结果调整：

```bash
# 左从臂
ros2 run so101_hardware calibrate_arm \
  --arm follower \
  --port /dev/ttyACM0 \
  --calib-file ~/.calibrate/so101_follower_left_calibrate.json

# 右从臂
ros2 run so101_hardware calibrate_arm \
  --arm follower \
  --port /dev/ttyACM1 \
  --calib-file ~/.calibrate/so101_follower_right_calibrate.json

# 左主臂
ros2 run so101_hardware calibrate_arm \
  --arm leader \
  --port /dev/ttyACM2 \
  --calib-file ~/.calibrate/so101_leader_left_calibrate.json

# 右主臂
ros2 run so101_hardware calibrate_arm \
  --arm leader \
  --port /dev/ttyACM3 \
  --calib-file ~/.calibrate/so101_leader_right_calibrate.json
```

校准文件会被 `so101_dual_arm.yaml` 复用，不需要额外生成合并标定文件。

## 遥操作启动

启动双臂遥操作：

```bash
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_dual_arm \
  control_mode:=teleop
```

可另开终端查看相机：

```bash
rviz2
# 或
ros2 run rqt_image_view rqt_image_view
```

## 数据采集

启动双臂遥操作并开启 episode 录制：

```bash
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_dual_arm \
  control_mode:=teleop \
  record:=true \
  record_mode:=episodic \
  record_visualizer:=rerun
```

另开终端触发和停止单条 episode：

```bash
ros2 run dataset_tools record_cli
```

录制数据默认保存到 YAML 的 `recording.bag_base_dir`，当前配置为 `~/rosbag/episodes`。双臂数据通常位于该目录下的 `so101_dual_arm` 子目录。

## 数据转换

将 ROS bag 转换为 LeRobot 数据集：

```bash
ros2 run dataset_tools bag_to_lerobot \
  --bags-dir ~/rosbag/episodes/so101_dual_arm \
  --robot-config $(ros2 pkg prefix robot_config)/share/robot_config/config/robots/so101_dual_arm.yaml \
  --out ~/datasets/so101_dual_arm_v1
```

转换时会读取 robot YAML 的 contract 和双臂标定配置，用于生成 LeRobot 数据和动作/关节转换 metadata。

## 网络环境

分布式推理时，两端必须在同一 ROS 2 domain 内，并允许跨机器发现：

```bash
export ROS_DOMAIN_ID=<domain-id>
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

如果环境里安装并验证过 CycloneDDS，也可以两端统一使用 `rmw_cyclonedds_cpp`。不要两端混用不同 RMW。

## YAML 配置

双臂配置文件位于：

```text
src/robot_config/config/robots/so101_dual_arm.yaml
```

### 分布式推理入口

`control_modes.model_inference.inference` 描述本机/edge 侧如何连接板端推理服务：

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      model: so101_act
      execution_mode: distributed
      cloud_inference_topic: /preprocessed/batch
      cloud_result_topic: /inference/action
      request_timeout: 10.0
    executor:
      type: topic
      mode: model_inference
      control_frequency: 30.0
      watermark_threshold: 40
      min_queue_size: 0
```

字段含义：

| 字段 | 本机/edge 侧含义 | 板端/cloud 侧要求 |
|------|------------------|-------------------|
| `execution_mode` | 必须为 `distributed`，本机只做前/后处理并通过 topic 调用板端推理 | 板端不读这份 robot YAML，启动 `cloud_inference.launch.py` 即可 |
| `cloud_inference_topic` | 本机发布预处理 batch 的 topic | 板端订阅 topic 必须一致，默认 `/preprocessed/batch` |
| `cloud_result_topic` | 本机订阅动作结果的 topic | 板端发布 topic 必须一致，默认 `/inference/action` |
| `request_timeout` | 本机等待板端推理结果的超时时间 | 应高于板端模型耗时和网络抖动 |
| `control_frequency` | 本机 action dispatcher 播放动作频率 | 不影响板端模型推理速度 |
| `watermark_threshold` | 本机开始播放前缓存的动作队列水位 | 应覆盖端到端延迟，例如 30 Hz 下 `40` 约为 1.33 秒缓存 |
| `min_queue_size` | 播放时允许的最小队列水位 | 双臂实机验证中可设为 `0`，避免高延迟下长期 hold |

如果修改 topic 名称，必须本机 YAML 和板端 launch 参数同时修改。

### 本机模型配置

`models.so101_act` 是本机/edge 侧读取模型 metadata 的位置。即使推理在板端执行，本机仍需要读取 LeRobot 配置、统计信息和输入输出 contract：

```yaml
models:
  so101_act:
    path: ./models/dual_arm/pretrained_model
    policy_type: act
    device: cpu
    lerobot_norm_mode: range_m100_100
```

建议本机使用仓库相对路径，避免把个人绝对路径写进 YAML。该路径必须在本机/edge host 上存在，并至少包含策略 metadata（例如 `config.json` 和统计信息），因为 edge 侧需要读取输入输出契约和归一化配置。板端模型目录不在这里配置，而是在板端启动命令里通过 `policy_path:=...` 指定。

### 观测 contract

编译模型的 `config.json` 中的 `input_features` 是最终契约。当前双臂 ACT 模型要求图像键为：

```yaml
contract:
  observations:
    - key: observation.images.top
      topic: /camera/top/image_raw
    - key: observation.images.left
      topic: /camera/left_wrist/image_raw
    - key: observation.images.right
      topic: /camera/right_wrist/image_raw
    - key: observation.state
      topic: /joint_states
```

注意：topic 名可以仍是 `/camera/left_wrist/image_raw`，但 contract key 必须匹配模型，例如 `observation.images.left`，不能写成 `observation.images.left_wrist`。

## 分布式推理启动命令

以下示例使用占位变量：

```bash
export IB_ROBOT_WS=<IB_Robot 工作区路径>
export BOARD_MODEL_DIR=<板端模型目录>
export ROS_DOMAIN_ID=<domain-id>
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
```

### 板端启动纯推理节点

在板端加载 ROS 环境后启动 cloud inference：

```bash
cd $IB_ROBOT_WS
source install/setup.bash
export ROS_DOMAIN_ID=<domain-id>
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch inference_service cloud_inference.launch.py \
  policy_path:=$BOARD_MODEL_DIR \
  device:=ascend_om
```

如果板端使用其他推理后端，替换 `device` 即可。例如 CPU 后端：

```bash
ros2 launch inference_service cloud_inference.launch.py \
  policy_path:=$BOARD_MODEL_DIR \
  device:=cpu
```

如需覆盖分布式 topic，板端使用 `input_topic:=...` 和 `output_topic:=...`，并且必须和本机 YAML 的 `cloud_inference_topic` / `cloud_result_topic` 保持一致。默认 topic 通常无需显式传参。

### 本机启动机器人和 edge 推理代理

在本机启动双臂 robot launch：

```bash
cd $IB_ROBOT_WS
source install/setup.bash
export ROS_DOMAIN_ID=<domain-id>
export ROS_LOCALHOST_ONLY=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 launch robot_config robot.launch.py \
  robot_config:=so101_dual_arm \
  control_mode:=model_inference \
  execution_mode:=distributed \
  use_sim:=false
```

## 常见问题

### 板端显示 ready，但没有推理

检查本机是否已经启动 `robot.launch.py`。如果本机没启动，`/preprocessed/batch` 通常只有板端 subscriber，没有 publisher。

```bash
ros2 topic info /preprocessed/batch
```

### 本机看不到板端节点

确认两端环境一致：

```bash
echo $ROS_DOMAIN_ID
echo $ROS_LOCALHOST_ONLY
echo $RMW_IMPLEMENTATION
```

两端 `ROS_DOMAIN_ID` 和 `RMW_IMPLEMENTATION` 必须一致，`ROS_LOCALHOST_ONLY` 必须为 `0`。

### 模型输入 key 不匹配

以板端模型目录中的 `config.json` 为准，检查 `input_features`。YAML 的 `contract.observations[].key` 必须与模型 key 完全一致。

例如模型要求：

```text
observation.images.top
observation.images.left
observation.images.right
observation.state
```

则 YAML 不应使用 `observation.images.left_wrist` 或 `observation.images.right_wrist`。

### 队列经常 empty 或动作响应慢

先看本机 `action_dispatcher` 日志和板端推理耗时。如果板端平均推理约 380 ms，本机端到端约 780 ms，`30 Hz` 下 `watermark_threshold: 40` 约提供 1.33 秒缓存，通常比 `20` 更稳。

如果仍频繁 empty，可以先降低 `control_frequency` 或增大 `watermark_threshold`，不要优先改模型后端。

### 实机异常运动

立即停止两端推理和本机控制 launch：

```bash
# 本机
pkill -f robot.launch.py
pkill -f action_dispatcher
pkill -f act_inference_node

# 板端
pkill -f pure_inference_node
```

然后确认 command topic 没有活跃 publisher，再排查 contract、标定、动作维度和左右臂顺序。
