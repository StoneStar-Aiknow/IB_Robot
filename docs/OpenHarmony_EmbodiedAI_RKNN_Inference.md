# OpenHarmony EmbodiedAI 1.0.1 RKNN NPU 推理指南

本文档覆盖：将训练好的 ACT 策略模型转换为 RKNN 格式，并在 OpenHarmony EmbodiedAI 1.0.1（RK3588）开发板上通过 NPU 运行推理。

> **板端环境前提**：RoboFrame 发布包的 `install.sh` 已自动完成全部板端配置（rknnlite、Python pysite、系统库、SSH 等）。以下内容假设 `install.sh` 已执行完毕。

## 1. 在主机上将 ONNX 转换为 RKNN

### 1.1 创建专用 RKNN 虚拟环境

rknn-toolkit2 要求 `torch<=2.4.0` + `numpy<=1.26.4`，与 lerobot 的 `torch>=2.7` + `numpy>=2.0` 冲突，需要单独的虚拟环境：

```bash
python3 -m venv .venv-rknn
source .venv-rknn/bin/activate
pip install rknn-toolkit2==2.3.2
```

### 1.2 导出并转换

```bash
source .venv-rknn/bin/activate

# 从 ONNX 转换
python src/model_utils/model_utils/export_onnx_rknn.py \
    --onnx models/502000/act_ros2_rknn.onnx \
    --output models/502000/act_ros2_rknn.rknn \
    --dtype float16

# 或直接从 policy checkpoint 转换
python src/model_utils/model_utils/export_onnx_rknn.py \
    --policy_path models/502000 \
    --output models/502000/act_ros2_rknn.rknn \
    --dtype float16
```

转换结果：`act_ros2_rknn.rknn`（约 114 MB，float16）。确保模型目录下只有一个 `*.rknn` 文件。

## 2. 部署与启动

### 2.1 配置 YAML

在 `robot_config` YAML（如 `so101_single_arm.yaml`）中需要修改两处：

**① 定义 RKNN 模型**（`models:` 节）：

```yaml
models:
  so101_act_rknn:
    path: models/502000/pretrained_model   # 模型目录路径（绝对或相对均可）
    policy_type: act
    device: rknn                            # 指定 RKNN 后端
    lerobot_norm_mode: range_m100_100
```

**② 选择该模型**（`control_modes.model_inference.inference.model`）：

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      model: so101_act_rknn    # ← 改为上面定义的模型名（原来是 so101_act）
```

### 2.2 两种运行方式

根据是否使用真实硬件，有两种运行方式：

#### 方式一：分布式（Ubuntu 仿真 + 板端 NPU 推理）

Ubuntu 主机负责 Gazebo 仿真与 Edge 侧预处理/后处理；端侧开发板负责 NPU 纯推理。两台机器必须位于同一局域网，设置相同的 `ROS_DOMAIN_ID`。

**Ubuntu 主机（仿真 + Edge）**：

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    execution_mode:=distributed \
    use_sim:=true
```

**端侧开发板（NPU Cloud 节点）**：

```bash
source /data/roboframe/scripts/robooh_1.0.1.env
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=<your_model_path> \
    device:=rknn
```

#### 方式二：单板全链路（板端独立运行）

开发板独立运行完整闭环：摄像头采集 → NPU 推理 → 机械臂控制，无需外接 Ubuntu 主机。

```
开发板 (RK3588, OpenHarmony EmbodiedAI 1.0.1)
├── usb_cam_node_exe × 2          (top + wrist 相机, MJPEG 640x480)
├── static_transform_publisher × 4 (TF: base→camera, gripper→camera, optical)
├── lerobot_policy_node × 1        (RKNN NPU 推理, ACT 策略)
├── action_dispatcher_node × 1     (动作分发, 20Hz)
└── so101_hardware                 (ros2_control, /dev/ttyACM0)
     ↳ arm_position_controller / gripper_position_controller
```

数据流：`相机 Image → 推理节点 (NPU ~500ms) → Action Dispatcher → Joint Commands → 机械臂`

**前置条件**：内核已启用 `CONFIG_USB_ACM=y`（SO-101 机械臂）。预编译内核镜像获取详见 [README.OpenHarmony](../README.OpenHarmony.md) FAQ。

```bash
source /data/roboframe/scripts/robooh_1.0.1.env

# 清理残留进程
pkill -9 -f "ros2 launch\|lerobot_policy_node\|action_dispatcher_node\|usb_cam_node_exe"

# 启动
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=false \
    control_mode:=model_inference
```

## 3. 机械臂校准

首次使用前必须在板端执行校准（交互式操作，需要手动转动机械臂关节）：

```bash
ssh root@<board_ip>
source /data/roboframe/scripts/robooh_1.0.1.env

ros2 run so101_hardware calibrate_arm --arm follower --port /dev/ttyACM0
```

校准 JSON 保存在 `~/.calibrate/so101_follower_calibrate.json`（SSH 环境 `HOME=/data/root`）。

如果通过 HDC shell（`HOME=/`）运行 launch，推理节点会找不到校准文件，需要符号链接：

```bash
mkdir -p /.calibrate
ln -sf /data/root/.calibrate/so101_follower_calibrate.json /.calibrate/so101_follower_calibrate.json
```

## 4. 推理管线说明

RKNN 后端与其他后端（CPU/GPU）共享同一套预处理/推理/后处理管线：

- **预处理**：`LeRobotPreprocessor` 根据 `config.json` 对观测值做归一化（图像归一化 + 状态归一化）
- **推理**：`PureInferenceEngine` 检测到 `device: rknn` 后，通过 `RKNNPolicyWrapper` → `RKNNRuntimeSession` 加载 `.rknn` 模型并在 NPU 上执行推理。RKNN 专属的 NHWC 布局转换在 session 内部自动完成
- **后处理**：`LeRobotPostprocessor` 将推理输出反归一化为关节角度

## 5. 预期输出与排障

### 预期输出

```
[usb_cam_node_exe-1] [INFO] [top_camera]: Starting 'top' (/dev/video20) at 640x480 via mmap (mjpeg2rgb) at 30 FPS
[usb_cam_node_exe-2] [INFO] [wrist_camera]: Starting 'wrist' (/dev/video22) at 640x480 via mmap (mjpeg2rgb) at 60 FPS
[lerobot_policy_node-7] [INFO] [act_inference_node]: Using inference_backend=rknn, tensor_device=cpu
[lerobot_policy_node-7] [INFO] [act_inference_node]: DispatchInfer Action Server ready
[lerobot_policy_node-7] [INFO] [act_inference_node]: ✓ First inference complete (monolithic): total=~500ms
[action_dispatcher_node-8] [INFO] [action_dispatcher]: ✓ First inference received: chunk=100
```

### 已知 warning 与排障

| 现象 | 原因 | 处理 |
|------|------|------|
| `robot_description not found` | URDF 包未部署，ros2_control 不需要它 | 忽略 |
| `Camera calibration file not found` | 未做相机内参标定 | 忽略 |
| `Query dynamic range failed (RKNN_ERR_MODEL_INVALID)` | 静态 shape 模型的正常警告 | 忽略 |
| `Ignoring unexpected goal response` | DDS 残留，重启前未清理进程 | 启动前 `pkill` |
| `unknown control 'white_balance_temperature_auto'` | USB 摄像头不支持该 V4L2 控制 | 忽略 |
| `/dev/ttyACM0` 不存在 | 内核缺少 `CONFIG_USB_ACM=y` | 刷入预编译内核，见 [README.OpenHarmony](../README.OpenHarmony.md) FAQ |
| `RKNN model file not found` | 模型目录下没有 `.rknn` 文件 | 确认模型文件已推送到 YAML 配置的 `path` 目录 |
| `Calibration file not found` | SSH vs HDC 的 HOME 不同 | 符号链接校准文件（§3） |
| 推理节点 SIGSEGV | env 未正确加载 | 确认 `source robooh_1.0.1.env` 已执行 |
| `input[0] need 2dims input, but 4dims` | 输入顺序不匹配 | RKNN 编译器会重排输入，重新导出后需验证 |

## 6. RKNN 运行时技术细节

### 输入顺序

RKNN 编译器会重排模型输入。ACT 模型原始 ONNX 输入为 `[cam_high, cam_left, state]`，转换后的 RKNN 模型期望 `[state, cam_high, cam_left]`。重新导出模型后务必通过测试推理验证输入顺序。

### NHWC 布局

RKNNLite 期望 4D 图像输入为 NHWC（1,H,W,C）布局。`RKNNRuntimeSession` 内部自动做 NCHW→NHWC 转换，无需手动处理。如果绕过 session 直接调用 `RKNNLite.inference()`，需自行确保 NHWC。
