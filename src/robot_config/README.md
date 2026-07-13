# robot_config

ros2_control 和外设的统一机器人配置系统。

## 概述

本软件包为机器人硬件提供统一的配置系统，桥接以下组件：

- **ros2_control**：用于关节/电机控制接口
- **外设**：用于相机和其他设备（通过现有 ROS2 驱动）
- **tensormsg**：用于 ML 策略 I/O 契约

目标是建立机器人硬件配置的单一数据源，消除不同配置系统之间的重复。

## 特性

- **单一 YAML 配置**：在一个文件中定义 ros2_control、相机和 ML 契约
- **使用现有 ROS2 相机驱动**：
  - `usb_cam` 用于 USB 相机（基于 OpenCV）
  - `realsense2_camera` 用于 RealSense D400 系列
- **TF 发布**：自动发布相机坐标系变换
- **标定支持**：标准 ROS2 camera_info_manager 集成
- **tensormsg 集成**：契约通过名称引用外设
- **RealSense contract relay**：在 `robot_config` 内部将驱动原生 topic 收口到统一 `/camera/{name}/...` 接口

## 架构

```
robot_config YAML（单一数据源）
        │
        ├───► ros2_control（关节/电机）
        │       └───► so101_hardware 插件
        │
        ├───► 相机驱动（现有 ROS2 包）
        │       ├───► usb_cam（USB 相机）
        │       └───► realsense2_camera（RealSense D400）
        │               └───► topic_relay（统一 contract topic）
        │
        └───► tensormsg 契约（ML I/O）
                └───► PolicyBridge / EpisodeRecorder
```

## Sim camera pose override（仅限仿真标定，opt-in）

仿真启动时，URDF / MJCF 中相机位姿默认来自 robot YAML 与
`launch_builders/sim_backend/camera_presets.py:PRESETS`。为了让开发者在仿真里
调出的视角可以跨重启复用，launch adapter 层会通过
`launch_builders/sim_backend/camera_overrides.py` 读取一条不进版本控制的用户级旁路：

| 项 | 说明 |
|---|---|
| 文件路径 | `~/.ros/ibrobot/sim_camera_overrides/<camera_name>.yaml` |
| 写入方 | `dataset_tools.camera_alignment`（stub）和 `sim_models.sim_camera_adjuster`（真实位姿） |
| 字段 | `parent_frame`, `pose.{x,y,z,roll,pitch,yaw}`, `fovy_deg` |
| 直接生效平台 | `gazebo` |
| MuJoCo 路径 | 不直接读取同一套角度，而是由 `mujoco_adapter.py` 做显式坐标系转换 |
| 文件缺失或 stub | 回退到 `PRESETS`，保持历史默认行为 |

这条 override 只服务于开发者标定，不应该被运行时业务逻辑当作新的 SSOT。
如果仿真相机姿态和 YAML 里看到的不一致，先检查这个目录里是否残留了历史 override。

## 配置示例

```yaml
robot:
  name: so101_single_arm
  type: so101
  robot_type: so_101

  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
    port: /dev/ttyACM0
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
    reset_positions:
      "1": 0.0813
      "2": 3.7905

  peripherals:
    - type: camera
      name: top
      driver: opencv  # 使用 usb_cam 包
      index: 0
      width: 640
      height: 480
      fps: 30
      frame_id: camera_top_frame
      optical_frame_id: camera_top_optical_frame
      # 可选 ISP 参数（usb_cam 全部 V4L2 控件，详见 usb_cam_node.cpp:65-85）：
      # brightness: 128            # 0-255
      # contrast: 128              # 0-255
      # saturation: 128            # 0-255
      # sharpness: 128             # 0-255
      # gain: 0                    # 0-255
      # auto_white_balance: false  # 与 white_balance 互斥
      # white_balance: 4600        # 2000-10000 K
      # autoexposure: false        # 与 exposure 互斥
      # exposure: 312              # V4L2 absolute exposure
      # autofocus: false
      # focus: 0                   # 0-1023

    - type: camera
      name: wrist
      driver: realsense  # 使用 realsense2_camera 包
      serial_number: "12345678"
      width: 640
      height: 480
      fps: 30
      depth_width: 640
      depth_height: 480
      frame_id: camera_wrist_frame

  contract:
    observations:
      - key: observation.images.top
        topic: /camera/top
        peripheral: top  # 引用上面的相机
        image:
          resize: [480, 640]
      - key: observation.current
        topic: /so101_follower/joint_currents
        type: ibrobot_msgs/msg/JointCurrent
        selector:
          names: ["current.1", "current.2", "current.3", "current.4", "current.5", "current.6"]
```

## 控制模式配置

robot_config 包支持双控制模式，以满足不同 AI 模型的需求：

### 可用控制模式

#### 1. teleop 模式（人工遥操作）

**适用于：** 人工遥操作设备（leader arm、游戏手柄、VR设备）

**特点：**
- 实时直接控制
- 零延迟直通（< 5ms）
- 支持多种输入设备
- 内置安全过滤（关节限位）

**配置：**
```yaml
robot:
  default_control_mode: "teleop"

  control_modes:
    teleop:
      description: "人工遥操作模式（直接控制）"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: false
        force_disable: true

  teleoperation:
    enabled: true
    # 单设备使用 active_device；双臂等多输入设备使用 active_devices。
    active_device: "so101_leader"
    # active_devices: ["left_leader", "right_leader"]
    devices:
      - name: "so101_leader"
        type: "leader_arm"
        port: "/dev/ttyUSB0"
        calib_file: "$(env HOME)/.calibrate/so101_leader_calibrate.json"
        target:
          arm_joint_names: ["1", "2", "3", "4", "5"]
          gripper_joint_names: ["6"]
          arm_command_topic: "/arm_position_controller/commands"
          gripper_command_topic: "/gripper_position_controller/commands"
```

多设备遥操作时，`active_devices` 按名称选择 `devices` 中的多个输入设备。每个
设备可通过 `target` 指定要控制的关节组和控制器命令话题；未指定时回退到机器人级
`joints.arm` / `joints.gripper` 以及默认单臂控制器话题。

**启动命令：**
```bash
# 遥操作模式（episodic 录制）
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=teleop \
  record:=true \
  record_mode:=episodic \
  use_sim:=false

# 遥操作模式（episodic + Rerun）
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=teleop \
  record:=true \
  record_mode:=episodic \
  record_visualizer:=rerun \
  use_sim:=false

# 另一个终端启动录制客户端
ros2 run dataset_tools record_cli

# 录制完成后转换为 LeRobot 数据集
ros2 run dataset_tools bag_to_lerobot \
  --bags-dir ~/rosbag/episodes/so101_single_arm \
  --robot-config src/robot_config/config/robots/so101_single_arm.yaml \
  --out /path/to/output_dataset
```

#### 2. model_inference 模式（高频位置控制）

**适用于：** 端到端模仿学习模型（ACT、pi0、Diffusion Policy）

**特点：**
- 高频控制（50-100Hz）
- 低延迟（1-3ms）
- 直接基于话题的位置命令
- 反应迅速、运动流畅

**配置：**
```yaml
robot:
  default_control_mode: "model_inference"

  control_modes:
    model_inference:
      description: "高频端到端控制模式（ACT/pi0）"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: true
        execution_mode: "distributed" # 或 "monolithic" (单机零拷贝)
        model: so101_act
        attention_viz:
          enabled: false
          mode: file
          interactive_masking: false
          mask_save_dir: gui_interactions
```

**启动的控制器：**
- `arm_position_controller` (JointGroupPositionController)
- `gripper_position_controller` (ForwardCommandController)

模型配置中的 `lerobot_norm_mode` 决定 LeRobot 动作/观测与 `ros2_control`
弧度命令之间的转换方式。`range_m100_100` 使用机械臂 `[-100,100]`、
夹爪 `[0,100]`；`degrees` 对机械臂关节使用 centered degrees，但
`joints.gripper` 中的夹爪关节仍保持 `[0,100]` 开合语义。
真机模式从 `ros2_control.calib_file` 读取舵机校准范围；`use_sim:=true`
仿真模式不依赖该校准文件，而是从生成后的 URDF 关节 `limit` 读取弧度范围。

LeRobot 转换 metadata 中的标定来源字段保持稳定契约。这里的
`calibration_source` / `calibration_sources` 是数据集 metadata 输出字段，
不是 `ros2_control` YAML 输入 schema：

- `calibration_source`：兼容旧消费者的单字符串字段，始终取第一个解析到的标定文件路径。
- `calibration_sources`：完整的多标定文件路径列表，多标定源场景应优先读取该字段。

单臂旧式 `ros2_control.calib_file` 会映射到 `arm` 命名空间：

```yaml
robot:
  ros2_control:
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
```

双臂或更多来源通过 `ros2_control.xacro_args.calib_file_<namespace>` 声明。后缀即 LeRobot 转换使用的关节命名空间，也会作为 xacro/URDF 参数名的一部分。`<namespace>` 允许字母、数字和下划线；建议优先使用 `left`、`right`、`front` 这类与本体/机械臂语义相关的名称，便于和 `joint_names`、数据集 metadata 及策略特征对齐：

```yaml
robot:
  ros2_control:
    xacro_args:
      calib_file_front: $(env HOME)/.calibrate/front_calibrate.json
      calib_file_left: $(env HOME)/.calibrate/left_calibrate.json
      calib_file_right: $(env HOME)/.calibrate/right_calibrate.json
      calib_file_1: $(env HOME)/.calibrate/extra_calibrate.json
```

从磁盘加载 SO-101 数字关节标定时，`robot_config` 会按每个来源的 key 后缀转换为内部命名空间键，例如 `"1"` 到 `"6"` 映射为 `joint1_arm` 到 `joint6_arm`、`jointN_left` / `jointN_right` 或 `jointN_1`。`calib_file_<namespace>` 支持任意数量的唯一命名空间；如果合并后的标定键冲突，加载会直接报错。`calib_file_1` 的 namespace 就是 `1`，不会额外推断为 `left` 或 `right`。

LeRobot 转换 metadata 中的标定来源字段保持稳定契约：

- `calibration_source`：兼容旧消费者的单字符串字段，始终取第一个解析到的标定文件路径。
- `calibration_sources`：完整的多标定文件路径列表，多标定源场景应优先读取该字段。

当机器人通过 `ros2_control.calib_file` 配置单个标定文件时，这两个字段都指向同一来源；当机器人通过 `ros2_control.xacro_args.calib_file_1`、`calib_file_2` 等按编号配置多个标定文件时，不需要额外合并标定文件，`calibration_source` 仍保持首个路径，完整有序列表写入 `calibration_sources`。

**命令接口：**
```bash
# 机械臂位置命令
ros2 topic pub /arm_position_controller/commands std_msgs/msg/Float64MultiArray "data: [1.0, 2.0, 3.0, 4.0, 5.0]"
```

### Tracing（ros2_tracing / LTTng）

`robot.launch.py` 支持两个 tracing 相关 launch 参数：

- `enable_tracing:=true`：在启动时开启 LTTng tracing session
- `trace_session_name:=...`：覆盖默认 session 名 `ib_robot_trace`

按包内架构，`robot.launch.py` 只保留编排职责；LTTng session 的创建、命名冲突处理、
以及 shutdown 时的 stop/destroy 生命周期都由
`robot_config/launch_builders/tracing.py` 统一管理。

### Voice ASR（语音识别）

`robot_config` 通过 `robot.voice_asr` 作为语音识别节点的机器人级单一配置来源，并由
`robot_config/launch_builders/voice_asr.py` 注入到 `voice_asr_service` 节点。

常用启动覆盖参数：

```bash
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  voice_asr_auto_start:=true \
  voice_asr_device_name:=Blackwire \
  voice_asr_realtime_pre_roll_seconds:=0.5
```

| Launch 参数 | 作用 |
| --- | --- |
| `voice_asr_auto_start` | 临时覆盖 `robot.voice_asr.enabled`，设为 `true` 时强制启动 ASR 节点 |
| `voice_asr_device_index` | 临时覆盖 `robot.voice_asr.device_index` |
| `voice_asr_device_name` | 临时覆盖 `robot.voice_asr.device_name`，优先按设备名匹配 |
| `voice_asr_realtime_pre_roll_seconds` | 临时覆盖实时识别 pre-roll 时长 |

`voice_asr_service` 的包级默认值与 `robot_config` 中的 `VoiceASRConfig` 默认值保持同步；
具体机器人仍应以 `config/robots/<robot>.yaml` 中的 `robot.voice_asr` 为准。

### 具身 AI 流水线（Embodied AI Pipeline）

`robot_config` 通过 `robot.embodied` 字段统一管理具身 AI 链路配置，但不直接启动
具身业务节点。完整运行时由 `embodied_bringup` 消费同一份 YAML 并编排下游节点，
以保持依赖方向为 `embodied_bringup -> robot_config`。

**前提条件**：具身流水线当前只在 `moveit_planning` 控制模式下可用。

#### 启动参数

| Launch 参数 | 作用 |
| --- | --- |
| `with_embodied` | 在 `robot_config` 基础 launch 中仅保留兼容覆盖；完整具身链路请使用 `embodied_bringup` |

```bash
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=moveit_planning \
  use_sim:=true \
  moveit_display:=false
```

#### YAML 配置结构

```yaml
embodied:
  enabled: false              # 默认关闭；通过 embodied_bringup launch 临时开启
  debug_tracing: true

  timeouts:
    task_budget_sec: 180.0         # 任务端到端总预算
    scene_freshness_sec: 0.5       # 图像/深度新鲜度门槛
    model_idle_timeout_sec: 120.0  # 大模型输出空闲超时
    rpc_timeout_sec: 5.0           # action/server/service 统一 RPC 超时
    gripper_settle_sec: 1.5        # 夹爪稳定等待时间

  planner:
    mode: vlm_api             # rule / vlm_api / hybrid
    scene_sources:
      primary_camera_topic: /camera/front_camera/color/image_raw
      primary_camera_info_topic: /camera/front_camera/color/camera_info
      primary_aligned_depth_topic: /camera/front_camera/aligned_depth_to_color/image_raw
      primary_pointcloud_topic: /camera/front_camera/depth/color/points
      ee_pose_topic: /robot_status/ee_pose
      joint_state_topic: /joint_states
      require_depth: true
      require_pointcloud: false
    vlm_api:
      provider: openai_compatible
      base_url: http://localhost:8000/v1
      model: Qwen3.5-9B
      api_key_env: ""

  entry:
    visual_games:
      sorting_hat:
        enabled: false        # 趣味视觉游戏默认关闭
        trigger_aliases: [分院帽, 奔月帽, 风月帽, 分月帽]

  execution:
    relative_motion_reference_frame: base
    relative_motion_step_m: 0.03
    relative_motion_direction_mapping:
      forward:  [1, 0, 0]
      backward: [-1, 0, 0]
      left:     [0, 1, 0]
      right:    [0, -1, 0]
      up:       [0, 0, 1]
      down:     [0, 0, -1]

  safety:
    workspace:
      x: [-0.05, 0.45]
      y: [-0.35, 0.35]
      z: [0.05, 0.55]

  named_poses:
    home:           {position: {x: 0.15, y: 0.0, z: 0.30}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
    observe_table:  {position: {x: 0.20, y: 0.0, z: 0.35}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}

  named_targets:
    demo_object:
      observe_pose:  {position: {x: 0.25, y: 0.0, z: 0.26}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
      pregrasp_pose: {position: {x: 0.25, y: 0.0, z: 0.16}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
      grasp_pose:    {position: {x: 0.25, y: 0.0, z: 0.10}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
      lift_pose:     {position: {x: 0.25, y: 0.0, z: 0.25}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
```

`embodied.entry.visual_games` 声明入口层视觉趣味游戏（如分院帽）的触发别名与开关；
camera/VLM/timeout 仍由 `embodied.perception` 统一管理。`validate_config()` 强制一致性：
任一游戏 `enabled=true` 而 `embodied.perception.enabled=false` 时返回错误，配置阶段即拦截。

更多具身节点说明，详见各子包 README：
- [`embodied_agent`](../embodied_agent/README.md)
- [`vlm_task_planner`](../vlm_task_planner/README.md)
- [`perception_service`](../perception_service/README.md)
- [`skill_library`](../skill_library/README.md)
- [`safety_guard`](../safety_guard/README.md)

#### 2. moveit_planning 模式（轨迹规划控制）

**适用于：** 基于规划的模型（VoxPoser、VLM、目标条件化）

**特点：**
- 轨迹插值和时间参数化
- 通过 MoveIt 动作接口执行
- 碰撞检测和避障
- 更平滑的轨迹

**配置：**
```yaml
robot:
  default_control_mode: "moveit_planning"

  control_modes:
    moveit_planning:
      description: "MoveIt 轨迹规划模式"
      controllers:
        - joint_state_broadcaster
        - arm_trajectory_controller
        - gripper_trajectory_controller
```

**启动的控制器：**
- `arm_trajectory_controller` (JointTrajectoryController)
- `gripper_trajectory_controller` (JointTrajectoryController)

**命令接口：**
```bash
# 通过 MoveIt 动作执行轨迹
ros2 action send_goal /arm_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{...}"
```

### 完整控制模式配置示例

```yaml
robot:
  name: so101_single_arm
  type: so101
  default_control_mode: "model_inference"  # 或 "teleop" 或 "moveit_planning"

  joints:
    arm: ["1", "2", "3", "4", "5"]
    gripper: ["6"]
    all: ["1", "2", "3", "4", "5", "6"]

  # 控制模式配置
  control_modes:
    teleop:
      description: "人工遥操作模式"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller

    model_inference:
      description: "高频端到端控制模式"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller

    moveit_planning:
      description: "MoveIt轨迹规划模式"
      controllers:
        - joint_state_broadcaster
        - arm_trajectory_controller
        - gripper_trajectory_controller

  # 遥操作配置
  teleoperation:
    enabled: true
    active_device: "so101_leader"
    devices:
      - name: "so101_leader"
        type: "leader_arm"
        port: "/dev/ttyUSB0"
        calib_file: "$(env HOME)/.calibrate/so101_leader_calibrate.json"

  # 硬件配置
  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
    port: /dev/ttyACM0
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
    reset_positions:
      "1": 0.0813
      "2": 3.7905

  # 外设（相机、传感器）
  peripherals:
    - type: camera
      name: top
      driver: opencv
      index: 0
      width: 640
      height: 480
      fps: 30

  # ML 契约
  contract:
    observations:
      - key: observation.images.top
        topic: /camera/top
        peripheral: top
        image:
          resize: [480, 640]
    actions:
      - key: action
        topic: /arm_position_controller/commands  # 根据模式变化
        ros_type: std_msgs/msg/Float64MultiArray
        names: ["1", "2", "3", "4", "5", "6"]
```

### 模式切换工作原理

1. **配置阶段：**
   - `robot.launch.py` 从 YAML 读取 `default_control_mode`
   - 可通过 `control_mode:=xxx` 命令行参数覆盖
   - 验证模式是否存在于 `control_modes` 部分

2. **控制器生成：**
   - 仅生成所选模式中列出的控制器
   - 确保无控制器冲突（同一关节不能被多个控制器控制）

3. **动作分发集成：**
   - `action_dispatch` 节点从 `robot_config` 读取当前模式
   - 实例化适当的执行器（TopicExecutor 或 ActionExecutor）
   - 为上游推理服务提供统一 API

### 控制模式故障排除

#### 模式未切换

**问题：** 命令行覆盖未生效

**解决方案：** 确保 `control_mode` 参数拼写正确：
```bash
ros2 launch robot_config robot.launch.py control_mode:=moveit_planning use_sim:=true
```

#### 控制器冲突

**问题：** 同一关节被多个控制器控制

**解决方案：** 检查配置，确保每种模式使用互斥的控制器：
```yaml
control_modes:
  model_inference:
    controllers:
      - arm_position_controller      # 位置控制器
  moveit_planning:
    controllers:
      - arm_trajectory_controller    # 轨迹控制器（不同类型）
```

#### 执行器类型不匹配

**问题：** 推理服务发送位置命令但启动了轨迹控制器

**解决方案：** 确保控制模式与模型类型匹配：
- ACT/pi0 模型 → `model_inference` 模式
- VoxPoser/VLM 模型 → `moveit_planning` 模式
- 人工遥操作 → `teleop` 模式

## 使用方法

### 启动机器人

```bash
# 启动真实硬件
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm

# 启动仿真
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

# 契约级 mock 仿真
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true sim_platform:=mock control_mode:=model_inference

# MoveIt 规划模式（带 RViz）
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning use_sim:=true

# MoveIt 模式无 RViz（headless）
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning use_sim:=true moveit_display:=false

# 分布式推理 — 单机调试（Edge + Cloud 同时启动）
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=model_inference execution_mode:=distributed use_sim:=true cloud_local:=true

# 分布式推理 — 跨机器部署（端侧仅启动 Edge，Cloud 在算力机器上单独启动）
# 端侧：
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=model_inference execution_mode:=distributed use_sim:=true
# 算力机器（需设置相同 ROS_DOMAIN_ID）：
# ros2 launch inference_service cloud_inference.launch.py policy_path:=/path/to/model device:=cuda
# 端侧开发板 Ascend NPU：
# ros2 launch inference_service cloud_inference.launch.py policy_path:=/path/to/model device:=npu
```

### 验证配置

```bash
# 直接使用 Python 验证机器人配置文件
python3 src/robot_config/robot_config/scripts/validate_config.py \
    src/robot_config/config/robots/so101_single_arm.yaml
```

## 相机驱动

### USB 相机（通过 `usb_cam`）

```yaml
- type: camera
  name: usb_cam
  driver: opencv
  index: 0
  width: 640
  height: 480
  fps: 30
  pixel_format: mjpeg2rgb
  frame_id: camera_frame
  camera_info_url: file://$(env HOME)/.ros/camera_info/top.yaml
```

**安装：**
```bash
sudo apt install ros-humble-usb-cam
```

### RealSense 相机（通过 `realsense2_camera`）

```yaml
- type: camera
  name: realsense
  driver: realsense
  serial_number: "12345678"
  width: 640
  height: 480
  fps: 30
  depth_width: 640
  depth_height: 480
  depth_fps: 30
  enable_depth: true
  enable_color: true
  align_depth: false
```

**安装：**
```bash
# Ubuntu
sudo apt install ros-humble-librealsense2*
# openEuler
sudo dnf install ros-humble-librealsense2*
```

## 相机标定

相机内参可以存储在标准 ROS2 位置：

```yaml
- type: camera
  name: top
  driver: opencv
  index: 0
  width: 640
  height: 480
  camera_info_url: file://$(env HOME)/.ros/camera_info/top_camera.yaml
```

可以使用标准 ROS2 相机标定工具创建标定文件：

```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 \
  --square 0.024 \
  image:=/camera/top/image_raw
```

### 色彩 / ISP 校准 — Override 机制

YAML 是相机参数的 **单一真理来源（SSOT）**，但日常调机时手动改 YAML
不便。为此 robot_config 在加载相机参数时按以下顺序合并：

1. `peripherals/[name]` YAML 默认值；
2. `~/.ros/ibrobot/camera_isp_overrides/{camera_name}.json` 中的 override
   （由 `camera_isp_calibrator` 写入，键为 V4L2 参数名）；

**override 中的值覆盖 YAML**。这样校准结果可以跨次启动复用而不污染
SSOT；删除 JSON 即可回退到 YAML 默认。

加载逻辑见 `robot_config/launch_builders/camera_isp_overrides.py`，
合并点见 `robot_config/launch_builders/perception.py`（opencv 分支）。
配套校准工具：`ros2 run dataset_tools camera_isp_calibrator`。

## tensormsg 集成

robot_config 通过允许观察通过名称引用外设来与 tensormsg 契约集成：

```yaml
# 在 robot_config 中
peripherals:
  - type: camera
    name: top
    width: 640
    height: 480

# 在 contract 部分
contract:
  observations:
    - key: observation.images.top
      topic: /camera/top
      peripheral: top  # 自动填充外设的 width、height、fps
```

当契约加载时，将自动包含外设定义中的相机元数据。

## 故障排除

### 相机无法打开

检查 USB 权限：
```bash
ls -l /dev/video*
sudo chmod 666 /dev/video0
```

或将用户添加到 `video` 组：
```bash
sudo usermod -a -G video $USER
```

### RealSense 相机未找到

安装 librealsense2：
```bash
# Ubuntu
sudo apt install librealsense2-utils librealsense2-dev
sudo apt install ros-humble-librealsense2*

# openEuler
sudo dnf install librealsense2-utils librealsense2-devel
sudo dnf install ros-humble-librealsense2*
```

检查相机是否连接：
```bash
realsense-viewer
```

### 标定文件未找到

确保路径正确并以 `file://` 开头：
```yaml
camera_info_url: file:///home/user/.ros/camera_info/top.yaml
```

### 控制器加载失败

如果遇到 "Controller already loaded" 错误，运行清理脚本：
```bash
./scripts/cleanup_ros.sh
```

## 仿真依赖（MuJoCo）

`use_sim:=true` 模式依赖 `ros-humble-mujoco-ros2-control`，通过 rosdep 声明为运行时依赖：

```bash
# 安装所有依赖（含 MuJoCo 仿真包）
rosdep install --from-paths src --ignore-src -y
```

该命令会自动安装 `ros-humble-mujoco-ros2-control` 及其传递依赖
（`mujoco_ros2_control_msgs`、`mujoco_ros2_control_plugins`、`mujoco_vendor`）。
不再需要手动初始化 git submodule。

### 仿真推理控制面板

在 MuJoCo 仿真推理调试时，推荐使用仓库根目录下的
`scripts/model_infer_panel.py`，提供 Start/Stop 推理、随机场景、AutoTest
全量评估等控制能力：

```bash
python3 scripts/model_infer_panel.py
# 浏览器访问 http://<robot-ip>:8766
```

相机图像可视化请使用已有的 Rerun 链路：

```bash
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=model_inference \
  use_sim:=true \
  record_visualizer:=rerun
```

## 参考资料

- [usb_cam GitHub](https://github.com/ros-drivers/usb_cam) - ROS2 USB 相机驱动
- [realsense-ros GitHub](https://github.com/realsenseai/realsense-ros) - Intel RealSense ROS2 封装
- [action_dispatch README](../action_dispatch/README.md) - 详细执行器文档
- [docs/architecture.md](../../docs/architecture.md) - 系统架构概览

## 许可证

Apache-2.0
