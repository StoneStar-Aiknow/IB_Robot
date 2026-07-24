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

## Grasp execution target gripper

`robot.grasp_execution.planner_node` 会原样传给 `grasp_planner_node`。当 GraspGen source gripper
与目标执行器不一致，且执行侧已有目标夹爪 mesh tabletop hard gate 时，可设置
`enable_source_gripper_tabletop_sweep: false`，跳过源夹爪逐候选扫描；`enable_tabletop_filter` 仍应保持
`true`，以继续输出 table plane 和 object-top。

真机抓取配置可在 robot YAML 的 `robot.grasp_execution.target_gripper` 下声明目标夹爪几何。
SO101 单动爪使用 `fixed_finger_contact_ee` 作为固定指侧参考点，`closing_axis_ee` 表示从固定指
指向目标宽度中心的方向。`manipulation_execution/pick_executor_node` 会根据 GraspGen 候选的
`target_width_m` 计算有效接触中心：

```text
dynamic_fixed_finger_margin_m = min(
    fixed_finger_margin_max_m,
    fixed_finger_margin_m
        + max(0, fixed_finger_margin_width_ref_m - target_width_m)
        * fixed_finger_margin_width_gain,
)

effective_center = fixed_finger_contact_ee
                 + closing_axis_ee * 0.5 * (target_width_m + width_clearance_m)
                 + closing_axis_ee * dynamic_fixed_finger_margin_m
```

`fixed_finger_margin_m` 是额外远离固定指的基础安全距离，用于降低固定指先碰物体边缘或上表面的风险。
当前 SO101 RealSense 抓取配置默认基础值为 `0.006 m`；目标窄于 `fixed_finger_margin_width_ref_m=0.035 m`
时按 `fixed_finger_margin_width_gain=0.25` 增加安全余量，并由 `fixed_finger_margin_max_m=0.012 m` 封顶。

`target_gripper.fixed_finger_base_side` 是独立的候选硬约束。启用后，执行器在 `base` 坐标系 XY 平面
计算“目标宽度中心到固定指”与“目标宽度中心到 `reference_point_base`”的夹角余弦；低于
`min_alignment_cos` 的候选会在 IK/FK 准备前拒绝。SO101 默认参考机器人 base 原点且阈值为 `0.0`，
因此固定指必须位于物体朝机器人一侧，移动指从外侧闭合。无法获得可靠目标宽度区间时也会拒绝，
避免在固定指方向未知时继续执行。

`target_gripper.fixed_finger_robust_gap` 在下降完成、夹爪闭合前执行第二层硬检查。它把当前接触点残差
投影到实际闭合轴，并计算：

```text
effective_gap = fixed_finger_gap_m + contact_error_along_closing_axis_m
required_gap = fixed_finger_target_gap_m - max_target_gap_deficit_m
```

朝固定指方向的误差为负，会缩小 `effective_gap`。SO101 默认最多允许相对目标间隙损失 `0.003 m`；
不足时先退回 pregrasp，再尝试下一候选，避免在已知固定指侧覆盖不足时闭合夹爪。

`target_gripper.ik_orientation_guard` 约束 position-only IK 的实际 FK 朝向。SO101 的 joint5 对 TCP 位置
几乎不产生梯度，因此超出执行门限时会保持在 seed 附近；执行器将超限 joint5 seed 翻转 `±π`，让固定指和
活动指换侧，再比较 GraspGen 目标和实际 FK 的接近轴、180° 对称闭合轴直线及固定指内侧关系。当前配置使用：

```yaml
ik_orientation_guard:
  enabled: true
  approach_axis_ee: [0.0, 0.0, 1.0]
  closing_axis_180_symmetric: true
  joint5_abs_max: 2.0
  max_approach_error_deg: 25.0
  max_closing_error_deg: 20.0
```

`joint5_abs_max` 同时约束监督式测试脚本和 Hermes 执行器。两条链路都会先把超过半圈的解映射到
等价的 `[-π/2, π/2]` 分支，再以该门限和 FK 轴误差决定是否接受候选。

闭合轴的 180° 对称只表示两指闭合直线相同，不表示固定指身份可忽略。执行器必须再用实际 FK 位姿执行
`fixed_finger_base_side` 硬检查；固定指仍在外侧或目标宽度区间缺失时拒绝当前候选并继续 candidate fallback。

候选准备加速由同一 `grasp_execution.ik` 配置控制：

```yaml
ik:
  worker_count: 4
  worker_namespace_prefix: /ik_worker
  auto_start_workers: true
```

`embodied_pipeline.launch.py` 会自动启动对应数量的隔离 MoveIt worker。Hermes executor 与监督式脚本
都固定一份共同 `/joint_states` seed，将候选按 worker 分片并按原顺序合并；最终补偿与运动不进入 worker。
候选进入 worker 前的 SO101 mesh/tabletop 检查也采用与监督式脚本相同的凸包缓存和批量向量化路径。

`robot.grasp_execution.prepared_candidate_scoring` 控制 IK/FK 后软排序。SO101 使用候选目标宽度区间和
候选规划姿态计算固定指到目标前缘的间隙，并以动态 margin 为期望值计算包络分数。该分数与
接触点 XY/Z 质量、目标体积质心距离和 GraspGen 置信度加权，只改变执行顺序，不作为候选硬拒绝条件。

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
        pipelines:
          policy:
            model_path: models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515
            deployment: cpu
            execution_mode: monolithic # 或 distributed
            request_timeout: 5.0
            default_task: ""
            runtime_options: {}
      executor:
        type: topic
        mode: model_inference
        inference_pipeline: policy
        queue_size: 100
        watermark_threshold: 50
        control_frequency: 20.0
```

**启动的控制器：**
- `arm_position_controller` (JointGroupPositionController)
- `gripper_position_controller` (ForwardCommandController)

机器人 recording 配置中的 `lerobot_norm_mode` 决定 LeRobot 动作/观测与 `ros2_control`
弧度命令之间的转换方式。`range_m100_100` 使用机械臂 `[-100,100]`、
夹爪 `[0,100]`；`degrees` 对机械臂关节使用 centered degrees，但
`joints.gripper` 中的夹爪关节仍保持 `[0,100]` 开合语义。
真机模式从 `ros2_control.calib_file` 读取舵机校准范围；`use_sim:=true`
仿真模式不依赖该校准文件，而是从生成后的 URDF 关节 `limit` 读取弧度范围。

Observation 的 `align` 配置同时服务离线数据和在线推理，但两个时间阈值含义不同：

- `tol_ms`：`strategy: asof` 的时间对齐容差，离线录制/转换和在线采样都会使用；
  值小于等于 `0` 时退化为 `hold`。`hold` 和 `drop` 不使用该值。
- `max_age_ms`：在线推理允许复用的最大样本年龄，基于节点本地接收时钟计算，不能通过
  回拨推理请求时间戳绕过。缺失或只有未来时间戳的样本始终会
  被拒绝；大于 `0` 时还会拒绝超龄样本。以上情况都返回 `observation_not_ready`，而不是补零运行模型。

```yaml
contract:
  observations:
    - key: observation.images.top
      align:
        strategy: hold
        tol_ms: 1500
        max_age_ms: 500
```

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

### 真机运行时配置合成

SO101 手眼标定会把串口、相机分辨率、标定外参和 leader 配置保存在主机本地 runtime YAML。
启动完整抓取链路前，使用仓库 SSOT 刷新能力与安全配置，同时保留这些主机字段：

```bash
source .shrc_local && python3 scripts/synthesize_so101_grasp_runtime_config.py
```

默认输入为 `config/robots/so101_handeye_realsense_only.yaml`，输出为
`/tmp/so101_handeye_realsense_grasp.yaml`。合成器只从旧 runtime 保留 `ros2_control`、
`peripherals`、`contract` 和 `teleoperation`；`grasp_execution` 与 `embodied` 始终来自仓库
SSOT，避免旧 runtime 隐藏新增技能或安全策略。可先传 `--dry-run` 只做校验。

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

  skill_templates:
    wave_hello:
      description:
        summary: "Wave hello or goodbye with the wrist."   # ≤120 字符
        category: social_greeting                          # 任意非空字符串
        when_to_use: ["greet someone", "say hi or bye"]    # 非空字符串列表
        aliases_zh: ["打招呼", "挥手"]                      # MCP catalog 与规则入口共用的中文别名
        aliases_en: ["hello", "wave"]
        motion_scope: [wrist]                              # base|shoulder|elbow|wrist|gripper|arm
        anchor_pose: home                                  # 必须在 named_poses 中已定义，或 "none"
        intensity: moderate                                # subtle|moderate|large
        duration_sec_estimate: 8.0                         # 含 1.0 秒初始夹爪归一化并留有余量
        requires_motion_params: false                      # 布尔
        rule_entry: true                                   # 是否把 aliases_zh 注入规则解析器
      initial_gripper_state: closed
      primitive_sequence:
        - primitive_name: move_to_joint_positions
          joint_positions: {"1": 0.02, "2": 0.54, "3": -0.82, "4": -0.18, "5": 0.02}
          duration_sec: 2.0
        - primitive_name: move_through_joint_positions
          trajectory_template:
            type: single_joint_wave_v1
            waypoint_duration_sec: 0.05
            active_waypoint_count: 16
            repeat_count: 3
            base_pose: {"1": 0.02, "2": 0.54, "3": -0.82, "4": -0.18, "5": 0.02}
            joint: "5"
            amplitude: 0.35
        - primitive_name: move_to_joint_positions
          joint_positions: {"1": 0.02, "2": 0.54, "3": -0.82, "4": -0.18, "5": 0.02}
          duration_sec: 2.0

  named_targets:
    demo_object:
      observe_pose:  {position: {x: 0.25, y: 0.0, z: 0.26}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
      pregrasp_pose: {position: {x: 0.25, y: 0.0, z: 0.16}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
      grasp_pose:    {position: {x: 0.25, y: 0.0, z: 0.10}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
      lift_pose:     {position: {x: 0.25, y: 0.0, z: 0.25}, orientation: {x: 0.0, y: 1.0, z: 0.0, w: 0.0}}
```

#### skill_templates 与 description 契约

`embodied.skill_templates` 是技能执行的单一事实来源。每个 skill 模板可挂一个
结构化 `description` 块，作为 MCP catalog 可发现性与中文别名的唯一来源。
`aliases_zh` 始终用于 catalog 文档；仅当 `rule_entry: true` 且
`requires_motion_params: false` 时，launch 才会把这些别名注入规则解析器。
`robot_config.loader._validate_skill_description` 在加载时强校验字段类型与受控词表。

受控词表：
- `motion_scope`：`base` / `shoulder` / `elbow` / `wrist` / `gripper` / `arm`
- `intensity`：`subtle` / `moderate` / `large`
- `anchor_pose`：必须引用 `named_poses` 中已定义的位姿，或填 `"none"`
- `rule_entry`：可选布尔值；设为 `true` 时允许无动态运动参数的技能进入确定性规则入口
- `disabled`：可选布尔值；设为 `true` 时从 planner allowlist、规则入口、resolver、safety guard
  和 MCP catalog/validate/execute 的启用技能集合中统一排除
- `duration_sec_estimate`：必须为正数，并覆盖确定性手臂 motion/wait 总时长、`open`/`closed`
  初始夹爪归一化的 1.0 秒，以及每个显式 `open_gripper`/`close_gripper` primitive 的 1.0 秒，且保留执行余量

绝对关节轨迹必须在 `move_through_joint_positions` 前显式使用
`move_to_joint_positions` 进入首个 waypoint，并在需要时用另一条带正数
`duration_sec` 的 `move_to_joint_positions` 显式返回手势基位。trajectory generator
不会消费 `return_to_base` 一类隐式返回字段。

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

# 分布式推理 — 单机调试（YAML 中 policy pipeline 必须为 distributed）
# 终端 1：Edge
ros2 launch robot_config robot.launch.py config_path:=/absolute/path/to/so101_single_arm_distributed.yaml control_mode:=model_inference use_sim:=true
# 终端 2：Cloud
ros2 launch inference_service cloud_inference.launch.py pipeline_id:=policy model_path:=/absolute/path/to/policy_bundle deployment:=cpu

# 分布式推理 — 跨机器部署（端侧仅启动 Edge，Cloud 在算力机器上单独启动）
# 端侧：
ros2 launch robot_config robot.launch.py config_path:=/absolute/path/to/so101_single_arm_distributed.yaml control_mode:=model_inference use_sim:=true
# 算力机器（需设置相同 ROS_DOMAIN_ID）：
# ros2 launch inference_service cloud_inference.launch.py pipeline_id:=policy model_path:=/absolute/path/to/policy_bundle deployment:=cuda
```

相对 `model_path` 只相对于绝对路径环境变量 `WORKSPACE` 解析。Pipeline ID 决定默认 Action、
reset、health 和分布式 topic；多个 pipeline 时 `executor.inference_pipeline` 必须明确选择一个。

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
