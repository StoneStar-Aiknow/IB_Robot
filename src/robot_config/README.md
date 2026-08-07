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
- **RealSense contract normalization**：将驱动原生 topic 收口到统一 `/camera/{name}/...` 接口；高带宽
  RGB-D 可由驱动直接 remap，CameraInfo relay 仅负责规范化 `frame_id`
- **通用模型服务编排**：通过顶层 `perception_services.services` 列表启动任意强类型 model-service plugin

### ros2_control 启动门禁

真机控制器由 `robot_config/controller_spawner` 串行加载、配置和激活；该入口修复 ROS 2 Humble 官方 spawner
未向 `load_controller` 传递 service call timeout 的问题，并在调用超时后重新读取 lifecycle 状态，接受服务端
已经完成的加载。随后统一的 `wait_for_controllers` 再确认 robot YAML 为当前控制模式声明的全部 controller 都是
`active`，通过后才启动 MoveIt、teleop 或 task executor。发现、service call 和 switch 的等待上限都来自
robot YAML 的 `robot.controller_startup_timeout.hardware`，不再维护第二套 hardware lifecycle readiness 判定。

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
        │               ├───► direct topic remap（高带宽 RGB-D）
        │               └───► topic_relay（CameraInfo/frame_id 规范化）
        │
        └───► tensormsg 契约（ML I/O）
                └───► PolicyBridge / EpisodeRecorder
```

### 通用模型服务 SSOT

模型服务使用顶层 `perception_services`，不属于具体消费者。每个 enabled entry 必须显式选择 manifest 中的命名
deployment，不能直接填写 `backend` 或 `device`：

```yaml
robot:
  perception_services:
    services:
      - id: depth_front
        enabled: true
        required: true
        bundle_path: models/depth/front
        deployment: cpu
        adapter_class: depth_service.plugin:DepthServicePlugin
        service_type: depth_msgs/srv/EstimateDepth
        endpoint: /depth/front
        node_name: depth_front_service
        runtime_options: {}
```

配置 loader 验证 bundle/manifest/deployment、plugin 和 service type 语法，并拒绝重复 ID、node name 和 endpoint。
Launch builder 对每个 enabled entry 启动一个相互独立的 `perception_service/model_service_node`，不维护模型家族
白名单或 family-to-executable map。Manifest/deployment fingerprint 是结构化运行身份；artifact 内容校验属于
模型打包流程，不通过服务启动时扫描大文件完成。通用层只用 dummy/echo plugin 做 CI 参考，生产 adapter 和具体
service entry 由消费功能自行注册。

语义建图在 `semantic_mapping.perception.semantic_roles` 中仅把 `sam2_masks`、`ram_plus_tags`、
`siglip2_image`、`siglip2_text` 和可选 `gdino_confirmation` 绑定到上述 service ID。语义层验证每个 role 的
精确 service type、required/optional policy、manifest semantic identity，以及 SigLIP image/text embedding
metadata 兼容性；通用 parser 仍不维护模型 family 词表。有效 role bindings、service declarations、deployment
fingerprint 和 semantic identity fingerprint 共同生成与路径和列表顺序无关的正 63-bit configuration generation。

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

## Observation video transport

Observation transport is part of the contract and is absent by default. An omitted
`transport` field keeps the existing DDS image behavior. Explicit RTP is only valid
for `sensor_msgs/msg/Image` observations using `rgb8` or `bgr8`, and requires a
distributed inference pipeline. RTP configuration is fail-closed: it does not fall
back to DDS when a stream or codec is unavailable.

```yaml
observations:
  - key: observation.images.top
    topic: /camera/top/image_raw
    type: sensor_msgs/msg/Image
    peripheral: top
    image: {resize: [480, 640], encoding: rgb8}
    transport:
      mode: rtp
      stream_id: top
      endpoint: {host: 192.168.10.20, port: 5004}
      codec: h264
      encoder_backend: nvidia  # 本机 RTX 使用 NVENC；也可用 auto/software/ascend
      decoder_backend: ascend
      h264: {profile: main, bitrate_bps: 4000000, gop_frames: 15}
      media:
        width: 640
        height: 480
        frame_rate_hz: 30
        pixel_format: nv12
        color_space: bt709
        color_range: limited
      buffer: {sender_queue_frames: 2, receiver_queue_packets: 256, decoded_frame_capacity: 32, retention_ms: 1000}
      readiness: {keyframe_timeout_ms: 3000, timestamp_mapping_max_age_ms: 1000, max_inter_camera_skew_ms: 50}
      security: none
```

The contract fingerprint includes stream identity, endpoint, codec/media reconstruction
semantics, buffering/readiness limits, and image metadata. Consequently endpoint
changes require matching edge and cloud deployments; there are no excluded endpoint
overrides in this milestone. RTP/UDP has no authentication, confidentiality, or
integrity protection and is limited to a trusted robot network. Use an explicit
`mode: dds` contract for rollback or recording-compatible deployments.

Development examples are available in `config/robots/dev_rtp_single_camera.yaml`
and `config/robots/dev_rtp_multi_camera.yaml`; production profiles remain DDS by
default.

`nvidia` 当前仅支持 encoder，典型组合是 edge 使用 `encoder_backend: nvidia`，310B cloud 使用
`decoder_backend: ascend`。`auto` 的 encoder 探测顺序为 `ascend`、`nvidia`、`software`；显式选择
不可用 backend 时直接拒绝启动。

## Grasp execution target gripper

`robot.grasp_execution.planner_node` 中的 ROS 参数会传给 `grasp_planner_node`；`host_runtime` 是
bringup 消费的进程启动配置，不会注入 ROS 参数。当 GraspGen source gripper 与目标执行器不一致，
且执行侧已有目标夹爪 mesh tabletop hard gate 时，可设置
`enable_source_gripper_tabletop_sweep: false`，跳过源夹爪逐候选扫描；`enable_tabletop_filter` 仍应保持
`true`，以继续输出 table plane 和 object-top。

Ascend 板端可分别在 `perception_node` 和 `planner_node` 下限制 CPU 数值库线程：

```yaml
host_runtime:
  omp_threads: 4
  blas_threads: 1
```

`embodied_bringup` 会据此设置进程级 `OMP_NUM_THREADS` 和 `OPENBLAS_NUM_THREADS`，并让 GNU OpenMP
worker 使用被动等待，避免 NPU 请求完成后继续抢占 CPU。该配置只影响对应节点进程，不影响 MoveIt、
相机驱动或其他机器人配置。

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

`target_gripper.fixed_finger_robust_gap` 在候选 IK/FK 准备阶段执行硬检查。它把预测接触点残差投影到
实际闭合轴，并计算：

```text
effective_gap = fixed_finger_gap_m + contact_error_along_closing_axis_m
required_gap = fixed_finger_target_gap_m - max_target_gap_deficit_m
gap_deficit = max(0, required_gap - effective_gap)
pass = gap_deficit <= measurement_tolerance_m
```

朝固定指方向的误差为负，会缩小 `effective_gap`。SO101 默认最多允许相对目标间隙损失 `0.003 m`，并给
IK/FK、TF 和舵机误差保留 `0.001 m` 容差；超过该范围的候选在 approach/descend 之前拒绝。候选准备阶段的
`robust_gap_headroom` 仍记录未经容差放宽的原始余量用于排序。

下降运动成功即进入 commit-to-grasp 状态：`close_gripper` 是下降后的第一条动作；低位实测在闭爪后继续以
best-effort 方式写入 pose diagnostics，诊断异常不会阻止闭爪或后续验证。执行器不再退回 pregrasp 或切换
候选。这样所有可恢复的候选淘汰都发生在低位运动之前，避免机械臂到达物体后因边界测量噪声突然上升。

`candidate_target_offset_base_m` 是候选目标在 `base_frame` 中的三轴平移补偿。它统一作用于 grasp、approach、
lift 和规划接触点，不改变相机测得的物体宽度端点。SO101 hand-eye profile 使用旧 310P marker test
验证过的 `[0.0, 0.0, -0.008]`；该值属于机器人/手眼执行几何，因此由 robot YAML 管理，action 客户端不提供
覆盖参数。

`target_gripper.ik_orientation_guard` 约束 position-only IK 的实际 FK 朝向。SO101 的 joint5 对 TCP 位置
几乎不产生梯度，因此超出执行门限时会保持在 seed 附近；执行器将超限 joint5 seed 翻转 `±π`，让固定指和
活动指换侧，再比较 GraspGen 目标和实际 FK 的接近轴、180° 对称闭合轴直线及固定指内侧关系。当前配置使用：

```yaml
ik_orientation_guard:
  enabled: true
  approach_axis_ee: [0.0, 0.0, 1.0]
  closing_axis_180_symmetric: true
  joint5_home_max_delta_rad: 1.5707963267948966
  joint5_limit_epsilon_rad: 0.001
  joint5_stage_continuity: true
  joint5_stage_max_delta_rad: 1.5707963267948966
  max_approach_error_deg: 25.0
  max_closing_error_deg: 20.0
  moveit_orientation_search:
    enabled: true
    approach_tolerance_deg: 15.0
    free_rotation_tolerance_deg: 180.0
    constraint_weight: 1.0
    max_attempts: 3
```

HOME joint5 由同一 robot YAML 的 `ros2_control.reset_positions["5"]` 提供。初始候选只检查
`abs(candidate_joint5 - home_joint5) <= joint5_home_max_delta_rad + joint5_limit_epsilon_rad`，不把观察位
joint5 当作抓取安全原点；候选选定以后，approach、pregrasp、grasp 和 lift 才使用阶段连续性门阻止真正的
半圈翻腕。Hermes 与监督式客户端都向同一个 `/manipulation/execute_pick` 发送 `PickObject` goal。

MoveIt LMA 仍使用 `position_only_ik: true`。只有初始 FK 接近轴超过 25°硬阈值时，执行器才把上述
当前 SO101 profile 不启用 `moveit_orientation_search`。LMA 已配置为 position-only，OrientationConstraint
不会为 5-DOF 机械臂创造额外自由度，反而可能把本应由最终 FK 门禁解释的姿态误差变成 `NO_IK_SOLUTION`。
最终 IK/FK 结果仍必须通过 25°/20° 硬门禁。

可靠开口参数由 `prepared_candidate_scoring` 声明，因为准备阶段是从这个块里读取它们的：

```yaml
prepared_candidate_scoring:
  reliable_max_opening_m: 0.072
  moving_finger_min_clearance_m: 0.003
```

这两个值进入 `fixed_finger_envelope_score`，按 `moving_gap = reliable_max_opening_m - far_extent` 折算成
`moving_score`，再与 `fixed_score` 加权成候选软分数 —— **它们是排序偏好，不是硬拒绝**。把它们写到
`target_gripper` 下面会让准备阶段读不到，静默回落到硬编码默认值。

真正的固定指硬拒绝是 `target_gripper.fixed_finger_robust_gap`，在降到抓取位姿之后、闭合之前用实测
接触残差评估一次；不通过则撤回 pregrasp 并以 `FIXED_FINGER_ROBUST_GAP_REJECTED` 继续 candidate
fallback。准备阶段目前没有对应的前置硬门，规划位姿固定指间隙已为负的候选仍会被执行一次才被拒。

闭合轴的 180° 对称只表示两指闭合直线相同，不表示固定指身份可忽略。执行器必须再用实际 FK 位姿执行
`fixed_finger_base_side` 硬检查；固定指仍在外侧或目标宽度区间缺失时拒绝当前候选并继续 candidate fallback。

候选准备加速由同一 `grasp_execution.ik` 配置控制：

```yaml
ik:
  worker_count: 4
  worker_namespace_prefix: /ik_worker
  auto_start_workers: true
```

`embodied_pipeline.launch.py` 会自动启动对应数量的隔离 MoveIt worker。正式执行器为整批候选
固定一份共同 `/joint_states` seed，并按候选排名固定 round-robin 分配到 worker，再按原候选顺序
合并结果；最终补偿与运动不进入 worker。候选进入 worker 前的 SO101 mesh/tabletop 检查
也由该执行器的唯一公共实现完成，使用凸包缓存和批量向量化路径。

SO101 真机的 MoveIt 完成语义也由同一 robot YAML 的 `moveit` 域声明：

```yaml
moveit:
  motion_status_hold_s: 0.0
  motion_feedback_timeout_s: 0.3
  motion_feedback_tolerance_rad: 0.12
  motion_require_tf_sync: true
  motion_hardware_feedback_topic: /so101_follower/joint_currents
```

网关只在 MoveIt 终态之后同时看到新的硬件读取心跳、收敛关节样本和覆盖该样本时间戳的末端 TF 时
返回成功，不再依赖固定 `0.3 s` sleep。由该屏障覆盖的 `contact_realign.settle_sec` 和
`pose_diagnostics.settle_sec` 可设为 `0.0`；相机与夹爪稳定等待不在此屏障覆盖范围内。仿真启动会
清空硬件心跳话题，只保留关节与 TF 屏障。

`robot.grasp_execution.prepared_candidate_scoring` 控制 IK/FK 后软排序。SO101 使用候选目标宽度区间和
候选规划姿态计算固定指到目标前缘的间隙，并以动态 margin 为期望值计算包络分数。该分数与
接触点 XY/Z 质量、目标体积质心距离和 GraspGen 置信度加权。`robust_gap_headroom_weight` 和
`robust_gap_headroom_scale_m` 进一步把 IK/FK 预测接触残差对应的安全间隙余量归一化后纳入排序；
软排序本身只改变执行顺序；启用 `fixed_finger_robust_gap` 时，同一预测余量还会在运动前执行独立硬门禁。

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
`target.arm_command_topic` 与 `target.gripper_command_topic` 同时构成控制权边界：每个
活动设备必须独占它实际发布的手臂和夹爪控制器话题，任一话题被两个活动设备共享都会在
启动前报错。SO-101 VR 的夹爪话题由 `vr_config.so101_gripper_topic` 指定；其 Placo 手臂
输出当前使用标准 `/arm_position_controller/commands`。因此双臂等多设备配置必须同时拆分
arm 和 gripper 两类话题，不能只拆分手臂话题。
当前 SO-101 Placo 是按目标手臂唯一的执行资源，同一 launch 最多选择一个 Phone、
SO-101 VR 或 Xbox Cartesian 输入；多个 leader 等非 Cartesian 输入不受此限制。

手机设备只使用内置 WebPhone；显式配置时 `phone_config.backend` 必须为 `webphone`。HTTPS/WSS
端口、TLS 文件和 `command_stale_s` 必须定义在 `phone_config.web` 中；启动构建器会
校验端口、超时和证书/密钥配对，并将设备级 `control_frequency` 原样传给节点。
WebPhone 还要求 `command_stale_s + 1 / control_frequency <= 0.22s`；该式约束
“检测并发起 stop 请求”的时间，不是机械臂物理停止的硬实时保证。默认 stale 为
0.18s，50Hz 配置可显式使用 0.2s。
WebPhone 统一使用 Placo clutch 相对位姿契约：浏览器可提供 WebXR AR 位姿，或在不支持
WebXR 时提供光流虚拟位姿；两条跟踪路径都不再把位姿微分成速度。
Phone 因此要求 `teleoperation.cartesian.solver: placo_servo`，不提供速度模式
开关；旧配置中 `input_mode: pose` 仍可读取，`velocity` 会在启动前报错。Phone 使用相对
基线位姿，因此 `end_effector_bounds` 的每个轴必须严格满足 `min < 0 < max`。
手机使用的 Placo `position_only` 位于
`teleoperation.cartesian.placo_servo.position_only`；launch 会把解析后的配置传给手机的
Placo 链路。遥操作 Home 的唯一目标来自 `ros2_control.reset_positions`；launch 按所选手臂
关节顺序校验并注入 Placo。Phone 与 SO-101 VR 都调用同一个
`/so101_placo_servo_node/return_home` Action，并等待新鲜 `/joint_states` 的最大关节误差连续
稳定后得到终态，不再使用 Cartesian named pose、状态话题或固定等待时间。launch 还会将
`command_stale_s` 注入 Phone 专用 Placo 命令租约；VR/Xbox 的 YAML 默认租约为关闭。
该租约只在 Phone 本周期取得有效控制命令或正在执行受控 Home 时续租；空输入、传输/转换
异常不会用“进程仍存活”冒充有效命令。Home 要求 `reset_positions` 覆盖全部选中手臂关节、
数值有限且位于 Placo 关节限位内；Phone 启动会拒绝 MoveIt Servo，因为 Phone 固定使用
Placo pose 且 Home 由 Placo 执行。Home 终态后仍要求
deadman 松开再按，运行期间夹爪保持最后目标。launch 还会把 `safety.estop_topic` 同时注入
Placo 与独立 VR 节点，使该路径满足 `E-stop > Home/start/pose/twist`，不依赖 `TeleopNode` 转发急停。
手机 pose 模式还必须至少启用 WebXR AR 或光流降级之一。

WebPhone 不提供账号鉴权，仅支持受信内部网络。Origin 和单客户端限制不等于身份认证；
禁止公网映射、反向/云隧道、访客 Wi-Fi 和不可信 VPN，建议通过独立控制网段及防火墙限制
HTTP/WSS 端口来源，并在不使用遥操时停止服务。

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

### 真机手眼配置

SO101 抓取使用同级独立配置 `config/robots/so101_handeye_realsense_grasp.yaml`。用户应直接在
这份 YAML 中填写从动臂串口、相机序列号和 leader 配置；`scripts/handeye_calibrator.py` 质量检查
通过后会就地更新 `peripherals[name=wrist].transform`。多台物理机器人应分别复制独立 YAML，避免
不同实例的端口和标定值互相覆盖。`config_path` 仍可用于加载 workspace 外部的完整 robot YAML，
但不再支持第三层 overlay 合成。

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
robot:
  name: example_robot
  # The selected Gateway mode must be a key in control_modes.
  control_modes:
    moveit_planning: {}
  skill_required_control_mode: moveit_planning

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
          category: social_greeting                            # 任意非空字符串
          when_to_use: ["greet someone", "say hi or bye"]    # 非空字符串列表
          aliases_zh: ["打招呼", "挥手"]                      # description / 规则解析器元数据，不属于公开 catalog
          aliases_en: ["hello", "wave"]
          motion_scope: [wrist]                                # base|shoulder|elbow|wrist|gripper|arm
          anchor_pose: home                                    # 必须在 named_poses 中已定义，或 "none"
          intensity: moderate                                  # subtle|moderate|large
          duration_sec_estimate: 8.0                           # 含 1.0 秒初始夹爪归一化并留有余量
          requires_motion_params: false                        # 布尔
          rule_entry: true                                     # 是否把 aliases_zh 注入规则解析器
        capability:
          schema_version: 1
          summary: "Wave hello or goodbye with the wrist."
          domain: social_greeting
          moves_robot: true
          required_control_mode: moveit_planning
          parameters:
            type: object
            additionalProperties: false
            properties: {}
            required: []
          recovery_policy: never_retry
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

#### Capability Gateway 公开契约

`robot.embodied.skill_templates.<skill>.capability` 是 Capability Gateway 的公开 SSOT；它不是从
`description` 或 primitive sequence 派生。所有未禁用的模板都必须声明 `schema_version: 1` 以及
`summary`、`domain`、`moves_robot`、`required_control_mode`、`parameters` 和 `recovery_policy`。

| 字段 | 校验规则 |
| --- | --- |
| `summary` / `domain` | 非空字符串 |
| `moves_robot` | 布尔值 |
| `required_control_mode` | `teleop`、`model_inference` 或 `moveit_planning`，且必须等于机器人级 `skill_required_control_mode` |
| `parameters` | 严格 object schema：`type: object`、`additionalProperties: false`；只允许 `target_name`、`place_name`、`motion_direction`、`motion_distance` 属性，`required` 必须唯一且引用已声明属性 |
| `recovery_policy` | `never_retry`、`ask_user` 或 `recover_safe_pose` |

字符串参数只能使用 `type` 与非空 `enum`；`motion_direction` 的 enum 只能包含六个方向
`forward`、`backward`、`left`、`right`、`up`、`down`。`motion_distance` 必须是 `type: number`、
`exclusiveMinimum: 0`，并以 `meters` 或 `degrees` 标明 `unit`。其他 schema key 或请求属性会在加载时被拒绝。

`load_robot_config_dict()` 是 launch、CLI 和 catalog 共用的规范化加载入口；它会在返回配置前执行上述
capability、模板和 Gateway 一致性校验。若 `embodied.skill_templates` 非空，
`skill_required_control_mode` 必须是 `control_modes` 的非空成员；每个启用能力的
`required_control_mode` 必须与它完全相等。

共享配置解析的选择顺序为：显式 `config_path`、显式 `config_name`、`ROBOT_CONFIG`、`ROBOT_NAME`、
默认 `so101_single_arm`。按名称查找时先查已安装 `robot_config` 的 `config/robots/`，再查源码树的
`config/robots/`；显式路径必须存在。公开 catalog 仅输出 capability 字段、命名位姿名称、timeout policy
和 digest。primitive sequence、关节/笛卡尔坐标、目标绑定、ROS service/action/topic 名称仍是私有实现数据。

#### skill_templates 与 description 契约

`embodied.skill_templates` 是技能执行的单一事实来源。每个 skill 模板可挂一个
结构化 `description` 块，用于人类可读的发现信息与中文别名；公开 catalog 的字段以相邻的
`capability` 块为准，且不包含 `description` 或其中的 aliases。
`aliases_zh` 是 description 与规则解析器元数据，不会出现在 `robot-skill` 或其他公开 capability catalog 中。
仅当 `rule_entry: true` 且 `requires_motion_params: false` 时，launch 才会把这些别名注入规则解析器。
`robot_config.loader._validate_skill_description` 在加载时强校验字段类型与受控词表。

受控词表：
- `motion_scope`：`base` / `shoulder` / `elbow` / `wrist` / `gripper` / `arm`
- `intensity`：`subtle` / `moderate` / `large`
- `anchor_pose`：必须引用 `named_poses` 中已定义的位姿，或填 `"none"`
- `rule_entry`：可选布尔值；设为 `true` 时允许无动态运动参数的技能进入确定性规则入口
- `disabled`：可选布尔值；设为 `true` 时从 planner allowlist、规则入口、resolver、safety guard
  和 `robot-skill` catalog/validate/execute 的启用技能集合中统一排除
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
  align_depth: true
  # 高带宽 image/pointcloud 由驱动直接改名，避免通过第二个 DDS relay 复制。
  # CameraInfo 仍使用轻量 relay，把 frame_id 规范为 optical_frame_id。
  direct_topic_remap: true
  # 仅在确有 PointCloud2 消费者时开启；RGB-D 抓取可直接从对齐深度构造点云。
  enable_pointcloud: false
```

`direct_topic_remap` 默认是 `false`，保留旧的全 relay 行为。对单机高带宽 RealSense pipeline，建议设为
`true`；下游仍使用 `/camera/{name}/image_raw` 和
`/camera/{name}/aligned_depth_to_color/{image_raw,camera_info}`，不需要了解驱动原生 topic 前缀。

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
