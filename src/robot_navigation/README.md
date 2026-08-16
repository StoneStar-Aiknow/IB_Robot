# robot_navigation

机器人导航包，集成了语音识别、Nav2 导航、全向轮底盘桥接和定位融合功能。

## 功能特性

- **语音控制**: 基于 `voice_asr_service` (sherpa-onnx 本地 ASR) + `voice_control` 关键词匹配的语音导航
- **导航控制**: Nav2 导航 Goal 客户端，支持语音触发导航，到达后自动触发机械臂推理
- **底盘桥接**: `cmd_vel_bridge_node` 通过 IK/FK 将标准 `/cmd_vel` 桥接到 ros2_control 全向轮速度指令 (rad/s)，并发布里程计
- **定位融合**: EKF (robot_localization) 融合底盘里程计速度，RTAB-Map 视觉 SLAM 提供全局定位修正
- **建图保存**: `save_rtabmap_map` 包装 Nav2 map_saver，将 `/rtabmap/map` 保存为 `map.yaml/map.pgm`
- **任务联动**: 语音 → 导航 → 到达 → 触发 action_dispatcher 评估，形成完整任务链

## 系统架构

```text
  用户语音 ──► voice_asr_node (sherpa-onnx) ──► /voice_command
                                                     │
                                                     ▼
                                              voice_control (关键词匹配)
                                                     │
                                                     ▼
                                         /voice_asr/keyword_matched (JSON)
                                                     │
                                                     ▼
                                            nav2_goal_client
                                                     │
                                                     ▼
                                         Nav2 NavigateToPose Action
                                                     │
                                                     ▼
                                        Nav2 规划 + 发布 /cmd_vel
                                                     │
                                                     ▼
                                        到达目标 ──► /action_dispatcher/start_evaluate

  /cmd_vel ─────► cmd_vel_bridge_node (IK) ──► /base_velocity_controller/commands (rad/s)
                    │  (FK)
                    └──► /odom (nav_msgs/Odometry)
                              │
                              ▼
                    EKF (robot_localization) ──► TF: odom → base_link (30Hz)
                                                   │
  RTAB-Map (视觉 SLAM) ──► TF: map → odom (~1Hz) ◄──┘
```

### TF 树结构

```text
map ──(RTAB-Map)──► odom ──(EKF)──► base_link ──► ... ──► sensor frames
```

| TF 变换 | 发布者 | 频率 | 说明 |
|---------|--------|------|------|
| `map → odom` | RTAB-Map | ~1Hz | 视觉 SLAM 全局定位修正 |
| `odom → base_link` | EKF | 30Hz | 融合底盘里程计速度，平滑输出 |

注意: `cmd_vel_bridge_node` 仅发布 `/odom` 话题（供 EKF 订阅），不发布 TF（`publish_tf: false`）。
Nav2 使用静态地图与 RTAB-Map 提供的 `map → odom` 定位结果，不并行启动 AMCL，避免多个节点竞争发布同一全局定位 TF。

## 节点列表

| 节点 | 功能 | 入口点 |
|------|------|--------|
| `voice_control` | 语音关键词匹配 + 导航桥接 (sherpa-onnx 本地 ASR) | `robot_navigation.voice_control` |
| `nav2_goal_client` | Nav2 导航 Goal 客户端 + 评估触发 | `robot_navigation.nav2_goal_client` |
| `cmd_vel_bridge_node` | cmd_vel → ros2_control 桥接 + 里程计发布 | `robot_navigation.cmd_vel_bridge_node` |
| `save_rtabmap_map` | 保存 RTAB-Map OccupancyGrid 地图 | `robot_navigation.save_rtabmap_map` |

## 使用入口

先按目的选择入口；完整机器人链路统一由 `robot_config` 启动，`robot_navigation` 只保留导航子系统、节点和 PC 端 RViz 预设。

| 你要做什么 | 看哪一节 | 推荐命令入口 |
|---|---|---|
| 复刻建图与导航环境 | [环境复刻](#环境复刻) | `robot_config:=lekiwi_mapping` / `robot_config:=lekiwi_navi` |
| 跑本包测试 | [测试验证](#测试验证) | `colcon test --packages-select robot_navigation` |
| 验证完整 Gazebo/Nav2 仿真 | [测试验证](#测试验证) | `NAV_TEST_PROFILE=full colcon test --packages-select robot_config ...` |
| 实机建图 | [建图流程](#建图流程) | `ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_mapping` |
| 实机导航 | [导航流程](#导航流程) | `ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_navi control_mode:=teleop` |
| PC 端观察 | [PC 端 RViz 观察](#pc-端-rviz-观察) | `lekiwi_mapping_rviz.launch.py` / `lekiwi_navigation_rviz.launch.py` |
| 单独调试导航节点 | [底层调试入口](#底层调试入口) | `ros2 run robot_navigation ...` |

`robot_config` 的优势：YAML 单一数据源，自动启动控制器、相机、TF、定位、Nav2 和导航节点，并通过 `control_mode` 切换运行模式。

## 环境复刻

### 1. 运行环境选择

LeKiwi 建图和导航支持两类运行方式：

| 模式 | ROS 节点运行位置 | Ubuntu PC 角色 | 适用场景 |
|---|---|---|---|
| openEuler 开发板主运行 | 资源较充足的开发板 | SSH + RViz 远程观察 | 实机复刻、板端验证 |
| Ubuntu 单机运行 | Ubuntu PC / 笔记本 | 运行所有节点 + RViz | 本机调试、算法验证、没有开发板的联调 |

跨机器部署时，开发板和 PC 必须网络互通，并使用相同的 `ROS_DOMAIN_ID`。openEuler 方案下 Ubuntu 仍然是必需的观察端；Ubuntu 方案下 Ubuntu 是运行端。

典型组合：

| 组合 | ROS 节点跑在 | Ubuntu 角色 | 网络连接 | 设备接入 | 小车能否自由跑 |
|---|---|---|---|---|---|
| A | openEuler @ 开发板 | SSH + RViz | 网线接 Ubuntu PC | RealSense/底盘接开发板 | 否，小车在脚边 |
| B | openEuler @ 开发板 | SSH + RViz | 网线接 Ubuntu 笔记本 | RealSense/底盘接开发板 | 是，笔记本跟着小车 |
| C | openEuler @ 开发板 | SSH + RViz | WiFi | RealSense/底盘接开发板 | 是，PC 不动 |
| D | Ubuntu @ 笔记本 | 运行 + RViz | 无跨机通信 | RealSense/底盘接笔记本 | 是，笔记本跟着小车 |
| E | Ubuntu @ PC | 运行 + RViz | 无跨机通信 | RealSense/底盘接 PC | 否，小车在脚边 |

推荐先用组合 A 验证主链路：网线稳定，小车在脚边，小范围完成建图、保存地图、加载地图和导航。需要小车真正自由跑时，再切到 B/C/D。

### 2. 终端环境

每个新终端都需要先加载 ROS 和工作区环境。具体加载方式参考 IB Robot 仓主目录下的 README，并确保开发板和 PC 使用相同的 `ROS_DOMAIN_ID`，例如 `<your_id>`。

### 3. 编译

编译方式参考 IB Robot 仓主目录下的 README。本文件只说明导航相关入口、测试和排查方法，不重复维护完整工作区编译命令。

### 4. 硬件与网络注意事项

- 使用 RealSense 深度相机时，必须接 USB 3.0 蓝口，不建议使用长 USB 延长线；深度流数据量大，USB 2.0 或长线容易导致掉帧、重枚举或无图像。
- 底盘控制链默认走 `/dev/ttyACM0`，插拔顺序变化时可能变成 `/dev/ttyACM1`。
- 推荐计算侧和执行侧分开供电：开发板 + USB 外设一套电源，底盘/机械臂电机一套电源，避免电机瞬时电流导致板子重启或 RealSense 掉线。
- 完整导航链路会同时运行 RealSense、RTAB-Map、ros2_control、Nav2 和相关桥接节点，对 CPU 与内存资源要求较高；建议使用具备较高算力和较大内存资源的开发板，资源水平至少接近中高端边缘计算板，才能更稳定地跑完整链路。
- WiFi 下如果 SSH/DDS 卡顿，优先确认两端 `ROS_DOMAIN_ID` 一致，再检查 DDS 实现、网卡绑定、单播 peers、路由器频段和 RealSense 带宽配置。

### 5. robot_config 入口

| 入口 YAML | 用途 | 启动命令 |
|---|---|---|
| `lekiwi_mapping.yaml` | base-only 建图：底盘 + RealSense + RTAB-Map mapping + cmd_vel_bridge | `ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_mapping` |
| `lekiwi_navi.yaml` | 完整导航：加载静态地图 + Nav2 + RTAB-Map localization + EKF + cmd_vel_bridge | `ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_navi control_mode:=teleop` |

建图和导航使用独立配置文件。`lekiwi_mapping` 只负责建图主链；`lekiwi_navi` 消费保存好的地图做定位与导航。

`lekiwi_navi.yaml` 中导航相关配置形态如下：

```yaml
navigation:
  enabled: true
  nav2_bringup:
    enabled: true
    map_file: "$(env HOME)/.ros/ibrobot/maps/rtabmap.yaml"
  ekf_rtabmap:
    enabled: true
    rtabmap:
      rtabmap_args: "--Mem/InitWMWithAllNodes true --Mem/IncrementalMemory true --Mem/PermanentMemory false --Mem/STMSize 8 --Reg/Force3DoF true"
  cmd_vel_bridge:
    enabled: true
    publish_tf: false         # EKF 发布 TF，bridge 只发布 /odom 话题
    max_radps: 4.602          # 最大轮速 (rad/s)
    cmd_vel_topic: /cmd_vel
    joint_states_topic: /joint_states
    odom_topic: /odom
  robot_navigation:
    enabled: true
    enable_voice_control: true
    destinations:
      point_a: {x: 0.0, y: 0.2, theta: 1.5708}  # rad (90 deg)
      point_b: {x: 0.2, y: 0.0, theta: 0.0}
  rviz:
    enabled: true
```

## 建图流程

建图入口使用 `lekiwi_mapping.yaml`，链路包含 base-only ros2_control、RealSense、TF、RTAB-Map mapping 和 `/cmd_vel` 底盘桥接。它不会自动启动键盘遥控，需要另一个终端发布 `/cmd_vel`。

终端 A 启动建图主链：

```bash
ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_mapping
```

终端 B 启动键盘遥控：

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

`teleop_twist_keyboard` 默认发布到 `/cmd_vel`，会被 `lekiwi_mapping.yaml` 中的 `navigation.cmd_vel_bridge.cmd_vel_topic: /cmd_vel` 消费。常用键位：`i` 前进、`,` 后退、`j` 左转、`l` 右转、`k` 停止；`w/x/e/c/q/z` 是速度倍率调整键。

终端 C 保存地图：

```bash
ros2 run robot_navigation save_rtabmap_map -f ~/.ros/ibrobot/maps/rtabmap
```

默认等价于：

```bash
ros2 run nav2_map_server map_saver_cli -t /rtabmap/map -f ~/.ros/ibrobot/maps/rtabmap
```

输出约定：

```text
~/.ros/ibrobot/maps/rtabmap.yaml     # Nav2 map_server 加载
~/.ros/ibrobot/maps/rtabmap.pgm      # 占据栅格图像
~/.ros/ibrobot/maps/rtabmap.db       # RTAB-Map localization 复用
```

后续导航默认从 `lekiwi_navi.yaml` 的 `navigation.nav2_bringup.map_file: "$(env HOME)/.ros/ibrobot/maps/rtabmap.yaml"` 加载地图。

## 导航流程

启动导航主链：

```bash
ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_navi control_mode:=teleop
```

RViz 启动后，使用 `2D Goal Pose` 或 `Nav2 Goal` 在地图上点击目标。Nav2 会规划全局路径并通过 `/cmd_vel` 驱动底盘，局部规划器会根据代价地图避障。

如果需要导航评估模式并联动推理链：

```bash
ros2 launch robot_config robot.launch.py use_sim:=false robot_config:=lekiwi_navi control_mode:=navi
```

## PC 端 RViz 观察

开发板上已经启动 `lekiwi_mapping` 或 `lekiwi_navi` 后，PC 端不需要在板端本地打开 RViz。只要网络互通且 `ROS_DOMAIN_ID` 一致，PC 可以直接观察远端 ROS 图。

PC 端同样需要先加载 ROS 和工作区环境，具体方式参考 IB Robot 仓主目录下的 README。如果 PC 本地也有同一份工作区，并希望使用工作区里的 launch/config，再 source 该 overlay。两端 `ROS_DOMAIN_ID` 必须一致，例如 `<your_id>`。

先确认能看到远端 ROS 图：

```bash
ros2 topic list
ros2 topic echo /tf
```

建图观察：

```bash
ros2 launch robot_navigation lekiwi_mapping_rviz.launch.py
```

该预设打开 `TF`、`RobotModel`、`/rtabmap/map`、`/rtabmap/cloud_map`。

导航观察：

```bash
ros2 launch robot_navigation lekiwi_navigation_rviz.launch.py
```

该预设打开 `TF`、`RobotModel`、`/map`、`/plan`。

自定义 RViz 配置：

```bash
ros2 launch robot_navigation lekiwi_mapping_rviz.launch.py rviz_config:=/path/to/custom.rviz
```

## 单点验证

如果一键 `robot_config` 入口因相机型号、内核版本、DDS 或 TF 问题失败，先按下面顺序做单点验证。

### 1. RealSense standalone 出流

```bash
ros2 launch realsense2_camera rs_launch.py
```

RealSense 的 profile、分辨率、帧率、同步和深度对齐参数可以根据具体设备、算力和场景按需修改与调优。单点验证时重点确认彩色图、深度图和 camera_info 等基础话题能够稳定发布，并且 TF 与时间戳能被下游 RTAB-Map 正常消费。

### 2. 静态 TF

```bash
ros2 run tf2_ros static_transform_publisher \
  0 0 0.3 0 0 0 base_link camera_link
```

这只用于单点验证 RTAB-Map。正式建图/导航时，`robot_config` 会按 `lekiwi_*.yaml` 中 `peripherals.realsense.transform` 自动发布。

### 3. RTAB-Map 消费 RGBD

```bash
ros2 launch rtabmap_launch rtabmap.launch.py \
  rgb_topic:=/camera/camera/color/image_raw \
  depth_topic:=/camera/camera/depth/image_rect_raw \
  camera_info_topic:=/camera/camera/color/camera_info \
  frame_id:=camera_link \
  approx_sync:=true \
  queue_size:=20 \
  rtabmap_viz:=false \
  rviz:=false \
  database_path:=/tmp/rtabmap_true_$(date +%Y%m%d_%H%M%S).db \
  rtabmap_args:="--Mem/InitWMWithAllNodes true --Mem/IncrementalMemory true --Mem/PermanentMemory false --Mem/STMSize 8"
```

判断 RTAB-Map 真的吃到数据：日志出现 `Odom: quality=...`，`rtabmap` 主节点 `Rate=1.00s`。如果一直 `Did not receive data since 5 seconds!`，先检查 topic 名和 TF。`approx_sync=false` 可作为板端 fallback，但默认配置仍保留 `true`。

### 4. 测试卫生

每轮前检查真实进程、ROS graph 和 graph cache，避免残留进程污染：

```bash
ps -ef | grep -E 'realsense2_camera_node|rtabmap|rgbd_odometry|static_transform_publisher' | grep -v grep
ros2 node list
ros2 topic list
ros2 daemon stop
ros2 daemon start
```

旧 `/tmp/rtabmap.db` 可能引入脏库错误。每轮单点验证用新的 `database_path:=/tmp/rtabmap_..._$(date ...).db`。测板端频率时先关闭 `rqt_image_view` 等 PC 侧 viewer，避免外部订阅者影响测量。

## 底层调试入口

本节用于排查 `robot_config` 完整链路之外的导航子系统问题，例如单独确认 Nav2 子系统、RViz 预设或某个 `robot_navigation` 节点是否可运行。正常建图和导航仍优先使用前文的 `robot_config` 入口。

```bash
# Nav2 子系统入口：map_server + Nav2 navigation_launch.py
# 通常由 robot_config 根据 lekiwi_navi.yaml 间接包含
ros2 launch robot_navigation nav2_bringup.launch.py

# 指定地图
ros2 launch robot_navigation nav2_bringup.launch.py map:=/path/to/rtabmap.yaml

# PC 端打开建图观察 RViz 预设
ros2 launch robot_navigation lekiwi_mapping_rviz.launch.py

# PC 端打开导航观察 RViz 预设
ros2 launch robot_navigation lekiwi_navigation_rviz.launch.py

# 单独运行节点
ros2 run robot_navigation voice_control
ros2 run robot_navigation nav2_goal_client
ros2 run robot_navigation cmd_vel_bridge_node
ros2 run robot_navigation save_rtabmap_map

# Legacy/debug-only EKF + RTAB-Map entry. For full LeKiwi bringup, use robot_config.
ros2 launch robot_navigation ekf_rtabmap_launch.py
```

直接启动 `nav2_bringup.launch.py` 时，调用方需要确保以下节点已经由 `robot_config` 或其他入口启动：

1. `robot_state_publisher` 和 `/tf_static`
2. ros2_control、controller spawner 与 `/joint_states`
3. `/odom`、`map -> odom` 或等价定位链路
4. 需要时单独启动 `nav2_goal_client`、语音节点和 RViz

## 测试验证

`colcon test` 用于单点验证某个包或模块的功能，便于把 `robot_navigation` 的 standalone 测试和 `robot_config` 维护的完整仿真测试分开执行。

### 1. 通用编译

```bash
colcon build --packages-up-to robot_config robot_navigation --base-paths src --packages-skip sim_models
```

### 2. robot_navigation standalone 测试

这些测试不启动完整机器人、Gazebo 或 Nav2 bringup，适合 Ubuntu 和 openEuler 都执行。

```bash
colcon test --packages-select robot_navigation --base-paths src --event-handlers console_direct+
```

预期结果：`robot_navigation` 单元测试与软件闭环 E2E 通过。

### 3. robot_config minimal 测试

openEuler、无 Gazebo 或无桌面环境跑 minimal profile。该 profile 会显式跳过 Gazebo/Nav2 完整仿真，只验证测试发现和 skip 策略。

```bash
NAV_TEST_PROFILE=minimal colcon test --packages-select robot_config --base-paths src --event-handlers console_direct+ --pytest-args -k test_navigation_simulation
```

预期结果：`robot_config` navigation simulation 测试 `3 skipped`，skip 原因为 minimal profile 禁用 Gazebo 仿真。

### 4. robot_config full 仿真测试

Ubuntu 且已安装 Gazebo/ros_gz 时运行 full profile。该测试由 `robot_config` 维护，启动 Gazebo、ros2_control、控制器、Nav2、`robot_navigation` 节点和测试辅助节点。

```bash
NAV_TEST_PROFILE=full colcon test --packages-select robot_config --base-paths src --event-handlers console_direct+ --pytest-args -k test_navigation_simulation
```

预期结果：3 个 Gazebo/Nav2 导航 E2E 通过。测试进程会打印临时日志目录，例如 `/tmp/robot_config_nav_sim_xxx`，其中包含 `robot_launch.log`、`odom_bridge.log`、`gt_odom_node.log` 和 `cmd_vel_relay.log`。

### 5. Gazebo GUI 可视化测试

GUI 不是必跑项，只用于 Ubuntu 桌面环境人工观察 Gazebo 窗口。pytest 仍会自动发送导航目标，Gazebo 窗口只负责可视化。

```bash
SIM_GUI=1 NAV_TEST_PROFILE=full colcon test --packages-select robot_config --base-paths src --event-handlers console_direct+ --pytest-args -k test_navigation_simulation
```

## ROS 接口

### voice_control

| 类型 | 话题/服务 | 类型 | 方向 | 说明 |
|------|-----------|------|------|------|
| 订阅 | `/voice_command` | `std_msgs/String` | 输入 | voice_asr_node 输出的识别文本 |
| 订阅 | `/voice_asr/keywords` | `std_msgs/String` | 输入 | 动态更新关键词 (JSON) |
| 发布 | `/voice_asr/keyword_matched` | `std_msgs/String` | 输出 | 匹配的关键词 (JSON) |
| 发布 | `/voice_asr/nav_stop` | `std_msgs/String` | 输出 | 停止导航命令 |
| 服务客户端 | `/action_dispatcher/stop_evaluate` | `std_srvs/Trigger` | 调用 | 匹配 "停止" 时调用 |
| 服务客户端 | `/voice_asr_node/set_hotwords` | `ibrobot_msgs/srv/SetHotwords` | 调用 | 启动时注册热词 |

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `topic_text` | `/voice_command` | voice_asr_node 输出话题 |
| `topic_keyword_matched` | `/voice_asr/keyword_matched` | 匹配结果输出话题 |
| `topic_nav_stop` | `/voice_asr/nav_stop` | 导航停止话题 |
| `keywords_json` | `{}` | 关键词 JSON（从 launch 传入） |
| `keywords_file` | `""` | 关键词 JSON 文件路径 |
| `destinations_json` | `{}` | 目的地名称 → 坐标映射 |

### nav2_goal_client

| 类型 | 话题/服务 | 类型 | 方向 | 说明 |
|------|-----------|------|------|------|
| 订阅 | `/voice_asr/keyword_matched` | `std_msgs/String` | 输入 | 语音命令 |
| 订阅 | `/voice_asr/nav_stop` | `std_msgs/String` | 输入 | 语音停止导航命令 |
| Action | `navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | 调用 | Nav2 导航目标 |
| 服务客户端 | `/action_dispatcher/start_evaluate` | `std_srvs/Trigger` | 调用 | 到达后触发评估 |

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `x` | `0.0` | 默认目标 X 坐标 |
| `y` | `0.0` | 默认目标 Y 坐标 |
| `theta` | `0.0` | 默认目标朝向 (弧度) |
| `timeout_sec` | `60.0` | 导航超时 (秒) |
| `global_frame` | `map` | 目标坐标系 |
| `enable_feedback` | `true` | 是否打印 Nav2 反馈 |
| `trigger_evaluation` | `false` | 到达后是否触发 action_dispatcher 评估 |
| `subscribe_voice` | `true` | 是否订阅语音命令 |
| `topic_keyword_matched` | `/voice_asr/keyword_matched` | 语音命令话题 |
| `topic_nav_stop` | `/voice_asr/nav_stop` | 语音停止导航话题 |

### cmd_vel_bridge_node

| 类型 | 话题 | 类型 | 方向 | QoS | 说明 |
|------|------|------|------|-----|------|
| 订阅 | `/cmd_vel` | `geometry_msgs/Twist` | 输入 | Reliable | 速度指令 (vx, vy, vtheta) |
| 订阅 | `/joint_states` | `sensor_msgs/JointState` | 输入 | Best Effort | 轮子反馈 (joints "7", "8", "9") |
| 条件订阅 | `/motion_mode/navigation_enabled` | `std_msgs/Bool` | 输入 | Transient Local | 导航命令授权；抓取模式持续输出零速 |
| 发布 | `/base_velocity_controller/commands` | `std_msgs/Float64MultiArray` | 输出 | Reliable | 原始轮速 [left, back, right] rad/s |
| 发布 | `/odom` | `nav_msgs/Odometry` | 输出 | Reliable | 里程计 |
| 条件发布 | `/motion_mode/base_navigation_enabled` | `std_msgs/Bool` | 输出 | Reliable | 已清除旧命令并输出零速的模式确认 |
| 条件发布 | TF: `odom → base_link` | TransformStamped | 输出 | - | 仅当 `publish_tf: true` 时发布 |

**参数**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `wheel_radius` | `0.05` | 轮子半径 (m) |
| `base_radius` | `0.125` | 底盘半径 (m) |
| `max_radps` | `4.602` | 最大轮速 (rad/s)，对应 base_vel_max_raw=3000 steps/s |
| `odom_frame` | `odom` | 里程计坐标系 |
| `base_frame` | `base_link` | 机器人基坐标系 |
| `publish_tf` | `true` | 是否发布 TF（启用 EKF 时应设为 `false`） |
| `control_frequency` | `50.0` | 控制频率 (Hz) |
| `cmd_timeout` | `0.5` | 无指令超时后归零输出 (s) |
| `cmd_vel_topic` | `/cmd_vel` | 订阅的速度指令话题 |
| `joint_states_topic` | `/joint_states` | 订阅的关节状态话题（轮子反馈） |
| `odom_topic` | `/odom` | 发布的里程计话题 |
| `motion_mode_enabled` | `false` | 是否启用抓取/导航常驻互锁 |
| `navigation_enabled_on_startup` | `true` | 互锁启用时的初始底盘授权状态 |
| `navigation_enabled_topic` | `/motion_mode/navigation_enabled` | Gateway 发布的模式状态 |
| `navigation_mode_ack_topic` | `/motion_mode/base_navigation_enabled` | 底盘模式确认话题 |

**运动学**: 三个全向轮安装角度分别为 150°、-90°、30°（对应 joint 7/left, 8/back, 9/right）。IK 通过 3×3 矩阵 `M @ [vx, vy, vtheta]` 计算轮速，输出单位为 rad/s（ros2_control 速度指令接口）。FK 通过 M 的伪逆将轮子反馈还原为机体速度。超过 `max_radps` 时按比例缩放所有轮速。

互锁启用后，抓取模式会忽略新的 `/cmd_vel` 并在每个控制周期发布三轮零速。任何模式切换都会先清除缓存速度，
因此进入导航后必须收到一条新的 `/cmd_vel` 才会移动，旧的 Nav2 指令不会在切换后恢复执行。

## 关键词配置

`config/keywords.json` 定义语音关键词（正则表达式）和动作映射：

```json
{
  "keywords": {
    "去.*a点|到.*a点|a点": {
      "type": "destination",
      "info": {"destination": "point_a"}
    },
    "捡.*蓝色方块|拿.*蓝色方块|蓝色方块": {
      "type": "action",
      "info": {"task_description": "Pick up the blue square"}
    },
    "停止|停下": {
      "type": "stop",
      "info": {"task_description": "Stop current action"}
    }
  }
}
```

`destination` 类型的 `destination` 字段通过 `destinations_json` 参数（来自 `robot_config`）解析为实际坐标 (x, y, theta)。

| 类型 | 作用 | info 字段 |
|------|------|-----------|
| `destination` | 触发导航 | `destination` — 在 destinations_json 中查找坐标 |
| `action` | 设置任务描述 | `task_description` — 到达后传给 action_dispatcher |
| `stop` | 停止导航和推理 | `task_description` — "Stop current action" |

## 定位融合 (EKF)

`config/ekf.yaml` 配置 `robot_localization` EKF 节点，30Hz，2D 模式：

| 输入源 | 话题 | 融合内容 | 作用 |
|--------|------|---------|------|
| 底盘里程计 | `/odom` | X/Y 速度 + 偏航角速度 | 高频实时速度（来自 cmd_vel_bridge FK） |

**设计说明**:
- EKF 仅融合底盘里程计速度（`/odom`），输出平滑的 `odom → base_link` TF
- **不**融合 RTAB-Map 视觉里程计（`/rtabmap/odom`），因为两者 child_frame 不同会导致冲突
- RTAB-Map 通过 `map → odom` TF 做全局定位修正，EKF 不参与该环节
- `publish_tf: true`：EKF 接管 `odom → base_link` TF 发布，`cmd_vel_bridge` 的 `publish_tf` 需设为 `false`

## RTAB-Map 配置

RTAB-Map 以定位模式运行（`localization: true`），通过视觉 SLAM 发布 `map → odom` TF。

关键参数：
- `frame_id: camera_link` — RTAB-Map 使用的相机坐标系，由 `robot_config` 的 RealSense adapter 和 TF 配置提供
- `approx_sync: true` — RGB 和 Depth 近似同步
- `--Reg/Force3DoF true` — 强制 2D 模式，防止相机倾斜导致的 pitch/yaw 偏差（对 2D 全向轮机器人至关重要）
- `--Mem/InitWMWithAllNodes true` — 初始化时加载所有节点
- `--Mem/STMSize 8` — 短期记忆大小
- 更新频率约 ~1Hz（`map → odom` TF），受视觉特征提取计算开销限制（`Vis/MaxFeatures: 1000`）

## Nav2 配置概要

`config/nav2_params.yaml` 配置完整 Nav2 栈：

| 组件 | 关键参数 |
|------|---------|
| **Localization** | 不启动 AMCL；RTAB-Map 使用保存地图发布 `map → odom`，EKF 发布 `odom → base_link` |
| **Controller** | DWB 局部规划器，20Hz，max vel 0.26 m/s，max theta 1.0 rad/s，goal tolerance xy: 0.05 / yaw: 0.1 |
| **Local costmap** | 3×3m 滚动窗口，0.05m 分辨率，robot radius 0.22m，`/scan` 话题，`transform_tolerance: 3.0` |
| **Global costmap** | Static + Obstacle + Inflation 层，`transform_tolerance: 3.0` |
| **Planner** | NavfnPlanner，tolerance: 0.5 |
| **Behaviors** | spin, backup, drive_on_heading, assisted_teleop, wait |
| **Velocity smoother** | max [0.26, 0.26, 1.0]，odom_topic: `/odometry/filtered` |

### 重要配置说明

- **`use_sim_time: False`**: 所有 Nav2 节点均使用系统时间。实车必须为 `False`，否则 TF 查找失败 ("Transform data too old")
- **`transform_tolerance: 3.0`**: RTAB-Map 更新频率约 1Hz，需较宽松的 TF 容差避免 "Transform timeout"
- **DWB 参数调优**: `sim_time: 1.0`（轨迹预测时长），`PathAlign.scale: 12.0`，`GoalAlign.scale: 8.0`（降低权重减少短距离导航振荡）

## 完整工作流程

```text
 1. 用户说："去a点"
 2. voice_asr_node (sherpa-onnx) 本地语音识别，输出 "去a点" 到 /voice_command
 3. voice_control 订阅 /voice_command，用正则匹配关键词，从 destinations_json 解析 point_a 的坐标
 4. 发布 JSON 到 /voice_asr/keyword_matched
 5. nav2_goal_client 收到消息，发送 NavigateToPose Action 给 Nav2
 6. Nav2 进行全局/局部规划，通过 controller_server 发布 /cmd_vel
 7. cmd_vel_bridge_node 将 /cmd_vel 通过 IK 转换为全向轮角速度 (rad/s)
 8. 发布到 /base_velocity_controller/commands，经由 ros2_control → lekiwi_hardware 驱动电机
 9. cmd_vel_bridge_node 同时通过 FK 计算里程计，发布 /odom
10. EKF 融合 /odom 速度数据，发布 odom → base_link TF
11. RTAB-Map 通过视觉 SLAM 发布 map → odom TF 进行全局修正
12. 到达目标后，nav2_goal_client 检查是否有缓存的 task_description
13. 如果有，调用 /action_dispatcher/start_evaluate 触发机械臂推理
14. 如果用户说"停止"，voice_control 调用 /action_dispatcher/stop_evaluate 并取消导航
```

## 启动文件

| 文件 | 启动内容 |
|------|---------|
| `nav2_bringup.launch.py` | Nav2 子系统：map_server + Nav2 navigation_launch.py |
| `ekf_rtabmap_launch.py` | Legacy/debug-only RTAB-Map + EKF 入口；正式 LeKiwi 实机链路由 `robot_config` 启动 |
| `lekiwi_mapping_rviz.launch.py` | PC 端建图观察 RViz 预设 |
| `lekiwi_navigation_rviz.launch.py` | PC 端导航观察 RViz 预设 |

## 目录结构

```
robot_navigation/
├── config/
│   ├── keywords.json              # 关键词正则 → 动作映射
│   ├── ekf.yaml                   # EKF 传感器融合配置
│   ├── config.rviz                # RViz2 可视化配置
│   └── nav2_params.yaml           # Nav2 完整参数栈
├── launch/
│   ├── nav2_bringup.launch.py     # Nav2 子系统入口
│   ├── ekf_rtabmap_launch.py      # Legacy/debug-only EKF + RTAB-Map
│   ├── lekiwi_mapping_rviz.launch.py
│   └── lekiwi_navigation_rviz.launch.py
├── robot_navigation/
│   ├── voice_control.py           # 语音关键词匹配 + 导航桥接
│   ├── nav2_goal_client.py        # Nav2 Action 客户端 + 评估触发
│   └── cmd_vel_bridge_node.py     # cmd_vel 桥接 + IK/FK + 里程计
├── test/
│   ├── test_voice_control.py      # 语音控制 pytest 测试
│   ├── test_cmd_vel_bridge.py     # cmd_vel 桥接 pytest 测试（IK/FK/里程计）
│   ├── test_nav2_goal_client.py   # Nav2 Goal 客户端 pytest 测试
│   └── e2e/
│       ├── mock_servers.py        # Mock Trigger/Action 服务
│       └── test_pipeline_software.py   # 软件闭环 E2E（无物理仿真）
└── README.md
```

完整机器人启动链路下的导航仿真测试由 `robot_config` 维护。

## 依赖

### ROS2 包

- `nav2_bringup` — Nav2 导航栈
- `robot_localization` — EKF 传感器融合
- `robot_state_publisher` — URDF 坐标变换
- `joint_state_publisher` — 关节状态发布
- `rtabmap_launch` — RTAB-Map 视觉 SLAM
- `rviz2` — 可视化
- `robot_config` — 配置加载（destinations、contract）
- `action_dispatch` — 动作分发与评估（通过服务调用联动）

### Python 包

- `sherpa-onnx` — 本地语音识别（由 voice_asr_service 包提供）
- `ibrobot_msgs` — 自定义消息/服务（SetHotwords 等）

## 故障排除

### 导航时机器人不走直线（偏移/扭转）

导航偏移通常由以下原因叠加导致，按优先级排查：

1. **`use_sim_time` 配置错误**: 实车必须设为 `False`，否则 TF 查找失败
   ```bash
   ros2 param get /controller_server use_sim_time  # 应返回 "false"
   ```

2. **RTAB-Map pitch 偏差**: 相机安装倾斜会导致 yaw 偏差，需启用 `--Reg/Force3DoF true`
   ```bash
   ros2 run tf2_ros tf2_echo map odom  # 检查是否有非零 pitch/roll
   ```

3. **全局定位 TF 冲突**: 多个节点同时广播 `map → odom` 会导致 TF 跳变。正式 LeKiwi 实机链路不启动 AMCL，由 RTAB-Map 负责 `map → odom`
   ```bash
   ros2 run tf2_ros tf2_echo map odom
   ros2 topic echo /rtabmap/info
   ```

4. **EKF 订阅了不存在的话题**: EKF 无法更新会导致 TF 停滞
   ```bash
   ros2 topic hz /odom  # 确认有数据
   ```

5. **TF 发布冲突**: `cmd_vel_bridge` 和 EKF 同时发布 `odom → base_link` TF
   ```bash
   ros2 param get /cmd_vel_bridge publish_tf  # 启用 EKF 时应为 "false"
   ```

### 麦克风无法使用

```bash
sudo usermod -a -G audio $USER
# 重新登录后生效
```

### 语音识别无输出

```bash
# 检查 voice_asr_node 是否在运行
ros2 node list | grep voice

# 检查 /voice_command 是否有数据
ros2 topic echo /voice_command

# 检查 voice_control 是否收到文本
ros2 topic echo /rosout --filter "msg.name == 'voice_control'"
```

### 底盘无响应

```bash
# 检查 cmd_vel_bridge 是否正常发布
ros2 topic echo /base_velocity_controller/commands

# 检查里程计输出
ros2 topic echo /odom

# 检查 /cmd_vel 是否有输入
ros2 topic echo /cmd_vel
```

### 导航不触发评估

```bash
# 检查 nav2_goal_client 参数
ros2 param get /nav2_goal_client trigger_evaluation

# 检查 action_dispatcher 服务是否可用
ros2 service list | grep action_dispatcher
```

### EKF 融合异常

```bash
# 检查输入源是否正常
ros2 topic hz /odom

# 检查 EKF 输出
ros2 topic echo /odometry/filtered

# 检查 TF 树是否完整
ros2 run tf2_tools view_frames
```

### 语音命令未被 nav2_goal_client 接收

```bash
# /voice_asr/keyword_matched 话题需持续发布（非 --once），QoS 类型可能不匹配
# 测试时使用 -r 2 持续发布
ros2 topic pub -r 2 /voice_asr/keyword_matched std_msgs/msg/String "{data: '{\"type\": \"destination\", \"info\": {\"destination\": \"point_a\"}}'}"
```
