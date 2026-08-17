# LeKiwi 腕部 RealSense 手眼标定命令

本文件记录 LeKiwi（SO-101 从动臂腕装 RealSense）重新标定的完整命令流程。

前提条件：

- robot config：`src/robot_config/config/robots/lekiwi_handeye_realsense_grasp.yaml`
- robot config 中的从动臂端口：默认 `/dev/ttyACM0`
- 主臂端口：示例使用 `/dev/ttyACM1`，但必须以实际 USB 枚举为准
- 长期部署应改用 `/dev/serial/by-id/...`，避免重插后 ACM 编号互换
- 腕部 RealSense 话题（`lekiwi_handeye_realsense_grasp` 的实际 remap）：
  - RGB：`/camera/wrist/image_raw`
  - 对齐深度图：`/camera/wrist/aligned_depth_to_color/image_raw`
  - CameraInfo：`/camera/wrist/aligned_depth_to_color/camera_info`
- 机器人坐标系：
  - 基座坐标系：`base`
  - 末端执行器坐标系：`gripper`
- ChArUco 标定板：
  - 仓库提供的打印图：`outputs/calibration/charuco_A4_6x4_18mm_13mm_DICT_5X5_100_300dpi.png`
  - ArUco 字典：`DICT_5X5_100`
  - 棋盘格数：`6 x 4`
  - 格子边长：`0.018 m`
  - 标记边长：`0.013 m`
  - 最多 ChArUco 角点：`15`

`--squares-x/y` 是方格数量，不是内角点数量。标定命令里的
`--dictionary`、`--squares-x/y`、`--square-length`、`--marker-length`
必须和实物打印板一致；参数不一致会先在单帧 PnP 阶段产生几十像素级
重投影误差，样本会被持续拒绝。

不要在 Bash 中 source `install/setup.zsh`，请使用 `.shrc_local`。

重要：真机启动使用 `robot_config:=lekiwi_handeye_realsense_grasp`。启动前检查该 YAML；
它可能仍把 `ros2_control.port` 指到 `/dev/ttyACM0`。当 `/dev/ttyACM0`
是主臂时，主臂会被当作从动臂写入 follower 标定并上电保持。硬件激活不会自动执行
`reset_positions`，但端口选错仍可能损坏标定或导致后续控制指令作用于错误机械臂。

## 0. 可选：清理残留的机器人节点

仅在之前的启动被中断、ROS 图中存在残留节点时使用。

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && \
./scripts/cleanup_ros.sh
```

## 1. 检查眼在手（Eye-In-Hand）前置条件

```bash
cd ~/IB_Robot && \
  source .shrc_local && \
  export ROS_DOMAIN_ID=218 && \
  python3 scripts/check_handeye_preconditions.py \
    --robot-config src/robot_config/config/robots/lekiwi_handeye_realsense_grasp.yaml \
    --camera-name wrist
```

预期：前置检查通过，确认腕部/夹爪安装相机的配置正确。

## 2. 终端 A：启动遥操作设备（主臂）

将主臂连接到 USB 端口，用于遥操作控制从动臂移动到不同标定姿态。
如果当前已经按抓取流程启动过 `control_mode:=moveit_planning`，需要先执行第 0 步清理，
否则第 3 步会看到 `arm_trajectory_controller` / `gripper_trajectory_controller`
仍处于 active。手眼标定采样需要 teleop 的 position controller：
`arm_position_controller` / `gripper_position_controller`。
当前 `robot.launch.py` 不支持 `teleop_auto_config` 或 `teleop_leader_port` 启动参数；
`control_mode:=teleop` 只读取 robot YAML 中的 `robot.teleoperation`。启动终端 A 前，
先把 robot YAML 中的主臂配置显式打开。

把下面命令里的 `--leader-port` 改成实际主臂串口；它不能和
`robot.ros2_control.port` 指向的从动臂串口相同：

```bash
cd ~/IB_Robot && \
  source .shrc_local && \
  python3 scripts/configure_so101_handeye_teleop.py \
    --robot-config src/robot_config/config/robots/lekiwi_handeye_realsense_grasp.yaml \
    --leader-port /dev/ttyACM1
```

```bash
cd ~/IB_Robot && \
  source .shrc_local && \
  export ROS_DOMAIN_ID=218 && \
  ros2 launch robot_config robot.launch.py \
    robot_config:=lekiwi_handeye_realsense_grasp \
    control_mode:=teleop \
    use_sim:=false \
    moveit_display:=false
```

如果启动日志出现 `WARNING: Teleop mode requested but teleoperation config not found`
或 `Active device 'so101_leader' not found`，说明 robot YAML 中的
`robot.teleoperation` 尚未正确配置，需先回到本节开头重新启用主臂配置。
启动日志里必须看到加载 `lekiwi_handeye_realsense_grasp.yaml`。

等待以下输出：

```text
Controllers are active
```

确认遥操作已就绪：握住主臂，缓慢移动从动臂应跟随运动。
如果终端 A 持续打印 `Joint '<n>' clipped ...`，说明 leader 目标超出了 follower
安全限位。轻微且偶发的裁剪可以先把主臂摆回中位；如果持续裁剪，不要开始第 5 步采样，
先重新摆正主臂或检查 leader 校准/关节 wrap 状态。

## 3. 终端 B：验证运行状态

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 control list_controllers
```

预期的控制器状态：

```text
joint_state_broadcaster active
arm_position_controller active
gripper_position_controller active
```

检查关节状态：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic echo /joint_states --once
```

检查腕部相机：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic list | grep -E '/camera/.*/(image_raw|camera_info|aligned_depth_to_color)'
```

检查脚本使用的 RealSense 话题是否存在：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic info /camera/wrist/image_raw
```

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic info /camera/wrist/aligned_depth_to_color/camera_info
```

读取一帧真实图像，确认 topic 能收到数据：

```bash
cd ~/IB_Robot && \
  source .shrc_local && \
  export ROS_DOMAIN_ID=218 && \
  ros2 topic echo /camera/wrist/image_raw \
    --qos-reliability best_effort \
    --once
```

RealSense 图像 topic 使用 best-effort QoS。若省略 `--qos-reliability best_effort`，
`ros2 topic echo` 可能出现 reliability 不兼容，表现为 topic 有 publisher 但收不到图像。

## 4. 终端 C：用 RViz 实时查看腕部相机

启动 RealSense 后，通过 RViz 实时查看图像，确认标定板完整、清晰、曝光正常，并且移动主臂时画面确实随腕部相机变化。

优先使用当前 robot_config 重映射后的 RGB topic：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && rviz2
```

RViz 打开后按以下方式查看图像：

1. 点击左下角 `Add`。
2. 选择 `By display type` -> `Image`。
3. 在新建的 `Image` display 中把 `Topic` 设为 `/camera/wrist/image_raw`。
4. 确认画面稳定刷新，标定板完整、清晰、曝光正常。

如果 Topic 下拉里没有 `/camera/wrist/image_raw`，可以直接在 `Topic` 输入框手动输入该
topic 并按回车。若仍不刷新，把 Image display 的 QoS `Reliability` 改为
`Best Effort`。临时观察底层 RealSense raw 图像时，也可以输入
`/camera/wrist_camera/color/image_raw`；但正式标定命令仍使用
`/camera/wrist/image_raw`。

如果 RViz 的 Image 显示黑屏或不刷新，先确认当前 ROS 图中实际发布的图像 topic：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic list | grep -E 'color/image_raw|image_raw'
```

当前 `lekiwi_handeye_realsense_grasp` 应使用 `/camera/wrist/image_raw`。不要把
`/camera/camera/color/image_raw` 或 `/camera/wrist_camera/color/image_raw` 作为本流程默认输入。

重要：RViz Image 显示、`handeye_calibrator --image-topic` 必须使用同一个 RGB topic；
`--camera-info-topic` 使用 `/camera/wrist/aligned_depth_to_color/camera_info`，该 topic
包含与对齐深度图一致的有效相机内参。

对应关系：

| RGB topic | CameraInfo topic |
|---|---|
| `/camera/wrist/image_raw` | `/camera/wrist/aligned_depth_to_color/camera_info` |

确认事项：

- ChArUco 标定板应完整进入画面，不能只露出一部分。
- 标定板不要过曝、欠曝或运动模糊。
- 采样时标定板不要贴近图像边缘。
- 遥操作改变腕部姿态时，画面应稳定刷新。

## 5. 终端 D：运行交互式手眼标定

整个标定过程中保持 ChArUco 标定板固定不动。通过主臂遥操作将机械臂移动到新姿态，松手等待稳定后在本终端按回车采样。

如果第 4 步 RViz 能看到 `/camera/wrist/image_raw` 的真实画面，使用默认命令：

下面命令匹配仓库提供的标定板
`outputs/calibration/charuco_A4_6x4_18mm_13mm_DICT_5X5_100_300dpi.png`：
6 x 4 个 ChArUco 方格，方格边长 18 mm，marker 边长 13 mm。
如果使用的是另一个实物标定板，必须把 `--squares-x/y`、`--square-length`、
`--marker-length` 和 `--dictionary` 改成实物板的真实参数；参数不一致会导致几十像素级重投影误差。

```bash
cd ~/IB_Robot && \
  source .shrc_local && \
  export ROS_DOMAIN_ID=218 && \
  python3 scripts/handeye_calibrator.py \
    --image-topic /camera/wrist/image_raw \
    --camera-info-topic /camera/wrist/aligned_depth_to_color/camera_info \
    --base-frame base \
    --ee-frame gripper \
    --dictionary DICT_5X5_100 \
    --squares-x 6 \
    --squares-y 4 \
    --square-length 0.018 \
    --marker-length 0.013 \
    --samples 35 \
    --min-samples 20 \
    --min-corners 10 \
    --max-reprojection 1.0 \
    --method park \
    --output-json outputs/handeye/wrist_handeye_new.json \
    --robot-config src/robot_config/config/robots/lekiwi_handeye_realsense_grasp.yaml \
    --camera-name wrist \
    --max-translation-std 0.01 \
    --max-rotation-rms 2.0 \
    --max-reprojection-mean 1.0
```

参数说明：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--max-reprojection` | `1.0` | 单帧重投影误差上限（px），超过自动拒绝 |
| `--squares-x/y` | `6` / `4` | ChArUco 方格数量，不是内角点数量；6 x 4 方格最多检测到 15 个 ChArUco 角点 |
| `--square-length` / `--marker-length` | `0.018` / `0.013` | 方格边长和 marker 边长，单位 m，必须和实物打印板一致 |
| `--dictionary` | `DICT_5X5_100` | ArUco 字典，必须和打印板一致 |
| `--max-translation-std` | `0.01` | 标定结果平移标准差上限（m/轴） |
| `--max-rotation-rms` | `2.0` | 标定结果旋转 RMS 上限（度） |
| `--max-reprojection-mean` | `1.0` | 标定结果平均重投影误差上限（px） |
| `--robot-config` | （空） | 指定后，质量检查通过则自动写入变换 |
| `--camera-name` | （空） | 配置文件中相机外设名称，`--robot-config` 时必填 |

标定结束后脚本会自动执行质量检查：

```text
Quality check:
  PASS: auto-wrote transform to src/robot_config/config/robots/lekiwi_handeye_realsense_grasp.yaml
```

如果质量不达标：

```text
Quality check:
  FAIL: translation std y=0.02500m > 0.01m
  Quality check FAILED — config file was NOT updated.
  Adjust thresholds or re-calibrate with better samples.
```

采样指导：

- 采集 `25-35` 个有效样本。
- 通过主臂遥操作移动从动臂，变换腕部的横滚、俯仰和偏航，不要仅平移相机。
- 标定板应出现在图像的中心、左侧、右侧、上方和下方各区域。
- 超过 `--max-reprojection`（默认 `1.0 px`）的样本会被自动拒绝。
- 拒绝明显模糊或部分遮挡的标定板画面。
- 每次采样前松开主臂，等待从动臂完全静止后再按回车。

## 6. 确认配置文件已更新

标定成功后，传给 `--robot-config` 的 robot YAML 中 `peripherals[name=wrist].transform`
字段已自动更新。抓取时直接使用同一份 robot 配置读取外参，避免 robot YAML
和本地 hand-eye JSON 报告保存两套不同结果。JSON 报告是本地生成的调试/留档产物，
不会作为仓库默认标定结果提交。如果质量检查失败，脚本只写 JSON 报告，不会更新
`robot_config`。

终端 A 和正式 `pick_executor_node` 都应读取同一份
`src/robot_config/config/robots/lekiwi_handeye_realsense_grasp.yaml`。

也可打开生成的 JSON 报告查看完整指标：

```bash
cd ~/IB_Robot && python3 -m json.tool outputs/handeye/wrist_handeye_new.json
```

不需要复制 JSON，也不再向监督式客户端传手眼文件或 robot-config 路径。重启统一
pipeline 加载新 YAML 后，保持 `authorize_motion:=false`，先通过正式规划服务做无运动验证：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && \
ros2 service call /grasp_planner/plan_grasp ibrobot_msgs/srv/PlanGrasp \
  "{text_prompt: 'banana', confidence_threshold: 0.1, grasp_threshold: 0.5, debug_output_mode: 'diagnostic'}"
```

`PickObject.MODE_PLAN_ONLY` 是 Gateway 内部诊断字段，当前 v1 catalog 未对外公开，不能自行构造
delegated action goal 绕过 Gateway。

运行上述命令前，必须停止标定期间的终端 A–D，执行第 0 步清理，然后按
[`so101_banana_pick_pipeline_commands.md`](so101_banana_pick_pipeline_commands.md) 第 2 节重启统一 pipeline，
以加载新的相机变换。
