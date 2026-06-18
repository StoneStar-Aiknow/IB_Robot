# SO101 腕部 RealSense 抓取流水线命令

本文件记录使用腕部相机进行目标检测、GraspGen 6-DOF 抓取规划和 MoveIt 抓取执行的完整命令流程。

前提条件：

- 机器人配置：`so101_handeye_realsense_only`
- 运行时配置路径：`/tmp/so101_handeye_realsense_grasp.yaml`
- 运行时配置中的从动臂端口：`/dev/ttyACM1`
- 主臂端口：通常为 `/dev/ttyACM0`，但必须以实际 USB 枚举为准
- 腕部 RealSense 话题（`so101_handeye_realsense_only` 的实际 remap）：
  - RGB：`/camera/wrist/image_raw`
  - 对齐深度图：`/camera/wrist/aligned_depth_to_color/image_raw`
  - CameraInfo：`/camera/wrist/aligned_depth_to_color/camera_info`
- 实时网页预览脚本：`scripts/camera_topic_viewer.py`
- 检测包：`detection_service`
- 检测服务：`/grounded_sam2/detect_and_segment`
- GraspGen 抓取服务：`/grasp_planner/plan_grasp`
- 抓取脚本：`scripts/test_banana_handeye_pick.py`
- 抓取脚本仍支持读取本地生成的 hand-eye JSON 报告，但真机抓取推荐显式传
  `--handeye-source robot-config --robot-config /tmp/so101_handeye_realsense_grasp.yaml`。
  这样计算用的手眼外参与终端 A 实际加载的相机 TF 始终来自同一份 runtime YAML。

首次运行前需要安装可选检测/抓取依赖并下载检测模型：

`--with-grasp` 会编译 GraspGen 的 CUDA `pointnet2_ops` 扩展，运行前必须已安装
CUDA toolkit，并确保 `CUDA_HOME` 指向包含 `bin/nvcc` 的 CUDA 根目录。CPU-only
环境只能先运行 `--with-detection` 验证检测依赖。

```bash
cd ~/IB_Robot && ./scripts/setup.sh --with-detection --with-grasp
cd ~/IB_Robot && ./scripts/download_detection_models.sh
```

修改代码或首次运行前，先构建相关包：

```bash
cd ~/IB_Robot && source .shrc_local && colcon build --symlink-install --merge-install --packages-select \
  ibrobot_msgs detection_service grasp_service robot_config dataset_tools
```

不要在 Bash 中 source `install/setup.zsh`，请使用 `.shrc_local`。

重要：真机启动必须传 `config_path:=/tmp/so101_handeye_realsense_grasp.yaml`。
如果只传 `robot_config:=so101_handeye_realsense_only`，launch 会加载仓库内默认 YAML；
该默认 YAML 可能把 `ros2_control.port` 指到 `/dev/ttyACM0`。当 `/dev/ttyACM0`
是主臂时，主臂会被当作从动臂上电控制，并在启动时移动到 `reset_positions`。

关键坐标关系：

- GraspGen 返回的是相机坐标系下的 Robotiq 2F-140 抓取器位姿，不是目标质心，也不是 SO101 的 `gripper` link 位姿。
- SO101 的 MoveIt 目标 link 是 `gripper`，其原点在夹爪舵机/腕部壳体附近，不是指尖接触中心。
- 测试脚本会执行以下变换链：

```text
T_base_so101_ee =
  T_base_gripper_current *
  T_gripper_camera_from_robot_config *
  T_camera_graspgen *
  T_graspgen_so101_ee
```

运行脚本时必须看到类似日志，确认不是把两个坐标系直接重合：

```text
GRASPGEN_EE_ALIGNMENT graspgen_contact=(0.0000,0.0000,0.1950) target_contact=(0.0050,0.0000,-0.0750) adapter_xyz=(-0.0050,0.0000,0.1200) adapter_rpy=(3.1416,0.0000,0.0000) auto_width_compensation=True translation_auto=True
```

## 0. 可选：清理残留的机器人节点

在之前的启动被中断或机器人不再响应时使用。

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && \
  pkill -f "ros2 launch robot_config robot.launch.py"; \
  pkill -f move_group; \
  pkill -f moveit_gateway.py; \
  pkill -f task_executor_node; \
  pkill -f ros2_control_node; \
  pkill -f realsense2_camera_node; \
  pkill -f robot_state_publisher; \
  pkill -f static_transform_publisher; \
  ros2 daemon stop
```

清理后重新启动前，建议确认 MoveIt action server 没有重复残留。正常情况下
`/move_action` 应该没有 action server；终端 A 启动后只能有一个 `/move_group`
action server：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 action info /move_action
```

如果旧的 Grounded-SAM2 节点残留，也需停止：

```bash
pkill -f grounded_sam2_node
pkill -f grasp_planner_node
```

## 1. 终端 A：启动机器人、RealSense、控制器、MoveIt

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch robot_config robot.launch.py \
  robot_config:=so101_handeye_realsense_only \
  config_path:=/tmp/so101_handeye_realsense_grasp.yaml \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false
```

`teleop_auto_config` 默认开启：这里使用 `control_mode:=moveit_planning` 时会自动关闭
`robot.teleoperation`，避免主臂遥操继续向 position controller 发命令并与 MoveIt
轨迹控制抢控制权。但 `moveit_planning` 仍会启动 `ros2_control`，所以必须确认
`ros2_control.port` 指向从动臂，而不是主臂。

启动日志里必须看到：

```text
config_path: /tmp/so101_handeye_realsense_grasp.yaml
Loading config from: /tmp/so101_handeye_realsense_grasp.yaml
```

等待以下输出：

```text
Controllers are active
MoveIt Gateway fully initialized
TaskExecutor ready
```

## 2. 终端 B：启动 Grounded-SAM2 检测服务

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 run detection_service grounded_sam2_node --ros-args \
  -p rgb_topic:=/camera/wrist/image_raw \
  -p depth_topic:=/camera/wrist/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/wrist/aligned_depth_to_color/camera_info
```

等待以下输出：

```text
GroundedSAM2Node ready
```

## 3. 终端 C：启动 GraspGen 抓取规划服务

测试管线默认使用 GraspGen 的 6-DOF 抓取候选，不再直接使用 mask 质心作为抓取点。

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 run grasp_service grasp_planner_node --ros-args \
  -p depth_topic:=/camera/wrist/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/wrist/aligned_depth_to_color/camera_info \
  -p detect_service:=/grounded_sam2/detect_and_segment \
  -p save_debug_outputs:=false \
  -p debug_output_dir:=outputs/grasp_pipeline \
  -p enable_collision_filter:=true \
  -p enable_tabletop_filter:=true \
  -p require_tabletop_filter:=false \
  -p tabletop_filter_mode:=adaptive \
  -p tabletop_clearance:=0.002 \
  -p tabletop_pregrasp_distance:=0.08 \
  -p adaptive_tabletop_clearance_max:=0.002 \
  -p num_grasps:=5000 \
  -p topk_num_grasps:=1000
```

`save_debug_outputs:=false` 表示默认不为每次请求写完整点云和预览。调试输出改由
`PlanGrasp` 请求里的 `debug_output_mode` 按次控制：`diagnostic` 只写
`grasp_result.json`，`full` 还会写点云、gripper、PNG/HTML 预览。

上面的 GraspGen 参数是 2026-06-16 对 `banana` 连续规划对比后的推荐值：
`num_grasps:=5000`、`topk_num_grasps:=1000` 提高候选采样覆盖，
`tabletop_clearance:=0.002` 比默认 `0.003` 更容易保留贴桌但仍有正 clearance 的香蕉候选。
本轮复测中该组合 5 次规划成功 4 次，优于默认配置 4 次成功 2 次。

等待以下输出：

```text
GraspPlannerNode ready
```

## 4. 终端 D：启动 RealSense 实时网页预览（抓取时保持运行）

在终端 A/B/C 都正常后启动这个终端，并在后续 `--detect-only` 或完整抓取期间保持运行。
该脚本只订阅 RGB 图像并输出 MJPEG 网页，不会发送机器人控制命令。

优先查看与检测服务相同的 RealSense RGB topic：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && python3 scripts/camera_topic_viewer.py \
  --topic /camera/wrist/image_raw \
  --mode mjpeg \
  --host 0.0.0.0 \
  --port 8765
```

在机器人主机本机浏览器打开：

```text
http://127.0.0.1:8765
```

如果从另一台电脑浏览，打开：

```text
http://<机器人主机IP>:8765
```

如果 `8765` 端口被占用，把命令中的 `--port 8765` 改成 `--port 8766`。

如果网页打开但黑屏或不刷新，先确认当前 RealSense RGB topic：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic list | grep -E 'wrist|color/image_raw|image_raw'
```

正常抓取推荐继续使用 `/camera/wrist/image_raw`。当前 `so101_handeye_realsense_only`
会把 RealSense RGB raw 重映射到该 topic；不要使用 `/camera/camera/color/image_raw`
或 `/camera/wrist_camera/color/image_raw` 作为本流程的默认 RGB 输入。

注意：抓取调试时，网页预览、Grounded-SAM2 的 `rgb_topic`、抓取脚本的 `--rgb-topic`
应尽量使用同一个 RGB topic。这样浏览器里看到的画面就是检测和 GraspGen 实际使用的画面。

## 5. 终端 E：运行状态检查

检查控制器：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 control list_controllers
```

预期状态：

```text
joint_state_broadcaster active
arm_trajectory_controller active
gripper_trajectory_controller active
```

检查关节状态：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic echo /joint_states --once
```

检查动作服务：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 action list | grep -E 'execute_task_plan|follow_joint_trajectory|move_action'
```

检查 MoveIt action server 数量。必须只有一个 `/move_group` server；如果出现两个，
先回到第 0 步清理残留节点，否则会出现实际轨迹执行成功但客户端收到 `STATUS_ABORTED`：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 action info /move_action
```

预期：

```text
Action: /move_action
Action clients: 1
    /moveit_gateway
Action servers: 1
    /move_group
```

检查腕部相机 topic 有发布者：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic info /camera/wrist/image_raw
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic info /camera/wrist/aligned_depth_to_color/image_raw
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic info /camera/wrist/aligned_depth_to_color/camera_info
```

检查检测服务：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 service list | grep /grounded_sam2/detect_and_segment
```

检查 GraspGen 抓取服务：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 service list | grep /grasp_planner/plan_grasp
```

检查 IK 服务：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 service list | grep /compute_ik
```

## 6. 终端 E：仅移动到观测姿态

此步骤不执行检测、计算、IK 或抓取。

推荐使用的观测姿态（调参阶段使用）：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && python3 scripts/test_banana_handeye_pick.py \
  --prompt banana \
  --observe-only \
  --handeye-source robot-config \
  --robot-config /tmp/so101_handeye_realsense_grasp.yaml \
  --observe-x 0.08 \
  --observe-y -0.23 \
  --observe-z 0.25
```

备选的更高观测姿态：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && python3 scripts/test_banana_handeye_pick.py \
  --prompt banana \
  --observe-only \
  --handeye-source robot-config \
  --robot-config /tmp/so101_handeye_realsense_grasp.yaml \
  --observe-x 0.00 \
  --observe-y -0.22 \
  --observe-z 0.30
```

## 7. 终端 E：检测与 GraspGen 冒烟测试

直接调用服务：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 service call /grounded_sam2/detect_and_segment ibrobot_msgs/srv/DetectSegment \
  "{text_prompt: 'banana', confidence_threshold: 0.1}"
```

保存一帧可视化快照：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 run detection_service grounded_sam2_snapshot \
  --prompt banana \
  --confidence-threshold 0.1 \
  --rgb-topic /camera/wrist/image_raw \
  --depth-topic /camera/wrist/aligned_depth_to_color/image_raw \
  --camera-info-topic /camera/wrist/aligned_depth_to_color/camera_info \
  --out-dir outputs/grounded_sam2
```

直接调用 GraspGen 服务，确认能返回抓取候选：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 service call /grasp_planner/plan_grasp ibrobot_msgs/srv/PlanGrasp \
  "{text_prompt: 'banana', confidence_threshold: 0.1, grasp_threshold: 0.5, debug_output_mode: 'diagnostic'}"
```

返回里的 `debug_output_dir` 是本次请求写出的诊断目录，`diagnostic_details` 会列出
失败阶段、失败原因、mask/depth 统计、raw grasp 数量和过滤结果。

## 8. 终端 E：运行仅规划流水线

此步骤会移动到观测姿态、调用 GraspGen 生成抓取候选、将候选抓取位姿变换到基座坐标系、检查 IK，然后在抓取前退出。
执行时保持第 4 步网页预览终端运行，用浏览器确认香蕉在画面中且运动后没有离开视野。

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && python3 scripts/test_banana_handeye_pick.py \
  --prompt banana \
  --detect-only \
  --handeye-source robot-config \
  --robot-config /tmp/so101_handeye_realsense_grasp.yaml \
  --debug-output-mode diagnostic \
  --target-offset-z -0.008 \
  --min-contact-z -0.0101 \
  --observe-x 0.08 \
  --observe-y -0.23 \
  --observe-z 0.25
```

预期最后一行输出：

```text
FLOW_RESULT success=True
```

同时确认输出中包含：

```text
GRASPGEN_EE_ALIGNMENT ...
GRASPGEN_CANDIDATE_ACCEPT ...
```

完整抓取前必须检查候选高度日志。`contact_base` 是 GraspGen 接触点变换到
`base` 后的位置，`target_ee_grasp` 是要发给 MoveIt 的目标 `gripper` link 位置。
如果看到 `reason=height_guard_failed`，或者 `contact_base.z`/`target_ee_grasp.z`
低于桌面，不要运行完整抓取，先同步手眼外参。

本次验证过的 `--detect-only` 关键日志形态如下：

```text
GRASPGEN_RESULT success=True n=11 ...
GRASPGEN_DEBUG_OUTPUT dir=...
GRASPGEN_DIAGNOSTIC ...
GRASPGEN_CANDIDATE idx=0 ... contact_base=(...) target_ee_grasp=(...)
GRASPGEN_CANDIDATE_ACCEPT idx=0 ...
PICK skipped=True reason=detect_only
FLOW_RESULT success=True
```

## 9. 终端 E：运行完整抓取流水线

仅在 `--detect-only` 结果正常后执行。
执行期间保持第 4 步网页预览终端运行，用浏览器实时观察 RealSense RGB 画面。

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && python3 scripts/test_banana_handeye_pick.py \
  --prompt banana \
  --handeye-source robot-config \
  --robot-config /tmp/so101_handeye_realsense_grasp.yaml \
  --debug-output-mode full \
  --confidence-threshold 0.3 \
  --grasp-threshold 0.2 \
  --min-grasp-confidence 0.60 \
  --graspgen-centroid-confidence-window 0.06 \
  --graspgen-topdown-weight 0.35 \
  --target-offset-z -0.008 \
  --min-contact-z -0.0101 \
  --observe-x 0.10 \
  --observe-y -0.16 \
  --observe-z 0.25
```

注意：`tabletop_filter_mode`、`adaptive_tabletop_*` 是 `grasp_planner_node` 的 ROS 参数，
不能追加到 `scripts/test_banana_handeye_pick.py` 命令后。若脚本报
`unrecognized arguments: -p ...`，说明这些 `-p` 参数放错位置了。请在第 3 步启动
`grasp_planner_node` 时设置，或在节点已运行时使用 `ros2 param set /grasp_planner ...`。

本次完整流程验证通过时，关键日志形态如下：

```text
TASK_RESULT success=True steps=2 ... msg=All steps completed successfully
GRASPGEN_RESULT success=True n=11 ...
GRASPGEN_CANDIDATE_ACCEPT idx=0 ...
TASK_SEND id=banana_graspgen_pick steps=7 ...
TASK_RESULT success=True steps=7 ... msg=All steps completed successfully
FLOW_RESULT success=True
```

## 10. 常用调参参数

如果变换后的目标位置存在系统性偏移，可手动添加基座坐标系修正：

```bash
--target-offset-x 0.00 --target-offset-y 0.00 --target-offset-z 0.00
```

当前现场点云拟合到的桌面高度约为 `base.z=-0.012 m`。抓取脚本默认
`--min-contact-z 0.0` 是相对机器人 `base` 坐标系的保守绝对下限，不是桌面检测高度；
本现场建议使用 `2 mm` 桌面余量：

```bash
--min-contact-z -0.0101
```

若仍感觉夹爪偏上，可使用本次现场验证的基座 Z 下压，不要直接关闭高度保护：

```bash
--target-offset-z -0.008 --min-contact-z -0.0101
```

如果临时使用本地 hand-eye JSON 报告，建议同时传运行时配置路径做一致性检查：

```bash
--robot-config /tmp/so101_handeye_realsense_grasp.yaml
```

一旦看到 `HANDEYE_CONFIG_CHECK ... translation_delta_m` 或 `rotation_delta_deg`
超出阈值，先同步手眼外参；不要用 `--target-offset-*` 掩盖两套外参不一致。

示例：如果夹爪在基座 Y 轴负方向偏了 `5 cm`，添加：

```bash
--target-offset-y 0.05
```

GraspGen 是默认目标来源。只有临时对比旧质心逻辑时才加：

```bash
--target-source centroid
```

GraspGen 置信度与候选数量：

```bash
--grasp-threshold 0.5 --min-grasp-confidence 0.0 --max-candidates 80
```

草莓等贴桌、低矮目标建议让 `grasp_planner_node` 使用自适应桌面过滤，而不是直接关闭
`enable_tabletop_filter`：

启动 `grasp_planner_node` 时设置：

```bash
-p tabletop_filter_mode:=adaptive \
-p adaptive_tabletop_low_profile_height:=0.035 \
-p adaptive_tabletop_clearance_min:=0.001 \
-p adaptive_tabletop_clearance_max:=0.002 \
-p adaptive_tabletop_pregrasp_min:=0.02 \
-p adaptive_tabletop_auto_tune:=true \
-p adaptive_tabletop_retry_clearances:=0.002,0.001
```

如果 `grasp_planner_node` 已经在运行，用参数服务动态切换：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && \
  ros2 param set /grasp_planner tabletop_filter_mode adaptive
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && \
  ros2 param set /grasp_planner adaptive_tabletop_clearance_max 0.002
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && \
  ros2 param set /grasp_planner adaptive_tabletop_clearance_min 0.001
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && \
  ros2 param set /grasp_planner adaptive_tabletop_pregrasp_min 0.02
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && \
  ros2 param set /grasp_planner adaptive_tabletop_auto_tune true
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.bash && \
  ros2 param set /grasp_planner adaptive_tabletop_retry_clearances '0.002,0.001'
```

然后运行 `scripts/test_banana_handeye_pick.py`；不要在该 Python 脚本后追加 `-p ...`。

`adaptive_tabletop_auto_tune` 默认开启。若低矮目标首轮 adaptive 桌面过滤后没有候选，
节点会在 `tabletop_best_candidate_clearance_m` 未明显低于桌面的前提下，自动尝试
`adaptive_tabletop_retry_clearances` 中更小的 clearance。本轮草莓检测中，
`adaptive_tabletop_clearance_max:=0.002` 是当前推荐默认值，比旧的 `0.003` 更容易保留贴桌的向下抓候选。
推荐优先从以下观测位姿开始：

```bash
--observe-x 0.10 --observe-y -0.17 --observe-z 0.27
```

判断候选是否“往下抓”时，看日志中的 `approach_axis_base=(x,y,z)` 和
`topdown_score`。若 `z` 接近 `-1.0`，例如 `-0.95`，说明 GraspGen 候选是沿
基座 Z 方向向下接近目标；`topdown_score` 越接近 `1.0`，排序越偏向 top-down。

如果 `adaptive` 模式仍出现 `failure_stage: tabletop_filter` 且
`tabletop_object_height_m` 很低，先查看 `tabletop_auto_tuned`、
`tabletop_auto_tune_attempts` 和 `tabletop_auto_tune_reason`。如果 auto-tune 仍无法救回，
可临时用 `soft` 模式做诊断；真机执行前必须查看 `grasp_result.json` 和预览图，
确认 `tabletop_relaxed` 候选没有明显穿桌：

```bash
-p tabletop_filter_mode:=soft -p adaptive_tabletop_hard_floor:=-0.002
```

重点诊断字段：`tabletop_filter_mode`、`tabletop_low_profile`、`tabletop_relaxed`、
`tabletop_object_height_m`、`tabletop_clearance_used_m`、
`tabletop_pregrasp_distance_used_m`、`tabletop_best_candidate_clearance_m`、
`tabletop_auto_tuned`、`tabletop_auto_tune_attempts`、`tabletop_auto_tune_reason`。

按请求控制 GraspGen 调试输出：

```bash
--debug-output-mode default     # 跟随 grasp_planner_node 的 save_debug_outputs 参数
--debug-output-mode none        # 不写任何调试文件
--debug-output-mode diagnostic  # 只写 grasp_result.json，适合常规排障
--debug-output-mode full        # 写 grasp_result.json、点云、gripper 和预览图
```

默认会在相近置信度的 GraspGen 候选里，同时考虑 top-down 姿态偏好和
“接触点更接近检测到的 3D 目标中心”的候选。top-down 分数使用 base 坐标系下的
`approach_axis_base.z` 计算，越接近从上往下抓，`topdown_score` 越接近 `1.0`：

```bash
--graspgen-rank-by-centroid \
--graspgen-centroid-confidence-window 0.06 \
--graspgen-topdown-weight 0.35 \
--graspgen-topdown-min-z -0.25
```

日志里重点看：

```text
GRASPGEN_RANK ... order=idx:centroid_dist/conf/td=topdown/s=combined,...
GRASPGEN_CANDIDATE ... approach_axis_base=(...) topdown_score=... contact_camera=(...) centroid_dist_camera=...
```

如果高置信候选是侧向/平行抓，但另一个候选的 `topdown_score` 明显更高，combined score 会优先尝试更接近 top-down 的候选。
如需减弱 top-down 偏好，降低 `--graspgen-topdown-weight`；如需关闭 top-down 打分，设置：

```bash
--graspgen-topdown-weight 0
```

如需关闭检测质心重排但保留 top-down 打分，可加 `--no-graspgen-rank-by-centroid` 并保留
`--graspgen-topdown-weight` 大于 `0`。如需回到纯 GraspGen 置信度顺序，同时设置：

```bash
--no-graspgen-rank-by-centroid --graspgen-topdown-weight 0
```

SO101 是 5-DOF，但单动夹爪补偿是在 GraspGen 候选姿态下计算的。脚本默认会把
GraspGen 候选姿态发送给 MoveIt，让补偿计算和实际执行使用同一套夹爪方向：

```bash
--execute-grasp-orientation
```

如需临时只验证位置目标，可关闭姿态执行；此时日志会提示宽度补偿假设的 GraspGen 姿态
可能和实际执行姿态不一致：

```bash
--no-execute-grasp-orientation
```

如果要让 IK 过滤也检查候选姿态，可临时加：

```bash
--ik-check-orientation
```

抓取高度保护默认开启，防止把桌面下方目标发送给 MoveIt：

```bash
--min-approach-z 0.04 --min-grasp-z 0.02 --min-contact-z 0.0
```

如果日志出现：

```text
GRASPGEN_CANDIDATE_REJECT ... reason=height_guard_failed ...
```

说明当前手眼外参或 GraspGen/SO101 适配把目标变换到了桌面下方。先修手眼外参，
不要用 `--allow-out-of-workspace` 绕过该保护。

GraspGen 抓取器坐标系到 SO101 `gripper` 坐标系的默认适配：

```bash
--graspgen-contact-x 0.0 --graspgen-contact-y 0.0 --graspgen-contact-z 0.195 \
--target-contact-x 0.005 --target-contact-y 0.0 --target-contact-z -0.075 \
--graspgen-to-ee-roll 3.141592653589793 --graspgen-to-ee-pitch 0.0 --graspgen-to-ee-yaw 0.0
```

脚本会由以上 contact-center 自动计算 `--graspgen-to-ee-x/y/z`，默认结果约为：

```bash
--graspgen-to-ee-x -0.005 --graspgen-to-ee-y 0.0 --graspgen-to-ee-z 0.120
```

含义：把 GraspGen 的 Robotiq 接触中心对齐到 SO101 夹爪有效接触中心，而不是把
GraspGen 抓取器原点直接对齐到 SO101 的 `gripper` link 原点。
SO101 指尖最末端约在 `gripper.z=-0.105`，但香蕉抓取更适合用指腹夹持中心，
默认使用 `gripper.z=-0.075`。如果实测仍然在香蕉外侧收手，可继续把
`--so101-contact-z` 调到 `-0.065`；如果插入过深或碰撞，再调回 `-0.085`。

如果实测 SO101 指尖有效接触点不同，优先调 `--target-contact-*`。
旧的 `--so101-contact-*` 仍是兼容别名，但新配置和脚本逻辑都使用 target-gripper 命名。
只有确认整体存在固定基座坐标偏差时，再调 `--target-offset-*`。

当前 `grasp_service` 会为每个 GraspGen 候选自动估算机器人无关的目标宽度：
`target_width_m` 是分割点云沿 GraspGen 源夹爪闭合轴投影后的宽度，不需要用户手动估算目标宽度。
目标夹爪补偿不写在 `grasp_service` 中，而是由执行脚本读取 runtime
`robot_config` 里的 `grasp_execution.target_gripper` 几何参数后，在 source-gripper 到
target-gripper adapter 中按候选动态补偿。SO101 只是其中一个 target-gripper 配置实例。

推荐在 runtime YAML 中保留以下配置：

```yaml
grasp_execution:
  source_gripper: robotiq_2f_140
  source_contact_point: [0.0, 0.0, 0.195]
  adapter:
    source_to_ee_rpy: [3.141592653589793, 0.0, 0.0]
  execution_scoring:
    confidence_weight: 1.0
    contact_distance_weight: 1.0
    contact_distance_scale_m: 0.06
    topdown_weight: 0.70
  target_gripper:
    type: asymmetric_single_moving_jaw
    ee_frame: gripper
    # Fixed finger reference; fallback width keeps old center near [0.005,0,-0.075].
    fixed_finger_contact_ee: [-0.014, 0.0, -0.075]
    # Vector from fixed finger contact toward target width center.
    closing_axis_ee: [1.0, 0.0, 0.0]
    width_clearance_m: 0.003
    min_width_m: 0.008
    max_width_m: 0.080
    fallback_width_m: 0.035
    # Optional override; script default is 0.75 to reject clipped width estimates.
    width_quality_min: 0.75
```

日志里重点看：

```text
GRASPGEN_CANDIDATE ... target_width=... width_quality=... width_comp=auto:measured_width=...:used_width=...:quality=... target_contact=(...) adapter_xyz=(...)
```

脚本默认 `--target-width-quality-min 0.75`。旧的 `--so101-width-quality-min` 是兼容别名。
`target_width_quality=0.5` 通常表示宽度估计被
GraspGen 的 min/max 截断过，不再作为可靠自动宽度；脚本会改用 `fallback_width_m`，避免把
SO101 单动爪补偿拉偏。如果 `width_comp=fallback`，说明候选宽度质量不足或宽度无效；如果要在宽度无效时拒绝候选，添加：

```bash
--target-auto-width-required
```

`closing_axis_ee` 必须和 GraspGen 源夹爪的宽度轴通过 `source_to_ee_rpy` 对齐。当前
Robotiq 2F-140 源夹爪宽度轴是局部 `X`，`source_to_ee_rpy=(pi,0,0)` 下 SO101
`+X` 映射到 GraspGen `+X`，因此这里使用 `[1.0, 0.0, 0.0]`。这里的轴表示从固定指
接触点指向目标宽度中心的方向，不是活动爪的转轴。不要把单动爪补偿写成
`[0.0, 1.0, 0.0]`，否则宽度补偿会偏到与 GraspGen 夹爪宽度轴正交的方向。

`fixed_finger_contact_ee` 不是旧的 SO101 有效接触中心。旧中心约为
`[0.005, 0.0, -0.075]`，动态补偿会计算：

```text
effective_center = fixed_finger_contact_ee + closing_axis_ee * 0.5 * (target_width_m + width_clearance_m)
```

因此当前用 `[-0.014, 0.0, -0.075]` 作为固定指侧参考点；当宽度退回
`fallback_width_m=0.035` 且 `width_clearance_m=0.003` 时，effective center 仍约为
`[0.005, 0.0, -0.075]`，不会破坏旧的静态接触中心假设。

如果要临时关闭 SO101 自动宽度补偿并回到固定 contact-center 对齐：

```bash
--no-target-auto-width-compensation
```

旧质心模式下 SO101 夹爪的 TCP 参数：

```bash
--tcp-x 0.005 --tcp-y 0.0 --tcp-z -0.105 --tip-clearance-z 0.008
```

仅检测调试时禁用 IK 过滤：

```bash
--detect-only --no-ik-filter
```

## 11. 常见故障

`Detect service is not available`（检测服务不可用）：

- 启动终端 B。
- 检查：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 service list | grep grounded_sam2
```

`GraspGen service is not available`（GraspGen 服务不可用）：

- 启动终端 C。
- 检查：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 service list | grep /grasp_planner/plan_grasp
```

`GraspGen returned zero candidates`（没有抓取候选）：

- 先看 `outputs/grasp_pipeline/` 下的调试输出，确认 mask、深度和点云是否正确。
- 可以临时在终端 C 关闭严格几何过滤：

```bash
-p enable_collision_filter:=false -p enable_tabletop_filter:=false -p require_tabletop_filter:=false
```

`Motion timed out after 60.0s`（运动超时）：

- 任务已发送，但控制器或 `/joint_states` 未激活。
- 重启终端 A。
- 检查：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 control list_controllers
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic echo /joint_states --once
```

`MoveIt reported unsuccessful` / `STATUS_ABORTED`，但日志里控制器显示轨迹已完成：

- 优先检查是否有重复 `/move_group` action server。
- 典型 gateway 日志：

```text
Ignoring unexpected goal response. There may be more than one action server for the action 'move_action'
Action 'move_action' was unsuccessful: STATUS_ABORTED.
```

- 检查：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 action info /move_action
```

- 如果 `Action servers` 下出现两个 `/move_group`，执行第 0 步清理残留节点，然后只启动一个终端 A。

`IK service is not available`（IK 服务不可用）：

- MoveIt 尚未完全启动。
- 等待出现 `You can start planning now!`，或重启终端 A。

抓取位置偏差较大：

- 手眼标定质量差，请重新执行
  `docs/so101_handeye_calibration_commands.md`（可调低 `--max-reprojection` 以排除更多低质量样本）。
- 脚本会打印 `HANDEYE_QUALITY`，如果显示 `status=bad`，请勿信任抓取结果。
- 如果 `--detect-only` 中 `contact_base.z` 或 `target_ee_grasp.z` 已经低于桌面，
  说明规划输入目标本身已经在桌面下方，不是 MoveIt 执行偏差。
- 如果 `/tmp/so101_handeye_realsense_grasp.yaml` 与本地生成的 hand-eye JSON 报告
  不一致，先同步外参并重启终端 A。例如当前推荐的 `ee_to_camera_link` 数值为：

```yaml
transform:
  parent_frame: gripper
  x: -0.03893349096435185
  y: 0.046093712828857866
  z: -0.02852409709828743
  roll: 3.1220710111747603
  pitch: 1.08946543832224
  yaw: -1.4382020056359806
```

- 当前默认路径使用 GraspGen 候选，不使用质心；如果偏差仍大，优先检查 GraspGen debug 输出里的
  `grasp_result.json`、相机 frame_id、手眼外参，以及 `--graspgen-to-ee-*` 坐标系适配量。
- 如果 `GRASPGEN_EE_ALIGNMENT` 没有出现，或 `adapter_xyz=(0.0000,0.0000,0.0000)` 且 `adapter_rpy=(0.0000,0.0000,0.0000)`，说明运行的不是修正后的脚本。
