# SO101 腕部 RealSense 抓取流水线

面向真机抓取调试的最小命令集：腕部 RealSense -> Grounded-SAM2 -> GraspGen -> MoveIt -> 可选抓取验证。

## 固定约定

- 仓库：`~/IB_Robot`
- ROS 域：`ROS_DOMAIN_ID=218`
- robot config：`so101_handeye_realsense_only`
- runtime config：`/tmp/so101_handeye_realsense_grasp.yaml`
- 腕部 RGB：`/camera/wrist/image_raw`
- 腕部对齐深度：`/camera/wrist/aligned_depth_to_color/image_raw`
- 腕部 CameraInfo：`/camera/wrist/aligned_depth_to_color/camera_info`
- 检测服务：`/grounded_sam2/detect_and_segment`
- 抓取规划服务：`/grasp_planner/plan_grasp`
- 抓取验证服务：`/grasp_verifier/verify_grasp`
- 执行脚本：`scripts/test_banana_handeye_pick.py`

真机启动必须传 `config_path:=/tmp/so101_handeye_realsense_grasp.yaml`。不要只传
`robot_config:=so101_handeye_realsense_only`，否则可能加载仓库默认 YAML，把主臂端口当成从动臂控制。

常用命令都从仓库根目录执行：

```bash
cd ~/IB_Robot
```

## 0. 首次准备

安装依赖并下载模型：

```bash
./scripts/setup.sh --with-perception --with-grasp
./scripts/download_perception_models.sh
```

构建相关包：

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select \
  ibrobot_msgs perception_service manipulation_service robot_config dataset_tools
```

## 1. 清理残留节点

上次启动被中断、机器人不响应、MoveIt action server 重复时执行：

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && \
  pkill -f "ros2 launch robot_config robot.launch.py"; \
  pkill -f move_group; \
  pkill -f moveit_gateway.py; \
  pkill -f task_executor_node; \
  pkill -f ros2_control_node; \
  pkill -f realsense2_camera_node; \
  pkill -f "wrist_.*_relay"; \
  pkill -f grounded_sam2_node; \
  pkill -f grasp_planner_node; \
  pkill -f grasp_verifier_node; \
  pkill -f rviz2; \
  ros2 daemon stop
```

## 2. 启动服务

### 终端 A：机器人 + RealSense + MoveIt

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 launch robot_config robot.launch.py \
  robot_config:=so101_handeye_realsense_only \
  config_path:=/tmp/so101_handeye_realsense_grasp.yaml \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false
```

等待日志：

```text
Controllers are active
MoveIt Gateway fully initialized
TaskExecutor ready
```

### 终端 B：Grounded-SAM2

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 run perception_service grounded_sam2_node --ros-args \
  -p rgb_topic:=/camera/wrist/image_raw \
  -p depth_topic:=/camera/wrist/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/wrist/aligned_depth_to_color/camera_info
```

等待：

```text
GroundedSAM2Node ready
```

### 终端 C：GraspGen

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 run manipulation_service grasp_planner_node --ros-args \
  -p depth_topic:=/camera/wrist/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/wrist/aligned_depth_to_color/camera_info \
  -p detect_service:=/grounded_sam2/detect_and_segment \
  -p save_debug_outputs:=false \
  -p debug_output_dir:=outputs/grasp_pipeline \
  -p enable_collision_filter:=false \
  -p enable_tabletop_filter:=true \
  -p require_tabletop_filter:=false \
  -p tabletop_filter_mode:=diagnostic \
  -p tabletop_clearance:=0.002 \
  -p tabletop_pregrasp_distance:=0.08 \
  -p enable_object_cloud_completion:=true \
  -p object_cloud_completion_mode:=mask_depth_inpaint \
  -p object_cloud_completion_max_points:=5000 \
  -p object_cloud_completion_kernel_size:=5 \
  -p object_cloud_completion_min_neighbors:=6 \
  -p enable_object_cloud_prismatic_extrude:=true \
  -p object_cloud_prismatic_extrude_max_points:=8000 \
  -p object_cloud_prismatic_extrude_layers:=8 \
  -p sync_max_age_sec:=0.8 \
  -p input_buffer_size:=90 \
  -p num_grasps:=5000 \
  -p topk_num_grasps:=1000
```

等待：

```text
GraspPlannerNode ready
```

说明：`-p ...` 是 `grasp_planner_node` 参数，只能放在本命令后面，不要追加到 Python 抓取脚本后。

### 终端 C2：抓取验证，可选

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && ros2 run manipulation_service grasp_verifier_node --ros-args \
  -p gripper_joint:=6 \
  -p joint_state_topic:=/joint_states \
  -p joint_current_topic:=/so101_follower/joint_currents \
  -p wrist_depth_topic:=/camera/wrist/aligned_depth_to_color/image_raw \
  -p gripper_closed_position:=0.0 \
  -p gripper_contact_min_opening:=0.08 \
  -p gripper_no_contact_max_opening:=0.03 \
  -p current_contact_threshold_a:=0.08
```

腕部相机被大目标遮挡时不会单独判失败，只记录诊断证据。

### 终端 D：RViz，可选

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && rviz2
```

添加 `Image` display，topic 选 `/camera/wrist/image_raw`，Reliability 设为 `Best Effort`。

## 3. 启动后检查

运行时抓取几何配置：

```bash
grep -A16 "target_gripper:" /tmp/so101_handeye_realsense_grasp.yaml
```

应确认内容等价于：

```text
fixed_finger_contact_ee: [-0.014, 0.0, -0.080]
fixed_finger_margin_m: 0.003
fixed_finger_margin_max_m: 0.008
fixed_finger_margin_width_ref_m: 0.035
fixed_finger_margin_width_gain: 0.25
```

`fixed_finger_contact_ee.z` 是 SO101 夹爪坐标系里的固定指接触深度，不是 base-Z 下压量。完整抓取日志里应看到 `TARGET_WIDTH_COMP` 和 `width_comp=...fixed_finger_margin...`，说明动态固定指 margin 已从 runtime config 生效。

控制器：

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 control list_controllers
```

应看到：

```text
joint_state_broadcaster active
arm_trajectory_controller active
gripper_trajectory_controller active
```

服务和 action：

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 service list | grep -E 'detect_and_segment|plan_grasp|compute_ik'
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 action info /move_action
```

`/move_action` 必须只有一个 `/move_group` action server。若有多个，回到第 1 步清理。

相机：

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic info /camera/wrist/image_raw
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic info /camera/wrist/aligned_depth_to_color/image_raw
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic info /camera/wrist/aligned_depth_to_color/camera_info
```

## 4. 只移动到观测姿态

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && python3 scripts/test_banana_handeye_pick.py \
  --prompt banana \
  --observe-only \
  --handeye-source robot-config \
  --robot-config /tmp/so101_handeye_realsense_grasp.yaml \
  --observe-x 0.08 \
  --observe-y -0.23 \
  --observe-z 0.25
```

## 5. 冒烟测试

检测：

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 service call /grounded_sam2/detect_and_segment ibrobot_msgs/srv/DetectSegment \
  "{text_prompt: 'banana', confidence_threshold: 0.1}"
```

抓取规划：

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 service call /grasp_planner/plan_grasp ibrobot_msgs/srv/PlanGrasp \
  "{text_prompt: 'banana', confidence_threshold: 0.1, grasp_threshold: 0.5, debug_output_mode: 'diagnostic'}"
```

需要保存一帧检测快照时：

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 run perception_service grounded_sam2_snapshot \
  --prompt banana \
  --confidence-threshold 0.1 \
  --rgb-topic /camera/wrist/image_raw \
  --depth-topic /camera/wrist/aligned_depth_to_color/image_raw \
  --camera-info-topic /camera/wrist/aligned_depth_to_color/camera_info \
  --out-dir outputs/grounded_sam2
```

## 6. 仅规划，不抓取

会移动到观测姿态、检测、规划、做 IK/执行侧筛选，抓取前退出。

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && python3 scripts/test_banana_handeye_pick.py \
  --prompt banana \
  --detect-only \
  --handeye-source robot-config \
  --robot-config /tmp/so101_handeye_realsense_grasp.yaml \
  --debug-output-mode full \
  --execution-debug-preview \
  --task-goal-timeout-s 60.0 \
  --so101-tabletop-filter \
  --so101-tabletop-clearance -0.020 \
  --target-offset-z 0.000 \
  --min-contact-z -0.045 \
  --observe-x 0.10 \
  --observe-y -0.16 \
  --observe-z 0.22
```

通过标志：

```text
GRASPGEN_RESULT success=True ...
GRASPGEN_CANDIDATE_ACCEPT ...
PICK skipped=True reason=detect_only
FLOW_RESULT success=True
```

完整抓取前检查 `outputs/grasp_pipeline/...`：

- `execution_candidates.json`
- `grasp_preview_so101_execution.html`
- `grasp_preview_so101_execution.svg`
- `grasp_preview_execution_stages.svg`

如果出现 `height_guard_failed`、`so101_tabletop_failed`，或候选明显在桌面下，不要继续完整抓取。

## 7. 完整抓取

只在第 6 步正常后执行。

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && python3 scripts/test_banana_handeye_pick.py \
  --prompt banana \
  --handeye-source robot-config \
  --robot-config /tmp/so101_handeye_realsense_grasp.yaml \
  --debug-output-mode full \
  --execution-debug-preview \
  --pick-diagnostics \
  --task-goal-timeout-s 60.0 \
  --so101-tabletop-filter \
  --so101-tabletop-clearance -0.020 \
  --confidence-threshold 0.3 \
  --grasp-threshold 0.2 \
  --min-grasp-confidence 0.0 \
  --graspgen-centroid-confidence-window 0.06 \
  --graspgen-topdown-weight 0.35 \
  --graspgen-topdown-min-z -0.25 \
  --contact-realign-tolerance 0.008 \
  --target-offset-z 0.000 \
  --min-contact-z -0.045 \
  --observe-x 0.10 \
  --observe-y -0.16 \
  --observe-z 0.22
```

通过标志：

```text
TASK_RESULT success=True ...
GRASPGEN_CANDIDATE_ACCEPT ...
TASK_SEND id=banana_graspgen_pick_approach ...
TASK_SEND id=banana_graspgen_pick_pregrasp_realign ...
TASK_SEND id=banana_graspgen_pick_grasp ...
FLOW_RESULT success=True
```

执行后重点看同目录的：

- `pick_pose_diagnostics.json`：实际位姿误差和接触点 residual。
- `execution_candidates.json`：最终 selected 候选和被拒原因。
- `grasp_preview_so101_execution.html`：执行侧 3D 预览。

关键日志解释：

- `CONTACT_REALIGN phase=approach`：approach 高位接触点对齐。
- `PREGRASP_REALIGN`：按物体最高点加 `--pregrasp-realign-clearance` 计算最后安全对齐高度。
- `CONTACT_REALIGN phase=pregrasp`：pregrasp 高位接触点对齐。
- `PREGRASP_REALIGN_APPLY ... ignored_z_delta=...`：只把 pregrasp 的 XY 修正应用到最终下降，Z 修正被忽略，避免最终夹持被安全高度 realign 抬高。
- `CONTACT_REALIGN_CHECK phase=grasp`：低位只检查 residual，不再横向 realign。

如果固定指在目标前侧，仍可能出现固定指先碰边导致漏抓。此时不要只继续增大 `fixed_finger_margin_max_m`，还要在 `grasp_preview_so101_execution.html` 里检查 selected 候选的 SO101 mesh 方向，优先选择活动指能把目标扫回夹口的候选。

## 8. 抓取后验证

当前是手动调用，抓取脚本不会自动触发。建议完整抓取 lift 后保持 `0.5-1.0s` 再调用：

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && \
  ros2 service call /grasp_verifier/verify_grasp ibrobot_msgs/srv/VerifyGrasp \
  "{task_id: 'pick_001', text_prompt: 'banana', expected_target_width_m: 0.035, post_grasp_wait_s: 0.2}"
```

结果解释：

- `STATUS_SUCCESS(1)`：融合证据认为抓住。
- `STATUS_FAILED(0)`：融合证据倾向没抓住。
- `STATUS_UNCERTAIN(2)`：证据不足；重观察或保守重试，不要直接当失败。

`evidence` 中重点看 `gripper_position`、`gripper_current_abs_a`、`wrist_visibility`。

## 9. 常用调参

目标位置有固定偏差时：

```bash
--target-offset-x 0.00 --target-offset-y 0.00 --target-offset-z 0.00
```

当前现场常用桌面/高度保护：

```bash
--min-contact-z -0.045 --so101-tabletop-clearance -0.020
```

GraspGen 候选：

```bash
--grasp-threshold 0.5 --min-grasp-confidence 0.0 --max-candidates 80
```

Top-down 偏好：

```bash
--graspgen-rank-by-centroid \
--graspgen-topdown-weight 0.35 \
--graspgen-topdown-min-z -0.25
```

关闭 top-down 和质心重排，回到 GraspGen 置信度顺序：

```bash
--no-graspgen-rank-by-centroid --graspgen-topdown-weight 0
```

临时只验证位置目标：

```bash
--no-execute-grasp-orientation
```

临时禁用 IK 过滤，仅用于检测排障：

```bash
--detect-only --no-ik-filter
```

调试输出级别：

```bash
--debug-output-mode none        # 不写文件
--debug-output-mode diagnostic  # 只写 grasp_result.json
--debug-output-mode full        # 写点云、预览和执行侧诊断
```

## 10. 常见问题

`Detect service is not available`：启动终端 B，检查：

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 service list | grep grounded_sam2
```

`GraspGen service is not available`：启动终端 C，检查：

```bash
source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 service list | grep /grasp_planner/plan_grasp
```

`No synchronized depth/CameraInfo for detection frame`：

- 确认 B/C/RViz 都使用 `/camera/wrist/...` 同一组 topic。
- 确认三路相机 topic 都有发布者。
- 若有重复节点或 topic 重名，回到第 1 步清理。

`GraspGen returned zero candidates`：

- 用 `--debug-output-mode full` 看 mask、depth、`grasp_result.json`。
- 临时降低 `--grasp-threshold`。
- 必要时临时关闭服务端严格过滤做对比，但真机执行前必须保留执行侧 guard。

`Motion timed out after 60.0s`：

- 重启终端 A。
- 检查控制器和 `/joint_states`。

`STATUS_ABORTED` 但控制器像是完成了：

- 检查 `/move_action` 是否有多个 `/move_group` action server。
- 有重复就回到第 1 步清理。

抓取位置偏差大：

- 先看 `HANDEYE_QUALITY` 和 runtime YAML 中 wrist 相机 transform。
- 确认命令使用 `--handeye-source robot-config --robot-config /tmp/so101_handeye_realsense_grasp.yaml`。
- 若 `GRASPGEN_EE_ALIGNMENT` 没出现，说明脚本或环境不是当前版本。
