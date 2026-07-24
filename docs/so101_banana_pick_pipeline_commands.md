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
- Hermes MCP 服务名：`ibrobot`
- Hermes 抓取技能：`pick_object`，不要写成 `pick-object`

真机启动必须传 `config_path:=/tmp/so101_handeye_realsense_grasp.yaml`。不要只传
`robot_config:=so101_handeye_realsense_only`，否则可能加载仓库默认 YAML，把主臂端口当成从动臂控制。
每次拉取抓取/Hermes 能力更新后，先运行 runtime 合成器；它保留本机串口、相机和手眼标定，
并从仓库 SSOT 更新 `grasp_execution`、`embodied` 及安全策略。

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

构建完整抓取与 Hermes 调用链：

```bash
cd ~/IB_Robot && source .shrc_local && colcon build --symlink-install --merge-install --packages-up-to \
  embodied_bringup robot_mcp manipulation_execution
```

把仓库能力配置合入已标定的 runtime YAML：

```bash
cd ~/IB_Robot && source .shrc_local && python3 scripts/synthesize_so101_grasp_runtime_config.py
```

## 1. 清理残留节点

上次启动被中断、机器人不响应、MoveIt action server 重复时执行：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && \
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
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 launch robot_config robot.launch.py \
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

### 终端 A2：并行 IK/FK Worker，监督式脚本需要

当前 SO101 robot_config 默认使用 4 个隔离 worker；运行监督式脚本前启动：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 launch robot_moveit so101_ik_workers.launch.py \
  worker_count:=4 \
  namespace_prefix:=ik_worker \
  use_sim_time:=false
```

这些 worker 只提供 `/ik_worker_<n>/compute_ik` 和 `/ik_worker_<n>/compute_fk`，不会发送运动命令。如需临时退回串行候选准备，显式传 `--ik-worker-count 0`。

该终端只用于单独运行监督式脚本。通过第 11 节的 `embodied_pipeline.launch.py` 启动 Hermes 抓取时，
launch 会读取 `grasp_execution.ik.worker_count` 并自动启动同样的 worker，不要重复启动本终端。

### 终端 B：Grounded-SAM2

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 run perception_service grounded_sam2_node --ros-args \
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
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 run manipulation_service grasp_planner_node --ros-args \
  -p depth_topic:=/camera/wrist/aligned_depth_to_color/image_raw \
  -p camera_info_topic:=/camera/wrist/aligned_depth_to_color/camera_info \
  -p detect_service:=/grounded_sam2/detect_and_segment \
  -p save_debug_outputs:=false \
  -p debug_output_dir:=outputs/grasp_pipeline \
  -p enable_collision_filter:=false \
  -p enable_tabletop_filter:=true \
  -p enable_source_gripper_tabletop_sweep:=false \
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

### 终端 C2：抓取验证，完整抓取必需

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && ros2 run manipulation_service grasp_verifier_node --ros-args \
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
完整抓取默认使用 `--grasp-verification required`，若本服务未启动，脚本会在任何抓取动作前退出。

### 终端 D：RViz，可选

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && rviz2
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
fixed_finger_margin_m: 0.006
fixed_finger_margin_max_m: 0.012
fixed_finger_margin_width_ref_m: 0.035
fixed_finger_margin_width_gain: 0.25
```

`fixed_finger_contact_ee.z` 是 SO101 夹爪坐标系里的固定指接触深度，不是 base-Z 下压量。完整抓取日志里应看到 `TARGET_WIDTH_COMP` 和 `width_comp=...fixed_finger_margin...`，说明动态固定指 margin 已从 runtime config 生效。

控制器：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 control list_controllers
```

应看到：

```text
joint_state_broadcaster active
arm_trajectory_controller active
gripper_trajectory_controller active
```

服务和 action：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 service list | grep -E 'detect_and_segment|plan_grasp|compute_ik|compute_fk|move_to_configuration|verify_grasp'
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 action info /move_action
```

`/move_action` 必须只有一个 `/move_group` action server。若有多个，回到第 1 步清理。

相机：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic info /camera/wrist/image_raw
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic info /camera/wrist/aligned_depth_to_color/image_raw
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 topic info /camera/wrist/aligned_depth_to_color/camera_info
```

## 4. 只移动到观测姿态

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && python3 scripts/test_banana_handeye_pick.py \
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
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 service call /grounded_sam2/detect_and_segment ibrobot_msgs/srv/DetectSegment \
  "{text_prompt: 'banana', confidence_threshold: 0.1}"
```

抓取规划：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 service call /grasp_planner/plan_grasp ibrobot_msgs/srv/PlanGrasp \
  "{text_prompt: 'banana', confidence_threshold: 0.1, grasp_threshold: 0.5, debug_output_mode: 'diagnostic'}"
```

需要保存一帧检测快照时：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 run perception_service grounded_sam2_snapshot \
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
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && python3 scripts/test_banana_handeye_pick.py \
  --prompt banana \
  --detect-only \
  --handeye-source robot-config \
  --robot-config /tmp/so101_handeye_realsense_grasp.yaml \
  --debug-output-mode diagnostic \
  --no-execution-debug-preview \
  --ik-worker-count 4 \
  --centroid-source volume \
  --task-goal-timeout-s 60.0 \
  --so101-tabletop-filter \
  --so101-tabletop-clearance 0.000 \
  --ik-fk-contact-max-xz-error 0.020 \
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

- `grasp_result.json`
- `execution_candidates.json`
- `prepared_candidate_ranking.json`

detect-only 选中的记录使用 `reason: selected_detect_only`，不能把它误认为已经执行过真实抓取。

如果出现 `height_guard_failed`、`so101_tabletop_failed`，或候选明显在桌面下，不要继续完整抓取。

`--so101-tabletop-filter` 优先使用 PlanGrasp 响应中的 execution table plane，不需要为了正常抓取
生成完整 PLY 和 HTML/SVG/PNG 预览。如果响应中没有可用平面，候选会 fail closed 并以
`so101_tabletop_failed` 拒绝，不会静默放行。只有排查点云或夹爪几何时才临时启用
`--debug-output-mode full --execution-debug-preview`。

同时检查终端日志中的：

- `DETECTION ... surface_xyz=... volume_xyz=...`：当前默认使用 `volume_xyz` 做候选重排和主目标点。
- `GRASPGEN_RANK ... centroid_camera=...`：应对应 `--centroid-source volume` 的体积质心。
- `grasp_preview_so101_execution.html`：同时显示主质心和备选质心；如果 surface/volume 分离明显，优先看体积质心是否落在 SO101 两指通道内。

## 7. 完整抓取

只在第 6 步正常后执行。

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && python3 scripts/test_banana_handeye_pick.py \
  --handeye-source robot-config \
  --robot-config /tmp/so101_handeye_realsense_grasp.yaml \
  --debug-output-mode diagnostic \
  --no-execution-debug-preview \
  --ik-worker-count 4 \
  --pick-diagnostics \
  --no-pick-diagnostics-detect \
  --grasp-verification required \
  --recover-after-close-failure \
  --grasp-verification-probe-lift-height 0.030 \
  --grasp-verification-probe-lift-speed 0.020 \
  --task-goal-timeout-s 60.0 \
  --so101-tabletop-filter \
  --so101-tabletop-clearance -0.020 \
  --confidence-threshold 0.3 \
  --grasp-threshold 0.2 \
  --min-grasp-confidence 0.0 \
  --centroid-source volume \
  --graspgen-centroid-confidence-window 0.06 \
  --graspgen-topdown-weight 0.35 \
  --graspgen-topdown-min-z -0.25 \
  --ik-fk-contact-compensation \
  --ik-fk-contact-tolerance 0.003 \
  --ik-fk-contact-max-iterations 6 \
  --ik-fk-contact-max-correction 0.030 \
  --ik-fk-contact-max-xz-error 0.020 \
  --max-execution-attempts 1 \
  --final-lift 0.050 \
  --lift-speed 0.020 \
  --contact-realign-tolerance 0.008 \
  --target-offset-z 0.000 \
  --min-contact-z -0.045 \
  --observe-x 0.10 \
  --observe-y -0.16 \
  --observe-z 0.22  \
  --prompt banana 
```

通过标志：

```text
TASK_RESULT success=True ...
GRASPGEN_CANDIDATE_ACCEPT ...
TASK_SEND id=banana_graspgen_pick_approach ...
TASK_SEND id=banana_graspgen_pick_pregrasp_realign ...
MOVE_CONFIGURATION_RESULT label=descend_to_ik_fk_compensated_grasp success=True ...
GRASP_VERIFY label=close success=True status=success ...
TASK_SEND id=banana_graspgen_pick_probe_lift ...
GRASP_VERIFY label=probe_lift success=True status=success ...
GRASP_VERIFY label=lift success=True status=success ...
FLOW_RESULT success=True
```

执行后重点看同目录的：

- `pick_pose_diagnostics.json`：实际位姿误差和接触点 residual。
- `grasp_verification.json`：close、3 cm probe lift 和最终 lift 的夹爪位置、电流及融合判定。
- `execution_candidates.json`：最终 selected 候选和被拒原因。
- `prepared_candidate_ranking.json`：候选软排序和各评分项。

默认命令不生成执行侧 HTML/SVG/PNG 预览。需要视觉排障时，临时改为
`--debug-output-mode full --execution-debug-preview`。

关键日志解释：

- `CONTACT_REALIGN phase=approach`：approach 高位接触点对齐。
- `PREGRASP_REALIGN`：按物体最高点加 `--pregrasp-realign-clearance` 计算最后安全对齐高度。
- `CONTACT_REALIGN phase=pregrasp`：pregrasp 高位接触点对齐。
- `PREGRASP_REALIGN_APPLY ... ignored_z_delta=...`：只把 pregrasp 的 XY 修正应用到最终下降，Z 修正被忽略，避免最终夹持被安全高度 realign 抬高。
- `CONTACT_REALIGN_CHECK phase=grasp`：低位只检查 residual，不再横向 realign、回撤或 abort。
- `IK_FK_CONTACT_COMP`：最终下降前使用 IK 关节解做 FK，按预测接触点的 base-X/Y residual 迭代修正命令。
- `IK_FK_CANDIDATE`：候选 IK 解的完整接触点误差；`z_error` 是 X/Y 补偿无法消除的误差，超过上限时候选会在执行前被拒绝。
- `PREPARED_CANDIDATE_RANK`：按固定爪包络、目标体积质心距离、IK/FK 接触误差和候选置信度的综合软分排列执行顺序。
- `IK_FK_CANDIDATE_RANK`：仅在固定爪软排序关闭时使用，按无法补偿的 `z_error` 从小到大排列执行顺序。
- `prepared_candidate_ranking.json`：Hermes 执行层和本监督式脚本都会输出的 IK/FK 后软排序明细，优先保留目标位于固定爪与活动爪包络内的候选；固定爪前缘间隙偏好只降级排名，不单独拒绝候选。
- `IK_FK_GEOMETRY_CHECK`：用候选实际 IK 解的 FK 姿态重新检查夹爪网格和桌面间隙，detect-only 阶段也会执行。
- `MOVE_CONFIGURATION_RESULT`：通过 MoveIt 执行补偿时使用的同一组 IK 关节角，避免网关重新求解。
- `action=log_only_realign_threshold_exceeded` / `action=log_only_abort_threshold_exceeded`：说明 residual 超过对应日志阈值，但仍继续 close；这是现场策略，避免肉眼可抓位置被阈值误判后反复 retry。

如果固定指在目标前侧，仍可能出现固定指先碰边导致漏抓。此时不要只继续增大 `fixed_finger_margin_max_m`，还要在 `grasp_preview_so101_execution.html` 里检查 selected 候选的 SO101 mesh 方向，优先选择活动指能把目标扫回夹口的候选。

漏抓排查时不要漏看体积质心：`volume_xyz` 代表目标主体位置，比可见表面 `surface_xyz` 更适合判断物体是否在两指通道内。若 contact residual 看起来不大，但 `volume_xyz` 已经在固定指外侧或夹口外，仍会出现固定指先碰、活动指扫不回的漏抓。

## 8. 抓取后验证

完整抓取脚本会自动在三个阶段调用验证服务：

- `close`：闭合后、任何抬升前确认已经夹住；失败或不确定时保持夹爪闭合，先垂直撤回到
  pregrasp 高度，再打开夹爪并返回观察位。可用 `--no-recover-after-close-failure` 恢复旧的原位停止行为。
- `probe_lift`：以 `0.02` 速度抬升 `3 cm`，在低高度检查是否立即滑脱。
- `lift`：监督式验证仅抬升到 `5 cm`，确认目标仍在夹爪中；验证通过后再单独设计转运轨迹。

任一阶段返回 `STATUS_FAILED(0)` 或 `STATUS_UNCERTAIN(2)` 时，`required` 策略都会输出
`FLOW_RESULT success=False`，不会再把动作执行完成当成抓取成功。`optional` 仅允许在服务完全不可用时跳过；
服务一旦返回明确结果，失败或不确定仍会使流程失败。close 恢复只把机械臂带回可重新规划的安全观察位，
不会把本轮失败改成成功；`probe_lift` / `lift` 验证失败仍停在对应抬升位置。

需要单独检查当前夹持状态时，也可以手动调用：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && \
  ros2 service call /grasp_verifier/verify_grasp ibrobot_msgs/srv/VerifyGrasp \
  "{task_id: 'pick_001', text_prompt: 'banana', expected_target_width_m: 0.035, post_grasp_wait_s: 0.2}"
```

结果解释：

- `STATUS_SUCCESS(1)`：融合证据认为抓住。
- `STATUS_FAILED(0)`：融合证据倾向没抓住。
- `STATUS_UNCERTAIN(2)`：证据不足；重观察或保守重试，不要直接当失败。

`evidence` 中重点看 `gripper_position`、`gripper_current_abs_a`、`wrist_visibility`。脚本会同时把完整结果写入
`grasp_verification.json`，并在 selected candidate 下保存 `grasp_verification` 记录。

默认关闭 `--pick-diagnostics-detect`。夹爪贴近目标后腕部视野严重遮挡，重新分割会增加 close-to-lift 延迟，
并可能产生错误质心。显式启用时，距离实际接触点超过
`--pick-diagnostics-max-target-contact-distance` 的检测只记录为 `plausible: false`，不会覆盖抓取前可信目标。

## 9. 常用调参

目标位置有固定偏差时：

```bash
--target-offset-x 0.00 --target-offset-y 0.00 --target-offset-z 0.00
```

这些参数是单次运行的 base-frame 执行修正，不是手眼标定修正。GraspGen 模式会在 workspace、
桌面间隙和 IK/FK 检查前把修正应用到候选抓取位姿，因此修改后必须先用相同参数执行
`--detect-only`，确认仍有安全候选，再进行真实抓取。不要根据单个目标的结果直接把修正写入机器人
SSOT；不同目标的接触高度可能不同。

2026-07-17 的 marker 实测中，两次无 Z 修正的执行接触点分别比观测体积质心高 `7.10 mm` 和
`8.03 mm`，闭合电流均为 `0 A`。使用以下目标特定修正后，接触点高度差降至 `2.10 mm`，close、
probe lift 和 final lift 三阶段验证均为 `confidence=1.00`：

```bash
--prompt marker --target-offset-z -0.008
```

该值目前只作为 marker 测试覆盖项保留。已有 banana 样例在 `--target-offset-z 0.000` 下成功，
因此没有足够证据把 `-0.008 m` 设为全局默认值。

需要连续测试并在每次成功后释放目标时：

```bash
--release-after-success \
--release-drop-height-m 0.015 \
--release-settle-s 1.0
```

脚本会先完成 close、probe lift 和 final lift 三阶段验证，再下降到抓取位姿上方 `15 mm` 打开夹爪。
低位释放可以减少目标从 `5 cm` 高度直接掉落后的弹跳。close 验证失败会保持闭合并撤回到安全高度后
打开；probe/final retention 验证失败会在当前抬升位打开，再返回观察位。

prepared candidate 软排序 A/B 可用：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && source install/setup.bash && \
  python3 scripts/run_marker_grasp_ab_trials.py --trials-per-mode 10
```

runner 按 baseline、enhanced 交替执行。baseline 传入 `--no-prepared-candidate-scoring`，enhanced 传入
`--prepared-candidate-scoring`；其他抓取、偏移、安全和验证参数保持一致。每次尝试写独立日志，并在
`outputs/grasp_ab/<timestamp>_marker_ab/summary.json` 汇总结果。目标释放后如果移动到图像边界，需暂停
并重新居中，避免把视野丢失误判为候选排序效果。

2026-07-17 marker 正式 10+10 次测试结果为 baseline `5/10`、enhanced `5/10`。enhanced 成功
候选的平均体积质心距离由 `11.0 mm` 降到 `7.2 mm`，平均 IK/FK XY residual 由 `12.5 mm`
降到 `8.1 mm`，但本样本未观察到最终成功率提升。完整报告位于：

```text
outputs/grasp_ab/formal_marker_ab_20260717_report.md
```

最终抓取 X/Y 方向受 5-DOF 姿态误差影响时，优先使用动态 IK/FK 接触点补偿，而不是直接写死
`target-offset-x` / `target-offset-y`：

```bash
--ik-fk-contact-compensation \
--ik-fk-contact-tolerance 0.003 \
--ik-fk-contact-max-iterations 6 \
--ik-fk-contact-max-correction 0.030 \
--ik-fk-contact-max-xz-error 0.020
```

补偿同时调整 base-X 和 base-Y，Z 不补偿。若预测需要超过最大 X/Y 修正量，或未补偿的 Z 误差超过
`--ik-fk-contact-max-xz-error`，候选会被拒绝；执行最终下降时会通过
`/moveit_gateway/move_to_configuration` 使用同一组 IK 关节解。候选筛选和最终下降前都会基于该
IK 解的 FK 姿态重新检查 SO101 网格桌面间隙。

监督式真实抓取使用的桌面/高度保护：

```bash
--min-contact-z -0.045 --so101-tabletop-clearance 0.000
```

GraspGen 候选：

```bash
--grasp-threshold 0.5 --min-grasp-confidence 0.0 --max-candidates 80
```

`--max-candidates` 是 IK/FK 检查预算。脚本会先对 GraspGen 返回的全部候选做质心、top-down 和
置信度重排，再对全部候选执行 fixed-finger side、workspace、height 和 SO101 tabletop 等廉价
几何检查，最后只把排序靠前的 80 个通过者送入 IK/FK。这样不会因为前 80 个源候选被廉价门限
拒绝而丢掉后续可执行候选；`<=0` 时会检查全部通过廉价门限的候选。

`--ik-worker-count 4` 只并行候选准备。脚本先从 `/joint_states` 固定一份共同 IK seed，再把通过
workspace/height/tabletop 检查的候选分片到 4 个独立 MoveIt worker，最后按原候选顺序汇总。
最终抓取前的接触补偿、`move_to_configuration` 和所有运动仍走主 MoveIt 串行链路。默认值为 0，
因此未显式启用时行为仍是单 `/compute_ik`、`/compute_fk` 服务。

Top-down 偏好：

```bash
--graspgen-rank-by-centroid \
--centroid-source volume \
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

启用 `--so101-tabletop-filter` 时，即使传入 `none` / `diagnostic` / `default`，脚本也会强制请求
`full`，因为 SO101 执行侧 tabletop sweep 需要 `scene_cloud.ply`。

## 10. 常见问题

`Detect service is not available`：启动终端 B，检查：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 service list | grep grounded_sam2
```

`GraspGen service is not available`：启动终端 C，检查：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 && ros2 service list | grep /grasp_planner/plan_grasp
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

## 11. 启动 Hermes 抓取链路

终端 1：启动完整 ROS 抓取节点：

```bash
cd ~/IB_Robot && source .shrc_local && \
python3 scripts/synthesize_so101_grasp_runtime_config.py && \
export ROS_DOMAIN_ID=218 && source install/setup.bash && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_handeye_realsense_only \
  config_path:=/tmp/so101_handeye_realsense_grasp.yaml \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false \
  with_embodied:=true
```

该 launch 会自动启动：

- `grounded_sam2_node`：GDINO + SAM 检测分割。
- `grasp_planner_node`：GraspGen 抓取规划。
- `grasp_verifier_node`：抓取结果验证。
- `pick_executor_node`：正式抓取执行。
- 4 个 `/ik_worker_<n>/compute_ik`、`compute_fk`：与监督式脚本相同的候选并行准备池。
- `skill_executor_node` 和 `safety_guard_node`：Hermes 技能执行与安全校验。

不要再单独启动同名节点，否则会出现重复 service/action server。

终端 2：启动 Hermes：

```bash
cd ~/IB_Robot && hermes --cli
```
