# SO101 腕部 RealSense 抓取流水线

面向 310P 板端和 PC 真机抓取调试的最小命令集：腕部 RealSense -> Grounding-DINO/SAM2 named deployments ->
GraspGen -> MoveIt -> 可选抓取验证。两个平台使用相同硬件（SO101 机械臂 + 腕部 RealSense），
仅推理硬件不同：310P 用 Ascend NPU，PC 用 NVIDIA CUDA。

## 固定约定

- 仓库：`~/IB_Robot`
- ROS 域：`ROS_DOMAIN_ID=218`
- 单机抓取隔离：`ROS_LOCALHOST_ONLY=1`（pipeline、检查命令和客户端必须一致）
- 310P robot config：`so101_handeye_realsense_grasp`（Ascend NPU 推理）
- PC robot config：`so101_handeye_realsense_grasp_pc`（NVIDIA CUDA 推理）
- 腕部 RGB：`/camera/wrist/image_raw`
- 腕部对齐深度：`/camera/wrist/aligned_depth_to_color/image_raw`
- 腕部 CameraInfo：`/camera/wrist/aligned_depth_to_color/camera_info`
- 检测服务：`/perception/grasp/grounding_detect`
- 抓取规划服务：`/grasp_planner/plan_grasp`
- 抓取验证服务：`/grasp_verifier/verify_grasp`
- 统一执行入口：`/manipulation/execute_pick`（`manipulation_execution/pick_executor_node`）
- 监督式客户端：`ros2 run manipulation_execution pick_action_client`
- Hermes 控制面：`ibrobot-control` Agent Skill -> `robot-skill` -> ROS Capability Gateway
- Hermes 抓取技能：`pick_object`，不要写成 `pick-object`

两个配置文件硬件部分完全相同（机械臂端口、腕部相机话题、手眼标定值），唯一差异是推理后端：

| 配置项 | 310P (`so101_handeye_realsense_grasp`) | PC (`so101_handeye_realsense_grasp_pc`) |
|---|---|---|
| GraspGen 后端 | `ascend_local` | `local_cuda` |
| GraspGen 模型 | `/root/graspgen_310p_bundle` | `models/grasp/checkpoints/` |
| 感知 bundle | `*_ascend` + `ascend_310p` deployment | `grounded_sam2_swint_ogc` + `torch_cuda` deployment |
| 感知 service | `GroundingDINORawDetectPlugin` + `SegmentDetectionsPlugin` 分开 | `GroundingDetectPlugin` 合并 |
| segment_service | `/perception/grasp/segment_detections` | `""`（inline，不需要单独 service） |

抓取配置只发布 RGB、对齐深度和 CameraInfo。GraspGen 会从对齐深度直接构造目标点云，因此不再启动
未被抓取链消费的 ROS `PointCloud2` 发布/转发；大尺寸 RGB-D 消息由 RealSense 节点直接 remap 到稳定话题，
避免多次冷启动后 Python relay 与 Fast DDS UDP 队列叠加。

启动前必须检查对应 YAML 中的 `ros2_control.port`、相机序列号和手眼标定值。

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
  embodied_bringup robot_skill_cli perception_service manipulation_execution
```

在 `src/robot_config/config/robots/``so101_handeye_realsense_grasp.yaml` 或
`so101_handeye_realsense_grasp_pc.yaml` 中填写当前机器的硬件和标定值。

310P 板端启动前先在本机执行静态 preflight，不会产生机械臂运动：

```bash
cd /root/IB_Robot && source .shrc_local
test -f models/perception/grounding_dino_swint_seq8_1280x720_ascend/inference_manifest.json
test -f models/perception/sam2_hiera_tiny_ascend/inference_manifest.json
test -f /root/graspgen_310p_bundle/inference_manifest.json
test -e /dev/ttyACM0
test -e /dev/video0
python3 scripts/check_handeye_preconditions.py \
  --robot-config src/robot_config/config/robots/so101_handeye_realsense_grasp.yaml \
  --camera-name wrist \
  --check-files
ros2 pkg prefix robot_config
ros2 pkg prefix perception_service
ros2 pkg prefix manipulation_service
ros2 pkg prefix manipulation_execution
ros2 pkg prefix embodied_bringup
ros2 pkg prefix robot_moveit
ros2 pkg prefix so101_hardware
```

PC 端 preflight 使用 CUDA bundle：

```bash
cd ~/IB_Robot && source .shrc_local
test -f models/perception/grounded_sam2_swint_ogc/inference_manifest.json
test -f models/grasp/checkpoints/graspgen_robotiq_2f_140.yml
test -f models/grasp/checkpoints/graspgen_robotiq_2f_140_gen.pth
test -e /dev/ttyACM0
python3 scripts/check_handeye_preconditions.py \
  --robot-config src/robot_config/config/robots/so101_handeye_realsense_grasp_pc.yaml \
  --camera-name wrist \
  --check-files
```

任何命令失败都先补齐设备、标定文件、bundle 或板端构建，不要启动抓取执行。

## 1. 清理残留节点

上次启动被中断、机器人不响应、MoveIt action server 重复时执行：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && \
./scripts/cleanup_ros.sh
```

## 2. 启动服务

只使用统一 bringup，不再分别启动 robot、IK worker、planner、verifier 或 executor。

### 终端 A：统一抓取 pipeline

先确认工作区无人、急停可用、夹爪与桌面无干涉。第 4–8 节的监督式 action 会产生机械臂运动，
因此操作员必须在启动时显式授权：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_handeye_realsense_grasp \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false \
  with_embodied:=true \
  authorize_motion:=true
```

PC 端把 `robot_config` 换成 `so101_handeye_realsense_grasp_pc`，其余参数不变：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_handeye_realsense_grasp_pc \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false \
  with_embodied:=true \
  authorize_motion:=true
```

切换平台只需改 `robot_config` 参数，不需要改任何代码或重新编译。

该 launch 会自动启动：

- SO101 ros2_control、RealSense、MoveIt 和 task executor。
- `grasp_grounding_detect` 与 `grasp_segment_detections` generic model host。
- `grasp_planner_node`、`grasp_verifier_node` 和唯一 `pick_executor_node`。
- `robot.grasp_execution.ik.worker_count` 个隔离 IK/FK worker。
- `skill_executor_node` 和 `safety_guard_node`，为抓取执行提供统一 primitive/安全路径。

等待关键日志：

```text
Controllers are active
MoveIt Gateway fully initialized
TaskExecutor ready
GraspPlannerNode ready
PickExecutor ready: action=/manipulation/execute_pick ... ik_workers=4
```

如果只需查看 Gateway/技能目录而不允许任何运动，把上述命令的 `authorize_motion` 设为
`false`。该模式下不要运行第 4–8 节的 action 命令。

腕部相机被大目标遮挡时不会单独判失败，只记录诊断证据。完整抓取由
`robot.grasp_execution.verification: required` 要求验证服务；若服务未启动，正式 executor
会在任何抓取动作前退出。

### 终端 B：RViz，可选

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && rviz2
```

添加 `Image` display，topic 选 `/camera/wrist/image_raw`，Reliability 设为 `Best Effort`。

## 3. 启动后检查

运行时抓取几何配置：

```bash
grep -A16 "target_gripper:" src/robot_config/config/robots/so101_handeye_realsense_grasp.yaml
# PC 端改为 so101_handeye_realsense_grasp_pc.yaml
```

应确认内容等价于：

```text
fixed_finger_contact_ee: [-0.014, 0.0, -0.080]
fixed_finger_margin_m: 0.006
fixed_finger_margin_max_m: 0.012
fixed_finger_margin_width_ref_m: 0.035
fixed_finger_margin_width_gain: 0.25
```

`fixed_finger_contact_ee.z` 是 SO101 夹爪坐标系里的固定指接触深度，不是 base-Z 下压量。
当 planner debug 模式为 `diagnostic` 或 `full` 时，可用 `prepared_candidate_ranking.json` 中的
`target_width_m`、`fixed_finger_gap_m`、`fixed_finger_target_gap_m` 和 `moving_finger_gap_m`
复核动态固定指 margin 的结果。

控制器：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && ros2 control list_controllers
```

应看到：

```text
joint_state_broadcaster active
arm_trajectory_controller active
gripper_trajectory_controller active
```

服务和 action：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && ros2 service list | grep -E 'grounding_detect|segment_detections|plan_grasp|compute_ik|compute_fk|move_to_configuration|verify_grasp'
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && ros2 action info /move_action
```

`/move_action` 必须只有一个 `/move_group` action server。若有多个，回到第 1 步清理。

相机：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && ros2 topic info /camera/wrist/image_raw
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && ros2 topic info /camera/wrist/aligned_depth_to_color/image_raw
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && ros2 topic info /camera/wrist/aligned_depth_to_color/camera_info
```

## 4. 只移动到观测姿态

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
  ros2 run manipulation_execution pick_action_client \
  --prompt banana \
  --mode observe_only
```

观测位姿和速度来自正在运行的 robot config（310P 或 PC），不允许由测试客户端覆盖。

## 5. 冒烟测试

感知服务 readiness：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && ros2 service list | \
  grep -E '/perception/grasp/(grounding_detect|segment_detections)'
```

抓取规划：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && ros2 service call /grasp_planner/plan_grasp ibrobot_msgs/srv/PlanGrasp \
  "{text_prompt: 'banana', confidence_threshold: 0.1, grasp_threshold: 0.5, debug_output_mode: 'diagnostic'}"
```

## 6. 仅规划，不抓取

会移动到观测姿态、检测、规划、做 IK/执行侧筛选，抓取前退出。

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
  ros2 run manipulation_execution pick_action_client \
  --prompt banana \
  --mode plan_only \
  --timeout-s 240
```

通过标志：

```text
PICK_FEEDBACK phase=planning ...
PICK_FEEDBACK phase=selecting ...
PICK_ACTION_RESULT success=True ... candidate=<index> ...
FLOW_RESULT success=True
```

默认 `planner.debug_output_mode: none`，因此 plan-only 以终端日志、`PICK_ACTION_RESULT` 和其
`pipeline_timings_json` 为准。如果启动前已把 debug 模式改为 `diagnostic` 或 `full`，再检查
`PICK_ACTION_RESULT.debug_output_dir` 指向的目录：

- `grasp_result.json`
- `pick_frame_diagnostics.json`
- `prepared_candidate_ranking.json`

plan-only result 的 `candidate_index` 只表示正式 pipeline 已完成候选准备，不能把它误认为已经执行过真实抓取。

如果出现 `NO_SAFE_GRASP_CANDIDATES`、`TARGET_TABLETOP_UNAVAILABLE`、
`TARGET_GEOMETRY_FAILED`，或候选明显在桌面下，不要继续完整抓取。

正式配置中的 `target_geometry.tabletop_filter` 优先使用 PlanGrasp 响应中的 execution table plane，不需要为了正常抓取
生成完整 PLY 和 HTML/SVG/PNG 预览。如果响应中没有可用平面，候选会 fail closed 并以
`TARGET_TABLETOP_UNAVAILABLE` 拒绝，不会静默放行。只有排查点云或夹爪几何时，才临时把
`robot.grasp_execution.planner.debug_output_mode` 改为 `full`，并重启正式 executor。

同时检查终端日志中的：

- `PlanGrasp('banana'): <n> grasps, ...`：候选生成成功且数量非零。
- `PIPELINE_TIMING stage=candidate_geometry_ranking`：源排序与 SO101 廉价几何门禁已完成。
- `prepared candidate rank: ...`：`execution_scoring.centroid_source: volume` 已参与候选排序。

## 7. 完整抓取

只在第 6 步正常后执行。

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
  ros2 run manipulation_execution pick_action_client \
  --prompt banana \
  --mode execute \
  --timeout-s 240
```

候选阈值、IK worker 数、接触补偿、恢复、速度和验证策略全部来自启动正式 executor 时使用的
`robot.grasp_execution`，客户端不再维护第二套参数。

固定指 robust-gap 只在批量候选准备阶段作为 hard gate。候选已经到达 pregrasp 并完成在线接触补偿后，
复算结果仅记录诊断，不会再在下降闭爪前撤回或切换候选。

通过标志：

```text
PICK_FEEDBACK phase=planning ...
PICK_FEEDBACK phase=selecting ...
PICK_FEEDBACK phase=close ...
PICK_FEEDBACK phase=verify_probe ...
PICK_FEEDBACK phase=verify_lift ...
PICK_ACTION_RESULT success=True ... candidate=<index> ...
FLOW_RESULT success=True
```

如果 planner debug 模式为 `diagnostic` 或 `full`，执行后重点看同目录的：

- `pick_pose_diagnostics.json`：实际位姿误差和接触点 residual。
- `grasp_verification.json`：close、3 cm probe lift 和最终 lift 的夹爪位置、电流及融合判定。
- `pick_frame_diagnostics.json`：capture-time TF 查询时间戳和回退模式。
- `prepared_candidate_ranking.json`：候选软排序和各评分项。

当前 executor 不生成独立的 SO101 execution HTML/SVG/PNG 预览。需要视觉排查 planner
点云和源候选时，临时把
`robot.grasp_execution.planner.debug_output_mode` 改为 `full`。

关键日志解释：

- `PIPELINE_TIMING stage=graspgen_request`：完整 `PlanGrasp` 请求耗时。
- `PIPELINE_TIMING stage=candidate_geometry_ranking`：源排序、SO101 adapter、workspace/height/tabletop
  等廉价几何门禁的合并耗时。
- `PIPELINE_TIMING stage=candidate_ik_fk`：候选 IK/FK 准备耗时；同行的 `workers` 和 `candidates`
  表示动态工作队列规模。
- `IK worker verification passed`：主 MoveIt 与全部 worker 在共同 seed/验证位姿上结果一致；
  `cached=true` 表示命中进程内缓存。
- `pick candidate preparation failed ... code=...`：候选在 IK/FK、接触补偿、joint5、固定指侧或
  最终网格门禁失败。
- `prepared candidate rank: ...`：按固定爪包络、体积质心距离、IK/FK 接触误差、
  robust-gap headroom 和 confidence 的综合软分排列执行顺序。
- `contact realign phase=approach|pregrasp`：安全高度的接触点对齐 residual。
- `grasp prediction candidate=...`：最终下降使用的 IK/FK 预测位姿、闭合轴误差和接触 residual。
- `PICK_FEEDBACK phase=descend` 后必须直接出现 `phase=close`；`close_gripper` 是下降成功后的第一条动作。
- `pose diagnostic label=grasp ... action=...`：闭爪后以 best-effort 记录低位 residual；`log_only_*` 不触发
  横向 realign、候选切换，也不能中断闭爪或抓后验证。
- `grasp verification phase=verify_close|verify_probe|verify_lift`：三阶段抓后验证证据。
- `pick candidate failed ... code=...`：当前物理执行候选失败；`retryable=true` 时才可切换到下一个
  已准备候选。
- `post-success release completed`：成功验证后的低位释放已完成。
- `PICK_ACTION_RESULT ... pipeline_timings_json=...`：客户端收到的最终结果、候选、尝试数、
  验证状态和分阶段计时。

`prepared_candidate_ranking.json` 是上述软排序的结构化明细，仅在 planner debug 模式为
`diagnostic` 或 `full` 时写出。

如果固定指在目标前侧，仍可能出现固定指先碰边导致漏抓。此时不要只继续增大
`fixed_finger_margin_max_m`，还要检查 `pick candidate preparation failed` 的
`FK_FIXED_FINGER_BASE_SIDE_*` 错误、`prepared candidate rank` 和 `prepared_candidate_ranking.json`
中的固定指内侧/包络评分。

漏抓排查时不要漏看体积质心：`volume_xyz` 代表目标主体位置，比可见表面 `surface_xyz` 更适合判断物体是否在两指通道内。若 contact residual 看起来不大，但 `volume_xyz` 已经在固定指外侧或夹口外，仍会出现固定指先碰、活动指扫不回的漏抓。

## 8. 抓取后验证

正式 `pick_executor_node` 会自动在三个阶段调用验证服务：

- `close`：闭合后、任何抬升前确认已经夹住；失败或不确定时保持夹爪闭合，先垂直撤回到
  pregrasp 高度，再打开夹爪并返回观察位；该策略由 `recover_after_close_failure` 配置。
- `probe_lift`：按 `probe_lift_velocity_scaling: 0.05` 抬升 `3 cm`，在低高度检查是否立即滑脱。
- `lift`：当前正式流程抬升到 `5 cm`，确认目标仍在夹爪中；验证通过后再单独设计转运轨迹。

任一阶段返回 `STATUS_FAILED(0)` 或 `STATUS_UNCERTAIN(2)` 时，`required` 策略都会输出
`FLOW_RESULT success=False`，不会再把动作执行完成当成抓取成功。`optional` 仅允许在服务完全不可用时跳过；
服务一旦返回明确结果，失败或不确定仍会使流程失败。close 恢复只把机械臂带回可重新规划的安全观察位，
不会把本轮失败改成成功；`probe_lift` / `lift` 验证失败时，当前 SSOT 会在对应
抬升位打开夹爪并返回观察位，由 `recover_after_retention_failure: true` 控制。

需要单独检查当前夹持状态时，也可以手动调用：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
  ros2 service call /grasp_verifier/verify_grasp ibrobot_msgs/srv/VerifyGrasp \
  "{task_id: 'pick_001', text_prompt: 'banana', expected_target_width_m: 0.035, post_grasp_wait_s: 0.2}"
```

结果解释：

- `STATUS_SUCCESS(1)`：融合证据认为抓住。
- `STATUS_FAILED(0)`：融合证据倾向没抓住。
- `STATUS_UNCERTAIN(2)`：证据不足；重观察或保守重试，不要直接当失败。

`evidence` 中重点看 `gripper_position`、`gripper_current_abs_a`、`wrist_visibility`。当 planner
debug 模式为 `diagnostic` 或 `full` 时，正式 executor 会同时把完整结果写入
`grasp_verification.json`。

正式 executor 不在 close-to-lift 阶段重新运行目标分割。夹爪贴近目标后腕部视野严重遮挡，重新分割既会
增加延迟，也容易产生错误质心；抓后判定统一由 `grasp_verifier_node` 的夹爪位置、电流和可见性证据完成。

## 9. 常用调参

所有会改变候选、安全门禁、IK/FK、速度、恢复或验证结果的参数都必须修改
`src/robot_config/config/robots/` 下对应平台的 YAML（`so101_handeye_realsense_grasp.yaml` 或
`so101_handeye_realsense_grasp_pc.yaml`）的 `robot.grasp_execution`，然后重启
正式 executor。监督式 action 客户端只允许选择目标、运行模式、预算和是否在成功后释放，禁止再用 CLI
构造第二套运行配置。

目标位置出现固定偏差时，不再通过监督式 CLI 注入单次 base-frame offset。先用 `plan_only` 和
`pick_pose_diagnostics.json` 判断偏差来自手眼标定、目标夹爪接触几何还是 IK/FK residual，再修改对应的
robot YAML SSOT，并重启 executor。不要根据单个目标的一次结果写入全局修正。

需要连续测试并在每次成功后释放目标时：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
ros2 run manipulation_execution pick_action_client \
  --prompt marker \
  --mode execute \
  --release-after-success \
  --release-drop-height-m 0.015 \
  --repeat 5
```

正式 executor 会先完成 close、probe lift 和 final lift 三阶段验证，再下降到抓取位姿上方 `15 mm` 打开夹爪。
低位释放可以减少目标从 `5 cm` 高度直接掉落后的弹跳。close 验证失败会保持闭合并撤回到安全高度后
打开；probe/final retention 验证失败会在当前抬升位打开，再返回观察位。

连续测试必须优先使用 `--repeat`，不要在 shell 循环里反复启动 `ros2 run`。复用同一个 action client
participant 可以避免大 ROS graph 中反复发现端点造成 UDP 接收缓冲拥塞。若日志出现
`PICK_ACTION_GOAL_RESPONSE_RECOVERY`，表示 goal 已按原 UUID 查询结果，客户端不会重发动作。

最终抓取 X/Y 方向受 5-DOF 姿态误差影响时，优先在 robot YAML 中使用动态 IK/FK 接触点补偿，
不要在监督式客户端中维护一次性 X/Y 偏移：

```yaml
contact_compensation:
  enabled: true
  xy_tolerance_m: 0.003
  max_iterations: 6
  max_correction_m: 0.030
  max_z_error_m: 0.020
```

补偿同时调整 base-X 和 base-Y，Z 不补偿。若预测需要超过最大 X/Y 修正量，或未补偿的 Z 误差超过
`contact_compensation.max_z_error_m`，候选会被拒绝；执行最终下降时会通过
`/moveit_gateway/move_to_configuration` 使用同一组 IK 关节解。候选筛选和最终下降前都会基于该
IK 解的 FK 姿态重新检查 SO101 网格桌面间隙。

监督式真实抓取使用的桌面/高度保护：

```yaml
candidate_selection:
  min_contact_z: -0.045
target_geometry:
  tabletop_clearance_m: -0.020
```

GraspGen 候选：

```yaml
planner:
  grasp_threshold: 0.5
candidate_selection:
  min_confidence: 0.0
  max_candidates: 80
```

`candidate_selection.max_candidates` 是 IK/FK 检查预算。正式 executor 会先对 GraspGen 返回的全部候选做质心、top-down 和
置信度重排，再对全部候选执行 fixed-finger side、workspace、height 和 SO101 tabletop 等廉价
几何检查，最后只把排序靠前的 80 个通过者送入 IK/FK。这样不会因为前 80 个源候选被廉价门限
拒绝而丢掉后续可执行候选；`<=0` 时会检查全部通过廉价门限的候选。

`robot.grasp_execution.ik.worker_count: 4` 只并行候选准备。正式 executor 先从 `/joint_states` 固定一份共同
IK seed，4 个独立 MoveIt worker 从共享动态队列领取候选，最后按原候选顺序汇总。最终抓取前的接触补偿、
`move_to_configuration` 和所有运动仍走主 MoveIt 串行链路。设为 `0` 时退回主 `/compute_ik`、
`/compute_fk` 服务串行准备。

Top-down 偏好：

```yaml
execution_scoring:
  centroid_source: volume
  contact_distance_weight: 1.0
  topdown_weight: 0.35
candidate_selection:
  topdown_min_z: -0.25
```

关闭 top-down 和质心重排，回到 GraspGen 置信度顺序：

```yaml
execution_scoring:
  contact_distance_weight: 0.0
  topdown_weight: 0.0
```

临时只验证位置目标：

```yaml
ik:
  check_orientation: false
```

只做感知排障时直接调用第 5 节的 `/grasp_planner/plan_grasp` 服务。`PickObject.MODE_PLAN_ONLY` 仍会执行
正式 IK/FK 和安全筛选，不提供绕过 IK guard 的模式。

调试输出级别：

```yaml
planner:
  debug_output_mode: none        # 不写文件
# debug_output_mode: diagnostic  # 只写 grasp_result.json
# debug_output_mode: full        # 写点云和预览
```

启用 `target_geometry.tabletop_filter` 时不再强制 `full`；正式 executor 直接使用 `PlanGrasp` 响应中的
execution table plane 做 SO101 网格 hard gate，不依赖 `scene_cloud.ply`。

## 10. 常见问题

`GroundingDetect service not available` 或 `segment_service_unavailable`：检查终端 A 的 generic model host：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && ros2 service list | grep /perception/grasp
```

`GraspGen service is not available`：检查终端 A 的 planner 服务：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && ros2 service list | grep /grasp_planner/plan_grasp
```

`No synchronized depth/CameraInfo for detection frame`：

- 确认 generic model host 与可选 RViz 都使用 `/camera/wrist/...` 同一组 topic。
- 确认三路相机 topic 都有发布者。
- 若有重复节点或 topic 重名，回到第 1 步清理。

`GraspGen returned zero candidates`：

- 临时把 `robot.grasp_execution.planner.debug_output_mode` 改为 `full`，查看 mask、depth 和
  `grasp_result.json`。
- 临时降低 `robot.grasp_execution.planner.grasp_threshold`。
- 必要时临时关闭服务端严格过滤做对比，但真机执行前必须保留执行侧 guard。

`Motion timed out after 60.0s`：

- 重启终端 A。
- 检查控制器和 `/joint_states`。

`STATUS_ABORTED` 但控制器像是完成了：

- 检查 `/move_action` 是否有多个 `/move_group` action server。
- 有重复就回到第 1 步清理。

抓取位置偏差大：

- 先运行 `scripts/check_handeye_preconditions.py`，并检查 robot YAML 中 wrist 相机 transform。
- 确认启动配置与当前平台匹配（310P 用 `so101_handeye_realsense_grasp`，PC 用
  `so101_handeye_realsense_grasp_pc`），正式 executor 会从同一 robot YAML 获取
  相机变换和目标夹爪几何。
- 若正式 executor 没有 `prepared candidate rank`、`grasp prediction candidate` 或
  `pick candidate preparation failed` 日志，检查当前安装空间是否已重新构建。

## 11. 启动 Hermes 抓取链路

### 11.1 启动 pipeline

Hermes 抓取复用同一组 pipeline 节点（planner、verifier、executor、IK worker），不要为 Hermes
再启动第二组。310P 使用 Ascend NPU 板端推理，必须先从 PC 通过 SSH 登录 310P；PC 流程使用
NVIDIA CUDA 本机推理，直接在 PC 终端执行。

终端 A：先 SSH 登录 310P，再在板端启动完整 pipeline：

```bash
ssh root@<310p-ip>
cd /root/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_handeye_realsense_grasp \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false \
  with_embodied:=true \
  authorize_motion:=true
```

终端 A：启动完整 pipeline（PC）：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_handeye_realsense_grasp_pc \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false \
  with_embodied:=true \
  authorize_motion:=true
```

等待日志：

```text
Controllers are active
MoveIt Gateway fully initialized
TaskExecutor ready
GraspPlannerNode ready
PickExecutor ready: action=/manipulation/execute_pick ... ik_workers=4
```

### 11.2 启动 Hermes CLI

对于 PC 流程，保持终端 A 的 pipeline 运行，另开 PC 终端 B，加载与 pipeline 相同的 ROS
发现环境后启动交互式 Hermes CLI：

```bash
cd ~/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
hermes --cli
```

对于 310P 流程，保持第一个 SSH 终端中的 pipeline 运行，另开 PC 终端 B，再次 SSH 登录
同一块 310P，然后在 310P 上启动 Hermes CLI：

```bash
ssh root@<310p-ip>
cd /root/IB_Robot && source .shrc_local && export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1 && source install/setup.bash && \
hermes --cli
```

PC 和 310P 抓取都由 Hermes 使用 `ibrobot-control` 完成，不在 Hermes 终端手工执行裸 `ros2`、
primitive、MoveIt 或 controller 运动命令。首先在 Hermes CLI 中输入无运动校验请求：

```text
请使用 ibrobot-control，针对 so101_handeye_realsense_grasp_pc 配置检查 Gateway 状态，
依次完成 list-skills、describe pick_object 和 validate pick_object，目标是 banana。
只校验，不要执行机械臂运动。
```

310P 的 Hermes CLI 使用同样的请求，但配置名必须改为
`so101_handeye_realsense_grasp`。

Hermes 应通过 `robot-skill` 按 `status -> list-skills -> describe -> validate` 顺序返回 Gateway 与
`pick_object` 契约的实际校验结果。任一步失败、Gateway 未就绪或运动未授权时都停止，
Hermes 不启动或重启 pipeline，也不修改 `authorize_motion`。

### 11.3 在 Hermes CLI 中执行抓取

只有终端 A 的 pipeline 已由操作员显式设置 `authorize_motion:=true`，且上一步 `validate`
成功后，操作员才在同一 Hermes CLI 会话中给出本次明确的运动确认和全新 task ID：

```text
我已完成现场安全检查，并明确确认本次机械臂抓取运动。
请继续使用 ibrobot-control，以 task ID pick-banana-001 执行刚才校验的
pick_object，目标是 banana。失败、超时或停止状态未知时不要自动重试。
```

Hermes 应在 CLI 中展示 `robot-skill` 返回的 JSONL feedback 和唯一 terminal result。如需取消，
应取消当前 execute 进程或使用同一 task ID；只有 terminal result 才能证明任务已收敛。

两个平台的抓取目标和 task ID 都必须由操作员明确提供；310P 的 Hermes 会话与 pipeline
必须保持在同一块 310P 上，不要在 PC 本地另起一个 Hermes 会话跨机绕过 Gateway。
