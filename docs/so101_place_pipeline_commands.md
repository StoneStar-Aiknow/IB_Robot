# LeKiwi 固定位置释放与单关节观察位容器验证

`place_in_container` 不根据视觉结果规划候选放置位。它使用 robot YAML 中已经确认的固定关节目标执行释放，
再用请求中的 `container_name` 检测指定容器，并用 `target_name` 检测释放后的物品。顺序如下：

```text
place_container（3 号 raw 1500）→ open_gripper → 仅 3 号移动到 raw 1700 → 视觉验证 → 仅 3 号返回 raw 1500
```

`place_container` 只决定机械臂移动到哪里；`container_name` 只决定释放后 GDINO 检测哪个容器。指定容器不会
改变放置轨迹，也不会触发容器位姿估计、IK 或放置位规划。验证阶段对容器和物品的分割 mask 做二维包含判定，
要求物品中心位于容器区域内，且物品 mask 位于容器区域内的比例达到配置阈值。

`place_container` 是已由操作员在真机上确认的固定关节目标，写在两个 LeKiwi robot YAML 的
`placement_execution.motion.place_joint_positions` 中。当前释放目标（仅 1–5 号关节，弧度）为：

```text
1: -0.047553
2: -0.073631
3: -0.840621  # raw 1500
4:  1.497165
5: -1.570790
```

夹爪 6 号关节不属于放置位，流程在到位后单独调用 `open_gripper`。随后仅 3 号关节移动到验证目标
`-0.533825`（raw 1700）后采集图像验证，验证完成后仅 3 号返回 `-0.840621`（raw 1500）。
不移动到 `observe_table`，也不读取 `/robot_status/ee_pose` 或写入笛卡尔 `named_poses.place_container`。

## 启动

每个终端先加载项目环境；下例使用项目真机测试域：

```bash
cd /home/zf/IB_Robot
source .shrc_local
export ROS_DOMAIN_ID=218 ROS_LOCALHOST_ONLY=1
source install/setup.bash
```

310P 启动命令（Ascend NPU）：

```bash
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=lekiwi_handeye_realsense_grasp \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false \
  with_embodied:=true \
  authorize_motion:=true
```

PC 启动命令（NVIDIA CUDA）：

```bash
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=lekiwi_handeye_realsense_grasp_pc \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false \
  with_embodied:=true \
  authorize_motion:=true
```

## 执行前检查

- 停止 Hermes、任务规划器和其他动作调用方，确认 Gateway idle；
- 物品已稳定夹持，人员和工具退出工作区；
- robot YAML 中的固定 1–5 号关节目标已在现场验证，打开夹爪不会碰撞容器；
- 3 号电机 raw 1700 的验证位能让腕部相机看到容器内部和释放后的物品；执行器会把释放后的 RGB、检测 mask、开爪
  JointState 和判定结果写入 `placement_execution.debug_output_root`，便于离线审计和重放；
- 指定 `container_name` 后，视野中只检测到一个匹配该描述的目标容器；
- 腕部相机能够看到容器内部，`target_name` 和 `container_name` 与现场物品、容器描述一致；`target_name` 会
  直接作为 GDINO 文本查询，建议包含与现场一致的颜色等外观特征（例如 `red marker`），避免泛化的 `marker`
  把视野中的橙色夹爪误识别为第二个目标；
- 准备急停；失败、超时或释放状态未知时不得自动重试。

## 通过公开技能执行

`validate` 只做参数和安全预检，不创建执行任务，因此不使用 `--task-id`。`execute` 会创建可查询、可取消的
Gateway 任务，CLI 强制要求提供唯一且非空的 `--task-id`；每次新的真机执行都应使用新的 ID。

310P：

```bash
robot-skill --config-name lekiwi_handeye_realsense_grasp \
  validate place_in_container \
  --target-name "red marker" \
  --container-name "black bowl" \
  --timeout-sec 60

robot-skill --config-name lekiwi_handeye_realsense_grasp \
  execute place_in_container \
  --target-name "red marker" \
  --container-name "black bowl" \
  --task-id place-marker-310p-001 \
  --timeout-sec 60
```

PC：

```bash
robot-skill --config-name lekiwi_handeye_realsense_grasp_pc \
  validate place_in_container \
  --target-name "red marker" \
  --container-name "black bowl" \
  --timeout-sec 60

robot-skill --config-name lekiwi_handeye_realsense_grasp_pc \
  execute place_in_container \
  --target-name "red marker" \
  --container-name "black bowl" \
  --task-id place-marker-pc-001 \
  --timeout-sec 60
```

所有放置请求都必须走上述 Gateway；`/manipulation/execute_place` 是内部
delegated action，不能由 CLI 直接构造 `PlaceObject` goal。成功结果应满足：

```text
success=True place_succeeded=True release_status=1 verification_status=1
```

`release_status=1` 但 `verification_status=2` 表示物品明确在容器外；`verification_status=3` 表示视觉证据
不足或相互矛盾。两者都不得自动重新开爪，应由操作员查看现场后决定下一步。

## 离线证据重放

放置执行器从开爪前建立版本化证据目录，目录内包括：

- `placement_manifest.json`：固定位置流水线版本、配置和 executor identity；
- `open_gripper_joint_state.json`：开爪完成后的新鲜夹爪反馈；
- `sample_XX_rgb.npy`、`sample_XX_*_mask.npy` 和 `sample_XX.json`：每个新鲜验证帧及检测结果；
- `placement_result.json`：实际终态。

在不启动 ROS 或真机的情况下重放：

```bash
source .shrc_local
placement_replay outputs/placement_pipeline/<evidence-directory>
```

没有当前 manifest 的历史记录（例如旧三维容器规划输出）会报告 `legacy_incompatible`，不会被当作当前放置结果。
