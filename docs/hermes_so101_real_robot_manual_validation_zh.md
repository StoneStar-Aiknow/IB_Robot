# Hermes 控制 SO-101 真机手动验证指南

本文用于在 Ubuntu/openEuler 源码工作区中，手动验证当前代码能否通过 Hermes 自然语言请求控制
SO-101 单臂机器人。

验证链路必须是：

```text
Hermes 自然语言请求
  -> robot-skill
  -> Agent plan / validate / confirm / execute
  -> Capability Gateway
  -> Safety Guard
  -> MoveIt
  -> ros2_control
  -> SO-101 真机
```

不得使用裸 ROS action、MoveIt、controller 或 primitive 命令发送运动。`ros2 control
list_controllers` 仅用于只读检查控制器状态。

## 1. 固定约定

- 本文示例工作区为 `/mnt/data/lwh/IB_Robot_0803`。如果仓库位于其他目录，只修改
  `IBROBOT_WS`。
- 在当前工作区内构建、启动和验证；使用该工作区根目录的 `venv`。
- 所有 ROS 进程使用 `ROS_DOMAIN_ID=52`。
- 每次真机启动必须显式传入 `moveit_display:=true`。
- 不生成 keystore，不设置任何 `ROS_SECURITY_*` 环境变量，也不使用 SROS2 enclave。
- `authorize_motion:=true` 只能由完成现场安全检查的操作员在启动命令中显式设置。
- 执行运动前，必须检查控制器状态，并由用户确认 Hermes 展示的 exact plan。
- 正常关闭或重启服务前，必须先通过 Hermes 执行 `recover_safe_pose` 并确认成功，然后才能运行
  `./scripts/cleanup_ros.sh`。

## 2. 启动前安全检查

启动前确认：

1. 机械臂已经完成校准，校准文件与当前真机匹配。
2. 串口设备存在且没有被其他进程占用。
3. 机械臂周围没有人员、线缆或其他障碍物。
4. 急停或断电手段可立即使用。
5. 机器人当前姿态不会在控制器激活时碰撞环境。
6. 当前 Git 分支和待验证代码正确。

可执行以下只读检查：

```bash
export IBROBOT_WS=/mnt/data/lwh/IB_Robot_0803
cd "$IBROBOT_WS"

git branch --show-current
test -e /dev/ttyACM0 && ls -l /dev/ttyACM0
test -f "$HOME/.calibrate/so101_follower_calibrate.json" && \
  ls -l "$HOME/.calibrate/so101_follower_calibrate.json"
```

任一必要设备或校准文件缺失时停止，不要启动真机。

## 3. 构建当前工作区

代码更新后，在工作区根目录执行完整构建。`.shrc_local` 会激活当前工作区根目录下的 `venv`：

```bash
export IBROBOT_WS=/mnt/data/lwh/IB_Robot_0803
cd "$IBROBOT_WS"
source .shrc_local

./scripts/build.sh
```

完整构建已经针对当前 commit 成功完成时，可以跳过本步骤。构建失败时停止，不要继续使用旧的
install overlay 启动真机。

## 4. 终端 1：启动真机流水线

打开第一个终端：

```bash
export IBROBOT_WS=/mnt/data/lwh/IB_Robot_0803
cd "$IBROBOT_WS"
source .shrc_local
source install/setup.bash
export ROS_DOMAIN_ID=52
export ROS_LOG_DIR=/tmp/ibrobot-real52

ros2 launch embodied_bringup embodied_pipeline.launch.py \
  config_path:="$IBROBOT_WS/src/robot_config/config/robots/so101_single_arm.yaml" \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=true \
  authorize_motion:=true \
  with_embodied:=true \
  with_perception:=false \
  entry_mode:=hermes
```

保持该终端前台运行。启动过程中应重点确认：

- SO-101 hardware 成功 `activate`；
- 没有舵机通信、校准或串口错误；
- RViz 正常启动；
- controller spawner 没有退出码 1；
- Gateway、Safety Guard 和 Agent plan 节点进入 ready 状态。

## 5. 终端 2：检查控制器和 Gateway

打开第二个终端：

```bash
export IBROBOT_WS=/mnt/data/lwh/IB_Robot_0803
cd "$IBROBOT_WS"
source .shrc_local
source install/setup.bash
export ROS_DOMAIN_ID=52
export ROS2CLI_DISABLE_DAEMON=1

ros2 control list_controllers
```

以下三个控制器必须全部为 `active`：

```text
joint_state_broadcaster
arm_trajectory_controller
gripper_trajectory_controller
```

只要有一个控制器不是 `active`，就不要启动 Hermes、不要发送动作，也不要反复重启机械臂服务。

控制器全部 active 后，继续检查受控入口：

```bash
ROBOT_CONFIG_PATH="$IBROBOT_WS/src/robot_config/config/robots/so101_single_arm.yaml"

robot-skill --config-path "$ROBOT_CONFIG_PATH" status
robot-skill --config-path "$ROBOT_CONFIG_PATH" list-skills
robot-skill --config-path "$ROBOT_CONFIG_PATH" describe open_gripper_skill
robot-skill --config-path "$ROBOT_CONFIG_PATH" describe recover_safe_pose
```

`status` 至少需要证明：

- `motion_authorized` 为 `true`；
- `control_plane_ready` 为 `true`；
- `active_control_mode` 为 `moveit_planning`；
- 没有其他 active task；
- `open_gripper_skill` 和 `recover_safe_pose` 都处于可执行状态。

任一命令非零退出或返回 not-ready/unauthorized 时停止，不要进入运动验证。

## 6. 终端 3：启动 Hermes

打开第三个终端：

```bash
export IBROBOT_WS=/mnt/data/lwh/IB_Robot_0803
cd "$IBROBOT_WS"
source .shrc_local
source install/setup.bash
export ROS_DOMAIN_ID=52

hermes-robot \
  --config-path "$IBROBOT_WS/src/robot_config/config/robots/so101_single_arm.yaml" \
  -- --cli
```

`hermes-robot` 会检查 Hermes 版本、`robot-skill`、已安装的 `ibrobot-control` skill、Gateway
control plane 和 Agent plan 接口。它不会启动机器人，也不会替操作员开启运动授权。

预检查失败时停止，并记录完整错误码和错误信息。

## 7. 使用自然语言验证打开夹爪

进入 Hermes 后输入：

```text
请控制真实机器人完成“打开夹爪”。
必须严格按 ibrobot-control 流程执行：
status -> list-skills -> plan-workflow -> describe -> validate-plan。
请先展示完整 workflow steps、每步参数、plan digest、registry epoch/generation/digest
和 fresh task ID。在我明确回复“确认执行”之前，不要调用 confirm-plan 或 execute-plan。
不要调用裸 ROS、MoveIt、controller 或 primitive。
```

Hermes 必须先完成以下只读阶段：

1. `robot-skill status`
2. `robot-skill list-skills`
3. 使用原始自然语言和 typed workflow 创建一次 `plan-workflow`
4. `robot-skill describe open_gripper_skill`
5. `robot-skill validate-plan --plan-token ...`
6. 展示 exact step、参数、plan digest、registry identity 和新的 task ID

确认以下内容完全正确：

- 计划只有一个 `open_gripper_skill` step；
- 没有额外技能、参数或未说明的运动；
- validation 返回 allowed；
- registry identity 与当前 Gateway status 一致；
- task ID 是本次请求的新 ID。

确认无误后，在 Hermes 中输入：

```text
确认执行
```

Hermes 随后应使用 exact tuple 执行一次 `confirm-plan`，再使用返回的 confirmation token 执行一次
`execute-plan`。如果显式使用 `--timeout-sec`，confirm 和 execute 必须使用完全相同的值。

等待唯一 terminal result。通过标准为：

```text
success: true
error_code: 空
executed_step_count: 1
```

同时现场确认夹爪实际打开。只看到 goal accepted、feedback 或进程退出都不能视为成功。

## 8. 回到安全原位

完成动作验证后，不要直接关闭 ROS。继续在同一个 Hermes 会话中输入：

```text
请让真实机器人回到安全原位。
必须使用 recover_safe_pose，并严格执行
status -> list-skills -> plan-workflow -> describe -> validate-plan。
先展示完整 workflow steps、参数、plan digest、registry identity 和新的 task ID，
等待我明确回复“确认执行”后，才能调用 confirm-plan 和 execute-plan。
```

确认计划只有一个 `recover_safe_pose` step、validation allowed 且 identity 正确后输入：

```text
确认执行
```

必须等待 terminal result 返回 `success: true`，并现场确认机械臂已经回到安全原位。

## 9. 正常关闭与重启

仅在 `recover_safe_pose` 已返回成功并确认机器人处于安全原位后执行：

```bash
export IBROBOT_WS=/mnt/data/lwh/IB_Robot_0803
cd "$IBROBOT_WS"
./scripts/cleanup_ros.sh
```

确认旧进程已经退出后，才能按照本文第 4 节重新启动。不要用直接 kill controller manager 的方式代替
正常恢复和清理顺序。

## 10. 失败、超时与取消处理

以下任一情况发生时，停止当前流程，不要自动重试，不要更换 task ID 后再次执行：

- Gateway not-ready 或 unauthorized；
- controller 未全部 active；
- plan 缺少、增加或重排了请求步骤；
- validate/confirm/execute 任一步非零退出；
- action timeout、transport failure 或结果缺少 terminal result；
- 返回 `SKILL_CANCEL_TIMEOUT` 或其他停止状态未知错误。

正在执行的 `execute-plan` 可通过 SIGINT/SIGTERM 请求取消，但必须等待 terminal result。也可以由另一受控
客户端执行：

```bash
robot-skill --config-path "$ROBOT_CONFIG_PATH" cancel-plan --task-id "<TASK_ID>"
```

“取消请求已发送”不等于“机器人已停止”。停止状态未知时不得继续发送 `recover_safe_pose` 或任何其他动作，
也不得把系统描述为已经安全停止；应先进行现场安全处置和状态确认。

如果 Gateway 不可用，以至于无法按要求执行 `recover_safe_pose`，不要盲目重启服务。先恢复受控入口或由现场
操作员完成安全处置，再决定是否清理和重启。

## 11. 验收记录

一次完整通过的真机验证至少应记录：

- Git commit 和分支；
- robot config 路径；
- `ROS_DOMAIN_ID=52`；
- 启动参数包含 `moveit_display:=true`、`authorize_motion:=true`；
- 三个 controller 的 active 状态；
- Gateway status 中的 registry epoch、generation 和 digest；
- `open_gripper_skill` 的 plan digest、task ID 和 terminal result；
- `recover_safe_pose` 的 plan digest、task ID 和 terminal result；
- 现场观察结果；
- 最终执行 `cleanup_ros.sh` 的时间与结果。

只有打开夹爪和回到安全原位两个动作都获得唯一、成功的 terminal result，才可以将本轮 Hermes 真机功能
验证标记为通过。
