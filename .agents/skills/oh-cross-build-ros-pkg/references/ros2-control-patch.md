# Special Case: `ros2_control` / `controller_manager`

## When to Read

- 交叉编译 `ros2_control` 或 `controller_manager` 时
- SO-101 真机依赖 `ros2_control_node` + `controller_manager` + spawner 时
- spawner 崩溃或卡在 switch controller 时

SO-101 真机依赖 `ros2_control_node` + `controller_manager` + spawner。OpenHarmony musl 环境下有一个必须确认的关键补丁。

## 源码来源

```text
https://gitcode.com/openharmony-robot/ros_ros2_control
tag OpenHarmony-Embodied-v1.0.1-Release (commit c742704)
```

包含 `ros2_control`、`controller_manager`、`realtime_tools` 等包（版本 2.53.0）。该 fork 已包含 `realtime_tools` 的 `__OHOS__` 平台适配（跳过 `mlockall`、thread affinity 等 glibc-only 路径）。

## controller_manager mutex 补丁（关键）

`controller_manager/src/controller_manager.cpp` 的 `switch_controller()` 中，`std::defer_lock` 在 musl 上会导致 `condition_variable::wait_for` 不稳定，必须改为直接加锁：

```diff
-  std::unique_lock<std::mutex> switch_params_guard(switch_params_.mutex, std::defer_lock);
+  std::unique_lock<std::mutex> switch_params_guard(switch_params_.mutex);
```

RoboFrame 发布包的 `install.sh` 会通过 patched `.so` 覆盖机制自动应用此修复（`/data/roboframe/install/patches/lib/`）。

## 验证

正常 launch 日志应包含：

```text
[ros2_control_node]: Successful initialization of hardware 'RobotSystem'
[spawner_joint_state_broadcaster_group]: Loaded joint_state_broadcaster
[spawner_joint_state_broadcaster_group]: Configured and activated all the parsed controllers list
```

如果 spawner 崩溃或卡在 switch controller，说明 `controller_manager` 未包含 mutex 补丁。
