# sim_models

IB-Robot 仿真相关的场景资产、场景编译器和开发者标定辅助工具。

## 当前职责

| 在范围内 | 不在范围内 |
|---|---|
| Gazebo / MuJoCo 场景模板与道具资产 | 机器人本体 SSOT 配置 |
| 场景编译与运行时装配 | 业务控制逻辑 |
| 开发者用仿真标定辅助工具 | 真机相机标定 |

## 相机标定相关入口

### `ros2 run sim_models sim_camera_adjuster`

交互式 Gazebo 代理相机调参工具，支持：

- 实时调节 `xyz / rpy / fovy`
- 读取已有 override 作为初值
- 保存到 `~/.ros/ibrobot/sim_camera_overrides/<camera>.yaml`
- MuJoCo 目标下自动拉起一个 headless Gazebo 做位姿采集

常见调用方式：

```bash
ros2 run sim_models sim_camera_adjuster \
    --camera /camera/top/image_raw \
    --robot-config so101_single_arm
```

### `sim_models.aruco_spawner`

Python API，用于在仿真里动态插入和清理 ArUco A4 标定板：

```python
from sim_models.aruco_spawner import (
    spawn_aruco_gazebo,
    despawn_aruco_gazebo,
    verify_mujoco_model,
)
```

`dataset_tools.camera_alignment` 在仿真路径下会调用这组接口。

## 与其他包的关系

```text
dataset_tools.camera_alignment
    ├─► sim_models.aruco_spawner
    └─► ~/.ros/ibrobot/sim_camera_overrides/<camera>.yaml
                                 │
                                 ▼
robot_config.launch_builders.sim_backend.camera_presets
```

`~/.ros/ibrobot/sim_camera_overrides/` 是开发者标定侧的 opt-in 旁路，不是新的配置单一数据源。
