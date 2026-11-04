# robot_runtime_sim

机器人运行时的仿真平台提供方：Gazebo 与 MuJoCo 的平台 spec、默认 embodiment、仿真专用节点与
仿真环境测试支撑。运行时契约层（`robot_runtime`）不依赖任何仿真器，只通过
`ibrobot.sim_platforms` entry point 发现这里的平台。

## 职责

- **平台 spec**（`platforms.py`）：`GAZEBO_SPEC` / `MUJOCO_SPEC` 通过
  `ibrobot.sim_platforms` entry point 注册，声明平台名、默认 embodiment 的 `module:attr`
  引用与传感器 traits。该模块必须 import 轻量（D1：仅标准库加
  `robot_runtime.sim_platforms` 类型）——构建接口描述会导入它。
- **Embodiment**（`gazebo_embodiment.py` / `mujoco_embodiment.py`）：平台的默认
  `SimEmbodiment`，从 `robot_runtime.launch_support` 迁出。包含描述渲染器
  （`render_gazebo_description` / `render_mujoco_description`）与整套 launch 组合
  （`gazebo_embodiment_actions` / `mujoco_embodiment_actions`）。
- **平台代码**（`gazebo.py` / `mujoco.py`）：传感器 URDF/SDF/MJCF 生成、桥接条目、
  相机参数树与版本检查。
- **仿真专用节点**（D6 节点归属：builder 随可执行文件一起迁移）：
  - `joint_current_publisher`（`sim_nodes.simulated_joint_current_actions`）；
  - `finite_cloud_filter`（`gazebo.lidar_finite_filter_nodes` /
    `mujoco.mujoco_lidar_filter_nodes`）；
  - `imu_settle_filter`（`gazebo.imu_settle_filter_nodes` /
    `mujoco.mujoco_imu_filter_nodes`）；
  - `odometry_twist_relay`（`mujoco.odometry_twist_relay_node`）；
  - `wait_for_clock`。
- **测试支撑**（`testing/sim_environment.py` / `testing/sim_ground_truth.py`）：
  仿真环境探测（require-or-skip 规则）与真值流，从 `robot_runtime.testing` 迁出。

## 如何新增一个仿真平台

1. 写一个 import 轻量的 spec 模块（只依赖 `robot_runtime.sim_platforms` 的类型）：

   ```python
   # my_pkg/platforms.py
   from robot_runtime.sim_platforms import SimPlatform, SimSensorTraits

   MY_PLATFORM = SimPlatform(
       name="my_platform",          # 匹配 ^[a-z][a-z0-9_]*$，不能叫 sdk
       package="my_pkg",            # 未安装报错时提示安装的包
       embodiment="my_pkg.embodiment:MY_EMBODIMENT",  # module:attr
       world=None,                  # 通用 launch 的 world 钩子（D5，可后补）
       scene_task=None,
       sensor_traits=SimSensorTraits(),  # 默认 traits：不做平台特化改写
   )
   ```

2. 在你包的 `setup.py` 注册 entry point：

   ```python
   entry_points={
       "ibrobot.sim_platforms": ["my_platform = my_pkg.platforms:MY_PLATFORM"],
   }
   ```

3. 实现 embodiment（`SimEmbodiment` 协议：`render_description(...)` 与 `actions(...)`）。
   可复用核心 `robot_runtime.launch_support` 的公共 builder（spawner、facade 节点、
   缺失帧 TF），也可以复用本包 `gazebo.py` / `mujoco.py` 的传感器生成代码。

4. 在机器人 profile 里声明 `sim_platforms: [sdk, my_platform]` 并选择
   `sim_platform: my_platform`。声明只是语法（D2）：未安装插件的主机仍能加载该
   profile；只有有效选择（`use_sim` 为真且平台生效）才要求已注册。

## 如何覆盖 embodiment 或传感器 traits

一个机器人可以不写任何 ros2_control 代码就替换平台在本机上的 embodiment 或 traits
（D3/D4）：在 profile 的平台段声明 `simulation.<platform>.embodiment` 为 `module:attr`
字符串。

```yaml
simulation:
  mujoco:
    embodiment: my_robot_pkg.embodiment:MY_MUJOCO_EMBODIMENT
```

- 覆盖的 embodiment 需实现同样的 `SimEmbodiment` 协议；解析是惰性的（每次调用都
  重新解析），失败的 import 或缺失属性按 `SimPlatformError` 报错，错误信息带 profile
  路径、平台名与引用。
- embodiment 可以带 `sensor_traits` 属性（可选）：存在时替换平台 traits（D4），
  影响接口描述与 `RuntimeStatus` 能力面在物理选择下的传感器投影。
- `validate_profile` 在加载时校验引用的字符串形式（必须是 `module:attr` 且两侧都
  非空：`:attr`、`module:`、空串与不含冒号的形式都按 `ProfileError` 拒绝并点名
  profile，注册性与可导入性仍留到有效选择时才查）；不设置时使用平台的默认
  embodiment。

## 运行平台测试

```bash
colcon test --merge-install --packages-select robot_runtime_sim
```

需要真实仿真器的用例按 `IBROBOT_REQUIRE_SIM_PLATFORMS`（逗号分隔平台名）门控：默认
缺环境跳过并给出原因；验收跑设置该变量后跳过转为失败。真值与环境探测来自
`robot_runtime_sim.testing`。
