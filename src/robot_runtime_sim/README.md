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
       name="my_platform",  # 匹配 ^[a-z][a-z0-9_]*$，不能叫 sdk
       package="my_pkg",  # 未安装报错时提示安装的包
       embodiment="my_pkg.embodiment:MY_EMBODIMENT",  # module:attr
       world=None,  # 通用 launch 的 world 钩子（D5，可后补）
       scene_task=None,
       sensor_traits=SimSensorTraits(),  # 默认 traits：不做平台特化改写
   )
   ```

2. 在你包的 `setup.py` 注册 entry point：

   ```python
   entry_points = {
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

## 选哪种接法：平台由谁控制、时间由谁推进

| 平台情况 | 例子 | 接法 |
|---|---|---|
| 仿真自己按时钟推进，机器人用 ros2_control | Gazebo、MuJoCo | 本包已有的平台：`SimPlatform` + 默认 embodiment（ros2_control、桥接、facade），`/clock` 由仿真发布 |
| 仿真自己推进，没有 ros2_control，但有自己的控制接口 | 厂商仿真 SDK、远程仿真服务 | 优先写 ros2_control 硬件接口，归入上一类；或者自定义 embodiment，起一个直接提供运行时公共契约的桥接节点（同 X2 厂商运动控制层的做法） |
| **没有控制器，仿真每一步都要等运行时给目标**（平台每拍来要动作，或只提供 `reset/step`） | RoboDojo（平台来要）、gym 式环境（只有 `reset/step`） | **`direct_control`**：运行时自己当控制器，见下节 |
| 本来能单步，但评测要求可复现 | MuJoCo、Isaac Sim 都能暂停 | 可以有意走 `direct_control` 的 host_step |

## direct_control：没有控制器的平台

`robot_runtime_sim.direct_control` 是这一类平台共用的运行时。运行时自己把
`MoveToPose` / `ComputeIk` / 关节和夹爪命令变成每一步的关节目标，平台只负责物理。
按需导入：选 Gazebo / MuJoCo 时不会加载它。

| 模块 | 职责 |
|---|---|
| `engine.py` | 纯逻辑（不导入 rclpy）：会话与 reset 代次（旧代次的请求一律拒绝）、指令生命周期（去重、取消、停止、墙钟期限、平台断联记为 UNKNOWN）、逐步生成目标、空闲不推进 |
| `ports.py` | 平台适配器端口：`AdapterDescriptor`、`ObservationFrame`、`PlatformCodec`、`PlatformResult`；读取 profile 的 `simulation.<platform>` 段 |
| `host_step.py` | 第二种推进方式：平台只提供 `reset/step` 时，由运行时扮演主循环 |
| `kinematics.py` | 位姿与关节路径的纯计算 |
| `node.py` | ROS 外壳（可执行文件 `direct_control_runtime`）：公共运行时契约、`/simulation/*` 会话接口、`/clock` |

三种时间各管一件事：

| 时间 | 用途 |
|---|---|
| 步号（会话内） | 会话记账；平台的步数预算 |
| ROS 时间 `/clock` | = 本进程所有会话累计的仿真时间，reset 不回退，发布序列不回跳。只在平台推进时前进，空闲时停住；消息时间戳都是仿真时间，订阅方用 `use_sim_time` |
| 墙钟期限 | 任务与动作的期限都按墙钟算（task_dispatch 单步运动、技能执行器的等待、运行时单条命令、平台长时间无调用），用来发现链路卡死，取值远大于正常动作所需；平台卡死时调用方照样能退出 |

运行时节点本身用墙钟（它就是时钟源），状态和关节消息按墙钟定时发布、暂停时照常发布，
消息时间戳是对应观测的仿真时间。

引擎没有评测概念。评测（准入、判分通道、任务说明等）是可选的外挂层：通过节点参数
`extensions`（`module:attr`，节点建好后调用一次）挂上，可以装运动闸门
（`SimulationCore.set_motion_gate`）、监听平台结果（`add_platform_result_listener`）并提供
自己的接口。不挂时，平台自己的判分结果只写日志，不发布到任何话题。

### 接入一个没有控制器的平台

1. 在 profile 里声明平台：`use_sim: true`、`sim_platforms: [my_platform]`、
   `sim_platform: my_platform`；`simulation.my_platform` 段写 `clock_mode`（`platform_pull` 或
   `host_step`）、`dt_s`、`supported_commands`、`ik`、`motion`、`adapter_config`。
2. 写适配器：工厂 `create_adapter(profile, overrides)`，返回带 `descriptor`、`codec`、
   `start(core, logger)`、`stop()` 的对象（参考 `robot_runtime_sim.testing.direct_control_fakes`）。
3. 写 embodiment：`actions()` 起 `robot_runtime_sim` 的 `direct_control_runtime`，参数
   `platform`、`adapter`（适配器工厂的 `module:attr`）、`use_sim_time: false`。适配器只由这里
   指定，没有单独的适配器注册表。
4. 注册平台：`ibrobot.sim_platforms` 入口点指向轻量的 `SimPlatform` spec（`world=None` 表示
   世界由平台自己管）。完整示例见 `robodojo_robot`。

现在只有 RoboDojo 一个真实平台；host_step 只用 CPU 替身测过。以后可能要补的扩展点：
运行时侧 IK 求解器（现在 IK 只能来自平台，`ik: platform`）、仿真自由运行时按 `/clock` 采样
下发目标（free_run）、移动底盘类运动组。

## 运行平台测试

```bash
colcon test --merge-install --packages-select robot_runtime_sim
```

需要真实仿真器的用例按 `IBROBOT_REQUIRE_SIM_PLATFORMS`（逗号分隔平台名）门控：默认
缺环境跳过并给出原因；验收跑设置该变量后跳过转为失败。真值与环境探测来自
`robot_runtime_sim.testing`。
