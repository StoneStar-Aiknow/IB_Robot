# X2 MuJoCo 独立仿真

`aimdk_mujoco` 是独立的 `ament_python` 包，用 MuJoCo 在本地运行 X2 的站立和平面行走。
当前 SDK 策略表现为低抬脚的贴地小步，尚未通过正常步态验收；身体前移不等于行走成功。
不需要 AimDK 运行时、`aimdk_msgs`、`aimdk_robot` 或厂商进程，不连接真实机器人。
它不发布公共机器人运行时的 `ACTIVE` 或就绪声明，也不把仿真真值伪装成厂商里程计。

本包不是 `sim_platform:=mujoco` 的运行时提供方。
通用 `robot_config` launch 不能用它接入 Agent、推理或录制。
X2 运行时的 `sim_platform:=sdk` 仍使用无物理的厂商协议 mock。

后续平台化改造将另起 OpenSpec 变更，由维护者确认后实施。
目标是在 `aimdk_msgs` 厂商协议接缝替换 MC/HAL，保留运行时和数据面。
物理宿主计划对齐 `mujoco_ros2_control`，复用场景、传感器和时钟机制。
实施前需确认 effort 接口、动态 PD 增益和策略同步。
还需确认 launch 组装、外部 overlay 和仿真时间的新鲜度判断。

## 本地资源

需要 ROS 2 Python 环境、MuJoCo 和 ONNX Runtime。ROS 节点启动后始终为暂停的模型预览；
运行策略需要提交 action 显式选择控制器，并提供具有合法使用权限的匹配策略权重。
Ubuntu full profile 的 `setup.sh` 安装固定版本的 MuJoCo。
ONNX Runtime 已由项目 requirements 安装，无需再次安装。
其他非 LIBERO 环境可在仓库根目录安装同一版本清单，再构建本包：

```bash
source .shrc_local && python3 -m pip install -r requirements/mujoco-sim.txt
source .shrc_local && ./scripts/build.sh --aimdk
```

`--aimdk` 构建 X2 运行时及其工作区依赖，不下载模型或策略资源。
支持的平台还构建独立工具 `aimdk_mujoco`，不把它注册为运行时仿真平台。
其中 `aimdk_robot` 运行时需要外部提供的 `aimdk_msgs` overlay，独立仿真不依赖它。
只需要独立仿真时，也可使用 `./scripts/build.sh -- --packages-up-to aimdk_mujoco`，同时构建专用接口包 `aimdk_interfaces`。

MuJoCo 必须为 `3.12.0`，与 `mujoco_vendor` 和项目版本清单一致。
不要在要求 `mujoco<3.9` 的 LIBERO 环境安装本包；请使用独立环境。
`colcon` 不会自动安装 pip 依赖。

openEuler 的标准 setup 不安装 MuJoCo，本工具在该平台默认不构建。
本 PR 未验证 openEuler 上的物理运行或测试。
只有自行准备匹配依赖后，才可显式启用构建：

```bash
source .shrc_local && IBR_BUILD_INCLUDE_AIMDK_MUJOCO_ON_OPENEULER=1 ./scripts/build.sh -- --packages-up-to aimdk_mujoco
```

该开关不安装引擎，也不代表 openEuler 运行验收通过。

本包不自带策略权重或厂商资产。
策略及相关模型资产只在本地使用，应放在仓库外的独立资源目录中，不能随源码提交。
本包源码的 Apache-2.0 许可证不授予第三方策略、网格或模型的再分发权。
使用前请自行核查 SDK 和策略的许可证及授权范围，不要把这些文件上传到 PR。

在仓库根目录、包已可导入的环境下准备资源：

```bash
source .shrc_local && python -m aimdk_mujoco.assets --assets /absolute/path/to/x2-assets
```

公开机器人资源的下载与策略权重独立，准备资源时不需要策略，也不会复制权重到资产目录。
运行时读取准备好的资产目录，不需要外部 AimDK 安装路径。
没有默认资产目录；准备、启动和验证时都必须通过 `--assets` 或 launch 的 `assets:=` 显式指定。
下文的 `/absolute/path/to/x2-assets` 应替换为同一个实际目录，建议使用绝对路径，避免依赖启动目录。
下载器只准备公开资产，不下载或复制任何 ONNX 权重；运行时也不会自动获取权重。
机器人模型下载自公开仓库 `AgibotTech/agibot_x2_urdf`，固定版本
`77f43eb0904dae4c48ccd9154fee824f8ffd4d38`，遵循 MulanPSL-2.0，许可证保存在资产目录。
下载慢时可用 `--robot-cache <SDK中的X2_URDF-v1.3.0目录>` 提供网格缓存，
每个文件仍须匹配公开版本的 Git blob 哈希；准备后复制成本地普通文件，不创建外部软链接。
中断后可加 `--resume` 继续；完成的目录不会被覆盖。`assets.json` 只记录公开机器人文件、许可证、来源和 SHA256，不包含策略。

## 支持的控制配置

项目已提供 [config/mc_rl_velocity.yaml](config/mc_rl_velocity.yaml)，构建时安装到
`share/aimdk_mujoco/config/`，无需用户另行创建 YAML。其内容为：

```yaml
controller: mc_rl_velocity
policy: policy.onnx
```

| 选择 | 配置文件 | `motion` | 支持的策略及用途 | 模型历史验证范围 |
| --- | --- | --- | --- | --- |
| ROS 启动预览 | 无策略选择 | 无 | 暂停模型预览，不读取任何 ONNX | 不执行主动平衡或行走 |
| `mc_rl_velocity` | `config/mc_rl_velocity.yaml` | 空字符串 | AimDK `extra/mc-rl/x2_deploy_pkg/models/policy.onnx`，用于主动平衡和速度控制 | 历史测试已验证加载、平移和停止；低抬脚步态尚未通过验收 |
| `bmimic` | `config/bmimic.yaml` | `kuailechongbai` | `kuailechongbai1-Full2.onnx`，151 维重力观测的参考动作跟踪 | 已完成 2820 帧无头动力学测试；未做视觉质量验收，不声称结束后站稳 |
| `bmimic` | `config/bmimic.yaml` | `fan_dance` | `policy_FanDance.onnx`，扇子舞 | 已完成配置的 1171 帧无头动力学测试 |
| `bmimic` | `config/bmimic.yaml` | `niuniu_dance` | `policy_niuniu_Dance.onnx`，扭扭舞 | 已完成模型全部 1160 帧无头动力学测试 |
| `bmimic` | `config/bmimic.yaml` | `charleston` | `policy_dance_ces_full300.onnx`，查尔斯顿舞 | 已完成配置的 2518 帧无头动力学测试 |
| `bmimic` | `config/bmimic.yaml` | `dingding` | `dingding_v5_gc.onnx`，Dingding 动作 | 已完成配置的 1551 帧无头动力学测试 |
| `bmimic` | `config/bmimic.yaml` | `kuailechongbai_video` | `kuailechongbai_fromvideo_inpainted.onnx`，视频版快乐崇拜 | 已完成配置的 2820 帧无头动力学测试 |

目前支持上述两种控制器，不是任意 ONNX 选择器。
速度控制权重的 SHA256 为 `5ca4ebc3ae12e902cb4346ecc4b8532e3a2404a41f0a56e631a2d0316e297a71`；
请自行合法获取，并手动放到 `<assets>/policy.onnx`。本项目不提供或自动复制该权重。
其他未验证的策略不能通过改名为 `policy.onnx` 使用。

`mc_rl_velocity` YAML 必须且只能包含这两个字段。
`policy` 是相对资产目录的权重文件路径，也可使用 `rl_models/mc_rl_velocity.onnx` 这样的嵌套路径，
不是相对 YAML 所在目录。禁止绝对路径、`..` 路径穿越，以及通过软链接逃出资产目录。
请将已获授权的权重放在配置的相对路径下。
速度适配器固定关节顺序、观测历史、增益、默认姿态和权重 SHA256 等契约；
速度适配器每次选择加载时单独校验哈希，支持已知匹配的 500 维输入、29 维输出权重。
这不是通用厂商配置加载器，不支持任意 ONNX；
仅修改文件名或 YAML 不会增加模型支持。

### 快乐崇拜和 BMIMIC

将已获授权的权重手动放到 `<assets>/kuailechongbai1-Full2.onnx`，使用
[config/bmimic.yaml](config/bmimic.yaml) 中的 `kuailechongbai` 动作。
执行帧为 1 至 2820（包含两端），50 Hz，共 56.4 秒。不自动复制 SDK 权重。
BMIMIC 配置不含哈希，运行时不校验 BMIMIC 权重哈希；速度控制器的已知权重哈希门禁不变。

BMIMIC 不接受非零 `/simulation/cmd_vel`。完整执行后暂停，不自动循环或切换策略；
回放通过新的重置 action 请求，不需要重启。终点不是已验证的静止姿态，
不能把暂停后模型不倒误认为主动站稳。暂停、取消和恢复规则见下文。

六个动作合并在同一个 BMIMIC 配置中，顶层必须包含 `controller: bmimic` 和 `motions` 映射。
每个动作条目必须包含 `policy`、`start_count`、`end_count`，另允许可选正整数 `frame_count`；
不接受 `sha256` 或其他字段。`policy` 同样相对资产目录，并受上述路径限制。
例如，合并配置中的快乐崇拜和扇子舞条目为：

```yaml
controller: bmimic
motions:
  kuailechongbai:
    policy: kuailechongbai1-Full2.onnx
    start_count: 1
    end_count: 2820
  fan_dance:
    policy: policy_FanDance.onnx
    start_count: 0
    end_count: 1170
    frame_count: 1787
```

BMIMIC 没有默认动作，action 的 `motion` 必须显式指定 `motions` 中的名称；空值或未知名称都会失败。
离线物理 CLI 则通过 `--controller-config <bmimic.yaml> --motion <名称>` 选择动作。
旧版导出缺少 `time_step_total` 元数据时，必须在对应动作条目中显式声明经图检查确认的参考表行数；
两者同时存在时必须相同。不会凭文件名猜长度或静默放宽区间。
同契约的新权重无需新建 Python 控制器，但必须提供有效帧区间并单独验证和验收。
元数据和接口检查不证明权重等价，也不保证任意 ONNX 可安全使用；替换权重不能沿用既有测试结论。
加载时检查 float 输入 `obs[1,151]`、`time_step[1,1]`，命名输出及模型元数据。
29 个非头部关节顺序、默认角度、PD 增益和动作缩放从模型元数据读取并检查；头部不参与策略动作。
观测为参考关节位置/速度（58）、骨盆投影重力（3）、角速度乘 0.25（3）、
相对默认姿态的关节位置（29）、速度乘 0.05（29）、上次动作（29），不堆叠五帧。
每帧先读取相同 `time_step` 的参考表，再运行 actor，避免参考缓存错后一帧。
动作限幅为 [-100,100]，目标为默认角度加模型动作缩放乘输出。
该契约不同于 Unitree 的 154 维参考锚点旋转观测；不支持按维度猜测或填零适配。

历史真实模型动力学定向测试完成了 2820 帧，最低骨盆高度约 0.489 m，最大倾斜约 34.4 度，
未触发 46 度/0.3 m 跌倒门槛。结束时仍有运动，因此只说明有限时域内未触发故障，
不是跟踪精度、视觉舞蹈质量、结束后主动平衡或真机适用性的验收。
模型中的世界坐标参考身体输出不用于重写仿真根位姿。

### 其他舞蹈配置

其余五个动作条目复用相同 BMIMIC 适配器。把有授权的对应权重手动放到
`<assets>/<表中的模型文件名>`，将 action 的 `motion` 改为所选名称即可，
例如 `fan_dance`；仍使用同一个随包安装的 `config/bmimic.yaml`，不下载或复制权重。

| `motion` 名称 | 执行区间（包含两端） | 时长（50 Hz） | 最低骨盆高度 | 最大倾斜 |
| --- | --- | --- | --- | --- |
| `fan_dance` | 0～1170 | 23.42 s | 0.453 m | 11.3 度 |
| `niuniu_dance` | 0～1159 | 23.20 s | 0.581 m | 27.1 度 |
| `charleston` | 0～2517 | 50.36 s | 0.471 m | 23.0 度 |
| `dingding` | 0～1550 | 31.02 s | 0.584 m | 9.6 度 |
| `kuailechongbai_video` | 1～2820 | 56.40 s | 0.513 m | 29.0 度 |

区间参考厂商 `motion_list.yaml`，但该目录本身全部注释，不代表厂商已启用动作。
扭扭舞目录终点 1170 超出实际参考表的 1160 行，本项目显式改为 1159，
不依赖 ONNX 内部限幅反复播放尾帧。扇子舞有 1787 行、查尔斯顿有 6572 行，
这里仅支持目录指定的片段，没有宣称验证剩余帧；这三份旧模型用各自动作条目的 `frame_count` 记录完整参考表长度。
已检查五份模型参考输出不依赖 `obs`，可按当前帧先读取参考再执行 actor。

表中数据来自本地真实权重和原有物理模型的历史测试，不修改根位姿或放宽跌倒门槛。
这些数据不是新 action 生命周期的验收结果。
这五个策略都未做 GUI 视觉质量验收，不验证跟踪精度或真机适用性；执行结束暂停也不代表主动平衡。
视频版快乐崇拜的最终平面位移约 2.42 m，需要注意观察范围，不能把它等同于原地舞蹈。

## ROS 启动和策略请求

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && python -m aimdk_mujoco.node --assets /absolute/path/to/x2-assets --headless
```

或者只启动一次 launch，然后在其他终端提交策略请求：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 launch aimdk_mujoco simulation.launch.py assets:=/absolute/path/to/x2-assets headless:=true
```

节点 CLI 仅使用必填的 `--assets`、可选的 `--headless` 和 `--config-dir`；
launch 参数仅有 `assets`、`headless`、`config_dir`，没有 `controller_config` 或 `motion`。
配置目录默认是安装后的 `share/aimdk_mujoco/config/`，可用 `--config-dir <目录>` 或
`config_dir:=<目录>` 覆盖。省略 `--headless`，或设置 `headless:=false`，
可打开 MuJoCo 被动查看器，需要可用的图形环境；无头模式不导入查看器。
即使权重已存在，节点也始终从暂停预览开始，不读取 ONNX，不自动运行策略。

策略请求接口为 `/simulation/execute_policy`，类型为 `aimdk_interfaces/action/ExecuteSimulationPolicy`。这是 X2 仿真专用接口，不属于公共 Robot Runtime 契约：

| Goal 字段 | 约束 |
| --- | --- |
| `controller` | `bmimic` 或 `mc_rl_velocity`，从配置目录选择相应 YAML |
| `motion` | BMIMIC 的目录条目名称；速度控制器必须为空字符串 |
| `reset_before_start` | 新选择或回放为 `true`；同一未完成选择续跑为 `false` |
| `transition_from_current` | 无重置交接为 `true`；必须同时 `reset_before_start: false`，会保留当前 `MjData` |
| `duration_s` | BMIMIC 必须为 `0`，执行配置帧区间；速度控制器必须为有限值，`0 < duration_s <= 3600`，限定运行时长 |

另有内部交接控制器 `recover_stand`：它复用速度策略，在当前状态不瞬移的前提下，
尝试连续稳定 1 秒后结束并暂停。请求必须使用
`reset_before_start: false`、`transition_from_current: true`、`1 <= duration_s <= 30`；
它不是从摔倒状态起身，低于 0.5 m、倾斜超过 25 度、速度过大或已锁存故障都会失败。
交接到 BMIMIC 还要求站稳、关节目标差小于 0.5 rad，且首个目标的力矩保护检查通过。

例如先执行快乐崇拜，等待完成后再提交 10 秒速度控制请求，无需重新 launch：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 action send_goal --feedback /simulation/execute_policy aimdk_interfaces/action/ExecuteSimulationPolicy '{controller: bmimic, motion: kuailechongbai, reset_before_start: true, duration_s: 0.0}'
```

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 action send_goal --feedback /simulation/execute_policy aimdk_interfaces/action/ExecuteSimulationPolicy '{controller: mc_rl_velocity, motion: "", reset_before_start: true, duration_s: 10.0}'
```

速度 goal 本身不指定行走速度；运行期间通过 `/simulation/cmd_vel` 输入，零速度时策略仍执行主动平衡。
反馈字段为 `state`（`preparing`、`running` 或 `paused`）、`completed_steps`、`total_steps`；
结果字段为 `success`、`message`、`completed_steps`。完成后冻结物理，不自动循环。

### 暂停、取消和重置

- 两种控制器的 `/simulation/stop` 和 `/simulation/stand`（`std_srvs/srv/Trigger`）都暂停物理和时钟，goal 保持 active；不是站立控制，也不证明站稳。
- 加载期间收到 stop/stand 时，加载完成后仍保持暂停，须显式 start 才能推进物理。后台加载不会阻塞窗口与取消；取消后须等该次加载结束才能接受新 goal，过期结果不会激活。
- `/simulation/start` 只恢复当前 active 且暂停的 goal；没有这样的 goal 时不能启动新动作，也不能恢复已经取消的 goal。
- Action cancel 冻结物理并结束当前 goal。需使用支持取消的 action 客户端，例如通过 `rclpy` goal handle 的 `cancel_goal_async()`；stop 不等于 cancel，不要用关闭 CLI 终端代替取消确认。
- 当前 goal 只要仍 active（包括暂停），新 goal 都会被拒绝。中途切换必须先取消；已完成的选择也只能以 `reset_before_start: true` 回放。
- 取消后只有同一未完成的控制器和动作可用 `reset_before_start: false` 提交新 goal 续跑；新选择或切换默认重置。
- 需要保留当前物理姿态进行交接时，先取消当前 goal，再用 `reset_before_start: false` 和
  `transition_from_current: true` 请求 `recover_stand` 或下一个已验证的控制器。该路径只允许
  通过有限状态和目标力矩保护，不保证任意动作都能安全衔接。
- 每次选择都重新加载本地 ONNX，不缓存会话，不下载或复制权重。先预加载并校验成功，再执行所需重置；无效配置、动作或权重使该请求 abort，不扰动现有物理状态，也不锁存运行故障。

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 service call /simulation/stop std_srvs/srv/Trigger '{}'
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 service call /simulation/start std_srvs/srv/Trigger '{}'
```

初始化时设置直立姿态并以固定姿态 PD 稳定 0.5 秒，保存稳定后的初始状态；
这不是从地面躺姿起身，也不是主动平衡验收。重置复用同一模型、`MjData` 和窗口，
恢复保存的稳定初态，但保持仿真时间单调，不是物理平滑过渡，不能将瞬间复位视为策略动作。
运行时按单调墙钟以 50 Hz 调度，每次用 20 个 PD 子步推进 0.02 秒；较慢时不突发补算。
除显式重置外，动作由力矩控制推进，不通过重写根位姿移动。
所有仿真输入和输出话题默认位于 `/simulation/` 下，不订阅真实机器人的 `/cmd_vel`，
也不发布全局 `/joint_states`、`/imu/data`、`/diagnostics` 或 `/clock`。
因此，仅设置 `use_sim_time:=true` 不会让其他节点使用本仿真的时钟。
需要仿真时间的消费节点应设置 `use_sim_time:=true`，并将其时钟订阅从 `/clock` 重映射到
`/simulation/clock`，即添加 `--ros-args -p use_sim_time:=true -r /clock:=/simulation/clock`。

如果确实要把本仿真作为当前 ROS 域的全局时钟源，可在独立节点启动时显式将
`/simulation/clock` 重映射到 `/clock`：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && python -m aimdk_mujoco.node --assets /absolute/path/to/x2-assets --headless --ros-args -r /simulation/clock:=/clock
```

此时消费节点只需设置 `use_sim_time:=true`。启用前应确认同一域中没有其他 `/clock` 发布者，
避免接管其他仿真或真实机器人节点的时间。验证工具使用默认 `/simulation/clock`，不使用此重映射。

### 速度输入

`/simulation/cmd_vel` 使用 `geometry_msgs/Twist`，只接受平面速度：

| 分量 | 范围 |
| --- | --- |
| `linear.x` | [-0.3, 0.5] m/s |
| `linear.y` | [-0.3, 0.3] m/s |
| `angular.z` | [-0.2, 0.2] rad/s |

任何分量非有限、平面指令越界，或其他三个分量非零，都拒绝整条消息并清零当前指令，
不做静默限幅。距离最后一条有效消息达到 0.5 秒墙钟时间后，自动请求零速度；
仅在运行中的速度 goal 内执行零速度主动平衡；输入或看门狗都不会恢复暂停或启动新 goal。
超时不依赖 `/simulation/clock`，不会因仿真时间停止而失效。

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && ros2 topic pub --rate 10 /simulation/cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.2}}'
```

## 输出和故障

| 接口 | 含义 |
| --- | --- |
| `/simulation/ground_truth/odom` | `nav_msgs/Odometry`，`world` 到 `pelvis` 的物理真值，线速度旋转至子坐标系 `pelvis`，角速度也在该坐标系 |
| `/simulation/joint_states` | 策略观测顺序的关节名称、位置和速度，不伪造力矩 |
| `/simulation/imu/data` | `pelvis` 姿态和机体系角速度；无加速度测量，`linear_acceleration_covariance[0] = -1` |
| `/simulation/clock` | MuJoCo 仿真时间；预览、暂停、取消和完成后不推进，重置不倒退 |
| `/simulation/diagnostics` | 仿真状态；物理故障时为 `ERROR`，不是公共运行时能力声明 |

不发布 `/odom`，不启动厂商栈，也不发布 TF。使用输出时请明确区分物理真值与传感器估计。
实际物理或推理运行故障（如跌倒、非有限数值或推理失败）永久锁存，指令清零，停止物理推进和
状态、时钟发布。节点保留服务和周期诊断；start/stop/stand 均不能解除故障，
新 goal 或重置请求也不能恢复。排查后需显式重启进程。
这与新请求预加载/校验失败不同，后者只 abort 该请求，不锁存现有物理核心。

## 验证

所有策略验证都必须事先在 YAML 指定的资产相对路径放置已获授权的匹配权重。
模型预览不能作为物理或行走验收。不启动 ROS 图的离线物理 CLI 保持不变：
使用 `--controller-config <yaml>`、BMIMIC 的 `--motion <名称>` 和原有 `--verify` 入口。

```bash
source .shrc_local && python -m aimdk_mujoco.physics --assets /absolute/path/to/x2-assets --controller-config src/robots/aimdk/aimdk_mujoco/config/mc_rl_velocity.yaml --verify --report /tmp/opencode/x2-acceptance.json
```

`verify_ros` 先启动通用预览节点，再提交 `mc_rl_velocity` action，
通过 `/simulation/cmd_vel` 测试零速平衡、前进和零速停止，随后取消 goal 并确认常驻节点仍存活，记录真值并退出：

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && python -m aimdk_mujoco.verify_ros --assets /absolute/path/to/x2-assets --report /tmp/opencode/x2-ros-acceptance.json --check-pause-resume
```

它将资产和可选配置目录传给通用节点，默认使用安装的速度配置，不再接受或传递
`--controller-config`、`--motion`。策略仍只从 YAML 指定的资产相对路径加载。
加 `--gui` 可同时观察窗口；应确保当前域没有其他同名仿真接口。
`--check-pause-resume` 额外检查暂停一秒墙钟期间仿真时间固定、遥测持续，以及 start 后时间继续推进。
它只报告 `translation_passed`，不验脚部步态，不能独立作为行走验收。
无重置交接的真实权重验收使用独立入口：

```bash
source .shrc_local && python -m aimdk_mujoco.verify_handoff --assets /absolute/path/to/x2-assets --report /tmp/opencode/x2-handoff-acceptance.json
```

该入口只启动并清理自己创建的仿真进程，执行一次快乐崇拜、取消、回稳、速度控制、
再次回稳和扇子舞；只有首次策略选择重置，后续交接保留仿真状态。
单元测试无需 X2 资产：节点测试使用隔离 ROS 域和假物理核心；物理回归使用小型生成模型，
覆盖历史帧排列、边界速度、推理失败、数值失稳及速度参考点。
正式包验收使用 `colcon test --merge-install --packages-select aimdk_mujoco`。

### 有界无重置交接验收

独立入口 `verify_handoff` 使用真实本地权重和现有 Action API，不改变 `verify_ros` 的速度验证流程：

```bash
source .shrc_local && ROS_DOMAIN_ID=89 python -m aimdk_mujoco.verify_handoff --assets /absolute/path/to/x2-assets --report /tmp/opencode/x2-handoff.json
```

它自行通过 launch 启动独立仿真，继承调用者的 `ROS_DOMAIN_ID`；未设置时随机选择本地域。
域中已有仿真接口时拒绝执行，不接管或清理其他进程。退出时先取消未完成 goal，再向自己启动的
launch 发送 SIGINT，等待优雅退出；超时仅升级终止自己拥有的进程组。
可加 `--config-dir <目录>` 或 `--gui`，使用前准备好快乐崇拜、速度控制、扇子舞三份授权权重，
路径仍来自支持的控制器配置，不下载或复制权重。

固定序列为：快乐崇拜初次以 `reset_before_start=true` 启动，收到运行反馈的
`completed_steps >= 500`（50 Hz 下至少 10 秒仿真时间）后取消；随后依次提交
`recover_stand(duration_s=10)`、`mc_rl_velocity(duration_s=5)`、
`recover_stand(duration_s=10)`、`bmimic/fan_dance` 完整区间。
后四个 goal 均设置 `reset_before_start=false`、`transition_from_current=true`，不复位物理状态；
恢复站立达到连续一秒稳定条件可提前成功，10 秒是恢复时限，不是强制等待。
速度阶段只有收到 `running` 反馈后才以 10 Hz 持续发布 `linear.x=0.4`，避免准备期间丢弃命令。

JSON 报告记录各阶段反馈、终态、遥测计数/仿真时长及通过/失败，旁边的 `.log` 保存 launch 输出。
诊断故障、超时、提前或错误终态、遥测中断/非有限值、时钟倒退/异常跳变、明显位置瞬移、
步数与仿真时间不符，以及终态后节点不存活均失败；每个 goal 执行墙钟上限为 90 秒。
时钟按每个话题独立检查以容忍跨话题投递顺序；暂停/加载期间允许同一时间戳重复。
这是实验性有限时域交接验收，不证明任意状态可交接、细微关节连续性、舞蹈视觉质量或正常步态。

常驻 action 版本已在本地真实权重上验证快乐崇拜完整 2820 帧 GUI 执行、取消与续跑、
舞蹈/速度控制器切换、并发拒绝、缺失权重请求失败后可继续使用，以及全程时钟不倒退。
速度 ROS 验证通过暂停/恢复、10 秒前进约 1.705 m、零速停止和取消后节点存活检查。
这些结果不包含其余五份舞蹈的常驻 action 或 GUI 验收，也不改变脚部步态尚未通过的结论。

BMIMIC 无头有限时域验证使用同一个 `physics --verify` 入口；它显式开始动作，
检查整个配置区间的动力学故障，记录最终状态，完成后暂停，不执行速度控制验收：

```bash
source .shrc_local && python -m aimdk_mujoco.physics --assets /absolute/path/to/x2-assets --controller-config src/robots/aimdk/aimdk_mujoco/config/bmimic.yaml --motion kuailechongbai --verify --report /tmp/opencode/x2-bmimic-acceptance.json
```

`verify_ros` 仍仅用于速度控制器的平移/暂停测试，不能作为舞蹈验收入口。

## 步态调查

2026-10-26 对本地同一策略、模型、参数实测：`vx=0.2`、前进 10 秒，身体前移约 1.70 米。
去掉前 2 秒过渡后，左右脚分别检测到 16 / 15 次持续离地小摆动，
脚底碰撞球最低点最大离地仅约 9.6 / 6.3 毫米；单脚支撑占约 41%。
这不是代码重写根位姿产生的平移，也不是完全不迈步，但存在低抬脚和落脚滑移，
视觉上接近拖步。`vx=0.1` 时双脚持续接地，不能认为正常行走。

观测排列、关节映射、增益和权重与 SDK 参考一致；按参考取消额外状态刷新，
以及临时对比更高摩擦、无滑移求解和训练 armature，均未消除低抬脚现象。
这些实验没有改进已安装的模型或控制参数。暂不能把问题归结为某一个接线错误，
也没有通过动画抬脚或修改根位姿掩盖它；更换或重新训练策略仍需后续验证。

`physics --verify` 现在额外检查两脚的持续摆动、单脚支撑、脚底离地和承重接触点速度。
项目当前验收门槛为两脚各至少 3 次持续摆动、最大脚底离地至少 15 毫米，
单脚支撑至少 10%，承重滑移中位数不超过 0.02 m/s、95 分位不超过 0.1 m/s。
这些是项目的保守验收条件，不是对所有低速步态的通用定义；接触瞬间速度须结合原始轨迹判断。
当前权重会明确报告 `passed: false`，不再把稳定平移报告成正常行走。
报告路径示例：`/tmp/opencode/x2-gait-acceptance.json`。
