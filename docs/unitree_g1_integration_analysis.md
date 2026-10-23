# Unitree G1 接入可行性分析

> 分析阶段文档，**不含任何实现**。
> 基线：**upstream master `ca7b8c23`**（2026-10-12 重新对齐）。PR #422（运行时契约本体）与
> PR #426（AgiBot X2 bridge）都已合入 master，合入版比 2026-09-18 分析时的 PR head 多出约 23k 行，
> 本文已按合入版重新核对（变化汇总见 §2.3）。旧基线 `6835cc1c` 保留在 tag `backup/unitree-g1-pre-master`。
> 厂商材料下载于 2026-09-17，存放在本 worktree 的 `unitree/`（已 gitignore）。

## 0. 已定决策（用户；Q1–Q4 2026-09-18，Q5–Q6 2026-10-12）

| # | 问题 | 决策 | 后果 |
|---|---|---|---|
| Q1 | G1 上是否跑策略推理 | **跑** | 必须发布公共 model → 需要 vendoring `g1_description` URDF，且需要一处 `robot_runtime` 契约改动放开 `authority`（§6.1） |
| Q2 | 与 G1 的通信通道 | **纯 ROS 2 消息** | vendoring `unitree_ros2` 的 `unitree_hg`/`unitree_go`/`unitree_api`；不链接 `libunitree_sdk2.a`；RMW 固定 `rmw_cyclonedds_cpp` |
| Q3 | 基线 | **PR #426** → 现为 **upstream master**（#426 已合入） | 直接继承 registry v3、6 个 `ibrobot_msgs` 定义、`build.sh` 分组、独立性门禁多机器人化；#422/#426 均已合入，G1 是**直接对 master 的单个 PR**，不再有 PR 栈 |
| Q4 | 目标构型 | **29 DoF + Dex3 灵巧手** | URDF 取 `g1_29dof_with_hand_rev_1_0.urdf`（43 个 actuated 关节 = 29 身体 + 7×2 手）；末端走 `hand.multi_joint` 而非 `gripper.1d` |

| Q5 | Dex3 手指是否进策略动作空间 | ~~方案 B~~ **已被 Q9 取代** | 手指不进 `gripper` 组，作为 `hand.multi_joint` 的逐手通道 |
| Q6 | 腰是否进 `arm` 组 | **不进**（2026-10-12） | `arm` = 14 个臂关节；腰走独立的 `waist_stream` 通道 |

| Q7 | `joint_groups` 划分规则 | ~~放宽为"互斥子集"~~ **已被 Q9 取代** | 不发布 model 即不触发划分规则 |
| Q8 | 归一化表顺序 | ~~保持机器人关节顺序~~ **已被 Q9 取代** | |
| Q9 | 多动作空间与策略推理的落地路线 | **路线 1**（2026-10-12）：并入进行中的 OpenSpec change `refactor-embodiment-control-bindings` | G1 首个 PR 不发布公共 model、不改契约层；Q5/Q7/Q8 的临时方案作废，换算权威与按特征绑定由该 change 决定。OpenSpec：`unitree-g1-runtime-migration`（新建）+ 该 change 的 design D9 / tasks 5.6–5.7 |

## 1. 结论

**可以接入，且是目前为止最容易接入的一款厂商机器人**——前提是走 PR #426 建立的
「vendor bridge」路线，而不是 PR #422 的 ros2_control 路线。

三条核心判断：

| 判断 | 依据 |
|---|---|
| **架构完全匹配** | G1 与 AgiBot X2 是同构问题：厂商自带实时平衡/步态控制器，上层只能在"受保护层"下发指令。PR #426 的 D1（bridge 而非 ros2_control hardware interface）、D3（runtime mode ↔ 厂商 FSM）、D4（TORQUE_OFF 在双足上是摔倒）、D7（厂商已发布的传感器只声明不重驱）逐条都成立 |
| **许可证比 X2 干净** | `unitree_sdk2`、`unitree_ros`（含 `g1_description`）、`unitree_ros2` 全部 **BSD-3-Clause**。X2 被 D8 卡住的"`aimdk_msgs` 无 license 不能 vendoring"问题在 G1 上**不存在** |
| **不需要链接厂商二进制** | `unitree_ros2` 提供 BSD-3 的纯 ROS 2 消息包 `unitree_hg` / `unitree_go` / `unitree_api`，G1 的全部接口（含 RPC 服务）都能用纯 Python ROS 2 节点访问，不必链接 `libunitree_sdk2.a` |

前置决策已在 §0 定案。**§6.1、§6.2、§6.4 中关于 model / 分组 / 归一化的方案已被 Q9 取代，保留为分析记录**；
以 OpenSpec `unitree-g1-runtime-migration/design.md` 为准。

## 2. 两个参考 PR 做了什么

### PR #422 `refactor/robot-so101` — 运行时契约本体

4 个 commit，356 文件，+26206/-5361。引入了：

- **`src/robot_runtime/`**：运行时契约层。`contract.py`（中立 topic/service 名）、
  `capabilities.py`（冻结的能力词表，extension-only）、`profile.py`（运行时参数 SSOT）、
  `modes.py`、`facade_node.py`（ros2_control 实现）、`mock_runtime_node.py`（无硬件实现）、
  `interface_description.py` + JSON schema（版本化的公共接口描述）、
  `test/test_conformance.py`（**可执行的契约**，按 capability 裁剪用例）
- **`src/robots/<vendor>/<robot>_*` 目录结构**：so101 拆成
  `so101_robot` / `so101_hardware` / `so101_description` / `so101_motion` / `so101_sdk` / `so101_suite`，
  feetech 驱动独立为 `src/robots/feetech/feetech_sdk`
- 所有通用消费者（`action_dispatch`、`robot_teleop`、`task_dispatch`、`manipulation_execution`、
  `skill_library`、`dataset_tools`）改为只依赖契约名，不再依赖具体机器人

关键点：`mock_runtime_node` 与 `facade_node` 通过**同一套 conformance 套件**，
证明了契约不预设 ros2_control——这正是 G1 能接入的前提。

### PR #426 `refactor/robot-aimdk` — 厂商 bridge 参考实现（已合入）

下表是 2026-09-18 按 PR head 统计的构成（35 个文件、+5708 行）；合入版的 `aimdk_robot`
本体已增长到约 9200 行，见下方骨架与 §2.3。

| 类别 | 文件 | 行数 | G1 能否复用 |
|---|---|---|---|
| 能力词表 v2→v3 | `robot_runtime/capabilities.py` + test | ~110 | ✅ 直接复用（G1 不需要再加名字，见 §4） |
| 中立消息类型 | `ibrobot_msgs`：`ExecuteNamedMotion.action`、`SpeakText.srv`、`PlayExpression.srv`、`SetLedPattern.srv`、`StartRelocalization.srv`、`PowerState.msg` | ~178 | ✅ 直接复用 |
| 构建分组 / 门禁 | `scripts/build.sh`（group 选择）、`scripts/verify_runtime_independence.sh` | ~167 | ⚠️ 需加 G1 条目（各约 20 行） |
| **机器人包本体** | `src/robots/aimdk/aimdk_robot/**` | **~4180** | ❌ 需要为 G1 重写 |
| 部署配置 | `robot_config/config/robots/aimdk_x2.yaml` + test | ~196 | ❌ 需要为 G1 重写 |
| conformance 扩展 | `robot_runtime/test/test_conformance.py` | ~39 | ✅ 已覆盖 humanoid 形状 |

`aimdk_robot` 包内部结构（这就是 G1 包要照抄的骨架；行数为 master `ca7b8c23`，括号内为 PR head 时）：

```
aimdk_robot/
├── profiles/x2_ultra.yaml            799 行 (618)   ← 厂商事实的唯一落点
├── profiles/x2_ultra_omnihand.yaml   936 行 (778)   ← 末端变体 = profile 变体
├── aimdk_robot/projection.py         823 行 (436)   ← 纯函数：关节投影/模式映射/Twist 换算/停止规划
├── aimdk_robot/runtime_node.py      1777 行 (1036)  ← 契约服务端
├── aimdk_robot/vendor_gateway.py     571 行 (322)   ← 厂商 service/topic 封装 + 仲裁生命周期
├── aimdk_robot/vendor_mock_node.py   667 行 (318)   ← 说厂商 IDL 的 mock，用于无硬件验证
├── aimdk_robot/named_motion.py       359 行 (149)
├── aimdk_robot/localization.py        78 行
├── launch/runtime.launch.py          112 行         ← 错峰启动（厂商限制 ≤2 节点/秒）
├── README.md                         644 行         ← 含首次真机接触记录与上机步骤
└── test/  3 个文件                   2420 行 (928)
```

增长主要来自评审与 2026-09-22 首次真机接触：输出路径上的准入（stop 先闭锁再请平台 hold、
导航门撤销/仲裁丢失时下一 tick 丢弃缓存命令）、按 profile 声明的**厂商逐端点 QoS**、
命名动作以平台任务状态判完成且拒绝 cancel、各上身通道独立过期、静默反馈源退出聚合关节态。
这些都是 G1 bridge 同样要做的事，**工作量估算应以合入版为准**（§7）。

设计决策记录在 `openspec/changes/aimdk-runtime-migration/design.md`（D1–D14，557 行）。

### 2.3 合入后新增、对 G1 有直接影响的约定（2026-10-12 核对）

| # | 来源 | 约定 | 对 G1 的影响 |
|---|---|---|---|
| 1 | `aimdk_x2.yaml` | `robot.type` 必须等于所选 runtime profile 的公共机器人标识（X2 为 `x2_ultra`），厂商家族写在 `robot_type` | G1 部署配置写 `type: g1_29dof_dex3`（与 profile 名一致）、`robot_type: unitree_g1` |
| 2 | `interface_description.py` | `teleoperation.command_stale_s` 必须在 `(0, 1]` | G1 profile 若声明遥操作需遵守 |
| 3 | `capabilities.py` | `interaction.expression` / `interaction.led` 参数新增 `priority_levels` | G1 的 `LedControl` 需声明优先级层（即使只有一层） |
| 4 | aimdk README「Vendor QoS」 | 厂商端点 QoS 是 profile 数据，best-effort 对 reliable 订阅会静默失败 | Unitree `rt/*` topic 的 QoS 必须按真机写进 profile（§6.3 新增一项） |
| 5 | aimdk README「Declared sensors」 | 只声明**真机实际在发**的端点（X2 因固件未发而撤掉了 `perception.lidar`、GNSS、relocalization） | G1 的声明集合同样要以真机 `ros2 topic info -v` 为准 |
| 6 | `e01b1645` / `d27a4ebc` | `skill_library` 新增 `runtime_named_motion` 执行器 + `aimdk_x2_skills` 部署（无手臂面、只暴露命名动作） | G1 的 `ArmActionClient` 预置动作（挥手、握手、拍手、飞吻…）走 `motion.named` 后，**技能侧零代码**即可复用，只需一个 `unitree_g1_skills` 部署 + catalog profile |
| 7 | `robot_runtime/peripherals.py`、D12 | 部署的 `peripherals:` 与 `navigation.fast_lio` 以 fragment 形式交给 runtime，由 runtime 组合驱动；共享辅助已支持 **RealSense** 与 **Livox MID-360**，并有 `fast_lio_nodes`；独立性门禁的 `THIRD_PARTY_ALLOWED` 已含 `livox_ros_driver2 fast_lio fast_calib` | G1 的 D435 / MID-360 / FAST-LIO **不需要写任何驱动代码**，G1 的 `runtime.launch.py` 照 LeKiwi 的方式调用 `peripheral_nodes` + `fast_lio_nodes` 即可（修正了原 D7 的判断） |
| 8 | `ca7b8c23` | 有 runtime provider 的导航/混合阶段，在 Collision Monitor 改接到 runtime 的 `/cmd_vel` 之前，**启动时直接失败** | G1 首个 PR 不做 Nav2 导航阶段，只做 `base.cmd_vel` 遥控/策略；与 LeKiwi 现状一致 |
| 9 | X2 `design.md` D5 | X2 设计里的 D5 **选项 2** 就是"profile 声明 `calibration: vendor_internal` 时 `authority` 接受 `urdf`"，当时为 X2 选了选项 1 | §6.1 的推荐方案**不是新发明**，而是已评审过、留给"有可分发 URDF 的厂商机器人"的预案；G1 正好满足其前提 |

契约层本身（`model_metadata.py` 的 authority/joint_groups 规则、registry v3 词表名）**没有变化**，
§4、§6.1、§6.2 的结论全部仍然成立。

## 3. Unitree G1 侧的事实

均来自实际下载的代码，非推测。

### 3.1 控制分层（与 X2 同构）

| 层 | 通道 | 语义 | 对应 X2 |
|---|---|---|---|
| **高层 RPC** | `LocoClient`（service name `sport`，api_id 7001–7111） | `SetVelocity(vx,vy,ω,duration)`、`SetFsmId`、`SetBalanceMode`、`SetStandHeight`、`SetSwingHeight`、`SwitchToUserCtrl` / `SwitchToInternalCtrl` | `SetMcAction` + `/aima/mc/locomotion/velocity` |
| **手臂流式（受保护）** | DDS topic `rt/arm_sdk`，类型 `unitree_hg/LowCmd` | 平衡控制器仍在跑；`motor_cmd[29].q` 是 **0→1 的权重系数**，用于把手臂控制权从内置控制器渐变移交给用户；**无需 CRC** | `/mc/upper_body_command`（MC 层） |
| **低层全身（危险）** | DDS topic `rt/lowcmd`，`unitree_hg/LowCmd` + **CRC32 必填** | 接管全部 29 个电机，内置平衡控制器让出；等价于 X2 的 HAL 层 | `/aima/hal/joint/*/command`（`expert_joint_tier`，默认关） |
| **状态** | `rt/lowstate`（`unitree_hg/LowState`：`imu_state` + `motor_state[35]` + `mode_machine` + `wireless_remote`） | 全身关节 + IMU + 手柄，一条 topic | `/aima/hal/joint/{arm,head,waist,leg}/state` 四条 |
| **末端** | `rt/dex3/{left,right}/{cmd,state}`（Dex3 灵巧手，7 DoF×2）、`rt/inspire/{cmd,state}`（因时手）、Dex1 夹爪 | 末端类型是硬件变体 | OmniPicker / OmniHand |
| **交互** | `AudioClient`：`TtsMaker`、ASR、`PlayStream`/`PlayStop`、`GetVolume`/`SetVolume`、`LedControl(R,G,B)` | 平台自带语音与灯带 | `PlayTts` / `SetPmuLed` |
| **预置动作** | `ArmActionClient`（service `arm`，api_id 7106–7113）：`ExecuteAction` / `GetActionList` / 自定义动作 | 挥手等预置手臂动作，**动作表可运行时查询** | `SetMcPresetMotion` |
| **AGV / 升降** | `g1/agv/g1_agv_client.hpp`（G1-D 轮式底盘变体） | 变体 | — |

`rt/arm_sdk` 无需 CRC 这一点很关键：**默认控制路径可以完全用 Python 实现**，
只有"专家层"`rt/lowcmd` 才需要复刻 C++ 结构体内存布局来算 CRC32。

### 3.2 关节清单

`include/unitree/dds_wrapper/robots/g1/defines.h` 中 `JointIndex`（29 DoF）：

```
0-5    左腿 hip_pitch/roll/yaw, knee, ankle_pitch/roll
6-11   右腿 同上
12-14  waist_yaw / waist_roll / waist_pitch
15-21  左臂 shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw
22-28  右臂 同上
29     arm_sdk 权重位（非关节）
```

`ArmJoints` 常量 = 14 臂 + 3 腰 = **17 个可通过 `rt/arm_sdk` 控制的关节**。

URDF 侧（`g1_29dof_rev_1_0.urdf`）：29 个 revolute 关节，命名与 SDK 枚举**逐一对应**
（`left_shoulder_pitch_joint` ↔ `LeftShoulderPitch`），且带完整
`lower/upper/effort/velocity`（例：`left_wrist_pitch_joint` ±1.6144 rad, effort 5, vel 22）。
URDF 还内置 `d435_joint`（RealSense D435）与 `mid360_joint`（Livox MID360）——
本仓库已有 `src/livox_ros_driver2` + `src/fast_lio` + `src/fast_calib` 子模块，可直接对接。

变体矩阵由 `mode_machine` 标识（README 表格）：23dof/29dof、锁腰/不锁腰、
4010/5010 腕电机、带手/不带手、`g1_dual_arm`（仅双臂，9）。
**`mode_machine` 在 `LowState` 里可读**，且 `LowCmd` 的 `check_mode_machine()` 会校验——
这比 X2 的 `GetHandType` 更彻底：整机构型运行时可验证。

### 3.3 许可证与分发形态

| 仓库 | License | 形态 |
|---|---|---|
| `unitree_sdk2` | BSD-3-Clause | **预编译**：`lib/{x86_64,aarch64}/libunitree_sdk2.a`（55M）+ 预编译 CycloneDDS `.so`（25M）。头文件与 IDL 是源码，核心库无源码。Ubuntu 20.04 / gcc 9.4 构建 |
| `unitree_ros`（`g1_description`） | BSD-3-Clause | 纯 URDF/MJCF + meshes（该目录约数十 MB） |
| `unitree_ros2` | BSD-3-Clause | **纯消息包源码**：`unitree_hg`、`unitree_go`、`unitree_api`（`Request{header{identity{id,api_id}}, parameter, binary}` / `Response`）+ CycloneDDS 配置 |

`unitree_api/msg/Request` 的结构与 `LocoClient::Call(api_id, json_parameter, data)`
一一对应，意味着 **LocoClient / ArmActionClient / AudioClient 的全部调用都可以在
ROS 2 Python 节点里用发 `Request` + 收 `Response` 复现**，不必链接 `.a`。

## 4. 能力词表映射（G1 → registry v3）

PR #426 把词表从 v2 提到 v3 就是为了容纳"厂商自带更多能力"的机器人。
G1 的能力**全部落在现有名字里，不需要新增任何 capability**：

| G1 能力 | 机制 | capability（v3 已有） |
|---|---|---|
| 全身关节状态 | `rt/lowstate` → `/joint_states` + `/unitree/body_joint_states` | `joint.state` |
| 手臂/腰流式 | `rt/arm_sdk`（17 关节 + weight 斜坡） | `joint.position_stream` |
| 行走 | `LocoClient::SetVelocity` ← `/cmd_vel` | `base.cmd_vel` |
| FSM 姿态（坐/蹲/起身/damping） | `SetFsmId` / `SwitchToInternalCtrl` | `motion.posture` |
| 预置手臂动作 | `ArmActionClient::ExecuteAction`（`GetActionList` 可查表） | `motion.named` |
| Dex1 夹爪 | `rt/dex1/*` | `gripper.1d` |
| Dex3 灵巧手 7DoF×2 | `rt/dex3/{left,right}/cmd` | `hand.multi_joint`（v3 已声明参数 schema） |
| TTS | `AudioClient::TtsMaker` | `interaction.tts` |
| ASR | `AudioClient` ASR api | `interaction.audio_capture` |
| 扬声器播放 | `PlayStream` / `PlayStop` | `interaction.audio_playback` |
| 灯带 | `LedControl(R,G,B)` | `interaction.led` |
| IMU | `LowState.imu_state` | `perception.imu` |
| D435 相机 | RealSense 驱动（**非** SDK 自带，需自行驱动） | `perception.camera` |
| MID360 LiDAR | `livox_ros_driver2`（本仓库已有） | `perception.lidar` |
| 电池 | `unitree_hg/BmsState`、`MainBoardState` | `power.state` |
| 急停/阻尼 | `SetFsmId(damping)` / `SwitchToInternalCtrl(PASSIVE)` | `runtime.stop` |

**唯一可能缺的**：`interaction.expression`（G1 无表情屏）——不声明即可，无需改词表。

`localization.pose` / `localization.map` 在 G1 上无厂商 SLAM，应当**不声明**，
改由本仓库已有的 `fast_lio` + MID360 提供（那是本项目自己的栈，不是运行时能力）。

## 5. X2 ↔ G1 逐条设计决策对照

| 决策 | X2 的结论 | G1 是否成立 | 差异 |
|---|---|---|---|
| **D1** bridge 而非 ros2_control | 成立 | ✅ 完全成立 | G1 同样厂商拥有实时层；`rt/arm_sdk` 的 weight 语义是异步权限移交，`hardware_interface::write()` 表达不了 |
| **D2** 公共关节组只取契约子集 | arm 14 + gripper 2，腿/腰/头不入组 | ✅ 成立，但更好 | G1 `rt/arm_sdk` **可以控腰**（`ArmJoints` 含 3 个 waist）。定为 arm=14 臂，腰 3 走独立 `waist_stream` 通道且不入组；腿 12 只读（§6.4） |
| **D3** runtime mode ↔ 厂商 FSM | MC action | ✅ 成立 | G1 侧是 `SetFsmId` + `SwitchToUserCtrl/InternalCtrl`，语义更简单；`arm_sdk` 的 weight 斜坡要成为 mode 进入/退出的一部分 |
| **D4** TORQUE_OFF 在双足上是摔倒 | `soft_estop`，姿态条件放行 | ✅ 完全成立 | G1 对应 `SetFsmId(damping)`；真正 passive 只在坐/趴时允许 |
| **D5** 无公共 model（无标定文件） | 选项 1：不发布 model，推理/录制 fail closed | ⚠️ **G1 选选项 2** | G1 URDF 是 BSD-3 且线上值已是 SI 弧度，正好满足 X2 D5 选项 2 的前提。见 §6.1 |
| **D6** 仲裁 input-source 生命周期 | ADD→ENABLE，超时丢失 | 🔄 **形态不同** | G1 无命名 input-source，取而代之是 `SwitchToUserCtrl` + `arm_sdk` weight + `mode_machine` 校验。仍需状态机，但更简单 |
| **D7** 厂商传感器只声明不重驱 | 相机/LiDAR/IMU 都由厂商发布 | ⚠️ **部分不成立** | G1 只有 IMU 由厂商发布（`LowState.imu_state`）；D435 与 MID-360 **不由 SDK 发布**，走 LeKiwi 的 D12 路线：runtime 用共享的 `peripheral_nodes` + `fast_lio_nodes` 组合驱动，无需新代码（§2.3 #7） |
| **D8** 厂商包不可 vendoring（无 license） | 阻塞 | ✅ **不成立（好事）** | BSD-3，可 vendoring。但预编译 `.a` 带来另一类问题，见 §6 Q2 |
| **D9** 三层无硬件验证 | pure / vendor mock / conformance | ✅ 直接照搬 | `unitree_vendor_mock` 说 `unitree_hg` + `unitree_api` IDL |
| **D10** 加中立封装暴露厂商能力 | registry v3 | ✅ 复用，无需再扩 | 见 §4 |
| **D11** 音频并入项目音频契约 | `/audio/capture_stamped` 等 | ✅ 成立 | G1 同样自带麦克风阵列与扬声器 |
| **D12** 具身 primitive 另开 PR | `embodied.enabled: false` | ✅ 照搬 | |
| **D13** 遥测翻译不代理 | `PowerState` / `DiagnosticArray` | ✅ 成立 | 源是 `BmsState` / `MainBoardState` |
| **D14** 低层专家层默认关 | `expert_joint_tier: false` | ✅ 完全成立 | G1 对应 `rt/lowcmd` + CRC32；关着就不必实现 CRC |


## 6. 决策落地与新增待决项

> 注（2026-10-12，Q9）：本节 6.1 / 6.2 / 6.4 是路线 1 确定前的分析，其中对 `model_metadata`、
> `joint_groups`、归一化顺序的改动**不在 G1 首个 PR 中做**。

### 6.1 Q1「跑策略推理」落到代码上的具体改动

`pipeline_policy_node` 只读 `joint_groups["all"]`（`inference_service/pipeline_policy_node.py`），
但要拿到 `all`，就必须通过 `validate_model_metadata`，也就必须发布公共 model。
实机路径上的阻碍是以下几处（行号为 master `ca7b8c23`，已复核）：

| 位置 | 现状 | 需要 |
|---|---|---|
| `robot_runtime/model_metadata.py:285` | `"authority": "urdf" if simulated else "calibration"` | 实机也允许 `urdf` |
| `robot_runtime/model_metadata.py:308` | `if positions and not simulated and calibration is None: raise ValueError("physical model requires effective calibrated conversion authority")` | G1 没有 calibration 文件（线上值已是 SI 弧度），此处必须放行 |
| `robot_runtime/model_metadata.py:315` | `elif simulated and robot_description:` 才从 URDF 建换算表 | 实机 + `vendor_internal` 也要走 URDF 建表，否则换算表为空 |
| `robot_runtime/model_metadata.py:340` | 实机上 gripper 关节**跳过** URDF 限位裁剪（假设 calibration 才是范围权威） | `vendor_internal` 时 URDF 就是范围权威，不能跳过；方案 B 下 14 个手指关节都受此影响 |
| `robot_runtime/model_metadata.py:64` + `robot_runtime/schemas/interface_description.schema.json:443` | `authority` 枚举 = `calibration | urdf` | 若选"新增 `vendor` 值"则要扩枚举 |

两种做法：

- **（推荐）复用现有 `urdf` 值**＝X2 design D5 的**选项 2**：profile 声明 `hardware.calibration: vendor_internal` 时，
  `authority` 取 `urdf`、不要求 calibration 文件、从 URDF 建换算表并按 URDF 裁剪限位。
  改动 = `model_metadata.py` 上表前四处约 10–15 行 + 测试，**不动 schema 枚举**，
  对 SO-101/LeKiwi/X2 零影响（它们不声明该键）
- **新增 `vendor` 枚举值**：语义更准确（"厂商内部标定，对外已是物理量"），
  但要同时改 `model_metadata.py`、JSON schema、`interface_description` 及其测试

两者都属于**对 #422 契约层的改动**，会落在 G1 的 PR 里、影响下游两个已有机器人，
按仓库惯例需要在 OpenSpec design 里单列一条决策（对应 X2 的 D5）。

vendoring 清单（Q1 + Q2 共同要求）：

| 包 | 来源 | License | 说明 |
|---|---|---|---|
| `src/robots/unitree/unitree_msgs`（或拆成 `unitree_hg`/`unitree_go`/`unitree_api`） | `unitree_ros2/cyclonedds_ws/src/unitree/` | BSD-3 | 纯 `.msg` 源码，无二进制 |
| `src/robots/unitree/g1_description` | `unitree_ros/robots/g1_description` | BSD-3 | 只取 `g1_29dof_with_hand_rev_1_0.urdf` + 其引用的 meshes |

两者都要在 PR 的「第三方材料及许可证」披露里写明来源、commit 与 BSD-3。

### 6.2 已定（Q5 = B）：Dex3 的 14 个手指关节进 `gripper` 组

这是 Q1（跑推理）+ Q4（Dex3）叠加后**新冒出来**的问题，必须在写 profile 前定。

`validate_model_metadata` 要求 `joint_groups` 恰好是 `{all, arm, gripper, base}`，
且 `arm + gripper + base` 必须**划分** `all`。Dex3 是 7 DoF×2，不是 1 维夹爪。
下游对这几个组的既有语义：

| 消费者 | 读什么 | 约束 |
|---|---|---|
| `inference_service/pipeline_policy_node.py` | `joint_groups["all"]` | 只关心 `all`，不关心怎么分组 |
| `task_dispatch/task_executor_node.py` | `joint_groups["gripper"]` | **"exactly one joint required"** |
| `robot_teleop/public_target.py`、`so101_robot/teleoperation.py` | 一个 arm 组 + 一个 gripper | 单臂单夹爪 |
| `manipulation_execution/pick_executor_node.py` | `joint_groups["arm"]` | |
| `joint_conversion.py` | `gripper` 成员 | 仅改归一化区间：gripper → `0..100`，其余 → `-100..100` |

注意最后一行：**`gripper` 成员身份在数值上只影响归一化区间**，并不强制"一个开度值"。
于是有三条路：

| 方案 | `arm` / `gripper` / `all` | 策略能控 | 代价 |
|---|---|---|---|
| **A. 仅双臂** | arm=14，gripper=[]，all=14 | 14 臂关节 | Dex3 只能靠 `hand.multi_joint` 接口或预置手势另行驱动，**策略学不到手指动作**；与双臂+灵巧手数据集不匹配 |
| **B. 手指塞进 gripper** | arm=14，gripper=14 指，all=28 | 28 | 不改契约即可跑通推理；但违反 `task_executor_node` 的"exactly one gripper"（该层对 G1 关闭，照 X2 的 `embodied.enabled: false`），且"gripper"一词名不符实 |
| **C. 扩 `joint_groups` 增加 `hand` 组** | arm=14，hand=14，gripper=[]，all=28 | 28 | 语义最干净；但要改 `model_metadata` 的划分规则 + JSON schema + `interface_description`，是比 6.1 更大的契约面改动，且 X2 的 D2 明确把"扩组"列为**未经明确决策不得采取**的升级路径 |

另有一个独立维度：`rt/arm_sdk` 的 `ArmJoints` 常量含 **3 个 waist 关节**（`WaistYaw/Roll/Pitch`），
即腰也可在保平衡前提下被流式控制。腰要不要进 `arm` 组（→ all=17 或 31），同样要一并定。

> **已定：B**（2026-10-12）。`task_executor_node` 的单夹爪路径对 G1 不启用；**C** 留作后续单独的契约 change。
> 腰**不进** `arm` 组（Q6）。

### 6.3 需向真机核对的厂商事实

这些无法从代码推断，列为用户执行的 gate（X2 在 2026-09-22 首次上机后，profile 中多项值改为实测值，
G1 应预期同样的修订）：

1. **`mode_machine` 值**。`LowState.mode_machine` 可运行时读取。仓库里 29 DoF + Dex3 的 URDF
   只有 `g1_29dof_with_hand_rev_1_0`（`mode_machine = 5`，4010 腕电机）。
   若实机是较新的 mode 15/16/18（5010 腕），**没有现成的 29dof+Dex3 URDF**，
   需按仓库自带的 `merge_g1_29dof_and_inspire_hand.ipynb` 的方式自行合并。
   `LowCmd::check_mode_machine()` 会校验不匹配，必须先确认。
2. **RPC 请求/响应 topic 名**。`unitree_api/msg/Request{header.identity.{id,api_id}, parameter}`
   的载荷结构已确认，但 `sport` / `arm` / `audio` 三个 service 对应的 DDS topic 名
   （形如 `rt/api/sport/request`）需在真机上核对。
3. **网卡与 DDS 域**。`CYCLONEDDS_URI` 必须指定 G1 所在网卡名（厂商 `setup.sh` 里是 `enp3s0` 占位）。
4. **Dex3 `mode` 位域**。`HandCmd.motor_cmd[i].mode` 是打包字节
   `id:4 | status:3 | timeout:1`（见 `example/g1/dex3/g1_dex3_example.cpp` 的 `RIS_Mode_t`），
   bridge 必须复刻这个编码。
5. **D435 / MID360 的网络与板端可用性**。驱动组合已有共享实现（§2.3 #7），但 MID-360 的
   `lidar_ip` / host 网卡、`realsense2_camera` 在 G1 计算板上是否可装，需上机确认。
6. **停止语义**。`SetFsmId(damping)` 与 `SwitchToInternalCtrl(PASSIVE)` 的实际时延与姿态前提。
7. **厂商端点 QoS**（合入版 X2 的教训，§2.3 #4）。`rt/lowstate`、`rt/arm_sdk`、`rt/dex3/*`、
   RPC request/response 各自的 reliability/durability/depth，用 `ros2 topic info -v` 实测后写入 profile。

### 6.4 同一台 G1 上的多种模型（只控双臂 / 双臂+腰 / 带手指…）

**现有框架已经支持的部分：**

- 一个模型的输入输出由部署配置里的 `contract`（`observations` / `actions`）决定，
  **按关节名选取**：`selector.names: ["position.left_shoulder_pitch_joint", ...]`
  经 `tensormsg.utils.dot_get` 在 `JointState.name` 中查找，不依赖机器人的关节顺序。
- 动作按段发布到 runtime 的命令通道（`joint.<channel>`，每个通道有固定关节表），
  SO-101 已经是"一个 action 张量切成 `arm_stream` + `gripper_stream` 两段"。
- 一个部署配置对应一份 contract；不同动作空间的模型 = 不同部署配置（或不同 `base_contract`），
  这与 SO-101 / LeKiwi 现有"每种用法一个 yaml"的做法一致。

据此 G1 的建议形态：

| 层 | 设计 |
|---|---|
| profile `joints`（= model `all`） | 全部 43 个，顺序为 **14 臂 → 14 手指 → 3 腰 → 12 腿**（Q8，见下）。`joint.state` 也发布全部 43 个；`LowState` 的电机下标顺序（腿在前）由 bridge 重排 |
| `joint_groups` | `arm` = 14 臂，`gripper` = 14 手指（Q5=B），`base` = []，腰与腿**不入组**（Q6） |
| 命令通道 | `arm_stream`（14）、`waist_stream`（3）、`hand_left_stream`（7）、`hand_right_stream`（7），腿无通道 |
| 未被下发的通道 | bridge **锁存进入模式时的位置**并持续下发（`rt/arm_sdk` 的 weight 是 17 个关节共用的，腰一旦被接管就必须有目标值）；不能用"每 tick 回填实测值"，那样腰会在重力下缓慢下沉 |
| 每个模型 | 一份部署配置：只控双臂的模型只绑 `arm_stream`（+手指通道）；带腰的模型额外绑 `waist_stream` |
| 数据采集 | 录**全集**（43 个关节状态 + 所有已下发通道），训练时按需选特征子集；同一份数据可以喂给不同动作空间的模型 |

**两处契约相关的决定：**

1. **划分规则放宽（Q7，已定）**。`validate_model_metadata`（`model_metadata.py:71-73`）要求
   `arm + gripper + base` 恰好**划分** `all`；腰不进 `arm` 组 ⇒ 腰无法出现在 `all` 里 ⇒
   带腰的模型拿不到腰的归一化参数。改为"三组互不相交，且都是 `all` 的子集"（约 3 行 + 测试）。
   已核对的消费者——`interface_binding`、`robot_teleop/public_target`、`so101_robot/teleoperation`、
   `task_executor_node`、`pick_executor_node`、`dataset_tools/bag_to_lerobot`——都只按组名取组，
   不依赖"`all` 恰好等于三组之和"；SO-101 / LeKiwi / X2 仍满足原划分，行为不变。不动 JSON schema。
2. **归一化保持机器人关节顺序（Q8，已定）**。`pipeline_policy_node.py:738` 按 `joint_groups["all"]`
   建换算表，在 `_rad_to_lerobot` / `_lerobot_to_rad` 中按下标作用于状态/动作向量。不改。
   由此得到的**部署约束**：contract 的 `observation.state` / `action` 必须是 `all` 的**前缀**
   （按 `all` 顺序、从第 0 个起连续）；需要非前缀子集的模型，就取到覆盖它的最短前缀，
   在模型内部做选择/映射。为让常用模型都是前缀，profile 的 `joints` 按
   **14 臂 → 14 手指 → 3 腰 → 12 腿** 排列：

   | 模型 | 取 `all` 的前缀 |
   |---|---|
   | 只控双臂 | 前 14 |
   | 双臂 + Dex3 | 前 28 |
   | 双臂 + Dex3 + 腰 | 前 31 |
   | 双臂 + 腰（不含手指） | 前 31，模型内部忽略手指的 14 维 |

   注意：换算循环对长度不符只截断不报错（`if index < converted.shape[-1]`），
   若 contract 写成了非前缀子集会**静默**用错关节的上下限。G1 部署配置的测试里应断言
   contract 的 selector 名恰为 `all` 的前缀，在 G1 包内兜住，不改推理节点。

**目前不支持、也不建议首个 PR 做的：**同一个部署配置内按控制模式切换不同动作空间的模型
（contract 现在是部署级的，不是 pipeline 级的）。需要时再单独提 change 把 contract 下放到 pipeline。

## 7. 工作量估算（按已定决策、以合入版 `aimdk_robot` 为参照）

| 工作项 | 估算 | 说明 |
|---|---|---|
| `unitree_msgs` vendoring（`unitree_hg`/`unitree_go`/`unitree_api`） | ~40 个 `.msg` + 包壳 | Q2；纯拷贝 + 归属声明 |
| `g1_description` vendoring（URDF + meshes） | 1 个 ROS 包 | Q1；只取 29dof+hand 一个变体及其 meshes |
| `robot_runtime` authority 放开（X2 D5 选项 2） | ~15 行 + ~80 行测试 | **触及契约层**，对现有机器人零行为变化 |
| `joint_groups` 划分规则放宽为"互斥子集"（§6.4-1） | ~3 行 + ~40 行测试 | 契约层；现有机器人仍满足原规则 |
| G1 部署配置的"contract 为 `all` 前缀"断言（§6.4-2） | ~30 行测试 | 放在 `robot_config` 的 G1 配置测试里 |
| `profiles/g1_29dof_dex3.yaml` | 700–850 行 | 含逐端点 QoS、peripherals 默认值；无仲裁/表情/relocalization |
| `projection.py`（关节投影 / FSM 映射 / Twist 换算 / weight 斜坡 / Dex3 位域 / 停止规划） | 650–800 行 | 纯函数，可先行 |
| `runtime_node.py` | 1400–1700 行 | 输出路径准入、通道独立过期等照合入版；少掉 head 模式与 input-source 仲裁 |
| `vendor_gateway.py`（`Request`/`Response` RPC 关联 + FSM 状态机 + `mode_machine` 校验） | 500–650 行 | |
| `vendor_mock_node.py` | 600–700 行 | 说 `unitree_hg` + `unitree_api` IDL |
| `named_motion.py`（`ArmActionClient` 预置动作 + FSM 姿态） | 250–350 行 | |
| `launch/runtime.launch.py` | 150–200 行 | `CYCLONEDDS_URI` 网卡参数、RMW 固定、`peripheral_nodes` + `fast_lio_nodes` |
| 测试三件套（profile / projection / bridge） | 1800–2300 行 | |
| README（含上机步骤） | 400–550 行 | |
| `robot_config/config/robots/unitree_g1.yaml`（+ 可选 `unitree_g1_skills.yaml`）+ test | ~250 行 | 技能侧复用 `runtime_named_motion`，无新执行器 |
| `build.sh` 分组 + `verify_runtime_independence.sh` 条目 | ~45 行 | 照 `AIMDK_RUNTIME_MEMBERS` 写法 |
| **合计（不含 vendoring 的 .msg/mesh）** | **≈ 7000–8500 行** | 原估 4000–4700 行是按 #426 PR head 算的，合入版 X2 本体约 9200 行 |

如果想控制单个 PR 规模，可以把"authority 放开 + vendoring + profile + projection（纯函数）"
作为第一个 PR，bridge 节点与部署接线作为第二个。

vendoring 的消息与 mesh 会让 PR 的 diff 行数远超 2000 行门槛，因此
`atomgit-pr` skill 的 **`## Reuse Self-Check` 章节为强制**（四个固定字段）。

## 8. 建议的推进顺序

1. 起 OpenSpec change `unitree-g1-runtime-migration`，照 `aimdk-runtime-migration/design.md`
   的体例把 D1–D14 对 G1 重裁一遍（§5 的表是草稿），并把 §6.1 的 authority 改动
   单列为一条决策（对应 X2 的 D5，但结论相反），§6.4-1 的划分规则放宽也单列一条
2. 分三阶段落地（照 #426 的 Migration Plan）：
   - 阶段 1：vendoring + packaging + `projection.py` + profile，纯函数测试通过，**不接触厂商流量**
   - 阶段 2：`vendor_gateway` + `runtime_node` 对着 `unitree_vendor_mock` 跑，conformance 通过
   - 阶段 3：authority 改动 + `build.sh` 分组 + 独立性门禁 + `robot_config` 部署接线
     （可选：`unitree_g1_skills` 部署，复用 `runtime_named_motion`）
3. §6.3 的七项真机核对列为用户执行的 gate，agent 不自行勾选
4. PR 阶段：先 `[WIP]` 提交暂缓双平台 Docker；因为动了 `package.xml` 依赖与构建脚本，
   转正式检视时**必须**补齐 Ubuntu + openEuler 双平台纯净 Docker 的 `setup.sh + build.sh` 证据

## 附：本地材料位置

```
<worktree>/unitree/   # 本地上游工作副本，gitignored
├── unitree_sdk2/     # BSD-3，头文件 + IDL + 预编译 .a（按 Q2 决策，仅作接口参考，不链接）
├── unitree_ros/      # BSD-3，sparse checkout → robots/g1_description
└── unitree_ros2/     # BSD-3，cyclonedds_ws/src/unitree/{unitree_hg,unitree_go,unitree_api}
```

已加入本 worktree 的 `.gitignore`（`/unitree/`），与 #426 处理 `/aimdk/` 的方式一致。
