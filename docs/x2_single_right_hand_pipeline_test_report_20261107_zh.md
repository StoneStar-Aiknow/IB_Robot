# X2 单右手与统一 YAML 功能测试报告

## 1. 测试范围与结论

测试日期为 2026-11-07，时间均采用北京时间。
机器人保留双臂，只安装右侧灵巧手。
本报告统计 02:02:29 至 02:44:21 的新总配置记录。
两端的总入口均为
[`aimdk_x2_voice_mirror_grasp.yaml`](../src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml)。

本轮包含 8 次请求：7 次 Agent 文字请求和 1 次 Gateway 定向请求。
Gateway 指能力网关（Capability Gateway）。
执行记录中，5 次请求成功，3 次准备位请求失败。
表中策略顺序为 `hand/body`。

| 功能 | 配置策略 | 结果 |
| --- | --- | --- |
| Gateway 定向挥手 | `online/online` | 1 次成功 |
| Agent 文字挥手、飞吻 | `online/online` | 各 1 次成功 |
| Agent 文字回抓取准备位 | `online/online` | 2 次失败，路径间隙检查拒绝 |
| Agent 文字回抓取准备位 | `off/off` | 1 次失败，右肩到位误差超限 |
| Agent 文字开始、结束照镜子 | `off/off` | 各 1 次成功 |

最终总配置为 `off/off`，两端已核验生效。
本轮没有在最终策略下重测挥手和飞吻。
成功指 Gateway 返回 `SUCCEEDED`，不代表现场动作质量验收。
抓取准备位尚未完整通过，右手张开阶段均未执行。
本轮未执行苹果抓取。

本轮未执行握手、举手、鼓掌、自我介绍、自由姿态和休息位。
双手配置仅通过软件配置测试，尚无本轮双手真机结果。

## 2. YAML 继承与引用关系

### 2.1 机器人配置继承链

下列文件均位于 `src/robot_config/config/robots/`。
箭头表示左侧文件通过 `base_config` 继承右侧文件。

```text
aimdk_x2_voice_mirror_grasp.yaml
  -> aimdk_x2_voice_mirror_all.yaml
    -> aimdk_x2_voice_mirror.yaml
      -> aimdk_x2_magicmirror.yaml
        -> aimdk_x2_skills.yaml
          -> aimdk_x2.yaml
```

| 文件 | 本层职责 |
| --- | --- |
| [`aimdk_x2.yaml`](../src/robot_config/config/robots/aimdk_x2.yaml) | 厂商运行时、关节和控制模式基础配置 |
| [`aimdk_x2_skills.yaml`](../src/robot_config/config/robots/aimdk_x2_skills.yaml) | 技能执行基础设施与命名动作 |
| [`aimdk_x2_magicmirror.yaml`](../src/robot_config/config/robots/aimdk_x2_magicmirror.yaml) | OmniHand 运行时、视觉服务与照镜子会话 |
| [`aimdk_x2_voice_mirror.yaml`](../src/robot_config/config/robots/aimdk_x2_voice_mirror.yaml) | Agent、语音桥和语音输入 |
| [`aimdk_x2_voice_mirror_all.yaml`](../src/robot_config/config/robots/aimdk_x2_voice_mirror_all.yaml) | 同时启用手势与照镜子，组合录音和命名动作 |
| [`aimdk_x2_voice_mirror_grasp.yaml`](../src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml) | 抓取服务、任务姿态、安装侧与部署覆盖项 |

加载器递归合并父配置，子配置覆盖同名值。
映射按字段合并，普通列表替换父列表。
总入口的 `perception_services.services.__append__` 追加抓取服务。
这个声明保留父配置的 YOLOX 与 PEAR 服务。
YOLOX 是人体检测模型，PEAR 是人体姿态参数模型。
最终值以合并后的消费配置快照为准。

总入口将手势、照镜子和抓取统一到 `policy_stream`。
总入口还选择 X-ASR 模型、Agent 模型服务和录音路径。

### 2.2 技能目录与姿态资产

以下 3 类文件通过引用连接，不构成新的 `base_config` 层：

| 文件或字段 | 引用方式 | 职责 |
| --- | --- | --- |
| [同名技能目录 profile](../src/agent/skill_catalog/config/profiles/aimdk_x2_voice_mirror_grasp.yaml) | `embodied.skill_catalog_profile` | 暴露手势、照镜子、抓取和准备位技能 |
| [`x2_example.yaml`](../src/robot_config/config/task_poses/x2_example.yaml) | `task_poses.file` | 提供准备位、休息位、关节目标和批准标记 |
| `commissioning/fixed_poses.json` | `pose_composer.fixed_library` | 提供离线编译的固定姿态与模型资产 |

技能目录和机器人配置虽然同名，但属于不同目录。
安装侧只在机器人总配置中声明。
`task_poses.hands: installed` 根据安装侧选择开手动作。
这个选择保留双臂目标和批准标记。

技能目录的 `pick_object` 复用 `aimdk_x2_grasp_task_poses` 实现。
抓取参数仍在本总入口中声明。
`aimdk_x2_grasp.yaml` 不在本入口的继承链中。

## 3. 单手、双手与现场批准

修改总 YAML 中的这 2 个字段即可选择安装侧：

```yaml
robot:
  grasp_execution:
    backend_options:
      installed_hands: [right]
      arm_side: right
```

| 硬件状态 | `installed_hands` | `arm_side` |
| --- | --- | --- |
| 只安装右手，本轮真机配置 | `[right]` | `right` |
| 只安装左手 | `[left]` | `left` |
| 安装双手，按目标侧选择抓取臂 | `[left, right]` | `auto` |

单手配置拒绝 `arm_side: auto`。
两端必须使用相同的安装声明。
修改安装侧、任务姿态、批准值或关节限制后，重新编译姿态库。
编译器从总 YAML 读取输入，无需另写现场机器人 YAML。
编译命令见[统一配置指南](x2_voice_mirror_grasp_config_zh.md#姿态批准与固定姿态库)。

仓库示例姿态的硬件批准值仍为 `false`。
现场部署沿用操作员已批准的目标与 `true` 标记。
现场修改没有改变准备位角度和到位容差。
仓库配置本身不会自动授予硬件批准。

| 现场核验项 | 结果 |
| --- | --- |
| 两端安装侧 | 均为 `[right]`，抓取侧为 `right` |
| 新旧准备位快照 | 完全一致，摘要为 `7e76950c…ac327d` |
| 固定姿态库 | 两端均包含 12 个固定姿态 |
| 固定姿态库摘要 | `e5c4bc31…fdc50d` |
| 新旧关节限制 | 一致 |
| 硬件与空手操作批准 | 运行快照中均为 `true` |
| 最终源码 YAML 校验值 | 两端均为 `6f15561c…22cab` |

完整摘要见[本轮证据 JSON](test_evidence/x2_voice_mirror_grasp_20261107.json)。

## 4. 本次代码整理

单手执行基础来自提交 `a1698a27f`。
本次整理使两端直接使用仓库总 YAML。

| 修改位置 | 行为变化 |
| --- | --- |
| `robot_runtime/hand_mounts.py` | 共用安装侧校验，拒绝重复、未知和非字符串值 |
| `robot_config/task_poses.py` | 从安装声明选择开手侧，校验后冻结姿态摘要 |
| `robot_config/fixed_poses.py` | 共用编译和运行时路径解析，保留内容校验 |
| `pose_composer/compile_library.py` | 新增 `--robot-config`，从总 YAML 读取编译输入 |
| `embodied_bringup/launch_builders/embodied.py` | 解析录音路径中的 `WORKSPACE` 替换 |
| `aimdk_grasp/joints.py` | 复用公共安装侧校验，避免两套解析规则 |
| `x2_grasp_head.launch.py` | Orin 不再校验仅在 310P 使用的模型资产 |
| 总 YAML 与技能目录说明 | 共用单手、双手入口，明确配置归属 |
| 配置与启动测试 | 覆盖安装侧切换、路径解析、参数冲突和分布式加载 |

编译器保留原来的显式参数入口。
`--robot-config` 拒绝混用 `--profile`、`--limits` 和 `--task-poses`。
编译器不要求目标姿态库已经存在。
运行时仍拒绝内容或姿态摘要不匹配的资产。

## 5. 构建、启动与文字测试方法

### 5.1 环境与软件回归

310P 使用 openEuler Embedded、ROS 2 Humble 和 Python 3.11。
Orin 使用 Ubuntu、ROS 2 Humble、CUDA 和 cuRobo。
两端使用 `ROS_DOMAIN_ID=0` 与 `rmw_fastrtps_cpp`。
310P 承载 Agent、Gateway、语音和视觉模型服务。
Orin 承载抓取头，相机使用现场已有驱动。

本机使用工作树自有虚拟环境。
本次通过 `scripts/build.sh` 构建 5 个包：
`robot_runtime`、`robot_config`、`pose_composer`、
`embodied_bringup`、`aimdk_grasp`。
定向构建忽略未选中的源码包，复用已构建的依赖。
正式定向验收通过 `colcon test` 执行。

| 包 | 通过 | 跳过 | 失败或错误 |
| --- | ---: | ---: | ---: |
| `robot_config` | 73 | 0 | 0 |
| `pose_composer` | 24 | 0 | 0 |
| `aimdk_grasp` | 146 | 2 | 0 |
| `embodied_bringup` | 73 | 6 | 0 |
| `skill_catalog` | 5 | 0 | 0 |
| 合计 | 321 | 8 | 0 |

这是本次修改范围的定向结果，不是全部包的全量测试。
证据 JSON 保存测试范围和结果文件摘要。
跳过项未获得通过结论。

### 5.2 启动两端

前置条件：两端源码、模型和固定姿态库已部署。
现场环境脚本内部加载本端 `.shrc_local` 和厂商消息环境。
真机目标变量必须在加载环境前设置。
首次启动曾继承离线环境，造成 DDS 配置冲突。
DDS 指数据分发服务（Data Distribution Service）。
使用下列顺序后，两端启动检查通过。

1. 在 Orin 部署根目录启动抓取头。

   ```bash
   export HZ_TARGET=robot HZ_ROS_DOMAIN_ID=0
   source commissioning/env_orin.sh
   ros2 launch aimdk_grasp x2_grasp_head.launch.py \
     robot_config:=aimdk_x2_voice_mirror_grasp \
     config_path:="$PWD/src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml"
   ```

2. 在 310P 部署根目录启动主管线。

   ```bash
   export VOICE_POSE_TARGET=robot
   source commissioning/env.sh
   ros2 launch embodied_bringup embodied_pipeline.launch.py \
     robot_config:=aimdk_x2_voice_mirror_grasp \
     config_path:="$PWD/src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml" \
     control_mode:=model_inference with_moveit:=false \
     use_sim:=false authorize_motion:=true
   ```

`config_path` 显式选择本端仓库源码中的同一总入口。
本轮没有通过额外的现场机器人 YAML 覆盖总入口。
模型和姿态库仍属于本端部署资产。

3. 等待 Agent、Gateway 和语音服务就绪。
4. 核对消费快照中的安装侧与路径策略。

本轮最终快照为 `/tmp/ibrobot_interfaces__eyk0aq1/robot.yaml`。
现场指针为 `commissioning/live_config.path`。
重启会生成新的消费快照。
`ready=true` 只证明接口准入，不能替代抓取准备检查。

### 5.3 通过 Agent 发送文字

1. 在 310P 部署根目录启动文字客户端。

   ```bash
   export VOICE_POSE_TARGET=robot
   source commissioning/env.sh
   python3 -m ibrobot_agent.chat_tui \
     --channel-id voice_asr --principal-id local_operator \
     --robot-scope aimdk_x2_voice_mirror_grasp \
     --session-id x2-text-check
   ```

2. 输入一条测试指令。
3. 等待这条指令返回终态。
4. 输入下一条已授权的测试指令。

客户端把文字交给 `/agent/request`。
Agent 规划后，通过 Gateway 执行技能。
本轮挥手、飞吻和准备位测试均逐条提交。
本轮照镜子先返回启动成功，再提交结束指令。
记录中没有自动重试。

查看日志：

```bash
tail -n 100 -F "$(cat commissioning/pipeline.logpath)"
```

Orin 使用 `commissioning/head.logpath` 查看抓取头日志。

## 6. 新 YAML 真机测试结果

### 6.1 请求与终态

| 编号 | 入口与文字 | Agent 技能 | 策略 | 终态 |
| --- | --- | --- | --- | --- |
| T01 | Gateway 定向挥手 | `wave_hand`，调用方指定 | `online/online` | `SUCCEEDED` |
| T02 | Agent：请挥手一次 | `wave_hand` | `online/online` | `SUCCEEDED` |
| T03 | Agent：请做一次飞吻动作 | `blow_kiss` | `online/online` | `SUCCEEDED` |
| T04 | Agent：回到抓取准备位 | `return_to_task_pose(grasp_ready)` | `online/online` | `HAND_SWEEP_COLLISION` |
| T05 | Agent：回到抓取的初始位置 | `return_to_task_pose(grasp_ready)` | `online/online` | `HAND_SWEEP_COLLISION` |
| T06 | Agent：回到抓取的初始位置 | `return_to_task_pose(grasp_ready)` | `off/off` | `TASK_POSE_NOT_REACHED` |
| T07 | Agent：开始照镜子 | `mirror_start` | `off/off` | `SUCCEEDED` |
| T08 | Agent：结束照镜子 | `mirror_stop` | `off/off` | `SUCCEEDED` |

### 6.2 路径检查与准备位

总配置最初使用 `hand: online, body: online`。
手部路径检查计入 `0.1 rad` 的手指保持误差包络。
T04 和 T05 均在轨迹下发前拒绝执行。

碰撞体对为 `right_wrist_yaw_link/R_thumb_abad`。
模型间隙分别为 `5.993 mm` 和 `5.995 mm`。
本次计算要求的间隙为 `10.181 mm`。
这些结果不证明实物发生碰撞。

按操作员指令，总配置随后改为：

```yaml
pose_composer:
  path_validation:
    hand: "off"
    body: "off"
```

双引号保证 YAML 将 `off` 解析为字符串。
两端重启后，消费快照和执行回执均确认为 `off/off`。
执行器保留入口、端点、反馈与执行准入检查。
`off` 不证明完整路径已经通过碰撞检查。

T06 的手臂轨迹已执行结束。
单次到位检查测得右肩误差为 `0.111417 rad`。
允许误差为 `0.1 rad`。
最差关节为 `right_shoulder_roll_joint`。
系统因此返回 `TASK_POSE_NOT_REACHED`。
稳定时间窗尚未获得验证。

3 次准备位请求的 `completed_hands` 均为空。
左手阶段标记为 `hand action not requested`。
右手阶段标记为 `stage not reached`。
这些记录没有证明右手执行器故障。
到位容差在本轮保持原值。

### 6.3 照镜子与关闭结果

T07 启动成功后，会话从 `WAIT` 进入 `STANDBY`。
会话随后进入 `MIRRORING/running`。
Agent 收到启动终态后，客户端才提交 T08。
T08 触发 `wrap_up`，会话返回 `WAIT`。
结束回执确认运行时返回 `idle`。
结束后 Gateway 的 `busy=false`，Agent 仍就绪。

本轮验证了文字规划、会话启动和会话结束。
本轮没有量化人体跟随误差或录制视频验收。
语音服务初始化记录不代表识别准确率或播音质量验收。

按操作员指令，本轮测试后关闭了两端管线。
310P 进程组 `145636` 和 Orin 进程组 `392883` 均已清空。
关闭范围为主管线与抓取头，厂商相机驱动独立运行。

## 7. 证据与复核边界

结构化证据见
[`x2_voice_mirror_grasp_20261107.json`](test_evidence/x2_voice_mirror_grasp_20261107.json)。
证据包含请求标识、规划摘要、终态、错误和阶段计时。
证据也包含配置对比、软件测试结果和关闭记录。
旧现场配置的 21 次测试不计入本报告。

| 阶段 | 310P 日志 | Orin 日志 |
| --- | --- | --- |
| `online/online` | `pipeline_unified_yaml_20261107_014334.log` | `head_unified_yaml_20261107_014331.log` |
| 最终 `off/off` | `pipeline_unified_yaml_20261107_023130.log` | `head_unified_yaml_20261107_023051.log` |

日志位于本端部署目录的 `commissioning/logs/`。
本轮未把厂商模型、离线几何资产或录音纳入提交。

## 8. 动作时延

所有时延单位均为秒，保留 3 位小数。
发送至终态来自 Agent 请求账本的时间差。
执行器时延来自日志中的 `duration_sec`。
T01 不经过 Agent 文字入口，未统计文字请求总时延。
失败行记录失败返回耗时，不是成功完成耗时。

| 编号 | 动作 | 策略 | 终态 | 发送至终态 / s | 执行器 / s |
| --- | --- | --- | --- | ---: | ---: |
| T01 | Gateway 挥手 | `online/online` | 成功 | — | 6.502 |
| T02 | Agent 挥手 | `online/online` | 成功 | 8.413 | 5.251 |
| T03 | Agent 飞吻 | `online/online` | 成功 | 7.478 | 5.406 |
| T04 | 回准备位 | `online/online` | 路径检查拒绝 | 4.587 | 1.810 |
| T05 | 回初始位置 | `online/online` | 路径检查拒绝 | 6.430 | 1.661 |
| T06 | 回初始位置 | `off/off` | 到位检查失败 | 8.298 | 4.547 |
| T07 | 开始照镜子 | `off/off` | 成功 | 1.973 | 0.060 |
| T08 | 结束照镜子 | `off/off` | 成功 | 5.521 | 3.076 |

准备位阶段计时如下。
运动调用包含下发前的检查。
T04 和 T05 的运动调用未下发手臂轨迹。
“—”表示没有执行该阶段。

| 编号 | 准备总耗时 / s | 规划 / s | 运动调用 / s | 到位检查 / s | 右手张开 / s |
| --- | ---: | ---: | ---: | ---: | ---: |
| T04 | 1.661 | 0.006 | 0.322 | — | — |
| T05 | 1.522 | 0.010 | 0.297 | — | — |
| T06 | 4.357 | 0.006 | 2.903 | 0.001 | — |
