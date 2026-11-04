# 灵犀 X2 抓取技能（pick_object）部署与使用

本文说明如何在灵犀 X2（OmniHand 灵巧手）上部署 `pick_object` 技能，并用 `robot-skill` 调用它。

技能用左手抓取一个用文字指定的物体，抬起并保持，再放回原处，最后让左臂回到初始位置。
X2 没有容器，所以 X2 实现不运送物体。

## 架构

抓取分布在两台主机上：

| 主机 | 运行内容 |
| --- | --- |
| 机器人侧主机（NVIDIA GPU 主机或 Ascend 310P） | X2 运行时 `aimdk_runtime` 与数据面 `aimdk_dataplane`、Capability Gateway、技能栈、感知模型服务、抓取管线 `grasp_pipeline_node` |
| X2 的 Orin（开发计算单元，`10.0.1.41`） | 抓取头 `x2_grasp_head_node`：头部相机采集与配准、碰撞场景、cuRobo 规划、目标监视 |

```text
robot-skill / Hermes / planner
  -> /embodied/execute_skill
     Gateway：motion_authorized、运行时状态、切换到 policy_stream（厂商 UPPERBODY_REMOTE_SPLIT）
  -> grasp_pipeline_node（机器人侧主机）
       检测、分割、生成候选：感知模型服务
       采集、场景、规划、目标监视：/x2_grasp_head/*（Orin 上的抓取头）
       左臂与左手流式控制：joint.arm_stream、joint.hand_stream
  -> aimdk_dataplane -> 厂商 MC 层
```

厂商禁止跨计算单元订阅原始相机流。因此抓取头和头部相机都在 Orin 上运行。
两台主机之间只传压缩采集、目标 mask、抓取候选、规划结果和监视状态。
一次请求的压缩采集为 0.45 MB（1280x720）。

两套部署配置共用 `aimdk_x2_grasp.yaml`。按机器人侧主机的加速器选择一套：

| 配置 | 机器人侧主机 | 检测 | 分割 | GraspGen-X |
| --- | --- | --- | --- | --- |
| `aimdk_x2_grasp_cuda` | NVIDIA GPU | grounded SAM2，`torch_cuda`，同时返回 mask | 无 | `torch_cuda` |
| `aimdk_x2_grasp_310p` | Ascend 310P | GroundingDINO，`ascend_310p` | SAM2 prompt，`ascend_310p` | `torch_npu` |

两台主机必须启动同一套配置。

## 安全须知

- 运动默认未授权。`authorize_motion:=false` 时，`pick_object` 返回 `MOTION_NOT_AUTHORIZED`，机器人不动。
- 抓取会移动左臂和左手。每次真机运行时，操作员必须在现场，并能随时调用 `/runtime/stop`。
- vendor mock 与真机使用同名的厂商接口。如果在机器人网络上启动 mock，真机会执行发给 mock 的请求。
  在 mock 上运行时，设置 `ROS_LOCALHOST_ONLY=1` 和一个机器人没有使用的 `ROS_DOMAIN_ID`。
- 如果左手半握，下一次请求会让抓取头重建规划器。重建在 Orin 上需要 50 s，请求会超时。

## 前置条件

硬件和网络：

- 一台机器人侧主机，有 NVIDIA GPU 或 Ascend 310P。
- 机器人侧主机用网线接到 X2 的机器人网络，配置一个静态地址 `10.0.1.x/24`。
  机器人单元已占用 `10.0.1.40`、`10.0.1.41`、`10.0.1.42`。
- 机器人侧主机与 X2 使用同一个 `ROS_DOMAIN_ID` 和 `RMW_IMPLEMENTATION`。`ROS_LOCALHOST_ONLY` 未设置或为 `0`。

软件：

- 两台主机都有 ROS 2 Humble，都有本仓库同一个提交的代码。
- 机器人侧主机有与机器人固件匹配的 AimDK overlay（`aimdk_msgs`）。
  overlay 不在仓库内，构建方法见 `src/robots/aimdk/aimdk_robot/README.md`「Vendor dependency」。
- Orin 的 Python 环境有 NVIDIA Jetson 版 PyTorch。PyPI 的 aarch64 CUDA wheel 不含 Orin 的 `sm_87` 内核，
  不能用于 Orin。

模型（放在两台主机的 `models/` 下）：

| bundle | 使用的配置 | 获取方法 |
| --- | --- | --- |
| `grounded_sam2_swint_ogc` | `aimdk_x2_grasp_cuda` | `./scripts/download_perception_models.sh` |
| `grounding_dino_swint_seq8_1280x720` | `aimdk_x2_grasp_310p` | `package_ascend_perception_bundles --model-type grounding_dino` |
| `sam2.1_hiera_tiny_prompt_ascend` | `aimdk_x2_grasp_310p` | `package_ascend_perception_bundles --model-type sam2` |
| `graspgenx_omnihand_left_x2` | 两套都用 | `package_graspgenx_torch_bundle` |

`graspgenx_omnihand_left_x2` 需要 OmniHand 左手的夹爪描述。夹爪描述的再分发条款还没有确认，所以仓库不提供它。
向维护者获取夹爪描述，再按 `src/perception_service/README.md` 第 10 节打包。

Orin 也需要 `graspgenx_omnihand_left_x2`。规划器用它的夹爪描述，执行器用它的 manifest 核对执行器身份。
两台主机上的这个 bundle 必须完全相同。

## 准备网络和时钟

警告：主机时钟与机器人时钟相差大于 0.1 s 时，运行时拒绝进入 `ACTIVE`。
时钟相差大于 0.2 s 时，平台丢弃运行时发出的全部命令，并且两侧都不报告错误。

1. 让机器人侧主机跟随机器人的 PTP 时钟。X2 的时钟是 PTP domain 1，用 `phc2sys -O 0` 把 PTP 时间直接写入系统时钟。
2. 用机器人的消息时间戳确认时钟一致：

   ```bash
   ros2 topic echo --once /aima/mc/common/state --field header.stamp --qos-reliability best_effort
   date +%s.%N
   ```

3. 把机器人侧主机的 UDP 缓冲调到与机器人单元相同的值：

   ```bash
   sudo sysctl -w net.core.rmem_max=104857600 net.core.wmem_max=104857600
   sudo sysctl -w net.core.rmem_default=2097152 net.core.wmem_default=2097152
   ```

   如果缓冲保持内核默认值 212 KB，主机会丢弃 UDP 数据报。可靠话题随后反复重传，2.76 MB 的检测请求无法到达。

4. 确认机器人侧主机能列出 `/aima/` 话题：

   ```bash
   ros2 topic list | grep '^/aima/'
   ```

如果机器人侧主机有多个 IPv4 地址，Fast DDS 默认向全部地址发布数据，对端按每个地址各发一份。
2026-10-22 实测，机器人侧主机有 6 个地址时，厂商 `/tf` 从 45 Hz 降到 20 Hz。
当前 `scripts/dds` 的 `dds-config` 不支持只绑定一块网卡。建议机器人侧主机只保留机器人网卡的地址。

## 安装和构建

在两台主机上分别执行以下步骤。

1. 安装依赖。在 NVIDIA 主机上，`setup.sh` 在 `cuda` 加速器下安装 cuRobo 和 Warp：

   ```bash
   ./scripts/setup.sh
   ```

   cuRobo 来自 `third_party/wheels/curobo/` 中经过审计的纯 Python wheel，运行时用 NVRTC 编译 CUDA 内核。
   同一个 wheel 也用于 Orin。

2. 加载环境。机器人侧主机还要加载 AimDK overlay：

   ```bash
   source .shrc_local
   source <AimDK overlay>/install/local_setup.bash
   ```

3. 构建：

   ```bash
   ./scripts/build.sh
   ```

4. 设置 `WORKSPACE` 为仓库根目录。如果不设置，`robot_config` 拒绝全部相对的 `bundle_path`。`.shrc_local` 会设置它。

在 Orin 上，用 venv 里的 colcon 构建。`/usr/bin/colcon` 生成的入口脚本使用系统 Python，
系统 Python 找不到 venv 里的 NVRTC 头文件，cuRobo 编译内核会失败。

## 启动

前置条件：

- 机器人已上电，平台处于上半身遥控模式 URS（`UPPERBODY_REMOTE_SPLIT`）。
- 左手 OmniHand 已激活：厂商手部状态里左手的 10 个关节都有 `L_` 开头的名字。
- 左手张开。左手与 GraspGen-X 夹爪描述的张开预设相差不超过 0.025 rad。
- 左臂在初始位置 `HOME_COMMAND`：肩 pitch 0.4 rad，肘 -1.2 rad，其余关节 0 rad。每个关节的偏差不超过 0.1 rad。
- 桌面上有目标物体，与其他物体分开。建议把物体放在机器人左前方 40 至 46 cm 处。2026-10-26 实测，这个范围内的物体能规划成功。

机器人处于稳定站立（`STAND_DEFAULT`）时，OmniHand 没有激活：左手关节没有名字，位置都是 0。
抓取头预热时检查左手是否张开。如果左手没有激活或没有张开，预热失败，日志出现 `Initial left hand is not open`，抓取后端一直不就绪。
因此，启动前用 AimDK 厂商工具依次执行 4 步：稳定站立、切换到 URS、发送上半身命令、张开左手。
本仓库还没有提供这 4 步的工具（待补）。

如果左臂偏离 `HOME_COMMAND` 大于 0.1 rad，执行器不自动复位。先把左臂移回初始位置，再发起抓取。

步骤：

1. 在 Orin 上启动抓取头（以 CUDA 部署为例）：

   ```bash
   ros2 launch aimdk_grasp x2_grasp_head.launch.py robot_config:=aimdk_x2_grasp_cuda
   ```

   抓取头等待机器人侧主机的运行时就绪，所以可以先启动它。
   如果配置不是 OmniHand X2、`aimdk_robot` 运行时和远程抓取后端的组合，launch 在启动任何节点前退出。

2. 在机器人侧主机上启动运行时、Gateway、技能栈、模型服务和抓取管线：

   ```bash
   ros2 launch embodied_bringup embodied_pipeline.launch.py \
     robot_config:=aimdk_x2_grasp_cuda control_mode:=moveit_planning \
     use_sim:=false authorize_motion:=true
   ```

   `control_mode` 必须是 `moveit_planning`，因为技能目录用这个名字要求控制模式。X2 不运行 MoveIt。

3. 等待日志出现 `Runtime ready: aimdk_robot (..., ACTIVE)` 和 `consumer snapshot: <路径>`。记下快照路径。
4. 等待抓取管线报告后端就绪。后端预热完成前，管线拒绝抓取请求。
   第一次启动时，抓取头编译 cuRobo 内核，需要 3 至 8 min。

## 调用技能

1. 查看技能状态：

   ```bash
   SNAP=<上面日志里的快照路径>
   robot-skill --config-path "$SNAP" status
   robot-skill --config-path "$SNAP" describe pick_object
   ```

2. 只校验，不执行：

   ```bash
   robot-skill --config-path "$SNAP" validate pick_object --target-name "a bottle"
   ```

3. 执行。每次执行前运行第一行，生成一个新的任务 ID：

   ```bash
   TASK_ID="pick-$(date +%Y%m%d-%H%M%S)"
   robot-skill --config-path "$SNAP" execute pick_object --target-name "a bottle" --task-id "$TASK_ID"
   ```

4. 如果要中止，取消同一个任务：

   ```bash
   robot-skill --config-path "$SNAP" cancel --task-id "$TASK_ID"
   ```

Gateway 用任务 ID 识别一次执行，每个任务 ID 只能执行一次。
如果任务 ID 已经用过，Gateway 拒绝执行。请求与上次相同时返回 `DUPLICATE_TASK_ID`，不同时返回 `TASK_ID_CONFLICT`。

如果 `robot-skill` 不在 `PATH` 中，用 `python3 -m robot_skill_cli.cli` 代替。

`--target-name` 是检测模型的文字提示，用英文名词短语，例如 `a bottle`、`a cup`、`an apple`。
技能超时为 120 s。2026-10-23 实测，一次完整抓取用时 24 至 41 s，规划用时 1.6 至 2.0 s。

## 结果

`execute` 的结果包含成功标志、错误码和一份抓取报告。报告列出每个步骤的耗时和每次采集的结果。

| 错误码 | 含义 | 处理 |
| --- | --- | --- |
| `MOTION_NOT_AUTHORIZED` | 启动时没有设置 `authorize_motion:=true` | Gateway 不派发请求 |
| `INVALID_TARGET` | `--target-name` 为空 | 给出目标的文字提示 |
| `NO_EXECUTABLE_CANDIDATE` | 没有得到可执行的抓取。消息写出原因，包括 3 种：没有检测到目标、场景中找不到支撑面、全部候选都规划失败 | 检查物体是否在头部相机视野内。把物体移到左前方 40 至 46 cm 处，与其他物体分开 |
| `RPC_FAILED` | 模型服务或抓取头没有响应 | 检查两台主机上的服务是否在运行 |
| `OBSERVATION_UNAVAILABLE` | 图像时刻的关节反馈缺失，或与图像相差大于 100 ms | 重试 |
| `TARGET_MOVED` | 接近过程中目标移动 | 系统停止手臂，退回，再规划一次 |
| `WATCH_FAILED` | 目标监视无法判断，或抓取头停止上报状态 | 系统让手臂停在原处，不退回 |
| `MOTION_FAILED` | 手臂没有跟随下发的目标 | 检查运行时状态 |
| `EFFECTOR_FAILED` | 左手没有完成张开或闭合 | 检查左手状态 |
| `GRASP_VERIFICATION_FAILED` | 闭合后没有握住物体，或搬运时物体滑落 | 系统放开物体并退回 |
| `SKILL_TIMEOUT` | 技能超过 120 s | 检查第一次请求是否触发了规划器重建 |

## 停止

1. 在机器人侧主机上按 `Ctrl+C` 停止 `embodied_pipeline.launch.py`。
2. 在 Orin 上按 `Ctrl+C` 停止 `x2_grasp_head.launch.py`。
3. 确认两台主机上没有残留的模型服务进程。残留的 GraspGen-X 进程继续占用 NPU 或 GPU。

## 已知限制

- 双主机部署只用于 X2，是临时方案（`aimdk_grasp.split`）。
  如果要把全部后端放在一个进程里运行，选择 `aimdk_grasp.backend:x2_grasp_backends`，并删除 `backend_options.head`。
- 右手不在碰撞模型中。规划时右臂锁定。
- 只用左手抓取。

更多细节见：

- `src/robots/aimdk/aimdk_grasp/README.md`：观测、场景、规划、手部控制和目标监视
- `src/robots/aimdk/aimdk_robot/README.md`：运行时、数据面、网络和时钟要求
- `src/robot_config/config/robots/aimdk_x2_grasp.yaml`：部署配置和各项参数的依据
