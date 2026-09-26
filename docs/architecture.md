# IB-Robot 架构文档

IB-Robot 将 LeRobot 模型与数据流程接入 ROS 2 机器人系统，通过机器人公开接口、观测／动作契约和模型执行抽象，连接具身 Agent、策略推理、遥操作、数据采集与训练。

本文说明模块职责、配置归属与关键数据流。环境准备和启动命令见 [README 运行指南](../README.md#运行指南)，具体参数以各包文档和所选机器人配置为准。

## 目录

- [架构说明](#架构说明)
- [配置与契约](#配置与契约)
- [Robot Runtime 与本体适配](#robot-runtime-与本体适配)
- [具身 Agent 与技能执行](#具身-agent-与技能执行)
- [具身模型统一执行框架](#具身模型统一执行框架)
- [VLA 执行优化](#vla-执行优化)
- [感知、操作与 SLAM 导航](#感知操作与-slam-导航)
- [数据与训练](#数据与训练)
- [启动编排与部署](#启动编排与部署)
- [通信、配置与观测](#通信配置与观测)
- [模块与扩展入口](#模块与扩展入口)

## 架构说明

![IB-Robot 架构图](pictures/ib-robot-architecture.drawio.svg)

架构按职责组织，而不是把所有包排列成固定的进程调用顺序：

- **具身 Agent 与技能**：把外部 Agent 或自然语言请求转为可校验、可确认、可取消的技能执行。
- **数据与训练**：组织遥操作、采集、转换、训练与评估流程。
- **模型执行与 VLA 优化**：统一异构计算资源上的模型执行，并协调策略推理与连续动作执行。
- **感知、操作与空间能力**：提供场景理解、语音、抓取、追踪、SLAM 和导航服务，按应用需要组合。
- **Robot Runtime**：以统一的状态、能力和接口描述连接不同形态的机器人本体。

底部有两条独立的适配路径：机器人本体经 Robot Runtime 接入业务，CPU／GPU／NPU 经模型会话接入统一执行框架。替换机器人不应要求业务层理解厂商控制接口；替换推理设备也不应要求业务层直接调用厂商模型 API。

配置、通信与 Profiling 贯穿这些职责。架构图中的功能块可以跨节点或跨设备部署，不要求同时启动。

## 配置与契约

配置按所描述对象划分归属，同一事实由对应的配置或运行时发布，消费者不再各自维护副本。

| 配置或契约 | 负责内容 | 主要入口 |
| --- | --- | --- |
| 应用 YAML | 选择运行时、声明所需能力、绑定公开接口、定义业务控制模式、观测／动作 Contract、推理 pipeline 和可选子系统 | [robot_config](../src/robot_config/README.md) |
| 机器人 runtime profile | 本体身份、硬件端口、标定、传感器、模式、控制器或厂商 MC 适配参数 | [robot_runtime](../src/robot_runtime/README.md)、各机器人运行时 |
| 运行时接口描述 | 实际公开的端点、消息类型、QoS、坐标系、模型与能力信息 | [公开接口 schema](robot_interface_schema.md) |
| 模型 manifest | 模型语义、产物、命名 deployment、后端 runtime profile 与 tensor 绑定 | [inference_manifest](../src/inference_manifest/README.md) |
| 技能目录快照 | 技能定义、参数、primitive 契约、机器人上下文及目录身份 | [skill_catalog](../src/skill_catalog/README.md)、[skill_library](../src/skill_library/README.md) |

`robot_config` 支持 `base_config` 继承与启动前校验。采用 `runtime.provider` 的配置把本体参数交给机器人 profile，应用只选择运行时并声明绑定关系；旧式无 provider 配置仍有自身的启动路径，不能把其硬件字段直接套用到新运行时。

`RuntimeStatus` 携带运行时状态、能力和公开接口描述。启动编排取得有效接口快照、核对应用需要的能力并完成绑定，随后启动消费者；上层不根据机器人名称猜测话题或控制器名。

观测／动作 Contract 定义语义字段、类型、来源、频率与时间对齐要求。`tensormsg` 负责 ROS 消息与 tensor 编解码，观测缓冲负责按时间戳组织输入。采集、推理与转换共享契约；数据集还保存采集时的契约和单位转换快照，避免离线转换依赖后来修改的机器人配置。

## Robot Runtime 与本体适配

`robot_runtime` 定义公共契约、能力词表、profile 加载、状态与模式接口、停止语义，以及 mock 参考实现和一致性测试。它提供的是统一本体抽象，不承担策略推理或业务任务规划。

机器人适配包实现这些接口，并拥有本体相关的运动控制、传感器接入和厂商依赖：

| 本体路径 | 实现方式 | 业务层看到的边界 |
| --- | --- | --- |
| SO-101 | `so101_robot` 组合 SDK、硬件适配、模型描述、ros2_control 与 `so101_motion` 运动服务 | 公共状态、模式、关节／夹爪接口和运动服务 |
| 灵犀 X2 | `aimdk_robot` 通过 AimDK 桥接厂商 MC、模式与输入源管理；将厂商传感器及本体能力映射为公开接口 | 公共运行时接口与本体声明的能力，不暴露厂商控制细节 |
| LeKiwi 等现有配置 | 按所选 YAML 组合现有硬件、底盘、导航与遥操作模块 | 以该配置实际提供的接口和能力为准 |
| Mock runtime | 内存状态与确定性运动学参考实现 | 用于契约测试及通用业务与本体实现的隔离验证 |

传感器驱动属于机器人适配的实现细节，可以启动本体自带驱动，也可以声明厂商已经发布的传感器接口。应用消费公开的相机、关节、IMU 等数据，不统一接管所有机器人的驱动进程。

模式切换和停止由运行时契约协调，具体执行由控制器或厂商 MC 完成。以 SO-101 为例，profile 声明 `idle`、`stream`、`policy_stream`、`trajectory` 等模式及对应控制器；业务选择匹配的模式，不自行拼装控制器激活集。公共 facade 不传送关节命令，也不实现轨迹插值或实时命令仲裁。

SO-101 的 `simulated:=true` 使用 SDK 模拟传输，复用其控制栈，但不提供物理仿真。旧式 Gazebo／MuJoCo、`hardware_mock`、公共 mock runtime 和 benchmark 是不同用途的路径，不能用一个模拟开关相互替代。

详见 [SO-101 运行时](../src/robots/so101/so101_robot/README.md)、[X2 运行时](../src/robots/aimdk/aimdk_robot/README.md) 与 [公共运行时契约](../src/robot_runtime/README.md)。

## 具身 Agent 与技能执行

具身技能把任务理解与物理执行分开。默认 Hermes 入口通过 `robot-skill` 访问受控 Gateway；`ibrobot_agent` 提供另一条自然语言入口，接入相同的计划与执行边界。

```text
Hermes / 外部 Agent → robot-skill ──────────┐
自然语言 → ibrobot_agent → 交互控制器 ──────┤
                                          ↓
                     计划 → 校验 → 展示与确认 → 执行
                                          ↓
                   技能目录快照 / 安全预检 / 技能执行网关
                                          ↓
                 受限 primitive 或受保护的操作、导航执行器
                                          ↓
                           机器人公开运动与能力接口
```

| 模块 | 职责 |
| --- | --- |
| `robot_skill_cli` | 对 Agent 提供能力发现、状态查询和计划交互的受控 CLI |
| `ibrobot_agent` | 自然语言交互与计划生成入口 |
| `embodied_agent` | 计划生命周期、确认、执行及取消协调 |
| `skill_catalog` | 编译带身份与机器人上下文的技能目录快照 |
| `safety_guard` | 只读预检，不直接下发运动 |
| `skill_library` | 技能执行网关，拆解有限 primitive 或委托受保护执行器 |
| `embodied_bringup` | 组合基础机器人和具身节点，并协调启动就绪条件 |

执行受到目录身份、计划状态、运行时能力和运动授权约束。`authorize_motion` 由操作员在具身启动入口显式提供，Agent 不自行打开授权，也不绕过网关直接向控制器发送命令。该授权控制具身技能链路，不替代本体、遥操作或策略执行各自的准入机制。

技能可调用操作、导航和本体能力；策略动作流则由推理与动作调度链路处理。两者共享机器人公开接口，但不等于所有模型输出都要经过 Agent 计划。

详见 [具身启动](../src/embodied_bringup/README.md)、[robot-skill](../src/robot_skill_cli/README.md) 与 [自然语言 Agent](../src/ibrobot_agent/README.md)。

## 具身模型统一执行框架

统一执行框架以 `ModelSession` 为后端模型资源抽象，由运行时句柄提供请求与生命周期边界，承载策略、感知、语音等模型的执行。

| 层次 | 职责 |
| --- | --- |
| Pipeline / adapter / plugin | 业务输入输出、模型家族预处理和后处理、ROS 服务形状 |
| `ModelRuntimeHandle` | 请求准入、deadline、取消、健康状态与关闭等待 |
| `ModelSession` | 加载、执行和释放后端模型对象、设备资源、buffer 或 worker |
| 命名 deployment | 指定后端、目标设备、产物和 runtime ABI 绑定 |

`ModelRuntimeFactory` 负责按注册信息构造运行时，不是业务请求入口。`inference_service` 提供策略 pipeline、通用 `model_service_node` 和可选多模型调度子系统；感知与语音可使用通用宿主或各自节点，不强制经过策略动作链。

模型 bundle 的 `inference_manifest.json` 使用 schema-v3 描述部署。同一模型可以声明多个 deployment，通过对应产物和后端会话适配不同硬件；统一接口不意味着同一编译产物可以直接运行在所有设备上。仓库的会话实现覆盖 Torch、ONNX、Ascend、Hisilicon、RKNN 和 HMM，实际选择由 manifest 决定。

### 本机策略执行

```text
机器人公开观测 → Contract 编解码与时间对齐 → policy pipeline
  → 模型预处理 → ModelRuntimeHandle → ModelSession → 模型后处理
  → 策略动作块 → action_dispatch → 机器人公开命令接口
```

`action_dispatch` 请求推理并管理动作块、时序平滑、调度与执行器。它不拥有机器人运动规划，也不应直接理解厂商硬件 API。

### 边端协同推理

```text
机器人侧：观测接入 / 请求与会话管理
  → DDS 请求与非图像数据；可选 H.264 RTP/UDP 图像传输
算力侧：观测重建 → 模型处理与执行 → 推理结果
  → DDS 结果与状态
机器人侧：结果校验与动作调度 → 同一套机器人执行接口
```

分布式 pipeline 保留相同的动作执行边界。算力侧运行 `PureInferenceNode`，通过 `CloudBackendRuntime` 执行模型；当前分布式 policy 的模型 processors 由云端持有，不能简单把全部预处理／后处理都归给 Edge。

两端核对 pipeline、bundle、deployment 和契约身份，并使用会话与状态消息协调请求。启动算力侧时，`robot_config_path` 指向该机器可读取的配置，`deployment` 是 manifest 中的命名部署，不是设备自动识别别名。

详见 [推理服务](../src/inference_service/README.md)、[模型部署契约](../src/inference_manifest/README.md) 和 [视频传输](observation_video_streaming.md)。

## VLA 执行优化

这一功能域围绕推理响应、动作块长度和连续运动的平滑性组织策略，由模型执行与动作调度协作完成。

| 策略 | 作用 |
| --- | --- |
| Auto Horizon | 根据观测与注意力信息调整动作块的可执行长度 |
| 时序集成（Temporal Ensemble） | 融合重叠动作，改善连续动作的平滑性 |
| 投机执行（Speculative Execution） | 提前计算候选结果，并在使用前校验 |
| Real-Time Chunking（RTC） | 通过实时动作分块协调推理延迟与连续执行 |

图中将这些策略统一放在“VLA 执行优化”中；代码中的模型推理、`action_dispatch` 和机器人执行器仍分别承担自己的职责。实时性优化不改变机器人模式、停止或技能授权的边界。

配置入口见 [动作分发策略](../src/action_dispatch/README.md) 与 [推理 runtime options](../src/inference_service/README.md)。

## 感知、操作与 SLAM 导航

这些模块提供可组合能力，不构成每个应用都必须依次经过的固定流水线。

| 能力 | 主要模块 | 职责与边界 |
| --- | --- | --- |
| 场景理解 | `perception_service` | 视觉模型调用、目标识别与场景信息服务 |
| 抓取规划与验证 | `manipulation_service` | 生成或验证抓取候选，不直接承担完整物理执行 |
| 闭环操作 | `manipulation_execution` | 抓取、放置、模仿执行，协调感知结果与受保护动作 |
| 任务步骤 | `task_dispatch` | 按任务步骤调用运动、夹爪等公开接口 |
| 语音交互 | `voice_asr_service` / `voice_tts_service` | 语音识别、语音合成及音频交互 |
| SLAM 与导航 | `robot_navigation` 及配置选择的 SLAM／Nav2 组件 | 定位、几何地图、导航命令与底盘适配 |
| 目标追踪 | `object_tracker` | RGB-D 目标追踪及导航跟随 |
| 语义地图 | `semantic_mapping` | 基于 RGB-D 与 TF 的持久化三维语义对象地图，与导航几何地图分工 |

机器人是否可执行抓取、底盘导航或厂商预置动作，由其声明的能力与应用配置决定；配置一个上层节点不会凭空增加本体能力。

## 数据与训练

```text
遥操作设备 → 输入适配与准入 → 机器人执行
                              ↓
                     观测与动作 / Contract
                              ↓
          Episode 录制 → LeRobot 数据集 → 质量处理与训练
                              ↓
                   策略训练 / 蒸馏 / 评估
                              ↓
                  模型导出与转换 → 模型 bundle → 部署
```

遥操作支持 Xbox、同构硬件、手机 IMU、VR／动捕等输入形式。输入设备和机器人输出接口分离；机器人运行时负责本体接入，遥操作模块根据配置将输入映射到公开命令，并遵守准入、反馈新鲜度与 deadman 等约束。

`dataset_tools` 提供 episodic 录制、交互式 `record_cli`、`bag_to_lerobot`、质量分析和评估工具。录制服务和客户端是两个角色，Rerun 可视化只是可选辅助进程，不能代替客户端。数据集保存契约与转换快照，离线转换使用数据集自身的元信息。

训练流程连接 LeRobot 策略训练、蒸馏学习和数据质量权重等优化方法。训练参数、所需版本与补丁条件集中在 [模型训练指南](model_training_guide.md)，架构层不重复维护命令。`model_utils` 负责相关导出与转换工具，发布的 bundle 只保留 manifest 引用的模型产物及元数据；转换中间产物使用 `models/_work/`。

采集与训练不是自动上线链路：训练得到的模型需要完成评估、部署配置和接口核对，再进入机器人推理流程。

详见 [遥操作](../src/robot_teleop/README.md)、[数据集工具](../src/dataset_tools/README.md) 与 [模型工具](../src/model_utils/model_utils/README.md)。

## 启动编排与部署

### 根据职责选择入口

| 场景 | 入口 | 编排职责 |
| --- | --- | --- |
| 独立 SO-101 本体 | `so101_robot runtime.launch.py` | 读取 profile，组合机器人控制、运动服务、公开状态与传感器 |
| 推理、遥操作与采集 | `robot_config robot.launch.py` | 加载应用配置、启动 provider、取得公开接口快照、绑定并启动业务消费者 |
| Hermes／Agent 技能链路 | `embodied_bringup embodied_pipeline.launch.py` | 复用基础机器人启动过程，在就绪后组合计划、安全和技能节点 |
| 分布式算力侧 | `inference_service cloud_inference.launch.py` | 加载模型部署、配置契约和通信端点 |

采用 runtime provider 时，启动就绪判断基于运行时状态与能力核对；旧式路径可能使用控制器就绪检查。`robot_config` 本身不直接启动具身技能节点，仅设置 `with_embodied` 不能代替具身流水线入口。

独立 runtime 和会代为启动该 runtime 的完整应用是两种组合方式，不应对同一本体重复启动。应用控制模式与运行时模式也不同：前者选择业务流程，后者控制本体允许的命令通道，两者通过配置映射。

### 部署边界

Ubuntu／openEuler 源码工作区通过 `.shrc_local` 加载环境，使用 `scripts/build.sh` 构建。OpenHarmony 使用交叉构建后的板端发布包和板端环境入口，不能直接照搬源码工作区命令。

同一系统内的机器人、推理、技能与录制进程需要一致的 ROS Domain 和兼容的 RMW／QoS 配置；各机器路径可以不同，但模型与契约身份必须匹配。具体命令见 [README](../README.md#运行指南) 与 [OpenHarmony 指南](../README.OpenHarmony.md)。

## 通信、配置与观测

图中的 **IBMW · ROS 2 / DDS** 表达通信层。Topic、Service 和 Action 连接观测、模型、任务及机器人能力；`ibrobot_msgs` 定义项目共享的消息、服务和 action，`tensormsg` 处理业务 tensor 与 ROS 数据之间的转换。

图像可通过 `observation_transport` 接入，并按部署选择 H.264 RTP/UDP。DDS 仍承担请求、结果、状态与非图像观测；图像传输不替代业务会话和执行契约。

右侧配置设施分别管理应用组合、runtime profile、模型部署与启动绑定；DDS 可视化配置归属通信配置和管理职责，不替代上述配置来源。

**Profiling** 使用 `ibrobot_tracing` 采集并分析推理、动作与跨节点调用信息。可选 Web 工作台提供浏览器分析界面，需要单独准备依赖和构建。Trace 是观测数据，不改变业务调度；主机调用耗时也不自动等价于 GPU／NPU kernel 的完成时间。

详见 [Tracing Core](../src/ibrobot_tracing/README.md) 与 [Profiling Web](../tools/ibrobot_tracing_web/README.md)。

## 模块与扩展入口

| 扩展任务 | 主要入口 | 需要保持的边界 |
| --- | --- | --- |
| 接入新机器人 | `robot_runtime`、`src/robots/`、应用 YAML | 本体实现公共状态、能力、接口和停止契约，通用包不依赖厂商细节 |
| 接入新模型或硬件后端 | `inference_manifest`、`inference_service` 的 adapter／plugin 与 `model_sessions` | 模型语义处理与设备资源分离，通过命名 deployment 绑定 |
| 新增技能 | `skill_catalog`、`skill_library`、所需操作／导航执行器 | 保留目录身份、预检、授权、确认和取消边界 |
| 新增遥操作输入 | `robot_teleop` 与机器人输入配置 | 输入映射复用公开输出接口，保留设备准入与停止行为 |
| 扩展数据与评估 | `dataset_tools`、`model_utils`、`benchmark` | 复用契约和数据集元信息，不另造本体标定来源 |

各包 README 描述本包接口与职责；新增能力先明确归属，再扩展配置与契约。验证规则见 [开发与测试](testing.md)。
