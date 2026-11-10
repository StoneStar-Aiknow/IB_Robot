# 薄 CLI 与通用 RobotClient 重构设计

状态：首批代码已实现，通过本机集成验收。日期：2026-10-29。
代码基线：`35fb9106`。
范围：机器人发现与连接、在线技能操作、工作流调用、任务查询与取消。
交互入口包括 CLI 和简易 Web 界面。

### 实施记录

本次实现 A、B、C、W 的首批功能。
`robot_client` 已独立于机器人配置和目录编译器。
默认 CLI 已接入发现、连接和在线查询。
通信桥、驻留进程和工作流控制器已迁入公共库。
旧导入路径保留兼容导出。
Web 控制台复用客户端准备、提交和观测任务。

2026-11-02 兼容路径增量（g1-reliability-client 批次）：

- `--config-name/--config-path` 兼容路径的账本与工作流命令
  （`status`、`cancel`、`reload-catalog`、`plan-workflow`、`run-workflow`、
  `validate-plan`、`confirm-plan`、`execute-plan`、`cancel-plan`）改用
  `load_transport_context`：只解析端点映射与超时策略，不再本地编译技能目录；
  状态查询与取消不再被本地目录缺失或损坏阻塞。这些命令的输出与超时行为
  不变（transport-only 上下文与完整上下文的端点和超时策略一致）。
- `closed_loop_runner` 改用 `robot_client.connection.bridge_kwargs` 统一
  端点映射，不再手写枚举（此前会漏掉 prepare/program 等可配置端点）；
  `robot_id` 保持原空值，线格式不变。
- Agent/Hermes 迁移（D 阶段）尚未开始：Hermes 子命令仍由包装器强制
  `--config-path`（兼容路径），`ibrobot_agent` 仍按 launch 参数直连
  RosBridge。二者迁移需要发现/连接工作流变更，留待集成阶段单独实施。

D、E 阶段尚未完成。
CLI 兼容路径仍依赖原有部署包。
Hermes、自然语言 Agent 和视觉游戏尚未全部迁移。
因此，本次不宣称整个 CLI 已完成精简安装。

实际接口见 [robot_client README](../src/agent/robot_client/README.md)。
Web 使用说明见 [Web README](../src/agent/robot_skill_web/README.md)。
后文保留目标设计，未实现的条目不作为完成声明。

### 本机验收结果

- 8 个修改涉及的 ROS 包构建成功。
- 7 个测试包完成 `colcon test` 和结果检查。
- 共 1393 项通过、15 项跳过，错误和失败均为 0。
- 52 个新增或修改的 Python 文件通过 Ruff 检查。
- 隔离 ROS 测试覆盖跨进程发现、连接和状态查询。
- 浏览器测试覆盖登录、发现、连接、预览、提交和取消。

浏览器测试还覆盖刷新恢复和 390 px 移动布局。
Web 重启恢复测试使用持久引用，不重复发送动作。
所有执行测试使用 mock，未操作真实机器人。
双平台 Docker 和真实机器人验收尚未执行。

## 1. 设计结论

采用「薄命令行接口（CLI）＋公共客户端库＋现有服务端」。
新增 `robot_client` 库，不新增执行决策服务。
现有 daemon 只复用通信连接。
Gateway 继续持有执行准入和任务账本。
计划服务继续负责计划准备与校验。

重构目标是删除客户端对机器人部署资源的依赖。
目标不是把 `cli.py` 原样搬到另一个目录。
客户端只认识公开技能名、参数契约和执行引用。
厂商动作 ID、关节配置和技能编译留在服务端。
默认连接流程改为「发现机器人 → 选择 → 连接」。
操作员不填写连接 YAML、服务名或机器人 profile。

首阶段保留现有 ROS 2 协议。
客户端仍需 ROS 2 运行环境和消息包。
浏览器通过本地 HTTP 适配层访问 `RobotClient`。
简易 Web 界面纳入本次重构范围。
浏览器不直接连接 ROS 2。
跨网络部署和公网访问仍属于独立设计。

## 2. 重构前代码与实际约束

本节记录代码基线的行为，不描述重构后的默认入口。

| 位置 | 当前行为 | 重构含义 |
| --- | --- | --- |
| [cli.py](../src/agent/robot_skill_cli/robot_skill_cli/cli.py) 的 `main()` | `list-skills` 使用本地目录。`status` 进入本地运行配置加载 | 查询尚未脱离机器人配置 |
| [catalog.py](../src/agent/robot_skill_cli/robot_skill_cli/catalog.py) 的 `load_runtime_context()` | 加载 YAML，再调用 `compile_local_snapshot()` | 任务查询与取消也会受本地资源影响 |
| `catalog.py` 的 `capability_view_from_snapshot()` | 校验 3 类摘要，再投影公开字段 | 可复用算法，但需拆除编译器导入 |
| [bridge_daemon.py](../src/agent/robot_skill_cli/robot_skill_cli/bridge_daemon.py) | 启动时导入 `cli.bridge_kwargs` 和 `load_runtime_context` | 通信层反向依赖 CLI 和目录编译 |
| [interactive_control.py](../src/agent/robot_skill_cli/robot_skill_cli/interactive_control.py) | 编排发现、准备、显示、确认、执行与停止 | 存在可复用流程，但不能直接改名为公共库 |
| [ibrobot_agent/node.py](../src/agent/ibrobot_agent/ibrobot_agent/node.py) | 包含 Program 提交和终态查询流程 | 后续应复用客户端，保留 Agent 策略 |
| [package.xml](../src/agent/robot_skill_cli/package.xml) | 直接依赖 `robot_config`、`skill_catalog`、`embodied_agent` | 仅改函数调用不足以完成依赖解耦 |

### 2.1 两种执行语义必须分开

[ExecuteProgram](../src/ibrobot_msgs/action/ExecuteProgram.action) 是完整请求提交接口。
Gateway 收到请求后执行准入、持久化和调度。
接口明确规定：显示计划不是服务端执行门槛。

CLI 的 `run-workflow` 具有额外的交互约束。
CLI 先显示计划并刷新输出，再请求执行。
`InteractiveController` 当前仍调用计划确认接口。
随后，支持 Program 的 bridge 发送 `ExecuteProgram`。
Program 请求没有 `confirmation_token` 字段。

因此，首阶段保留 CLI 的先显示后提交顺序。
首阶段也保留生产 Agent 的直接 Program 路径。
禁止把本地显示回执描述成服务端授权凭证。
若要求所有客户端强制预览，必须另改服务端协议。
本次重构不包含这项语义变更。

### 2.2 参数扩展存在协议边界

[WorkflowStep](../src/ibrobot_msgs/msg/WorkflowStep.msg) 使用固定字段。
现有 CLI 也使用固定参数选项。
首阶段支持现有参数契约内的新技能名。
例如，增加 G1 预置动作不要求客户端认识厂商 ID。
全新参数类型仍需协议版本和客户端支持。
本设计不以通用 JSON 参数掩盖这项限制。

## 3. 职责与依赖方向

```text
robot-skill CLI       Hermes adapter       ibrobot_agent / Web backend
       |                    |                         |
       +--------------------+-------------------------+
                            |
                       RobotClient
             connection / catalog / workflow / execution
                            |
                       Transport API
                   /                  \
          in-process RosBridge    resident bridge proxy
                                        |
                                  existing daemon
                   \                  /
                       ROS 2 interfaces
                            |
              plan service + Capability Gateway
                            |
                 protected executor -> runtime -> SDK
```

| 层 | 保留的职责 | 禁止承担的职责 |
| --- | --- | --- |
| CLI | 参数语法、目标选择、输出、信号、退出码 | 编译目录、厂商映射、任务准入 |
| Web 界面与 HTTP 适配层 | 表单、预览、会话、结果显示 | 复制客户端流程、直连 SDK、授予运动权限 |
| `RobotClient` | 目标绑定、目录投影、版本核对、流程编排、任务观测 | 运动授权、任务调度、权威任务状态 |
| transport / daemon | 目标绑定、消息转换、连接复用、传输错误 | 技能选择、自动重放、目录编译、凭证管理 |
| 计划服务 | 固定计划内容、计划校验、现有确认流程 | 取代 Gateway 实时准入 |
| Gateway | 幂等、参数复核、授权、并发、账本、执行准入 | 信任客户端预检来跳过复核 |
| runtime | 厂商协议、动作映射、物理状态与停止能力 | 解释终端参数或 Agent 提示词 |

新增公共库包：`src/agent/robot_client/`。
包内按职责拆模块，不首批拆成多个传输包。
Web 入口独立放在 `src/agent/robot_skill_web/`。
Web 包依赖公共库，公共库不依赖 Web 包。

```text
robot_client/
  client.py               # Bound facade: previews, submissions, ledger
  transport.py            # Neutral GatewayTransport and endpoint identity
  connection.py           # ROS scope, private JSON state, bridge parameters
  models.py               # Immutable public values
  errors.py               # Typed errors, independent of CLI exit codes
  catalog.py              # Versioned capability queries and projection
  workflow.py             # Prepare, presentation barrier, submit
  execution.py            # Query, events, cancellation, reconciliation
  transports/
    protocol.py           # Transport interface
    ros_bridge.py         # Existing ROS transport implementation
    resident.py           # Existing daemon client and lifecycle
```

公共接口不返回 ROS goal handle、future 或 `argparse.Namespace`。
核心模块不导入 `rclpy`，ROS 实现在 transport 内。
首阶段安装包仍声明 ROS 依赖，不宣称纯 Python 独立部署。
`robot_client` 禁止依赖 CLI、机器人配置或厂商包。

规范化 JSON 已以 `embodied_common.canon` 为准。
快照摘要算法移入 `embodied_common` 的独立模块。
原 `skill_catalog.digest` 保留兼容导出。
迁移必须保持规范化字节和摘要结果一致。
禁止客户端复制一份算法后独立演化。
禁止最终在线依赖树引入目录编译器或技能资源。

当前 `embodied_common` 仍声明图像和推理描述依赖。
首阶段不拆这个公共包，也不承诺最小安装体积。
本次验收重点是删除本体资源和编译器依赖。
若安装体积成为要求，再单独拆分基础契约包。

## 4. 目标绑定（原「发现与连接」已移除）

2026-11 决定：从公开客户端移除 `discover` / `connect` / `disconnect`、`--target`、
无配置在线分发、选择与凭证持久化，以及服务端 `GetGatewayConnectionInfo` 接口。
删除提交的覆盖面去向见 Issue #238；跨机器最小 descriptor 设计也留在该 Issue，
不在本设计扩写。

### 4.1 CLI 与 Web 的绑定规则

CLI 与 Web 使用同一套配置解析顺序：
`--config-path` > `--config-name` > `ROBOT_CONFIG` > `ROBOT_NAME`。
四者都没有时以退出码 2 报错，不猜测默认机器人，也不连接任何 Gateway。
Web 启动只读取本地配置，不扫描 ROS 图、不连接 Gateway、不发送动作。

```bash
robot-skill --config-name unitree_g1 status
ROBOT_CONFIG=unitree_g1 robot-skill list-skills
ros2 run robot_skill_web robot-skill-web --config-name unitree_g1
```

配置解析后，应用边界把已加载的 `embodied` 映射交给
`robot_client.gateway_transport()`，得到中立的 `GatewayTransport`：
规范化 endpoint 映射、`robot_id`/`robot_name` 和 `caller_scope`。
`robot_client` 不读取配置文件，也不依赖 `robot_config`、`skill_catalog` 或 CLI。

### 4.2 目标身份

`target_key` 由规范化 endpoint 映射、机器人身份、调用方作用域和当前 ROS 环境
（`ROS_DOMAIN_ID`、`RMW_IMPLEMENTATION`、local-only/安全/profile 等变量）共同派生，
沿用既有 canonical JSON 与 SHA-256 规则：

- 同一配置文件内容变化导致 endpoint 变化时，目标不同。
- 不同配置文件解析出同一 endpoint 集时，目标相同（路径只作为来源提示）。
- 同名机器人在不同 ROS domain/安全配置下是不同目标。

该摘要只用于路由与引用校验，不是认证，也不能替代 Gateway 的 `authorize_motion`。
浏览器不接收 endpoint 明细，只接收公开目标标识。
执行引用携带 `target_key`、`robot_id` 和 `caller_scope`；跨目标引用在客户端和 Web
后端都被拒绝，不会发到 Gateway。

### 4.3 预览新鲜度与恢复

`prepare_workflow()` 读取 Gateway 业务状态并记录 `robot_name`、`registry_epoch`、
`registry_generation`、`registry_digest` 和 `capability_digest`。
`registry_epoch` 每次 Gateway 进程启动都会变化；目录 reload/重编会改变 generation
与摘要。`submit_prepared()` 在展示前重新读取这些字段，任一变化即作废旧预览。
计划本体保存在 Gateway 进程内，重启后自然无法确认；预览校验与计划令牌互为补充。

daemon 预热只接受 `--config-name` / `--config-path`，与 CLI 使用同一解析规则。
旧的部署导出连接描述文件输入已移除；需要跨机描述时按 Issue #238 讨论。

### 4.4 边界

ROS domain 仍是通信边界：目标身份包含 ROS 环境，但 `ROS_DOMAIN_ID` 不是认证。
部署侧负责 domain、中间件与安全配置；客户端不扫描其他 domain，也不改运行中的 ROS context。

## 5. 公共客户端接口

下列示例对应首批公开接口。

```python
transport = gateway_transport(embodied_config, robot_id="g1", robot_name="g1")
client = RobotClient(transport)
client.get_status()
client.list_skills()
client.describe_skill(name)
client.list_poses()
prepared = client.prepare_workflow(request)
ref = client.submit_prepared(
    prepared["preview_id"], presenter=..., persist_reference=...
)
client.get_execution(ref)
client.get_execution_events(ref, after_sequence=...)
client.cancel_execution(ref)
client.wait_execution(ref, deadline=...)
client.close()
```

构造时绑定唯一目标，不要求所有服务同时就绪。
`get_status()` 不要求计划服务和目录服务可用。
任务查询与取消也独立探测各自端点。
这样，目录故障不会阻断已有任务的停止请求。

客户端以字典副本返回预览，内部保存准确计划。
预览携带步骤、计划身份、Gateway 业务身份和预算。
`submit_prepared()` 检查预览属于当前连接会话。
`presenter` 成功返回代表输出已完成并刷新。
`persist_reference` 必须在发送前完成持久化。
这项回执只约束调用顺序，不授予运动权限。

`ExecutionRef` 明确区分标识符：

| 字段 | 含义 |
| --- | --- |
| `target_key` | ROS 环境范围与规范 endpoint/机器人摘要，不是授权凭证 |
| `robot_id`、`caller_scope` | 服务端查询作用域 |
| `request_id` | Program 幂等键，提交前生成并保存 |
| `execution_id` | Gateway 分配的执行 ID，回执前允许缺失 |
| `legacy_task_id` | 仅供旧 action 适配器使用 |

禁止把旧 `task_id` 猜成 Program 的 `request_id`。
当前 CLI 工作流确实生成不同的两类 ID。
兼容层保存明确映射，新接口直接返回 `ExecutionRef`。
调用方可保存执行引用，文件中不含 token。
跨进程恢复先重新认证，再查询服务端账本。
本地引用只定位执行，不恢复任务状态。

同步门面供 CLI 使用。
事件和停止通道支持独立消费。
事件回调不能占用唯一的取消通信通道。
接入已有 ROS executor 的 Agent 时，允许注入 transport。
客户端只关闭自己创建的节点、线程和 context。

## 6. 在线目录与命令行为

### 6.1 首阶段复用快照

在线 `list-skills` 和 `describe` 改为读取服务端目录。
查询步骤为 status、指定版本快照、摘要校验、公开投影。
保留现有 3 类摘要核对，不静默降级。
快照适配器只读取结构数据，不编译技能。

缓存键包含连接身份和 registry 三元组。
三元组为 `epoch`、`generation`、`digest`。
同一三元组内容变化时返回摘要错误。
禁止缓存保存 token、内部模板或厂商配置文件。
首阶段完整快照仅在校验时作为临时数据存在。
实时 ready 状态不与静态目录共用有效期。

纯查询遇到版本竞争时，最多重新读取 1 次。
第 2 次仍冲突时返回明确的版本错误。
准备完成后的计划不自动替换为新版本。
服务端不可达时，默认查询返回不可达错误。
禁止把旧缓存显示成当前可执行能力。

| 命令 | 新在线路径 | 是否依赖完整目录 |
| --- | --- | --- |
| `status` | 直接查询 Gateway 状态 | 否 |
| 新 `execution-status` / `execution-events` | 按执行引用查询账本 | 否 |
| 新 `cancel-execution` | 按执行引用取消 | 否 |
| 原 `cancel` / `cancel-plan` | 保留旧 action 标识语义 | 否 |
| `list-skills` / `describe` / `list-poses` | 服务端快照投影 | 是，首阶段 |
| `validate` / `execute` | 公共客户端准备请求，服务端复核 | 是，首阶段 |
| `run-workflow` | 公共客户端编排计划和执行 | 是，兼容阶段 |
| `reload-catalog` | 显式转发管理请求 | 否，不在本地重编译 |

离线检查改为显式开发命令。
开发命令继续使用 `skill_catalog` 编译器。
离线输出必须标记「未连接运行中的 Gateway」。
在线命令故障时禁止自动切换到离线模式。
保留现有 JSON envelope、退出码和终端显示规则。
目录来源变化必须写入迁移说明。

迁移期间新增显式 `--offline` 过渡选项。
精简部署不包含离线依赖，选用时返回安装提示。
开发部署通过独立离线入口提供编译功能。

原 `execute` 首阶段仍发送单技能 action。
禁止静默改成 Program 后复用原 `task_id`。
两类执行通过不同适配器共享连接和错误处理。

### 6.2 公共能力协议作为后续阶段

拟新增 `GetGatewayCapabilities`，不修改内部快照语义。
响应包括服务端身份、协议版本和功能标识。
公开目录包括技能名、说明、参数、超时上限和语义级别。
目录同时携带 registry 身份和公开内容摘要。
客户端只计算公开摘要，不重建内部 registry 摘要。
两类摘要必须有不同字段名和规范化规则。

响应排除 primitive 模板、厂商 ID、关节表和文件路径。
动态 ready 状态继续从 status 读取。
客户端把能力就绪显示为观测值，而非执行承诺。
未知协议主版本拒绝执行，不能按旧结构猜测。

首阶段按连接信息选择已知协议适配器。
旧 Gateway 兼容路径使用导出的连接描述。
禁止用 Python `hasattr` 判断远端服务能力。
连接功能标识由 A 阶段接口提供。
E 阶段再增加公开能力目录的版本协商。
`GetSkillSnapshot` 继续用于服务端内部同步。

## 7. 工作流、取消与不确定结果

### 7.1 保留 CLI 的交互顺序

```text
prepare -> verify exact plan -> display + flush
        -> check stop -> existing validate/confirm adapter
        -> persist ExecutionRef -> submit once
        -> observe events / query ledger -> terminal or observation unknown
```

首阶段复用现有 `PrepareAgentPlan` 和确认接口。
兼容适配器保留现有版本、步骤与预算核对。
首阶段不新增第二个服务端计划存储。
后续公共协议稳定后，再简化确认步骤。

显示失败、输出刷新失败或提交前停止时，不发送 goal。
提交引用保存失败时，也不发送 goal。
提交后发生的输出失败不能证明执行失败。
客户端必须按执行引用查询服务端。

计划过期或 registry 变化时，返回版本错误。
调用方重新准备后，必须重新显示实际计划。
禁止自动替换已显示计划后继续执行。
服务端在提交时仍独立复核参数、版本和实时准入。

### 7.2 任务状态只以服务端为准

客户端的「观测未知」与账本 `UNKNOWN` 分开表达。
网络超时只说明客户端缺少证据。
网络超时不能直接把账本状态改成 `UNKNOWN`。
账本 `UNKNOWN` 的隔离与运维对账保持现有规则。

| 故障或事件 | 客户端行为 |
| --- | --- |
| 发送前确认失败 | 返回本地错误，不发送 goal |
| goal 回执超时 | 保留幂等键，查询账本，不自动重发 |
| `GetExecution` 暂未找到 | 保持观测未知，不推断未执行 |
| 相同幂等键、不同内容 | 透传服务端冲突，禁止换键绕过 |
| daemon 在提交后断开 | 新连接只查询或取消，不重放 goal |
| 取消返回 `accepted` | 显示取消已请求，继续查询终态 |
| runtime 不支持物理取消 | 透传限制，禁止声称机器人已停止 |
| 事件重复或序号缺口 | 按执行 ID 和序号去重，再用账本补齐 |
| Gateway 重启 | 废弃旧准备态，使用持久化账本查询已有执行 |
| 查询旧执行时 registry 已更新 | 不因目录版本阻断查询或取消 |

库错误包含 `code`、阶段、执行引用和观测确定性。
库保留服务端错误码，不自行解释为 CLI 退出码。
CLI 在边缘映射现有退出码。
重试策略只允许有界的只读查询重试。
取消重试沿用服务端单调停止语义。

### 7.3 daemon 与权限

daemon 迁入 transport 层，保留原启动命令兼容入口。
daemon 接收已核对的连接描述，不拉取目录。
保留 socket 用户隔离、对端 UID 核对和独立发送队列。
保留 ROS 环境与代码版本参与连接分组的规则。
调用方凭证按会话传递，禁止跨会话缓存复用。
daemon 不保存权威任务状态或授予调用方权限。

Hermes 的受控入口限制保留在 Hermes 适配器。
自然语言规划、提示词和语音策略不进入公共库。
公共库不提供 primitive 的通用执行入口。
库只暴露公共技能与 Program 接口。

## 8. 迁移阶段与文件改动

| 阶段 | 主要改动 | 完成标准 |
| --- | --- | --- |
| A：目标绑定（原发现与连接，2026-11 移除） | 删除发现接口、候选选择、选择/凭证持久化与 `--target`；CLI/Web 改为启动时解析部署配置并绑定中性 transport | 无配置时明确报错；状态、查询、取消不加载目录；覆盖面归档见 Issue #238 |
| B：在线目录 | 拆分摘要算法。增加快照投影。CLI 查询改用服务端。隔离离线工具 | 删除本地技能资源后仍可列举、描述和执行现有技能 |
| C：统一调用流程 | 迁移 `InteractiveController` 的公共流程。封装 Program 查询、取消和事件 | CLI 与第二个调用方共用客户端，原交互顺序不变 |
| W：简易 Web | 单页工作台与 HTTP 适配层。启动绑定部署配置，复用公共客户端 | 浏览器建立会话、预览、执行、查询和取消，不发现或切换目标 |
| D：收敛依赖 | 迁移 Hermes 与 Agent 适配层。清理包依赖。补齐在线视觉游戏目录 | 精简客户端不安装机器人配置、编译器或技能包 |
| E：公共协议 | 增加公开能力接口和特性协商。删除正常查询的内部快照依赖 | 升级目录实现不要求升级客户端 |

A 至 D 完成后，才宣称「在线 CLI 已变薄」。
E 是后续优化，不阻塞先删除本地资源依赖。
每个阶段保持可运行，禁止一次性替换所有入口。
W 依赖 C，作为公共库第二个调用方的验收入口。
W 不等待 E，也不依赖 Hermes 或自然语言 Agent。

旧导入路径保留兼容导出，再逐个迁移消费者。
`robot_skill_cli.ros_bridge` 暂时导出新实现。
`robot_skill_cli.interactive_control` 暂时提供旧接口适配。
原 `output.py` 和 `human_output.py` 留在 CLI。
禁止新库反向导入这两个输出模块。

`list-games` 和 `describe-game` 当前也读取本地配置。
首阶段明确保留这部分兼容路径。
D 阶段补充服务端公开游戏目录和参数契约。
游戏启动与结果查询继续使用现有独立接口。
游戏目录不完整时，禁止宣称整个 CLI 无配置依赖。

依赖声明变化会触发仓库双平台验证门禁。
执行 Docker 前，必须确认 PR 是 WIP 还是正式检视。
本次执行本机构建和隔离 ROS 测试。
本次不运行 Docker 或真机动作。

## 9. 验收与回归矩阵

| 类别 | 必测场景 | 通过条件 |
| --- | --- | --- |
| 依赖隔离 | 精简安装删除机器人 YAML、profile、技能目录 | 在线状态、目录、执行、查询和取消仍可用 |
| 导入隔离 | 禁止导入 `robot_config`、编译器和 CLI | `robot_client` 的在线路径正常运行 |
| 故障隔离 | 快照或计划服务不可达 | status、账本查询与取消仍可用 |
| 目标绑定 | 无配置、`--config-name`、`--config-path`、`ROBOT_CONFIG` 四种入口 | 前三者按优先级解析；无配置 exit 2；可查询状态 |
| 重名与冲突 | 同名机器人不同 endpoint 或不同 ROS 域 | 视为不同目标；引用与历史不跨目标串用 |
| 移除旧端点 | 旧页面或脚本调用发现/选择接口 | `/api/robots`、`/api/connections` 返回 404，不退化盲连 |
| 目标绑定 | 默认目标变化、旧标签页或 Gateway 重启 | 不串用计划，不把取消发送到其他实例 |
| 凭证保护 | 会话建立提交旧登录字段 | 400 拒绝；会话只读本地绑定；浏览器不能覆盖作用域 |
| 目录权威 | 本地离线目录与服务端不同 | 在线结果只反映服务端 |
| 动态技能 | mock 新增 G1 预置动作 | 客户端无源码修改，列举并提交正确技能名 |
| 多部署 | 同一入口指向 G1、SO-101、LeKiwi、X2 配置 | 无厂商分支；换部署需换配置；无任务跨目标取消 |
| 精确版本 | 准备后 reload、重启或回收快照 | 原计划拒绝执行，要求重新准备与显示 |
| 显示顺序 | 输出抛错、flush 失败、显示时停止 | goal 发送次数为 0 |
| 生产 Agent | 直接 Program 提交 | 不意外增加 CLI 交互门槛 |
| 重复提交 | 同键同内容与同键异内容 | 账本去重或冲突，物理执行次数不增加 |
| 不确定结果 | goal 回执丢失、daemon 中断 | 无自动重放，可按引用查询与取消 |
| 权限隔离 | 错误 scope、token 或 robot ID | 服务端拒绝，无跨用户查询或取消 |
| 任务标识 | legacy task ID 与 Program request ID 不同 | 适配器准确定位执行，不猜测 ID |
| 取消能力 | 取消已接受但动作未结束 | 不输出已停止或成功终态 |
| 事件恢复 | 重复、丢失、乱序与过期事件游标 | 去重补齐，无法补齐时明确标记缺口 |
| 输出兼容 | TTY、管道、Hermes managed、显式 JSON | envelope、退出码和完整标识符保持兼容 |
| 生命周期 | Agent 注入已有 ROS executor | 关闭客户端不关闭调用方 executor |
| 协议边界 | 不支持的新参数或协议主版本 | 明确拒绝，不忽略字段或猜测执行 |

纯逻辑测试覆盖转换、标识、状态机和错误映射。
ROS 集成测试使用隔离 domain 和 mock Gateway。
正式包验收入口为 `colcon test`。
测试必须确认进程正常退出，超时不计通过。
新增测试依赖遵循包声明，禁止形成依赖环。
现有 CLI 测试结果不能代替新库验收。

## 10. 简易 Web 界面

### 10.1 首版范围

首版面向单个操作员，同一时刻绑定 1 台机器人。
启动时用 `--config-name` / `--config-path` / `ROBOT_CONFIG` 指定部署配置，
页面不再发现、选择或切换目标。
会话建立后直接进入工作台，页面固定显示目标身份。
换目标需要重启 Web 并传入另一个配置。

首版只提供 5 项功能：

- 建立浏览器会话（目标固定）。
- 查看连接、授权、控制模式、忙碌状态和更新时间。
- 搜索技能，查看说明、就绪原因和验证状态。
- 填写参数，生成并查看准确执行计划。
- 请求执行，查看进度、执行引用和终态。
- 请求取消，继续观察服务端结果。

首版只执行单技能任务，使用单步骤 Program。
技能列表只开放 `planner_visible` 的高层技能。
参数表单只支持客户端已识别的公开参数类型。
未知类型显示「当前界面不支持」，执行按钮禁用。
G1 未验证动作始终保留「未验证」标记。
技能 ready 不代表已完成真机验证。

首版不包含聊天、视频、遥操作和工作流编辑器。
首版也不提供目录重载、运动授权开关和运维对账。

### 10.2 页面布局

使用「等待会话」和工作台两个状态，不增加多级菜单。
等待态只显示已绑定的目标名称，不提供列表或连接按钮。
会话建立后显示目标、授权和忙碌状态。

桌面端左右布局，窄屏按技能、计划、结果顺序堆叠。
目标身份与任务取消入口始终可见。
页面以文本标签区分状态，不只依靠颜色。

```text
+--------------------------------------------------------------+
| Robot: G1   Connected   Authorization: OFF   Updated: ...     |
+--------------------------+-----------------------------------+
| Search skills            | Selected skill                    |
|                          | Description / readiness / warning |
| Skill A     Ready        | Parameters: [...]                 |
| Skill B     Not ready    | [Preview plan]                    |
| Skill C     Unverified   |                                   |
|                          | Plan: target / steps / parameters |
|                          | Registry version / budget         |
|                          | [Execute this plan]               |
+--------------------------+-----------------------------------+
| Execution: ...   State: ...   Progress: ...                   |
| [Request cancellation]       Result / error / event details  |
+--------------------------------------------------------------+
```

「预览计划」只准备计划，不提交运动。
计划区域完成渲染后，才允许点击「执行此计划」。
修改技能或参数后，立即废弃原预览。
执行期间禁止重复点击提交，取消入口保持可用。
断线时显示最后观测时间，不把旧数据标成在线。

取消按钮标注「请求取消」，不使用「急停」名称。
界面不能替代机器人硬件急停。

### 10.3 部署与接口

```text
Browser: bundled HTML / CSS / JavaScript
    -> same-origin HTTP adapter: robot_skill_web
    -> RobotClient
    -> existing ROS transport / daemon
    -> existing plan service + Gateway
```

前端使用静态资源，不引入前端框架或构建链。
Python HTTP 适配层与静态资源同源部署。
适配层直接调用库，禁止通过 subprocess 执行 CLI。
适配层只保存会话和提交关联记录。
执行状态仍从 Gateway 账本获取。

默认入口为 `robot-skill-web`。
首批 Web 入口只支持启动配置绑定，不提供候选选择或连接描述文件。
首版只绑定 `127.0.0.1`，不提供公网监听选项。
以下接口已接入本地 HTTP 适配层。

| HTTP 接口 | 行为 |
| --- | --- |
| `POST /api/session` | 把浏览器会话绑定到启动配置解析出的唯一目标（空对象） |
| `GET /api/status` | 读取目标和状态，不加载技能目录 |
| `GET /api/skills` | 读取在线公开目录和参数描述 |
| `POST /api/previews` | 准备单步骤计划，返回预览引用和有效期 |
| `POST /api/executions` | 按预览引用提交一次执行 |
| `GET /api/executions/{ref}` | 查询执行与新增事件，不依赖技能目录 |
| `POST /api/executions/{ref}/cancel` | 请求取消，不等待物理动作结束 |

预览引用绑定会话、目标摘要、准确计划、Gateway 业务身份、预算与请求 ID。
执行请求只接受预览引用，不接受替换后的步骤。
HTTP 适配层把点击执行转换为客户端的预览完成回执。
回执仅约束本 Web 入口，不作为 Gateway 授权凭证。
计划过期或版本变化时，界面要求重新预览。

提交接口异步返回执行引用，不等待任务终态。
HTTP 的 `202` 仅表示接收请求，不表示 Gateway 已准入。
任务等待与取消使用独立工作通道。

首版通过轮询读取状态和任务结果，不引入 WebSocket。
拟议默认周期：页面状态 2 秒，活动任务 1 秒。
每类轮询最多保留 1 个在途请求。
查询失败时有界退避，界面保留明确的断线提示。
轮询只观测任务，不驱动服务端执行。

### 10.4 会话、安全与恢复

会话 Cookie 使用 `HttpOnly` 和 `SameSite=Strict`。
适配层限制 Host，并核对变更请求的 Origin。
变更请求还必须通过跨站请求伪造防护校验。
接口禁止跨域访问，未认证请求不能读取机器人数据。
`POST /api/session` 只接受空对象，旧登录/凭证字段被拒绝；
会话建立被拒绝达到 5 次后限制 60 秒。
只保留 1 个有效浏览器会话，替换会话不改变目标或任务状态。

Gateway token 只保存在后端进程中。
目标只来自启动时的部署配置；浏览器不能指定候选、端点、scope 或 token。
`GET /api/session` 只返回公开目标标识和未确认任务引用，不返回 endpoint 明细。
调用方作用域由部署配置决定。
Gateway 沿用原有权限检查，不因 Web 会话而放宽。
技能说明按纯文本渲染，禁止执行目录返回的 HTML。

适配层在发送 goal 前持久化执行引用与请求 ID。
适配层用持久化的原子提交标记阻止并发发送。
同一预览的并发提交复用原请求 ID。
HTTP 重复请求只返回关联引用，不再次发送 goal。
提交结果不明时，适配层按原引用查询账本。

关联记录不是任务账本，不保存权威执行状态。
关联记录不保存凭证，并限制为服务进程用户可读。

刷新页面后，界面恢复当前会话的执行引用并查询。
后台重启后，操作员重新建立会话，再查询已保存的引用。
后台重启使旧预览失效，但不重放已提交任务。
其他部署未确认的任务会阻止当前部署建立会话，必须先核对。
关闭页面或会话过期不会隐式取消任务。
界面必须提前说明这项行为。
重新登录后仍可查询和请求取消已有任务。

### 10.5 Web 验收

| 场景 | 通过条件 |
| --- | --- |
| 精简客户端安装 | 页面完成状态、技能、预览、执行与取消流程 |
| 无配置启动 | 退出码 2，不启动服务，不连接任何网关 |
| 发现/选择接口 | `/api/robots`、`/api/connections` 返回 404 |
| 默认启动 | 用部署配置启动，页面建立会话后直接进入工作台 |
| 目标固定 | 旧标签页与旧预览不能提交到其他部署；换目标需重启 Web |
| 活动或未知任务 | 禁止新准备与提交，保留原执行引用与取消入口 |
| 只打开页面或预览 | goal 发送次数为 0 |
| 参数变化或计划过期 | 原预览不能提交 |
| 双击、双标签页或 HTTP 请求重复 | 同一预览最多发送 1 次 goal |
| HTTP 回执丢失或页面刷新 | 使用原引用查询，不重复执行 |
| Web 后台重启 | 原任务可查询，旧预览失效，无自动重放 |
| 任务观测超时 | 取消接口仍可响应，不伪造终态 |
| 未登录、跨域或错误会话 | 不泄露状态，不发送运动或取消请求；旧登录字段返回 400 |
| 技能说明含 HTML 或脚本 | 页面只显示文本，无脚本执行 |
| 窄屏与键盘操作 | 表单、计划与取消入口均可操作 |

接口测试与浏览器测试使用 mock Gateway。
正式包测试通过 `colcon test` 注册并执行。
浏览器测试额外核对渲染、按钮状态和重复提交行为。
本次文档更新不运行 Web 服务或真机动作。

## 11. 本次范围之外

- 不改厂商 SDK、G1 动作映射或物理取消能力。
- 不增加拥有执行决策权的 daemon。
- 保留 Gateway 的幂等、授权、并发和隔离策略。
- 不自动恢复或重放结果未知的运动。
- 不把客户端缓存变成权威目录或执行账本。
- 不把自然语言规划器迁入公共客户端。
- 不新增任意 JSON 到厂商动作的透传接口。
- 不以 CLI 重构名义改写生产 Program 的执行门槛。
- 不提供公网 Web、多用户权限管理或浏览器遥操作。

建议先实施 A 和 B，验证依赖删除确实生效。
随后实施 C 和 W，用简易 Web 验证公共库复用。
再实施 D，完成其余调用方迁移和部署隔离。
公共能力协议单独评审，避免扩大首批变更范围。
