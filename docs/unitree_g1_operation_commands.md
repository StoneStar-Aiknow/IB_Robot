# 宇树 G1 启动与操作命令

归档日期：2026-10-30。

本页归档本次 Ubuntu 工作区的真机预置动作流程。
操作入口包括 Web 和命令行界面（CLI）。
调用顺序为操作入口 → Gateway → G1 runtime → 官方软件开发工具包（SDK）。
本页不是新机器安装指南。

清理状态：2026-10-30，按操作员要求删除了临时部署目录。
旧配置和原始排障记录均已删除。
本页已撤下依赖旧目录的 Gateway 和 CLI 启动命令。
再次启动真机前，必须重新配置并验证部署。

## 1. 适用范围与前置条件

- 工作区已构建，`.shrc_local` 和 `install/` 可用。
- 已准备并验证新的真机配置。
- 机器人已开机，并处于运控模式。
- 本次部署使用网卡 `eno1`，预期 FSM 值为 `802`。
- FSM 指厂商运控有限状态机，不是 Gateway 控制模式。
- 操作员在现场，已确认活动空间和物理停止手段。
- 本页仅覆盖预置动作，不覆盖移动、手臂流式控制或 Revo2 手部控制。

警告：预置动作依赖运控模式。调试模式不适用于本页。

警告：Web 的「请求取消」不是硬件急停。
关闭页面或退出进程，也不代表机器人已停止。

1. 确认本机没有重复运行的 G1 部署。
2. 打开 2 个新终端。
3. 在终端 A 启动 Gateway。
4. 在终端 B 启动 Web。

## 2. 已清理的临时部署

以下目录已删除，仅保留路径用于识别历史会话：

```text
/home/lhj/orca/tmp-sdk-verify/web-authorized-g1-20261029-tiklbnw2
```

删除范围包括以下内容：

- `operator-robot.yaml` 和 `operator-profile.yaml` 临时配置。
- `execution-secret.env` 旧执行密钥。
- `config/ibrobot/credentials.json` 旧 CLI 凭证绑定。
- 启动日志、进程号、诊断结果和网络抓包。

`session.env` 已在此前删除。
其他会话目录和执行账本未修改。
删除文件不等于撤销其他进程内存中的凭证。
当前直接连接流程不使用这些文件。

本次机器人操作系统（ROS）通信配置为 `ROS_DOMAIN_ID=146`。
中间件为 `rmw_fastrtps_cpp`，`ROS_LOCALHOST_ONLY=0`。
Gateway、Web 和 CLI 必须使用相同的 ROS 域和发现配置。
Web、Gateway 和 CLI 使用同一部署配置绑定的直接连接。
该流程不生成密钥，也不加载密钥文件。

厂商 SDK 单独使用网卡 `eno1` 和厂商 domain `0`。
ROS domain 与厂商 domain 不是同一个配置项。

原始排障记录不再可用。
第 6 节保留当时记录的超时设置和现场观察。

警告：网络抓包和诊断记录可能包含敏感数据。
禁止将敏感数据写入本文、Git、截图或聊天记录。

新部署必须重新确认目标机器人、网卡和 FSM。
禁止只修改目录名后复用其他机器人的配置。

### DDS 配置准备

DDS 指数据分发服务。
工作区使用 `dds-config` 管理 DDS 后端和发现方式。
默认配置保存在 `$HOME/.config/ibrobot/dds/config.yaml`。
`source .shrc_local` 会加载已保存的默认配置。
工具不设置 ROS 域，下面的启动命令仍显式保留域 `146`。
本次删除文件不修改用户默认配置，也不切换运行中的进程。

警告：切换通信配置前，必须确认没有活动任务或结果未知的任务。
旧的任务引用绑定原通信范围，不能跨范围自动恢复。

1. 打开新终端。
2. 检查当前 DDS 配置。

```bash
cd /home/lhj/orca/workspaces/IB_Robot/unitree-runtime-migration-171 &&
export ROS_DOMAIN_ID=146 ROS_LOCALHOST_ONLY=0 &&
source .shrc_local &&
dds-config show
```

如果已有符合部署要求的配置，保留现有配置。
如果尚未配置，按以下步骤设置。

1. 执行 `dds-config set`。
2. 选择 Fast DDS。
3. 选择部署需要的发现方式。
4. 核对网卡和地址。
5. 确认保存配置。

同一用户启动 Gateway、Web 和 CLI 时使用同一配置。
禁止在受管配置上叠加旧的 RMW 或 XML 环境变量。
embodied 启动入口会应用原有的 Fast DDS UDPv4 处理。
完整配置说明见 [DDS 配置](../scripts/dds/README.md)。

## 3. 终端 A：启动 Gateway 和 runtime

旧 Gateway 启动命令依赖已删除的配置，现已撤下。
清理后新增了最小部署配置。
当前启动使用第 3 节的 `g1-presets` 配置。
完整真机部署流程见 [G1 runtime 说明](../src/robots/unitree/unitree_g1_robot/README.md)。

配置目录为 `$HOME/.config/ibrobot/deployments/g1-presets/`。
目录只包含 `robot.yaml` 和 `runtime-profile.yaml`。
配置使用 `eno1`、预期 FSM `802` 和 10 秒 SDK 超时。
配置不启用手部控制，不保存密钥或凭证。

警告：`authorize_motion:=true` 允许后续运动请求。
启动本身不执行挥手，也不自动切换厂商 FSM。
只有现场条件符合第 1 节时，才使用运动授权。

1. 核对机器人型号、手部硬件和网卡。
2. 准备新的启动配置和 runtime profile。
3. 确认没有重复运行的 G1 部署。
4. 核对 ROS 通信配置和目标身份。
5. 使用新配置完成只读检查。
6. 在现场确认安全后，启动授权部署。

最小配置适用于本次 G1，不能直接用于其他机器人。
以下命令先启动只读验证，不允许提交运动。

```bash
(
cd /home/lhj/orca/workspaces/IB_Robot/unitree-runtime-migration-171 || exit
source .shrc_local || exit
export ROS_DOMAIN_ID=146 ROS_LOCALHOST_ONLY=0
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  config_path:="$HOME/.config/ibrobot/deployments/g1-presets/robot.yaml" \
  use_sim:=false control_mode:=named_motion \
  authorize_motion:=false with_perception:=false
)
```

本次流程不生成密钥。
Gateway 将 `robot_skill_cli` 标准操作员令牌固定为空值。
如果启动环境有旧的执行令牌，该值不会增加权限。
密钥和令牌不写入配置、ROS 参数或 launch 日志。

只读检查使用 `authorize_motion:=false`。
修改启动参数需要先正常退出原部署，再启动新部署。
禁止为同一机器人同时启动 2 套 runtime。

启动日志应显示运行时就绪和预置动作目录生成。
2026-10-30 的会话发现了 23 个预置动作。
动作数量随机器人返回的目录变化，不是固定验收值。

完整技能链路入口为 `embodied_bringup embodied_pipeline.launch.py`。
该入口包含 Gateway 和 runtime。
无需再单独启动 `unitree_g1_robot runtime.launch.py`。
本次机器人安装了 Revo2 手部。
原临时预置动作 profile 不控制手部。
禁止将默认 Dex3 profile 用于本次 Revo2 机器人。

## 4. 终端 B：启动 Web

Web 启动时不加载密钥，也不生成登录码。
Web 需要已配置并运行的 Gateway。
独立启动 Web 不会恢复已删除的真机部署。

Web 在启动时按部署配置绑定目标机器人。
客户端核对配置解析出的 endpoint 和 Gateway 报告的机器人身份。
绑定只读取本地配置。
建立浏览器会话不准备计划，也不提交动作。
会话建立后，界面会自动恢复同一目标的未结束任务。

警告：本流程依赖受信 ROS 网络。
本流程不验证操作员身份。
`ROS_DOMAIN_ID` 不是安全边界，也不是加密。
网络内进程可能发送请求。
只在本机单操作员部署中使用该流程。

1. 进入同一工作区。
2. 加载工作区和 DDS 配置。
3. 启动 Web 服务。

```bash
cd /home/lhj/orca/workspaces/IB_Robot/unitree-runtime-migration-171 &&
export ROS_DOMAIN_ID=146 ROS_LOCALHOST_ONLY=0 &&
source .shrc_local &&
ros2 run robot_skill_web robot-skill-web --config-name unitree_g1 --port 8765
```

`--config-name` 必须指向本次部署使用的配置；也可改用 `--config-path` 或
`ROBOT_CONFIG`，规则与 `robot-skill` 相同。四者都没有时 Web 以退出码 2 退出。
Web 启动只读取本地配置，不连接 Gateway，也不发送动作。
Web 启动不再引用本次会话目录。
本工作区的 Web 可执行文件位于 ROS 包目录。
使用 `ros2 run`，无需手动修改 `PATH`。
Web 仅监听本机 `127.0.0.1`。
终端只显示访问地址。

```text
http://127.0.0.1:8765
```

打开网页后，页面自动建立本机会话并显示已绑定的机器人。
页面不再显示发现列表，也没有「切换机器人」或「断开连接」入口。
目标是启动配置解析出的唯一目标。
换目标需要重启 Web 并传入另一个部署配置。

Web 会话有效期为 8 小时。
会话过期后，刷新页面即可重新建立同类会话。
重新建立会话不会改变目标、取消任务或重发任务。
旧浏览器会话失效，但任务状态不变。
其他部署的未确认任务会阻止会话建立，需要先在对应部署上核对。

会话建立被拒绝达到 5 次后，限制 60 秒。
会话 Cookie 和跨站请求伪造（CSRF）令牌自动管理。
操作员无需维护 Web 登录密钥。

### 浏览器操作

1. 打开终端显示的地址。
2. 核对页面显示的本次 G1 目标。
3. 核对授权状态和忙碌状态。
4. 选择要执行的技能。
5. 检查自动生成的计划。
6. 点击「执行 <技能名>」。

建立会话不查询也不提交动作。
建立会话后仍需 Gateway 的运动授权才能执行。
打开页面不会执行机器人动作。

界面自动准备计划，不需要手动点击预览按钮。
自动准备计划不会发送动作。
执行说明是审计文本，不是动作参数。
默认说明为「通过 Web 请求执行技能：<技能名>」。
修改参数或执行说明后，界面重新准备计划。

警告：只有现场操作员确认安全后，才点击执行按钮。
结果未知或执行超时时，禁止重复点击或换请求 ID 重发。

## 5. CLI 查询与单次挥手

CLI 与 Web 是 2 个操作入口。
两者复用同一个 Gateway，不要同时提交动作。

旧 CLI 配置和密钥文件已删除。
CLI 直接发现并连接目标。
发现和连接不需要本地机器人资源。
标准连接不加载密钥或凭证文件。
连接验证不代表操作员身份验证。
连接验证也不代表运动授权。

### 查询状态与目录

1. 打开第 3 个终端。
2. 加载工作区和 DDS 配置。
3. 用部署配置绑定目标，再查询状态和目录。

```bash
cd /home/lhj/orca/workspaces/IB_Robot/unitree-runtime-migration-171 &&
export ROS_DOMAIN_ID=146 ROS_LOCALHOST_ONLY=0 &&
source .shrc_local &&
robot-skill --config-name unitree_g1 status &&
robot-skill --config-name unitree_g1 list-skills &&
robot-skill --config-name unitree_g1 describe g1_wave_above_head
```

`--config-name` 必须与本次部署一致；也可用 `--config-path` 或 `ROBOT_CONFIG`。
不再支持 `robot-skill discover` / `connect` / `disconnect` 和 `--target`；
目标在启动时由配置确定，换目标就是换配置与 ROS 域。
配置绑定不会创建凭证绑定，也不会恢复已删除的配置。

技能名以当前服务端目录为准。
如果目录没有 `g1_wave_above_head`，停止使用挥手示例。

### 显式执行 1 次挥手

警告：下面的命令会提交真实运动请求。
只有机器人空闲、授权有效且现场操作员确认安全时，才执行。

1. 核对当前动作目录和现场状态。
2. 为本次请求填写唯一请求 ID。
3. 执行 1 次挥手命令。

```bash
robot-skill run-workflow \
  --text '现场确认安全后，执行一次头顶挥手' \
  --request-id g1-wave-manual-001 \
  --workflow-json '[{"schema_version":1,"skill_name":"g1_wave_above_head"}]'
```

`run-workflow` 展示计划后提交执行，不是只读预览命令。
新请求使用新请求 ID，不能用新 ID 绕过结果未知状态。
如果请求超时，保留原请求 ID 和完整终端输出。
先核对 Gateway 日志和现场状态，再决定后续操作。

CLI 的兼容流程不保证生成 Program 执行引用文件。
禁止将输出中的 `task_id` 当作 Program 执行引用。
Web 会保存自身请求的执行引用，用于恢复查询。

取消请求不保证中断厂商预置动作。
禁止用恢复动作 `99` 代替急停。

## 6. 10 秒超时设置

历史会话的 SDK 远程过程调用（RPC）超时为 10 秒。
原 `operator-profile.yaml` 已删除，字段记录如下：

```yaml
vendor:
  rpc:
    timeout_s: 10.0
```

这只是字段位置示意，不是完整 profile。
禁止用这段片段覆盖整个配置文件。
此记录不会自动配置新部署的超时。

- 仓库默认超时未修改。
- Gateway 的独立 `rpc_timeout_sec` 仍为 5 秒。
- 修改 profile 后，原进程不会自动加载新值。
- 超时不是动作完成信号，也不是允许自动重试的条件。

2026-10-30 的挥手请求返回 `SUCCEEDED`。
操作员确认挥手完成，双臂停止。
Gateway 记录的执行时长约为 7.30 秒。
这不是单独测量的 SDK 响应耗时或物理动作时长。

## 7. 退出与故障处理

警告：先确认动作结束，再退出软件。
`Ctrl+C` 不能作为经过验证的物理停止手段。

1. 停止提交新任务。
2. 查询活动任务或结果未知的任务。
3. 在现场确认机器人停止。
4. 在终端 B 按 `Ctrl+C` 退出 Web。
5. 在终端 A 按 `Ctrl+C` 退出 Gateway。

| 现象 | 检查或处理 |
| --- | --- |
| Web 无法发现 G1 | 确认 Gateway 已就绪，核对各终端的 ROS 域和 `dds-config show` 输出 |
| Web 连接失败 | 核对目标身份、Gateway 状态和 ROS 通信配置 |
| `MOTION_NOT_AUTHORIZED` | 核对启动参数；运动授权必须来自现场操作员 |
| `PRESET_HARDWARE_DISABLED` | 核对本次 operator profile，禁止关闭准入检查绕过错误 |
| `service ownership query failed` | 核对机器人运控模式、网卡和厂商服务状态 |
| `NAMED_MOTION_TIMEOUT` / SDK `3104` | 将结果视为未知，查询原任务并观察机器人，禁止自动重发 |
| 端口 `8765` 已占用 | 确认原 Web 实例及其任务状态，再决定退出原实例或使用其他端口 |
| 修改配置后仍是旧行为 | 正常退出原部署后重启，禁止启动重复实例 |

警告：禁止删除执行记录来解除未知结果保护。
真机取消、软件急停和完整退出清理尚未全部验收。

## 8. 参考文档

- [G1 runtime 说明](../src/robots/unitree/unitree_g1_robot/README.md)
- [Web 控制台说明](../src/agent/robot_skill_web/README.md)
- [RobotClient 连接说明](../src/agent/robot_client/README.md)
