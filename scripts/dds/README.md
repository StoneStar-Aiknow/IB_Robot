# DDS 配置

`dds-config` 为从 IB-Robot 源码工作区启动的新进程选择 DDS 后端和发现方式。目前仅支持 Bash；在工作区根目录通过项目环境入口使用：

```bash
source .shrc_local
dds-config show
dds-config set
```

问答依次选择 Fast DDS 或 CycloneDDS、通信范围，以及需要时使用的本机网卡、IPv4 和对端地址。选择题直接回车采用显示的默认选项；最后的应用确认默认为否，按 Ctrl-C 可取消。

## 命令与生效范围

| 命令 | 当前终端 | 当前用户默认配置 |
|---|---|---|
| `dds-config set` | 问答确认后立即生效 | 保存 |
| `dds-config set --tmp` | 问答确认后立即生效 | 不修改 |
| `dds-config set --config CONFIG.yaml` | 应用指定文件 | 保存 |
| `dds-config set --config CONFIG.yaml --tmp` | 应用指定文件 | 不修改 |
| `source .shrc_local` | 应用保存的默认配置；若没有默认配置，恢复本终端接管前环境 | 只读 |
| `dds-config unset` | 恢复接管前环境 | 移除 |
| `dds-config unset --tmp` | 恢复接管前环境 | 保留 |
| `dds-config show` | 查看当前配置及用户默认配置 | 只读 |

默认配置位于 `$HOME/.config/ibrobot/dds/config.yaml`，同一用户的不同工作区共用。`set` 会在应用前检查 YAML、本机网卡和路由、依赖及生成的配置；输入无效时不会修改当前 DDS 环境。本版没有独立的校验命令。不带 `--config` 的 `set` 打开问答界面；默认配置损坏时，可通过 `--config` 指定有效文件重新设置，或用 `unset` 移除。

`set` 和 `unset` 必须通过当前终端注册的 Bash 函数运行；管道、命令替换或子 shell 无法修改父 shell。更新工具后重新 `source .shrc_local` 加载函数，同时会应用已保存的默认配置。

只有随后启动的进程继承新环境；已经运行的节点及其他终端不会热切换。`unset` 不删除生成的 XML，也不卸载系统 DDS 包，因为已有进程可能仍在使用。

## 配置格式与适用运行库

本版在 Ubuntu/openEuler 的 `/opt/ros/humble` 下检查以下系统包版本：

| 后端 | DDS 版本 | RMW 包及版本 |
|---|---|---|
| Fast DDS | 2.6.11 | rmw_fastrtps_cpp 6.2.10 |
| CycloneDDS | 0.10.5 | rmw_cyclonedds_cpp 1.3.4 |

缺少依赖或版本不在支持范围时会报错；本提交尚不自动安装或升级依赖。不支持任意厂商 XML、自定义运行库 overlay、调优参数或 Zsh 激活。

仅本机发现的最小 YAML 示例：

```yaml
schema_version: 1
backend: fastdds
network:
  discovery: local_only
backends:
  fastdds:
    profile:
      source: generated
      preset: fastdds/conservative-v1
      parameters: {}
  cyclonedds:
    profile:
      source: generated
      preset: cyclonedds/conservative-v1
      parameters: {}
```

选择 CycloneDDS 时，将 `backend` 改为 `cyclonedds`。若要在局域网通过组播自动发现，将 `network` 替换为下方内容，并填写实际的本机网卡与 IPv4：

```yaml
network:
  discovery: lan_multicast
  interface: eth0
  address: 192.0.2.10
```

通过指定对端的单播地址引导发现时，使用 `static_peers`，并提供非空的对端列表：

```yaml
network:
  discovery: static_peers
  interface: eth0
  address: 192.0.2.10
  peers: [192.0.2.20, 192.0.2.30]
```

示例地址只是占位值。本机地址必须属于所选网卡，前往 peer 的路由也必须走该网卡；只有一个合格 IPv4 时可以省略 `address`。`lan_multicast` 要求网卡支持组播，`static_peers` 不要求组播。peer 地址仅用于引导发现，不是访问控制白名单，也不意味着所有 peer 必须在线。Fast DDS 的内置传输不提供此生成 XML 原先自定义 UDP transport 的接口白名单：所选地址用于发现 locator 与路由检查；工具也不固定数据传输 locator，以保留厂商默认的本机 SHM 选择。因而不保证该进程只在所选网卡打开 UDP socket、向该网卡发送组播或在其他网卡不通数据；若部署要求严格的网卡隔离，应另行验证或采用网络层约束。

无组播模式将发现索引扩展至 0..32：CycloneDDS 使用 `ParticipantIndex=auto` 和 `MaxAutoParticipantIndex=32`，Fast DDS 为初始 peer 补充索引 4..32 的发现端口，不更改内置传输或数据 locator。端口被占用、参与进程数超过范围仍可能导致发现失败。Fast DDS 此时按设置时的 `ROS_DOMAIN_ID`（未设置为 0）计算发现端口，要求该值为 0..232；运行期间修改 domain 会被受管入口拒绝，应重新 `source .shrc_local` 或 `set`。工具不设置 `ROS_DOMAIN_ID` 或 `ROS_LOCALHOST_ONLY`；跨机时仍需两端 domain 一致，且 `ROS_LOCALHOST_ONLY=1` 与跨机发现冲突。

## 状态、错误与恢复

`show` 分开显示保存的默认选择和当前终端配置；有效配置相同时合并显示。状态含义：

- `ready`：当前配置快照、生成内容、本机运行库与网卡事实、环境映射均通过检查。
- `not ready`：无法确认当前配置；请查看诊断信息。
- `未启用 dds-config`：本终端未由工具接管，ROS 和用户既有环境决定使用哪个后端。

未设置 `RMW_IMPLEMENTATION` 不等于 DDS 不可用，ROS 会选择默认 RMW。这里的任何状态都不能证明远端已发现或业务数据已送达。

若手工修改受管的 RMW/XML 环境变量，重新执行 `set`，或 `source .shrc_local` 恢复已保存的默认配置；修复前 `show` 和 launch 检查会持续提示冲突。

恢复环境时会保留原变量的值、是否存在及 export 属性。若一个终端仅继承了另一终端的 DDS 设置，它无法推断接管前的原环境；请在最初设置的终端运行 `unset`。HOME 改变、内部状态损坏或变量具有不支持的属性时，需处理来源并在新终端初始化。

失败的配置操作会保持或恢复操作前的终端环境。加载项目环境时，若默认 DDS 配置激活出现可恢复错误，source hook 会报告错误并标记配置未就绪，同时允许一般开发环境继续初始化；`source` 返回 0 不等于 DDS 已 ready。不会静默换后端。普通 `set` / `unset` 失败会返回非零状态。

如果生成的 XML 或目录损坏，仅重复 `set` 不能修复不可变目录。报错会指出目标目录并给出安全、不覆盖已有文件的移动命令。先停止引用该配置的节点，将目录移到提示的备份位置（若已存在则另取备份名），再重新 `set`。工具不会自动删除或覆盖仍可能被使用的文件；备份保留供检查。单纯缺失或过期的生成目录可直接通过 `set` 准备。

生成目录只包含 `profile.xml` 和完整性清单。更新工具或本机事实改变后，重新 `source .shrc_local` 会在保存的默认配置仍有效且本机检查通过时生成新产物；临时配置可重新 `set --tmp`。不会在 `source` 时安装包、覆盖旧产物或改变已运行进程；本机事实无效时仍会报错并标为未就绪。

`--json` 返回机器可读结果。`set` / `unset` 的运行错误在 stderr 输出包含 `error`、`code`、`advice` 的 JSON；无效参数沿用命令行标准帮助提示。终端标题遵循 `NO_COLOR` 与 `TERM=dumb`。

## 应用启动检查

以下入口会在启动节点前检查继承的受管 DDS 环境：

- `robot_config/robot.launch.py`
- `robot_config/teleop_operator.launch.py`
- `embodied_bringup/embodied_pipeline.launch.py`
- `inference_service/cloud_inference.launch.py`（推理和录制）

已知失败、启动环境与受管记录不一致以及不支持的记录，都会在节点启动前给出原因和修复建议。唯一的入口局部例外是 embodied 对受管 Fast DDS 严格预检成功后显式设置的 UDPv4 内置传输变量；其他入口和其他变量依旧严格检查。robot 入口在构造延迟启动的消费者前还会再检查一次；此时若失败，provider 可能已运行，随后走现有 launch 停止路径。

检查只读取继承的环境：不重新探测文件、库或网卡，不创建 DDS participant，不检查远端 peer 或跨机厂商组合，也不验证端到端 QoS 和数据送达。launch 文件导入错误可能发生在检查运行之前。未受管环境沿用既有行为；其他独立 launch、`ros2 run`、RViz 和外部客户端不受全局拦截。

生成 XML 配置所选发现策略；Fast DDS 不再关闭内置传输、强制仅用 UDPv4 或固定数据 locator，允许已装版本自动生成 UDP/共享内存等默认数据 locator。无组播模式仍显式指定发现单播 locator 和初始 peer；是否实际使用 SHM 由厂商和运行环境决定，`ready` 不证明 SHM 已使用。`local_only` 限制发现范围，不是内置 UDP 网卡的隔离保证。CycloneDDS 不再强制 `Transport=udp` 或人为设置 participant 上限。`local_only` 与 `static_peers` 明确关闭 CycloneDDS 组播并采用 `ParticipantIndex=auto` 来实现单播发现；`lan_multicast` 沿用其对所选网卡的默认组播策略。embodied 启动入口是已有的显式例外：用 Fast DDS 时仍为本次 launch 设置 `FASTDDS_BUILTIN_TRANSPORTS=UDPv4`（包括受管 Fast），以保持其原有的服务发现修复；使用 CycloneDDS 时不设置 Fast 专用变量。独立 robot 启动不会继承 embodied 的例外。Robot Runtime 的 readiness 与接口绑定是独立检查，不随本工具更改。
