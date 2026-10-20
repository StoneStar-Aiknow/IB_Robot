# Ubuntu 与 Ascend310P 两端时钟对齐操作手册

本文记录已验证的硬件 PTP 操作顺序：先同步两端网卡 PHC，再同时采集各自主机时钟到 PHC 的映射。
系统 `date` 无需相同；trace 中的原始时间戳需通过映射转换到共同参考轴，不能直接相减。

映射 API、误差预算与限制见 [clock_calibration.md](clock_calibration.md)。以下命令在 **Bash** 中执行。
本文不启动机器人，也不调整系统 `CLOCK_REALTIME`；既有 NTP 保持运行，不为本流程启动 `phc2sys`。

## 1. 确认设备与准备工具

本次验证环境：

| 项目 | Ubuntu 主端 | 310P 从端 |
|---|---|---|
| IP / 接口 | `192.168.7.100` / `enp129s0` | `192.168.7.101` / `eth0` |
| 驱动 / PHC | `igc` / `/dev/ptp0` | `xgmac` / `/dev/ptp0` |
| LinuxPTP | 3.1.1 | 4.3 |
| 角色 | MASTER，固定 grandmaster | SLAVE，跟随该 grandmaster |
| PTP domain | 97 | 97 |

其他机器应替换接口、PHC、IP 和 grandmaster identity。PTP domain 与 ROS domain 独立。

安装缺少的系统工具；已有工具无需重复安装：

```bash
# Ubuntu
sudo apt-get install linuxptp ethtool acl

# 310P，root 终端；使用主机已配置、适用的仓库
dnf install linuxptp ethtool
```

分别检查各自接口，例如：

```bash
# Ubuntu
ethtool -T enp129s0
ethtool -i enp129s0
ptp4l -v
pmc -v
pgrep -a -x ptp4l
pgrep -a -x phc2sys
```

310P 换为 `eth0`。确认存在 `hardware-transmit`、`hardware-receive`、`hardware-raw-clock`，
`PTP Hardware Clock: 0` 对应 `/dev/ptp0`。如果已有 PTP 服务占用该 PHC，先明确其用途；不要同时启动第二个 servo。

两端各自进入 IB_Robot 工作区，加载本工作区环境并构建工具：

```bash
source .shrc_local
./scripts/build.sh --mixin release no-test -- --symlink-install --packages-select ibrobot_tracing
source .shrc_local
ros2 run ibrobot_tracing ibrobot-clock --help
```

## 2. 创建本次记录目录并放行网络

每次启动 PTP 使用新的共同会话名。以下示例名只用一次，之后必须替换。
在两端工作区根目录分别执行相同命名：

```bash
ALIGN_SESSION=align-20260921-run02
ALIGN_DIR="$PWD/tmp/clock-align/$ALIGN_SESSION"
mkdir -p "$PWD/tmp/clock-align"
mkdir "$ALIGN_DIR"
```

末级目录用 `mkdir` 拒绝覆盖旧实验。后续每个新终端都需要进入正确工作区，重新设置相同的
`ALIGN_SESSION` 和 `ALIGN_DIR`，但不要再次创建该目录。

本次使用 **UDPv4/E2E/HARDWARE**。Ubuntu 若启用 UFW，先检查已有规则，缺少时添加：

```bash
sudo ufw status numbered
sudo ufw allow in on enp129s0 from 192.168.7.101 to any port 319 proto udp
sudo ufw allow in on enp129s0 from 192.168.7.101 to any port 320 proto udp
```

另一端或中间设备如果也有防火墙，应允许这两个对等主机间的 UDP 319/320 及 PTP 组播流量。
不要关闭全部防火墙。硬件时间戳支持并不能保证每个驱动的二层 PTP 路径可用，相关已知问题见第 8 节。

## 3. 写入完整主从配置，前台启动 PTP

用编辑器将下列内容保存到各自主机的 `$ALIGN_DIR/ptp4l.cfg`。

Ubuntu 配置：

```ini
[global]
domainNumber 97
network_transport UDPv4
delay_mechanism E2E
time_stamping hardware
tx_timestamp_timeout 100
logSyncInterval -3
logMinDelayReqInterval -3
logAnnounceInterval 0
summary_interval 0
uds_address /tmp/ibrobot-align-master.sock
logging_level 6

[enp129s0]
masterOnly 1
```

310P 配置：

```ini
[global]
domainNumber 97
network_transport UDPv4
delay_mechanism E2E
time_stamping hardware
tx_timestamp_timeout 100
slaveOnly 1
logSyncInterval -3
logMinDelayReqInterval -3
logAnnounceInterval 0
summary_interval 0
uds_address /tmp/ibrobot-align-slave.sock
uds_ro_address /tmp/ibrobot-align-slave-ro.sock
logging_level 6

[eth0]
```

`slaveOnly` 是本次 LinuxPTP 4.3 验证过的兼容参数，会提示改名为 `clientOnly`；该提示不表示启动失败。
`masterOnly` 用于本次主端 3.1.1。若换版本，先核对工具支持的选项。

在两个前台终端依次启动主端、从端，并保持终端打开：

```bash
# Ubuntu 的 PTP 终端
sudo ptp4l -f "$ALIGN_DIR/ptp4l.cfg" -m -q 2>&1 | tee "$ALIGN_DIR/ptp4l.log"
```

```bash
# 310P 的 root PTP 终端
ptp4l -f "$ALIGN_DIR/ptp4l.cfg" -m -q 2>&1 | tee "$ALIGN_DIR/ptp4l.log"
```

使用独立的 Unix socket，不复用其他 PTP 服务。该流程同步 slave PHC，不将系统时间同步到 PHC。

## 4. 为采集授予最小权限，确认真正锁定

Ubuntu 另开普通用户终端，进入同一工作区并设置本次目录变量。
PTP 启动、socket 已出现后，保存 PHC 权限并授予临时访问：

```bash
test -S /tmp/ibrobot-align-master.sock
getfacl -p /dev/ptp0 > "$ALIGN_DIR/phc0-acl-before.txt"
sudo setfacl -m "u:$(id -un):r" /dev/ptp0
sudo setfacl -m "u:$(id -un):rw" /tmp/ibrobot-align-master.sock
```

这些 ACL 命令每次会话只执行一次；不要再次覆盖最初的权限备份。PHC 仅授予读取，
PMC 主 socket 的访问权限应只授予受信任的实验用户。结束时按第 7 节恢复；daemon 重启后 socket ACL 需重新检查。
310P 的 root 终端无需额外 ACL。

Ubuntu 检查：

```bash
pmc -u -b 0 -d 97 -s /tmp/ibrobot-align-master.sock -i /tmp/ibrobot-align-check-master.sock \
  'GET DEFAULT_DATA_SET' 'GET PORT_DATA_SET' 'GET TIME_STATUS_NP' \
  'GET CURRENT_DATA_SET' 'GET PORT_PROPERTIES_NP' | tee "$ALIGN_DIR/pmc-ready.txt"
```

310P 检查：

```bash
pmc -u -b 0 -d 97 -s /tmp/ibrobot-align-slave.sock -i /tmp/ibrobot-align-check-slave.sock \
  'GET DEFAULT_DATA_SET' 'GET PORT_DATA_SET' 'GET TIME_STATUS_NP' \
  'GET CURRENT_DATA_SET' 'GET PORT_PROPERTIES_NP' | tee "$ALIGN_DIR/pmc-ready.txt"
```

间隔约一秒检查数次，确认稳定：

| 字段 | Ubuntu 预期 | 310P 预期 |
|---|---|---|
| `portState` | `MASTER` | `SLAVE` |
| `timestamping` | `HARDWARE` | `HARDWARE` |
| `interface` | `enp129s0` | `eth0` |
| `domainNumber` | 97 | 97 |
| `gmIdentity` | 等于本机 `clockIdentity` | 等于 Ubuntu 的 `clockIdentity` |
| `gmPresent` | 自己作为 GM 时可以是 false | 必须是 true |
| `ingress_time` | MASTER 可以是 0 | 非零，并随新同步更新 |

本次主时钟 identity 为 `c85309.fffe.a775b6`；其他机器应读取真实值，再在两端设置：

```bash
ALIGN_GM=c85309.fffe.a775b6
```

不能仅凭进程存在或日志出现一次 MASTER 就开始计时。如果 slave 为 LISTENING/UNCALIBRATED/FAULTY，
先排查；servo offset 接近零也只代表同步残差，不能单独作为误差认证。

## 5. 两端同时采集，再运行需要分析的业务

两个采集终端使用相同 session、domain、GM；主机路径可以不同。各自先 `source .shrc_local`。

Ubuntu：

```bash
source .shrc_local
ros2 run ibrobot_tracing ibrobot-clock capture \
  --phc /dev/ptp0 --interface enp129s0 --pmc-socket /tmp/ibrobot-align-master.sock \
  --domain 97 --grandmaster "$ALIGN_GM" --session-id "$ALIGN_SESSION" \
  --duration 300 --rate 50 --output "$ALIGN_DIR/ubuntu-clock.jsonl" \
  > "$ALIGN_DIR/capture.log" 2>&1
```

310P：

```bash
source .shrc_local
ros2 run ibrobot_tracing ibrobot-clock capture \
  --phc /dev/ptp0 --interface eth0 --pmc-socket /tmp/ibrobot-align-slave.sock \
  --domain 97 --grandmaster "$ALIGN_GM" --session-id "$ALIGN_SESSION" \
  --duration 300 --rate 50 --output "$ALIGN_DIR/310p-clock.jsonl" \
  > "$ALIGN_DIR/capture.log" 2>&1
```

上限为 600 秒/次。选定足以覆盖业务的时长，在 **两端采集已开始且各自样本状态有效后**，再从其他终端
启动 tracing 和 LIBERO/推理业务。可以在未写完最后一行时短暂避开读取，检查 JSONL 中完整的 `sample`
记录：应有有效的 `status` 且其 poll 时间持续更新。起始缺少状态及 settling 区间不用于测量。

业务结束后继续保留少量校准样本，然后等待采集自然结束或分别 Ctrl-C。PTP 进程必须覆盖整个采集期间。
最终只分析两端**共同有效时间区间内**的事件；采集器无法为第 3 步旧日志补做历史对时，也不会对新事件外推。
如果任一端/GM 重启或参考时钟跳变，应结束本次会话并使用新的 session 重新采集。

## 6. 检查采集质量并保存原始数据

在各自主机检查对应文件：

```bash
# Ubuntu
source .shrc_local
ros2 run ibrobot_tracing ibrobot-clock inspect "$ALIGN_DIR/ubuntu-clock.jsonl"
```

310P 使用同样入口检查 `"$ALIGN_DIR/310p-clock.jsonl"`。应存在足够的 `valid` 区间；
返回码为 0 仅说明文件可读，不意味着已锁定、覆盖完整或满足精度要求。

把从端的 JSONL、PTP/PMC 日志复制到主端本次记录目录，例如在 Ubuntu 执行：

```bash
scp "root@192.168.7.101:/root/Workspace/IB_Robot-libero-repro-20260920/tmp/clock-align/$ALIGN_SESSION/310p-clock.jsonl" \
  "$ALIGN_DIR/310p-clock.jsonl"
scp "root@192.168.7.101:/root/Workspace/IB_Robot-libero-repro-20260920/tmp/clock-align/$ALIGN_SESSION/ptp4l.log" \
  "$ALIGN_DIR/310p-ptp4l.log"
scp "root@192.168.7.101:/root/Workspace/IB_Robot-libero-repro-20260920/tmp/clock-align/$ALIGN_SESSION/pmc-ready.txt" \
  "$ALIGN_DIR/310p-pmc-ready.txt"
```

复制路径需按实际工作区替换。保存两端 tool/version、配置、日志、原始 trace、JSONL 及应用代码版本。
按 API 将事件映射后，两个时间点的 `uncertainty_ns` 相加，预算超限或映射无效的事件明确标为不可比较。
默认单端上限 50 µs、跨端上限 100 µs；这些是带假设的工程估计，不等于绝对精度证书。

本次实验曾用每端 400 个未参与校准的 PHC 读数以及 200 次独立 UDP 因果关系检查做复核。
在新的系统负载或不同网卡上应重新检查读取窗口和留出残差；不要将本次 45.741 µs 的最大估计误差
当作所有后续实验的固定误差。

## 7. 按顺序结束并恢复权限

1. 结束业务与 tracing，再结束两端校准采集。
2. 在两个前台 PTP 终端分别 Ctrl-C，确认本次 ptp4l 已退出。
3. Ubuntu 在普通用户终端恢复原始 PHC ACL：

```bash
sudo setfacl --restore="$ALIGN_DIR/phc0-acl-before.txt"
getfacl -p /dev/ptp0
```

若本次 PMC socket 在 daemon 退出后仍存在，确认其不再被使用，再撤销该 socket 的用户 ACL；不要删除其他进程的 socket。
此流程不会将 slave PHC 调回旧时间/频率，停止后它继续自由运行；校准文件不向停止后的时间外推。

工具包及规则可保留供下次实验。确认后续不需要 PTP 流量时，按规则原样撤销（不要与启动命令一起执行）：

```bash
sudo ufw delete allow in on enp129s0 from 192.168.7.101 to any port 319 proto udp
sudo ufw delete allow in on enp129s0 from 192.168.7.101 to any port 320 proto udp
```

最后核对 `pgrep -a -x ptp4l`、`pgrep -a -x phc2sys` 和 `timedatectl show -p NTP -p NTPSynchronized`。
不要全局杀 Python/PTP 进程；后台部署要记录自有 PID/进程组及恢复动作，避免只停父进程留下子进程。

## 8. 已验证的故障排查

| 现象 | 检查与处理 |
|---|---|
| 主端 `timed out while polling for tx timestamp` / FAULTY | 确认硬件时间戳及驱动状态；本次 igc 设置 `tx_timestamp_timeout 100` 后可用。仍失败时保留日志，不能改软件时间戳后沿用原硬件精度结论 |
| 从端二层 PTP 持续 `bad message` | 本次 xgmac 固定 `rx-fcs: on`，AF_PACKET 含 FCS 尾部；UDPv4 路径已验证，避免直接复用 L2 配置 |
| 从端一直 LISTENING | 检查同 domain、网卡、组播、UDP 319/320 和对等防火墙；Ubuntu 可查 `journalctl -k --since '-10 min'` 中的 UFW BLOCK |
| PHC/socket Permission denied | 确认选对 PHC；按第 4 节授予临时读取/socket 权限。daemon 重启会重建 socket |
| master `gmPresent=false` | 当 `portState=MASTER` 且自己的 clockIdentity 等于指定 GM 时属于正常情况；slave 必须 gmPresent=true |
| `outside_capture` | 两端校准与事件窗口不重叠；重新协调采集启动，不能以外推伪造缺失数据 |
| `stale_ptp_status` / `stale_sync` / `ptp_not_locked` | 记录故障区间，排查 daemon 停止/阻塞、链路中断，恢复后等待新状态重新稳定 |
| `reference_discontinuity` | 已观察到参考 PHC 跳变，结束该 capture，建立新的 session |
| 单次 PHC 读数残差达数百 µs | 先看读取窗口；本次重复读取选最短窗口后才具备微秒级留出校验条件 |

本手册验证的是同步和校准采集。CTF 合并、业务 request/frame 关联和 RTP 指标换算还需要显式消费映射，
运行了 PTP 不会自动修正已有代码中的跨机 monotonic 相减。
