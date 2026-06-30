---
name: oh-constraints
description: "OpenHarmony (EmbodiedAI 1.0.1 / RoboOH 1.0.1) 板端运行时约束汇总。凡是涉及 OpenHarmony 板端操作、板端 shell 脚本编写、交叉编译产物部署到板端、BQ3588HM/RoboPi 板端命令执行、musl/aarch64、toybox 限制、read-only rootfs、no /usr/bin/env、no systemd、no awk/getprop、LD_PRELOAD 干扰、SSH RemoteCommand、HDC 连接等任何 OpenHarmony 相关内容时，必须先读取本 skill 了解平台约束，再动手。Triggers for 'OpenHarmony', '板端', '板端脚本', '板端约束', 'BQ3588HM', 'RoboPi', 'toybox', 'musl', 'aarch64-ohos', 'cross-compile deploy', 'board shell script', '板端服务管理', 'shebang', '/usr/bin/env', 'no systemd', 'read-only root'."
---

# OpenHarmony 板端运行时约束

本 skill 汇总 OpenHarmony EmbodiedAI 1.0.1 (RoboOH 1.0.1) 板端的运行时约束。适用于 BQ3588HM、RoboPi 及任何运行该系统的 RK3588 aarch64/musl 开发板。

**凡是涉及 OpenHarmony 板端的工作（写脚本、部署二进制、执行命令、管理服务），必须先读本 skill，避免踩坑。**

## 1. musl libc（非 glibc）

- 动态链接器：`/lib/ld-musl-aarch64.so.1`（不是 `/lib/ld-linux-aarch64.so.1`）。
- C++ 运行时：`/system/lib64/libc++_shared.so`。
- 所有二进制必须为 `aarch64-linux-ohos`（musl）交叉编译产物，glibc 版二进制无法运行。
- 验证命令：`file <binary>` 应显示 `interpreter /lib/ld-musl-aarch64.so.1`。

## 2. toybox 工具箱限制

OpenHarmony 默认用 `toybox` 作为工具箱，命令集远小于 GNU coreutils：

| 缺失/受限 | 影响 | 替代方案 |
|-----------|------|---------|
| `awk` 不存在 | 文本处理、字段提取 | 用 `sed`/`cut`/shell 变量替代，或 `grep -o` |
| `getprop` 不存在（部分镜像） | 读取系统属性 | 不依赖 getprop，改用文件读取 |
| `find` 可能 segfault | 文件查找 | 用 `ls -R` 或明确路径，避免 `find /` |
| `ps -ef` 进程名截断 | 进程检测 | `ps` 只显示进程名（如 `openclaw`），不显示完整命令行 `node ... openclaw.mjs gateway ...`。**不要**用 `ps \| grep <full-cmdline>` 做唯一判断，改用 **pidfile + 端口检测** |
| `systemctl` 不存在 | 服务管理 | 见 §5 无 systemd |

**编写板端脚本的原则**：只用 POSIX sh + toybox 内置命令，避免 awk/find/getprop；进程管理用 pidfile + `netstat`/`kill -0`。

## 3. 无 `/usr/bin/env`（shebang 问题）

板端**没有** `/usr/bin/env`，只有 `/bin/env` 或 `/system/bin/env`。

- `#!/usr/bin/env node` 这类 shebang **无法执行**，报 `No such file or directory`。
- npm/npx/corepack/openclaw 等 JS CLI 工具的入口 shebang 都是 `#!/usr/bin/env node`。
- **解决方案**：删除 symlink，改用绝对路径 shell wrapper：
  ```sh
  #!/bin/sh
  exec /data/local/nodejs/bin/node /path/to/cli.mjs "$@"
  ```
- 切记：若 `bin/cli` 是 symlink 指向 `cli.mjs`，**先 `rm` symlink 再 `cat > wrapper`**，否则会跟随 symlink 覆盖原始 `.mjs` 文件。

## 4. bash 不在默认 PATH

- 默认非交互 PATH：`/usr/bin:/bin:/usr/sbin:/sbin`（无 `/data/bin`）。
- `bash` 通常在 `/data/bin/bash`（个人部署），**默认 PATH 找不到**。`sh` 在 `/bin/sh`（toybox）。
- **脚本一律用 `#!/bin/sh`**，确保 POSIX 兼容（无 bashism：无 `[[ ]]`、无数组、无 `BASH_*`、无 `source`（用 `.`））。
- 通过 SSH 管道执行脚本：`ssh ... 'sh -s' < script.sh`，不要用 `bash -s`。
- 若必须用 bash，显式指定 `/data/bin/bash`，并在脚本里 source 个人 env 先把 `/data/bin` 加入 PATH。

## 5. 无 systemd（服务管理）

- 板端**没有 systemd**，`systemctl` 不存在。
- 不要依赖 `install/start/stop` 子命令（如 `openclaw gateway install`）。
- **自管 pidfile + 端口检测**：
  ```sh
  # 启动
  <command> > logfile 2>&1 &
  echo $! > /path/service.pid
  # 检测存活
  PID=$(cat /path/service.pid)
  kill -0 "$PID" 2>/dev/null && echo running || echo stopped
  # 端口检测
  netstat -an | grep ":<port> " | grep LISTEN
  # 停止
  kill "$PID"; sleep 2; kill -9 "$PID" 2>/dev/null; rm -f /path/service.pid
  ```

## 6. 只读根文件系统

- `/`（rootfs）默认**只读**。`/system`、`/usr`、`/lib` 等不可写。
- **可写区域**：
  - `/data` —— 持久化可写（主要部署位置）。
  - `/data/local/tmp` —— 持久化临时目录。
  - `/tmp` —— 可能只读（rootfs 上），robooh env 会 `mount -t tmpfs tmpfs /tmp` 使其可写（RAM，重启清空）。
- 部署二进制、配置、日志一律放 `/data/local/<app>/`，不要写 `/usr/local` 或 `/opt`。
- 临时文件用 `/data/local/tmp`（持久）或确保 `/tmp` 已挂 tmpfs。
- 若需改根文件系统：`mount -o remount,rw /`，用完 `mount -o remount,ro /` 恢复。

## 7. LD_PRELOAD 干扰（robooh env）

RoboPi 的 `robooh_1.0.1.env` 设置：
```sh
export LD_PRELOAD=/sys_prod/robot/out/lib/libpython3.12.so.1.0
```
这会把 libpython 注入**每个子进程**，包括 node、openclaw 等 non-ROS 进程，可能干扰 native 模块加载。

- **ROS/Python 推理任务**：source robooh env（需要 LD_PRELOAD）。
- **Node.js / OpenClaw / 非 ROS 二进制**：**不要** source robooh env，用独立 env 脚本（仅设 PATH）。
- 若必须在 robooh 环境下跑 non-ROS 进程，先 `unset LD_PRELOAD`。

## 8. 无外网（常见）

板端常无外网连通，DNS 可能配置但出不通：
- `npm install`、`npm update`、update check 会失败（`EAI_AGAIN`、`fetch timeout`）。
- **解决方案**：主机侧离线准备安装前缀（`npm install -g --prefix <dir> --os=linux --cpu=arm64 --libc=musl`），打包推送到板端。
- 日志中的 `fetch timeout registry.npmjs.org` 属正常现象，不影响已部署服务运行。

## 9. 板端访问约束

OpenHarmony 板端有两种访问方式：**HDC（始终可用）** 和 **SSH（需配置）**。SSH 不通时 HDC 必通——每次重启后需通过 HDC 重新执行 `setup_sshd.sh` 启动 sshd（无 systemd，无法自启）。

> **完整的 HDC/SSH 连接、文件传输、SSH 配置流程详见 `oh-access` skill**（`.agents/skills/oh-access/SKILL.md`）。

## 10. 环境脚本分层

板端环境分多层，按需 source，**不要无脑全 source**：

| 脚本 | 作用 | 何时 source |
|------|------|------------|
| `/data/roboframe/scripts/robooh_1.0.1.env` | ROS 2 + Python + NPU 驱动 + LD_PRELOAD | 跑 ROS/推理时 |
| `/data/me.env` | 个人工具（bash/zsh/starship/terminfo）+ TMPDIR | 交互 shell 增强 |
| `/data/enter.env` | 一键入口（source robooh + me + tmpfs mount） | 交互登录 |
| `/data/local/nodejs/nodejs.env` | Node.js PATH | 跑 node/npm 时 |
| `/data/local/openclaw/openclaw.env` | OpenClaw PATH（含 nodejs.env） | 跑 openclaw 时 |

**非 ROS 服务（node/openclaw）只 source 自己的独立 env，不挂 robooh**，避免 LD_PRELOAD 干扰。

## 11. 脚本编写检查清单

编写板端 shell 脚本前，逐项确认：

- [ ] shebang 是 `#!/bin/sh`（不是 `#!/bin/bash` 或 `#!/usr/bin/env ...`）
- [ ] 无 bashism（无 `[[`、`[[ ]]`、数组、`BASH_SOURCE`、`source` 改用 `.`）
- [ ] 无 `awk`、无 `getprop`、无 `find /`
- [ ] 进程检测用 pidfile + `kill -0` + `netstat`，不用 `ps | grep <cmdline>`
- [ ] 所有写入路径在 `/data` 下（不是 `/usr`、`/opt`、`/tmp` 除非已挂 tmpfs）
- [ ] 日志输出到 `/data/local/tmp/` 或 `/data/local/<app>/`
- [ ] 服务用 pidfile 自管，不依赖 systemd
- [ ] JS CLI 入口用绝对路径 wrapper，不依赖 `/usr/bin/env`

## 参考文档

- `docs/OpenHarmony_EmbodiedAI_NodeJS_OpenClaw_Gateway.md` —— Node.js + OpenClaw 部署全过程，本文约束的来源
- `.agents/skills/oh-access/SKILL.md` —— HDC/SSH 连接与文件传输

## 板端路径布局

`install.sh` 执行后的典型路径：

| 内容 | 路径 |
|------|------|
| ROS 2 Humble（系统预装） | `/sys_prod/robot/install` |
| 系统依赖 / sysdeps | `/sys_prod/robot/out` |
| RoboFrame 包（交叉编译产物） | `/data/roboframe/install` |
| Python pysite（torch + 依赖） | `/data/roboframe/pysite` |
| env 脚本 | `/data/roboframe/scripts/robooh_1.0.1.env` |
| Python 3.12 | `/sys_prod/robot/out/bin/python3` |
| rknnlite（预装） | `/sys_prod/robot/out/lib/python3.12/site-packages` |
| librknnrt.so | `/vendor/lib64/librknnrt.so`（或 `/usr/lib/`） |

## RKNN 运行时约束

- **NumPy 优先级**：`torch` 必须看到 pysite 的 NumPy 1.26.4（`/data/roboframe/pysite`），而不是系统的 NumPy 2.4.0（`/sys_prod/robot/out/lib/python3.12/site-packages`），否则 `torch.from_numpy` 会失败。`robooh_1.0.1.env` 的 PYTHONPATH 已正确排序。
