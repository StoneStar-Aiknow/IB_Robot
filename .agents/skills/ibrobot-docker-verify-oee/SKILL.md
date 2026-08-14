---
name: ibrobot-docker-verify-oee
description: "在 openEuler Embedded (aarch64) Docker 容器中实际执行 setup.sh + build.sh。Use when the user explicitly asks for openEuler/OEE Docker verification, or when author-side PR creation/update workflows trigger the dependency/setup verification gate. Do not use automatically during PR review; review should check developer-provided Verification in the PR description."
---

# IB-Robot openEuler Embedded Docker Verification Skill

在全新 openEuler Embedded aarch64 Docker 容器中完整验证 `setup.sh` 和 `build.sh`。

> **⚠️ 与 Ubuntu 验证的两个核心差异：**
> 1. **以 root 用户操作**（openEuler Embedded 开发板默认 root，无需 sudo/testuser）
> 2. **所有 docker exec 命令必须 `chroot /root/openeuler_rootfs`** 进入 arm64 rootfs 环境才能执行

容器镜像通过 `--privileged` + `chroot /root/openeuler_rootfs` 进入 qemu-user 模拟的 arm64 环境，
模拟真实 openEuler Embedded 开发板的首用体验。

## When to Use

- 用户明确要求 "openEuler Docker 验证" / "oee container test" / "实际验证 openEuler setup/build"。
- 作者侧创建/更新 PR 流程（`atomgit-pr` 或 `ibrobot-git-flow`）触发依赖/setup 验证门禁，需要真实结果写入 PR 描述。
- 当前任务是验证本地对 `scripts/setup/platforms/openeuler-embedded-24.03.sh`、`scripts/setup.sh`、`scripts/setup/lerobot_patches.sh` 或 dnf/rosdep 相关逻辑的修改。
- 不要仅因为 PR review 触发本 skill。

## Review Boundary

- PR review 过程中，禁止因为 PR 修改了 `package.xml`、`setup.py`、setup 脚本或 build 文件就自动运行本 skill。
- 用户要求"review 这个 PR / 检查这个 PR 有没有问题"不等于授权运行 openEuler Docker 验证。
- review 默认只检查 PR 描述中开发者声明的 openEuler Embedded Verification。如果缺少或不完整，应作为阻塞性 review 问题要求开发者补充。
- 只有当用户在当前请求中明确要求 agent 实际执行 openEuler / 双平台 Docker setup/build 验证时，才运行本 skill。

## Prerequisites

- 宿主机已安装 Docker CLI（检查 `command -v docker`）。
- 宿主机已安装 openEuler aarch64 验证所需的 `qemu-user-static`（只检查不自动安装；缺失时提醒用户 `sudo apt install -y qemu-user-static`）。
- 当前用户有运行容器的权限。
- **必须检查验证镜像**：本地不存在或创建超过 30 天则重新拉取 `swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:env`。
- 网络：可访问 `repo.openeuler.org`、`eur.openeuler.openatom.cn`、`eulermaker.compass-ci.openeuler.openatom.cn`、华为 pip 镜像、`gitcode.com`。
- IB-Robot workspace 中有待验证的修改。

## Container Architecture

```
┌─ Docker container (x86_64) ──────────────────────┐
│  entrypoint: /bin/bash -l                        │
│  .bashrc auto-chroots on interactive login       │
│  ┌─ chroot /root/openeuler_rootfs (aarch64) ────┐│
│  │  openEuler Embedded Reference Distro         ││
│  │  openEuler ROS repos ( Embedded + SIG )      ││
│  │  ROS 2 Humble, python3, dnf, git, pip cache  ││
│  │  workspace at /root/IB_Robot                 ││
│  └───────────────────────────────────────────────┘│
└───────────────────────────────────────────────────┘
```

> **关于 ROS 安装：** `:env` 镜像已预装 ROS 2 Humble，因此本流程验证 setup.sh 的
> ROS 检测和复用路径。若要验证从零安装 ROS，必须另用不含 ROS 的基础镜像，不能把
> 本流程的成功结果描述为完整验证了 `install_ros.sh`。

所有 `docker exec` 命令需要通过 `chroot /root/openeuler_rootfs` 进入 arm64 环境。

> **注意：** 容器内 `which` 命令在 rootfs 中行为异常（即使工具已安装也报 not found），
> 应使用 `type git` 或直接用绝对路径 `/usr/bin/git` 来检测工具。

## Container Naming Convention

| Variable       | Value                                                    |
|----------------|----------------------------------------------------------|
| Container name | `verify-oee`                                             |
| User           | `root`（openEuler Embedded 默认 root 操作）              |
| Image          | `swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:env` |
| Rootfs path    | `/root/openeuler_rootfs`（容器内，chroot 前）            |
| Workspace      | `/root/openeuler_rootfs/root/IB_Robot`（chroot 内路径）  |
| Host workspace | 宿主机上 IB_Robot 项目根目录                              |

## Core Principle

> **setup.sh 和 build.sh 是唯一合法的软件安装途径。**

- 禁止手动 `dnf install` / `pip install` / `sed` patch 脚本 / 手动配置 ROS 仓库。
- 允许修复 chroot 基础设施（DNS、`/var/log`、`git safe.directory`）、通过环境变量传递镜像 URL。
- 完整的禁止/允许事项清单和论述见 [references/discipline.md](references/discipline.md)。

## Error Classification

验证完成后，必须收集 `setup.log` 和 `build.log` 中的所有 ERROR 行，并分类报告：

| 分类 | 定义 | 处理方式 |
|------|------|----------|
| **Fatal** | setup.sh 或 build.sh 未输出完成消息（`Setup complete` / `Build complete`）；进程异常退出 | 必须报告为阻塞问题 |
| **Non-fatal** | pip dependency resolver 警告；rosdep keys 未解析；updmap 字体映射缺失；其他不影响最终构建结果的警告 | 列出但标注为 non-fatal，不阻塞 |

报告时必须逐条列出所有 ERROR 行内容，并标注分类。

## Procedure Overview

Each phase's complete bash commands and detailed rationale live in
[references/procedure.md](references/procedure.md). The summary below is for
routing and mental model only; execute the actual commands by reading the
corresponding phase section.

| Phase | Purpose | Key Output |
|-------|---------|------------|
| **0** | Check host prerequisites (docker, qemu-user-static) | Pass/fail |
| **1** | Ensure `:env` image is fresh (pull if >30 days old) | Image ready |
| **2** | Start container, verify aarch64 emulation, fix chroot env (DNS, /var/log, git safe.directory) | Container ready |
| **3** | Inspect chroot environment (git, python3, dnf, ROS 2) | Environment confirmed |
| **4** | Prepare workspace — **choose one mode** (docker cp or git clone) | `SOURCE_MODE` |
| **5** | Run `setup.sh --yes --no-sudo`, capture log (20-40 min under qemu) | `/tmp/setup.log` |
| **6** | Run `build.sh`, capture log | `/tmp/build.log` |
| **7** | Collect ERROR lines, clean up container | Final report |

### Phase 4 Source Modes

- **方式 A — docker cp（默认）**: 从宿主机拷入当前项目目录，适合验证本地未提交改动。
- **方式 B — git clone**: 在容器内 clone 远程分支，适合验证已推送代码。注意 rootfs 中 `which` 不可靠，git 命令用绝对路径 `/usr/bin/git`。

## Variants

- **Iterative local testing** (re-run without full Phase 0-3):
  use the [quick-run one-liner](references/quick-run.md) to copy changed scripts and re-run setup.

## Key Differences from Ubuntu Verification

| Aspect | Ubuntu 22.04 (`ibrobot-docker-verify`) | openEuler Embedded (this skill) |
|--------|----------------------------------------|--------------------------------|
| Architecture | x86_64 native | aarch64 via qemu-user chroot |
| **User** | `testuser` + NOPASSWD sudo | **`root`（无需 sudo）** |
| **Command prefix** | `docker exec verify-ubuntu2204` | **`docker exec verify-oee bash -c 'chroot /root/openeuler_rootfs /bin/bash -c "..."'`** |
| Package manager | apt | dnf |
| ROS install | setup.sh 自动调用 `install_ros.sh` 安装 | setup.sh 自动调用 `install_ros.sh` 安装 |
| Speed | Native (fast) | Emulated (20-40 min for setup) |
| DNS | Works by default | Must copy resolv.conf |
| git-lfs | Available | Not available; lerobot_patches.sh auto-removes LFS hook |

## Troubleshooting

Common issues (DNS resolution, `/var/log` symlink, `dubious ownership`, GPGMEError, git-lfs hook,
`which` unreliable under qemu, pip ReadTimeout, etc.) are documented in
[references/pitfalls.md](references/pitfalls.md). Consult that table when a verification run fails
with an unfamiliar error.

## Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `IBR_LEROBOT_FORCE_REBUILD` | `1` | Rebuild lerobot patch branch from base commit |
| `ROSDISTRO_INDEX_URL` | Set by `setup.sh` | TUNA mirror for rosdistro index |
| `ROS_OS_OVERRIDE` | `rhel:8` | Set by platform script for rosdep compatibility |
| `SETUP_PIP_INDEX_URL` | Huawei mirror | Configured in `${VENV_PATH}/pip.conf` by setup |
