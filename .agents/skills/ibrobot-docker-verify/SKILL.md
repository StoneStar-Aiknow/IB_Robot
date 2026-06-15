---
name: ibrobot-docker-verify
description: "Execute setup.sh + build.sh in a clean Ubuntu 22.04 Docker container. Use when the user explicitly asks for Docker/setup/build verification, or when author-side PR creation/update workflows trigger the dependency/setup verification gate. Do not use automatically during PR review; review should check developer-provided Verification in the PR description."
---

# IB-Robot Docker Verification Skill

Full end-to-end validation of `setup.sh` and `build.sh` inside a pristine
Ubuntu 22.04 container — the closest approximation to a user's first-run
experience without requiring real hardware.

## When to Use

- User explicitly requests "Docker 验证" / "container test" / "实际验证 setup/build".
- An author-side PR creation/update workflow (`atomgit-pr` or `ibrobot-git-flow`)
  triggers the dependency/setup verification gate and needs real results for the
  PR description.
- The current task is to validate local changes to `scripts/setup.sh`,
  `scripts/setup/platforms/*.sh`, `scripts/setup/verify_env.sh`,
  `scripts/install_ros.sh`, or pip/apt dependency resolution.
- Do not infer this skill from PR review alone.

## Review Boundary

- During PR review, do **not** run this skill automatically just because a PR
  touches `package.xml`, `setup.py`, setup scripts, or build files.
- A user asking "review this PR" or "check whether this PR is OK" is not an
  explicit request to run Docker verification.
- PR review should inspect the developer-provided Verification section in the
  PR description. If required Ubuntu verification is missing or incomplete,
  raise a blocking review issue asking the developer to provide it.
- Only run this skill in a review session when the user explicitly asks the
  agent to perform the actual Ubuntu Docker setup/build verification.

## Prerequisites

- Docker CLI installed on the host. If `docker` is missing, stop and ask the
  user to install Docker before verification:
  ```bash
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker CLI is not installed. Install Docker on the host, then rerun verification."
    exit 1
  fi
  ```
- The current user has permission to run containers.
- The IB-Robot workspace has uncommitted or committed changes to validate.
- Network access to Aliyun apt mirror, TUNA ROS 2 repo, Huawei pip mirror,
  and `gitcode.com` / `atomgit.com` for lerobot submodule fetch.

## Container Naming Convention

| Variable      | Value                        |
|---------------|------------------------------|
| Container name | `verify-ubuntu2204`          |
| User           | `testuser`                   |
| Workspace      | `/home/testuser/IB_Robot`    |

## Verification Discipline

> **核心原则：setup.sh 和 build.sh 是唯一合法的软件安装途径。**

### 禁止事项

- **禁止手动安装软件包**：不允许在容器中手动执行 `apt install`、`pip install` 来安装 ROS、Python 包、系统依赖等。所有软件安装必须通过 `setup.sh` 完成。手动安装会绕过脚本逻辑，使验证结果失去意义。
- **禁止手动 patch 脚本**：不允许用 `sed`、`cp` 等方式修改容器内的 `install_ros.sh`、`setup.sh` 或平台脚本。如果脚本存在网络适配缺陷（如 GPG key URL 不可达、pip 镜像未传递），应修复脚本本身并提交到代码仓库，而不是在验证时临时 patch。
- **禁止手动配置 ROS 仓库或 GPG key**：ROS 安装必须由 `install_ros.sh` 完成。

### 允许事项

- **配置 OS 镜像源**：允许在容器创建阶段配置 apt 镜像源（如 Aliyun），这属于容器环境初始化。
- **创建用户和权限**：允许创建 `testuser`、配置 `NOPASSWD` sudo。
- **配置 locale 和时区**。
- **通过环境变量传递镜像 URL**：允许通过 `PIP_INDEX_URL`、`ROS_GPG_KEY` 等环境变量让脚本使用镜像源。这不是"手动安装"，而是让脚本的 `${VAR:-default}` 机制正确工作。

### 错误分类与报告

验证完成后，必须收集 `setup.log` 和 `build.log` 中的所有 ERROR 行，并分类报告：

| 分类 | 定义 | 处理方式 |
|------|------|----------|
| **Fatal** | setup.sh 或 build.sh 未输出完成消息（`Setup complete` / `Build complete`）；进程异常退出 | 必须报告为阻塞问题 |
| **Non-fatal** | pip dependency resolver 警告（如 numpy 版本冲突）；rosdep keys 未解析（自定义包不在 rosdistro）；其他不影响最终构建结果的警告 | 列出但标注为 non-fatal，不阻塞 |

报告时必须逐条列出所有 ERROR 行内容，并标注分类。

## Procedure

### Phase 0 — Check Host Docker CLI

> Run this on the host before any Docker operation. Do not continue if the
> `docker` command is unavailable.

```bash
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is not installed. Install Docker on the host, then rerun verification."
  exit 1
fi
```

### Phase 1 — Create and Provision Container

```bash
# 1.1 Start detached Ubuntu 22.04 container
docker run -d --name verify-ubuntu2204 \
  -e TZ=Asia/Shanghai \
  -e DEBIAN_FRONTEND=noninteractive \
  ubuntu:22.04 tail -f /dev/null

# 1.2 Install core prerequisites as root
docker exec verify-ubuntu2204 bash -c '
  apt-get update -qq &&
  apt-get install -y -qq \
    sudo git git-lfs locales python3 curl \
    gnupg2 lsb-release software-properties-common \
  > /dev/null 2>&1 &&
  useradd -m -s /bin/bash testuser &&
  echo "testuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/testuser &&
  locale-gen en_US.UTF-8 &&
  echo "container ready"
'
```

**Why NOPASSWD is required:** Docker `exec -d` (detached mode) allocates no
tty.  Ubuntu 22.04's default sudoers enables `use_pty`, which makes
`sudo -v` (validate credential cache) require a terminal even when the user
has `NOPASSWD:ALL`.  The `ensure_sudo_session` function in `setup.sh`
already works around this by calling `sudo -n true` first (non-interactive,
no tty needed), but the container still needs `NOPASSWD:ALL` so that
`sudo -n true` succeeds without a password prompt.

If you skip `NOPASSWD:ALL`, setup.sh will fail at `ensure_sudo_session` because `sudo -n true` returns non-zero
without either NOPASSWD or cached credentials — and there is no tty to
cache credentials through in detached mode.

### Phase 2 — Configure Mirrors

> **ROS 安装由 setup.sh 自动完成：** `setup.sh` 检测到 ROS 未安装时会调用
> `install_ros.sh` 自动安装（配置 ROS repo + apt 安装）。
> 不需要也不应该手动预装 ROS，让 setup.sh 完整跑一遍才能验证安装流程。

```bash
# 2.1 Aliyun apt mirror
docker exec verify-ubuntu2204 bash -c '
  sed -i "s|http://archive.ubuntu.com|http://mirrors.aliyun.com|g;
          s|http://security.ubuntu.com|http://mirrors.aliyun.com|g" \
    /etc/apt/sources.list
'
```

### Phase 3 — Copy Workspace

```bash
# 3.1 Copy the workspace into the container
docker cp <project_root> verify-ubuntu2204:/home/testuser/IB_Robot
docker exec verify-ubuntu2204 chown -R testuser:testuser /home/testuser/IB_Robot

# 3.2 Remove stale venv/build/install/log (copied from host, paths are wrong)
docker exec verify-ubuntu2204 bash -c '
  rm -rf /home/testuser/IB_Robot/{venv,build,install,log}
'
```

### Phase 4 — Run setup.sh

```bash
docker exec -d \
  -u testuser \
  -e HOME=/home/testuser \
  -e IBR_LEROBOT_FORCE_REBUILD=1 \
  -w /home/testuser/IB_Robot \
  verify-ubuntu2204 \
  bash -c 'DEBIAN_FRONTEND=noninteractive \
    bash scripts/setup.sh --yes --skip-submodules \
    > /tmp/setup.log 2>&1'
```

Monitor progress (pip downloads of torch/CUDA libraries take 10-20 min):

```bash
# Poll every 60s
docker exec verify-ubuntu2204 bash -c 'tail -5 /tmp/setup.log'
```

Wait until the log ends with:

```
Setup complete! Run ./scripts/build.sh to build the workspace.
```

### Phase 5 — Run build.sh

```bash
docker exec -d \
  -u testuser \
  -e HOME=/home/testuser \
  -w /home/testuser/IB_Robot \
  verify-ubuntu2204 \
  bash -c 'DEBIAN_FRONTEND=noninteractive \
    bash scripts/build.sh > /tmp/build.log 2>&1'
```

Monitor:

```bash
docker exec verify-ubuntu2204 bash -c 'tail -5 /tmp/build.log'
```

Wait until:

```
Build complete. Source with: source install/setup.sh
```

### Phase 6 — Inspect and Clean Up

```bash
# Collect all ERROR lines with context
docker exec verify-ubuntu2204 bash -c 'grep ERROR /tmp/setup.log || echo "(none)"'
docker exec verify-ubuntu2204 bash -c 'grep ERROR /tmp/build.log || echo "(none)"'

# Clean up
docker stop verify-ubuntu2204 && docker rm verify-ubuntu2204
```

> **错误报告要求**：必须逐条列出所有 ERROR 行，并按 Verification Discipline 中的分类标准标注 Fatal / Non-fatal。不能只给 ERROR 行数不给内容。

## Quick-Run One-Liner (for Iterative Testing)

When only the setup/platform scripts changed (ROS 2 already installed in a
running container), copy updated files and re-run without recreating the
container:

```bash
# Copy changed files
docker cp scripts/setup.sh verify-ubuntu2204:/home/testuser/IB_Robot/scripts/
docker cp scripts/setup/platforms/ubuntu-22.04.sh \
  verify-ubuntu2204:/home/testuser/IB_Robot/scripts/setup/platforms/

# Remove stale venv so setup recreates it with new code
docker exec verify-ubuntu2204 bash -c \
  'rm -rf /home/testuser/IB_Robot/{venv,build,install,log}'

# Re-run
docker exec -d -u testuser -e HOME=/home/testuser \
  -e IBR_LEROBOT_FORCE_REBUILD=1 \
  -w /home/testuser/IB_Robot \
  verify-ubuntu2204 \
  bash -c 'DEBIAN_FRONTEND=noninteractive \
    bash scripts/setup.sh --yes --skip-submodules \
    > /tmp/setup.log 2>&1'
```

## Known Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `sudo: a terminal is required to read the password` | Docker exec has no tty; `use_pty` in sudoers blocks `sudo -v` | Phase 1 **must** set `NOPASSWD:ALL`; `setup.sh` code uses `sudo -n true` first |
| `sh: 1: rosdep: not found` | Was caused by rosdepc calling `os.system('rosdep ...')` | Replaced rosdepc with direct `pip install rosdep` |
| `error loading sources list: Permission denied` | `write_rosdep_sources_list` wrote file as 600 root | Now does `chmod 644` after writing |
| `rosdep update` times out | `ROSDISTRO_INDEX_URL` not passed to platform script | Platform scripts now pass `env ROSDISTRO_INDEX_URL=...` |
| pip downloads from pypi.org at ~10 KB/s | No pip mirror configured in container | `ensure_workspace_venv` writes `${VENV_PATH}/pip.conf` |
| `The build time path ... doesn't exist` | Copied host venv has hardcoded `/home/xqw/...` paths | Always `rm -rf venv build install log` before re-running setup |
| `git: command not found` mid-setup | `install_ros.sh` apt install may remove git | Phase 1 already installed git; re-run `apt-get install -y git git-lfs` if needed |
| lerobot patch stack fetch fails | Submodule base commit not in local checkout | Rebase branch onto `upstream/master` before copying |

## Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `IBR_LEROBOT_FORCE_REBUILD` | `1` | Rebuild lerobot patch branch in container |
| `DEBIAN_FRONTEND` | `noninteractive` | Prevent tzdata etc. from blocking |
| `ROSDISTRO_INDEX_URL` | Set by `setup.sh` | TUNA mirror for rosdistro index |
| `SETUP_PIP_INDEX_URL` | Huawei mirror | Configured in `${VENV_PATH}/pip.conf` by setup |
