---
name: ibrobot-docker-verify
description: "Execute setup.sh + build.sh in a clean Ubuntu 22.04 ROS desktop-full Docker container with a host pip cache mount and stage timing. Use when the user explicitly asks for Docker/setup/build verification, or when author-side PR creation/update workflows trigger the dependency/setup verification gate. Do not use automatically during PR review; review should check developer-provided Verification in the PR description."
---

# IB-Robot Docker Verification Skill

Full end-to-end validation of `setup.sh` and `build.sh` inside a clean Ubuntu
22.04 container based on ROS 2 Humble desktop-full. The default flow copies the
current host workspace so uncommitted changes can be tested, while removing
host-built artifacts before creating a fresh venv and colcon workspace. A
clean remote clone is available only when the user explicitly requests it.

The ROS-ready image deliberately skips ROS first-install testing. Changes to
`scripts/install_ros.sh`, ROS repository setup, or ROS GPG-key handling require
the bootstrap variant documented below, based on plain `ubuntu:22.04`.

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
- If the user explicitly requests a remote commit or branch, its repository is
  reachable from the container.
- The host pip cache directory exists at `${PIP_CACHE_DIR:-$HOME/.cache/pip}`.
- Network access to Aliyun apt mirror, TUNA ROS 2 repo, Huawei pip mirror,
  and `gitcode.com` / `atomgit.com` for lerobot submodule fetch.

## Container Naming Convention

| Variable      | Value                        |
|---------------|------------------------------|
| Container name | `verify-ubuntu2204`          |
| User           | `testuser`                   |
| Workspace      | `/home/testuser/IB_Robot`    |
| Image          | `osrf/ros:humble-desktop-full-jammy` |
| Pip cache      | `/var/cache/ibrobot-pip`      |

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
- **挂载宿主机 pip cache**：只复用下载缓存，禁止挂载宿主机 `venv`、
  `build`、`install` 或整个源码工作区。默认流程使用 `docker cp` 复制源码，
  不是 bind mount，并在容器内删除宿主机构建产物。

### 错误分类与报告

验证完成后，必须收集 `setup.log` 和 `build.log` 中的所有 ERROR 行，并分类报告：

| 分类 | 定义 | 处理方式 |
|------|------|----------|
| **Fatal** | setup.sh 或 build.sh 未输出完成消息（`Setup complete` / `Build complete`）；进程异常退出 | 必须报告为阻塞问题 |
| **Non-fatal** | pip dependency resolver 警告（如 numpy 版本冲突）；rosdep keys 未解析（自定义包不在 rosdistro）；其他不影响最终构建结果的警告 | 列出但标注为 non-fatal，不阻塞 |

报告时必须逐条列出所有 ERROR 行内容，并标注分类。

## Procedure

### Phase 0 — Define Inputs and Check Host Prerequisites

> Run this on the host before any Docker operation. Do not continue if the
> `docker` command is unavailable.

```bash
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is not installed. Install Docker on the host, then rerun verification."
  exit 1
fi

IMAGE=osrf/ros:humble-desktop-full-jammy
CONTAINER=verify-ubuntu2204
PROJECT_ROOT=$(git rev-parse --show-toplevel)
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_PIP_CACHE=${PIP_CACHE_DIR:-"${HOME}/.cache/pip"}

if [ ! -d "${HOST_PIP_CACHE}" ]; then
  echo "Host pip cache does not exist: ${HOST_PIP_CACHE}"
  exit 1
fi

TOTAL_START=$(date +%s)
```

The source mode is selected in Phase 3. Use the default local workspace copy
unless the user explicitly asks for a clean remote branch or commit.

### Phase 1 — Prepare ROS Desktop-Full Image

> The desktop-full image matches the `ros-humble-desktop-full` level installed
> by `install_ros.sh`. Do not use `ros:humble-ros-base-jammy`; rosdep would need
> to install omitted desktop, RViz, rqt, xacro, and robot-state-publisher
> dependencies during every verification.

```bash
IMAGE_START=$(date +%s)
docker pull "${IMAGE}"
IMAGE_SECONDS=$(( $(date +%s) - IMAGE_START ))
```

The first pull is network-dependent and should be reported separately. Later
runs normally reuse the local image layers.

### Phase 2 — Create and Provision Container

```bash
# 2.1 Remove a stale verification container, then start the ROS-ready image.
CONTAINER_START=$(date +%s)
docker rm -f "${CONTAINER}" >/dev/null 2>&1 || true
docker run -d --name "${CONTAINER}" \
  --entrypoint /bin/bash \
  --mount "type=bind,src=${HOST_PIP_CACHE},dst=/var/cache/ibrobot-pip" \
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip \
  -e TZ=Asia/Shanghai \
  -e DEBIAN_FRONTEND=noninteractive \
  "${IMAGE}" -c 'sleep infinity'

# 2.2 Configure the OS mirror and install container bootstrap tools only.
docker exec "${CONTAINER}" bash -c '
  sed -i "s|http://archive.ubuntu.com|http://mirrors.aliyun.com|g;
          s|http://security.ubuntu.com|http://mirrors.aliyun.com|g" \
    /etc/apt/sources.list &&
  apt-get update -qq &&
  apt-get install -y -qq \
    sudo git git-lfs locales python3 curl \
    gnupg2 lsb-release software-properties-common \
    > /dev/null 2>&1
'

# 2.3 Match the container user to the host cache owner.
docker exec \
  -e HOST_UID="${HOST_UID}" \
  -e HOST_GID="${HOST_GID}" \
  "${CONTAINER}" bash -c '
    if ! getent group "${HOST_GID}" >/dev/null; then
      groupadd -g "${HOST_GID}" testuser
    fi
    group_name=$(getent group "${HOST_GID}" | cut -d: -f1)
    useradd -m -u "${HOST_UID}" -g "${group_name}" -s /bin/bash testuser
    echo "testuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/testuser
    locale-gen en_US.UTF-8 >/dev/null
    test -w /var/cache/ibrobot-pip
  '

CONTAINER_SECONDS=$(( $(date +%s) - CONTAINER_START ))
```

Mount pip at `/var/cache/ibrobot-pip`, not below `/home/testuser/.cache`.
Docker may create missing parent directories as root before `useradd`, which
can later break tools such as pre-commit even when the pip directory itself is
writable.

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

### Phase 3 — Prepare the Source Workspace

Choose exactly one source mode. Record the selected mode and source details in
the verification result and in the PR description's Verification section.

#### Default: Copy the Current Host Workspace

Use this mode unless the user explicitly requests a remote clone. It preserves
committed, uncommitted, and untracked changes from the current IB-Robot
workspace.

```bash
SOURCE_MODE="local workspace copy"
SOURCE_DETAILS="${PROJECT_ROOT}"

# Record exactly what local state is being tested before setup mutates
# submodules or installs hooks.
git -C "${PROJECT_ROOT}" rev-parse HEAD
git -C "${PROJECT_ROOT}" status --short --branch

docker cp "${PROJECT_ROOT}" "${CONTAINER}":/home/testuser/IB_Robot
docker exec "${CONTAINER}" \
  chown -R "${HOST_UID}:${HOST_GID}" /home/testuser/IB_Robot

# Never reuse host-specific virtual environments or build products.
docker exec "${CONTAINER}" bash -c '
  rm -rf /home/testuser/IB_Robot/{venv,build,install,log}
'

docker exec -u testuser -w /home/testuser/IB_Robot "${CONTAINER}" \
  git status --short --branch
```

If the host workspace contains large unrelated untracked files, report that
fact before copying. Do not silently switch to a remote clone because doing so
would omit the uncommitted changes the user may intend to verify.

#### Optional: Clone a Clean Remote Branch

Use this mode only when the user explicitly asks to validate a pushed commit,
fork branch, or clean upstream state. Replace the placeholders with the exact
requested remote and branch; do not default them silently.

```bash
SOURCE_MODE="remote clean clone"
SOURCE_DETAILS="<repo_url> branch <branch>"

docker exec \
  -u testuser \
  -e HOME=/home/testuser \
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip \
  -w /home/testuser \
  "${CONTAINER}" \
  git clone --branch "<branch>" --single-branch \
    "<repo_url>" /home/testuser/IB_Robot

docker exec -u testuser -w /home/testuser/IB_Robot "${CONTAINER}" \
  git status --short --branch
```

The remote-clone status must contain only the branch/tracking line before
setup. For either mode, record `git rev-parse HEAD`; for a local copy, also
preserve and report the pre-setup dirty status.

### Phase 4 — Run setup.sh

```bash
docker exec \
  -u testuser \
  -e HOME=/home/testuser \
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip \
  -e IBR_LEROBOT_FORCE_REBUILD=1 \
  -e DEBIAN_FRONTEND=noninteractive \
  -w /home/testuser/IB_Robot \
  "${CONTAINER}" \
  bash -c '
    SECONDS=0
    bash scripts/setup.sh --yes > /tmp/setup.log 2>&1
    status=$?
    printf "%s\n" "${status}" > /tmp/setup.status
    printf "SETUP_ELAPSED_SECONDS=%s\n" "${SECONDS}" > /tmp/setup.time
    exit "${status}"
  '
```

Verify the status is zero and the log contains:

```
Setup complete! Run ./scripts/build.sh to build the workspace.
```

### Phase 5 — Run build.sh

```bash
docker exec \
  -u testuser \
  -e HOME=/home/testuser \
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip \
  -e DEBIAN_FRONTEND=noninteractive \
  -w /home/testuser/IB_Robot \
  "${CONTAINER}" \
  bash -c '
    SECONDS=0
    bash scripts/build.sh > /tmp/build.log 2>&1
    status=$?
    printf "%s\n" "${status}" > /tmp/build.status
    printf "BUILD_ELAPSED_SECONDS=%s\n" "${SECONDS}" > /tmp/build.time
    exit "${status}"
  '
```

Verify the status is zero and the log contains:

```
Build complete. Source with: source install/setup.sh
```

### Phase 6 — Inspect, Report Timing, and Clean Up

```bash
# Collect source, status, timing, commit, and all ERROR lines with context.
printf 'source_mode=%s\n' "${SOURCE_MODE}"
printf 'source_details=%s\n' "${SOURCE_DETAILS}"
docker exec "${CONTAINER}" bash -c '
  cat /tmp/setup.status /tmp/setup.time
  cat /tmp/build.status /tmp/build.time
  git -C /home/testuser/IB_Robot rev-parse HEAD
  grep -n -E "ERROR|Error|error" /tmp/setup.log || true
  grep -n -E "ERROR|Error|error|Failed|failed" /tmp/build.log || true
'

TOTAL_SECONDS=$(( $(date +%s) - TOTAL_START ))
printf 'image_prepare=%ss\n' "${IMAGE_SECONDS}"
printf 'container_start=%ss\n' "${CONTAINER_SECONDS}"
docker exec "${CONTAINER}" cat /tmp/setup.time /tmp/build.time
printf 'total=%ss\n' "${TOTAL_SECONDS}"

# Clean up
docker stop "${CONTAINER}" && docker rm "${CONTAINER}"
```

> **错误报告要求**：必须逐条列出所有 ERROR 行，并按 Verification Discipline 中的分类标准标注 Fatal / Non-fatal。不能只给 ERROR 行数不给内容。

Report both totals when relevant:

- Cold total including the first image pull.
- Steady-state total using an already-present desktop-full image.

Also record the pip cache size before and after the run. A warm cache result
must not be presented as a cold-cache baseline.

The final report and any PR Verification section must explicitly state one of:

- `Source: local workspace copy`, with the host commit and whether the copied
  workspace contained uncommitted or untracked changes.
- `Source: remote clean clone`, with the exact repository URL, branch, and
  tested commit.

## Bootstrap Variant for ROS Installation Changes

The default desktop-full flow verifies dependency convergence and the complete
workspace build, but it cannot exercise the missing-ROS branch in
`install_ros.sh`. For changes to ROS installation, repository, or GPG-key
logic, repeat the same procedure with these differences:

```bash
IMAGE=ubuntu:22.04
```

- Keep the selected Phase 3 source mode and host pip cache mount.
- Let `setup.sh --yes` install ROS without manual intervention.
- Report the bootstrap result separately from the faster ROS-ready result.
- For release validation, optionally run with an empty pip cache directory to
  measure a true cold start.

## Quick-Run One-Liner (for Iterative Testing)

When iterating locally, repeat the matching Phase 3 source preparation rather
than changing source modes implicitly:

```bash
# Default local-copy mode: replace the previous container copy.
docker exec verify-ubuntu2204 rm -rf /home/testuser/IB_Robot
docker cp "${PROJECT_ROOT}" verify-ubuntu2204:/home/testuser/IB_Robot
docker exec verify-ubuntu2204 \
  chown -R "${HOST_UID}:${HOST_GID}" /home/testuser/IB_Robot

# Both modes: remove host or previous-run artifacts before setup.
docker exec verify-ubuntu2204 bash -c \
  'rm -rf /home/testuser/IB_Robot/{venv,build,install,log}'

# Re-run
docker exec -u testuser -e HOME=/home/testuser \
  -e PIP_CACHE_DIR=/var/cache/ibrobot-pip \
  -e IBR_LEROBOT_FORCE_REBUILD=1 \
  -w /home/testuser/IB_Robot \
  verify-ubuntu2204 \
  bash -c 'DEBIAN_FRONTEND=noninteractive \
    bash scripts/setup.sh --yes \
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
| `Permission denied: ~/.cache/pre-commit` | Bind mount caused Docker to create the home cache parent as root | Mount pip at `/var/cache/ibrobot-pip`, not below `~/.cache` |
| `The build time path ... doesn't exist` | Copied or iterative source retained host-specific venv/build paths | Remove `venv build install log`, then rerun setup from the selected source |
| `git: command not found` mid-setup | `install_ros.sh` apt install may remove git | Phase 1 already installed git; re-run `apt-get install -y git git-lfs` if needed |
| lerobot patch stack fetch fails | Submodule base commit not in local checkout | Rebase branch onto `upstream/master` before copying |

## Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `IBR_LEROBOT_FORCE_REBUILD` | `1` | Rebuild lerobot patch branch in container |
| `DEBIAN_FRONTEND` | `noninteractive` | Prevent tzdata etc. from blocking |
| `PIP_CACHE_DIR` | `/var/cache/ibrobot-pip` | Reuse host pip downloads without reusing the venv |
| `ROSDISTRO_INDEX_URL` | Set by `setup.sh` | TUNA mirror for rosdistro index |
| `SETUP_PIP_INDEX_URL` | Huawei mirror | Configured in `${VENV_PATH}/pip.conf` by setup |
