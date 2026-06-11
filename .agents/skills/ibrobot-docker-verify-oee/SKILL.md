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
- 用户要求“review 这个 PR / 检查这个 PR 有没有问题”不等于授权运行 openEuler Docker 验证。
- review 默认只检查 PR 描述中开发者声明的 openEuler Embedded Verification。如果缺少或不完整，应作为阻塞性 review 问题要求开发者补充。
- 只有当用户在当前请求中明确要求 agent 实际执行 openEuler / 双平台 Docker setup/build 验证时，才运行本 skill。

## Prerequisites

- Docker 已安装且当前用户有运行容器的权限
- **必须先检查验证镜像**，不能假设开发者本机已有该镜像；若本地不存在，或本地镜像创建时间距本次验证超过 30 天，则重新拉取：
  ```bash
  IMAGE=swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:latest
  CREATED=$(docker image inspect "$IMAGE" --format '{{.Created}}' 2>/dev/null || true)
  if [ -z "$CREATED" ]; then
    docker pull "$IMAGE"
  else
    NOW=$(date +%s)
    CREATED_TS=$(date -d "$CREATED" +%s)
    AGE_DAYS=$(( (NOW - CREATED_TS) / 86400 ))
    if [ "$AGE_DAYS" -gt 30 ]; then
      docker pull "$IMAGE"
    fi
  fi
  ```
- 网络：可访问 `repo.openeuler.org`、`eur.openeuler.openatom.cn`、
  `eulermaker.compass-ci.openeuler.openatom.cn`、华为 pip 镜像、`gitcode.com`
- IB-Robot workspace 中有待验证的修改

## Container Architecture

```
┌─ Docker container (x86_64) ──────────────────────┐
│  entrypoint: /bin/bash -l                        │
│  .bashrc auto-chroots on interactive login       │
│  ┌─ chroot /root/openeuler_rootfs (aarch64) ────┐│
│  │  openEuler Embedded Reference Distro         ││
│  │  openEuler ROS repos ( Embedded + SIG )      ││
│  │  python3, dnf, git                           ││
│  │  workspace at /root/IB_Robot                 ││
│  └───────────────────────────────────────────────┘│
└───────────────────────────────────────────────────┘
```

> **关于 ROS 安装：** setup.sh 会自动检测 ROS 是否已安装。若未安装，会调用
> `scripts/install_ros.sh` 完成安装（配置 openEuler ROS repo + dnf 安装）。
> **不需要也不应该手动预装 ROS**，让 setup.sh 完整跑一遍才能验证安装流程。

所有 `docker exec` 命令需要通过 `chroot /root/openeuler_rootfs` 进入 arm64 环境。

> **注意：** 容器内 `which` 命令在 rootfs 中行为异常（即使工具已安装也报 not found），
> 应使用 `type git` 或直接用绝对路径 `/usr/bin/git` 来检测工具。

## Container Naming Convention

| Variable       | Value                                                    |
|----------------|----------------------------------------------------------|
| Container name | `verify-oee`                                             |
| User           | `root`（openEuler Embedded 默认 root 操作）              |
| Image          | `swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:latest` |
| Rootfs path    | `/root/openeuler_rootfs`（容器内，chroot 前）            |
| Workspace      | `/root/openeuler_rootfs/root/IB_Robot`（chroot 内路径）  |
| Host workspace | 宿主机上 IB_Robot 项目根目录                              |

## Procedure

### Phase 0 — Ensure Required Image Is Fresh

> **必须执行。** openEuler Embedded 验证不使用本地默认镜像名，也不假设镜像已预装。
> 如果本地没有镜像，或镜像创建时间距本次验证超过 30 天，需要先从 SWR 重新拉取。
> 通过 `date` 获取当前时间，通过 `docker image inspect --format '{{.Created}}'` 获取本地镜像创建时间。

```bash
IMAGE=swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:latest
CREATED=$(docker image inspect "$IMAGE" --format '{{.Created}}' 2>/dev/null || true)

if [ -z "$CREATED" ]; then
  echo "Image not found locally, pulling: $IMAGE"
  docker pull "$IMAGE"
else
  NOW=$(date +%s)
  CREATED_TS=$(date -d "$CREATED" +%s)
  AGE_DAYS=$(( (NOW - CREATED_TS) / 86400 ))
  if [ "$AGE_DAYS" -gt 30 ]; then
    echo "Image is ${AGE_DAYS} days old, refreshing: $IMAGE"
    docker pull "$IMAGE"
  else
    echo "Using local image created ${AGE_DAYS} days ago: $IMAGE"
  fi
fi
```

### Phase 1 — Start Container and Fix chroot Environment

```bash
# 1.1 Start detached container (--entrypoint override to keep container alive)
#     NOTE: The image's default entrypoint /bin/bash -l triggers auto-chroot
#     via .bashrc on interactive login. For detached mode we override to
#     keep the container running, then manually chroot as needed.
docker run -d --name verify-oee --privileged \
  --entrypoint /bin/bash \
  swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:latest \
  -c "sleep infinity"

# 1.2 Verify aarch64 emulation
docker exec verify-oee chroot /root/openeuler_rootfs uname -m
# Expected: aarch64

# 1.3 Fix DNS (rootfs has no /etc/resolv.conf)
docker exec verify-oee bash -c \
  'rm -f /root/openeuler_rootfs/etc/resolv.conf && cp /etc/resolv.conf /root/openeuler_rootfs/etc/resolv.conf'

# 1.4 Fix /var/log (symlink target missing in rootfs)
docker exec verify-oee bash -c \
  'mkdir -p /root/openeuler_rootfs/var/volatile/log'

# 1.5 Mount /proc and /sys for chroot compatibility
docker exec verify-oee bash -c \
  'mount -t proc proc /root/openeuler_rootfs/proc 2>/dev/null; mount --bind /sys /root/openeuler_rootfs/sys 2>/dev/null'

# 1.6 Fix git safe.directory for UID mismatch after docker cp
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs git config --global --add safe.directory /root/IB_Robot
   chroot /root/openeuler_rootfs git config --global --add safe.directory /root/IB_Robot/libs/lerobot'
```

**Why --privileged:** chroot 和 qemu-user binfmt 模拟需要 privileged 权限。

**Why --entrypoint override:** 镜像默认 entrypoint `/bin/bash -l` 会在交互模式下
通过 `.bashrc` 自动 chroot。但 detached 模式下需要 override 以保持容器存活，
然后通过 `docker exec ... chroot /root/openeuler_rootfs` 手动进入 arm64 环境。

**Why chroot:** 容器镜像的 `.bashrc` 在交互 login 时自动
`exec chroot /root/openeuler_rootfs /bin/bash`。`docker exec` 命令在容器宿主空间
执行，必须手动 `chroot /root/openeuler_rootfs` 才能进入 arm64 环境。

### Phase 2 — Inspect chroot Environment

```bash
docker exec verify-oee bash -c 'chroot /root/openeuler_rootfs /bin/bash -c "
  uname -a
  cat /etc/os-release | head -3
  type git python3 dnf
  /usr/bin/git --version
  /usr/bin/python3 --version
  ls /opt/ros/humble/setup.bash 2>/dev/null || echo no-ros-yet
  cat /etc/yum.repos.d/openEulerROS.repo
"'
```

容器镜像应包含：git、python3、dnf + 两个 openEuler ROS repo 配置。
ROS 2 安装由 setup.sh 通过 `install_ros.sh` 自动完成，无需手动干预。

### Phase 3 — Prepare Workspace

提供两种方式，根据场景选择：

#### 方式 A：从宿主机 docker cp（适合本地代码验证）

> 用于验证本地未提交的改动。`docker cp` 拷入当前项目目录，可直接验证修改效果。

```bash
# 3.1 Copy current workspace into rootfs
docker cp <project_root> verify-oee:/root/openeuler_rootfs/root/IB_Robot

# 3.2 Remove stale artifacts (host paths are wrong in container)
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
```

#### 方式 B：在容器内 git clone（适合验证远程分支）

> 用于验证已推送到个人仓库分支的代码。注意 rootfs 中 `which` 不可靠，
> git 命令请用绝对路径 `/usr/bin/git`。

```bash
# 3.1 Clone the branch inside chroot
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "
    cd /root && /usr/bin/git clone -b <branch> <repo_url> /root/IB_Robot
  "'

# 3.2 Remove stale artifacts (if any)
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
```

### Phase 4 — Run setup.sh

```bash
docker exec -d verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "cd /root/IB_Robot && IBR_LEROBOT_FORCE_REBUILD=1 bash scripts/setup.sh --yes --no-sudo > /tmp/setup.log 2>&1"'
```

Monitor progress (qemu-user 模拟下 pip 安装很慢，全流程约 20-40 min):

```bash
# Poll (note: always chroot to read log)
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "tail -15 /tmp/setup.log"'
```

Wait until the log ends with:

```
Setup complete! Run ./scripts/build.sh to build the workspace.
```

### Phase 5 — Run build.sh

```bash
docker exec -d verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "cd /root/IB_Robot && source /opt/ros/humble/setup.bash && source venv/bin/activate && bash scripts/build.sh > /tmp/build.log 2>&1"'
```

Monitor:

```bash
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "tail -15 /tmp/build.log"'
```

Wait until:

```
Build complete. Source with: source install/setup.sh
```

### Phase 6 — Inspect and Clean Up

```bash
# Check results
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "grep -c ERROR /tmp/setup.log"'
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "grep -c ERROR /tmp/build.log"'

# Clean up
docker stop verify-oee && docker rm verify-oee
```

## Quick-Run (Iterative Testing)

When only scripts changed (ROS 2 + system deps already installed), copy
updated files and re-run without recreating the container:

```bash
# Copy changed files into rootfs
docker cp scripts/setup.sh verify-oee:/root/openeuler_rootfs/root/IB_Robot/scripts/
docker cp scripts/setup/platforms/openeuler-embedded-24.03.sh \
  verify-oee:/root/openeuler_rootfs/root/IB_Robot/scripts/setup/platforms/
docker cp scripts/setup/lerobot_patches.sh \
  verify-oee:/root/openeuler_rootfs/root/IB_Robot/scripts/setup/

# Clean and re-run
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
docker exec -d verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "cd /root/IB_Robot && IBR_LEROBOT_FORCE_REBUILD=1 bash scripts/setup.sh --yes --no-sudo > /tmp/setup.log 2>&1"'
```

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

## Known Pitfalls

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Couldn't resolve host name` | rootfs missing `/etc/resolv.conf` | Phase 1.3 copies from host container |
| `Config error: File exists: /var/log` | `/var/log` symlink target missing | Phase 1.4 creates `/var/volatile/log` |
| `/dev/stdout: No such file or directory` | `/proc` not mounted in chroot | Phase 1.5 mounts proc/sys |
| `dubious ownership in repository` | UID mismatch after `docker cp` | Phase 1.7 adds `safe.directory` |
| `gpg.errors.GPGMEError` during `rosdep install` | qemu-aarch64 emulation bug with Python `gpg` | setup.sh 自动禁用 `gpgcheck` |
| `git-lfs was not found` post-checkout hook | No git-lfs in rootfs | `lerobot_patches.sh` auto-removes hook when git-lfs missing |
| `ERROR: file:///root/IB_Robot/libs/lerobot does not appear to be a Python project` | Copied a linked worktree or an uninitialized submodule tree into the container, then ran setup without `--skip-submodules` | Use a standalone clone and run `git submodule update --init --recursive` before `docker cp` |
| `pip3 not found, cannot install colcon` | `platform_install_python_bootstrap` not called before `ensure_colcon` | `install_system_deps` calls bootstrap first |
| `python%{python3_pkgversion}-scipy` not found | `ROS_OS_OVERRIDE=rhel:8` uses RHEL naming; openEuler dnf can't match macro | Platform script skips `python3-scipy` in rosdep, installs via explicit `dnf install` |
| `rosdep install failed` for missing packages | Some ROS packages not in openEuler repos (e.g. `robot_localization`) | Platform script uses non-fatal rosdep + skip-keys |
| dnf outputs config dump instead of installing | Running dnf without `--nogpgcheck --setopt=strict=0` in chroot | Always use `dnf install -y --nogpgcheck` |
| `which git` returns nothing despite git being installed | `which` binary in rootfs behaves incorrectly under qemu-user | Use `type git` or absolute path `/usr/bin/git` instead |
| `ReadTimeoutError: files.pythonhosted.org` during pip install | Network instability under qemu-user emulation | Set `export PIP_DEFAULT_TIMEOUT=120` before running setup.sh |

## Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `IBR_LEROBOT_FORCE_REBUILD` | `1` | Rebuild lerobot patch branch from base commit |
| `ROSDISTRO_INDEX_URL` | Set by `setup.sh` | TUNA mirror for rosdistro index |
| `ROS_OS_OVERRIDE` | `rhel:8` | Set by platform script for rosdep compatibility |
| `SETUP_PIP_INDEX_URL` | Huawei mirror | Configured in `${VENV_PATH}/pip.conf` by setup |
