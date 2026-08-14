# Full Procedure — Phase 0 to Phase 7

> This document contains the complete bash commands for each phase.
> The main SKILL.md only lists the phase purposes; execute the actual
> commands by reading the corresponding section here.

## Phase 0 — Check Host Prerequisites

> **只允许在宿主机执行本节检查命令。** 本节只检查 `docker` 命令和宿主机 apt 包，
> 不执行 chroot、不挂载 rootfs、不进入 aarch64 环境。

```bash
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is not installed. Install Docker on the host, then rerun verification."
  exit 1
fi

missing_pkgs=()
for pkg in qemu-user-static; do
  if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
    missing_pkgs+=("$pkg")
  fi
done

if [ "${#missing_pkgs[@]}" -gt 0 ]; then
  echo "Missing host packages: ${missing_pkgs[*]}"
  echo "Install them on the host with: sudo apt update && sudo apt install -y ${missing_pkgs[*]}"
  exit 1
fi
```

## Phase 1 — Ensure Required Image Is Fresh

> **必须执行。** openEuler Embedded 验证不使用本地默认镜像名，也不假设镜像已预装。
> 如果本地没有镜像，或镜像创建时间距本次验证超过 30 天，需要先从 SWR 重新拉取。
> 通过 `date` 获取当前时间，通过 `docker image inspect --format '{{.Created}}'` 获取本地镜像创建时间。

```bash
IMAGE=swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:env
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

## Phase 2 — Start Container and Fix chroot Environment

```bash
# 2.1 Start detached container (--entrypoint override to keep container alive)
#     NOTE: The image's default entrypoint /bin/bash -l triggers auto-chroot
#     via .bashrc on interactive login. For detached mode we override to
#     keep the container running, then manually chroot as needed.
docker run -d --name verify-oee --privileged \
  --entrypoint /bin/bash \
  swr.cn-north-4.myhuaweicloud.com/openeuler-embedded-2/openeuler-ibrobot-dev:env \
  -c "sleep infinity"

# 2.2 Verify aarch64 emulation
docker exec verify-oee chroot /root/openeuler_rootfs uname -m
# Expected: aarch64

# 2.3 Fix DNS (rootfs has no /etc/resolv.conf)
docker exec verify-oee bash -c \
  'rm -f /root/openeuler_rootfs/etc/resolv.conf && cp /etc/resolv.conf /root/openeuler_rootfs/etc/resolv.conf'

# 2.4 Fix /var/log (symlink target missing in rootfs)
docker exec verify-oee bash -c \
  'mkdir -p /root/openeuler_rootfs/var/volatile/log'

# 2.5 Fix git safe.directory for UID mismatch after docker cp
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

**Host safety rule:** 任何 chroot 或 qemu/rootfs 相关命令都必须包在
`docker exec verify-oee ...` 里执行。不要在宿主机直接 chroot 到 aarch64 rootfs，
也不要把宿主机 `/proc`、`/sys`、`/dev` bind 到容器 rootfs；setup/build 不需要这些资源。

## Phase 3 — Inspect chroot Environment

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

容器镜像应包含：git、python3、dnf、ROS 2 Humble、pip 下载缓存和两个 openEuler ROS repo 配置。
setup.sh 仍须完整执行，以验证 ROS 检测、workspace venv 安装和项目依赖配置；无需手动干预。

## Phase 4 — Prepare Workspace

提供两种方式，根据场景选择：

### 方式 A：从宿主机 docker cp（适合本地代码验证）

> 用于验证本地未提交的改动。`docker cp` 拷入当前项目目录，可直接验证修改效果。

```bash
# 4.1 Copy current workspace into rootfs
docker cp <project_root> verify-oee:/root/openeuler_rootfs/root/IB_Robot

# 4.2 Remove stale artifacts (host paths are wrong in container)
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
```

### 方式 B：在容器内 git clone（适合验证远程分支）

> 用于验证已推送到个人仓库分支的代码。注意 rootfs 中 `which` 不可靠，
> git 命令请用绝对路径 `/usr/bin/git`。

```bash
# 4.1 Clone the branch inside chroot
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "
    cd /root && /usr/bin/git clone -b <branch> <repo_url> /root/IB_Robot
  "'

# 4.2 Remove stale artifacts (if any)
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "rm -rf /root/IB_Robot/{venv,build,install,log}"'
```

## Phase 5 — Run setup.sh

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

## Phase 6 — Run build.sh

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

## Phase 7 — Inspect and Clean Up

```bash
# Collect all ERROR lines with context
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "grep ERROR /tmp/setup.log || echo \"(none)\""'
docker exec verify-oee bash -c \
  'chroot /root/openeuler_rootfs /bin/bash -c "grep ERROR /tmp/build.log || echo \"(none)\""'

# Clean up
docker stop verify-oee && docker rm verify-oee
```

> **错误报告要求**：必须逐条列出所有 ERROR 行，并按 Verification Discipline 中的分类标准标注 Fatal / Non-fatal。不能只给 ERROR 行数不给内容。
