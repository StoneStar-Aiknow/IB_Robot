# ascend-vm 验证流程

本机安装了 ascend-vm 时，openEuler 验证使用本流程，不使用 Docker 流程。

ascend-vm 用 QEMU 运行完整的 openEuler Embedded aarch64 虚拟机，并通过 VFIO 直通 Atlas 300I Duo。和 Docker 流程相比，ascend-vm 有完整的内核、`/dev` 和真实 NPU，更接近真机。

## 与 Docker 流程的差异

| 方面 | Docker 流程 | ascend-vm 流程 |
|------|-------------|----------------|
| 模拟方式 | qemu-user，进入 chroot | qemu-system TCG，完整虚拟机 |
| CPU | 宿主机全部核 | 虚拟机配置的核数（默认 4） |
| NPU | 没有 | 有，`/dev/davinci*` |
| 环境复用 | 每次新建容器 | 复用同一台虚拟机 |
| 干净环境的范围 | 整个 rootfs | 只有代码目录和 venv |
| 速度 | 较快 | 较慢 |

ascend-vm 复用虚拟机，所以系统包、ROS 和 dnf 缓存会保留到下一次验证。第一次验证由 `setup.sh` 安装 ROS，之后的验证复用已安装的 ROS。Docker 流程的 `:env` 镜像也预装了 ROS，两者在这一点上相同。

如果变更切换了 ROS 包源，先在虚拟机中执行 procedure.md Phase 4.5 的预清理命令。命令去掉 `docker exec` 和 `chroot` 包装，直接在虚拟机中执行。

## Phase 0 — 检查宿主机

1. 确认 ascend-vm 已安装，虚拟机正在运行：

   ```bash
   command -v ascend-vm
   ascend-vm status
   ```

   `VM` 一节显示 `RUNNING`，并给出 SSH 端口。

   如果虚拟机没有运行，停止验证，请用户启动虚拟机。不要改用 Docker 流程。

2. 确认可以用密钥登录虚拟机：

   ```bash
   ssh -o BatchMode=yes -p 2222 root@127.0.0.1 true
   ```

   如果登录失败，停止验证，由用户用 `ssh-copy-id -p 2222 root@127.0.0.1` 添加公钥。不要改用 Docker 流程。Agent 不读取、不保存虚拟机 root 密码。

3. 在 `~/.ssh/config` 中为虚拟机配置别名。`LogLevel ERROR` 用于屏蔽登录横幅：

   ```text
   Host ascend-vm
       HostName 127.0.0.1
       Port 2222
       User root
       IdentityFile ~/.ssh/<key>
       IdentitiesOnly yes
       LogLevel ERROR
   ```

下文命令使用别名 `ascend-vm`。

## Phase 1 — 检查虚拟机环境

```bash
ssh ascend-vm 'uname -m; grep -E "^(NAME|BUILD_ID)=" /etc/os-release; nproc; df -BG /; ls /dev/davinci*; ls /opt/ros 2>/dev/null || echo no-ros-yet; ls /root/ibrobot-verify 2>/dev/null'
```

预期结果：

- `uname -m` 输出 `aarch64`。
- `/dev/davinci0` 存在。
- `df -BG /` 的 `Avail` 列不小于 `20G`。

20G 是估算值。2026-11-03 实测，第一次 setup 后剩余空间减少约 8G，主要是 venv 和 pip 缓存。build 的峰值没有实测，20G 在 setup 增量上留出了余量。

如果剩余空间不够，按顺序处理：

1. 删除 `/root/ibrobot-verify/` 下以前验证留下的目录。
2. 如果仍然不够，停止验证，告诉用户实际剩余空间。

## Phase 2 — 准备虚拟机

1. 放宽 dnf 下载超时。TCG 下 TLS 握手慢，dnf 默认的最低速率判定会中断下载：

   ```bash
   ssh ascend-vm 'grep -qE "^minrate" /etc/dnf/dnf.conf || printf "timeout=120\nminrate=100\nretries=10\n" >> /etc/dnf/dnf.conf'
   ```

2. 检查 libxml2，必要时修复。原因见 [pitfalls.md](pitfalls.md) 的 libxml2 一行和 #219：

   ```bash
   ssh ascend-vm 'readlink /usr/lib64/libxml2.so.2; rpm -qf /usr/lib64/libxml2.so.2.12.6 2>&1'
   ```

   如果 `libxml2.so.2` 指向 `libxml2.so.2.12.6`，并且这个文件不属于任何软件包，执行：

   ```bash
   ssh ascend-vm 'dnf reinstall -y --nogpgcheck libxml2; mv /usr/lib64/libxml2.so.2.12.6 /root/libxml2.so.2.12.6.bak; ldconfig; python3 -c "import ctypes; ctypes.CDLL(\"libxslt.so.1\")" && echo libxslt-ok'
   ```

   输出 `libxslt-ok` 才能继续。验证报告中写明执行了这项修复。

## Phase 3 — 准备代码快照

在宿主机上生成目标 commit 的独立快照，复制到虚拟机，并在虚拟机中核对 tree。

警告：下面的脚本必须在同一个 shell 中一次执行完。快照路径和 tree 都保存在 shell 变量里。分成多次执行时，变量会丢失。

前置条件：

- 宿主机安装了 git-lfs。虚拟机中没有 git-lfs，LFS 文件必须在宿主机上拉取。
- 目标 commit 已经推送到 PR 的源仓库。

脚本中的 3 个变量需要替换：

| 变量 | 取值 |
|------|------|
| `PROJECT_ROOT` | 包含目标 commit 的本地 checkout |
| `VERIFIED_COMMIT` | 目标 commit 的完整 SHA |
| `VERIFICATION_REPO` | PR 源分支所在仓库的 URL，通常是个人 fork |

`VERIFICATION_REPO` 按 PR 的源仓库填写，不按 remote 名称判断。有些 checkout 中 `origin` 指向上游仓库。

```bash
set -euo pipefail
PROJECT_ROOT=<local-checkout>
VERIFIED_COMMIT=<full-commit-sha>
VERIFICATION_REPO=<pr-source-repository-url>

command -v git-lfs >/dev/null
VERIFIED_COMMIT=$(git -C "${PROJECT_ROOT}" rev-parse "${VERIFIED_COMMIT}^{commit}")
VERIFIED_TREE=$(git -C "${PROJECT_ROOT}" rev-parse "${VERIFIED_COMMIT}^{tree}")
SNAPSHOT_ROOT=$(mktemp -d "/tmp/ibrobot-verify-${VERIFIED_TREE:0:12}.XXXXXX")
DEST=/root/ibrobot-verify/${VERIFIED_TREE:0:12}

# Snapshot of the target commit, outside the user's worktree.
git clone --quiet --no-hardlinks --no-checkout "${PROJECT_ROOT}" "${SNAPSHOT_ROOT}"
git -C "${SNAPSHOT_ROOT}" remote set-url origin "${VERIFICATION_REPO}"
git -C "${SNAPSHOT_ROOT}" checkout --quiet --detach "${VERIFIED_COMMIT}"
git -C "${SNAPSHOT_ROOT}" submodule --quiet update --init --recursive
git -C "${SNAPSHOT_ROOT}" lfs pull origin
git -C "${SNAPSHOT_ROOT}" lfs fsck
test "$(git -C "${SNAPSHOT_ROOT}" rev-parse HEAD^{tree})" = "${VERIFIED_TREE}"
test -z "$(git -C "${SNAPSHOT_ROOT}" status --porcelain --untracked-files=all)"

# Copy into a fresh directory in the VM. --no-same-owner makes root own the
# files; otherwise setup.sh fails on "detected dubious ownership".
ssh ascend-vm "rm -rf ${DEST} && mkdir -p ${DEST}"
tar -C "${SNAPSHOT_ROOT}" -cf - . | ssh ascend-vm "tar --no-same-owner -C ${DEST} -xf -"
ssh ascend-vm "git config --global --get-all safe.directory | grep -qxF ${DEST} || git config --global --add safe.directory ${DEST}"
test "$(ssh ascend-vm "git -C ${DEST} rev-parse HEAD^{tree}")" = "${VERIFIED_TREE}"

rm -rf -- "${SNAPSHOT_ROOT}"
echo "DEST=${DEST} VERIFIED_COMMIT=${VERIFIED_COMMIT} VERIFIED_TREE=${VERIFIED_TREE}"
```

- 脚本最后一行输出 `DEST`、`VERIFIED_COMMIT` 和 `VERIFIED_TREE`。记录这 3 个值，后续步骤和验证报告都要使用。
- 脚本中任何一步失败，脚本立即退出。宿主机上会留下 `/tmp/ibrobot-verify-<tree>.*`，删除后重新执行。
- 虚拟机中的 `git status` 会把 LFS 文件显示为已修改，原因是虚拟机中没有 git-lfs，不影响 tree 核对。
- 2026-11-03 实测，脚本用时 185 s。

## Phase 4 — 运行 setup.sh 和 build.sh

警告：setup 和 build 之间必须执行 lz4 门禁。setup 的 dnf 事务安装 `lz4-devel` 后，`/usr/lib64/liblz4.so` 指向不存在的 `liblz4.so.1.9.4`。不修复时，build 在 `livox_ros_driver2` 链接阶段报 `No rule to make target '/usr/lib64/liblz4.so'`。2026-11-03 的 ascend-vm 实测中，这个错误让 build 在 42 分钟后失败。

下面的命令在 setup 和 build 之间执行 lz4 门禁。命令在虚拟机中后台运行，退出码写入状态文件，SSH 断开不影响运行：

```bash
ssh ascend-vm "cd ${DEST} && (nohup bash -c '
  date +%s > setup.start
  PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 IBR_LEROBOT_FORCE_REBUILD=1 \
    bash scripts/setup.sh --yes --no-sudo --accel cann > setup.log 2>&1
  echo \$? > setup.status; date +%s > setup.end
  if [ \$(cat setup.status) = 0 ]; then
    if rpm -V lz4 lz4-devel 2>&1 | grep -q \"missing.*/usr/lib64/liblz4\" || [ ! -e /usr/lib64/liblz4.so.1.9.4 ]; then
      dnf reinstall -y --nogpgcheck lz4 lz4-devel > lz4-gate.log 2>&1
    fi
    test -e \$(readlink -f /usr/lib64/liblz4.so) && echo ok > lz4-gate.status || echo broken > lz4-gate.status
    date +%s > build.start
    bash scripts/build.sh > build.log 2>&1
    echo \$? > build.status; date +%s > build.end
  fi' < /dev/null > /dev/null 2>&1 &)"
```

- `--accel cann` 和 Docker 流程一致。虚拟机中有 `/dev/davinci*`，所以自动探测的结果也是 `cann`。
- `PIP_DEFAULT_TIMEOUT` 和 `PIP_RETRIES` 用于应对 pip 镜像的临时故障。
- 虚拟机的登录环境会加载 CANN 的 `PYTHONPATH`，`setup.sh` 会输出 `PYTHONPATH is set` 警告。这个警告不影响结果。
- setup 日志中 lerobot 对 torch、torchvision 或 numpy 版本的 pip 冲突提示，按 Non-fatal 分类，并在验证报告中单独列出。这组冲突是否预期，见 #219。
- setup 的系统依赖阶段会在系统 Python 中执行 `pip3 install colcon-common-extensions`。如果这一步超过 15 分钟没有结束，在虚拟机中执行 `dmesg | grep -i "out of memory"`。2026-11-03 实测，网络中断后这个 pip 进程占用 15.7 GB 内存，被内核 OOM 结束，setup 以 137 退出。已经安装的 colcon 不受影响，用新的代码目录重跑即可。详情见 #219。
- lz4 门禁的结果写入 `lz4-gate.status`。如果内容是 `broken`，build 会在 `livox_ros_driver2` 失败，按 procedure.md Phase 5.5 处理后重跑 build。

查看进度：

```bash
ssh ascend-vm "cd ${DEST}; cat setup.status build.status 2>/dev/null; tail -3 setup.log; tail -3 build.log 2>/dev/null"
```

`setup.status` 和 `build.status` 都是 `0`，并且日志中有 `Setup complete` 和 `Build complete`，验证才算通过。

参考耗时（2026-11-03，ascend-vm 3.1，4 个虚拟核，ROS 第一次安装）：

| 阶段 | 耗时 |
|------|------|
| setup | 2551 s |
| build（从头开始） | 约 2500 s 后在 `livox_ros_driver2` 失败（lz4 未修复） |
| build（修复 lz4 后在同一目录重跑） | 921 s |

如果 build 失败后修复了环境，可以在同一目录只重跑 `build.sh`。colcon 复用已经完成的包。

## Phase 5 — 收集结果并清理

1. 按 SKILL.md 的 Error Classification 收集并分类 `setup.log` 和 `build.log` 中的 ERROR 行。
2. 记录虚拟机状态：

   ```bash
   ssh ascend-vm 'readlink /usr/lib64/libxml2.so.2 /usr/lib64/liblz4.so; grep -E "^(timeout|minrate|retries)=" /etc/dnf/dnf.conf; grep BUILD_ID /etc/os-release'
   ```

3. 按下面的模板写验证报告。
4. 删除虚拟机中的代码目录：

   ```bash
   ssh ascend-vm "rm -rf ${DEST}"
   ```

### 验证报告模板

```text
openEuler 验证：PASS 或 FAIL
- 环境：ascend-vm <ascend-vm version 的输出>，虚拟机 BUILD_ID=<值>，<N> 个虚拟核，--accel cann
- Source：isolated committed snapshot
- Verified commit：<VERIFIED_COMMIT>
- Verified tree：<VERIFIED_TREE>
- setup：<setup.status>，<耗时> s
- build：<build.status>，<耗时> s，<完成的包数> 个包
- ERROR 行：逐条列出，标注 Fatal 或 Non-fatal
- 本次执行的修复：<Phase 2 和 lz4 门禁中本次实际执行的修复；没有就写“无”>
- 检查时的虚拟机状态：libxml2.so.2 -> <目标>，liblz4.so -> <目标>，dnf.conf 中的 timeout/minrate/retries
```

虚拟机是复用的。“本次执行的修复”只写本次运行实际执行的命令。以前运行留下的修复结果写在“检查时的虚拟机状态”中。

PR 的 `**Docker environment:**` 字段中，openEuler 一侧写 `openeuler-ascend-vm`。和 Ubuntu 一起时写 `ubuntu|openeuler-ascend-vm`。
