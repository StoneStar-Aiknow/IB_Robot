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

2. 确认可以用密钥登录虚拟机：

   ```bash
   ssh -o BatchMode=yes -p 2222 root@127.0.0.1 true
   ```

   如果登录失败，由用户用 `ssh-copy-id -p 2222 root@127.0.0.1` 添加公钥。Agent 不读取、不保存虚拟机 root 密码。

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
ssh ascend-vm 'uname -m; grep -E "^(NAME|BUILD_ID)=" /etc/os-release; nproc; df -h /; ls /dev/davinci*; ls /opt/ros 2>/dev/null || echo no-ros-yet'
```

预期结果：

- `uname -m` 输出 `aarch64`。
- `/dev/davinci0` 存在。
- 根分区剩余空间不小于 30 GB。

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

1. 在宿主机上按 procedure.md Phase 4 的“PR 证据”一节生成独立快照，执行到 `git status --porcelain` 检查为止。虚拟机中没有 git-lfs，LFS 文件必须在宿主机上拉取。

2. 把快照复制到虚拟机中的新目录。目录名使用 tree SHA 的前 12 位：

   ```bash
   DEST=/root/ibrobot-verify/${VERIFIED_TREE:0:12}
   ssh ascend-vm "rm -rf ${DEST} && mkdir -p ${DEST}"
   tar -C "${SNAPSHOT_ROOT}" -cf - . | ssh ascend-vm "tar --no-same-owner -C ${DEST} -xf -"
   ```

   警告：`tar` 必须使用 `--no-same-owner`。否则文件属主是宿主机用户，`setup.sh` 在同步子模块时报 `detected dubious ownership`。

3. 在虚拟机中核对 tree：

   ```bash
   ssh ascend-vm "git config --global --add safe.directory ${DEST} && git -C ${DEST} rev-parse HEAD HEAD^{tree}"
   ```

   输出的 tree 必须等于 `VERIFIED_TREE`。虚拟机中的 `git status` 会把 LFS 文件显示为已修改，原因是虚拟机中没有 git-lfs，不影响 tree 核对。

## Phase 4 — 运行 setup.sh 和 build.sh

在虚拟机中后台运行，退出码写入状态文件。SSH 断开不影响运行：

```bash
ssh ascend-vm "cd ${DEST} && (nohup bash -c '
  date +%s > setup.start
  PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=10 IBR_LEROBOT_FORCE_REBUILD=1 \
    bash scripts/setup.sh --yes --no-sudo --accel cann > setup.log 2>&1
  echo \$? > setup.status; date +%s > setup.end
  if [ \$(cat setup.status) = 0 ]; then
    date +%s > build.start
    bash scripts/build.sh > build.log 2>&1
    echo \$? > build.status; date +%s > build.end
  fi' < /dev/null > /dev/null 2>&1 &)"
```

- `--accel cann` 和 Docker 流程一致。虚拟机中有 `/dev/davinci*`，所以自动探测的结果也是 `cann`。
- `PIP_DEFAULT_TIMEOUT` 和 `PIP_RETRIES` 用于应对 pip 镜像的临时故障。
- 虚拟机的登录环境会加载 CANN 的 `PYTHONPATH`，`setup.sh` 会输出 `PYTHONPATH is set` 警告。这个警告不影响结果。

查看进度：

```bash
ssh ascend-vm "cd ${DEST}; cat setup.status build.status 2>/dev/null; tail -3 setup.log; tail -3 build.log 2>/dev/null"
```

`setup.status` 和 `build.status` 都是 `0`，并且日志中有 `Setup complete` 和 `Build complete`，验证才算通过。

## Phase 5 — 收集结果并清理

1. 按 SKILL.md 的 Error Classification 收集并分类 `setup.log` 和 `build.log` 中的 ERROR 行。
2. 记录以下信息：
   - 虚拟机 `BUILD_ID` 和 ascend-vm 版本（`ascend-vm version`）
   - setup 和 build 的耗时
   - Phase 2 中执行的修复
3. 删除虚拟机中的代码目录：

   ```bash
   ssh ascend-vm "rm -rf ${DEST}"
   ```

PR 的 `**Docker environment:**` 字段写明 openEuler 一侧使用 ascend-vm，例如 `ubuntu|openeuler-ascend-vm`。
