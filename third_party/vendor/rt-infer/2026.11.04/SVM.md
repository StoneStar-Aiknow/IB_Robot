# RT-Infer 310P1：SVM 驱动安装补充

日期：2026-11-03。适用：Ascend 310P1 RC、openEuler Embedded、5.10.0-openeuler。

# 1 为什么需要本补充

原 `2026.11.03-priority-eval.2` 包提供 scheduler、runtime 和 aicore_device.ko，但没有包含 SVM 替换步骤。某些板上的原版 svm.ko 对绑定调用返回成功，却不回填当前 runtime 读取的 PASID，导致首个硬件任务无法完成。

本补充提供与既有部署教程一致的 svm.ko。**仅有 CANN 环境和原生推理成功，不足以证明 SVM 接口满足软件调度要求。**

本补充不包含锁频、CPU 隔离或固件刷写操作。

|文件|用途|
|---|---|
|artifacts/svm.ko|配套 SVM 模块|
|svm_bind_probe.py|独立检查 PASID 回填，不下发算子|
|SHA256SUMS|文件校验|

模块 SHA256：`6441b69ea03273fb064571223c83f6fc665971391c9679db2687c77045b037b7`。

# 2 适用条件

以 root 操作，先退出 NPU 计算客户端和软调度器，保存当前业务记录。SVM 被基础驱动引用，应通过重启加载替换版本，不直接 rmmod/insmod。

```bash
uname -r
modinfo artifacts/svm.ko
sha256sum -c SHA256SUMS
fuser -v /dev/davinci0
```

目标内核和模块 vermagic 应为 `5.10.0-openeuler SMP mod_unload aarch64`，架构为 aarch64。内核配置或符号版本不同的设备需要单独适配，不能强制加载。

该模块已在既有验证板以及本次同事的 310P1 板上加载验证。本次同事板原模块哈希为 `b0c7aa72a15bc2857c60643497d5e6376e5e94f458dfb11822a2f8122b68b789`，替换后通过 PEAR 软件 FIFO、PFP 和原生数值检查。

# 3 备份、替换与重启

在本补充包根目录执行。下面会修改系统 SVM 模块，但不替换 CANN runtime。

```bash
set -e
NEW="$PWD/artifacts/svm.ko"
TARGET=/usr/local/Ascend/driver/host/svm.ko
BACKUP="/root/rtinfer-svm-backup-$(date +%Y%m%d-%H%M%S)"

test "$(id -u)" = 0
test "$(uname -r)" = 5.10.0-openeuler
test "$(modinfo -F vermagic "$NEW")" = '5.10.0-openeuler SMP mod_unload aarch64'
test -f "$TARGET"
test ! -e "$BACKUP"
if fuser /dev/davinci0 >/dev/null 2>&1; then echo '先正常退出 NPU 任务'; exit 1; fi
if pgrep -x aicore_schedule >/dev/null; then echo '先正常停止软调度器'; exit 1; fi

mkdir -m 0700 "$BACKUP"
cp -a "$TARGET" "$BACKUP/svm.before.ko"
sha256sum "$NEW" "$TARGET" > "$BACKUP/hashes.before"
cat /proc/sys/kernel/random/boot_id > "$BACKUP/boot.before"
MODE=$(stat -c '%a' "$TARGET")
test ! -e "$TARGET.rtinfer-new"
install -m "$MODE" -o root -g root "$NEW" "$TARGET.rtinfer-new"
cmp "$NEW" "$TARGET.rtinfer-new"
mv "$TARGET.rtinfer-new" "$TARGET"
sync
printf '原模块备份：%s\n' "$BACKUP"
systemctl reboot
```

保留打印的 BACKUP 路径。SSH 会随重启断开，设备启动后重新连接。

# 4 重启后验证

```bash
cat /proc/sys/kernel/random/boot_id
sha256sum /usr/local/Ascend/driver/host/svm.ko
modinfo -F vermagic /usr/local/Ascend/driver/host/svm.ko
grep '^svm ' /proc/modules
ls -l /dev/svm0

# 在补充包目录执行，不通过 priorityctl run 启动。
python3 svm_bind_probe.py --pid self --pasid -1
```

确认启动 ID 已变化、正确模块已加载、ioctl 成功，且 `after.pasid` 不再保持哨兵 -1；配套模块还应回填 TTBR 和 TCR。不要把某个固定 PASID 写死到程序中。

随后按原优先级包 README 检查环境，先做固定输入的原生对照，再做软件 FIFO 单模型及数值检查，通过后继续 NICE/PFP 与多进程实验。

软调度正常停止后仍需按原教程重启，才能恢复原生硬件调度；SVM 替换版本在重启后继续保留。

# 5 回滚

在维护窗口把 BACKUP 改为第 3 节保存的实际路径：

```bash
set -e
BACKUP=/root/rtinfer-svm-backup-实际时间
TARGET=/usr/local/Ascend/driver/host/svm.ko
install -m 0440 -o root -g root "$BACKUP/svm.before.ko" "$TARGET.rollback-new"
cmp "$BACKUP/svm.before.ko" "$TARGET.rollback-new"
mv "$TARGET.rollback-new" "$TARGET"
sync
systemctl reboot
```

无法正常启动时，通过设备串口或恢复环境还原同一备份文件。回滚至不回填 PASID 的模块后，当前软件调度路径将再次不满足前置条件。
