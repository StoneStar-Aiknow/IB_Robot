# Houmo LQ50-24G 驱动在 Orange Pi AI Pro (openEuler Embedded) 上的安装指南

## 背景

Orange Pi AI Pro（20T）运行 **openEuler Embedded (oEE)**，内核为 `5.10.0-openeuler`（Yocto/bitbake 自定义构建）。
后摩官方驱动文档面向标准 openEuler/Kylin V10 服务器版，且假设 DKMS 可用，
无法直接在 oEE 上按文档流程执行。本文档记录实际安装步骤，供后续复现。

## 环境信息

| 项目 | 值 |
|------|-----|
| 主板 | Orange Pi AI Pro（20T） |
| 系统 | openEuler Embedded (oEE), aarch64 |
| 内核 | `5.10.0-openeuler`（自编译） |
| 编译器 | GCC 12.3.1 |
| PCI 设备 | `0000:01:00.0 Class 0580: Device 1f6b:0c00` |
| 型号 | LQ50-24GB |
| 驱动版本 | V1.3.0 |

### 涉及的主机

| 主机 | 角色 |
|------|------|
| 开发机（本机） | 协调操作，中转文件 |
| Yocto 构建机 | 提供内核源码构建产物 |
| Orange Pi AI Pro（SSH） | 目标安装设备 |

## 前置条件

从后摩开发者社区下载安装包：

- 资源下载：[developer.houmoai.com/resources_v2](https://developer.houmoai.com/resources_v2)
- 安装说明：[openEuler 驱动安装](https://developer.houmoai.com/hmdoc/m50/software-manual/latest/system-installation-device-management/environment-deployment/system_software_installation_guide/linux/openeuler.html)

所需文件：

- `houmo-drv-xh2-1.3.0-1.aarch64.rpm` — 驱动 RPM（aarch64）
- `M50_M2_fw-xh2_v1.3.0.tar.gz` — 固件
- `houmo-examples-xh2_v1.3.0.zip` — SDK 示例

## 安装流程

### 第一步：从 Yocto 构建机获取内核构建树

oEE 板上不包含内核构建目录（`/lib/modules/<kernel>/build/` 不存在），
仓库中也无匹配的 `kernel-devel` 包。需从 Yocto 构建机提取。

**在构建机上执行：**

```bash
# 路径定义（根据实际 Yocto 构建路径调整）
KSDIR="/home/ubuntu/demo/build/3591rc-ibrobot/tmp/work-shared/3591rc/kernel-source"
BDIR="/home/ubuntu/demo/build/3591rc-ibrobot/tmp/work/3591rc-openeuler-linux/linux-openeuler/5.10-tag3591-r0/build"
OUT="/tmp/kernel-devsrc-full"

rm -rf "$OUT"
mkdir -p "$OUT"

# 1. rsync 内核源码（排除编译产物）
rsync -a \
  --exclude='*.o' --exclude='*.ko' --exclude='*.cmd' \
  --exclude='*.dwo' --exclude='*.lst' --exclude='*.gcno' \
  "${KSDIR}/" "${OUT}/"

# 2. 覆盖构建产物
cp "${BDIR}/.config" "${OUT}/.config"
cp "${BDIR}/Module.symvers" "${OUT}/Module.symvers"

# 3. 复制生成的头文件
rsync -a "${BDIR}/include/generated/" "${OUT}/include/generated/"
rsync -a "${BDIR}/include/config/" "${OUT}/include/config/"
rsync -a "${BDIR}/arch/arm64/include/generated/" "${OUT}/arch/arm64/include/generated/"

# 4. 删除不需要的大文件
rm -rf "${OUT}/arch/arm64/boot"

# 5. 打包
cd "$OUT"
tar czf /tmp/kernel-devsrc.tar.gz .
```

预期产物大小约 400MB。

### 第二步：传输到开发板

```bash
# 构建机 → 本机 → 开发板
scp ubuntu@<构建机IP>:/tmp/kernel-devsrc.tar.gz /tmp/
scp /tmp/kernel-devsrc.tar.gz 开发板:/tmp/
```

同时传输驱动 RPM：

```bash
scp houmo-drv-xh2-1.3.0-1.aarch64.rpm 开发板:/tmp/
```

### 第三步：在开发板上部署内核构建树

```bash
# 解压到 /usr/src/
sudo mkdir -p /usr/src/linux-5.10.0-openeuler
sudo tar xzf /tmp/kernel-devsrc.tar.gz -C /usr/src/linux-5.10.0-openeuler/

# 创建符号链接（kbuild 标准约定）
sudo ln -sfn /usr/src/linux-5.10.0-openeuler /lib/modules/5.10.0-openeuler/build
```

### 第四步：修复 Makefile 并原生编译 scripts

tar 包中的 Makefile 是构建环境的 wrapper（硬编码了 Yocto 路径），需要替换为真正的内核 Makefile：

```bash
# 替换顶层 Makefile（从 kernel-source 获取真实版本）
# 可从构建机重新拷贝：
# 可从构建机重新拷贝：
ssh ubuntu@<构建机IP> "cat ${KSDIR}/Makefile" > /tmp/real_makefile
scp /tmp/real_makefile 开发板:/tmp/real_makefile
sudo cp /tmp/real_makefile /usr/src/linux-5.10.0-openeuler/Makefile
```

安装编译工具并重编译 kernel scripts：

```bash
# OEE 上移除冲突包再安装 flex/bison
sudo dnf remove -y flex-help
sudo dnf install -y flex bison

# 编译 scripts（生成 modpost、fixdep 等 aarch64 原生二进制）
cd /usr/src/linux-5.10.0-openeuler
sudo make -j$(nproc) modules_prepare 2>&1 | tail -20
```

验证编译产物为 aarch64 二进制：

```bash
file scripts/mod/modpost scripts/basic/fixdep
# 应输出: ELF 64-bit LSB ... ARM aarch64 ...
```

### 第五步：安装 Houmo 驱动 RPM

RPM 依赖 `dkms`，OEE 仓库无此包。使用 `--nodeps` 跳过：

```bash
sudo rpm -i --nodeps --noscripts /tmp/houmo-drv-xh2-1.3.0-1.aarch64.rpm
```

RPM 会将文件部署到 `/usr/local/houmo-drv-xh2-1.3.0/`，
其中 `driver/` 目录包含内核模块源码。

### 第六步：编译内核模块

```bash
cd /usr/local/houmo-drv-xh2-1.3.0/driver
sudo make -j$(nproc) KDIR=/lib/modules/$(uname -r)/build
```

编译成功后产出 `xh2a_drv.ko`。

### 第七步：安装模块并加载

```bash
# 安装到内核模块目录
KVER=$(uname -r)
sudo mkdir -p /lib/modules/${KVER}/extra
sudo cp xh2a_drv.ko /lib/modules/${KVER}/extra/
sudo depmod -a

# 立即加载
sudo modprobe xh2a_drv

# 验证
lsmod | grep xh2a
```

### 第八步：配置 udev 和 symlinks

```bash
# udev 规则
sudo bash -c 'cat > /etc/udev/rules.d/99-xh2a.rules <<EOF
KERNEL=="xh2a_*", MODE="0666"
KERNEL=="xh2a-*", MODE="0666"
EOF'
sudo udevadm control --reload-rules && sudo udevadm trigger

# 工具链接
sudo ln -sf /usr/local/houmo-drv-xh2-1.3.0 /usr/local/houmo-sdk
```

### 第九步：开机自加载

```bash
# modprobe 方式
echo 'xh2a_drv' | sudo tee /etc/modules-load.d/houmo-xh2a.conf

# systemd 方式
sudo bash -c 'cat > /etc/systemd/system/houmo-xh2a.service <<EOF
[Unit]
Description=Load Houmo xh2a driver module
After=local-fs.target

[Service]
Type=oneshot
RemainAfterExit=true
ExecStart=/sbin/modprobe xh2a_drv

[Install]
WantedBy=multi-user.target
EOF'

sudo systemctl daemon-reload
sudo systemctl enable houmo-xh2a.service
```

### 第十步：验证

```bash
/usr/local/houmo-drv-xh2-1.3.0/tools/hm_smi/hm_smi -a
```

预期输出示例：

```
  Driver_Version         : V1.3.0
  Vendor                 : Houmo
  BDF                    : 0000:01:00.0
  Model                  : LQ50-24GB
  Firmware_Version       : V1.x.x
  DDR_Memory_Infos       :
    DDR_Memory_Total     : 24448.0MB
  IPU_Infos              :
    Core_Num             : 2
    Core_Freq            : 1400.0 Mhz
```

## 故障排除

### RPM 安装时 dkms 依赖报错
使用 `rpm -i --nodeps --noscripts` 绕过。OEE 无 DKMS 包，手动编译即可。

### `make scripts` 报 `flex: not found`
```bash
sudo dnf remove -y flex-help    # 与 flex 包冲突
sudo dnf install -y flex bison
```

### `make` 报 "No rule to make target `<path>/Makefile'"
顶层 Makefile 是 Yocto 构建环境的 wrapper，需替换为 kernel-source 的真实 Makefile。

### PCIe x1 而非文档中的 x4
LQ50 通过 M.2 接口连接，物理上受限于 M.2 插槽的 lane 数。
检查插槽规格和 BIOS/UEFI 中 PCIe lane 配置。

## 文件路径汇总

| 板上路径 | 内容 |
|---------|------|
| `/usr/src/linux-5.10.0-openeuler/` | 内核构建树 |
| `/lib/modules/5.10.0-openeuler/build` | 符号链接 ↑ |
| `/usr/local/houmo-drv-xh2-1.3.0/` | Houmo SDK 主目录 |
| `/usr/local/houmo-sdk` | 符号链接 ↑ |
| `/lib/modules/5.10.0-openeuler/extra/xh2a_drv.ko` | 驱动模块 |
| `/etc/udev/rules.d/99-xh2a.rules` | udev 权限规则 |
| `/etc/modules-load.d/houmo-xh2a.conf` | 开机加载配置 |
| `/etc/systemd/system/houmo-xh2a.service` | systemd 自启动服务 |
