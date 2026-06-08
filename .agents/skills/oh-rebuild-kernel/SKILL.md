---
name: oh-rebuild-kernel
description: "Rebuild and flash the Linux kernel (boot_linux.img) for the Bearkey BQ3588HM OpenHarmony board. Use when user needs to 'rebuild kernel', 'compile kernel', 'flash boot_linux', 'enable USB_ACM', 'enable USB_SERIAL_CH341', 'add kernel module', '内核编译', '刷入内核', '重新编译内核', 'kernel config', 'defconfig', or modify kernel configuration for the BQ3588HM board."
---

# Rebuild BQ3588HM OpenHarmony Kernel

Rebuild the Linux kernel `boot_linux.img` for the Bearkey BQ3588HM OpenHarmony board to add/enable kernel drivers (e.g., USB ACM, USB serial, joystick).

## When to Use This Skill

- User wants to enable a kernel driver that is not built into the stock kernel
- User says "内核编译", "重新编译内核", "enable USB driver", "add kernel config"
- SO-101 arm's `/dev/ttyACM0` is not appearing (needs `CONFIG_USB_ACM=y`)
- Gamepad or joystick device not recognized
- Any kernel-level driver support needed on the board

## Prerequisites

- BQ3588HM board with OpenHarmony EmbodiedAI 1.0.1
- OpenHarmony source code downloaded to `/data/oh_build` on the board (or accessible build environment)
- SSH or HDC shell access to the board
- `sudo` password: `admin`

## Verified Kernel Config Additions

The following configs were verified needed for IB_Robot peripherals:

### USB Serial (SO-101 arm)

```ini
CONFIG_USB_ACM=y           # SO-101 CH9102 chip reports as CDC ACM device
CONFIG_USB_SERIAL_CH341=y  # CH341 serial driver (backup)
```

> **Critical**: SO-101 uses CH9102 chip which reports as CDC ACM interface, NOT CH340/CH341.
> `CONFIG_USB_ACM=y` is the required driver, not just `USB_SERIAL_CH341`.

### Joystick / Gamepad (teleop mode)

```ini
CONFIG_INPUT_JOYDEV=y
CONFIG_INPUT_JOYSTICK=y
CONFIG_JOYSTICK_XPAD=y
CONFIG_HID_MICROSOFT=y
CONFIG_HID_SONY=y
CONFIG_HID_STEAM=y
CONFIG_HID_LOGITECH=y
CONFIG_HID_STEELSERIES=y
CONFIG_HID_WIIMOTE=y
```

## Step-by-Step Workflow

### 1. Locate defconfig

```bash
# On board (SSH/HDC)
cd /data/oh_build
# defconfig path for RK3588:
find . -name "arch/arm64_defconfig" -path "*/config/linux-6.6/rk3588/*"
# Expected: kernel/linux/config/linux-6.6/rk3588/arch/arm64_defconfig
```

### 2. Edit defconfig

Append the needed configs to the defconfig file:

```bash
# Example: enable USB_ACM
vi kernel/linux/config/linux-6.6/rk3588/arch/arm64_defconfig
# Append: CONFIG_USB_ACM=y
```

Or for a full IB_Robot setup, add all configs listed above.

### 3. Build boot_linux.img

```bash
cd /data/oh_build

# Full kernel build (may take 30-60 minutes on the board)
./build.sh -p bq3588 --ccache

# Output:
# out/bq3588/packages/phone/images/boot_linux.img
```

### 4. Backup and Flash

```bash
# BACKUP current boot_linux (important!)
dd if=/dev/block/by-name/boot_linux of=/data/boot_linux_backup.img

# Flash new kernel
dd if=/data/oh_build/out/bq3588/packages/phone/images/boot_linux.img of=/dev/block/by-name/boot_linux

# Reboot
reboot
```

### 5. Verify

```bash
# After reboot, check for USB ACM device
ls -la /dev/ttyACM0
# Expected: crw-rw---- 1 root radio 166, 0 ... /dev/ttyACM0

dmesg | grep cdc_acm
# Expected: cdc_acm 5-1.2:1.0: ttyACM0: USB ACM device

# Check joystick
ls /dev/input/js*
# Expected: /dev/input/js0 (when gamepad connected)

# Verify kernel config is active
zcat /proc/config.gz 2>/dev/null | grep USB_ACM
# Or: grep USB_ACM /boot/config-$(uname -r) 2>/dev/null
# Expected: CONFIG_USB_ACM=y
```

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `build.sh` fails with missing dependencies | Build environment incomplete | Follow OH source build guide |
| `dd` permission denied | Not root | Use `su` or `sudo` |
| Device still not appearing after flash | Wrong defconfig or wrong partition | Verify partition: `ls -la /dev/block/by-name/boot_linux` |
| Boot fails after flash | Bad kernel image | Restore backup: `dd if=/data/boot_linux_backup.img of=/dev/block/by-name/boot_linux` |
| `CONFIG_USB_ACM=y` but no `/dev/ttyACM0` | CH9102 not in ACM mode | Check with `lsusb`; some CH9102 variants need `usbserial` driver instead |

## Key Facts

- Board: Bearkey BQ3588HM (RK3588)
- Kernel version: Linux 6.6
- boot_linux partition: `/dev/block/by-name/boot_linux`
- OH source version: OpenHarmony EmbodiedAI 1.0.1
- Build command: `./build.sh -p bq3588 --ccache`
- Build output: `out/bq3588/packages/phone/images/boot_linux.img`
- `sudo` password: `admin`

## Related Skills

- `ibrobot-hdc`: File transfer to/from board
- `ibrobot-bq3588hm-oh`: Board runtime facts
- `bq3588-oh-rknn`: Running RKNN inference on board
