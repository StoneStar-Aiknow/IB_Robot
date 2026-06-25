# RoboFrame 高级部署指南

> 本文档涵盖 RoboFrame 在 OpenHarmony 上的高级配置：第三方 ROS 包交叉编译、内核驱动编译、RKNN 详细配置、全链路闭环、以及旧版 skh-run 环境叠加（已废弃）。

## 目录

- [第三方 ROS 2 包交叉编译](#第三方-ros-2-包交叉编译)
- [内核驱动编译（USB ACM / 手柄）](#内核驱动编译usb-acm--手柄)
- [RKNN NPU 推理详细配置](#rknn-npu-推理详细配置)
- [全链路闭环（真实硬件）](#全链路闭环真实硬件)
- [旧版 skh-run 环境叠加（已废弃）](#旧版-skh-run-环境叠加已废弃)

---

## 第三方 ROS 2 包交叉编译

板端预装运行时不包含所有包。纯 C/C++ 第三方 ROS 2 包（如 `usb_cam`）必须用 OHOS 交叉工具链编译。

### 以 usb_cam 为例

```bash
OH_CUSTOM_SRC="$OH_ROOT/custom_build_root/ibrobot_oh_ws/src"
git clone --depth 1 -b main https://github.com/ros-drivers/usb_cam.git "$OH_CUSTOM_SRC/usb_cam"

# 在已有的构建环境中编译
./scripts/openharmony/build_ibrobot_oh_custom.sh \
  --oh-root "$OH_ROOT" \
  --packages usb_cam
```

### 验证产物

```bash
file $OH_ROOT/custom_build_root/ibrobot_oh_ws/install/usb_cam/lib/usb_cam/usb_cam_node_exe
# 期望: ELF 64-bit LSB ... ARM aarch64 ... ld-musl-aarch64.so.1
```

### 部署到板端

```bash
scp -r $OH_ROOT/custom_build_root/ibrobot_oh_ws/install/usb_cam root@<board-ip>:/data/roboframe/install/
```

### ffmpeg 依赖

`usb_cam` 的 `mjpeg2rgb` 转换依赖 ffmpeg。若报 `libswscale.so.5` 找不到：

```bash
ssh root@<board-ip> 'mkdir -p /data/roboframe/install/usb_cam/lib && \
  ln -sf /data/out/lib/libswscale.so.5 /data/roboframe/install/usb_cam/lib/ && \
  ln -sf /data/out/lib/libavcodec.so.58 /data/roboframe/install/usb_cam/lib/ && \
  ln -sf /data/out/lib/libavutil.so.56 /data/roboframe/install/usb_cam/lib/'
```

### 已验证的第三方包

| 包 | 来源 | 版本 | 状态 |
| --- | --- | --- | --- |
| `usb_cam` | ros-drivers/usb_cam | main (0.8.x) | 已验证 640×480 MJPEG 30 FPS |

---

## 内核驱动编译（USB ACM / 手柄）

SO-101 机械臂使用 CH9102 芯片以 CDC ACM 模式报告，默认内核缺少 `CONFIG_USB_ACM`；手柄遥操作也需要 HID 驱动。

### 修改内核 defconfig

编辑 `kernel/linux/config/linux-6.6/rk3588/arch/arm64_defconfig`：

```diff
+CONFIG_USB_ACM=y
+CONFIG_USB_SERIAL_CH341=y
+CONFIG_INPUT_JOYDEV=y
+CONFIG_INPUT_JOYSTICK=y
+CONFIG_JOYSTICK_XPAD=y
+CONFIG_HID_MICROSOFT=y
+CONFIG_HID_SONY=y
+CONFIG_HID_STEAM=y
+CONFIG_HID_LOGITECH=y
```

### 编译并刷入

```bash
./build.sh -p bq3588 --ccache
# 产物: out/bq3588/packages/phone/images/boot_linux.img

dd if=/dev/block/by-name/boot_linux of=/data/boot_linux_backup.img
dd if=out/bq3588/packages/phone/images/boot_linux.img of=/dev/block/by-name/boot_linux
reboot
```

### 验证

```bash
ls -la /dev/ttyACM0
# crw-rw---- 1 root radio 166, 0 ... /dev/ttyACM0
dmesg | grep cdc_acm
# cdc_acm 5-1.2:1.0: ttyACM0: USB ACM device
```

---

## RKNN NPU 推理详细配置

### ONNX → RKNN 转换（主机）

```bash
python3 -m venv .venv-rknn
source .venv-rknn/bin/activate
pip install rknn-toolkit2==2.3.2

python tools/export_onnx_rknn.py \
    --onnx models/act_ros2_rknn.onnx \
    --output models/act_ros2_rknn.rknn \
    --dtype float16
```

### librknnrt.so 软链接

rknnlite 的 C 扩展硬编码搜索 `/usr/lib/librknnrt.so`：

```bash
ssh root@<board-ip> 'mount -o remount,rw / && mkdir -p /usr/lib && ln -sf /vendor/lib64/librknnrt.so /usr/lib/librknnrt.so'
```

> 根文件系统重启后恢复只读，需在启动脚本（如 `/vendor/etc/init/init.npu.cfg`）中加入自动创建命令。

### 独立验证 NPU 推理

```bash
ssh root@<board-ip> '
source /data/roboframe/scripts/robooh_1.0.1.env
python3 -c "
import numpy as np, time
from rknnlite.api import RKNNLite
rknn = RKNNLite()
rknn.load_rknn(\"/data/local/tmp/act_ros2_rknn.rknn\")
rknn.init_runtime(target=None)
state = np.random.randn(1, 14).astype(np.float32)
cam_high = np.random.randn(1, 3, 480, 640).astype(np.float32)
cam_left = np.random.randn(1, 3, 480, 640).astype(np.float32)
t0 = time.time()
outputs = rknn.inference(inputs=[state, cam_high, cam_left])
print(f\"shape={outputs[0].shape}, time={time.time()-t0:.3f}s\")
rknn.release()
"'
```

期望输出 `shape=(1, 100, 6)`，延迟约 121ms。

### RKNN 模型 YAML 配置

```yaml
models:
  so101_act_rknn:
    path: /data/roboframe/models/502000/pretrained_model   # 必须绝对路径
    policy_type: act
    device: rknn
    lerobot_norm_mode: range_m100_100

control_modes:
  model_inference:
    inference:
      enabled: true
      model: so101_act_rknn
```

### 版本兼容性

| 组件 | 版本 |
| --- | --- |
| rknn-toolkit2（主机，转换） | 2.3.2 |
| rknn-toolkit-lite2（板端，推理） | 2.3.0 |
| librknnrt.so（板端 NPU 驱动） | 2.4.1b0 |

---

## 全链路闭环（真实硬件）

### 闭环架构

```text
Board (RK3588, OpenHarmony)
├── usb_cam_node_exe × 2          (top + wrist 相机, MJPEG 640×480)
├── static_transform_publisher × 4 (TF: base→camera, gripper→camera, optical)
├── lerobot_policy_node × 1        (RKNN NPU 推理, ACT 策略)
├── action_dispatcher_node × 1     (动作分发, 20Hz)
└── so101_hardware                 (ros2_control, /dev/ttyACM0)
     ↳ arm_position_controller / gripper_position_controller
```

### 机械臂校准

```bash
ssh root@<board-ip>
source /data/roboframe/scripts/robooh_1.0.1.env
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 run so101_hardware calibrate_arm --arm follower --port /dev/ttyACM0
```

校准 JSON 保存在 `~/.calibrate/so101_follower_calibrate.json`。launch 环境下 `HOME=/data/local/tmp/ros_home`，需符号链接：

```bash
mkdir -p /data/local/tmp/ros_home/.calibrate
ln -sf /root/.calibrate/so101_follower_calibrate.json /data/local/tmp/ros_home/.calibrate/
```

### 启动全链路

```bash
ssh root@<board-ip>
source /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=false \
    control_mode:=model_inference \
    device:=rknn \
    2>&1 | tee /data/launch.log
```

### 预期日志

```text
[top_camera]: Starting 'top' at 640x480 via mmap (mjpeg2rgb) at 30 FPS
[wrist_camera]: Starting 'wrist' at 640x480 via mmap (mjpeg2rgb) at 60 FPS
[act_inference_node]: Using inference_backend=rknn, tensor_device=cpu
[act_inference_node]: ✓ First inference complete (monolithic): total=~500ms
[action_dispatcher]: ✓ First inference received: chunk=100
```

### 闭环性能

| 指标 | 数值 |
| --- | --- |
| NPU 推理延迟（稳定） | ~500 ms |
| 总端到端延迟 | ~520 ms |
| 控制频率 | 20 Hz |
| 相机帧率 | top: 30 FPS, wrist: 60 FPS |

---

## 旧版 skh-run 环境叠加（已废弃）

> **以下内容仅作为历史参考。** 当前推荐使用 `robooh_1.0.1.env` + `pack_roboframe_release.sh`，不再需要手动叠加 skh-run。

### 旧版架构（已废弃）

板端同时运行 ROS 2 节点和 PyTorch 推理需要叠加三套运行时：

| 生态 | 路径 | 说明 |
| --- | --- | --- |
| ROS 2 Humble | `/data/install` + `/sys_prod/robot/out` | OH 预编译 ROS 2 |
| RoboFrame | `/data/roboframe/install` | 交叉编译产物 |
| Torch 运行时 | `/data/local/skh-run/usr` | thirdparty_pytorch 的 skh-run |

**核心问题**：skh-run 自带 Python 3.12，与系统 Python 存在 ABI 冲突，必须用 `LD_PRELOAD` + `PYTHONPATH` + `LD_LIBRARY_PATH` 完整叠加。

### 旧版部署 skh-run

```bash
git clone https://gitcode.com/openharmony-robot/thirdparty_pytorch /tmp/thirdparty_pytorch
cd /tmp/thirdparty_pytorch && git lfs pull --include='test/skh-run.tar.gz'

hdc -t <board-ip>:8710 file send test/skh-run.tar.gz /data/local/skh-run.tar.gz
hdc -t <board-ip>:8710 shell 'cd /data/local && tar -zxpf skh-run.tar.gz'
```

### 旧版完整环境变量（顺序敏感）

```bash
# ① Python 运行时
export PYTHONHOME=/data/local/skh-run/usr
export LD_PRELOAD=${PYTHONHOME}/lib/libpython3.12.so.1.0:${PYTHONHOME}/lib/libomp.so
export LD_LIBRARY_PATH=${PYTHONHOME}/lib:/sys_prod/robot/out/lib:/data/install/lib:/vendor/lib64
export PYTHONPATH=${PYTHONHOME}/lib/python3.12/site-packages:/sys_prod/robot/out/lib/python3.12/site-packages

# ② ROS + RoboFrame
cd /data && . ./ros2ohos.env && . /data/roboframe/install/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ③ RoboFrame 特有路径
export PATH=${PYTHONHOME}/bin:$PATH
export PYTHONPATH=/data/roboframe/install/lerobot/src:/data/roboframe/install/inference_service/lib/python3.12/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=/data/roboframe/install/inference_service/lib:${PYTHONHOME}/lib/python3.12/site-packages/torch/lib:${LD_LIBRARY_PATH}
```

> **顺序关键**：`PYTHONHOME` + `LD_PRELOAD` 必须在 `source setup.sh` **之前**设置。

### 旧版手动安装 rknnlite

```bash
# 安装到 skh-run site-packages
ssh root@<board-ip> '
SITE=/data/local/skh-run/lib/python3.12/site-packages
cd $SITE && unzip -o /data/local/tmp/rknnlite.whl
# 重命名 .so 后缀
for d in api api/npu_config utils; do
  for f in $SITE/rknnlite/$d/*.cpython-312-aarch64-linux-gnu.so; do
    [ -f "$f" ] && mv "$f" "${f%-gnu.so}-ohos.so"
  done
done
'
```

### 旧版手动 wrapper 脚本

```bash
ssh root@<board-ip> '
NODE=/data/roboframe/install/inference_service/lib/inference_service/lerobot_policy_node
cp "$NODE" "${NODE}.real"
cat > "$NODE" << "WRAPPER"
#!/bin/sh
SKH_ROOT=/data/local/skh-run
export PYTHONHOME="${SKH_ROOT}"
export LD_PRELOAD="${SKH_ROOT}/lib/libpython3.12.so.1.0:${SKH_ROOT}/lib/libomp.so"
export LD_LIBRARY_PATH="${SKH_ROOT}/lib:/sys_prod/robot/out/lib:/data/roboframe/install/inference_service/lib:${LD_LIBRARY_PATH}"
export PYTHONPATH="/data/roboframe/install/lerobot/src:${SKH_ROOT}/lib/python3.12/site-packages:${PYTHONPATH}"
export ROS_DISTRO=humble ROS_VERSION=2 ROS_PYTHON_VERSION=3
exec "${SKH_ROOT}/bin/python3" "${NODE}.real" "$@"
WRAPPER
chmod +x "$NODE"
'
```

### 旧版附加依赖安装

RKNN 推理还需手动安装以下包到 skh-run site-packages：

| 包 | 说明 |
| --- | --- |
| rknnlite | RKNN 推理 API |
| ruamel.yaml | rknnlite 运行时依赖 |
| huggingface_hub | lerobot 配置加载依赖 |
| tqdm / filelock / fsspec / packaging | huggingface_hub 传递依赖 |
| psutil stub | 纯 Python stub |

```bash
# ruamel.yaml
ssh root@<board-ip> 'cd /data/local/skh-run/lib/python3.12/site-packages && tar xzf /data/local/tmp/ruamel.tar.gz'

# huggingface_hub + 依赖
ssh root@<board-ip> 'cd /data/local/skh-run/lib/python3.12/site-packages && tar xzf /data/local/tmp/hf-deps.tar.gz && tar xzf /data/local/tmp/hf-extra.tar.gz'

# psutil stub
ssh root@<board-ip> '
SITE=/data/local/skh-run/lib/python3.12/site-packages
mkdir -p $SITE/psutil
cat > $SITE/psutil/__init__.py << "PYEOF"
def cpu_percent(interval=None): return 0.0
def virtual_memory():
    class _M:
        total = 8037424 * 1024
        used = 1000000 * 1024
        available = 7000000 * 1024
        percent = 12.5
    return _M()
PYEOF
'
```
