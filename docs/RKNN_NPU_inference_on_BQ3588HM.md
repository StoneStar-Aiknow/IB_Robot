# BQ3588HM 开发板 RKNN NPU 推理指南

本文档介绍完整的端到端流程：将训练好的 ACT 策略模型转换为 RKNN 格式，并在 Bearkey BQ3588HM (RK3588) OpenHarmony 开发板上通过 NPU 运行推理。

## 前置条件

- 主机：Ubuntu 22.04，Python 3.10+
- 开发板：Bearkey BQ3588HM，OpenHarmony 系统，通过 HDC over TCP 访问
- HDC 工具路径：`/home/xqw/Research/oh_sdk/toolchains/hdc`
- 开发板网络地址：`192.168.136.111:8710`（根据实际网络调整）
- 已训练好的模型 checkpoint（如 `models/502000/`）

## 1. 在主机上将 ONNX 转换为 RKNN

### 1.1 创建专用 RKNN 虚拟环境

rknn-toolkit2 要求 `torch<=2.4.0` + `numpy<=1.26.4`，与 lerobot 的 `torch>=2.7` + `numpy>=2.0` 冲突，因此需要单独的虚拟环境：

```bash
python3 -m venv .venv-rknn
source .venv-rknn/bin/activate
pip install rknn-toolkit2==2.3.2
```

### 1.2 导出 ONNX 并转换为 RKNN

使用项目自带的导出脚本：

```bash
# 在项目根目录下
source .venv-rknn/bin/activate
python src/model_utils/model_utils/export_onnx_rknn.py \
    --onnx models/502000/act_ros2_rknn.onnx \
    --output models/502000/act_ros2_rknn.rknn \
    --dtype float16
```

也可以直接从 policy checkpoint 转换：

```bash
python src/model_utils/model_utils/export_onnx_rknn.py \
    --policy_path models/502000 \
    --output models/502000/act_ros2_rknn.rknn \
    --dtype float16
```

脚本会自动处理 onnx>=1.16 中 `onnx.mapping` 的兼容性问题。

**转换结果：** `act_ros2_rknn.rknn`（约 114 MB，float16），适用于包含两个 480x640 相机输入和 14 维状态向量的 ACT 模型。

## 2. 开发板环境配置

### 2.1 板端 Python 环境信息

| 项目 | 值 |
|------|------|
| Python 版本 | 3.12（musl libc） |
| 扩展模块后缀 | `.cpython-312-aarch64-linux-ohos.so` |
| site-packages 路径 | `/sys_prod/robot/out/lib/python3.12/site-packages/` |
| libpython 路径 | `/sys_prod/robot/out/lib/libpython3.12.so.1.0` |
| RKNN 运行时 | `/vendor/lib64/librknnrt.so`（v2.4.1b0） |
| rknnlite | 已预装在 site-packages 中 |

### 2.2 修复 rknnlite .so 文件后缀不匹配

rknnlite 预装的 `.so` 文件使用 `linux-gnu` 后缀，但板端 Python 期望 `linux-ohos` 后缀。需要逐个目录重命名：

```bash
HDC_BIN=/path/to/hdc
HDC_TARGET=<开发板IP>:8710

# api/ 目录
"$HDC_BIN" -t "$HDC_TARGET" shell 'for f in /sys_prod/robot/out/lib/python3.12/site-packages/rknnlite/api/*.cpython-312-aarch64-linux-gnu.so; do new="${f%-gnu.so}-ohos.so"; cp "$f" "$new"; done'

# api/npu_config/ 目录
"$HDC_BIN" -t "$HDC_TARGET" shell 'for f in /sys_prod/robot/out/lib/python3.12/site-packages/rknnlite/api/npu_config/*.cpython-312-aarch64-linux-gnu.so; do new="${f%-gnu.so}-ohos.so"; cp "$f" "$new"; done'

# utils/ 目录
"$HDC_BIN" -t "$HDC_TARGET" shell 'for f in /sys_prod/robot/out/lib/python3.12/site-packages/rknnlite/utils/*.cpython-312-aarch64-linux-gnu.so; do new="${f%-gnu.so}-ohos.so"; cp "$f" "$new"; done'
```

### 2.3 将 librknnrt.so 放到 rknnlite 查找的路径

rknnlite 会在 `/usr/lib/` 中查找 `librknnrt.so`。根文件系统默认只读，需先重新挂载：

```bash
"$HDC_BIN" -t "$HDC_TARGET" shell 'mount -o rw,remount /'
"$HDC_BIN" -t "$HDC_TARGET" shell 'mkdir -p /usr/lib'
"$HDC_BIN" -t "$HDC_TARGET" shell 'cp /vendor/lib64/librknnrt.so /usr/lib/'
```

> **注意：** 此修改在下次烧录固件前一直有效。如果重启后 `/usr/lib` 被还原，需要重新执行 `cp` 命令。

### 2.4 设置 LD_PRELOAD 解决 Python 符号可见性问题

板端 Python 动态链接了 `libpython3.12.so`，但 musl 的动态链接器不会自动将这些符号暴露给 `dlopen` 加载的扩展模块。需要通过 `LD_PRELOAD` 预加载：

```bash
export LD_PRELOAD=/sys_prod/robot/out/lib/libpython3.12.so.1.0
```

每次使用 rknnlite 运行 Python 前都必须设置此环境变量。

## 3. 部署模型并运行推理

### 3.1 推送 RKNN 模型到开发板

```bash
"$HDC_BIN" -t "$HDC_TARGET" file send models/502000/act_ros2_rknn.rknn /data/local/tmp/act_ros2_rknn.rknn
```

### 3.2 运行推理

```python
import numpy as np
import time

from rknnlite.api import RKNNLite

rknn = RKNNLite()
rknn.load_rknn("/data/local/tmp/act_ros2_rknn.rknn")
rknn.init_runtime(target=None)  # None = 使用本机 NPU

# 准备输入数据 — 注意：RKNN 会重排输入顺序为 [state, cam_high, cam_left]
state = np.random.randn(1, 14).astype(np.float32)        # 2D
cam_high = np.random.randn(1, 3, 480, 640).astype(np.float32)  # 4D NCHW
cam_left = np.random.randn(1, 3, 480, 640).astype(np.float32)  # 4D NCHW

t0 = time.time()
outputs = rknn.inference(inputs=[state, cam_high, cam_left])
t1 = time.time()

print(f"输出 shape: {outputs[0].shape}")  # (1, 100, 6)
print(f"推理耗时: {t1 - t0:.3f}s")        # RK3588 NPU 上约 121ms

rknn.release()
```

快速验证一行命令：

```bash
"$HDC_BIN" -t "$HDC_TARGET" shell 'LD_PRELOAD=/sys_prod/robot/out/lib/libpython3.12.so.1.0 python3 -c "
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
print(f\"output: shape={outputs[0].shape}, time={time.time()-t0:.3f}s\")
rknn.release()
"'
```

## 4. 关键技术说明

### 输入顺序

RKNN 编译器可能会重排模型输入。ACT 模型的原始 ONNX 输入为 `[cam_high, cam_left, state]`，转换后的 RKNN 模型期望 `[state, cam_high, cam_left]`。如果重新导出模型，务必通过测试推理验证输入顺序。

### 性能数据

| 指标 | 数值 |
|------|------|
| 模型大小（float16） | 约 114 MB |
| NPU 推理延迟 | 约 121 ms |
| 输出 shape | `(1, 100, 6)` — 100 个 action chunk × 6 自由度 |

### 版本兼容性

| 组件 | 版本 |
|------|------|
| rknn-toolkit2（主机，转换用） | 2.3.2 |
| rknn-toolkit-lite2（板端，推理用） | 2.3.2（预装） |
| librknnrt.so（板端） | 2.4.1b0 |
| RKNN 驱动 | 0.9.5 |

### 常见问题排查

| 现象 | 原因 | 解决方法 |
|------|------|----------|
| `ImportError: symbol not found (PyUnicode_FromFormat)` | Python 符号未暴露给 dlopen | 设置 `LD_PRELOAD=/sys_prod/robot/out/lib/libpython3.12.so.1.0` |
| `ModuleNotFoundError: No module named 'rknnlite.xxx'` | `.so` 文件后缀不匹配 | 重命名 `-gnu.so` 为 `-ohos.so`（见 2.2 节） |
| `Can not find dynamic library on RK3588!` | `/usr/lib/` 中没有 `librknnrt.so` | 从 `/vendor/lib64/` 复制（见 2.3 节） |
| `input[0] need 2dims input, but 4dims` | 输入顺序不匹配 | RKNN 重排了输入，state 需放在最前面（见 3.2 节） |
| `RKNN_ERR_MODEL_INVALID` 动态范围查询报错 | 静态 shape 模型的警告 | 可以安全忽略 |

## 5. 单板全链路：物理机械臂 + 摄像头 + RKNN 推理

前述章节将开发板作为「推理算力卡」与 Ubuntu 主机配合使用。本节介绍如何在 BQ3588HM 开发板上独立运行完整闭环：**摄像头采集 → RKNN NPU 推理 → 机械臂控制**，无需外接 Ubuntu 主机。

### 5.1 整体架构

```
BQ3588HM 开发板 (RK3588, OpenHarmony)
├── usb_cam_node_exe × 2          (top + wrist 相机, MJPEG 640x480)
├── static_transform_publisher × 4 (TF: base→camera, gripper→camera, optical)
├── lerobot_policy_node × 1        (RKNN NPU 推理, ACT 策略)
├── action_dispatcher_node × 1     (动作分发, 20Hz)
└── so101_hardware                 (ros2_control, /dev/ttyACM0)
     ↳ arm_position_controller / gripper_position_controller
```

数据流：`相机 Image → 推理节点 (NPU ~500ms) → Action Dispatcher → Joint Commands → 机械臂`

### 5.2 前置准备：内核驱动编译

开发板默认内核缺少 SO-101 机械臂和游戏手柄所需的 USB 驱动。需要重新编译内核 `boot_linux.img`。

> 本节对应独立 skill：`oh-rebuild-kernel`（详见 `.agents/skills/oh-rebuild-kernel/SKILL.md`）。

#### 5.2.1 获取 OpenHarmony 源码

```bash
# 在开发板上操作（HDC shell 或 SSH）
mkdir -p /data/oh_build
# 下载 OpenHarmony EmbodiedAI 1.0.1 源码到 /data/oh_build
# 参考：docs/OpenHarmony_EmbodiedAI_1.0.1_Release.md
```

#### 5.2.2 修改内核 defconfig

编辑 `kernel/linux/config/linux-6.6/rk3588/arch/arm64_defconfig`，确认以下选项已启用：

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
+CONFIG_HID_STEELSERIES=y
+CONFIG_HID_WIIMOTE=y
```

关键说明：
- **`CONFIG_USB_ACM=y`**：SO-101 机械臂使用 CH9102 芯片，该芯片以 CDC ACM 模式报告，需要 `USB_ACM` 驱动（不仅仅是 `USB_SERIAL_CH341`）
- 游戏手柄相关配置用于遥操作模式

#### 5.2.3 编译并刷入 boot_linux.img

```bash
# 在开发板上执行
cd /data/oh_build
./build.sh -p bq3588 --ccache

# 产物位于：
# out/bq3588/packages/phone/images/boot_linux.img

# 备份并刷入（mmcblk0p5 是 boot_linux 分区）
dd if=/dev/block/by-name/boot_linux of=/data/boot_linux_backup.img
dd if=out/bq3588/packages/phone/images/boot_linux.img of=/dev/block/by-name/boot_linux
reboot
```

#### 5.2.4 验证

重启后检查串口设备：

```bash
ls -la /dev/ttyACM0
# 应显示：crw-rw---- 1 root radio 166, 0 ... /dev/ttyACM0
dmesg | grep cdc_acm
# 应显示：cdc_acm 5-1.2:1.0: ttyACM0: USB ACM device
```

### 5.3 机械臂校准

首次使用前必须在板端执行校准（交互式操作，需要手动转动机械臂关节）：

```bash
# SSH 登录板端
ssh root@192.168.136.111

# 加载环境
cd /data && . ./ros2ohos.env && . /data/ibrobot/install/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 执行校准
ros2 run so101_hardware calibrate_arm --arm follower --port /dev/ttyACM0
```

校准完成后，JSON 保存在 `~/.calibrate/so101_follower_calibrate.json`（SSH 环境下 `HOME=/root`）。

推理节点查找的路径是 `$(env HOME)/.calibrate/so101_follower_calibrate.json`。如果使用 HDC shell（`HOME=/`），需要做符号链接：

```bash
# HDC shell 环境
mkdir -p /.calibrate
ln -sf /root/.calibrate/so101_follower_calibrate.json /.calibrate/so101_follower_calibrate.json
```

### 5.4 交叉编译第三方 ROS 2 包

板端 ROS 2 运行时（`/data/install`）不包含所有 IB Robot 需要的包。以下包需要通过 OH Docker 交叉编译工具链单独编译部署。

详细流程参考 `.agents/skills/oh-cross-build-ros-pkg/SKILL.md`。

#### 5.4.1 编译 usb_cam

usb_cam 用于 USB 摄像头图像采集，板端预装版本缺少必要的依赖库。需用 OH 交叉编译工具链重新编译：

```bash
# 构建根目录（已配置好的 OH 交叉编译环境）
OH_CUSTOM_ROOT=~/Research/bq3588_oh_ws/custom_build_root

# 确认源码已存在
ls ${OH_CUSTOM_ROOT}/ibrobot_oh_ws/src/usb_cam/package.xml
# 应输出 usb_cam 0.8.1

# Docker 交叉编译
docker run --rm -i \
    -e WS_ROOT=/mnt/ohos/tmp \
    -e OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18 \
    --name ibrobot-oh-build \
    -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
    -v "${OH_CUSTOM_ROOT}:/mnt/ohos/tmp" \
    voxelsky/ohos-ros-humble-builder:v0.1.5 \
    bash -lc "
set -euo pipefail
export OHOS_CPU=aarch64
export OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18
build-ros-humble --custom \
    --wd /mnt/ohos/tmp/ibrobot_oh_ws \
    --custom-prefix /data/ibrobot/install \
    --colcon-args --packages-select usb_cam
"

# 修复文件所有权
docker run --rm \
    -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
    voxelsky/ohos-ros-humble-builder:v0.1.5 \
    sh -c "chown -R $(id -u):$(id -g) /mnt/ohos/ibrobot_oh_ws/install/usb_cam || true"

# 验证产物
file ${OH_CUSTOM_ROOT}/ibrobot_oh_ws/install/usb_cam/lib/usb_cam/usb_cam_node_exe
# 应输出：ELF 64-bit LSB pie executable, ARM aarch64, ... ld-musl-aarch64.so.1

# 部署到板端
HDC_BIN=/path/to/hdc
HDC_TARGET=192.168.136.111:8710
"$HDC_BIN" -t "$HDC_TARGET" file send ${OH_CUSTOM_ROOT}/ibrobot_oh_ws/install/usb_cam/lib/usb_cam/usb_cam_node_exe /data/ibrobot/install/usb_cam/lib/usb_cam/usb_cam_node_exe
"$HDC_BIN" -t "$HDC_TARGET" file send ${OH_CUSTOM_ROOT}/ibrobot_oh_ws/install/usb_cam/lib/libusb_cam.so /data/ibrobot/install/usb_cam/lib/libusb_cam.so
"$HDC_BIN" -t "$HDC_TARGET" file send ${OH_CUSTOM_ROOT}/ibrobot_oh_ws/install/usb_cam/lib/libusb_cam_node.so /data/ibrobot/install/usb_cam/lib/libusb_cam_node.so
```

#### 5.4.2 其他纯 Python 包部署

以下 Python 包为纯 Python（无 C 扩展），可直接复制或符号链接到板端 site-packages：

| 包名 | 版本 | 用途 | 安装方式 |
|------|------|------|----------|
| `voice_asr_service` | - | 语音 ASR 服务 | 符号链接 |
| `robot_description` | - | URDF/meshes 数据包 | 符号链接 |
| `deepdiff` | 7.0.1 | 配置对比 | 主机 pip download → 解压 whl |
| `ordered-set` | 4.1.0 | deepdiff 依赖 | 主机 pip download → 解压 whl |
| `scservo_sdk` | - | 飞特舵机 SDK | 从主机复制 |
| `draccus` | 0.10.0 | LeRobot 配置框架 | 主机 pip download → 解压 whl |
| `mergedeep` | 1.3.4 | draccus 依赖 | 主机 pip download → 解压 whl |
| `mypy_extensions` | 1.1.0 | draccus 依赖 | 主机 pip download → 解压 whl |
| `pyyaml_include` | 1.4.1 | draccus 依赖 | 主机 pip download → 解压 whl |
| `toml` | 0.10.2 | draccus 依赖 | 主机 pip download → 解压 whl |
| `typing_extensions` | 4.15.0 | draccus 依赖 | 主机 pip download → 解压 whl |
| `typing_inspect` | 0.9.0 | draccus 依赖 | 主机 pip download → 解压 whl |

纯 Python whl 安装方法（板端无 pip）：

```bash
# 在主机下载
pip download <package>==<version> --no-deps -d /tmp/<pkg>

# 解压 whl 到板端 site-packages（用板端系统 Python 的 zipfile 模块）
HDC_BIN=/path/to/hdc
HDC_TARGET=192.168.136.111:8710

# 先发送 whl 到板端
"$HDC_BIN" -t "$HDC_TARGET" file send /tmp/<pkg>/<package>-<ver>-py3-none-any.whl /data/<package>.whl

# 在板端解压（板端 pip 不可用，用 Python zipfile）
ssh root@192.168.136.111 '/sys_prod/robot/out/bin/python3 -c "import zipfile; z=zipfile.ZipFile(\"/data/<package>.whl\"); z.extractall(\"/data/local/skh-run/usr/lib/python3.12/site-packages\"); print(\"OK\")"'
```

### 5.5 板端 ROS + Torch 双生态运行环境

板端同时运行 ROS 2 节点和 PyTorch 推理需要叠加两套运行时环境：

| 生态 | 基础路径 | 说明 |
|------|----------|------|
| ROS 2 Humble | `/data/install` + `/sys_prod/robot/out` | OH 预编译 ROS 2 |
| IB Robot 包 | `/data/ibrobot/install` | 交叉编译产物 |
| Torch 运行时 | `/data/local/skh-run/usr` | skh-run torch (aarch64) |
| RKNN 运行时 | `/vendor/lib64/librknnrt.so` | NPU 驱动 |

**核心问题**：Torch 运行时（skh-run）自带 Python 3.12 运行时（`libpython3.12.so`），与系统 Python 3.12 存在 ABI 冲突。不能单独用 `PYTHONPATH` 叠加（会 SIGSEGV），必须通过 `LD_PRELOAD` + `PYTHONPATH` + `LD_LIBRARY_PATH` 完整叠加。

#### 5.5.1 完整环境变量设置

```bash
# 每次启动前在 SSH 会话中执行

# 第一步：先设置 Python 运行时（setup.sh 内部需要 Python 扫描已安装包）
export PYTHONHOME=/data/local/skh-run/usr
export LD_PRELOAD=${PYTHONHOME}/lib/libpython3.12.so.1.0:${PYTHONHOME}/lib/libomp.so
export LD_LIBRARY_PATH=${PYTHONHOME}/lib:/sys_prod/robot/out/lib:/data/install/lib:/vendor/lib64
export PYTHONPATH=${PYTHONHOME}/lib/python3.12/site-packages:/sys_prod/robot/out/lib/python3.12/site-packages:/data/install/lib/python3.12/site-packages

# 第二步：加载 ROS 和 IB Robot 环境（此时 Python 可用，能正确扫描所有包）
cd /data && . ./ros2ohos.env && . /data/ibrobot/install/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 第三步：补充 IB Robot 特有的 Python 和动态库路径
export PATH=${PYTHONHOME}/bin:$PATH
export PYTHONPATH=/data/ibrobot/install/lerobot/src:/data/ibrobot/install/inference_service/lib/python3.12/site-packages:/data/ibrobot/install/tensormsg/lib/python3.12/site-packages:/data/ibrobot/install/robot_config/lib/python3.12/site-packages:/data/ibrobot/install/ibrobot_msgs/lib/python3.12/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=/data/ibrobot/install/ibrobot_msgs/lib:/data/ibrobot/install/inference_service/lib:/data/ibrobot/install/robot_config/lib:/data/ibrobot/install/so101_hardware/lib:${PYTHONHOME}/lib/python3.12/site-packages/torch/lib:${PYTHONHOME}/lib/python3.12/site-packages/torchaudio/lib:${LD_LIBRARY_PATH}
```

> **关键**：`PYTHONHOME` + `LD_PRELOAD` 必须在 `source setup.sh` **之前**设置，否则 `local_setup.sh` 调用 Python 扫描包时会失败，导致 `robot_description` 等包的路径不会被加入 `AMENT_PREFIX_PATH`。

#### 5.5.2 DDS 选择

**必须使用 CycloneDDS**。板端 FastRTPS 会崩溃：

```bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

### 5.6 板端推理代码适配

#### 5.6.1 RKNN 模式跳过 LeRobot Preprocessor

RKNN 模型的预处理（归一化等）已 baked into 模型中，不需要 LeRobot 的 `make_pre_post_processors`。`coordinator.py` 已修改为 RKNN 模式下跳过 LeRobotPreprocessor 和 LeRobotPostprocessor：

```python
# src/inference_service/inference_service/core/coordinator.py
is_rknn = self._device_name == "rknn"

if is_rknn:
    # RKNN: 使用基础 TensorPreprocessor（仅做 numpy→tensor 转换）
    self._preprocessor = TensorPreprocessor(device=self._device)
    self._postprocessor = TensorPostprocessor(device=self._device)
else:
    # 非 RKNN: 使用 LeRobot 完整预处理管线
    self._preprocessor = TensorPreprocessor(policy_path=policy_path, device=self._device)
    self._postprocessor = TensorPostprocessor(policy_path=policy_path, device=self._device)
```

这避免了在板端安装完整的 `lerobot` 依赖链（`huggingface_hub`、`diffusers`、`transformers` 等大量 C 扩展包）。

#### 5.6.2 模型配置

板端 YAML 配置（`so101_single_arm.yaml`）需指定 RKNN 模型：

```yaml
models:
  so101_act_rknn:
    path: /data/ibrobot/models/502000/pretrained_model  # 必须用绝对路径
    policy_type: act
    device: rknn
    lerobot_norm_mode: range_m100_100

control_modes:
  model_inference:
    inference:
      enabled: true
      model: so101_act_rknn  # 指向上面定义的 RKNN 模型
```

**注意**：模型 `path` 必须用绝对路径。相对路径会基于 `_workspace_root()`（源码安装路径）解析，在板端会指向错误位置。

### 5.7 开启 SSH 服务（可选但推荐）

HDC shell 的 PTY 缓冲区有限，大量日志会被截断。建议开启板端 SSH：

```bash
# 生成 host key
/sys_prod/robot/out/bin/ssh-keygen -A

# 启动 sshd（配置文件在 /sys_prod/robot/out/etc/sshd_config）
/sys_prod/robot/out/sbin/sshd -f /sys_prod/robot/out/etc/sshd_config -E /data/sshd.log

# 从主机连接
ssh root@192.168.136.111
```

SSH 的 PTY 缓冲区正常，适合长时间运行 launch 并查看完整日志。

### 5.8 完整启动流程

#### 5.8.1 清理残留进程

```bash
# 每次重启前清理（避免 DDS 残留通信）
pkill -9 -f "ros2 launch\|lerobot_policy_node\|action_dispatcher_node\|usb_cam_node_exe\|static_transform_publisher"
```

#### 5.8.2 启动全链路

```bash
# SSH 登录板端
ssh root@192.168.136.111

# 1. 先设置 Python 运行时（setup.sh 需要 Python）
export PYTHONHOME=/data/local/skh-run/usr
export LD_PRELOAD=${PYTHONHOME}/lib/libpython3.12.so.1.0:${PYTHONHOME}/lib/libomp.so
export LD_LIBRARY_PATH=${PYTHONHOME}/lib:/sys_prod/robot/out/lib:/data/install/lib:/vendor/lib64
export PYTHONPATH=${PYTHONHOME}/lib/python3.12/site-packages:/sys_prod/robot/out/lib/python3.12/site-packages:/data/install/lib/python3.12/site-packages

# 2. 加载 ROS + IB Robot 环境
cd /data && . ./ros2ohos.env && . /data/ibrobot/install/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# 3. 补充 IB Robot 路径
export PATH=${PYTHONHOME}/bin:$PATH
export PYTHONPATH=/data/ibrobot/install/lerobot/src:/data/ibrobot/install/inference_service/lib/python3.12/site-packages:/data/ibrobot/install/tensormsg/lib/python3.12/site-packages:/data/ibrobot/install/robot_config/lib/python3.12/site-packages:/data/ibrobot/install/ibrobot_msgs/lib/python3.12/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=/data/ibrobot/install/ibrobot_msgs/lib:/data/ibrobot/install/inference_service/lib:/data/ibrobot/install/robot_config/lib:/data/ibrobot/install/so101_hardware/lib:${PYTHONHOME}/lib/python3.12/site-packages/torch/lib:${PYTHONHOME}/lib/python3.12/site-packages/torchaudio/lib:${LD_LIBRARY_PATH}

# 4. 启动
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=false \
    control_mode:=model_inference \
    2>&1 | tee /data/launch.log
```

#### 5.8.3 预期输出

```
[usb_cam_node_exe-1] [INFO] [top_camera]: Starting 'top' (/dev/video20) at 640x480 via mmap (mjpeg2rgb) at 30 FPS
[usb_cam_node_exe-1] [INFO] [top_camera]: Timer triggering every 33 ms
[usb_cam_node_exe-2] [INFO] [wrist_camera]: Starting 'wrist' (/dev/video22) at 640x480 via mmap (mjpeg2rgb) at 60 FPS
[usb_cam_node_exe-2] [INFO] [wrist_camera]: Timer triggering every 16 ms
[lerobot_policy_node-7] [INFO] [act_inference_node]: Using inference_backend=rknn, tensor_device=cpu
[lerobot_policy_node-7] [INFO] [act_inference_node]: DispatchInfer Action Server ready
[lerobot_policy_node-7] [INFO] [act_inference_node]: lerobot_policy node ready (monolithic): policy_type=rknn, chunk_size=100
[lerobot_policy_node-7] [INFO] [act_inference_node]: ✓ First inference complete (monolithic): total=~500ms
[action_dispatcher_node-8] [INFO] [action_dispatcher]: ✓ First inference received: chunk=100
[action_dispatcher_node-8] [INFO] [action_dispatcher]: [stats] inferences=N, avg_latency=~500ms, queue=NN
```

#### 5.8.4 已知 warning（可安全忽略）

| Warning | 原因 | 处理 |
|---------|------|------|
| `robot_description not found` | URDF 包未部署，ros2_control 不需要它即可运行 | 忽略 |
| `Camera calibration file not found` | 未做相机内参标定 | 忽略 |
| `Query dynamic range failed (RKNN_ERR_MODEL_INVALID)` | 静态 shape 模型的正常警告 | 忽略 |
| `Ignoring unexpected goal response` | DDS 残留，重启前未清理进程 | 启动前清理 |
| `unknown control 'white_balance_temperature_auto'` | USB 摄像头不支持该 V4L2 控制 | 忽略 |

### 5.9 性能数据（单板闭环）

| 指标 | 数值 |
|------|------|
| NPU 推理延迟（首次含加载） | ~900ms |
| NPU 推理延迟（稳定） | ~500ms |
| 总端到端延迟（含预处理） | ~520ms |
| Action chunk size | 100 |
| 控制频率 | 20 Hz |
| 相机帧率 | top: 30 FPS, wrist: 60 FPS |
| 模型 | ACT (so101_act_rknn), 6 DoF |

### 5.10 故障排查

| 问题 | 原因 | 解决方法 |
|------|------|----------|
| `/dev/ttyACM0` 不存在 | 内核缺少 `CONFIG_USB_ACM=y` | 重新编译内核（5.2 节） |
| `ModuleNotFoundError: No module named 'draccus'` | LeRobot preprocessor 缺依赖 | 安装 draccus 及依赖（5.4.2 节），或确认 RKNN 模式已正确跳过 LeRobotPreprocessor |
| usb_cam crash (`libc++abi: terminating`) | 使用了错误的 usb_cam 二进制 | 用 OH 交叉编译版本替换（5.4.1 节） |
| `RKNN model file not found` | 模型 path 为相对路径 | YAML 中使用绝对路径 `/data/ibrobot/models/...` |
| `Calibration file not found` | SSH vs HDC 的 HOME 不同 | 符号链接校准文件（5.3 节） |
| `Cannot open device /dev/videoX` | usb_cam 枚举所有设备时的信息日志 | 忽略，只要目标设备正常打开 |
| 推理节点 SIGSEGV | LD_PRELOAD 未设置或 torch ABI 冲突 | 确认完整环境变量（5.5.1 节） |

## 6. 相关文件

| 文件 | 说明 |
|------|------|
| `models/502000/act_ros2_rknn.rknn` | RKNN 模型（float16，114 MB） |
| `models/502000/act_ros2_rknn.onnx` | RKNN 优化后的 ONNX（220 MB） |
| `src/model_utils/model_utils/export_onnx_rknn.py` | 导出脚本（ONNX → RKNN） |
| `.agents/skills/rknn-convert/convert_to_rknn.py` | 通用 ONNX → RKNN 转换器 |
| `.agents/skills/oh-cross-build-ros-pkg/SKILL.md` | OH 交叉编译第三方 ROS 包 skill |
| `.agents/skills/ibrobot-bq3588hm-oh/SKILL.md` | 板端 OpenHarmony 运行时事实 |
| `src/inference_service/inference_service/core/coordinator.py` | RKNN 模式跳过 LeRobotPreprocessor |
| `docs/OpenHarmony_EmbodiedAI_1.0.1_Release.md` | OH 源码获取参考 |
