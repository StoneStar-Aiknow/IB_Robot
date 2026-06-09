# BQ3588HM OpenHarmony ROS 安装与 IB_Robot 交叉编译说明

本文档面向 **Bearkey BQ3588HM + OpenHarmony 5.1** 的使用场景，整理两件事：

1. 如何在板端安装官方提供的 OpenHarmony ROS 2 Humble 运行时。
2. 如何在 Ubuntu 主机上交叉编译 IB_Robot 的 ROS 包，并部署到开发板。
3. 如何交叉编译并使用板端 `usb_cam`，包括 ffmpeg / `libswscale` / RViz 显示相关注意事项。

本文档不重复说明烧录、HDC/SSH 联网等基础内容；这些内容请先参考：

- [BQ3588HM_board_usage.md](./BQ3588HM_board_usage.md)

## References

- 官方二进制使用文档（包含 **开发板镜像**、**ROS 系统依赖二进制包**、**ROS 2 Humble 运行时二进制包**）：
  - <https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md>
- 官方 Docker 交叉编译文档：
  - <https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/docker-build.md>
- 其中与本文最相关的小节：
  - `docker-build.md` -> `用户自定义 ROS2 包/项目编译和使用`

> 说明：下面的下载项、目录约定和交叉编译流程，都是在官方文档基础上，结合本仓库的
> `scripts/openharmony/build_ibrobot_oh_custom.sh` 做的 IB_Robot 化整理。

## 缺少前置条件时先看哪里

- **如果主机侧没有可用的 `hdc`**：
  - 先看 `docs/BQ3588HM_board_usage.md` → **第一阶段：HDC 调试工具准备**
  - 再看本文 **1.4 OpenHarmony ROS SDK**
- **如果主机侧没有预设 `OH_ROOT` / `OH_DOWNLOAD_ROOT` / `OH_CUSTOM_ROOT`**：
  - 先看本文 **第 2 节：统一放到一个外部目录**
  - 再看本文 **第 4 节：`build_ibrobot_oh_custom.sh` 里的变量到底对应什么**

如果这些变量在当前 shell 里不存在，不要让自动化工具猜目录；请先按本文整理目录并导出
`OH_ROOT`，或者在脚本里显式传 `--root`、`--sdk-tar`、`--sysdeps-tar`、`--humble-tar`。

对于 `hdc`，也推荐不要在脚本里写死某个用户私有绝对路径；更稳妥的做法是先把 SDK 的
`toolchains` 目录导出到 `PATH`，并把这条导出写入 `~/.bashrc` 或 `~/.zshrc`，之后直接
使用 `hdc` 命令。

## 1. 需要准备的下载内容

对于 BQ3588HM（`aarch64`）场景，建议至少准备下面四类文件：

| 类别 | 官方用途 | 官方来源 |
| --- | --- | --- |
| BQ3588HM OpenHarmony 镜像 | 烧录开发板 | `usage.md` 中的 `oh5.1 bq3588 build` |
| `ohos-*-sysdeps-*.tar.gz` | OpenHarmony ROS 系统依赖包 | `usage.md` 中的 `ohos-sysdeps-xxx.tar.gz` |
| `ohos-humble-build-*.tar.gz` | OpenHarmony ROS 2 Humble 运行时发行版 | `usage.md` 中的 `ohos-humble-build-xxx.tar.gz` |
| `ohos-sdk-18-linux-aarch64-*.tar.gz` | 主机交叉编译时使用的 OHOS SDK | `docker-build.md` 中的 `ohos-ros-sdk-build` |

另外还需要一个 Docker 编译镜像：

```bash
docker pull voxelsky/ohos-ros-humble-builder:v0.1.5
```

官方文档当前给出的下载入口如下：

### 1.1 BQ3588HM 镜像

- 百度网盘：<https://pan.baidu.com/s/1BA5F8Ph7gpsrawpzvPEofA?pwd=kaq4>（提取码：`kaq4`）
- 交大云盘：<https://pan.sjtu.edu.cn/web/share/52224e51bcb98be6ab043c5846ddfb7f>（提取码：`m7fp`）

### 1.2 OpenHarmony ROS 系统依赖包

- 百度网盘：<https://pan.baidu.com/s/14b4YyQWxIBdKj2ZOu2I-VQ?pwd=sb3y>（提取码：`sb3y`）
- 交大云盘：<https://pan.sjtu.edu.cn/web/share/0bfcc408563cba4940905ba54607da38>（提取码：`dch8`）

### 1.3 OpenHarmony ROS 2 Humble 发行版

- 百度网盘：<https://pan.baidu.com/s/1562-HKLWZXbkNeMVHa3PNg?pwd=5tuy>（提取码：`5tuy`）
- 交大云盘：<https://pan.sjtu.edu.cn/web/share/9fe41dd3ac1fc712b9157a46778545c3>（提取码：`w5kz`）

### 1.4 OpenHarmony ROS SDK

- 百度网盘：<https://pan.baidu.com/s/168iE3OZT-5qswn24tf1oAA?pwd=k8wk>（提取码：`k8wk`）
- 交大云盘：<https://pan.sjtu.edu.cn/web/share/7c24241fbfb00c683ecedf73d6366fa4>（提取码：`e1hd`）

> 建议总是选择**日期最新且架构为 `aarch64`** 的文件。不要混用 `x86_64` 和 `aarch64`
> 产物。

## 2. 统一放到一个外部目录：推荐的主机目录布局

更推荐的做法是：**由用户自己指定一个统一的 OpenHarmony 主机目录**，把下载内容和交叉编译
目录都放在这里，并通过脚本参数 `--oh-root` 传给
`scripts/openharmony/build_ibrobot_oh_custom.sh`，而不是散落在多个临时目录中。

例如把这个统一目录记为 `<OH_ROOT>`：

```text
<OH_ROOT>/
├── downloads/
│   ├── images/
│   │   └── oh5.1-bq3588-build-...
│   ├── sdk/
│   │   └── ohos-sdk-18-linux-aarch64-....tar.gz
│   ├── sysdeps/
│   │   └── ohos-18-sysdeps-aarch64-....tar.gz
│   └── runtime/
│       └── ohos-humble-build-aarch64-....tar.gz
└── custom_build_root/
    ├── install/
    ├── ibrobot_oh_ws/
    │   └── src/
    ├── ohos-robot-toolchain/
    │   └── 18/native/
    ├── ros_ros2_base/
    └── version/
```

下文统一把这个目录记为：

```bash
export OH_ROOT="<your-unified-oh-root>"
export OH_DOWNLOAD_ROOT="$OH_ROOT/downloads"
```

脚本会默认按下面这个约定从 `OH_ROOT` 派生路径：

```text
OH_DOWNLOAD_ROOT = $OH_ROOT/downloads
OH_CUSTOM_ROOT   = $OH_ROOT/custom_build_root
```

这里的 `OH_ROOT` / `OH_DOWNLOAD_ROOT` / `OH_CUSTOM_ROOT` 都是**主机侧交叉编译变量**，
不是开发板上的环境变量。如果当前 shell 里没有这些变量，这通常是正常的；请先按本节准备
目录布局并自行导出，或者在调用脚本时改用显式参数，不要依赖“自动猜测主机目录”。

## 3. 板端安装 OpenHarmony ROS 2 Humble

这一部分对应官方 `usage.md` 的主流程。

### 3.1 将运行时包上传到开发板

如果板端 `/data` 里还没有这两个包，请先上传：

- `ohos-humble-build-*.tar.gz`
- `ohos-*-sysdeps-*.tar.gz`

例如使用 HDC：

```bash
HDC_BIN=<path-to-hdc>
HDC_TARGET=<board-ip>:8710

"$HDC_BIN" -t "$HDC_TARGET" file send \
  "$OH_DOWNLOAD_ROOT/runtime/ohos-humble-build-aarch64-....tar.gz" \
  /data/ohos-humble-build-aarch64.tar.gz

"$HDC_BIN" -t "$HDC_TARGET" file send \
  "$OH_DOWNLOAD_ROOT/sysdeps/ohos-18-sysdeps-aarch64-....tar.gz" \
  /data/ohos-18-sysdeps-aarch64.tar.gz
```

当前实验环境中，板子上已经验证存在的典型路径是：

- `/data/ohos-humble-build-aarch64-20260115100449.tar.gz`
- `/data/ohos-18-sysdeps-aarch64-20260115.tar.gz`

### 3.2 在板端解压并加载 ROS 环境

在开发板上执行：

```sh
cd /data
tar -zxpvf ohos-humble-build-aarch64.tar.gz
tar -zxpvf ohos-18-sysdeps-aarch64.tar.gz

# 注意：必须在 ros2ohos.env 所在目录执行
. ./ros2ohos.env
```

然后检查：

```sh
ros2 topic list
python3 --version
python3 -m pip --version
```

### 3.3 BQ3588HM 板子的额外注意事项

对于本仓库当前验证过的 BQ3588HM 开发板：

- `ros2ohos.env` 位于 `/data/ros2ohos.env`
- ROS 运行时目录通常是 `/data/install`
- sysdeps 目录通常是 `/data/out`
- 当前板端的 `/data/sysdeps.env` 已补过 `mount -o remount,rw /`，因此执行
  `. ./ros2ohos.env` 时可以正常准备 SSH 相关目录

如果你需要 HDC TCP 地址、SSH、公钥登录、只读根文件系统等细节，请继续看：

- [BQ3588HM_board_usage.md](./BQ3588HM_board_usage.md)

## 4. `build_ibrobot_oh_custom.sh` 里的变量到底对应什么

`scripts/openharmony/build_ibrobot_oh_custom.sh` 是我们把官方
`docker-build.md -> 用户自定义 ROS2 包/项目编译和使用` 流程，封装成 IB_Robot 专用脚本后的实现。

推荐把 `OH_ROOT` 作为这个脚本的一级输入，再由脚本自动展开出下载目录和交叉编译目录。

它开头定义的变量，建议理解为下面这张表：

| 变量 | 含义 | 推荐值 |
| --- | --- | --- |
| `OH_ROOT` | OpenHarmony 主机统一根目录 | `<your-unified-oh-root>` |
| `OH_DOWNLOAD_ROOT` | 下载内容总目录 | `$OH_ROOT/downloads` |
| `OH_CUSTOM_ROOT` | 交叉编译总根目录 | `$OH_ROOT/custom_build_root` |
| `OH_CUSTOM_WS` | 放 IB_Robot ROS 工作区的目录 | `$OH_CUSTOM_ROOT/ibrobot_oh_ws` |
| `OH_CUSTOM_SRC` | 交叉编译工作区的 `src/` | `$OH_CUSTOM_WS/src` |
| `OH_CUSTOM_TOOLCHAIN_ROOT` | 解压 OHOS SDK 的目录 | `$OH_CUSTOM_ROOT/ohos-robot-toolchain` |
| `OH_CUSTOM_SDK_TAR_GLOB` | 官方 `ohos-sdk-18-linux-aarch64-*.tar.gz` 的位置 | `$OH_DOWNLOAD_ROOT/sdk/...tar.gz` |
| `OH_CUSTOM_SYSDEPS_TAR_GLOB` | 官方 `ohos-*-sysdeps-*.tar.gz` 的位置 | `$OH_DOWNLOAD_ROOT/sysdeps/...tar.gz` |
| `OH_CUSTOM_HUMBLE_TAR_GLOB` | 官方 `ohos-humble-build-*.tar.gz` 的位置 | `$OH_DOWNLOAD_ROOT/runtime/...tar.gz` |
| `OH_CUSTOM_ROS2_BASE_REPO` | 自动克隆的 `ros_ros2_base` 仓库目录 | `$OH_CUSTOM_ROOT/ros_ros2_base` |
| `OH_CUSTOM_VERSION_REPO` | 自动克隆的 `version` 仓库目录 | `$OH_CUSTOM_ROOT/version` |
| `OH_CUSTOM_PREFIX` | 板端最终安装前缀 | `/data/ibrobot/install` |
| `OH_CUSTOM_IMAGE` | Docker 构建镜像 | `voxelsky/ohos-ros-humble-builder:v0.1.5` |

### 4.1 这些下载内容和变量的对应关系

你下载的文件，建议这样放：

```text
$OH_DOWNLOAD_ROOT/sdk/ohos-sdk-18-linux-aarch64-....tar.gz
$OH_DOWNLOAD_ROOT/sysdeps/ohos-18-sysdeps-aarch64-....tar.gz
$OH_DOWNLOAD_ROOT/runtime/ohos-humble-build-aarch64-....tar.gz
```

脚本运行时会自动做这些事：

1. 把 `ohos-humble-build-*.tar.gz` 解压到 `OH_CUSTOM_ROOT`，得到 `install/`
2. 把 `ohos-sdk-18-linux-aarch64-*.tar.gz` 解压到 `OH_CUSTOM_TOOLCHAIN_ROOT/18/native`
3. 把 `ohos-*-sysdeps-*.tar.gz` 里的 Python 3.12 / sframe 相关内容 overlay 到 SDK sysroot
4. 自动克隆：
   - `https://gitcode.com/openharmony-robot/ros_ros2_base.git`
   - `https://gitcode.com/openharmony-robot/version.git`
5. 从 IB_Robot 仓库复制交叉编译需要的包到 `OH_CUSTOM_WS/src`

所以这不是“又下载一堆和脚本无关的文件”；相反，这几个 tarball 就是脚本真正需要消费的输入。

## 5. 用我们的脚本交叉编译 IB_Robot

### 5.1 前提条件

主机侧准备好：

- Ubuntu 主机
- Docker
- `voxelsky/ohos-ros-humble-builder:v0.1.5`
- 下载好的：
  - `ohos-sdk-18-linux-aarch64-*.tar.gz`
  - `ohos-*-sysdeps-*.tar.gz`
  - `ohos-humble-build-aarch64-*.tar.gz`

### 5.2 推荐执行方式

先在主机上准备统一根目录：

```bash
export OH_ROOT="<your-unified-oh-root>"
```

然后在 IB_Robot 仓库根目录执行：

```bash
./scripts/openharmony/build_ibrobot_oh_custom.sh \
  --oh-root "$OH_ROOT" \
  --image voxelsky/ohos-ros-humble-builder:v0.1.5 \
  --packages ibrobot_msgs,tensormsg,robot_config,inference_service
```

只要 `OH_ROOT` 下的目录布局符合第 2 节约定，脚本会自动从：

- `$OH_DOWNLOAD_ROOT/sdk/`
- `$OH_DOWNLOAD_ROOT/sysdeps/`
- `$OH_DOWNLOAD_ROOT/runtime/`

解析 SDK、sysdeps 和 Humble runtime tarball。

如果你的文件不在默认布局里，再额外使用：

- `--sdk-tar`
- `--sysdeps-tar`
- `--humble-tar`

做精确覆盖即可。

### 5.3 这条命令实际做了什么

它本质上是在封装官方文档里的这类调用：

```bash
build-ros-humble --custom \
  --wd <项目工作目录> \
  --custom-prefix /data/ibrobot/install \
  --colcon-args --packages-select ibrobot_msgs tensormsg robot_config inference_service
```

脚本会在容器里把下面两个环境变量设好：

```bash
WS_ROOT=/mnt/ohos/tmp
OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18
```

并把你的 `OH_CUSTOM_ROOT` 挂载进容器。因此编译产物会直接落回主机目录。

## 6. 编译完成后产物在哪

编译完成后，重点看：

```text
$OH_CUSTOM_ROOT/ibrobot_oh_ws/install
```

这就是需要部署到板端的自定义 ROS 工作区安装结果。

如果你沿用默认前缀 `/data/ibrobot/install`，那推荐的打包和部署方式是：

```bash
cd "$OH_CUSTOM_ROOT/ibrobot_oh_ws"
tar -zcpf ibrobot-oh-install.tar.gz install
```

然后上传到板端：

```bash
HDC_BIN=<path-to-hdc>
HDC_TARGET=<board-ip>:8710

"$HDC_BIN" -t "$HDC_TARGET" file send \
  "$OH_CUSTOM_ROOT/ibrobot_oh_ws/ibrobot-oh-install.tar.gz" \
  /data/ibrobot-oh-install.tar.gz
```

在板端解压：

```sh
cd /data
mkdir -p /data/ibrobot
tar -zxpf ibrobot-oh-install.tar.gz -C /data/ibrobot
ls -lah /data/ibrobot/install
```

最终目录应当是：

```text
/data/ibrobot/install
```

## 7. 板端 `usb_cam` 包的交叉编译与使用

`usb_cam` 用于采集 BQ3588HM 板端 USB 摄像头图像。在当前验证过的环境中，建议把它视为
**需要单独交叉编译和部署的第三方 ROS 2 包**，不要直接依赖板端预装版本。

原因有三点：

1. 板端预装环境不一定包含 `usb_cam`，即使存在也可能和 `/data/ibrobot/install` 里的 overlay 顺序冲突。
2. `usb_cam` 是 C++ 节点，需要链接 OpenHarmony ROS 2 Humble 的 `rclcpp`、`class_loader` 等库，必须使用 OH 交叉工具链产物。
3. `usb_cam` 的图像格式转换依赖 ffmpeg 的 `libswscale` / `libavcodec` / `libavutil`，部署时要保证这些库可被动态链接器找到。

### 7.1 推荐部署形态

建议最终板端目录如下：

```text
/data/ibrobot/install/usb_cam/
├── bin/
├── lib/
│   ├── libusb_cam.so
│   ├── libusb_cam_node.so
│   ├── libswscale.so.5 -> /data/out/lib/libswscale.so.5
│   ├── libavcodec.so.58 -> /data/out/lib/libavcodec.so.58
│   ├── libavutil.so.56 -> /data/out/lib/libavutil.so.56
│   └── usb_cam/
│       └── usb_cam_node_exe
└── share/
```

`usb_cam_node_exe` 应是 OpenHarmony aarch64/musl ELF，而不是 shell wrapper：

```sh
file /data/ibrobot/install/usb_cam/lib/usb_cam/usb_cam_node_exe
# 期望包含：ELF ... arm64 ... /lib/ld-musl-aarch64.so.1
```

### 7.2 主机侧交叉编译 `usb_cam`

先确认主机侧已经按本文第 2 节准备好：

```bash
export OH_ROOT="<your-unified-oh-root>"
export OH_CUSTOM_ROOT="$OH_ROOT/custom_build_root"
export OH_CUSTOM_WS="$OH_CUSTOM_ROOT/ibrobot_oh_ws"
export OH_CUSTOM_SRC="$OH_CUSTOM_WS/src"
```

准备源码。建议使用 ROS 2 Humble 可用的 `ros-drivers/usb_cam` 版本；当前验证过的包版本为
`usb_cam 0.8.x`：

```bash
mkdir -p "$OH_CUSTOM_SRC"

# 如果已经有源码，先确认 package.xml 存在即可。
test -f "$OH_CUSTOM_SRC/usb_cam/package.xml" || \
  git clone --depth 1 https://github.com/ros-drivers/usb_cam.git "$OH_CUSTOM_SRC/usb_cam"
```

使用官方 OH Docker builder 交叉编译：

```bash
docker run --rm -i \
  -e WS_ROOT=/mnt/ohos/tmp \
  -e OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18 \
  --name ibrobot-oh-build \
  -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
  -v "${OH_CUSTOM_ROOT}:/mnt/ohos/tmp" \
  voxelsky/ohos-ros-humble-builder:v0.1.5 \
  bash -lc '
set -euo pipefail
export OHOS_CPU=aarch64
export OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18
build-ros-humble --custom \
  --wd /mnt/ohos/tmp/ibrobot_oh_ws \
  --custom-prefix /data/ibrobot/install \
  --colcon-args --packages-select usb_cam
'
```

如果 Docker 产物属主变成 root，可以修复一下：

```bash
docker run --rm \
  -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
  voxelsky/ohos-ros-humble-builder:v0.1.5 \
  sh -c "chown -R $(id -u):$(id -g) /mnt/ohos/ibrobot_oh_ws/install/usb_cam || true"
```

验证产物类型：

```bash
file "$OH_CUSTOM_WS/install/usb_cam/lib/usb_cam/usb_cam_node_exe"
file "$OH_CUSTOM_WS/install/usb_cam/lib/libusb_cam.so"
file "$OH_CUSTOM_WS/install/usb_cam/lib/libusb_cam_node.so"
```

### 7.3 部署到板端

推荐部署完整 `install/usb_cam` 目录，而不是只发二进制。这样 `share/ament_index`、库和可执行文件会保持一致：

```bash
HDC_BIN=hdc
HDC_TARGET=<board-ip>:8710

"$HDC_BIN" -t "$HDC_TARGET" shell 'mkdir -p /data/ibrobot/install'
"$HDC_BIN" -t "$HDC_TARGET" file send \
  "$OH_CUSTOM_WS/install/usb_cam" \
  /data/ibrobot/install/usb_cam
```

如果只想增量更新关键文件，至少需要发：

```bash
"$HDC_BIN" -t "$HDC_TARGET" file send \
  "$OH_CUSTOM_WS/install/usb_cam/lib/usb_cam/usb_cam_node_exe" \
  /data/ibrobot/install/usb_cam/lib/usb_cam/usb_cam_node_exe

"$HDC_BIN" -t "$HDC_TARGET" file send \
  "$OH_CUSTOM_WS/install/usb_cam/lib/libusb_cam.so" \
  /data/ibrobot/install/usb_cam/lib/libusb_cam.so

"$HDC_BIN" -t "$HDC_TARGET" file send \
  "$OH_CUSTOM_WS/install/usb_cam/lib/libusb_cam_node.so" \
  /data/ibrobot/install/usb_cam/lib/libusb_cam_node.so
```

部署后检查 ROS package 解析到的路径：

```sh
cd /data
. ./ros2ohos.env
. /data/ibrobot/install/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 pkg prefix usb_cam
# 期望：/data/ibrobot/install/usb_cam
```

如果输出仍然是 `/data/install`，说明预装包抢在 overlay 前面。需要检查 `AMENT_PREFIX_PATH` 顺序，
或临时屏蔽 `/data/install` 下的预装 `usb_cam` ament index 条目。

### 7.4 ffmpeg / `libswscale` 依赖

`usb_cam` 在 `mjpeg2rgb`、`yuyv` 等路径下会用到 ffmpeg 的图像转换库。板端常见库位置是
`/data/out/lib`：

```sh
ls -l /data/out/lib/libswscale.so* \
      /data/out/lib/libavcodec.so* \
      /data/out/lib/libavutil.so*
```

如果 `usb_cam` 启动时报类似下面的错误：

```text
Error loading shared library libswscale.so.5: No such file or directory
Error loading shared library libavcodec.so.58: No such file or directory
Error loading shared library libavutil.so.56: No such file or directory
```

可以在 `usb_cam` 自己的 lib 目录下建立符号链接：

```sh
mkdir -p /data/ibrobot/install/usb_cam/lib
ln -sf /data/out/lib/libswscale.so.5 /data/ibrobot/install/usb_cam/lib/libswscale.so.5
ln -sf /data/out/lib/libavcodec.so.58 /data/ibrobot/install/usb_cam/lib/libavcodec.so.58
ln -sf /data/out/lib/libavutil.so.56 /data/ibrobot/install/usb_cam/lib/libavutil.so.56
```

启动时看到下面的日志不是致命错误，只是说明没有硬件加速的 colorspace conversion：

```text
[swscaler @ ...] No accelerated colorspace conversion found from yuv422p to rgb24.
```

### 7.5 `LD_PRELOAD=libpython` 与 C++ 相机节点

运行 RKNN / LeRobot 推理时，Python 推理节点需要：

```sh
export PYTHONHOME=/data/local/skh-run/usr
export LD_PRELOAD=${PYTHONHOME}/lib/libpython3.12.so.1.0:${PYTHONHOME}/lib/libomp.so
```

但 `usb_cam_node_exe` 是 C++ 进程，不需要也不应该预加载 `libpython3.12.so.1.0`。在当前板端验证中，
把 `libpython` 注入 `usb_cam` 进程可能导致 ffmpeg / `libswscale` 路径出现异常终止或动态链接行为异常。

推荐做法不是写 shell wrapper，而是在 IB_Robot 的 launch builder 中给 `usb_cam` 节点单独设置
`additional_env`：

- 保留 `libomp.so`
- 去掉 `libpython3.12.so.1.0`
- 显式补齐 `/data/install/lib`、`/data/out/lib`、`/data/ibrobot/install/usb_cam/lib`
- 显式设置 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`

当前仓库的 `src/robot_config/robot_config/launch_builders/perception.py` 已包含这个处理。这样完整
`ros2 launch` 时，Python 推理节点仍然使用 `libpython`，但 C++ 相机节点不会被 `libpython` 干扰。

### 7.6 推荐相机配置：`mjpeg2rgb`

在 `src/robot_config/config/robots/so101_single_arm.yaml` 中，BQ3588HM 当前验证过的配置是：

```yaml
peripherals:
  - type: camera
    name: top
    driver: opencv
    index: 20
    width: 640
    height: 480
    fps: 30
    pixel_format: mjpeg2rgb
    frame_id: camera_top_frame

  - type: camera
    name: wrist
    driver: opencv
    index: 22
    width: 640
    height: 480
    fps: 60
    pixel_format: mjpeg2rgb
    frame_id: camera_wrist_frame
```

`mjpeg2rgb` 的优点是发布到 ROS 的 `sensor_msgs/Image.encoding` 为 `rgb8`，Ubuntu 侧 RViz 和常见
图像工具都能直接显示。`yuyv` 虽然也能采集，但可能发布为 `yuv422_yuy2`，模型侧可以解码，RViz
却可能显示 `No image` 或无法解码。

### 7.7 手动验证 `usb_cam`

单独验证 `usb_cam` 时，建议先不要带 `libpython`：

```sh
cd /data
. ./ros2ohos.env
. /data/ibrobot/install/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=55
export PYTHONHOME=/data/local/skh-run/usr
export PATH=${PYTHONHOME}/bin:$PATH
export LD_PRELOAD=${PYTHONHOME}/lib/libomp.so
export LD_LIBRARY_PATH=/data/ibrobot/install/usb_cam/lib:${PYTHONHOME}/lib:/sys_prod/robot/out/lib:/data/install/lib:/data/out/lib:/vendor/lib64

/data/ibrobot/install/usb_cam/lib/usb_cam/usb_cam_node_exe \
  --ros-args \
  -p camera_name:=top \
  -p video_device:=/dev/video20 \
  -p image_width:=640 \
  -p image_height:=480 \
  -p pixel_format:=mjpeg2rgb \
  -p framerate:=30.0 \
  -r __node:=top_camera \
  -r image_raw:=/camera/top/image_raw
```

正常日志应包含：

```text
Starting 'top' (/dev/video20) at 640x480 via mmap (mjpeg2rgb) at 30 FPS
Timer triggering ...
```

`Cannot open device: /dev/videoX` 在启动时可能出现很多行，这是 `usb_cam` 枚举其他 V4L2 设备的日志。
只要最后目标设备 `/dev/video20` / `/dev/video22` `Starting` 并 `Timer triggering`，即可忽略。

### 7.8 完整 IB_Robot launch 中的环境变量

完整 RKNN 推理 launch 仍然需要 Python / torch 运行时：

```sh
cd /data && . ./ros2ohos.env && . /data/ibrobot/install/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID=55
export PYTHONHOME=/data/local/skh-run/usr
export PATH=${PYTHONHOME}/bin:$PATH
export PYTHONPATH=/data/ibrobot/install/lerobot/src:/data/ibrobot/install/inference_service/lib/python3.12/site-packages:/data/ibrobot/install/tensormsg/lib/python3.12/site-packages:/data/ibrobot/install/robot_config/lib/python3.12/site-packages:/data/ibrobot/install/ibrobot_msgs/lib/python3.12/site-packages:${PYTHONHOME}/lib/python3.12/site-packages:/sys_prod/robot/out/lib/python3.12/site-packages:/data/install/lib/python3.12/site-packages
export LD_LIBRARY_PATH=/data/ibrobot/install/ibrobot_msgs/lib:/data/ibrobot/install/inference_service/lib:/data/ibrobot/install/robot_config/lib:/data/ibrobot/install/so101_hardware/lib:/data/ibrobot/install/usb_cam/lib:${PYTHONHOME}/lib:${PYTHONHOME}/lib/python3.12/site-packages/torch/lib:${PYTHONHOME}/lib/python3.12/site-packages/torchaudio/lib:/sys_prod/robot/out/lib:/data/install/lib:/data/out/lib:/vendor/lib64
export LD_PRELOAD=${PYTHONHOME}/lib/libpython3.12.so.1.0:${PYTHONHOME}/lib/libomp.so

ros2 launch robot_config robot.launch.py \
  robot_config:=so101_single_arm \
  use_sim:=false \
  control_mode:=model_inference \
  device:=rknn
```

当前验证过的正常日志包括：

```text
[top_camera]: Starting 'top' (/dev/video20) at 640x480 via mmap (mjpeg2rgb) at 30 FPS
[wrist_camera]: Starting 'wrist' (/dev/video22) at 640x480 via mmap (mjpeg2rgb) at 60 FPS
[top_camera]: Timer triggering ...
[wrist_camera]: Timer triggering ...
[act_inference_node]: DispatchInfer Action Server ready
[act_inference_node]: First inference complete
[action_dispatcher]: First inference received
```

### 7.9 Ubuntu RViz 看不到图像时

如果板端日志已经 `Timer triggering`，且模型推理也能收到图像，但 Ubuntu 侧 RViz 显示 `No image`，按下面顺序排查：

1. Ubuntu 和板端都设置同一个 `ROS_DOMAIN_ID`。
2. Ubuntu 和板端都使用 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`。
3. RViz 的 Image Display 选择 `/camera/top/image_raw` 或 `/camera/wrist/image_raw`。
4. RViz 的 Image Display 中将 `Reliability Policy` 改为 `Best Effort`。
5. `Fixed Frame` 可以先临时设置为 `camera_top_frame` 或 `camera_wrist_frame`，避免 TF 问题影响显示判断。
6. 确认 YAML 使用 `pixel_format: mjpeg2rgb`，这样发布编码为 `rgb8`，RViz 更容易直接显示。

如果需要确认板端确实发布了有效图片，可以在板端写一个小 subscriber 抓取 `/camera/*/image_raw` 到
PPM 文件。当前验证中，`mjpeg2rgb` 路径抓到的图像为 `640x480 rgb8`，能直接在 Ubuntu 打开。

### 7.10 常见问题速查

| 现象 | 可能原因 | 处理方式 |
| --- | --- | --- |
| `usb_cam_node_exe` 报 `librclcpp.so` / `libclass_loader.so` 找不到 | `LD_LIBRARY_PATH` 没包含 `/data/install/lib` 或 launch 子进程环境丢失 | 在 launch builder 的 `additional_env` 中显式补齐，手动测试时补 `LD_LIBRARY_PATH` |
| `rmw_fastrtps` `cast_or_create_topic` assertion | 默认 RMW 走了 Fast DDS | 设置 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| `libswscale.so.5` / `libavcodec.so.58` / `libavutil.so.56` 找不到 | ffmpeg 库不在动态链接器搜索路径 | 把 `/data/out/lib` 加到 `LD_LIBRARY_PATH`，或在 `usb_cam/lib` 下建符号链接 |
| `swscaler ... No accelerated colorspace conversion` | ffmpeg 走 CPU colorspace conversion | 可忽略，只是性能提示 |
| RViz `No image`，但模型能推理 | `yuyv` 发布 `yuv422_yuy2`，或 RViz QoS 不匹配 | 使用 `mjpeg2rgb`，RViz Image Display 改 `Best Effort` |
| 完整 RKNN launch 下相机崩，单独相机正常 | Python 推理环境的 `LD_PRELOAD=libpython` 影响 C++ 相机进程 | 给 `usb_cam` 节点单独设置 env，去掉 `libpython`，保留 `libomp` |

## 8. `ros2_control` / `controller_manager` 的 OpenHarmony 注意事项

SO-101 真机启动依赖 `ros2_control_node`、`controller_manager` 和 `controller_manager spawner`。
在 BQ3588HM OpenHarmony musl 环境中，这部分不能只依赖 Ubuntu 侧的 `ros2_control` 经验；需要使用
OpenHarmony 交叉编译产物，并记录实际源码来源和补丁。

### 8.1 当前验证过的源码来源

当前验证过的板端 `ros2_control` 基线来自 OpenHarmony 机器人仓库：

```text
https://gitcode.com/openharmony-robot/ros_ros2_control
commit c742704c6132ab81a0a34fef56f6422555e07e38
tag OpenHarmony-Embodied-v1.0.1-Release
```

该仓库把 `ros2_control`、`ros2_controllers`、`control_msgs`、`control_toolbox`、`realtime_tools` 等相关包放在
同一个仓库中。其中本次 SO-101 真机验证实际用到的核心包包括：

```text
controller_manager_msgs
hardware_interface
realtime_tools
controller_interface
controller_manager
ros2_control
```

版本确认：`controller_manager/package.xml` 和 `ros2_control/package.xml` 均为 `2.53.0`，
`realtime_tools/package.xml` 为 `2.15.0`。

### 8.2 OpenHarmony 基线中已有的关键适配

OpenHarmony fork 已经包含 `realtime_tools` 的平台适配。例如
`realtime_tools/src/realtime_helpers.cpp` 中对 `__OHOS__` 做了特殊处理：

```cpp
#elifdef __OHOS__
  return {false, "Memory locking is not supported on OpenHarmony OS."};
```

以及：

```cpp
#elifdef __OHOS__
  message = "Thread affinity is not supported on OpenHarmony OS.";
  return std::make_pair(false, message);
```

这避免在 OpenHarmony/musl 上继续走 Linux glibc 环境下的 `mlockall`、capability 和 thread affinity
路径。当前本地交叉编译工作区中的 `realtime_tools` 与该 OpenHarmony fork 一致，没有额外本地差异。

### 8.3 本地验证时额外确认的 `controller_manager` 补丁

仅使用上面的 OpenHarmony fork 还不够。本地交叉编译工作区：

```text
/home/xqw/Research/bq3588_oh_ws/custom_build_root/ibrobot_oh_ws/src/controller_manager_fix/
```

相对 `c742704` 还有两个已确认差异。

第一个是 `controller_manager/src/controller_manager.cpp` 中 `switch_controller()` 等待实时循环完成切换时，
必须先持有 `switch_params_.mutex` 再调用 `condition_variable::wait_for()`：

```diff
-  std::unique_lock<std::mutex> switch_params_guard(switch_params_.mutex, std::defer_lock);
+  std::unique_lock<std::mutex> switch_params_guard(switch_params_.mutex);
   if (!switch_params_.cv.wait_for(
         switch_params_guard, switch_params_.timeout, [this] { return !switch_params_.do_switch; }))
```

这才是本次验证中与 `SwitchController`/spawner 稳定性直接相关的修复点。使用 `std::defer_lock` 会把未加锁的
`unique_lock` 传给 `wait_for()`，在 OpenHarmony/musl 上会触发不稳定行为。

第二个是 `controller_manager/CMakeLists.txt` 中关闭测试构建：

```diff
-if(BUILD_TESTING)
+if(FALSE AND BUILD_TESTING)
```

这个改动只用于减少交叉编译测试依赖，不是运行时逻辑修复。

### 8.4 不再使用手动 spawner workaround

早期验证时，板端 `controller_manager spawner` / `SwitchController` 路径曾经触发过
`condition_variable::timed_wait` 相关崩溃。临时绕过方案是写 Python 脚本手动调用：

1. `/controller_manager/load_controller`
2. `/controller_manager/configure_controller`
3. `/controller_manager/switch_controller`

并把 `SwitchController.timeout` 设为 `0.0`。这个临时脚本曾命名为：

```text
scripts/oh_manual_spawn_controllers.py
```

它只是调试 workaround，不是当前推荐流程，也不需要纳入正式提交。当前推荐流程是部署修复后的
`controller_manager` 产物，让 IB_Robot launch 继续使用标准 `controller_manager` spawner：

```python
Node(
    package="controller_manager",
    executable="spawner",
    arguments=[
        "joint_state_broadcaster",
        "arm_position_controller",
        "gripper_position_controller",
        "--controller-manager",
        "controller_manager",
        "--activate-as-group",
    ],
)
```

也就是说，当前 `src/robot_config/robot_config/launch_builders/control.py` 不需要调用
`oh_manual_spawn_controllers.py`。

### 8.5 需要交叉编译哪些包

如果你的板端 ROS 运行时里 `controller_manager` 仍然会崩，或者换了新的 OpenHarmony ROS 基础包，需要重新确认
`ros2_control` 相关包是否已经包含上面的 `controller_manager` 修复。通常至少涉及这些包：

```text
controller_manager_msgs
hardware_interface
realtime_tools
controller_interface
controller_manager
ros2_control
```

具体集合取决于你的 OpenHarmony ROS 发行版和工作区依赖。构建方式仍然使用官方 Docker builder：

```bash
docker run --rm -i \
  -e WS_ROOT=/mnt/ohos/tmp \
  -e OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18 \
  --name ibrobot-oh-build \
  -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
  -v "${OH_CUSTOM_ROOT}:/mnt/ohos/tmp" \
  voxelsky/ohos-ros-humble-builder:v0.1.5 \
  bash -lc '
set -euo pipefail
export OHOS_CPU=aarch64
export OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18
build-ros-humble --custom \
  --wd /mnt/ohos/tmp/ibrobot_oh_ws \
  --custom-prefix /data/ibrobot/install \
  --colcon-args --packages-select \
    controller_manager_msgs \
    hardware_interface \
    realtime_tools \
    controller_interface \
    controller_manager \
    ros2_control
'
```

> 注意：上面的命令是说明“需要修复 ros2_control 时的交叉编译形态”。如果你使用的官方运行时已经
> 包含修复，且标准 spawner 已能正常加载控制器，就不需要重复编译这些包。反之，需要确认源码中
> `controller_manager/src/controller_manager.cpp` 已经采用加锁构造的 `switch_params_guard`。

### 8.6 如何确认当前状态是正常的

完整 launch 中应看到类似日志：

```text
[ros2_control_node]: Successful initialization of hardware 'RobotSystem'
[ros2_control_node]: Successful 'activate' of hardware 'RobotSystem'
[spawner_joint_state_broadcaster_group]: Loaded joint_state_broadcaster
[spawner_joint_state_broadcaster_group]: Loaded arm_position_controller
[spawner_joint_state_broadcaster_group]: Loaded gripper_position_controller
[spawner_joint_state_broadcaster_group]: Configured and activated all the parsed controllers list
[wait_for_active_controllers]: Controllers are active: joint_state_broadcaster, arm_position_controller, gripper_position_controller
```

如果出现下面情况，说明可能退回到了未修复版本或 RMW/环境配置不对：

| 现象 | 可能原因 | 处理方式 |
| --- | --- | --- |
| spawner 崩溃或卡在 switch controller | `SwitchController` 等待路径仍是未加锁 `std::defer_lock` 版本 | 确认部署的是修复后的 `controller_manager` / `ros2_control` |
| controller_manager service 找不到 | `ros2_control_node` 未启动或 DDS 不通 | 设置 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`，检查 `/controller_manager/list_controllers` |
| 控制器一直 inactive | spawner 没完成 activate | 看 spawner 日志，确认 controllers YAML 名称和 `so101_controllers.yaml` 一致 |
| 需要手动脚本才可以激活 | 仍在使用 workaround 状态 | 回到交叉编译并部署修复后的 ros2_control 产物 |

### 8.7 文档化原则

如果后续再次修改 `ros2_control` 或 `controller_manager` 源码，请把 patch 来源、修改点和部署路径记录下来。
不要只留下板端临时文件，否则很难判断当前系统是否依赖某个未提交的 workaround。

建议记录至少包含：

- 上游仓库和 commit / tag
- 修改文件和核心 diff
- 交叉编译命令
- 部署到板端的 package 路径
- 标准 spawner 是否能直接工作

## 9. 板端如何使用交叉编译出来的 IB_Robot 包

先加载官方 ROS for OpenHarmony 运行时：

```sh
cd /data
. ./ros2ohos.env
```

然后加载你自己的工作区：

```sh
cd /data/ibrobot
. install/setup.sh
```

之后就可以执行你部署进去的 ROS 2 包，例如：

```sh
ros2 pkg list | grep -E 'ibrobot_msgs|tensormsg|robot_config|inference_service'
```

如果你还需要在板端跑 `LeRobot + torch` 的 Cloud 推理链路，请继续看：

- [OpenHarmony_thirdparty_pytorch_validation.md](./OpenHarmony_thirdparty_pytorch_validation.md)

## 10. 和官方文档的关系

如果你只关心“下载二进制后怎么在板上用”，官方 `usage.md` 已经足够。

如果你只关心“如何手工用 Docker 交叉编译任意 ROS 项目”，官方 `docker-build.md` 的
`用户自定义 ROS2 包/项目编译和使用` 已经给出了通用流程。

而本文档额外补上的内容是：

1. **把 BQ3588HM 板端 ROS 安装和 IB_Robot 交叉编译放到一条连续流程里。**
2. **把 `build_ibrobot_oh_custom.sh` 的变量和实际下载内容一一对应起来。**
3. **补充 `usb_cam` 这类第三方 ROS 包的交叉编译、ffmpeg 依赖和板端运行注意事项。**
4. **记录 `ros2_control` / `controller_manager` 在 OpenHarmony musl 上的历史问题和正式修复原则。**
5. **明确建议把 SDK / sysdeps / runtime 放到仓库外的独立目录，而不是 IB_Robot 的 `tmp/`。**
6. **说明编译结果如何落到 `/data/ibrobot/install`，以及板端如何叠加加载。**
