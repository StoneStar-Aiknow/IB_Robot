# RoboFrame for OpenHarmony

> 基于 OpenHarmony 的端侧具身智能机器人框架，可支持在 OpenHarmony 上完成 CPU / NPU 推理、数据采集、机器人控制等功能。

RoboFrame 融合视觉-动作策略模型（ACT、Diffusion Policy 等 VLA 模型）与 ROS 2 机器人控制链路，在 OpenHarmony 上提供端到端的具身智能能力——涵盖 **CPU / NPU 模型推理**、**多模态数据采集**与**机械臂运动控制**。本 README 是 RoboFrame 在 OpenHarmony 上的整体介绍与上手指南，覆盖从源码获取、镜像烧录、交叉编译、第三方 ROS 包移植、内核驱动编译、板端运行时环境，到 NPU 推理与单板控制闭环的完整流程。

## 效果展示：OpenClaw 社交控制

RoboFrame 支持通过 OpenClaw AI Agent 进行远程社交控制，无论是在仿真环境还是真实 SO-101 机械臂上，都可以用自然语言下达指令：

|                            仿真演示 (Simulation)                            |                             真实硬件 (Real Robot)                            |
| :---------------------------------------------------------------------: | :----------------------------------------------------------------------: |
| ![仿真演示](docs/pictures/openclaw_sim.gif) | ![真实硬件](docs/pictures/openclaw_real.gif) |

## 目录

- [整体定位](#整体定位)
- [支持的硬件板卡](#支持的硬件板卡)
- [系统架构](#系统架构)
- [一、获取 OpenHarmony EmbodiedAI 源码与镜像](#一获取-openharmony-embodiedai-源码与镜像)
- [二、板端调试连接：HDC 与 SSH](#二板端调试连接hdc-与-ssh)
- [三、板端安装 OpenHarmony ROS 2 Humble 运行时](#三板端安装-openharmony-ros-2-humble-运行时)
- [四、交叉编译部署 RoboFrame 自有包](#四交叉编译部署-roboframe-自有包)
- [五、交叉编译部署第三方 ROS 2 包](#五交叉编译部署第三方-ros-2-包)
- [六、内核驱动编译（USB ACM / 手柄）](#六内核驱动编译usb-acm--手柄)
- [七、板端多运行时叠加环境](#七板端多运行时叠加环境)
- [八、RKNN NPU 推理部署](#八rknn-npu-推理部署)
- [九、单板全链路闭环与启动验证](#九单板全链路闭环与启动验证)
- [十、常见问题（FAQ）](#十常见问题faq)
- [十一、相关文档与生态](#十一相关文档与生态)

## 整体定位

RoboFrame 在 OpenHarmony 上提供**模型推理**、**数据采集**与**机器人控制**三类核心能力，并通过「主机交叉编译 + 板端运行」的系统架构落地。整体可分为三层：

| 层 | 角色 | 说明 |
| --- | --- | --- |
| **主机侧（Ubuntu x86_64）** | 交叉编译 + 模型转换 | 使用 Docker 交叉编译工具链把 RoboFrame 自有包和第三方 ROS 2 包编译为 aarch64/musl 产物；在主机上把 ONNX 转为 RKNN |
| **板端运行时（RK3588 OpenHarmony）** | ROS 2 + Torch + NPU | 叠加 OpenHarmony ROS 2 Humble 运行时、`thirdparty_pytorch` runtime 与 RKNN NPU 驱动 |
| **机械臂与外设** | 执行端 | SO-101 机械臂（ Feetech 舵机）、USB 摄像头、（可选）游戏手柄遥操作 |

> **核心约束**：OpenHarmony 使用 musl libc + aarch64，所有 C++ 节点和 Python 扩展模块都必须用 OpenHarmony 交叉工具链编译，**不能**直接复用 Ubuntu x86_64 或 glibc 构建产物。

## 支持的硬件板卡

本框架当前以 Rockchip RK3588（6 TOPS NPU）系列板卡为目标，已验证支持以下三款（均来自 OpenHarmony EmbodiedAI 1.0.1 Release）：

| 板卡 | 芯片 | NPU | 内存/存储 | 典型用途 |
| --- | --- | --- | --- | --- |
| **贝启 BQ3588HM** | RK3588 | 6 TOPS | 8GB LPDDR5 + 64GB eMMC | 具身智能机器人主控、端侧推理 |
| **曦胧 RoboPi** | RK3588 | 6 TOPS | 8GB + 64GB eMMC | 4×千兆网口（2 路 EtherCAT），工业级机器人控制 |
| **贝启 Robo3588** | RK3588 + RK1828 | 6 TOPS | 8GB + 64GB eMMC | 6×CAN、4×GMSL2 车载相机，人形 / AMR |

三款板卡的详细规格、固件下载与烧录方式见 [OpenHarmony EmbodiedAI 1.0.1 Release](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md)。

## 系统架构

```text
┌──────────────────────────────────────────────────────────────────┐
│  Host (Ubuntu 22.04 x86_64)                                      │
│                                                                  │
│  Docker: voxelsky/ohos-ros-humble-builder:v0.1.5                │
│    │  OHOS SDK (aarch64) + sysroot                              │
│    │  build-ros-humble --custom                                 │
│    ▼                                                             │
│  OH_ROOT/custom_build_root/roboframe_oh_ws/install/ ◄── aarch64/musl 产物 │
│                                                                  │
│  ONNX ── rknn-toolkit2 ──► *.rknn  (float16, NPU)               │
└──────────────────────────────────┬───────────────────────────────┘
                                   │ hdc file send / ssh scp
                                   ▼
┌──────────────────────────────────────────────────────────────────┐
│  Board (RK3588, OpenHarmony 5.1, musl)                           │
│                                                                  │
│  /data/roboframe/install/   ◄── RoboFrame 自有包 + 第三方 ROS 包  │
│  /data/install + /data/out  ◄── OpenHarmony ROS 2 Humble 运行时   │
│  /data/local/skh-run/usr    ◄── thirdparty_pytorch runtime       │
│  /vendor/lib64/librknnrt.so ◄── NPU 驱动                         │
│                                                                  │
│  usb_cam → 推理节点(RKNN) → action_dispatcher → ros2_control     │
└──────────────────────────────────────────────────────────────────┘
```

## 一、获取 OpenHarmony EmbodiedAI 源码与镜像

OpenHarmony EmbodiedAI 的源码获取、镜像编译与烧录流程（含 repo 工具准备、各板卡编译命令、瑞芯微烧录步骤及预编译固件下载）已统一维护在社区讨论中，请直接参考：

👉 [获取 OpenHarmony EmbodiedAI 源码与镜像](https://gitcode.com/org/openharmony-robot/discussions/4)

完成镜像烧录后，板卡即可进入后续的调试连接与交叉编译部署流程。

## 二、板端调试连接：HDC 与 SSH

### 2.1 HDC 工具准备

HDC（Hardware Device Connector）是与 OpenHarmony 设备交互的核心工具，来自 OpenHarmony SDK 的 `toolchains` 目录。

推荐把 SDK `toolchains` 目录加入 `PATH` 持久化：

```bash
# Bash
echo 'export PATH=<sdk-root>/toolchains:$PATH' >> ~/.bashrc && source ~/.bashrc
# Zsh
echo 'export PATH=<sdk-root>/toolchains:$PATH' >> ~/.zshrc && source ~/.zshrc

hdc list targets   # 验证
```

### 2.2 切换到 TCP 网络调试（推荐）

USB 连接下大文件传输容易产生僵尸会话，建议切换到局域网 TCP 模式：

```bash
hdc tmode port 8710                 # 板端开启监听
hdc shell ifconfig                  # 获取开发板局域网 IP
hdc tconn <board-ip>:8710           # 主机连接
hdc -t <board-ip>:8710 shell        # 明确指定目标设备
```

之后所有文件传输 / 命令执行统一使用 `hdc -t <board-ip>:8710 ...`。

### 2.3 开启 SSH（可选但推荐）

HDC shell 的 PTY 缓冲区有限，长时间 launch 日志会被截断。建议开启板端 SSH：

```bash
# 生成 host key 并启动 sshd（配置文件随 OH ROS runtime 提供）
hdc -t <board-ip>:8710 shell '/sys_prod/robot/out/bin/ssh-keygen -A'
hdc -t <board-ip>:8710 shell '/sys_prod/robot/out/sbin/sshd -f /sys_prod/robot/out/etc/sshd_config -E /data/sshd.log'
ssh root@<board-ip>
```

> **前置条件**：
> - `ros2ohos.env` 来自第三节安装的 OpenHarmony ROS 2 运行时，需先完成第三节部署，板端 `/data/ros2ohos.env` 才会存在。
> - `ros2ohos.env` 会 source 同目录的 `/data/sysdeps.env`。其中 `mount -o remount,rw /` 位于 `# setup sshd` 段落内、`if [ -f "${OHOS_ROS2_SYSDEPS}/etc/sshd_config" ]; then` 这一行的**紧随其后第一条命令**：
>
>   ```sh
>   # setup sshd
>   if [ -f "${OHOS_ROS2_SYSDEPS}/etc/sshd_config" ]; then
>       mount -o remount,rw /          # ◄── remount 在这里
>       mkdir -p /var/empty /var/run /root/.ssh /libexec
>       chmod 0555 /var/empty
>       # ...（首次还会 cp sshd 配置、改 /etc/passwd 允许 root 登录）
>       [ ! -f "/etc/ssh_host_rsa_key" ] && ssh-keygen -A || true   # 自动生成 host key
>   fi
>   ```
>
> - 因此**只需 source 过一次 `ros2ohos.env`**，remount、sshd 目录与 host key 都会自动就绪，随后即可启动 sshd。若你的镜像版本里 `sysdeps.env` 缺少这一行，按上面的位置（`# setup sshd` 的 `if` 之后、`mkdir` 之前）补上 `mount -o remount,rw /` 即可。

推荐使用习惯：**大文件传输 / 自动化脚本用 HDC/TCP，日常命令行 / 多终端调试用 SSH**。

## 三、板端安装 OpenHarmony ROS 2 Humble 运行时

### 3.1 下载运行时二进制

从 [usage.md](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md) 下载以下两类包（选择日期最新且架构为 `aarch64`）：

| 类别 | 文件 | 用途 |
| --- | --- | --- |
| ROS 系统依赖 | `ohos-*-sysdeps-*.tar.gz` | OpenHarmony ROS 系统级依赖库 |
| ROS 2 Humble 发行版 | `ohos-humble-build-*.tar.gz` | 板端 ROS 2 Humble 运行时 |

### 3.2 上传并解压

```bash
HDC_BIN=<path-to-hdc>
HDC_TARGET=<board-ip>:8710

"$HDC_BIN" -t "$HDC_TARGET" file send \
  ohos-humble-build-aarch64-*.tar.gz  /data/ohos-humble-build-aarch64.tar.gz
"$HDC_BIN" -t "$HDC_TARGET" file send \
  ohos-*-sysdeps-aarch64-*.tar.gz    /data/ohos-sysdeps-aarch64.tar.gz

"$HDC_BIN" -t "$HDC_TARGET" shell '
cd /data
tar -zxpvf ohos-humble-build-aarch64.tar.gz
tar -zxpvf ohos-sysdeps-aarch64.tar.gz
'
```

### 3.3 加载 ROS 环境并验证

```bash
hdc -t <board-ip>:8710 shell '
cd /data
. ./ros2ohos.env         # 必须在 ros2ohos.env 所在目录执行
ros2 topic list
python3 --version
'
```

BQ3588HM 上的关键路径约定：

- `ros2ohos.env` 位于 `/data/ros2ohos.env`
- ROS 运行时目录 `/data/install`
- sysdeps 目录 `/data/out`

## 四、交叉编译部署 RoboFrame 自有包

这是把 RoboFrame 框架部署到板端的核心步骤。RoboFrame 源码包按功能划分如下，均可纳入交叉编译工作区，再通过 `--packages` 按需选择：

```text
控制与规划
  robot_config          # 配置驱动中心、统一启动入口
  action_dispatch       # 统一动作执行器（Action Chunking 调度 / MoveIt 轨迹执行）
  robot_moveit          # MoveIt 2 运动规划集成
  robot_navigation      # 导航功能包
  omni_wheel_controller # 全向轮控制器插件
  task_dispatch         # 任务调度与分发

感知与交互
  perception_service    # 感知服务
  robot_teleop          # 遥操作控制（Leader Arm / Xbox 手柄）
  voice_asr_service     # 语音识别服务

推理与决策
  inference_service     # 多模型推理与部署服务
  vlm_task_planner      # VLM 任务规划
  embodied_agent        # 具身智能 Agent
  skill_library         # 技能库

数据与协议
  roboframe_msgs        # 统一消息/服务定义
  tensormsg             # ROS msg ↔ tensor 协议转换枢纽
  dataset_tools         # 数据集采集与转换工具

硬件接口
  so101_hardware        # SO-101 电机驱动接口
  lekiwi_hardware       # Lekiwi 底盘硬件驱动
  hardware_mock         # 硬件模拟（Mock）接口

模型与描述
  robot_description     # 统一 URDF/SRDF/MJCF 模型描述
  lekiwi_description    # Lekiwi 底盘 URDF/Mesh
  model_utils           # 模型工具库（ONNX/RKNN 导出等）

主机侧（一般不交叉编译到板端）
  sim_models            # 仿真场景模型（Gazebo/MuJoCo）
  attention_viz         # 注意力可视化
  safety_guard          # 安全守护
  embodied_common       # 公共基础库
  pymoveit2             # [子模块] MoveIt2 Python 接口
  rosclaw               # [子模块] 社交控制集成
```

> 最小推理/控制工作区默认交叉编译：`roboframe_msgs tensormsg robot_config inference_service dataset_tools`。若需要在板端使用 MoveIt 规划、导航或其它能力，把对应包名追加到 `scripts/openharmony/build_roboframe_oh_custom.sh` 的 `--packages` 列表即可。

### 4.1 前提条件

主机侧准备（Docker 环境搭建、交叉编译镜像获取与三类官方包下载的完整步骤，参见官方交叉编译文档 [docker-build.md](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/docker-build.md) 与二进制使用文档 [usage.md](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md)）：

- Ubuntu 22.04 主机，已安装 Docker
- Docker 镜像：`docker pull voxelsky/ohos-ros-humble-builder:v0.1.5`
- 已下载三类官方包：
  - `ohos-sdk-18-linux-aarch64-*.tar.gz`（OHOS SDK）
  - `ohos-*-sysdeps-*.tar.gz`（系统依赖）
  - `ohos-humble-build-aarch64-*.tar.gz`（ROS 2 运行时）

### 4.2 主机目录布局（OH_ROOT）

**推荐**用一个统一的外部根目录组织所有下载内容和交叉编译产物，通过 `OH_ROOT` 传给构建脚本，避免散落临时目录：

```text
<OH_ROOT>/
├── downloads/
│   ├── images/      oh5.1-bq3588-build-...
│   ├── sdk/         ohos-sdk-18-linux-aarch64-*.tar.gz
│   ├── sysdeps/     ohos-*-sysdeps-aarch64-*.tar.gz
│   └── runtime/     ohos-humble-build-aarch64-*.tar.gz
└── custom_build_root/
    ├── install/                 # OH ROS 2 Humble 运行时 (sysroot)
    ├── roboframe_oh_ws/
    │   ├── src/                 # 交叉编译工作区源码
    │   └── install/             # ◄── 交叉编译产物落点
    ├── ohos-robot-toolchain/
    │   └── 18/native/           # 解压后的 OHOS SDK + sysroot
    ├── ros_ros2_base/           # 脚本自动克隆
    └── version/                 # 脚本自动克隆
```

```bash
export OH_ROOT="<your-unified-oh-root>"
export OH_DOWNLOAD_ROOT="$OH_ROOT/downloads"
```

### 4.3 运行 RoboFrame 交叉构建脚本

RoboFrame 提供了封装好的交叉编译入口脚本，自动处理 SDK 解压、sysdeps overlay、`ros_ros2_base` 克隆、以及把待编译包复制到工作区：

```bash
export OH_ROOT="<your-unified-oh-root>"

./scripts/openharmony/build_roboframe_oh_custom.sh \
  --oh-root "$OH_ROOT" \
  --image voxelsky/ohos-ros-humble-builder:v0.1.5 \
  --packages roboframe_msgs,tensormsg,robot_config,inference_service
```

只要 `OH_ROOT` 目录布局符合 4.2 约定，脚本会自动从 `downloads/{sdk,sysdeps,runtime}/` 解析三类 tarball。文件不在默认布局时，可用 `--sdk-tar` / `--sysdeps-tar` / `--humble-tar` 精确覆盖。

> **LeRobot 运行时 patch**：打包板端 runtime 时，脚本会显式应用 `series.openharmony-5.1.0-musl.txt` patch 栈，确保 lazy-import 等 OpenHarmony 专用补丁真正进入部署产物（板端不安装完整训练依赖 `huggingface_hub`/`diffusers`/`transformers` 等）。**严禁手工复制 `lerobot/src` 进 install 树**，必须走此脚本。

#### 脚本内部做了什么

1. 把 `ohos-humble-build-*.tar.gz` 解压到 `custom_build_root/`，得到 `install/`
2. 把 `ohos-sdk-18-linux-aarch64-*.tar.gz` 解压到 `ohos-robot-toolchain/18/native`
3. 把 sysdeps 里的 Python 3.12 / sframe 内容 overlay 到 SDK sysroot
4. 自动克隆 `ros_ros2_base` 与 `version` 仓库
5. 把 RoboFrame 待编译包复制到 `roboframe_oh_ws/src`
6. 在 Docker 容器内调用 `build-ros-humble --custom --custom-prefix /data/roboframe/install`

### 4.4 产物位置与打包部署

编译产物在：

```text
$OH_ROOT/custom_build_root/roboframe_oh_ws/install
```

打包并部署到板端：

```bash
OH_CUSTOM_ROOT="$OH_ROOT/custom_build_root"
HDC_BIN=<path-to-hdc>
HDC_TARGET=<board-ip>:8710

cd "$OH_CUSTOM_ROOT/roboframe_oh_ws"
tar -zcpf roboframe-oh-install.tar.gz install

# 上传到板端
"$HDC_BIN" -t "$HDC_TARGET" file send \
  roboframe-oh-install.tar.gz /data/roboframe-oh-install.tar.gz

# 板端解压
"$HDC_BIN" -t "$HDC_TARGET" shell '
cd /data
mkdir -p /data/roboframe
tar -zxpf roboframe-oh-install.tar.gz -C /data/roboframe
ls -lah /data/roboframe/install
'
```

### 4.5 板端加载 RoboFrame 工作区

```sh
cd /data
. ./ros2ohos.env                    # 先加载官方 OH ROS 运行时
. /data/roboframe/install/setup.sh  # 再叠加 RoboFrame 工作区
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp   # 必须！

ros2 pkg list | grep -E 'roboframe_msgs|tensormsg|robot_config|inference_service'
```

> **关键**：板端必须使用 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`。默认的 `rmw_fastrtps_cpp` 在 OpenHarmony musl 上会触发 `cast_or_create_topic` 断言崩溃。

## 五、交叉编译部署第三方 ROS 2 包

板端预装运行时（`/data/install`）不包含所有需要的包。纯 C/C++ 第三方 ROS 2 包（如 `usb_cam`、`camera_ros`）必须用 OpenHarmony 交叉工具链编译，**不能**依赖 Ubuntu 构建产物。

下面以 `usb_cam`（USB 摄像头采集，推理链路必备）为例。

### 5.1 评估可行性

- **已知兼容**：纯 C/C++ 包、依赖标准 `rclcpp`/`rclpy` + 通用消息的包
- **已知有坑**：依赖 CUDA/GPU、x86 内联汇编、Qt/GTK GUI、硬依赖 glibc（OH 用 musl）的包
- 若包依赖 `/data/out/lib`、`/data/install/lib` 之外的库，需先交叉编译该库

板端预装常见库：ffmpeg（`libavcodec`/`libavformat`/`libswscale`）、OpenCV、`libv4l2`、标准 ROS 2 Humble 库。

### 5.2 准备源码

```bash
OH_CUSTOM_SRC="$OH_ROOT/custom_build_root/roboframe_oh_ws/src"
mkdir -p "$OH_CUSTOM_SRC"

git clone --depth 1 -b main https://github.com/ros-drivers/usb_cam.git "$OH_CUSTOM_SRC/usb_cam"
```

### 5.3 Docker 交叉编译

```bash
OH_CUSTOM_ROOT="$OH_ROOT/custom_build_root"
PACKAGE=usb_cam

docker run --rm -i \
    -e WS_ROOT=/mnt/ohos/tmp \
    -e OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18 \
    --name roboframe-oh-build \
    -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
    -v "${OH_CUSTOM_ROOT}:/mnt/ohos/tmp" \
    voxelsky/ohos-ros-humble-builder:v0.1.5 \
    bash -lc "
set -euo pipefail
export OHOS_CPU=aarch64
export OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18
build-ros-humble --custom \
    --wd /mnt/ohos/tmp/roboframe_oh_ws \
    --custom-prefix /data/roboframe/install \
    --colcon-args --packages-select ${PACKAGE}
"

# Docker 以 root 运行，修复产物属主
docker run --rm \
    -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
    voxelsky/ohos-ros-humble-builder:v0.1.5 \
    sh -c "chown -R $(id -u):$(id -g) /mnt/ohos/roboframe_oh_ws/install/usb_cam || true"
```

关键标志：

- `--custom`：启用 OHOS SDK 交叉编译模式
- `--custom-prefix /data/roboframe/install`：安装前缀与板端部署路径一致
- `--packages-select`：只编译目标包及其依赖

### 5.4 验证产物类型

```bash
file ${OH_CUSTOM_ROOT}/roboframe_oh_ws/install/usb_cam/lib/usb_cam/usb_cam_node_exe
# 期望: ELF 64-bit LSB ... ARM aarch64 ... dynamically linked ... ld-musl-aarch64.so.1
```

### 5.5 部署到板端

```bash
HDC_BIN=<path-to-hdc>
HDC_TARGET=<board-ip>:8710

# 推荐部署完整 install/usb_cam 目录（保持 lib/share/ament_index 一致）
"$HDC_BIN" -t "$HDC_TARGET" shell 'mkdir -p /data/roboframe/install'
"$HDC_BIN" -t "$HDC_TARGET" file send \
    ${OH_CUSTOM_ROOT}/roboframe_oh_ws/install/usb_cam \
    /data/roboframe/install/usb_cam
```

### 5.6 板端验证

```bash
hdc -t <board-ip>:8710 shell '
source /data/ros2ohos.env && source /data/roboframe/install/setup.sh && \
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
ros2 pkg prefix usb_cam
# 期望: /data/roboframe/install/usb_cam
'
```

### 5.7 ffmpeg / libswscale 依赖

`usb_cam` 的 `mjpeg2rgb`、`yuyv` 转换依赖 ffmpeg。若启动报 `libswscale.so.5` / `libavcodec.so.58` / `libavutil.so.56` 找不到，把板端 `/data/out/lib` 链入：

```sh
mkdir -p /data/roboframe/install/usb_cam/lib
ln -sf /data/out/lib/libswscale.so.5  /data/roboframe/install/usb_cam/lib/libswscale.so.5
ln -sf /data/out/lib/libavcodec.so.58  /data/roboframe/install/usb_cam/lib/libavcodec.so.58
ln -sf /data/out/lib/libavutil.so.56   /data/roboframe/install/usb_cam/lib/libavutil.so.56
```

### 5.8 第三方 ROS 包部署登记表

| 包 | 来源 | 版本 | 板端路径 | 状态 |
| --- | --- | --- | --- | --- |
| `usb_cam` | ros-drivers/usb_cam | main (0.8.x) | `/data/roboframe/install/usb_cam/` | 已验证 640×480 MJPEG 30 FPS |

> 其它纯 Python 包（如 `draccus`、`deepdiff`、`scservo_sdk` 等）无 C 扩展，可在主机 `pip download` 后解压 whl 到板端 site-packages，无需交叉编译。

## 六、内核驱动编译（USB ACM / 手柄）

SO-101 机械臂使用 CH9102 芯片以 **CDC ACM 模式** 报告，默认内核缺少 `CONFIG_USB_ACM`；游戏手柄遥操作也需要对应 HID 驱动。需重新编译并刷入 `boot_linux.img`。

### 6.1 修改内核 defconfig

编辑 `kernel/linux/config/linux-6.6/rk3588/arch/arm64_defconfig`，确认启用：

```diff
+CONFIG_USB_ACM=y              # SO-101 机械臂（CH9102 以 CDC ACM 报告，仅 CH341 不够）
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

### 6.2 编译并刷入 boot_linux.img

在 OpenHarmony 源码根目录：

```bash
./build.sh -p bq3588 --ccache
# 产物: out/bq3588/packages/phone/images/boot_linux.img

# 备份并刷入（boot_linux 分区为 mmcblk0p5）
dd if=/dev/block/by-name/boot_linux of=/data/boot_linux_backup.img
dd if=out/bq3588/packages/phone/images/boot_linux.img of=/dev/block/by-name/boot_linux
reboot
```

### 6.3 验证

```bash
ls -la /dev/ttyACM0
# crw-rw---- 1 root radio 166, 0 ... /dev/ttyACM0
dmesg | grep cdc_acm
# cdc_acm 5-1.2:1.0: ttyACM0: USB ACM device
```

## 七、板端多运行时叠加环境

板端同时运行 ROS 2 节点和 PyTorch 推理需要叠加三套运行时：

| 生态 | 基础路径 | 说明 |
| --- | --- | --- |
| ROS 2 Humble | `/data/install` + `/sys_prod/robot/out` | OH 预编译 ROS 2 |
| RoboFrame 包 | `/data/roboframe/install` | 交叉编译产物 |
| Torch 运行时 | `/data/local/skh-run/usr` | `thirdparty_pytorch` 的 `skh-run` (aarch64) |
| RKNN 运行时 | `/vendor/lib64/librknnrt.so` | NPU 驱动 |

**核心问题**：`skh-run` 自带 Python 3.12（`libpython3.12.so`），与系统 Python 3.12 存在 ABI 冲突，不能仅靠 `PYTHONPATH` 叠加（会 SIGSEGV），必须用 `LD_PRELOAD` + `PYTHONPATH` + `LD_LIBRARY_PATH` 完整叠加。

### 7.1 部署 torch runtime（skh-run）

从 [`thirdparty_pytorch`](https://gitcode.com/openharmony-robot/thirdparty_pytorch) 仓库拉取 LFS 大文件：

```bash
git clone https://gitcode.com/openharmony-robot/thirdparty_pytorch /tmp/thirdparty_pytorch
cd /tmp/thirdparty_pytorch
git lfs pull --include='test/skh-run.tar.gz'
```

上传到板端并解压：

```bash
HDC_BIN=<path-to-hdc>; HDC_TARGET=<board-ip>:8710
"$HDC_BIN" -t "$HDC_TARGET" file send test/skh-run.tar.gz /data/local/skh-run.tar.gz
"$HDC_BIN" -t "$HDC_TARGET" shell 'cd /data/local && tar -zxpf skh-run.tar.gz'
# 结果: /data/local/skh-run/usr
```

### 7.2 完整环境变量（顺序敏感）

```bash
# ① 先设置 Python 运行时（setup.sh 内部需要 Python 扫描已安装包）
export PYTHONHOME=/data/local/skh-run/usr
export LD_PRELOAD=${PYTHONHOME}/lib/libpython3.12.so.1.0:${PYTHONHOME}/lib/libomp.so
export LD_LIBRARY_PATH=${PYTHONHOME}/lib:/sys_prod/robot/out/lib:/data/install/lib:/vendor/lib64
export PYTHONPATH=${PYTHONHOME}/lib/python3.12/site-packages:/sys_prod/robot/out/lib/python3.12/site-packages:/data/install/lib/python3.12/site-packages

# ② 加载 ROS + RoboFrame 环境（此时 Python 可用，能正确扫描所有包）
cd /data && . ./ros2ohos.env && . /data/roboframe/install/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ③ 补充 RoboFrame 特有的 Python 和动态库路径
export PATH=${PYTHONHOME}/bin:$PATH
export PYTHONPATH=/data/roboframe/install/lerobot/src:\
/data/roboframe/install/inference_service/lib/python3.12/site-packages:\
/data/roboframe/install/tensormsg/lib/python3.12/site-packages:\
/data/roboframe/install/robot_config/lib/python3.12/site-packages:\
/data/roboframe/install/roboframe_msgs/lib/python3.12/site-packages:${PYTHONPATH}
export LD_LIBRARY_PATH=/data/roboframe/install/roboframe_msgs/lib:\
/data/roboframe/install/inference_service/lib:\
/data/roboframe/install/robot_config/lib:\
/data/roboframe/install/so101_hardware/lib:\
${PYTHONHOME}/lib/python3.12/site-packages/torch/lib:\
${PYTHONHOME}/lib/python3.12/site-packages/torchaudio/lib:${LD_LIBRARY_PATH}
```

> **顺序关键**：`PYTHONHOME` + `LD_PRELOAD` 必须在 `source setup.sh` **之前**设置，否则 `local_setup.sh` 调用 Python 扫描包时会失败，导致 `robot_description` 等包路径不会进入 `AMENT_PREFIX_PATH`。

> **C++ 节点不要预加载 libpython**：`usb_cam_node_exe` 是 C++ 进程，不应注入 `libpython3.12.so.1.0`（会干扰 ffmpeg/libswscale 链接）。请在 launch 配置中给相机节点单独设置 env：保留 `libomp.so`、去掉 `libpython`、显式补齐库路径与 `RMW_IMPLEMENTATION`。

## 八、RKNN NPU 推理部署

RK3588 内置 6 TOPS NPU。把训练好的 ACT 策略模型转为 RKNN 格式即可在板端 NPU 推理。

### 8.1 主机：ONNX → RKNN

`rknn-toolkit2` 要求 `torch<=2.4.0` + `numpy<=1.26.4`，与训练环境冲突，需专用虚拟环境：

```bash
python3 -m venv .venv-rknn
source .venv-rknn/bin/activate
pip install rknn-toolkit2==2.3.2

# 从 ONNX 转换
python tools/export_onnx_rknn.py \
    --onnx models/act_ros2_rknn.onnx \
    --output models/act_ros2_rknn.rknn \
    --dtype float16
```

转换结果：`act_ros2_rknn.rknn`（约 114 MB，float16）。

### 8.2 板端：修复 rknnlite .so 后缀

板端 Python 期望 `.cpython-312-aarch64-linux-ohos.so`，但预装的 rknnlite `.so` 使用 `linux-gnu` 后缀，需逐目录重命名：

```bash
HDC_BIN=<path-to-hdc>; HDC_TARGET=<board-ip>:8710
for d in api api/npu_config utils; do
  "$HDC_BIN" -t "$HDC_TARGET" shell "
    for f in /sys_prod/robot/out/lib/python3.12/site-packages/rknnlite/$d/*.cpython-312-aarch64-linux-gnu.so; do
      new=\"\${f%-gnu.so}-ohos.so\"; cp \"\$f\" \"\$new\";
    done"
done
```

### 8.3 板端：放置 librknnrt.so

rknnlite 在 `/usr/lib/` 查找 `librknnrt.so`，根文件系统默认只读，先重新挂载：

```bash
"$HDC_BIN" -t "$HDC_TARGET" shell 'mount -o rw,remount / && mkdir -p /usr/lib && cp /vendor/lib64/librknnrt.so /usr/lib/'
```

### 8.4 板端：LD_PRELOAD 解决 Python 符号可见性

板端 Python 动态链接 `libpython3.12.so`，但 musl 动态链接器不会把这些符号暴露给 `dlopen` 加载的扩展模块，需预加载：

```bash
export LD_PRELOAD=/sys_prod/robot/out/lib/libpython3.12.so.1.0
```

### 8.5 推送模型并验证

```bash
"$HDC_BIN" -t "$HDC_TARGET" file send models/act_ros2_rknn.rknn /data/local/tmp/act_ros2_rknn.rknn

"$HDC_BIN" -t "$HDC_TARGET" shell 'LD_PRELOAD=/sys_prod/robot/out/lib/libpython3.12.so.1.0 python3 -c "
import numpy as np, time
from rknnlite.api import RKNNLite
rknn = RKNNLite()
rknn.load_rknn(\"/data/local/tmp/act_ros2_rknn.rknn\")
rknn.init_runtime(target=None)  # None = 使用本机 NPU
state = np.random.randn(1, 14).astype(np.float32)
cam_high = np.random.randn(1, 3, 480, 640).astype(np.float32)
cam_left = np.random.randn(1, 3, 480, 640).astype(np.float32)
t0 = time.time()
outputs = rknn.inference(inputs=[state, cam_high, cam_left])
print(f\"output shape={outputs[0].shape}, time={time.time()-t0:.3f}s\")
rknn.release()
"'
# 期望: output shape=(1, 100, 6), time≈0.121s
```

> **输入顺序**：RKNN 编译器可能重排模型输入。ACT 模型 ONNX 输入为 `[cam_high, cam_left, state]`，转 RKNN 后期望 `[state, cam_high, cam_left]`。重新导出模型后务必测试验证。

### 8.6 RKNN 模型配置（YAML）

板端 YAML 配置需指定 RKNN 模型（`path` 必须用绝对路径）：

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

> RKNN 模式的预处理（归一化等）已 baked into 模型，推理服务在 `device:=rknn` 时会自动跳过 LeRobot 完整预处理管线，避免在板端安装大量 C 扩展训练依赖。

### 8.7 性能数据

| 指标 | 数值 |
| --- | --- |
| 模型大小（float16） | 约 114 MB |
| NPU 单次推理延迟 | 约 121 ms |
| 单板闭环稳定端到端延迟 | 约 500 ms |
| Action chunk size | 100 |
| 输出 shape | `(1, 100, 6)` — 100 × 6 DoF |

### 8.8 版本兼容性

| 组件 | 版本 |
| --- | --- |
| rknn-toolkit2（主机，转换） | 2.3.2 |
| rknn-toolkit-lite2（板端，推理） | 2.3.2（预装） |
| librknnrt.so（板端） | 2.4.1b0 |

## 九、单板全链路闭环与启动验证

本节说明在 BQ3588HM 上独立运行完整闭环：**摄像头采集 → NPU 推理 → 机械臂控制**，无需外接 Ubuntu 主机。

### 9.1 闭环架构

```text
BQ3588HM (RK3588, OpenHarmony)
├── usb_cam_node_exe × 2          (top + wrist 相机, MJPEG 640×480)
├── static_transform_publisher × 4 (TF: base→camera, gripper→camera, optical)
├── lerobot_policy_node × 1        (RKNN NPU 推理, ACT 策略)
├── action_dispatcher_node × 1     (动作分发, 20Hz)
└── so101_hardware                 (ros2_control, /dev/ttyACM0)
     ↳ arm_position_controller / gripper_position_controller
```

数据流：`相机 Image → 推理节点 (NPU ~500ms) → Action Dispatcher → Joint Commands → 机械臂`

### 9.2 机械臂校准

首次使用前必须在板端交互式校准：

```bash
ssh root@<board-ip>
cd /data && . ./ros2ohos.env && . /data/roboframe/install/setup.sh
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ros2 run so101_hardware calibrate_arm --arm follower --port /dev/ttyACM0
```

校准 JSON 保存在 `~/.calibrate/so101_follower_calibrate.json`（SSH 下 `HOME=/root`）。HDC shell 下 `HOME=/`，需做符号链接：

```bash
mkdir -p /.calibrate
ln -sf /root/.calibrate/so101_follower_calibrate.json /.calibrate/so101_follower_calibrate.json
```

### 9.3 清理残留进程

```bash
pkill -9 -f "ros2 launch\|lerobot_policy_node\|action_dispatcher_node\|usb_cam_node_exe\|static_transform_publisher"
```

### 9.4 启动全链路

```bash
ssh root@<board-ip>

# 按 7.2 节顺序设置完整环境变量后：
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=false \
    control_mode:=model_inference \
    device:=rknn \
    2>&1 | tee /data/launch.log
```

### 9.5 预期日志

```text
[top_camera]: Starting 'top' (/dev/video20) at 640x480 via mmap (mjpeg2rgb) at 30 FPS
[top_camera]: Timer triggering every 33 ms
[wrist_camera]: Starting 'wrist' (/dev/video22) at 640x480 via mmap (mjpeg2rgb) at 60 FPS
[act_inference_node]: Using inference_backend=rknn, tensor_device=cpu
[act_inference_node]: DispatchInfer Action Server ready
[act_inference_node]: ✓ First inference complete (monolithic): total=~500ms
[action_dispatcher]: ✓ First inference received: chunk=100
```

### 9.6 可安全忽略的 warning

| Warning | 原因 | 处理 |
| --- | --- | --- |
| `Camera calibration file not found` | 未做相机内参标定 | 忽略 |
| `Query dynamic range failed (RKNN_ERR_MODEL_INVALID)` | 静态 shape 模型正常警告 | 忽略 |
| `swscaler ... No accelerated colorspace conversion` | ffmpeg 走 CPU 色彩转换 | 忽略（仅性能提示） |
| `unknown control 'white_balance_temperature_auto'` | USB 摄像头不支持该 V4L2 控制 | 忽略 |

### 9.7 闭环性能数据

| 指标 | 数值 |
| --- | --- |
| NPU 推理延迟（首次含加载） | ~900 ms |
| NPU 推理延迟（稳定） | ~500 ms |
| 总端到端延迟（含预处理） | ~520 ms |
| 控制频率 | 20 Hz |
| 相机帧率 | top: 30 FPS, wrist: 60 FPS |

## 十、常见问题（FAQ）

| 问题 | 原因 | 解决方法 |
| --- | --- | --- |
| `/dev/ttyACM0` 不存在 | 内核缺少 `CONFIG_USB_ACM=y` | 重新编译内核（第六节） |
| `Assertion failed: ... cast_or_create_topic` | 使用了 `rmw_fastrtps_cpp` | `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| `librclcpp.so` / `libclass_loader.so` 找不到 | `LD_LIBRARY_PATH` 未含 `/data/install/lib` | 补齐库路径 |
| `libswscale.so.5` 找不到 | ffmpeg 库不在搜索路径 | 链入 `/data/out/lib`（5.7 节） |
| `ImportError: symbol not found (PyUnicode...)` | Python 符号未暴露给 dlopen | `LD_PRELOAD=libpython3.12.so.1.0`（8.4 节） |
| `ModuleNotFoundError: 'rknnlite.xxx'` | `.so` 后缀不匹配 | 重命名 `-gnu.so` → `-ohos.so`（8.2 节） |
| `Can not find dynamic library on RK3588!` | `/usr/lib` 缺 `librknnrt.so` | 从 `/vendor/lib64/` 复制（8.3 节） |
| `RKNN model file not found` | 模型 path 为相对路径 | YAML 用绝对路径（8.6 节） |
| `Calibration file not found` | SSH vs HDC 的 HOME 不同 | 符号链接校准文件（9.2 节） |
| 推理节点 SIGSEGV | `LD_PRELOAD` 未设置或 torch ABI 冲突 | 确认完整环境变量（7.2 节） |
| usb_cam crash（`libc++abi: terminating`） | 用了错误的（非 OH）usb_cam 二进制 | 用 OH 交叉编译版本替换（第五节） |
| RViz 显示 `No image` | QoS 不匹配或 `yuyv` 编码 | 用 `mjpeg2rgb`，RViz Image Display 改 `Best Effort` |
| `input[0] need 2dims input, but 4dims` | RKNN 输入顺序重排 | state 放最前面（8.5 节） |

## 十一、相关文档与生态

### 官方文档

- [OpenHarmony EmbodiedAI 1.0.1 Release 说明](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/usage.md)
- [Docker 交叉编译官方文档](https://gitcode.com/openharmony-robot/docs/blob/main/device-dev/docker-build.md)

### OpenHarmony 机器人生态仓库

| 仓库 | 说明 |
| --- | --- |
| [ros_ros2_base](https://gitcode.com/openharmony-robot/ros_ros2_base) | ROS 2 核心基础功能包（rcl/rclcpp/rclpy、ament、DDS、tf2 等） |
| [ros_ros2_control](https://gitcode.com/openharmony-robot/ros_ros2_control) | ros2_control 控制框架（含 OH 平台适配） |
| [ros_moveit2](https://gitcode.com/openharmony-robot/ros_moveit2) | MoveIt2 运动规划框架 |
| [ros_navigation2](https://gitcode.com/openharmony-robot/ros_navigation2) | Navigation2 自主导航 |
| [ros_peripheral](https://gitcode.com/openharmony-robot/ros_peripheral) | 传感器及外设驱动（相机/雷达/IMU/舵机） |
| [ros_ros2_misc](https://gitcode.com/openharmony-robot/ros_ros2_misc) | 第三方辅助功能包（SLAM/图像/行为树等） |
| [oh_robot_sim](https://gitcode.com/openharmony-robot/oh_robot_sim) | 具身智能模拟器框架（MuJoCo/Gazebo + Agent） |
| [tools_ohloha](https://gitcode.com/openharmony-robot/tools_ohloha) | ohloha 系统级包管理工具 |
| [tools_ohloha_pkgs](https://gitcode.com/openharmony-robot/tools_ohloha_pkgs) | 80+ 系统依赖库源码级迁移方案 |
| [thirdparty_pytorch](https://gitcode.com/openharmony-robot/thirdparty_pytorch) | 板端 PyTorch / Python runtime（`skh-run`） |

### 工具与镜像

| 资源 | 地址 |
| --- | --- |
| 交叉编译 Docker 镜像 | `docker pull voxelsky/ohos-ros-humble-builder:v0.1.5` |
| ROS 2 Humble Desktop 镜像 | `docker pull voxelsky/ros-humble-desktop-classic:v0.0.1` |
| OH Robot Sim Demo 镜像 | `docker pull voxelsky/ohos-rsim-nav2-demo:v0.0.1` |

---

**许可证**：Apache 2.0
