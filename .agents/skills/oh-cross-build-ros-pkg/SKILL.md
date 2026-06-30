---
name: oh-cross-build-ros-pkg
description: "Cross-compile third-party ROS 2 packages for OpenHarmony aarch64. Use when user needs to 'cross-compile ROS package', 'port ROS package to OH', 'build ROS for OpenHarmony', '移植ROS包', '交叉编译ROS', 'OH aarch64 build', 'usb_cam board', 'deploy ROS package to board', 'third-party ROS package'. Triggers for 'BQ3588HM', 'RoboPi'."
---

# Cross-Build Third-Party ROS 2 Packages for OpenHarmony

Port and deploy third-party ROS 2 packages (e.g., `usb_cam`, `camera_ros`, `nav2` etc.) to the OpenHarmony board using the Docker-based cross-compilation toolchain.

## When to Use This Skill

- User wants to add a new ROS 2 package to the board that is NOT part of IB_Robot source tree
- User says "移植XX包到板端", "板端编译XX", "cross-compile XX for OH"
- User needs a ROS driver (camera, lidar, etc.) on the board
- Package exists as a standard ROS 2 Humble package on GitHub / rosdistro

Do NOT use for:
- Building RoboFrame's own packages (`inference_service`, `robot_config`, etc.) — use `oh-build-roboframe` skill
- Board connectivity — use `oh-access`
- Running RKNN inference — see docs/OpenHarmony_EmbodiedAI_RKNN_Inference.md

## Architecture Overview

```
Host (x86_64 Ubuntu 22.04)
  +-- Docker container (voxelsky/ohos-ros-humble-builder:v0.1.5)
  |     Cross-compiles with OHOS SDK + toolchain
  |     Outputs aarch64 ELF binaries linked against musl
  |
  +-- OH_CUSTOM_ROOT/
        +-- ibrobot_oh_ws/src/<package>/   ← source
        +-- ibrobot_oh_ws/install/<package>/ ← build output
        +-- install/                        ← OH ROS 2 Humble runtime (sysroot)
        +-- ohos-robot-toolchain/18/native/ ← OHOS SDK + sysroot
```

Board receives deployable `install/<package>/` directory via `hdc file send`.

## Prerequisites

These must exist before starting (one-time setup, likely already done):

| Component | Path on Host | Purpose |
|-----------|-------------|---------|
| OH build root | `<oh_build_root>/custom_build_root/` | Contains all build artifacts |
| OH ROS 2 Humble runtime | `<build_root>/install/` | Cross-compilation sysroot |
| OHOS SDK | `<build_root>/ohos-robot-toolchain/18/native/` | Compiler + sysroot |
| Docker builder image | `voxelsky/ohos-ros-humble-builder:v0.1.5` | Build environment |
| HDC access to board | `hdc -t <board_ip>:8710` | Deployment transport |

Board-side prerequisites:

| Component | Board Path | Purpose |
|-----------|-----------|---------|
| OH ROS 2 runtime | `/sys_prod/robot/install` + `/sys_prod/robot/out` | Base ROS 2 libraries |
| ROS env script | `/data/roboframe/scripts/robooh_1.0.1.env` | Environment setup |
| IB Robot install | `/data/roboframe/install` | IB Robot packages |
| `setup.sh` chain | `/data/roboframe/install/setup.sh` | Package discovery |

## Step-by-Step Workflow

### Step 1: Assess Package Feasibility

Before building, verify the package is compatible:

```bash
# Check package dependencies
grep -E "<build_depend>|<exec_depend>|<depend>" <pkg>/package.xml
```

**Known compatible**: Pure C/C++ packages, ROS 2 nodelets using standard message types, packages depending on `rclcpp`/`rclpy` + common messages.

**Known problematic**:
- Packages requiring CUDA/GPU
- Packages with x86_64 inline assembly
- Packages needing Qt/GTK GUI
- Packages with hard dependency on glibc (OH uses musl)

**Library dependency checklist** — if the package needs a library beyond what's in `/sys_prod/robot/out/lib/` and `/sys_prod/robot/install/lib/`, you must also cross-compile that library first.

Common pre-installed libraries on the board:
- `libavcodec`, `libavformat`, `libswscale` (ffmpeg)
- `libopencv_*` (OpenCV)
- `libv4l2` (V4L2)
- Standard ROS 2 Humble libraries

### Step 2: Clone Package Source

```bash
# Workspace source directory
OH_WS_SRC=<build_root>/ibrobot_oh_ws/src

# Clone the package (use --depth 1 for speed, specify branch if needed)
git clone --depth 1 -b <branch> <repo_url> ${OH_WS_SRC}/<package_name>

# Example: usb_cam
git clone --depth 1 -b main https://github.com/ros-drivers/usb_cam.git ${OH_WS_SRC}/usb_cam
```

### Step 3: Cross-Build in Docker

Use the OH builder Docker image to cross-compile:

```bash
OH_CUSTOM_ROOT=<build_root>  # e.g., <oh_build_root>/custom_build_root
OH_IMAGE=voxelsky/ohos-ros-humble-builder:v0.1.5
PACKAGE=<package_name>  # e.g., usb_cam

docker run --rm -i \
    -e WS_ROOT=/mnt/ohos/tmp \
    -e OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18 \
    --name ibrobot-oh-build \
    -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
    -v "${OH_CUSTOM_ROOT}:/mnt/ohos/tmp" \
    "${OH_IMAGE}" \
    bash -lc "
set -euo pipefail
export OHOS_CPU=aarch64
export OHOS_SDK=/mnt/ohos/tmp/ohos-robot-toolchain/18
build-ros-humble --custom \
    --wd /mnt/ohos/tmp/ibrobot_oh_ws \
    --custom-prefix /data/roboframe/install \
    --colcon-args --packages-select ${PACKAGE}
"
```

**Key flags**:
- `--custom`: Uses the OHOS SDK cross-compilation mode
- `--custom-prefix /data/roboframe/install`: Sets the install prefix to match board deployment path
- `--packages-select`: Only builds the target package(s) + their dependencies in the workspace

### Step 4: Fix Ownership (if needed)

Docker runs as root; fix ownership of build artifacts:

```bash
docker run --rm \
    -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
    "${OH_IMAGE}" \
    sh -c "chown -R $(id -u):$(id -g) /mnt/ohos/ibrobot_oh_ws/install /mnt/ohos/ibrobot_oh_ws/build /mnt/ohos/ibrobot_oh_ws/log || true"
```

### Step 5: Verify Build Output

```bash
# Check binary type — must be aarch64 + musl
file ${OH_CUSTOM_ROOT}/ibrobot_oh_ws/install/${PACKAGE}/lib/*/$(basename ${PACKAGE}_node_exe || echo *)

# Expected output: "ELF shared object, 64-bit LSB arm64, dynamic (/lib/ld-musl-aarch64.so.1)"
```

### Step 6: Deploy to Board

```bash
hdc file send ${OH_CUSTOM_ROOT}/ibrobot_oh_ws/install/${PACKAGE} /data/roboframe/install/${PACKAGE}
```

### Step 7: Verify on Board

```bash
# Check package discovery
hdc shell 'source /data/roboframe/scripts/robooh_1.0.1.env && source /data/roboframe/install/setup.sh && ros2 pkg prefix ${PACKAGE}'

# Test run (use CycloneDDS!)
hdc shell 'source /data/roboframe/scripts/robooh_1.0.1.env && source /data/roboframe/install/setup.sh && \
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
    timeout 10 ros2 run ${PACKAGE} <executable> --ros-args <params>'
```

**CRITICAL**: Always set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` on the board. The default `rmw_fastrtps_cpp` crashes with assertion errors on OpenHarmony musl.

### Step 8: Fix Runtime Issues

Common runtime problems and fixes:

| Problem | Cause | Fix |
|---------|-------|-----|
| `No executable found` | Binary not marked executable or PATH issue | `chmod +x` the binary; use full path |
| `Assertion failed: ... cast_or_create_topic` | Using `rmw_fastrtps_cpp` | Set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| `cannot open shared object` | Missing library | Check with `ldd`; deploy missing lib to `/sys_prod/robot/out/lib/` |
| `Cannot open device` | Permission denied on `/dev/videoX` | `chmod 666 /dev/videoX` |
| Parameter not taking effect | ROS 2 parameter name mismatch | Check `declare_parameter()` names in source |

## Integration with IB Robot Launch System

Once the package is deployed, integrate it into the robot launch:

### For Camera Drivers

The robot YAML config at `src/robot_config/config/robots/<robot>.yaml` controls camera launch:

```yaml
peripherals:
  - type: camera
    name: top
    driver: opencv          # "opencv" → uses usb_cam_node_exe
    index: 0                # → /dev/video0 (or /dev/video20 on board)
    width: 640
    height: 480
    fps: 30
    pixel_format: mjpeg2rgb
    frame_id: camera_top_frame
```

The `driver: opencv` mapping is defined in `src/robot_config/robot_config/launch_builders/perception.py:52-107`:
- `driver: opencv` → launches `usb_cam_node_exe` from `usb_cam` package
- `driver: camera_ros` → launches `camera_ros_node` from `camera_ros` package
- `driver: realsense2_camera` → launches `realsense2_camera_node`

If adding a new driver type, you must edit `perception.py` to add the mapping.

### For Non-Camera Packages

Add the package to the appropriate launch builder:
- Navigation: `launch_builders/navigation.py`
- Voice/sensors: create a new builder
- Or add directly in `robot.launch.py`

## Deployed Packages Registry

| Package | Source | Version | Binary | Board Path | Status |
|---------|--------|---------|--------|-----------|--------|
| usb_cam | ros-drivers/usb_cam | main (0b1c9d7) | `usb_cam_node_exe` (aarch64/musl) | `/data/roboframe/install/usb_cam/` | Verified 30 FPS @ 640x480 MJPEG |
| | | | | | |

> **注意**：OpenHarmony EmbodiedAI 1.0.1 系统已内置 `usb_cam`（位于 `/sys_prod/robot/install`）。通常无需单独交叉编译，除非需要特定版本。

## Special Case: `ros2_control` / `controller_manager`

SO-101 真机依赖 `ros2_control_node` + `controller_manager` + spawner。OpenHarmony musl 环境下有一个必须确认的关键补丁。

### 源码来源

```text
https://gitcode.com/openharmony-robot/ros_ros2_control
tag OpenHarmony-Embodied-v1.0.1-Release (commit c742704)
```

包含 `ros2_control`、`controller_manager`、`realtime_tools` 等包（版本 2.53.0）。该 fork 已包含 `realtime_tools` 的 `__OHOS__` 平台适配（跳过 `mlockall`、thread affinity 等 glibc-only 路径）。

### controller_manager mutex 补丁（关键）

`controller_manager/src/controller_manager.cpp` 的 `switch_controller()` 中，`std::defer_lock` 在 musl 上会导致 `condition_variable::wait_for` 不稳定，必须改为直接加锁：

```diff
-  std::unique_lock<std::mutex> switch_params_guard(switch_params_.mutex, std::defer_lock);
+  std::unique_lock<std::mutex> switch_params_guard(switch_params_.mutex);
```

RoboFrame 发布包的 `install.sh` 会通过 patched `.so` 覆盖机制自动应用此修复（`/data/roboframe/install/patches/lib/`）。

### 验证

正常 launch 日志应包含：

```text
[ros2_control_node]: Successful initialization of hardware 'RobotSystem'
[spawner_joint_state_broadcaster_group]: Loaded joint_state_broadcaster
[spawner_joint_state_broadcaster_group]: Configured and activated all the parsed controllers list
```

如果 spawner 崩溃或卡在 switch controller，说明 `controller_manager` 未包含 mutex 补丁。

## Reference: Build Environment Variables

```bash
# Default values (override via CLI flags or environment)
OH_CUSTOM_ROOT=<oh_build_root>/custom_build_root
OH_CUSTOM_WS=${OH_CUSTOM_ROOT}/ibrobot_oh_ws
OH_CUSTOM_SRC=${OH_CUSTOM_WS}/src
OH_CUSTOM_TOOLCHAIN_ROOT=${OH_CUSTOM_ROOT}/ohos-robot-toolchain
OH_CUSTOM_IMAGE=voxelsky/ohos-ros-humble-builder:v0.1.5
OH_CUSTOM_PREFIX=/data/roboframe/install       # on-device install prefix
OH_BOARD_ROS_PREFIX=/sys_prod/robot/install            # on-device ROS 2 Humble prefix
```

## Reference: Board Environment

```bash
# Always source these two on the board before any ROS command
source /data/roboframe/scripts/robooh_1.0.1.env
source /data/roboframe/install/setup.sh

# Always set DDS implementation
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# For multi-machine (board + Ubuntu), use same domain
export ROS_DOMAIN_ID=42
```
