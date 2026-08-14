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

## Internal References

Read only the references needed for the current step:

| Purpose | Reference |
|---------|-----------|
| 8 步详细交叉编译流程（feasibility/clone/docker build/deploy/verify/runtime issues） | `references/step-by-step.md` |
| 集成到 IB Robot launch 系统、camera driver 映射、环境变量、已部署包清单 | `references/launch-integration.md` |
| `ros2_control` / `controller_manager` 的 musl mutex 补丁专项 | `references/ros2-control-patch.md` |

Do not expose these references as separate skills.

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

详细的 8 步流程（feasibility 评估、clone 源码、Docker 交叉编译、修复 ownership、验证构建产物、部署到板端、板端验证、修复 runtime 问题）见 `references/step-by-step.md`。

**CRITICAL**: Always set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` on the board. The default `rmw_fastrtps_cpp` crashes with assertion errors on OpenHarmony musl.

## Integration with IB Robot Launch System

部署完成后集成到 launch 系统的详细说明（camera driver 映射、非 camera 包的 launch builder、已部署包清单、构建环境变量、板端环境变量）见 `references/launch-integration.md`。

## Special Case: `ros2_control` / `controller_manager`

SO-101 真机依赖 `ros2_control`，OpenHarmony musl 环境下有一个必须确认的 mutex 补丁。详见 `references/ros2-control-patch.md`。
