# Integration with IB Robot Launch System

## When to Read

- 部署完第三方 ROS 2 包后需要集成到机器人 launch 系统时
- 需要配置 camera driver 或其他外设时
- 需要了解 `driver: opencv` 映射逻辑时
- 需要为非 camera 包添加 launch builder 时

## For Camera Drivers

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

The mapping is defined by `generate_camera_nodes()` in
`src/robot_config/robot_config/launch_builders/perception.py`:
- `driver: opencv` → launches `usb_cam_node_exe` from the `usb_cam` package
- `driver: camera_ros` → launches `camera_node` from the `camera_ros` package
- `driver: realsense` → launches `realsense2_camera_node` from the `realsense2_camera` package

If adding a new driver type, add its branch to `generate_camera_nodes()` rather than relying on a fixed source-line range.

## For Non-Camera Packages

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
