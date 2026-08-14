# Step-by-Step Cross-Build Workflow

## When to Read

- 需要执行具体的交叉编译步骤时
- 需要参考 Docker 命令、deploy 命令、board 验证命令时
- 需要查看 runtime issues 表时

## Step 1: Assess Package Feasibility

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

## Step 2: Clone Package Source

```bash
# Workspace source directory
OH_WS_SRC=<build_root>/ibrobot_oh_ws/src

# Clone the package (use --depth 1 for speed, specify branch if needed)
git clone --depth 1 -b <branch> <repo_url> ${OH_WS_SRC}/<package_name>

# Example: usb_cam
git clone --depth 1 -b main https://github.com/ros-drivers/usb_cam.git ${OH_WS_SRC}/usb_cam
```

## Step 3: Cross-Build in Docker

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

## Step 4: Fix Ownership (if needed)

Docker runs as root; fix ownership of build artifacts:

```bash
docker run --rm \
    -v "${OH_CUSTOM_ROOT}:/mnt/ohos" \
    "${OH_IMAGE}" \
    sh -c "chown -R $(id -u):$(id -g) /mnt/ohos/ibrobot_oh_ws/install /mnt/ohos/ibrobot_oh_ws/build /mnt/ohos/ibrobot_oh_ws/log || true"
```

## Step 5: Verify Build Output

```bash
# Check binary type — must be aarch64 + musl
file ${OH_CUSTOM_ROOT}/ibrobot_oh_ws/install/${PACKAGE}/lib/*/$(basename ${PACKAGE}_node_exe || echo *)

# Expected output: "ELF shared object, 64-bit LSB arm64, dynamic (/lib/ld-musl-aarch64.so.1)"
```

## Step 6: Deploy to Board

```bash
hdc file send ${OH_CUSTOM_ROOT}/ibrobot_oh_ws/install/${PACKAGE} /data/roboframe/install/${PACKAGE}
```

## Step 7: Verify on Board

```bash
# Check package discovery
hdc shell 'source /data/roboframe/scripts/robooh_1.0.1.env && source /data/roboframe/install/setup.sh && ros2 pkg prefix ${PACKAGE}'

# Test run (use CycloneDDS!)
hdc shell 'source /data/roboframe/scripts/robooh_1.0.1.env && source /data/roboframe/install/setup.sh && \
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && \
    timeout 10 ros2 run ${PACKAGE} <executable> --ros-args <params>'
```

**CRITICAL**: Always set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` on the board. The default `rmw_fastrtps_cpp` crashes with assertion errors on OpenHarmony musl.

## Step 8: Fix Runtime Issues

Common runtime problems and fixes:

| Problem | Cause | Fix |
|---------|-------|-----|
| `No executable found` | Binary not marked executable or PATH issue | `chmod +x` the binary; use full path |
| `Assertion failed: ... cast_or_create_topic` | Using `rmw_fastrtps_cpp` | Set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| `cannot open shared object` | Missing library | Check with `ldd`; deploy missing lib to `/sys_prod/robot/out/lib/` |
| `Cannot open device` | Permission denied on `/dev/videoX` | `chmod 666 /dev/videoX` |
| Parameter not taking effect | ROS 2 parameter name mismatch | Check `declare_parameter()` names in source |
