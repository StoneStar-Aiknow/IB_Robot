# Package-Specific Build Notes

## When to Read

- 构建特定 ROS 2 包时
- 需要了解某包的构建特性（C++ 组件、依赖、构建速度）

## robot_config (Python package)

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select robot_config
```
- Fast build (~1 second)
- Generates contracts at launch time (not build time)

## inference_service (Python package)

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select inference_service
```
- Requires lerobot in PYTHONPATH
- Treat `src/inference_service/package.xml` as the dependency SSOT; inspect its current declarations instead of maintaining a duplicate dependency list here

## so101_hardware (C++ + Python)

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select so101_hardware
```
- Has C++ ros2_control component
- Also has Python scripts
