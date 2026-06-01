# SO-101 Robotic Arm Hardware Package

Hardware driver package for the SO-101 robotic arm, providing high-performance C++ ros2_control interfaces and Python utilities.

## Overview

This package provides a complete hardware driver solution for the SO-101 robotic arm, supporting two main modes:
- **C++ ros2_control Plugin**: Direct communication via FTServo SDK, low latency, high performance. Ideal for production and control.
- **Python Utilities**: Tools for calibration, data collection (Leader Arm Publisher), and diagnostics.

## Key Features

- **Direct Communication**: Uses FTServo SDK for native communication with Feetech servos.
- **Mixed Package Build**: Uses `ament_cmake_python` to support both C++ plugins and Python scripts in a single package.
- **Startup Position Protection**: Supports `reset_positions` to prevent the arm from jumping to zero on startup (critical for mobile platforms).
- **Lifecycle Management**: Implements standard `on_init`, `on_configure`, `on_activate`, and `on_deactivate` states.
- **Safety**: Automatically disables motor torque on node shutdown.

## Architecture

```
ros2_control (Controller Manager)
      ↓
SO101SystemHardware (C++ Plugin)  ←──┐
      ↓                              │
FTServo SDK (C++)                    │
      ↓                              │
Feetech Servos (Hardware)            │
      ↑                              │
Python Utilities (Scripts) ──────────┘
```

## Dependencies

### Git Submodule
FTServo_Linux SDK is included as a git submodule:
```bash
git submodule update --init --recursive
```

### System Dependencies
- ROS 2 Humble
- nlohmann_json library
- hardware_interface, pluginlib, rclcpp_lifecycle
- pyserial (Python driver)

## Building

```bash
cd ~/Research/lerobot_ros2/src/ros2/ros2_ws
source /opt/ros/humble/setup.zsh
# Ensure successful mixed build by setting PYTHONPATH
PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH colcon build --packages-select so101_hardware
source install/setup.zsh
```

## Tool Documentation

- [arm_calibration_transfer](docs/tools/arm_calibration_transfer.md): Legacy calibration data migration and new follower calibration file generation
- [arm_calibration_checker](docs/tools/arm_calibration_checker.md): Real-robot verification process for calibration results

## Usage

### 1. Calibrating the Arm (Python)
Calibration must be performed before first use to generate JSON files in `~/.calibrate/`.
```bash
# Calibrate Follower arm
ros2 run so101_hardware calibrate_arm --arm follower --port /dev/ttyACM0
```

### 2. C++ ros2_control Plugin Configuration
Specify the hardware interface in your URDF:
```xml
<hardware>
  <plugin>so101_hardware/SO101SystemHardware</plugin>
  <param name="port">/dev/ttyACM0</param>
  <param name="calib_file">$(env HOME)/.calibrate/so101_follower_calibrate.json</param>
  <!-- Optional: Safe startup positions (JSON format, radians) -->
  <param name="reset_positions">{"1": 0.0, "2": 0.0}</param>
</hardware>
```

### 3. Leader Arm Publisher
Used for recording demonstration data or teleoperation:
```bash
ros2 run so101_hardware leader_arm_pub --port /dev/ttyACM0 --publish_rate 50.0
```

## Implementation Details

### Startup Positions (Reset Positions)
The `reset_positions` parameter allows you to specify initial joint positions.
- **Configured**: The arm moves smoothly to the specified pose upon activation.
- **Not Configured (Default)**: The arm preserves its current motor position without movement.

### Coordinate Conversion
The plugin handles conversion between steps and radians automatically:
- **Read**: `radians = ((steps - range_min) / range - 0.5) * 2.0 * PI`
- **Write**: `steps = (radians / (2.0 * PI) + 0.5) * range + range_min`

## Comparison: C++ Plugin vs Python Tools

| Feature | C++ Plugin (Production) | Python Tools (Dev/Calib) |
|---------|-------------------------|--------------------------|
| Latency | Very Low (Direct) | Higher (Python overhead) |
| Performance | High (Real-time) | Medium |
| Mode | ros2_control Interface | Topic Bridge / Scripts |
| Use Case | RL / Trajectory Execution | Calibration / Recording / Diagnostics |

## Constants Configuration
Shared constants can be accessed in Python via:
```python
from so101_hardware.calibration.constants import MOTOR_IDS, JOINT_NAMES, DEFAULT_SERIAL_PORT
```

## Troubleshooting

- **Serial Permissions**: Run `sudo chmod 666 /dev/ttyACM0` or add user to the `dialout` group.
- **Missing Calibration**: If you see `Calibration file not found`, run the `calibrate_arm` tool first.
- **Empty Submodule**: Ensure you have run `git submodule update`.

## License
TODO: License declaration