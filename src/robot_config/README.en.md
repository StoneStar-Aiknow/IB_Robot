# robot_config

Unified robot configuration system for ros2_control and peripherals.

## Overview

This package provides a unified configuration system for robot hardware that bridges:

- **ros2_control**: For joint/motor control interfaces
- **Peripherals**: For cameras and other devices (via existing ROS2 drivers)
- **tensormsg**: For ML policy I/O contracts

The goal is to have a single source of truth for robot hardware configuration, eliminating duplication between different configuration systems.

## Features

- **Single YAML configuration**: Define ros2_control, cameras, and ML contracts in one file
- **Uses existing ROS2 camera drivers**:
  - `usb_cam` for USB cameras (OpenCV-based)
  - `realsense2_camera` for RealSense D400 series
- **TF publishing**: Automatic camera frame transform publishing
- **Calibration support**: Standard ROS2 camera_info_manager integration
- **tensormsg integration**: Contracts reference peripherals by name

## Architecture

```
robot_config YAML (single source of truth)
        │
        ├───► ros2_control (joints/motors)
        │       └───► so101_hardware plugin
        │
        ├───► Camera drivers (existing ROS2 packages)
        │       ├───► usb_cam (USB cameras)
        │       └───► realsense2_camera (RealSense D400)
        │
        └───► tensormsg contracts (ML I/O)
                └───► PolicyBridge / EpisodeRecorder
```

## Configuration Example

```yaml
robot:
  name: so101_single_arm
  type: so101
  robot_type: so_101

  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
    port: /dev/ttyACM0
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
    reset_positions:
      "1": 0.0813
      "2": 3.7905

  peripherals:
    - type: camera
      name: top
      driver: opencv  # Uses usb_cam package
      index: 0
      width: 640
      height: 480
      fps: 30
      frame_id: camera_top_frame
      optical_frame_id: camera_top_optical_frame

    - type: camera
      name: wrist
      driver: realsense  # Uses realsense2_camera package
      serial_number: "12345678"
      width: 640
      height: 480
      fps: 30
      depth_width: 640
      depth_height: 480
      frame_id: camera_wrist_frame

  contract:
    observations:
      - key: observation.images.top
        topic: /camera/top
        peripheral: top  # References camera above
        image:
          resize: [480, 640]
```

## Capability Gateway Public Contract

`robot.embodied.skill_templates.<skill>.capability` is the public Capability Gateway SSOT. It is explicit metadata,
not a schema derived from `description` or a primitive sequence. Every enabled template must declare
`schema_version: 1`, `summary`, `domain`, `moves_robot`, `required_control_mode`, `parameters`, and
`recovery_policy`.

| Field | Validation |
| --- | --- |
| `summary` / `domain` | Non-empty strings |
| `moves_robot` | Boolean |
| `required_control_mode` | `teleop`, `model_inference`, or `moveit_planning`, exactly equal to global `skill_required_control_mode` |
| `parameters` | Strict object schema: `type: object`, `additionalProperties: false`; only `target_name`, `place_name`, `motion_direction`, and `motion_distance`; unique `required` entries that name declared properties |
| `recovery_policy` | `never_retry`, `ask_user`, or `recover_safe_pose` |

String parameter definitions permit only `type` and a non-empty `enum`; a `motion_direction` enum is limited to
`forward`, `backward`, `left`, `right`, `up`, and `down`. `motion_distance` must be a `number` with
`exclusiveMinimum: 0` and a `meters` or `degrees` `unit`. Any other schema key or public request property is rejected.

`load_robot_config_dict()` is the canonical normalized loader used by launch, CLI, and catalog consumers. It validates
the capability, template, and Gateway invariants before returning a config. When `embodied.skill_templates` is
non-empty, `skill_required_control_mode` must be a non-empty member of `control_modes`, and every enabled capability
must match it exactly.

Shared config selection is ordered as explicit `config_path`, explicit `config_name`, `ROBOT_CONFIG`, `ROBOT_NAME`,
then `so101_single_arm`. A name resolves first from the installed `robot_config/config/robots/` directory and then
from the source `config/robots/` directory; an explicit path must exist. The public catalog exposes only capability
fields, named-pose names, timeout policy, and a digest. Primitive sequences, joint/cartesian coordinates, target
bindings, and ROS service/action/topic names remain private implementation data.

## Control Mode Configuration

The robot_config package supports dual control modes for different AI model requirements:

### Available Control Modes

#### 1. teleop Mode (Human Teleoperation)

**Use for:** Human teleoperation devices (leader arm, gamepad, VR device)

**Characteristics:**
- Real-time direct control
- Zero-latency passthrough (< 5ms)
- Multiple input device support
- Built-in safety filters (joint limits)

**Configuration:**
```yaml
robot:
  default_control_mode: "teleop"

  control_modes:
    teleop:
      description: "Human teleoperation mode (direct control)"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: false

  teleoperation:
    enabled: true
    active_device: "so101_leader"
    devices:
      - name: "so101_leader"
        type: "leader_arm"
        port: "/dev/ttyUSB0"
        calib_file: "$(env HOME)/.calibrate/so101_leader_calibrate.json"
```

**Launch command:**
```bash
# Teleop mode (with auto recording)
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop record:=true
```

#### 2. model_inference Mode (High-Frequency Position Control)

**Use for:** End-to-end imitation learning models (ACT, pi0, Diffusion Policy)

**Characteristics:**
- High-frequency control (50-100Hz)
- Low latency (1-3ms)
- Direct topic-based position commands
- Reactive, fluid movements

**Configuration:**
```yaml
robot:
  default_control_mode: "model_inference"

  control_modes:
    model_inference:
      description: "High-frequency end-to-end control mode (ACT/pi0)"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: true
        pipelines:
          policy:
            model_path: models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515
            deployment: cpu
            execution_mode: monolithic # Or distributed
            request_timeout: 5.0
            default_task: ""
            runtime_options: {}
      executor:
        type: topic
        mode: model_inference
        inference_pipeline: policy
        queue_size: 100
        watermark_threshold: 50
        control_frequency: 20.0
```

**Launched controllers:**
- `arm_position_controller` (JointGroupPositionController)
- `gripper_position_controller` (ForwardCommandController)

The robot recording-level `lerobot_norm_mode` controls conversion between LeRobot
action/observation units and `ros2_control` radians. `range_m100_100` uses
arm `[-100,100]` and gripper `[0,100]`; `degrees` uses centered degrees for
arm joints while joints listed in `joints.gripper` keep `[0,100]` open/close
semantics.

Observation `align` settings serve both offline data processing and live inference, but their time limits have
different meanings:

- `tol_ms`: timestamp alignment tolerance for `strategy: asof`, used by both offline resampling and live sampling;
  values less than or equal to `0` fall back to `hold`. `hold` and `drop` do not use it.
- `max_age_ms`: maximum sample age accepted by live inference, measured from the node's local receipt clock so it
  cannot be bypassed by rewinding a request timestamp. Missing samples and histories containing only future-dated
  samples are always rejected; values greater than `0` also reject stale samples. These cases return
  `observation_not_ready` instead of running the model with zero padding.

```yaml
contract:
  observations:
    - key: observation.images.top
      align:
        strategy: hold
        tol_ms: 1500
        max_age_ms: 500
```

**Command interface:**
```bash
# Arm position commands
ros2 topic pub /arm_position_controller/commands std_msgs/msg/Float64MultiArray "data: [1.0, 2.0, 3.0, 4.0, 5.0]"

# Gripper position commands
ros2 topic pub /gripper_position_controller/commands std_msgs/msg/Float64MultiArray "data: [0.5]"
```

#### 2. moveit_planning Mode (Trajectory Planning)

**Use for:** Planning-based models (VoxPoser, VLM, goal-conditioned policies)

**Characteristics:**
- MoveIt integration (OMPL/Pilz planners)
- Time-parameterized trajectories
- Action-based execution with monitoring
- Collision avoidance support

**Configuration:**
```yaml
robot:
  default_control_mode: "moveit_planning"

  control_modes:
    moveit_planning:
      description: "MoveIt trajectory planning mode"
      controllers:
        - joint_state_broadcaster
        - arm_trajectory_controller
        - gripper_trajectory_controller
```

**Launched controllers:**
- `arm_trajectory_controller` (JointTrajectoryController)
- `gripper_trajectory_controller` (JointTrajectoryController)

**Command interface:**
```bash
# List available actions
ros2 action list

# Execute trajectory via action
ros2 action send_goal /arm_trajectory_controller/follow_joint_trajectory control_msgs/action/FollowJointTrajectory "{...}"
```

### Overriding Control Mode at Runtime

Control mode can be overridden via command line:

```bash
# Use default mode from config file
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm

# Override to teleop mode
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=teleop

# Override to model_inference mode
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=model_inference

# Override to moveit_planning mode
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=moveit_planning
```

### Mode Selection Decision Guide

```
What type of model are you using?
│
├─ Human teleoperation (leader arm, gamepad, VR)
│  └─ Use teleop mode
│     ├─ Real-time direct control
│     ├─ Zero-latency passthrough (< 5ms)
│     └─ Built-in safety filters
│
├─ End-to-end imitation learning (ACT, pi0, Diffusion)
│  └─ Use model_inference mode
│     ├─ Model outputs high-frequency position streams (50-100Hz)
│     ├─ Needs minimal latency (< 5ms)
│     └─ No trajectory planning required
│
└─ Planning-based (VoxPoser, VLM, goal-conditioned)
   └─ Use moveit_planning mode
      ├─ Model outputs sparse waypoints or goals
      ├─ Needs collision avoidance
      ├─ Requires MoveIt integration
      └─ Time-parameterized trajectories important
```

### Complete Configuration Example

```yaml
robot:
  name: so101_single_arm
  type: so101
  robot_type: so_101

  # Control mode management
  default_control_mode: "model_inference"  # Can be overridden via command line

  control_modes:
    teleop:
      description: "Human teleoperation mode (direct control)"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: false

    model_inference:
      description: "High-frequency end-to-end control mode (ACT/pi0)"
      controllers:
        - joint_state_broadcaster
        - arm_position_controller
        - gripper_position_controller
      inference:
        enabled: true
        pipelines:
          policy:
            model_path: models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515
            deployment: cpu
            execution_mode: monolithic # Or distributed
            request_timeout: 5.0
            default_task: ""
            runtime_options: {}
      executor:
        type: topic
        mode: model_inference
        inference_pipeline: policy

    moveit_planning:
      description: "MoveIt trajectory planning mode (VoxPoser/VLM)"
      controllers:
        - joint_state_broadcaster
        - arm_trajectory_controller
        - gripper_trajectory_controller

  # Unified joint configuration (DRY principle)
  joints:
    arm: ["1", "2", "3", "4", "5"]
    gripper: ["6"]
    all: ["1", "2", "3", "4", "5", "6"]

  # Hardware configuration
  ros2_control:
    hardware_plugin: so101_hardware/SO101SystemHardware
    port: /dev/ttyACM0
    calib_file: $(env HOME)/.calibrate/so101_follower_calibrate.json
    reset_positions:
      "1": 0.0813
      "2": 3.7905

  # Teleoperation configuration
  teleoperation:
    enabled: true
    active_device: "so101_leader"
    devices:
      - name: "so101_leader"
        type: "leader_arm"
        port: "/dev/ttyUSB0"
        calib_file: "$(env HOME)/.calibrate/so101_leader_calibrate.json"

  # Peripherals (cameras, sensors)
  peripherals:
    - type: camera
      name: top
      driver: opencv
      index: 0
      width: 640
      height: 480
      fps: 30

  # ML contract
  contract:
    observations:
      - key: observation.images.top
        topic: /camera/top
        peripheral: top
        image:
          resize: [480, 640]
    actions:
      - key: action
        topic: /arm_position_controller/commands  # Changes based on mode
        ros_type: std_msgs/msg/Float64MultiArray
        names: ["1", "2", "3", "4", "5", "6"]
```

### How Mode Switching Works

1. **Configuration Phase:**
   - `robot.launch.py` reads `default_control_mode` from YAML
   - Can be overridden via `control_mode:=xxx` command line argument
   - Validates mode exists in `control_modes` section

2. **Controller Spawning:**
   - Only controllers listed in the selected mode are spawned
   - Ensures no controller conflicts (same joint can't be controlled by multiple controllers)

3. **Action Dispatch Integration:**
   - `action_dispatch` node reads current mode from `robot_config`
   - Instantiates appropriate executor (TopicExecutor or ActionExecutor)
   - Provides unified API for upstream inference services

### Troubleshooting Control Modes

#### Mode not switching

**Problem:** Command line override not taking effect

**Solution:** Ensure `control_mode` parameter is correctly spelled:
```bash
# Correct
ros2 launch robot_config robot.launch.py control_mode:=moveit_planning

# Incorrect (typo)
ros2 launch robot_config robot.launch.py control_mode:=moveit_planing
```

#### Controller not starting

**Problem:** Controllers fail to activate

**Solution:** Check controller configuration in `so101_hardware/config/so101_controllers.yaml`:
```bash
# Verify controller exists
ros2 control list_controllers

# Check controller configuration
cat src/so101_hardware/config/so101_controllers.yaml | grep -A 10 "arm_trajectory_controller"
```

#### Action server not available

**Problem:** `FollowJointTrajectory` action not found in moveit_planning mode

**Solution:** Ensure trajectory controllers are active:
```bash
# List active controllers
ros2 control list_controllers | grep trajectory

# Should see:
# arm_trajectory_controller[joint_trajectory_controller/JointTrajectoryController] active
# gripper_trajectory_controller[joint_trajectory_controller/JointTrajectoryController] active
```

For more details, see:
- [action_dispatch README](../action_dispatch/README.md) - Detailed executor documentation
- [docs/architecture.md](../../docs/architecture.md) - System architecture overview


## Usage

### Launching the Robot

```bash
# Launch with real hardware
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm

# Launch with simulation
ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm use_sim:=true

# Distributed inference, single-host debug. The YAML policy pipeline must be distributed.
# Terminal 1: Edge
ros2 launch robot_config robot.launch.py config_path:=/absolute/path/to/so101_single_arm_distributed.yaml control_mode:=model_inference use_sim:=true
# Terminal 2: Cloud
ros2 launch inference_service cloud_inference.launch.py pipeline_id:=policy model_path:=/absolute/path/to/policy_bundle deployment:=cpu

# Distributed inference — cross-machine (Device only launches Edge; Cloud runs separately on GPU server)
# Device side:
ros2 launch robot_config robot.launch.py config_path:=/absolute/path/to/so101_single_arm_distributed.yaml control_mode:=model_inference use_sim:=true
# GPU server (set same ROS_DOMAIN_ID):
# ros2 launch inference_service cloud_inference.launch.py pipeline_id:=policy model_path:=/absolute/path/to/policy_bundle deployment:=cuda
```

Relative `model_path` values resolve only against an absolute `WORKSPACE` environment variable. The pipeline ID
derives default action, reset, health, and distributed topics. With multiple pipelines,
`executor.inference_pipeline` must select one explicitly.

### Validating Configuration

```bash
# Validate a robot config file with Python directly
python3 src/ros2/ros2_ws/src/robot_config/robot_config/scripts/validate_config.py \
    src/ros2/ros2_ws/src/robot_config/config/robots/so101_single_arm.yaml
```

## Camera Drivers

### USB Cameras (via `usb_cam`)

```yaml
- type: camera
  name: usb_cam
  driver: opencv  # Uses usb_cam package
  index: 0  # USB device index (/dev/video0)
  width: 640
  height: 480
  fps: 30
  pixel_format: bgr8  # bgr8, rgb8, mono8, yuyv, etc.
```

**安装 (Install):**
```bash
# Ubuntu
sudo apt install ros-humble-usb-cam
# openEuler
sudo dnf install ros-humble-usb-cam
```

### RealSense D400 Series (via `realsense2_camera`)

```yaml
- type: camera
  name: rs_cam
  driver: realsense  # Uses realsense2_camera package
  serial_number: "12345678"  # Device serial (optional)
  width: 640
  height: 480
  fps: 30
  depth_width: 640
  depth_height: 480
  depth_fps: 30
  enable_pointcloud: false
  align_depth: false
```

**安装 (Install):**
```bash
# Ubuntu
sudo apt install ros-humble-librealsense2*
# openEuler
sudo dnf install ros-humble-librealsense2*
```

## Camera Calibration

Camera intrinsics can be stored in the standard ROS2 location:

```yaml
- type: camera
  name: top
  driver: opencv
  index: 0
  width: 640
  height: 480
  camera_info_url: file://$(env HOME)/.ros/camera_info/top_camera.yaml
```

Calibration files can be created using the standard ROS2 camera calibration tools:

```bash
ros2 run camera_calibration cameracalibrator \
  --size 8x6 \
  --square 0.024 \
  image:=/camera/top/image_raw
```

## tensormsg Integration

The robot_config integrates with tensormsg contracts by allowing observations to reference peripherals by name:

```yaml
# In robot_config
peripherals:
  - type: camera
    name: top
    width: 640
    height: 480

# In contract section
contract:
  observations:
    - key: observation.images.top
      topic: /camera/top
      peripheral: top  # Auto-fills width, height, fps from peripheral
```

When the contract is loaded, it will automatically include the camera metadata from the peripheral definition.


The old `robot_interface` package used LeRobot's Robot class directly. This package replaces it with:

1. **No LeRobot dependency in ROS2 layer**: Uses ros2_control directly
2. **ros2_control native**: Standard ROS2 hardware interface
3. **Existing ROS2 camera drivers**: Uses `usb_cam` and `realsense2_camera` packages
4. **Single YAML configuration**: All hardware defined in one place

The two example configurations from `robot_interface` have been manually migrated to:
- `config/robots/so101_single_arm.yaml` (from `single_arm_banana.yaml`)
- `config/robots/so101_dual_arm.yaml` (from `dual_arms_pencil.yaml`)

## Troubleshooting

### Camera not opening

Check USB permissions:
```bash
ls -l /dev/video*
sudo chmod 666 /dev/video0
```

Or add user to `video` group:
```bash
sudo usermod -a -G video $USER
```

### RealSense camera not found

Install librealsense2:
```bash
# Ubuntu
sudo apt install librealsense2-utils librealsense2-dev
sudo apt install ros-humble-librealsense2*

# openEuler
sudo dnf install librealsense2-utils librealsense2-devel
sudo dnf install ros-humble-librealsense2*
```

Check camera is connected:
```bash
realsense-viewer
```

### Calibration file not found

Make sure the path is correct and starts with `file://`:
```yaml
camera_info_url: file:///home/user/.ros/camera_info/top.yaml
```

## References

- [usb_cam GitHub](https://github.com/ros-drivers/usb_cam) - USB camera driver for ROS2
- [realsense-ros GitHub](https://github.com/realsenseai/realsense-ros) - Intel RealSense ROS2 wrapper

## License

Apache-2.0
