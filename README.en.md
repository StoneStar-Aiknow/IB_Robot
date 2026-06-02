# IB-Robot

> IB-Robot (Intelligence Boom Robot): An integrated embodied AI framework merging the Hugging Face LeRobot ecosystem with ROS 2.

## New: OpenClaw Social Control Support!

IB-Robot now fully supports remote social control via the **[OpenClaw](https://github.com/openclaw/openclaw)** AI Agent! Whether in **Gazebo simulation** or on a **real SO-101 robot arm**, you can interact with the robot using natural language through Feishu, QQ, Discord, and more.

|                            Simulation Demo                            |                             Real Robot Demo                            |
| :---------------------------------------------------------------------: | :----------------------------------------------------------------------: |
| ![Simulation Demo](docs/pictures/openclaw_sim.gif) | ![Real Robot Demo](docs/pictures/openclaw_real.gif) |

---

## Project Positioning

**User Guide**: [IB-Robot Usage Guide](https://pages.openeuler.openatom.cn/embedded/docs/build/html/master/features/embodied_ai/index.html)

IB-Robot is an **integrated robot development framework** designed to bridge the gap between the Hugging Face LeRobot machine learning ecosystem and the ROS 2 robotics middleware. It provides a complete toolchain from data collection and training to real-world deployment.

### Core Integration Capabilities

| Dimension | LeRobot Ecosystem | ROS 2 Ecosystem | IB-Robot Solution |
|-----------|-------------------|-----------------|-------------------|
| **Data Flow** | Episode-based | Topic-based | Contract-driven bidirectional real-time conversion |
| **Temporal** | Discrete Steps | Continuous RT Stream | Automated alignment and high-frequency interpolation |
| **Control** | End-to-end Policies | Hierarchical Planning | **Dual-mode control (ACT vs. MoveIt)** |
| **Deployment** | Python Scripts | ROS 2 Nodes | Distributed edge-cloud collaborative deployment |

### Platform Support

The mainline workflow now supports three runtime platforms:

| Platform | Role | Current Support | Typical Scenarios |
|----------|------|-----------------|-------------------|
| **Ubuntu 22.04** | Host / developer machine | Full setup, build, simulation, recording, MoveIt, and inference workflows | Gazebo simulation, data collection, single-host inference, edge-cloud debugging |
| **openEuler Embedded 24.03** | Edge board | Setup, clean build, and board-side runtime verified | NPU inference, real robot control, recording client |
| **OpenHarmony 5.1** | Edge board | Board runtime workflow, HDC debugging, minimal inference workspace build helper, and LeRobot patch profile support | BQ3588HM board inference, HDC/SSH debugging |

## System Architecture

![IB-Robot Architecture](docs/pictures/architecture.png)

### Architecture Deep Dive

IB-Robot builds an end-to-end closed-loop system from perception and decision-making to execution, realizing seamless integration between ML and robotics:

1. **Multimodal Perception & Collection**:
   - **Low-level Perception**: Unified access to multiple cameras (USB/RealSense), LiDAR, and microphones via ROS 2 Drivers.
   - **Diverse Collection**: Supports **VR controllers, Xbox controllers, and mobile IMU** teleoperation for expert demonstration data.
2. **Protocol Conversion Hub (tensormsg)**:
   - Serving as the hub, tensormsg handles the bidirectional conversion between `ros_msg` and `tensor`, ensuring data type safety and consistency via the Contract mechanism.
3. **Inference & R&D Service**:
   - Supports various VLA (Vision-Language-Action) models (e.g., SmolVLA, Pi0.5) and end-to-end policy models (e.g., ACT, Diffusion Policy). The system **auto-detects backends** and launches them on demand based on the control mode.
4. **Unified Action Dispatcher**:
   - Acts as the robot's "cerebellum". In ACT mode, it handles **Action Chunking** scheduling and high-frequency interpolation; in planning mode, it interfaces with **MoveIt 2** for constrained trajectory execution, providing unified `RobotStatus` reporting.
5. **Configuration-Driven Center (robot_config)**:
   - Realizes "Spec-driven robot behavior". Defines joints, controller modes, and sensor parameters via a single YAML, supporting one-click switching between simulation and real-world environments.

---

## Repository Structure

```text
IB_Robot/                           # Main Workspace
├── .gitmodules                     # Git submodule configuration
├── README.md                       # Main documentation (Chinese)
├── README.en.md                    # Main documentation (English)
├── LICENSE                         # Apache 2.0 License
├── config.json                     # AI Agent configuration (AtomGit API tokens, etc.)
│
├── .agents/                        # AI Agent configuration directory
│   └── skills/                     # AI Agent skills (see .agents/skills/README.md)
│
├── libs/                           # External dependencies
│   ├── lerobot/                    # [Submodule] LeRobot training framework
│   └── atomgit_sdk/                # AtomGit API SDK
│
├── src/                            # Core source packages
│   ├── robot_config/               # System master control, specs, and launch entry
│   ├── action_dispatch/            # Unified action dispatcher (Dual-mode)
│   ├── task_dispatch/              # Task scheduling and dispatch service
│   ├── tensormsg/                  # LeRobot ↔ ROS 2 protocol conversion hub
│   ├── ibrobot_msgs/               # Unified system interfaces (Message/Action)
│   ├── dataset_tools/              # Dataset collection & conversion (Episode Recorder)
│   ├── robot_teleop/               # Teleoperation (Leader Arm/Xbox controller)
│   ├── robot_description/          # Unified URDF/SRDF/MJCF model descriptions
│   ├── lekiwi_description/         # Lekiwi chassis URDF/Mesh model descriptions
│   ├── robot_moveit/               # MoveIt 2 motion planning integration
│   ├── robot_navigation/           # Navigation package
│   ├── inference_service/          # Multi-model inference & deployment service
│   ├── so101_hardware/             # SO-101 motor driver interface
│   ├── lekiwi_hardware/            # Lekiwi chassis hardware driver interface
│   ├── hardware_mock/              # Hardware mock interface
│   ├── omni_wheel_controller/      # Omni-wheel controller plugin
│   ├── pymoveit2/                  # [Submodule] MoveIt2 Python interface
│   ├── rosclaw/                    # [Submodule] OpenClaw social control integration
│   ├── sim_models/                 # Simulation scene models (Gazebo/MuJoCo)
│   ├── model_utils/                # Model utility library
│   ├── attention_viz/              # Attention visualization tool
│   ├── voice_asr_service/          # Voice recognition service
│   └── workflows/                  # CI/CD configuration
│
├── docs/                           # Detailed architecture docs and dev guides
│   ├── pictures/                   # Architecture diagrams and demo GIFs
│   └── videos/                     # Demo videos (source files)
├── scripts/                        # Setup and verification scripts
└── build/                          # Build output (Auto-generated)
```

---

## Environment Initialization (First-time Setup)

**Important: This step only needs to be run once after the initial clone.**

### 0. System Requirements

- **OS**: Three-platform workflow support is available: Ubuntu host machines plus openEuler Embedded and OpenHarmony edge boards
- **ROS Version**: ROS 2 Humble
- **Python**: >= 3.10, defaults to the system Python. **Do NOT run in an active Conda environment to avoid dynamic library conflicts.**
- **Accelerator**: Supports NVIDIA GPU, Ascend 310B, Ascend 310P, or CPU-only fallback.

### 1. One-click Initialization

Run `./scripts/setup.sh`. This script automates heavy operations:

1. **Submodule Sync**: Runs `git submodule update --init --recursive` to download core source code.
2. **Platform and Accelerator Detection**: Detects Ubuntu / openEuler Embedded / OpenHarmony and NVIDIA GPU / Ascend 310B / 310P / CPU-only environments.
3. **ROS 2 Installation**: Detects and installs ROS 2 Humble and colcon tooling when missing.
4. **System Dependencies**: Installs C++ build tools, `nlohmann-json`, and other hardware driver dependencies via the system package manager.
5. **Virtual Environment (venv)**: Creates a `venv` directory in the root to isolate ML dependencies while reusing system `rclpy` through `--system-site-packages`.
6. **ML Stack Installation**: Installs `lerobot`, hardware dependencies, and the ROS-compatible NumPy 1.26.x series.
7. **Environment Verification**: Verifies `rosdepc`, `colcon`, `rclpy`, `lerobot`, and NumPy compatibility.

---

## Development Workflow

### 1. Load Environment

After opening a new terminal, load the environment from the project root:

```bash
cd /path/to/IB_Robot
source .shrc_local
```

> **Note**: `.shrc_local` automatically activates the venv, loads the ROS 2 environment, and sources the workspace `install/setup.sh`. You must run this in every new terminal, otherwise `ros2` commands and Python packages will not be available.

After the first build, also load the workspace overlay:

```bash
source install/setup.sh
```

### 2. Assign Domain ID

To avoid conflicts with other ROS 2 users on the same network, assign a unique Domain ID. **Required in every new terminal**:

```bash
export ROS_DOMAIN_ID=<Unique ID between 0-232>
```

> **Note**: When running across machines, all participating machines must use the **same `ROS_DOMAIN_ID`**.

### 3. Build Project

Run the unified build script after any code changes:

```bash
./scripts/build.sh --clean
```

*Note: `build.sh` now focuses on environment loading and workspace builds only. Python environment, editable `lerobot` installation, and NumPy compatibility are handled centrally by `setup.sh`.*

---

## Running Guide

All operations are triggered via the unified entry point `robot.launch.py` in the `robot_config` package. "Edge board" below refers to devices running **openEuler Embedded** or **OpenHarmony**.

Before any scenario, load the environment and set a unique `ROS_DOMAIN_ID`. When running across machines, all participating machines must use the **same `ROS_DOMAIN_ID`**.

```bash
source .shrc_local
export ROS_DOMAIN_ID=<Unique ID between 0-232>
```

For detailed sub-module documentation, see:

| Document | Description |
| :--- | :--- |
| [`src/inference_service/README.md`](src/inference_service/README.md) | Inference service architecture, single-host/distributed deployment, and NPU/GPU cloud node launch |
| [`docs/OpenHarmony_thirdparty_pytorch_validation.md`](docs/OpenHarmony_thirdparty_pytorch_validation.md) | OpenHarmony board `thirdparty_pytorch` / `skh-run` setup, deployment, and validation progress |
| [`src/robot_moveit/README.md`](src/robot_moveit/README.md) | MoveIt planning control, `/cmd_pose` usage, and headless launch |
| [`src/dataset_tools/README.md`](src/dataset_tools/README.md) | Episodic recording, `record_cli` usage, and `bag_to_lerobot` dataset conversion |

### I. Ubuntu Simulation

#### 1. Ubuntu Launch Simulation (Simulation & Controllers Only)

Suitable for verifying Gazebo, cameras, controllers, and basic ROS 2 topology without model inference.

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    use_sim:=true \
    with_inference:=false
```

#### 2. Ubuntu Launch Simulation with Model Inference

Explicitly switch to `model_inference` mode to control the simulated arm using local inference.

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    use_sim:=true
```

#### 3. Ubuntu Launch MoveIt Planning Control (Simulation)

This scenario starts MoveIt and RViz by default, exposing `/cmd_pose` for sending target poses.

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning \
    use_sim:=true
```

To disable RViz on headless or board environments:

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning \
    use_sim:=true \
    moveit_display:=false
```

Send a pose command to move the arm:

```bash
ros2 topic pub /cmd_pose geometry_msgs/Pose "{
  position: {x: 0.15, y: 0.0, z: 0.25},
  orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}
}" --once
```

Check end-effector pose feedback:

```bash
ros2 topic echo /robot_status/ee_pose
```

### II. Real Hardware

#### 1. Edge Board Launch MoveIt Planning Control (Real Hardware)

Same as Ubuntu MoveIt usage but with `use_sim` disabled for real robot control.

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning
```

The control interface uses the same topics:

```bash
ros2 topic pub /cmd_pose geometry_msgs/Pose "{
  position: {x: 0.15, y: 0.0, z: 0.25},
  orientation: {x: 0.0, y: 0.0, z: 0.707, w: 0.707}
}" --once
```

### III. Distributed Inference Deployment

The following uses the distributed deployment mode: the robot side runs only the Edge proxy node, while the compute side runs `cloud_inference.launch.py` separately.

#### 1. Ubuntu Single-host Debug (Edge + Cloud on Same Machine)

Suitable for development and integration testing, running both sides on one Ubuntu machine.

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    execution_mode:=distributed \
    use_sim:=true \
    cloud_local:=true
```

#### 2. Ubuntu Simulation + Edge Board NPU Inference

The Ubuntu host handles simulation and Edge-side pre/post-processing; the edge board handles cloud-side pure inference. Both machines must be on the same LAN with matching `ROS_DOMAIN_ID`.

**Ubuntu Host (Simulation + Edge)**

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=model_inference \
    execution_mode:=distributed \
    use_sim:=true
```

**Edge Board (NPU Cloud Node)**

```bash
ros2 launch inference_service cloud_inference.launch.py \
    policy_path:=/path/to/model \
    device:=npu
```

For a GPU server, replace `device:=npu` with `device:=cuda`.

Quick verification of the distributed pipeline:

```bash
ros2 node list | grep -E 'act_inference|pure_inference'
ros2 topic list | grep -E 'preprocessed|inference/action'
ros2 topic hz /inference/action
```

#### 3. OpenHarmony Board as Compute Side (RK3588)

- **Host-side build entry**: Use `scripts/openharmony/build_ibrobot_oh_custom.sh`. The script prepares the OpenHarmony cross-compilation directory and applies `series.openharmony-5.1-musl.txt` patches to `lerobot` during packaging.
- **Board-side PyTorch runtime**: Uses `thirdparty_pytorch`'s `test/skh-run.tar.gz`, deployed to `/data/local/skh-run`. IB_Robot's OpenHarmony inference node uses this runtime at the process level without polluting `/data/out/bin/ros2`.
- **Offline weights**: The ACT policy uses `resnet18` + `ResNet18_Weights.IMAGENET1K_V1`. If the board has no internet, pre-place `resnet18-f37072fd.pth` in `/root/.cache/torch/hub/checkpoints/`.
- **Current status**: BQ3588HM CPU inference verified; NPU inference pipeline is in progress.

### IV. Dataset Recording

Episodic recording always consists of two parts:

1. `robot.launch.py` starts the `episode_recorder` recording server
2. `ros2 run dataset_tools record_cli` starts the interactive recording client

`record_visualizer:=rerun` only launches the Rerun visualization sidecar; it does not replace `record_cli`.

#### 1. Ubuntu Recording Server + Ubuntu Recording Client

**Without Rerun**

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    use_sim:=false
```

**With Rerun**

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    record_visualizer:=rerun \
    use_sim:=false
```

**Client (same machine, different terminal)**

```bash
ros2 run dataset_tools record_cli
```

After starting `record_cli`, enter a task description to begin recording. Press Enter to end the current episode early.

#### 2. Ubuntu Recording Server + Edge Board Recording Client

This mode separates robot control from recording. The Ubuntu host runs the recording server; the edge board runs `record_cli`. Both must share the same `ROS_DOMAIN_ID`.

**Ubuntu Recording Server (optional Rerun)**

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    use_sim:=false
```

To enable visualization, add to the server command:

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=teleop \
    record:=true \
    record_mode:=episodic \
    record_visualizer:=rerun \
    use_sim:=false
```

**Edge Board Recording Client**

```bash
ros2 run dataset_tools record_cli
```

After recording, convert the episodic dataset to LeRobot format:

```bash
ros2 run dataset_tools bag_to_lerobot \
    --bags-dir ~/rosbag/episodes/so101_single_arm \
    --robot-config src/robot_config/config/robots/so101_single_arm.yaml \
    --out /path/to/output_dataset
```

For bag directory structure, `dataset.yaml` metadata, and more conversion options, see `src/dataset_tools/README.md`.

---

## Parameter Reference

| Parameter | Description | Default |
| :--- | :--- | :--- |
| `robot_config` | Robot config name (matches YAML in `config/robots/`) | `so101_single_arm` |
| `config_path` | Absolute path to config file (overrides `robot_config`) | Empty |
| `use_sim` | Use Gazebo simulation mode | `false` |
| `control_mode` | Override default mode (`model_inference` / `moveit_planning` / `teleop`) | From YAML |
| `with_inference` | Force enable/disable inference service (empty = auto-detect) | Empty |
| `execution_mode` | Inference execution mode (`monolithic` / `distributed`) | `monolithic` |
| `cloud_local` | In distributed mode, launch Cloud node on the same machine | `false` |
| `with_moveit` | Force enable/disable MoveIt core (empty = auto-detect) | Empty |
| `moveit_display` | Launch MoveIt RViz interface | `true` |
| `record` | Enable recording pipeline | `false` |
| `record_mode` | Recording mode (`continuous` / `episodic`) | `continuous` |
| `record_visualizer` | Recording visualizer (`none` / `rerun`) | `none` |
| `auto_start_controllers` | Automatically activate controllers on start | `true` |

---

## AI Agent Skills

IB-Robot includes built-in AI programming agent skills to help Claude Code, Gemini CLI, OpenCode, and other AI Agents better understand the project architecture and development workflow. For available skills, see [.agents/skills/README.md](.agents/skills/README.md).

### config.json Configuration

`config.json` stores configuration for AI Agents, currently used for AtomGit API integration:

```json
{
  "atomgit": {
    "token": "$ATOMGIT_TOKEN",
    "owner": "openEuler",
    "repo": "IB_Robot",
    "baseUrl": "https://api.atomgit.com"
  }
}
```

**Getting an AtomGit Personal Access Token**:

1. Visit <https://atomgit.com> and log in
2. Click your avatar (top-right) → Settings
3. Find "Access Tokens"
4. Click "New Token", grant `repo` and `pull_request` permissions
5. **Copy and save** the token immediately (shown only once)

Set the environment variable by adding the following to your `~/.zshrc` or `~/.bashrc`:

```bash
export ATOMGIT_TOKEN="your_token_here"
```

### Supported Agents

All clients compatible with the Agent Skills standard will automatically scan `.agents/skills/`. See [agentskills.io](https://agentskills.io).

---

## OpenClaw Social Control & Remote AI Agent

IB-Robot deeply integrates the [OpenClaw](https://github.com/openclaw/openclaw) AI Agent framework with the [RosClaw](https://github.com/PlaiPin/rosclaw) bridge, enabling remote robot control via natural language through Feishu, QQ, Discord, or Slack.

> **Acknowledgements**: Thanks to the OpenClaw team for the powerful AI agent framework, and RosClaw for the ROS 2 bridge solution.

### 1. Robot-Side Configuration (RosClaw & Bridge)

The robot side requires a WebSocket bridge driver and a discovery service.

- **Pull submodules**:
  Ensure the latest RosClaw submodule source is pulled:
  ```bash
  git submodule update --init --recursive
  ```
- **Install system dependencies**:
  ```bash
  # rosbridge_suite is required for WebSocket communication
  sudo apt-get update && sudo apt-get install -y ros-humble-rosbridge-suite
  ```
- **Start the robot**:
  First launch the robot (simulation or real hardware):
  ```bash
  # use_sim:=true for simulation, false for real hardware
  ros2 launch robot_config robot.launch.py robot_config:=so101_single_arm control_mode:=model_inference use_sim:=true with_inference:=false
  ```
- **Start the social bridge**:
  This project includes RosClaw as a submodule at `src/rosclaw`. Run the one-click script:
  ```bash
  # Auto-compiles the submodule and starts rosbridge_websocket, rosapi, and discovery nodes
  ./scripts/start_rosclaw.sh
  ```
  After launch, WebSocket service will be available on port `9090`.

### 2. Control-Side Configuration (OpenClaw)

- OpenClaw serves as the robot's "brain" and "frontend", connecting to social apps and invoking LLMs to understand commands.

> **Important**: Before using OpenClaw to control the robot, make sure the `ROS_DOMAIN_ID` on the OpenClaw side matches the robot side. Otherwise, OpenClaw will not be able to discover ROS 2 topics and services. You need to tell OpenClaw the corresponding `ROS_DOMAIN_ID` during the session.

- **Install OpenClaw**:
  Use the official quick-install script (requires Node.js 22+):
  ```bash
  # Install OpenClaw CLI
  npm install -g openclaw

  # Run the onboarding wizard to configure your LLM (e.g., GLM-4/5 or GPT-4)
  openclaw onboard
  ```
- **Install RosClaw plugin**:
  ```bash
  # Run from the IB_Robot root directory
  openclaw plugins install ./src/rosclaw/extensions/openclaw-plugin
  ```
- **Configure robot connection**:
  ```bash
  # Set the robot WebSocket address (replace with actual IP)
  openclaw config set plugins.entries.rosclaw.config.rosbridge.url "ws://<ROBOT_IP>:9090"
  ```
- **Inject IB-Robot specific skills**:
  To help the AI accurately understand units (radians) and vision topics, deploy the skill specification:
  ```bash
  mkdir -p ~/.openclaw/workspace/skills/ibrobot-control
  cp ./docs/ib_robot_social_skill.md ~/.openclaw/workspace/skills/ibrobot-control/SKILL.md
  ```
- **Start Gateway**:
  ```bash
  openclaw gateway
  ```

### 3. Interaction Examples

Once connected, you can interact via the web UI (`http://localhost:18789`) or through bound Feishu, QQ, or Discord:

- *"Show me the robot's current capabilities"* — Retrieve all sensor topics.
- *"Move the robot arm to the home position"* — AI automatically converts angles to **radians** based on the skill document.
- *"What's on the table?"* — AI captures and analyzes an image from `/camera/top/image_raw`.
- *"Pick up the bottle on the table"* — AI triggers IB-Robot's `DispatchInfer` action.

---

## FAQ

### 1. Residual Controllers

If controllers fail to start or ports are busy, run the cleanup script:

```bash
./scripts/cleanup_ros.sh
```

### 2. Shared Memory (SHM) Errors

If you see `RTPS_TRANSPORT_SHM Error`, try cleaning the cache:

```bash
sudo rm -rf /dev/shm/fastrtps_*
export ROS_LOCALHOST_ONLY=1
```

### 3. Simulation Window Not Displaying

If the simulation starts but no visualization window appears (e.g., MuJoCo/Gazebo), check the `DISPLAY` environment variable. On Wayland or some remote desktop environments, you may need to set it manually:

```bash
export DISPLAY=:1
```

---

**Maintainer**: IB-Robot Team\
**User Guide**: <https://pages.openeuler.openatom.cn/embedded/docs/build/html/master/features/embodied_ai/index.html>\
**Project Home**: <https://atomgit.com/openEuler/IB_Robot>\
**Feedback**: <https://atomgit.com/openEuler/IB_Robot/issues>
