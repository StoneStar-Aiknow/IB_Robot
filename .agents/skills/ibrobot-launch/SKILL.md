---
name: ibrobot-launch
description: "Launch IB-Robot on Ubuntu/openEuler source workspaces or deployed OpenHarmony boards. Use for 'launch', 'run robot', 'start simulation', 'start system', '启动机器人', '运行仿真', '测试推理', 'test inference', '遥操作调试', 'teleop', 'start so101', 'RoboPi launch', or 'OpenHarmony launch'. Distinguishes native Ubuntu/openEuler build-and-launch from OpenHarmony /data/roboframe runtime launch."
---

# IB-Robot Launch Skill

Use this skill to launch IB-Robot only after selecting the runtime platform. Ubuntu/openEuler and
OpenHarmony use different build, environment, shell, deployment, simulation, and logging rules.

## Platform Routing (Required)

Before running any command, determine which platform owns the ROS process:

| Runtime platform | Source of packages | Required workflow |
|---|---|---|
| Ubuntu / openEuler | Local IB_Robot source workspace | `ibrobot-build` -> source `.shrc_local` -> launch |
| OpenHarmony board | Deployed `/data/roboframe` release | read `oh-constraints` -> verify deployment -> source board env -> launch |

Do not infer the platform only from CPU architecture. Ask one short question if the user has not
said whether the process should run in the local workspace or on an OpenHarmony board.

### Shared ROS Requirements

- Choose a `ROS_DOMAIN_ID` for the session and use the same value for every participating process.
  Repository examples commonly use `42`; OpenHarmony validation examples commonly use `51`.
  These are examples, not protocol constants.
- Distributed processes must also use the same `RMW_IMPLEMENTATION`. Prefer the value already used
  by the deployment or test. OpenHarmony examples use `rmw_cyclonedds_cpp` unless the release guide
  explicitly requires another implementation.
- Keep environment setup and the `ros2` command in the same shell invocation.
- `robot_config/robot.launch.py` remains the unified full-system entry point on both platforms, but
  platform capabilities still differ.
- Simulation backend selection belongs in the robot YAML as `robot.simulation.platform`. For Mock,
  select a YAML containing `platform: mock` and launch with `use_sim:=true`.

## Ubuntu / openEuler Workflow

This path applies to a native source checkout on Ubuntu or openEuler, including openEuler Embedded
systems that have a normal glibc source workspace. It does not apply to OpenHarmony musl boards.

### 1. Build First

Load `ibrobot-build` and build from the repository root. For a targeted launch change:

```bash
source .shrc_local && colcon build --symlink-install --merge-install --packages-select robot_config
```

Use `source .shrc_local && ./scripts/build.sh` when the change affects the complete workspace.
Build and launch must be separate tool calls so a failed build stops the workflow.

### 2. Launch in One Shell

```bash
source .shrc_local && export ROS_DOMAIN_ID=<domain-id> && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py <args>
```

Required components:

- `.shrc_local`: ROS 2, Python venv, LeRobot `PYTHONPATH`, and workspace defaults.
- `ROS_DOMAIN_ID`: DDS isolation and discovery.
- `install/setup.zsh`: refreshed workspace overlay after the build.

### Common Native Launches

Simulation:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=true
```

### Native YAML Mock Simulation Test

Use Mock when the goal is to validate the observation/action contract or end-to-end inference
without a physical robot, camera, Gazebo, or MuJoCo. Mock is contract simulation, not physics
simulation: it publishes synthetic images and joint states, subscribes to configured actions, and
immediately feeds accepted actions back into joint state.

Only `control_mode:=model_inference` is supported. First select or create a dedicated robot YAML whose
SSOT contains:

```yaml
robot:
  simulation:
    platform: mock
```

Do not edit a shared production YAML in place merely to run a test. Prefer a dedicated Mock config
installed with `robot_config`, or select an external YAML with `config_path`. The recommended unified
launch then needs only `use_sim:=true`; it reads the Mock backend from the YAML and starts the
contract Mock, inference, and action dispatch together:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py \
    robot_config:=<mock-robot-config> \
    control_mode:=model_inference \
    use_sim:=true
```

For an external YAML, replace `robot_config:=<mock-robot-config>` with
`config_path:=/absolute/path/to/mock.yaml`; do not pass both.

The selected robot YAML must contain a valid model inference pipeline and model bundle path. If the
bundle or deployment differs from the YAML, update or select a test YAML instead of inventing
unsupported launch arguments.

In another shell with the same environment and domain, verify the Mock contract:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 topic hz /camera/top/image_raw

source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 topic echo /joint_states --once

source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 action info /inference/policy/dispatch
```

Trigger one end-to-end inference:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 action send_goal /inference/policy/dispatch \
    ibrobot_msgs/action/DispatchInfer \
    "{obs_timestamp: {sec: 0, nanosec: 0}, prompt: '', inference_id: 'native-mock-001', deadline: {sec: 0, nanosec: 0}}"
```

Pass criteria:

- Launch output contains `hardware_mock contract_mock active`.
- Synthetic camera frames and `/joint_states` are published at the YAML contract rates.
- `/inference/policy/dispatch` exists, accepts the goal, and returns `success: true`.
- Action output is consumed by Mock and reflected in a subsequent `/joint_states` message.
- No physical camera, serial device, controller manager, Gazebo, or navigation process is required.

Real hardware:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=false
```

MoveIt simulation:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning \
    use_sim:=true
```

Native Ubuntu normally owns GUI simulation and visualization. Do not attempt Gazebo or RViz on a
headless OpenHarmony board.

## OpenHarmony Workflow

This path applies to RoboOH/OpenHarmony boards such as RoboPi and BQ3588HM. Before any board
operation, load and follow `oh-constraints`. Use `oh-access` for HDC/SSH connection details.

### Non-Negotiable Board Rules

- Do not run `.shrc_local`, `scripts/setup.sh`, `colcon build`, or `ibrobot-build` on the board.
- Build the release on the Ubuntu host with `oh-build-roboframe`, then deploy it to
  `/data/roboframe`.
- Use POSIX `sh` syntax and `.` instead of `source` in board commands.
- Do not assume Bash, `/usr/bin/env`, systemd, a writable root filesystem, Gazebo, or RViz.
- Write logs and pidfiles under `/data/local/tmp` or another writable `/data` path.
- ROS/Python inference requires `/data/roboframe/scripts/robooh_1.0.1.env`; it intentionally sets
  board library paths and `LD_PRELOAD`.

### 1. Verify the Deployed Runtime

Run on the board:

```sh
test -f /data/roboframe/scripts/robooh_1.0.1.env
test -d /data/roboframe/install
. /data/roboframe/scripts/robooh_1.0.1.env
ros2 pkg list | grep -E 'robot_config|inference_service|hardware_mock|so101_hardware'
```

If the release or package is missing, stop. Build with `oh-build-roboframe` and deploy with
`oh-access`; never compensate by copying host `install/` files or `libs/lerobot/src` manually.

### 2. Set the Board Runtime Environment

Every board terminal or remote command must repeat the environment setup:

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=<domain-id>
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

For Houmo HMM deployments, additionally load the packaged HMM environment:

```sh
. /data/roboframe/scripts/setup/houmo_hmm_env.sh
```

Only add backend-specific environment scripts required by the chosen deployment. Do not load HMM
environment for RKNN or Torch runs.

### 3A. OpenHarmony YAML Mock Simulation Test

OpenHarmony does not run Gazebo or MuJoCo. Its supported Mock simulation is selected through the
robot YAML and provides synthetic observations plus an action feedback loop without accessing the
camera, serial port, controller manager, or physical robot.

The deployed robot YAML must set `robot.simulation.platform: mock`. Prefer installing a dedicated
named Mock config with the release rather than changing the production hardware config. The preferred
end-to-end path is the same unified launch used on native platforms: `use_sim:=true` selects the
backend declared by the YAML and starts Mock, the configured inference pipeline, and action dispatch
together:

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch robot_config robot.launch.py \
  robot_config:=<mock-robot-config> \
  control_mode:=model_inference \
  use_sim:=true
```

If the Mock YAML is deployed outside the package share directory, use its actual absolute
`config_path` instead of `robot_config`; do not assume a `/data/roboframe/config` directory exists.

For a temporary board-side test YAML, `sim_platform:=mock` is also a supported explicit override.
It takes precedence over `robot.simulation.platform` in the selected YAML. The verified PI0.5 HMM
Mock launch pattern is:

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
. /data/roboframe/scripts/setup/houmo_hmm_env.sh
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch robot_config robot.launch.py \
  config_path:=/data/local/tmp/pi05_hmm_so101_mock.yaml \
  control_mode:=model_inference \
  use_sim:=true \
  sim_platform:=mock
```

Use this override when the temporary YAML is intentionally reused across backends or does not embed
`simulation.platform: mock`. For a maintained robot configuration, keep the backend in YAML and omit
`sim_platform` so the configuration remains the SSOT.

The deployed YAML must point to a model bundle that exists on the board and select a valid named
deployment. Load the corresponding backend environment before launch when required, for example
`houmo_hmm_env.sh` for an HMM deployment.

From another board shell with the same environment, domain, and RMW implementation, verify:

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 topic hz /camera/top/image_raw
ros2 topic echo /joint_states --once
ros2 action info /inference/policy/dispatch
```

Trigger one end-to-end inference:

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 action send_goal /inference/policy/dispatch \
  ibrobot_msgs/action/DispatchInfer \
  "{obs_timestamp: {sec: 0, nanosec: 0}, prompt: '', inference_id: 'oh-mock-001', deadline: {sec: 0, nanosec: 0}}"
```

Pass criteria are identical to native Mock: active contract Mock, correctly published synthetic
observations, registered inference action, accepted goal with `success: true`, and action feedback in
`/joint_states`. Also confirm no real hardware device was opened.

### 3B. Full Real-Hardware Launch

Use the deployed absolute YAML path and disable simulation:

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch robot_config robot.launch.py \
  config_path:=/data/roboframe/install/robot_config/share/robot_config/config/robots/so101_single_arm.yaml \
  control_mode:=model_inference \
  use_sim:=false
```

Before allowing motion, verify all of the following:

- The YAML selects the intended named deployment and `execution_mode`.
- The model bundle, calibration file, serial device, and camera devices exist at the configured
  absolute paths.
- Required kernel drivers are enabled; SO-101 commonly needs `/dev/ttyACM0` and `CONFIG_USB_ACM=y`.
- Camera and robot observations match the policy contract.
- The user has explicitly requested real-hardware motion and the workspace is safe.

Do not silently replace unavailable hardware with simulation when the user requested a board test.

### 3C. OpenHarmony as Distributed Inference Side

Use the same domain and RMW implementation as the Ubuntu/openEuler peer. For a board-side cloud
inference service:

```sh
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=51
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
ros2 launch inference_service cloud_inference.launch.py \
  pipeline_id:=policy \
  model_path:=/data/models/<policy-bundle> \
  deployment:=<deployment-name>
```

Also verify `ROS_LOCALHOST_ONLY=0`, network reachability, matching bundle digest, deployment name,
and deployment fingerprint on both sides.

### Remote OpenHarmony Execution

Prefer SSH for long-running launch logs and HDC for recovery or when SSH is unavailable. Follow
`oh-access` instead of assuming a board IP.

HDC one-off command pattern:

```bash
HDC_BIN=hdc
HDC_TARGET=<board-ip>:8710
"$HDC_BIN" -t "$HDC_TARGET" shell \
  '. /data/roboframe/scripts/robooh_1.0.1.env && export ROS_DOMAIN_ID=51 && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && ros2 node list'
```

SSH automation must disable any configured interactive `RemoteCommand` and TTY allocation:

```bash
ssh -o RemoteCommand=none -o RequestTTY=no -o BatchMode=yes root@<board-ip> \
  '. /data/roboframe/scripts/robooh_1.0.1.env && export ROS_DOMAIN_ID=51 && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && ros2 node list'
```

Ask for the target when it is unknown. Do not hard-code a board address into the skill workflow.

## Launch Parameters

| Parameter | Meaning | Notes |
|---|---|---|
| `robot_config` | Named YAML under the package config directory | Convenient in a source workspace or installed package |
| `config_path` | Absolute robot YAML path | Prefer this on OpenHarmony deployments |
| `control_mode` | `teleop`, `model_inference`, `moveit_planning`, etc. | Must exist in the selected YAML |
| `use_sim` | Enable a configured simulation backend | Ubuntu/openEuler only for Gazebo/MuJoCo; use hardware mock for minimal board inference tests |
| `sim_platform` | Explicit CLI override of the YAML backend | Supported for temporary test configs; maintained configs should use `simulation.platform` in YAML |
| `with_inference` | Override inference auto-detection | Prefer YAML-driven behavior unless debugging |
| `inference_pipeline` | Select named pipeline | Must match the YAML and manifest contract |

## Verification After Launch

Run checks with the same platform environment, domain, and RMW settings as the launched process:

```bash
# Ubuntu/openEuler
source .shrc_local && export ROS_DOMAIN_ID=<domain-id> && ros2 node list
```

```sh
# OpenHarmony
. /data/roboframe/scripts/robooh_1.0.1.env
export ROS_DOMAIN_ID=<domain-id>
export RMW_IMPLEMENTATION=<rmw-implementation>
ros2 node list
```

Check the interfaces required by the chosen mode, not only process existence. Typical checks include
`ros2 action info /inference/policy/dispatch`, `ros2 topic list`, controller state, camera topics,
and one real inference goal.

## Process Management and Logs

### Ubuntu / openEuler

- Foreground launch is preferred while testing.
- ROS logs normally live under `~/.ros/log/`.
- Use repository cleanup tooling only from the source workspace and only when stale ROS processes
  are confirmed.

### OpenHarmony

- Foreground SSH is preferred for diagnosis because HDC PTY output may truncate long logs.
- There is no systemd. For background execution, use a pidfile and writable log path:

```sh
LOG=/data/local/tmp/ibrobot-launch.log
PIDFILE=/data/local/tmp/ibrobot-launch.pid
ros2 launch <package> <launch-file> <args> >"$LOG" 2>&1 &
echo $! >"$PIDFILE"
```

- Check with `kill -0 "$(cat "$PIDFILE")"`; stop with `kill`, wait, then use `kill -9` only if
  necessary. Remove the pidfile after shutdown.
- Do not use `systemctl`, `pkill -f` as the primary owner check, or `ps | grep <full-command>`.

## Troubleshooting

### Nodes Cannot Discover Each Other

Confirm every process uses identical `ROS_DOMAIN_ID` and `RMW_IMPLEMENTATION`. For distributed
operation, also confirm `ROS_LOCALHOST_ONLY=0` and network reachability.

### `ModuleNotFoundError: lerobot`

- Ubuntu/openEuler: source `.shrc_local` in the same shell.
- OpenHarmony: source `/data/roboframe/scripts/robooh_1.0.1.env`; if the staged LeRobot tree is
  absent, rebuild and redeploy the official release rather than copying source manually.

### Package or Launch File Not Found

- Ubuntu/openEuler: build first, then source `install/setup.zsh` in the launch shell.
- OpenHarmony: verify `/data/roboframe/install` and the package with `ros2 pkg list`. Rebuild with
  the default `oh-build-roboframe` package set if it is missing.

### OpenHarmony Native Process Crashes or RKNN Import Fails

Confirm the board env was loaded, the artifact is aarch64/musl compatible, the deployment exists in
`inference_manifest.json`, and the required NPU runtime is installed. For RKNN, verify:

```sh
python3 -c "from rknnlite.api import RKNNLite; print('RKNNLite OK')"
```

### Controller or Hardware Device Missing

Check the selected control mode, YAML paths, calibration, permissions, `/dev/ttyACM*`, camera
devices, and required OpenHarmony kernel configuration before changing launch logic.

## Quick Reference

| Task | Ubuntu / openEuler | OpenHarmony |
|---|---|---|
| Build | `source .shrc_local && ./scripts/build.sh` | Host: use `oh-build-roboframe`; never build on board |
| Environment | `source .shrc_local && export ROS_DOMAIN_ID=<id> && source install/setup.zsh` | `. /data/roboframe/scripts/robooh_1.0.1.env; export ROS_DOMAIN_ID=<id>; export RMW_IMPLEMENTATION=<rmw>` |
| Mock simulation | Select YAML with `simulation.platform: mock`, then `robot.launch.py robot_config:=<mock-config> use_sim:=true` | Same YAML-driven entry, or use `config_path:=<temp.yaml> use_sim:=true sim_platform:=mock` for an explicit temporary override |
| Real hardware | `robot.launch.py ... use_sim:=false` | `robot.launch.py config_path:=/data/roboframe/install/... use_sim:=false` |
| Logs | `~/.ros/log/` | `/data/local/tmp/` or another writable `/data` path |
| Access | Local shell | `oh-access` via SSH/HDC |

## Handoff to Other Skills

- Native build or build failure: `ibrobot-build`
- Native environment/import problem: `ibrobot-env`
- OpenHarmony runtime constraints: `oh-constraints` (mandatory before board operations)
- OpenHarmony host cross-build/package: `oh-build-roboframe`
- OpenHarmony connection/deployment: `oh-access`
- OpenHarmony third-party ROS package: `oh-cross-build-ros-pkg`
- Missing board kernel driver: `oh-rebuild-kernel`
- RKNN model conversion/package: `rknn-convert`
- Houmo HMM model packaging/runtime details: `hmm-convert`
