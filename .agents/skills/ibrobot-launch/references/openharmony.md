# OpenHarmony Launch Workflow

This path applies to RoboOH/OpenHarmony boards such as RoboPi and BQ3588HM. Before any board
operation, load and follow `oh-constraints`. Use `oh-access` for HDC/SSH connection details.

## Non-Negotiable Board Rules

- Do not run `.shrc_local`, `scripts/setup.sh`, `colcon build`, or `ibrobot-build` on the board.
- Build the release on the Ubuntu host with `oh-build-roboframe`, then deploy it to
  `/data/roboframe`.
- Use POSIX `sh` syntax and `.` instead of `source` in board commands.
- Do not assume Bash, `/usr/bin/env`, systemd, a writable root filesystem, Gazebo, or RViz.
- Write logs and pidfiles under `/data/local/tmp` or another writable `/data` path.
- ROS/Python inference requires `/data/roboframe/scripts/robooh_1.0.1.env`; it intentionally sets
  board library paths and `LD_PRELOAD`.

## 1. Verify the Deployed Runtime

Run on the board:

```sh
test -f /data/roboframe/scripts/robooh_1.0.1.env
test -d /data/roboframe/install
. /data/roboframe/scripts/robooh_1.0.1.env
ros2 pkg list | grep -E 'robot_config|inference_service|hardware_mock|so101_hardware'
```

If the release or package is missing, stop. Build with `oh-build-roboframe` and deploy with
`oh-access`; never compensate by copying host `install/` files or `libs/lerobot/src` manually.

## 2. Set the Board Runtime Environment

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

## 3A. OpenHarmony YAML Mock Simulation Test

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

## 3B. Full Real-Hardware Launch

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

## 3C. OpenHarmony as Distributed Inference Side

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

## Remote OpenHarmony Execution

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
