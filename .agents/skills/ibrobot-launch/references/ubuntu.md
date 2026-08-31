# Ubuntu / openEuler Launch Workflow

This path applies to a native source checkout on Ubuntu or openEuler, including openEuler Embedded
systems that have a normal glibc source workspace. It does not apply to OpenHarmony musl boards.

## 1. Build First

Load `ibrobot-build` and build from the repository root. For a targeted launch change:

```bash
source .shrc_local && ./scripts/build.sh -- --packages-select robot_config
```

Use `source .shrc_local && ./scripts/build.sh` when the change affects the complete workspace.
Never invoke raw `colcon build`; the build script applies layout and CMake settings itself.
Build and launch must be separate tool calls so a failed build stops the workflow.

## 2. Launch in One Shell

```bash
source .shrc_local && export ROS_DOMAIN_ID=<domain-id> && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py <args>
```

Required components:

- `.shrc_local`: ROS 2, Python venv, LeRobot `PYTHONPATH`, and workspace defaults.
- `ROS_DOMAIN_ID`: DDS isolation and discovery.
- `install/setup.zsh`: refreshed workspace overlay after the build.

## Common Native Launches

Simulation:

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=true
```

## Native YAML Mock Simulation Test

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

## Real Hardware Launch

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    use_sim:=false
```

## MoveIt Simulation

```bash
source .shrc_local && export ROS_DOMAIN_ID=42 && source install/setup.zsh && \
  ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm \
    control_mode:=moveit_planning \
    use_sim:=true
```

Native Ubuntu normally owns GUI simulation and visualization. Do not attempt Gazebo or RViz on a
headless OpenHarmony board.
