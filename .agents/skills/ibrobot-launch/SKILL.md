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

## Shared ROS Requirements

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

## Platform-Specific Workflows

Once the platform is determined, follow the corresponding workflow document for full commands:

- **Ubuntu / openEuler**: [references/ubuntu.md](references/ubuntu.md) — build first, then launch
  via `source .shrc_local && ros2 launch ...`. Covers simulation, Mock contract test, real hardware,
  and MoveIt planning.
- **OpenHarmony board**: [references/openharmony.md](references/openharmony.md) — verify deployed
  `/data/roboframe` release, source board env, then launch. Covers Mock test, real hardware,
  distributed inference, and remote SSH/HDC execution. **Mandatory**: read `oh-constraints` first.

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

- **Ubuntu/openEuler**: foreground launch preferred; logs under `~/.ros/log/`.
- **OpenHarmony**: foreground SSH preferred; no systemd — use pidfile + writable `/data` log path.

Full details (background execution, pidfile management, cleanup) in
[references/process-management.md](references/process-management.md).

## Troubleshooting

Common issues (node discovery, `ModuleNotFoundError: lerobot`, package not found, RKNN import
failure, hardware device missing) are documented in
[references/troubleshooting.md](references/troubleshooting.md).

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
