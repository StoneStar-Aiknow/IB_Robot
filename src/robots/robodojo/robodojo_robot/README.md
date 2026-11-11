# robodojo_robot

RoboDojo's dual-arm ARX X5 (Isaac Sim) as an IB-Robot robot, on the **`robodojo` simulation
platform**. This package registers the platform in `ibrobot.sim_platforms` and supplies its
embodiment, adapter and the X5 runtime profile. The runtime itself is the shared
[`robot_runtime_sim.direct_control`](../../../robot_runtime_sim/README.md) runtime: the engine,
session lifecycle and ROS services know nothing about RoboDojo.

| Path | Runs in | Role |
|---|---|---|
| `robodojo_robot/platforms.py` | anywhere (import-light) | `ROBODOJO_SPEC`: the `robodojo` platform (`world=None`: RoboDojo owns the world) |
| `robodojo_robot/embodiment.py` | launch | Starts `robot_runtime_sim`'s `direct_control_runtime` with this package's adapter (`robodojo_robot.adapter:create_adapter`); no ros2_control |
| `robodojo_robot/adapter.py` | ROS process | RoboDojo codec (`state.<side>_arm_joint_state`, `_ee_joint_state`, `_ee_pose`, `vision.<cam>.color` ↔ normalized frame / per-group targets), the policy model behind RoboDojo's ws server, platform-silence watchdog, head-camera overlay; RoboDojo's verdict → the session's platform result only |
| `robodojo_robot/policy_server.py` | ROS process | Hosts RoboDojo's own websocket `PolicyServer` (imported from `ROBODOJO_ROOT`) on loopback |
| `robodojo_robot/overlay.py` | ROS process | Table grid + gripper tips drawn from camera calibration (no scene content) |
| `robodojo_client/` | RoboDojo eval client (Isaac, py3.11, no ROS) | `deploy.eval_one_episode`: push observations with `episode_id`/`generation`, execute targets, answer IK with cuRobo, report the verdict, write per-episode tick ledgers (`artifacts.py`) |
| `profiles/arx_x5_dual.yaml` | — | Single source of groups, joints, TCP, cameras; `simulation.robodojo` holds the tick length, motion and adapter settings |

RoboDojo keeps its main loop (`main.py` → `run_eval`, platform pull). One RoboDojo episode is
one session of the runtime. Simulated time only advances while a motion runs; idle ticks are
not stepped, and `/clock` (which every consumer of the robot runs on) stands still. The robot
config is `robot_config/config/robots/robodojo_x5.yaml` (`simulation.platform: robodojo`).

```bash
export ROBODOJO_ROOT=<RoboDojo checkout>   # its conda env "RoboDojo"; CONDA_SH if conda is not on PATH
ros2 launch robodojo_robot runtime.launch.py profile:=arx_x5_dual [policy_port:=34960]   # runtime only
TASK=general_pickup EVAL_NUM=1 bash install/share/robodojo_robot/robodojo_client/run_eval_client.sh
python3 src/robot_runtime_sim/scripts/direct_control_smoke.py --group right_arm --out smoke.json --finish
```

`run_eval_client.sh` links the installed `robodojo_client/` files into
`$ROBODOJO_ROOT/XPolicyLab/policy/ibrobot_direct_control/` (re-linked when it points at another
checkout). Evaluation (admission, the verdict channel, the task instruction) is not part of this
package; an evaluation layer attaches to the runtime through its `extensions`.
