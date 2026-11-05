# pose_composer — natural-language arm poses for the AgiBot X2

Say *"双手向前伸直"*, *"做一个弯曲手臂的姿势"* or *"右手摸头顶"* and the X2 moves its
arms there. No per-pose trajectory is written in advance.

```
sentence ─► language model ─► pose description ─► solver ─► safety ─► path ─► runtime joint.trajectory
            (spec.SYSTEM_PROMPT)  (per arm: directions     (joint    (limits,   (planner,   (FollowJointTrajectory,
                                   or "hand to body point") angles)  clearance)  via poses)   replayed on the arm stream)
```

The language model never writes joint angles. It describes each arm either by
**directions** (`upper` = shoulder→elbow, `fore` = elbow→wrist, body frame) or by a
**body point** (`hand_to` one of `spec.LANDMARKS`, with an optional `offset_cm`
and `elbow` hint). The solver owns the kinematics; every frame is projected into
the safe set (joint travel, ≥2 cm from the body, 3 cm at the hands, gravity
torque), and the path between frames is checked and, when the straight path
would brush the body, routed through a via pose. The runtime validates the
trajectory once more against its own `vendor.arm.limits`.

## How the representation was chosen

In simulation (official X2 MuJoCo model, 33 instructions, 3 samples each, an
image judge calibrated against independent ratings), with DeepSeek-V4.1-Flash:

| Representation | T1–T3 weighted | held-out T6 | raw output safe | p50 |
|---|---|---|---|---|
| joint angles written by the model | 59–64 % | – | 85–87 % | 7–10 s |
| directions only | 82 % | 62 % | 100 % | 4.7 s |
| hand positions only | 79 % | 71 % | 100 % | 5.1 s |
| **directions or body point (this package)** | **92 %** | **79 %** | **100 %** | **3.3 s** |
| fixed pose library (reference) | 86 % | 25 % | 86 % | 2.6 s |

Known gaps: palm orientation is not described (palms-together reads as
"hands crossed"); the X2 shoulder barely adducts, so hands cannot reach the
ears or the chin squarely and such poses are reported as approximate.

## With the Agent

`robot_config` `aimdk_x2_free_pose` adds the `free_pose` skill to the X2 gesture
deployment. The planner puts the user's words for the pose in the step's
`target_name`; nothing else about the step changes.

```
ibrobot_agent ──(plan starts with free_pose)──► ~/compose  (ComposeArmPose)   composed while the plan is prepared,
     │                                                                          shown as a pose_preview event; a refused
     ▼                                                                          or unreachable pose is answered here
Gateway ─► pose_composer executor ──────────────► ~/execute (ExecuteArmPose)   runs that composition (or composes again
                                                                                 and says so), path from the measured pose
                                                                                 ─► runtime joint.trajectory
```

A stop request is forwarded through the Gateway; the runtime holds the arms
where they are. Only a pose that is the plan's first step is composed with
the plan; a later one is composed when it runs.

```bash
ros2 run aimdk_robot aimdk_fetch_x2_mjcf            # official model into ~/.cache/ibrobot (MulanPSL-2.0)
export ROS_DOMAIN_ID=<not the robot's> ROS_LOCALHOST_ONLY=1
AIMDK_MOCK_PHYSICS_MJCF=<scene.xml> ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=aimdk_x2_free_pose control_mode:=named_motion use_sim:=true authorize_motion:=true
ros2 run ibrobot_agent ibrobot_agent_chat --robot-scope aimdk_x2_skills
```

The composer's model is the `pose_composer.llm` block (OpenAI-compatible,
keys only by `api_key_env`).

## Requirements

The workspace venv's `mujoco` and `scipy`; the official model fetched as above;
the OmniHand runtime profile, which declares `joint.trajectory`,
`vendor.arm.limits` and the measured rest pose.
