# aimdk_grasp — AgiBot X2 grasp pipeline

Observation, scene estimation and grasp planning for the OmniHand X2, migrated
from the standalone X2 grasp workspace. The package reads the robot only through
the public interfaces `aimdk_robot` serves and calls the grasp models through the
typed perception services that `aimdk_x2_grasp.yaml` declares. It never imports
vendor (`aimdk_msgs`) types, so it runs anywhere the runtime's topics reach —
including the Orin planning container, which cannot load the vendor's prebuilt
message libraries.

The migration is incremental; this README records what has landed.

## Robot feedback

`aimdk_grasp.feedback` assembles the joint pose at an image time from the three
JointState views the runtime publishes, bound by interface ID from the public
description (`aimdk_grasp.feedback_ros.subscribe_feedback`):

| Interface | Endpoint | Provides |
|---|---|---|
| `joint.state` | `/joint_states` | both arms |
| `joint.head_state` | `/aimdk/head_joint_states` | head (camera placement) |
| `joint.hand_state` | `/aimdk/hand/joint_states` | the ten driven joints of each OmniHand |
| `runtime.status` | `/runtime_status` | health: `ACTIVE` with no faults; expires after 5 s without an update |

For an image stamp, each view's nearest sample must lie within 50 ms of it and
the views within 100 ms of each other, or the observation is refused. Arm and
head joint faults (vendor `error_code`) are reported by the runtime as
`<group> joint fault: <joint>=<code>` and make the feedback unhealthy; they are
not re-derived here. OmniHand `faultcode`s are not runtime faults: a finger
pressing an object reports one for as long as it grips.

The runtime publishes these views at the profile's `joint.state` rate (50 Hz).
Composite views use the oldest HAL header stamp among the groups they contain;
`joint.head_state` contains only the head, so a lagging waist or leg cannot label
a fresh camera pose with an older time as `joint.body_state` would. The X2 keeps
the HAL, the compute pack and the head camera on one PTP-synchronized clock (the
HAL leaves `meas_stamp` zero), so joint states and images are compared within one
clock domain, as in the standalone pipeline; the nearest sample is at most 10 ms
from the image.

## Planning model

`aimdk_grasp.planning_model.build_planning_model()` generates the planning URDF
of the X2 Ultra upper body with the left OmniHand from the official description
in `aimdk_description`; no hand-edited URDF is kept. It preserves the
conventions the pipeline's collision data and joint commands were built on:
hand joints are `R_*` aliases (native position times a fixed sign), `R_palm`
keeps the pipeline's tool-frame orientation, and the distal finger joints are
independent because the pipeline drives them from the measured coupling.

`config/x2_ultra/` holds the cuRobo configuration for that model. cuRobo 0.8
itself is installed by `scripts/setup.sh` on the cuda accelerator, from the
audited pure-Python wheel in `third_party/wheels/curobo/`; it compiles its CUDA
kernels at run time through NVRTC, so the same wheel serves the X2 Orin.

| File | Content |
|---|---|
| `robot.yml` | cuRobo `robot_cfg` (spheres, exclusions, locked joints, c-space) and the generator options (`planning_model`) |
| `hand_collision_cover.json` | certified sphere covers of six hand links (checked against the mesh file, its URDF origin and its scale) and the hand contact exclusions |
| `hand_collision_reference.json` | hand/wrist exclusions of the reference sphere fit |
| `torso_collision_cover.json` | certified torso cover, checked against the mesh file, its URDF origin and normalized scale; regenerate with `fit_x2_torso_cover <file>` |
| `leg_obstacle.json` | pelvis and left leg at the standing pose, for the fine collision check; regenerate with `export_x2_leg_obstacle <file>` |
| `provenance.json` | where each value comes from, and the measured coverage of the official meshes |

The collision data was fitted on the AimDK SDK hand and arm meshes of the
standalone workspace and is kept as is, with two exceptions: the shoulder roll
spheres are re-expressed into the official link frames (only the frame origin
differs), and the torso cover is refit on the official torso mesh, which has a
rear protrusion the SDK mesh lacks. The official meshes deviate from the SDK
ones by a few millimetres; `provenance.json` lists the per-link coverage.

The hand is the AimDK SDK hand (`palm_mount: legacy`, `thumb_mount: legacy`),
which the operator confirmed on the robot: the palm at the SDK position, 3 mm
closer to the wrist than the official model, and the four fingers and the thumb
on their SDK mounts. On the palm it is then exactly the hand of the GraspGen-X
gripper description, which the grasps are proposed for, and the hand collision
spheres sit where they were fitted (they are the SDK cover's, per link). The
official model's fingers are 0.53° about the wrist axis away from the SDK's and
its thumb roll axis ~75°; `"official"` builds it for comparison only.

Open item:

- **Right hand.** The right OmniHand is not in the collision model (the right
  arm is locked during planning), as in the standalone workspace.

## Robot model and collision spheres

`aimdk_grasp.robot_model.collision_robot_config()` loads `config/x2_ultra`,
verifies every cover against the description files it is bound to, and adds the
hand and assembly exclusions; it refuses a cover whose meshes changed, or whose
meshes the URDF moved or scaled inside their link.
`planning_urdf()` writes the generated URDF to a content-named file for cuRobo,
which does the collision checking on the GPU.

`aimdk_grasp.robot_spheres.RobotSphereModel` places the same spheres on the CPU
for given joint states (the scene's self filter and the planner's start state)
and bounds the left arm's and palm's reach (the scene crop and the grasp
candidate prefilter). `aimdk_grasp.hand.HandMapping` converts native left-hand
positions to the model's aliases and derives the unmeasured distal joints. Its
limits come from the planning model; the `open` / `close` presets come from the
deployed GraspGenX gripper description (`load_gripper_presets`), since grasp
candidates are generated for that hand shape.

The standalone workspace's exact CPU self- and scene-collision check is not
brought over: execution on the robot no longer calls it, and planning checks
run in cuRobo.

## Registration and scene estimation

`aimdk_grasp.registration.register_rgbd()` rectifies the head camera's RGB and
rasterizes each valid depth pixel's footprint into it with a nearest-surface
z-buffer, so depth steps are never averaged and sensor holes stay holes. The
kernel runs on NVIDIA Warp (`warp-lang`, installed by `scripts/setup.sh` with the
grasp dependencies) on the planning host's GPU (`device="cuda"`, about 3 ms per
1280x720 frame on an RTX 3090); Warp's CPU device runs the same kernel where no
GPU exists. Detection, segmentation and grasp generation run as perception
model services instead and can be deployed elsewhere, e.g. on Ascend. The X2's
head camera is mounted upside down, so the proposer shows the detector and the
segmenter the image turned upright (`proposer.upright_turns`, from the
image-time camera pose) and turns their boxes and masks back.

`aimdk_grasp.scene_estimation.estimate_scene()` turns the registered depth, the
target mask, the image-time camera pose and the robot's collision spheres into
the planner's torso-frame scene: the support plane as a cuboid, the target and
environment as voxel meshes, cropped to the left arm's reach, with the robot's
own body filtered out.

The planning model has no legs. Every fine validation scene also carries the
pelvis and left leg as one fixed mesh (`aimdk_grasp.leg_obstacle`, convex hulls of
the official collision meshes at the measured standing pose): a live approach
once swung the palm behind the torso into the left hip. The coarse scene gets it
as one box per hull, each side pulled in by 2 cm (`scene_estimation.leg_boxes`):
the mesh itself would cost every plan the hanging arm's 5-10 mm to the hip, and a
mesh query sees only near its surface, while a box has a distance inside it and
pushes a path out of the pelvis. A box's corners can stand out of its hull, so the
fine scene keeps the exact hulls and no boxes.

## Left-arm planner

`aimdk_grasp.planner.LeftArmPlanner` is the X2's cuRobo planner, the planning half
of its grasp backend for `manipulation_execution`'s grasp pipeline. It plans and
checks left-arm motion only, in `torso_link`, on CUDA: the right arm, head and
measured hand are locked into the model from feedback, and a change in them needs
a new planner. It holds two scene resolutions: the optimizer plans against the
coarse scene with a merged hand cover and a 2 cm activation distance, and every
returned state is checked against the fine scene with the full cover at zero
distance. No hand sphere may lie behind the torso's frontal plane. A segment the
fine check refuses says where (`LeftArmPlanner.violations`): the samples, the
links, the depth, and the deepest sphere's torso-frame centre, which tells the
table from the target, a neighbour or the robot's own body.

Grasp candidates arrive as GraspGenX gripper-base poses in the camera frame; the
deployed gripper description's `base_rotation` (`hand.load_gripper_description`)
turns them into `R_palm` goals, so the planner never assumes which hand the grasp
model was trained for. A plan is three timed segments (`trajectory.TimedTrajectory`):
an approach found by a batched contact screen and a goalset solve (its TrajOpt seed
lifts the hand up and out from where it hangs, then comes down into the approach pose
from 18 cm above, so the hand does not sweep through a low target on its way up), a
straight grasp line along the palm axis, and a vertical lift that leaves the carried target out of
its fine check (or a hold when the line cannot be planned). `warmup()` compiles and
captures every kernel and graph before the first request. A CUDA or Warp failure
raises `PlannerFault`, never a candidate failure, so the owner rebuilds the planner.

The planner is migrated from the standalone pipeline with its tuned constants
unchanged; the offline replay limits and the CPU sphere kinematics it carried for
analysis tools are not.

## Observations and revalidation

`aimdk_grasp.observation.RegisteredObservation` holds one registered frame, the
torso-frame camera pose at its image time and the feedback at that time, which
must be healthy and within 100 ms of the image. `ObservationBuffer` keeps the
newest observation and, separately, the newest current feedback: the request is
planned from the image, but starts from where the arm is now. Both are owned
copies, and reading them never refreshes their source time.

Before a plan is used, `aimdk_grasp.revalidation` compares it with what the robot
sees and measures now, and raises `ReobserveRequired` when it no longer holds:

| Check | Rejects |
|---|---|
| `require_fresh` | current feedback older than 100 ms, a scene older than 0.5 s, or an unhealthy runtime |
| `require_posture` | an arm, the head or the left hand moved more than 0.025 rad since planning |
| `require_stable_patch` | the target region changed in RGB (> 2 % of it) or in measured depth (> 5 %), or the camera geometry changed |

Depth counts only where both frames measured it: a transparent target's valid
pixels come and go between still frames.

## What replaced the standalone `x2_grasp_msgs`

| Standalone interface | Replacement | Why |
|---|---|---|
| `RobotFeedback.msg` | the runtime's JointState views above | the runtime is the only reader of vendor state; joint faults are part of its status |
| `RegisteredObservation.msg` | an in-process value inside the grasp node | observation and planning run in one node, so nothing crosses a process boundary |
| `CaptureRGBD.srv` | none needed | it existed to bridge the host observation node (which needed vendor messages) and the planning container; with vendor-free feedback both run in the container |
| `ExecuteGrasp.action` | not migrated | motion goes through `aimdk_robot`; nothing runs on the motion-control unit |

Camera transforms (`torso_from_camera`, `rgb_from_depth`) come from TF, which
the platform publishes.
