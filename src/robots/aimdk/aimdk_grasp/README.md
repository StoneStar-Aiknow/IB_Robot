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

## What replaced the standalone `x2_grasp_msgs`

| Standalone interface | Replacement | Why |
|---|---|---|
| `RobotFeedback.msg` | the runtime's JointState views above | the runtime is the only reader of vendor state; joint faults are part of its status |
| `RegisteredObservation.msg` | an in-process value inside the grasp node | observation and planning run in one node, so nothing crosses a process boundary |
| `CaptureRGBD.srv` | none needed | it existed to bridge the host observation node (which needed vendor messages) and the planning container; with vendor-free feedback both run in the container |
| `ExecuteGrasp.action` | not migrated | motion goes through `aimdk_robot`; nothing runs on the motion-control unit |

Camera transforms (`torso_from_camera`, `rgb_from_depth`) come from TF, which
the platform publishes.
