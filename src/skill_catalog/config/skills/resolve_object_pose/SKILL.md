# resolve_object_pose

Query the semantic map for a named object and return a stand-off navigation target `[x, y, theta_degrees]`.

This skill does not drive navigation. The Agent chains `resolve_object_pose` with `nav_abs_coordinate` to complete a "go to object" workflow.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `target_name` | string | yes | — | Manual semantic label (e.g. `banana`) |
| `stand_off_distance_m` | number | no | `0.2` | Distance from object to stand-off position |

## Usage

The skill returns `[x, y, theta_degrees]` where `theta` is in degrees (0 = map x-axis, positive = counterclockwise). The stand-off position is computed from the object position and the robot's current map-frame TF, so the robot approaches from its current side.

Agent workflow example:

```
robot-skill plan-workflow --text "去 banana 旁边" --workflow-json [
  {"skill_name": "resolve_object_pose", "target_name": "banana"},
  {"skill_name": "nav_abs_coordinate", "x": "<from previous>", "y": "<from previous>", "yaw": "<from previous>"}
]
```
