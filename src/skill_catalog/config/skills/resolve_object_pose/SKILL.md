# resolve_object_pose

Query the semantic map for a named object and return a stand-off navigation target `[x, y, theta_degrees]`.

This skill does not drive navigation. It queries the persisted semantic map and uses the current map-frame robot pose to calculate a stand-off target. The Agent can chain it with `nav_abs_coordinate` to complete a "go to object" workflow. It does not require online semantic-map frame processing to query an existing static map.

## Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `target_name` | string | yes | — | One of `banana`, `basket`, `paper ball`, `toy`, `ballpoint pen`, `grapes` |
| `stand_off_distance_m` | number | no | robot config | Navigation distance from the object in meters |

## Usage

The skill returns `[x, y, theta_degrees]` where `theta` is in degrees (0 = map x-axis, positive = counterclockwise). Use `--stand-off-distance-m` for a one-query override; when omitted, the skill uses `robot.semantic_mapping.target_watch.stand_off_distance_m`.

Agent workflow example:

```
robot-skill plan-workflow --text "去 banana 旁边" --workflow-json [
  {"skill_name": "resolve_object_pose", "target_name": "banana"},
  {"skill_name": "nav_abs_coordinate", "x": "<from previous>", "y": "<from previous>", "yaw": "<from previous>"}
]
```
