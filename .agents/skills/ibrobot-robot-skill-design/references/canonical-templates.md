# Canonical Skill Templates

Every template below must additionally carry the `description:` contract (see `contract-schema.md`). It is omitted here for brevity; never ship a skill without it.

## When to Read

- Step 7 (Implement After Confirmation) is being executed
- You need a concrete YAML starting point for a new skill

## Named Pose Skill

```yaml
<skill_name>:
  primitive_sequence:
    - primitive_name: move_to_named_pose
      pose_name: observe_table
```

## Anchored Cartesian Pattern

Use for "move around the observation pose".

```yaml
<skill_name>:
  primitive_sequence:
    - primitive_name: move_to_named_pose
      pose_name: observe_table
    - primitive_name: move_relative_ee
      motion_direction: up
      motion_distance: 0.04
    - primitive_name: move_relative_ee
      motion_direction: down
      motion_distance: 0.08
    - primitive_name: move_relative_ee
      motion_direction: up
      motion_distance: 0.04
    - primitive_name: move_to_named_pose
      pose_name: observe_table
```

## Request-Parameterized Relative Motion

```yaml
<skill_name>:
  primitive_sequence:
    - primitive_name: move_relative_ee
      motion_direction_from_request: true
      motion_distance_from_request: true
```

## Joint Wave Gesture

```yaml
<skill_name>:
  description:
    summary: "Wave hello or goodbye with the wrist (casual greeting gesture)."
    category: social_greeting
    when_to_use: ["greet someone", "say hi or bye"]
    do_not_use:
      - condition: "agree or say yes"
        instead_use: nod_yes
    aliases_zh: ["打招呼", "挥手"]
    motion_scope: [wrist]
    anchor_pose: home
    intensity: moderate
    duration_sec_estimate: 8.0
    requires_motion_params: false
    rule_entry: true
  primitive_sequence:
    - primitive_name: move_to_joint_positions
      joint_positions:
        "1": 0.02
        "2": 0.54
        "3": -0.82
        "4": -0.18
        "5": 0.02
      duration_sec: 2.0
    - primitive_name: move_through_joint_positions
      trajectory_template:
        type: single_joint_wave_v1
        waypoint_duration_sec: 0.05
        active_waypoint_count: 16
        repeat_count: 3
        base_pose:
          "1": 0.02
          "2": 0.54
          "3": -0.82
          "4": -0.18
          "5": 0.02
        joint: "5"
        amplitude: 0.35
        workspace_limits:
          model: so101_arm_v1
          points:
            ee:
              x: [-0.32, 0.25]
              y: [-0.42, 0.00]
              z: [0.05, 0.55]
    - primitive_name: move_to_joint_positions
      joint_positions:
        "1": 0.02
        "2": 0.54
        "3": -0.82
        "4": -0.18
        "5": 0.02
      duration_sec: 2.0
```
