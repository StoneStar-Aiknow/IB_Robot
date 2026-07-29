---
name: ibrobot-robot-skill-design
description: "Use when designing, adding, modifying, or validating an IB-Robot embodied skill, Hermes/Agent robot action, social gesture, observation-pose motion, gripper action, MoveIt/relative end-effector action, or when the user asks '设计机器人skill', '新增机器人动作', '新增具身技能', '让机器人做一个动作', 'Hermes 调用机器人', 'catalog 暴露机器人 skill', '庆祝动作', '挥手', '点头', or '观察位置上下左右'."
---

# IB-Robot Robot Skill Design Guide

This skill is an interactive design workflow for adding or changing robot skills in IB-Robot. It prevents semantic mistakes such as implementing "move around the observation pose" as an unrelated joint-space gesture.

## Core Rule

Do not implement a robot skill before the semantic anchor and motion space are explicit.

If the request says "around X", "near X", "at X", "观察位置", "桌面位置", or "目标物体旁边", model the skill as an anchored Cartesian or named-pose sequence unless the user explicitly asks for a pure joint-space gesture.

## Safety Boundary

All real robot execution must keep this path:

```text
Hermes / Agent / CLI
  -> /embodied/execute_skill
  -> skill_executor
  -> safety_guard
  -> existing primitive executor
```

Do not add direct hardware, `/task_executor/*`, MoveIt, ros2_control, or controller calls unless the user explicitly asks for an architecture change and an architecture review is performed.

## Mandatory Interaction Flow

Before editing files, ask only the minimum questions needed. Prefer a single multiple-choice question when possible. If the user already provided enough information, summarize the inferred design and ask for confirmation.

### Step 1: Classify Intent

Classify the requested skill into one primary type:

| Type | Use When | Preferred Primitive Pattern |
|------|----------|-----------------------------|
| `named_pose_skill` | Move to a configured pose | `move_to_named_pose` |
| `anchored_cartesian_pattern` | Move around/near a semantic pose | `move_to_named_pose` + `move_relative_ee` |
| `current_pose_cartesian_pattern` | Move relative to current EE pose | `move_relative_ee` |
| `joint_wave_pattern` | Simple one-joint gesture | `move_through_joint_positions` via `single_joint_wave_v1` |
| `multi_joint_wave_pattern` | Coordinated joint gesture/dance | `move_through_joint_positions` via `wave_dance_v1` |
| `gripper_or_wrist_skill` | Open/close/rotate gripper | gripper/rotate primitives |
| `composite_skill` | Ordered mix of the above | explicit primitive sequence |

### Step 2: Identify Anchor

Ask or infer the anchor:

- `observe_table` for "观察位置", "看桌面", "观察位".
- `home` for "安全位置", "回 home", "收回".
- `zero` for "零位".
- current EE pose for "从当前位置".
- target/object pose only if the runtime already supports resolving that target through existing primitives.

If the anchor is a named pose, the generated skill should normally begin with `move_to_named_pose` and, if requested or safer, end at the same anchor.

### Step 3: Choose Motion Space

Use the smallest correct motion space:

- Spatial language like up/down/left/right/forward/backward: use `move_relative_ee`.
- Pose language like home/observe/zero: use `move_to_named_pose`.
- Gesture language like wave/nod/shake/dance: use joint trajectory templates.
- Gripper language: use gripper primitives.

Do not use joint trajectories to approximate a Cartesian semantic requirement unless no existing Cartesian primitive can express it.

### Step 4: Define Parameters

Collect only parameters relevant to the type.

For `anchored_cartesian_pattern`:

- `anchor_pose`
- direction sequence using valid directions from robot_config: `forward`, `backward`, `left`, `right`, `up`, `down`
- distance per step in meters
- whether to return to the anchor
- timeout and whether RViz/real robot verification is required

For joint gesture patterns:

- base joint pose
- active joint(s)
- amplitude and repeat count
- waypoint duration
- workspace limits
- whether the final waypoint returns to base pose

### Step 5: Author the Description Contract (Mandatory)

Every new or modified skill MUST carry a `description:` block in the SSOT YAML (co-located with its `primitive_sequence`). This block is the single source of truth for how an Agent/Hermes caller and the rule parser pick THIS skill over its near-synonyms. Omitting it is an architecture violation, not a style choice.

Required schema (validated by `robot_config.loader._validate_skill_description`):

```yaml
<skill_name>:
  description:
    summary: "<= 120 chars, intent-first, not a mechanical description>"
    category: <observation|recovery|gripper|translation|rotation|dance|social_greeting|social_affirmation|social_emotion|...>
    when_to_use: [<short phrases describing when to pick this skill>]
    do_not_use:                             # MANDATORY for any skill with near-synonyms
      - condition: "<when NOT to pick this one>"
        instead_use: <existing_skill_name>  # must be a real skill in the same config
    aliases_zh: [<中文触发词>]               # ALSO drives the rule parser (single keyword source)
    aliases_en: [<english triggers>]
    motion_scope: [<base|shoulder|elbow|wrist|gripper|arm>]
    anchor_pose: <named pose | none>         # must exist in embodied.named_poses unless 'none'
    intensity: <subtle|moderate|large>       # safety-relevant: large motions near people need care
    duration_sec_estimate: <float > 0>
    requires_motion_params: <bool>           # true if it needs motion_direction/distance from the caller
    rule_entry: <bool>                       # true exposes aliases_zh to the deterministic rule parser
```

Disambiguation rule (the whole point of this contract):

- Before finalizing a new skill, list its near-synonyms among existing skills and add one `do_not_use` entry per synonym redirecting to the right alternative.
- Conversely, add反向 redirects on the existing synonym skills pointing at the new one when the boundary changes.
- `aliases_zh` is the SSOT for Chinese trigger keywords and catalog aliases; do NOT duplicate keyword lists in `embodied_common.command_parser` hardcode.
- `extract_skill_aliases` injects `aliases_zh` into the deterministic rule parser only when `rule_entry: true` and `requires_motion_params: false`.
- `summary` must be intent-driven ("Wave hello/goodbye with the wrist"), not mechanical ("Sinusoidal joint-5 motion").
- `do_not_use.instead_use` must reference a skill that actually exists in the same config — the loader rejects dangling redirects.
- `duration_sec_estimate` must cover deterministic arm motion plus 1.0 second for an `open`/`closed`
  `initial_gripper_state` and 1.0 second for every explicit `open_gripper`/`close_gripper` primitive, with margin.

Example (greeting cluster disambiguation):

```yaml
wave_hello:
  description:
    summary: "Wave hello or goodbye with the wrist (casual greeting gesture)."
    category: social_greeting
    when_to_use: ["greet someone", "say hi or bye", "wave to a person"]
    do_not_use:
      - condition: "agree or say yes"
        instead_use: nod_yes
      - condition: "formal raise-hand greet at the observe pose"
        instead_use: greet_observe_raise
      - condition: "dance rhythmically"
        instead_use: dance_basic
    aliases_zh: ["打招呼", "挥手", "挥挥手", "再见", "嗨"]
    motion_scope: [wrist]
    anchor_pose: home
    intensity: moderate
    duration_sec_estimate: 8.0
    requires_motion_params: false
    rule_entry: true
```

### Step 6: Produce A Design Record

Before code changes, show a concise design record and ask for confirmation unless the user explicitly requested full autonomy.

Use this format:

```markdown
Skill Design
Name: <skill_name>
Intent: <natural-language intent>
Robot Config: <robot config name>
Anchor: <named pose/current pose/none>
Motion Space: <named_pose/cartesian_relative/joint_trajectory/gripper/composite>
Primitive Sequence:
- <primitive details>
Description Contract: summary / category / when_to_use / do_not_use redirects / aliases_zh / motion_scope / intensity
Catalog Exposure: <yes/no; catalog doc auto-synthesized from description>
Safety Path: robot-skill -> /embodied/execute_skill -> skill_executor -> safety_guard
Validation Plan:
- ruff check <files>
- pytest <tests>
- colcon build --packages-select <packages>
- optional RViz/real robot validate_skill + execute_skill + recover_safe_pose
```

### Step 7: Implement After Confirmation

For normal robot skill changes, edit the smallest necessary set:

- `src/robot_config/config/robots/<robot>.yaml` for SSOT skill definitions, the `description:` contract, and `allowed_skills`.
- `src/embodied_common/embodied_common/trajectory_templates.py` only if a reusable trajectory template is truly needed.
- `src/embodied_common/embodied_common/skill_templates.py` when adding supported skill names or template expansion logic.
- The authoritative skill description is the YAML `description:` block, auto-surfaced by the `robot-skill` catalog.
- Tests under the affected packages.
- README files when launch commands, catalog usage, or public behavior changes.

Do not add backward-compatibility aliases for removed skill names unless there is persisted data, an external consumer, or an explicit user requirement.

## Canonical Templates

Every template below must additionally carry the `description:` contract from Step 5. It is omitted here for brevity; never ship a skill without it.

### Named Pose Skill

```yaml
<skill_name>:
  primitive_sequence:
    - primitive_name: move_to_named_pose
      pose_name: observe_table
```

### Anchored Cartesian Pattern

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

### Request-Parameterized Relative Motion

```yaml
<skill_name>:
  primitive_sequence:
    - primitive_name: move_relative_ee
      motion_direction_from_request: true
      motion_distance_from_request: true
```

### Joint Wave Gesture

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

## Validation Checklist

Always validate the design before real robot execution:

- The skill name is in `planning_policy.allowed_skills` if the planner should be allowed to generate it. This allowlist does not control catalog exposure.
- Catalog entries come from the current robot YAML; a configured skill is exposed unless its template sets `disabled: true`.
- Every primitive is in `SUPPORTED_PRIMITIVES`.
- Every named pose exists under `embodied.named_poses`.
- Literal `move_relative_ee` steps have valid direction and positive distance.
- Joint trajectories stay within joint limits and workspace limits.
- Every absolute joint trajectory has a positive-duration `move_to_joint_positions` entry that matches its first waypoint; returns are explicit primitives, never generator-only flags.
- The skill has a `description:` block with summary / category / when_to_use / motion_scope / intensity; the loader enforces this and rejects dangling `do_not_use.instead_use` targets.
- Every near-synonym skill has at least one `do_not_use` redirect so Agent/LLM callers can disambiguate (e.g. wave_hello vs greet_observe_raise vs nod_yes).
- Chinese trigger words live ONLY in `description.aliases_zh`; no parallel hardcoded keyword list in `command_parser`.
- Catalog `doc` is auto-synthesized from the `description:` block; do not hand-maintain per-skill prose.
- Real robot testing starts with RViz when the user wants visual confirmation.
- After real robot testing, execute `recover_safe_pose` unless the user explicitly wants to leave the robot at the final pose.

## Verification Commands

Run only on modified files/packages. Always source the project environment from the repository root.

```bash
source .shrc_local && ruff check <modified-python-files>
source .shrc_local && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest <related-tests>
source .shrc_local && colcon build --symlink-install --merge-install --packages-select <packages>
```

For real robot + RViz verification, use the user-requested ROS domain. If unspecified, ask before running hardware.

```bash
source .shrc_local
export ROS_DOMAIN_ID=<domain>
source install/setup.bash
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=true
```

Then call `robot-skill` or ROS actions through the guarded skill interface:

```text
robot-skill --config-name NAME validate SKILL ARGS
robot-skill --config-name NAME execute SKILL --task-id ID
robot-skill --config-name NAME execute recover_safe_pose --task-id ID
```

## Common Failure Modes

- Implementing an anchored spatial request as joint `base_pose` motion.
- Forgetting to add the skill to `planning_policy.allowed_skills` when the planner should generate it.
- Assuming planner `allowed_skills` controls catalog exposure; the catalog uses the current robot YAML and the template's `disabled` flag.
- Exposing a skill in the catalog but not updating catalog tests.
- Adding a new trajectory template when existing `move_relative_ee` primitives are sufficient.
- Running real robot tests without `ROS_DOMAIN_ID` in the same shell.
- Leaving RViz/runtime background processes running after verification without telling the user.
- Adding a skill without a `description:` block, leaving Agent/LLM callers unable to disambiguate it from near-synonyms.
- Hand-writing per-skill prose instead of the SSOT `description:` block, so the catalog and rule parser drift from the YAML.
- Declaring `do_not_use.instead_use` pointing at a skill that does not exist (loader will reject) or forgetting反向 redirects on the synonym skills.
- Duplicating Chinese keywords in `command_parser` hardcode instead of sourcing them from `description.aliases_zh`.
