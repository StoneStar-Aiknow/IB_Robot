---
name: ibrobot-robot-skill-design
description: "Use when designing, adding, modifying, or validating an IB-Robot embodied skill, Hermes/Agent robot action, social gesture, observation-pose motion, gripper action, MoveIt/relative end-effector action, or when the user asks '设计机器人skill', '新增机器人动作', '新增具身技能', '让机器人做一个动作', 'Hermes 调用机器人', 'catalog 暴露机器人 skill', '庆祝动作', '挥手', '点头', or '观察位置上下左右'."
---

# IB-Robot Robot Skill Design Guide

Interactive design workflow for adding or changing robot skills in IB-Robot. Prevents semantic mistakes such as implementing "move around the observation pose" as an unrelated joint-space gesture.

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

## Internal References

Read only the references needed for the current step:

| Purpose | Reference |
|---------|-----------|
| Author/validate the `description:` contract (Step 5) | `references/contract-schema.md` |
| Concrete YAML starting points for primitives (Step 7) | `references/canonical-templates.md` |
| Common failure modes to review before real robot execution | `references/failure-modes.md` |

Do not expose these references as separate skills.

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

Every new or modified skill MUST carry a `description:` block in the SSOT YAML. See `references/contract-schema.md` for the full schema, validation rules, and a disambiguation example. Key invariants:

- `summary` is intent-first, <= 120 chars.
- `do_not_use.instead_use` must reference a real skill in the same config.
- `aliases_zh` is the SSOT for Chinese trigger keywords (also drives the rule parser when `rule_entry: true`).
- `duration_sec_estimate` must cover arm motion plus gripper primitive overhead with margin.

### Step 6: Produce A Design Record

Before code changes, show a concise design record and ask for confirmation unless the user explicitly requested full autonomy.

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
- ./scripts/build.sh -- --packages-select <packages>
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

Concrete YAML templates for each skill type are in `references/canonical-templates.md`. Do not add backward-compatibility aliases for removed skill names unless there is persisted data, an external consumer, or an explicit user requirement.

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

For common mistakes to watch for, see `references/failure-modes.md`.

## Verification Commands

Run only on modified files/packages. Always source the project environment from the repository root.

```bash
source .shrc_local && ruff check <modified-python-files>
source .shrc_local && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest <related-tests>
source .shrc_local && ./scripts/build.sh -- --packages-select <packages>
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
