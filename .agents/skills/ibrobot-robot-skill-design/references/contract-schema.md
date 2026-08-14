# Description Contract Schema

Every new or modified skill MUST carry a `description:` block in the SSOT YAML (co-located with its `primitive_sequence`). This block is the single source of truth for how an Agent/Hermes caller and the rule parser pick THIS skill over its near-synonyms. Omitting it is an architecture violation, not a style choice.

## When to Read

- Step 5 of the design flow is being executed
- You need to author or validate a skill's `description:` block
- A new skill has near-synonyms among existing skills and needs disambiguation

## Required Schema

Validated by `robot_config.loader._validate_skill_description`:

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

## Disambiguation Rule

The whole point of this contract:

- Before finalizing a new skill, list its near-synonyms among existing skills and add one `do_not_use` entry per synonym redirecting to the right alternative.
- Conversely, add反向 redirects on the existing synonym skills pointing at the new one when the boundary changes.
- `aliases_zh` is the SSOT for Chinese trigger keywords and catalog aliases; do NOT duplicate keyword lists in `embodied_common.command_parser` hardcode.
- `extract_skill_aliases` injects `aliases_zh` into the deterministic rule parser only when `rule_entry: true` and `requires_motion_params: false`.
- `summary` must be intent-driven ("Wave hello/goodbye with the wrist"), not mechanical ("Sinusoidal joint-5 motion").
- `do_not_use.instead_use` must reference a skill that actually exists in the same config — the loader rejects dangling redirects.
- `duration_sec_estimate` must cover deterministic arm motion plus 1.0 second for an `open`/`closed`
  `initial_gripper_state` and 1.0 second for every explicit `open_gripper`/`close_gripper` primitive, with margin.

## Example: Greeting Cluster Disambiguation

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
