# Common Failure Modes

## When to Read

- Reviewing a skill design before real robot execution
- Debugging a skill that failed validation or behaved unexpectedly

## Failure Modes

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
