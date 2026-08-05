---
name: ibrobot-control
description: Use when a user asks Hermes or an Agent to discover, validate, execute, cancel, or stop an IB-Robot capability.
---

# IB-Robot Control

## Overview

Use only `robot-skill` and the ROS Capability Gateway. Discover before acting, require explicit user motion confirmation,
and report only what the CLI proves.

When launched by `hermes-robot`, the `robot-skill` executable on `PATH` is already bound to the preflighted robot config
and ROS domain. Invoke that exact executable directly. Never source `.shrc_local` or another setup script, inspect or
modify ROS/Python environment variables, search for robot configs or repositories, load `ibrobot-env`, use an absolute
`robot-skill` path, or add `--config-name`/`--config-path`. On any nonzero exit, report the exact CLI error and stop; a
failed command never proves that a status check completed.

## Natural-Language Plan Workflow

Run natural-language motion requests in this order. Replace placeholders with returned values.

1. Query the Gateway: `robot-skill status`.
2. Discover capabilities: `robot-skill list-skills`.
3. Generate a typed plan: `robot-skill plan-text --request-id REQUEST_ID --text TEXT`.
4. Read every selected contract with `robot-skill describe SKILL`, then run
   `robot-skill validate-plan --plan-token TOKEN`.
5. Show the exact ordered steps, parameters, plan digest, registry identity, and a fresh task ID. Obtain explicit user
   motion confirmation. General permission, a prior confirmation, or schedule pressure does not count.
6. Bind that exact tuple once with
   `robot-skill confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID`.
7. Execute only the returned confirmation token with
   `robot-skill execute-plan --plan-token TOKEN --confirmation-token CONFIRMATION_TOKEN --task-id ID`.

Natural-language single-Skill and Workflow requests both use the plan workflow above. For an explicitly selected single
skill, the direct `describe -> validate -> explicit user motion confirmation -> execute` path remains valid.

Stop on any failure, unavailable/not-ready Gateway, unauthorized motion, rejected validation, or missing confirmation.
Do not invent parameters absent from `describe`.
For an ordered multi-Skill request, call `plan-text` exactly once with the user's original wording. The returned single
plan must contain all ordered `workflow_steps`. If planning omits, reorders, or rejects a requested step, report that
exact result and stop; do not retry alternate phrasings and do not split the request into separately confirmed plans.

## Catalog Reload

When the operator asks to activate edited robot Skill YAML without restarting the robot, run exactly one
`robot-skill reload-catalog --request-id REQUEST_ID --force`. This reloads only the Gateway's configured catalog source;
it does not accept another path. Report `old_generation`, `generation`, `changed_skills`, and diagnostics. Then run
`robot-skill status` and require its registry identity to match the reload result before creating any new plan. A reload
timeout has an unknown activation result: stop and ask for a new user request before checking or retrying.

This is separate from Hermes `/reload-skills`, which only reloads Agent instruction files such as this `SKILL.md`.
Changing `nod_yes` implementation or manifest YAML requires `reload-catalog`, not `/reload-skills`.

## Cancellation

- For the currently running execute process, send SIGINT/SIGTERM and wait for its terminal JSONL `result`.
- Outside that process, use `robot-skill cancel --task-id ID`, then wait for terminal task state.
- For Agent plans, use `robot-skill cancel-plan --task-id ID`.
- **Cancellation requested is not robot stopped.** `SKILL_CANCEL_TIMEOUT`, transport failure, or an unknown result means
  the stop state is unknown; dispatch no further motion.

## Hard Boundaries

- The Agent **must not launch or restart the pipeline**.
- The Agent **must not enable motion authorization**; only the operator may set `authorize_motion`.
- The Agent **must not modify ROS parameters**.
- The Agent **must not source environment scripts, select a ROS domain, or discover another repository/config**.
- The Agent **must not call primitive, MoveIt, controller, or raw ros2 motion commands**.
- The Agent must not copy `docs/ib_robot_social_skill.md` as a control Skill.
- The Agent **must not automatically retry after failure, timeout, or unknown result**, including with a new task ID.

## Quick Reference

| Situation | Response |
|---|---|
| Catalog-only request | Use catalog commands; no motion confirmation is needed. |
| Runtime unavailable/unauthorized | Report the CLI error; do not start or alter infrastructure. |
| Terminal result | Report only public status/error fields. |
| Stop state unknown | Report uncertainty and send no new motion. |

## Rationalization Check

| Thought | Reality |
|---|---|
| "raw ROS is faster." | It bypasses the required control surface. |
| "More parameters make it safer." | Invented frame, orientation, velocity, or range arguments violate `describe`. |
| "It probably stopped; use a new ID." | Unknown is not stopped, and a new ID is still an automatic retry. |

## Red Flags

Prior permission, demo pressure, assumed stop, bypass commands, or an invented schema all mean: stop and restart the
required workflow.

## Common Mistakes

| Mistake | Correction |
|---|---|
| Treating accepted cancellation as physical stop | Wait for a terminal result or task ledger terminal state. |
| Retrying after checking idle | Require a new user request and repeat the entire workflow; never retry automatically. |
