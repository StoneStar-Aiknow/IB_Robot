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
3. Construct a fresh request ID directly in the conversation, such as `agent-request-YYYYMMDD-NNN`, then generate a
   typed plan: `robot-skill plan-workflow --request-id REQUEST_ID --text TEXT --workflow-json JSON`.
4. Read every selected contract with `robot-skill describe SKILL`, then run
   `robot-skill validate-plan --plan-token TOKEN`.
5. Construct the fresh task ID directly from the returned plan ID, such as `agent-task-PLAN_ID`, or use a session-local
   value such as `agent-task-YYYYMMDD-NNN`. Show the exact ordered steps, parameters, plan digest, registry identity, and
   that fresh task ID. Obtain explicit user
   motion confirmation. General permission, a prior confirmation, or schedule pressure does not count.
6. Bind that exact tuple once with
   `robot-skill confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID`.
7. Execute only the returned confirmation token with
   `robot-skill execute-plan --plan-token TOKEN --confirmation-token CONFIRMATION_TOKEN --task-id ID`.

By default, omit `--timeout-sec` from both commands so both use the current Gateway task budget. Do not derive a plan
budget from `default_skill_timeout_sec`. Only when the user explicitly requests a smaller budget, append the same
`--timeout-sec SEC` value to both commands. Never change the budget after confirmation.

`workflow-json` is an array of flat `WorkflowStep` objects. Skill arguments are top-level fields, for example
`[{"skill_name":"pick_object","target_name":"marker"}]`. Never use `skill`, a nested `parameters` object, or a bare
object. Invoke `plan-workflow` once with all three required options; do not probe it with an incomplete command.

Construct request IDs and task IDs directly in the conversation and `robot-skill` arguments. Do not call Python,
`uuidgen`, `date`, a shell, or any other helper tool to generate them. A command approval, including session-wide
approval, authorizes only that command and is not user motion confirmation. Only an explicit user response confirming
the displayed plan/task tuple permits `confirm-plan`.

Natural-language single-Skill and Workflow requests both use the plan workflow above. For an explicitly selected single
skill, the direct `describe -> validate -> explicit user motion confirmation -> execute` path remains valid.

Stop on any failure, unavailable/not-ready Gateway, unauthorized motion, rejected validation, or missing confirmation.
After a nonzero exit, do not run `--help`, inspect or change the environment, alter the JSON or timeout, query status, or
issue another `robot-skill` command in the same user request. Report the exact returned error instead.
Do not invent parameters absent from `describe`.
For an ordered multi-Skill request, call `plan-workflow` exactly once with the user's original wording and typed steps. The returned single
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

Stop the right goal with the right command. The two executors are different actions; the wrong command cannot
reach the active goal and will time out as `SKILL_CANCEL_TIMEOUT` while the robot keeps moving.

- For an executing **Agent plan** (`execute-plan`): issue `robot-skill cancel-plan --task-id ID`, then wait for a
  terminal `GoalStatus` ∈ {succeeded=4, canceled=5, aborted=6}. This is the only command that cancels the
  `/embodied/execute_agent_plan` goal. **Never** use `cancel` for an Agent plan — it targets the wrong action.
- `robot-skill cancel --task-id ID` cancels **only** a legacy single-skill `execute` goal
  (`/embodied/execute_skill`). Do not use it for `execute-plan`.
- Sending SIGINT/SIGTERM to the `execute-plan` process only kills the CLI; it does **not** reliably cancel the
  Gateway goal. Always issue `cancel-plan` to actually stop an Agent plan.
- **Cancellation requested is not robot stopped.** `SKILL_CANCEL_TIMEOUT`, transport failure, or an unknown result means
  the stop state is unknown; dispatch no further motion.

## Interactive Closed Loop: 别动 / 继续

Within one Hermes session the catalog query, plan/confirm/execute, stop, and continue steps form one closed loop.
Drive them with the closed vocabulary only; free text outside the grammar never acts.

1. **Discover (read-only).** `robot-skill status` then `robot-skill list-skills`. Do not plan or move.
2. **Reject out-of-catalog.** Any requested `skill_name` not in the current `planner_visible_names` is refused with
   `SKILL_REFERENCE_MISSING` before planning. Do not substitute or invent a skill.
3. **Prepare + present + confirm in session.** `plan-workflow` once, show the ordered steps, plan digest, registry
   identity, and a fresh task ID, then accept only the closed confirmation grammar
   (`确认执行当前计划` / `确认` / `确认执行` / `执行吧` / `好` / `好的` / `可以` / `是的` / `confirm`).
   Bind that single pending plan with `confirm-plan`. The session binds the sole unexpired pending plan; never accept a
   pending ID from the user or model.
4. **别动 → definite terminal.** On a stop phrase (`别动` / `停` / `停止` / `停下` / `stop` / `halt`), cancel the
   active plan with `cancel-plan` and wait for a terminal `GoalStatus` ∈ {succeeded=4, canceled=5, aborted=6}. Only a
   terminal result is a definite stopped state. `SKILL_CANCEL_TIMEOUT` or an unknown result means the stop state is
   unknown — report uncertainty and send no further motion.
5. **继续 → fresh-state continuation (breakpoint resume).** A continue phrase (`继续` / `继续吧` / `go on` / `continue`)
   is a **new user request**, not a retry. It is permitted only after a definite terminal (succeeded / canceled /
   aborted); `UNKNOWN` is refused and no motion is dispatched. Re-query `status` and `list-skills` on the fresh
   registry identity. Then resume from the breakpoint: read the prior terminal's `completed_step_count`, slice the
   original ordered steps to `original_steps[completed_step_count:]`, re-validate those remaining steps against the
   fresh catalog, and run a brand-new `plan-workflow` + `confirm-plan` + `execute-plan` on **only the remaining steps**
   with a **new `request_id` and `task_id`**. Fully-completed steps are skipped; the step that was interrupted
   mid-execution is re-run from its start (never resumed mid-skill). Never reuse the prior `plan_token`,
   `confirmation_token`, or `task_id`. If `completed_step_count` already equals the full step count, the plan is
   already complete — do not plan again; ask the user for a new request.

A reference implementation of this loop (catalog discovery, out-of-catalog rejection, in-session prepare/confirm,
stop-to-definite-terminal, and fresh-state continuation) lives in
`robot_skill_cli.interactive_control.InteractiveController` and is unit-tested without a ROS stack.

## Hard Boundaries

- The Agent **must not launch or restart the pipeline**.
- The Agent **must not enable motion authorization**; only the operator may set `authorize_motion`.
- The Agent **must not modify ROS parameters**.
- The Agent **must not source environment scripts, select a ROS domain, or discover another repository/config**.
- The Agent **must not call Python, `uuidgen`, `date`, a shell, or another helper tool to generate request/task IDs**.
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
| Treating command approval as motion confirmation | Wait for an explicit response confirming the displayed plan. |
| Retrying after checking idle | Require a new user request and repeat the entire workflow; never retry automatically. |
| Continuing after an unknown stop | `继续` needs a definite terminal (4/5/6). Unknown stop → refuse continuation, send no motion. |
| Re-running completed steps on `继续` | Resume from the breakpoint: plan only `original_steps[completed_step_count:]` with new IDs; skip fully-done steps, re-run the interrupted step from its start. |
