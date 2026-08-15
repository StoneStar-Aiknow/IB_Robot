---
name: ibrobot-control
description: "Use when a user asks Hermes or an Agent to discover, validate, execute, cancel, or stop IB-Robot capabilities through the `robot-skill` CLI and ROS Capability Gateway. Covers 'run a robot skill', 'execute robot action', 'cancel motion', 'stop robot', '执行机器人动作', '取消动作', '停止机器人', 'nod', 'wave', 'celebrate', 'look around', or interact with existing high-level robot skills. Requires explicit user motion confirmation before any physical motion; never bypasses the Gateway or calls raw ros2 / MoveIt / controller commands directly."
---

# IB-Robot Control

## Overview

Use only `robot-skill` and the ROS Capability Gateway. Discover before acting, present the exact plan before execution,
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
5. Show the exact ordered steps, parameters, plan digest, registry identity, and a fresh task ID, then flush the
   presentation to the user.
6. Immediately bind that exact tuple once with
   `robot-skill confirm-plan --plan-token TOKEN --plan-digest DIGEST --task-id ID --timeout-sec SEC`.
7. Execute only the returned confirmation token with
   `robot-skill execute-plan --plan-token TOKEN --confirmation-token CONFIRMATION_TOKEN --task-id ID --timeout-sec SEC
   --plan-id PLAN_ID --plan-digest DIGEST --registry-epoch EPOCH --registry-generation GENERATION
   --registry-digest REGISTRY_DIGEST --expected-step-count COUNT`.

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

Natural-language single-Skill and Workflow requests both use the plan workflow above. The internal `confirm-plan` call is
the Gateway's technical binding for the exact plan/task tuple, not a second user confirmation gate. For an explicitly
selected single skill, the direct `describe -> validate -> execute` path remains valid.

Stop on any failure, unavailable/not-ready Gateway, unauthorized motion, or rejected validation.
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

- When no process in this session owns the executing **Agent plan**, issue `robot-skill cancel-plan --task-id ID
  --plan-id PLAN_ID --plan-digest DIGEST --registry-epoch EPOCH --registry-generation GENERATION
  --registry-digest REGISTRY_DIGEST --expected-step-count COUNT`, then wait for the strictly validated terminal result.
  This is the only external command that cancels the `/embodied/execute_agent_plan` goal. **Never** use `cancel` for an
  Agent plan — it targets the wrong action.
- `robot-skill cancel --task-id ID` cancels **only** a legacy single-skill `execute` goal
  (`/embodied/execute_skill`). Do not use it for `execute-plan`.
- SIGINT/SIGTERM on the running `execute-plan` process requests root cancellation and waits for its terminal result.
  Outside that process, use `cancel-plan` with the same task ID.
- **Cancellation requested is not robot stopped.** `SKILL_CANCEL_TIMEOUT`, transport failure, or an unknown result means
  the stop state is unknown; dispatch no further motion.

## Interactive Closed Loop: 别动 / 继续

Within one Hermes session the catalog query, plan/confirm/execute, stop, and continue steps form one closed loop.
Drive them with the closed vocabulary only; free text outside the grammar never acts.

1. **Discover (read-only).** `robot-skill status` then `robot-skill list-skills`. Do not plan or move.
2. **Reject out-of-catalog.** Any requested `skill_name` not in the current `planner_visible_names` is refused with
   `SKILL_REFERENCE_MISSING` before planning. Do not substitute or invent a skill.
3. **Prepare + present + execute immediately (no confirmation gate).** `plan-workflow` once, show and flush the ordered steps,
   plan digest, registry identity, and a fresh task ID, then run `validate-plan` + `confirm-plan` + `execute-plan`
   back-to-back without waiting for a `确认` reply — execution starts right after presentation. The user catches a
   wrong workflow with `别动` (step 4) during execution instead of confirming beforehand. The `确认` grammar is not
   needed for this flow. The session binds the sole unexpired pending plan; never accept a pending ID from the user or model.
4. **别动 → definite terminal.** On a stop phrase (`别动` / `停` / `停止` / `停下` / `stop` / `halt`), latch stop across
   validation, internal confirmation, action-server waiting, goal submission, and goal acceptance. If this session owns
   a running `execute-plan` process, signal only that process and let it cancel and converge. Use external `cancel-plan`
   only when no local process owns that task. Never use both for one task. Suppress any new goal after a pre-submission
   stop. A proven fresh in-process task needs no CancelGoal; if the deterministic task may be an idempotent retry, the
   owner must cancel/converge the existing goal instead of synthesizing a local terminal. GoalStatus, success/error,
   plan ID/digest, registry identity, and completed-step
   bounds must agree. Only status 5 + `success=false` + `SKILL_CANCELLED` + exact identity proves cancellation.
   `SKILL_CANCEL_TIMEOUT`, identity mismatch, or an unknown result means the stop state is unknown — report uncertainty
   and send no further motion.
5. **继续 → fresh user request only.** A continue phrase is permitted only after a definite canceled terminal
   (`GoalStatus=5` plus `SKILL_CANCELLED`). In this baseline `--resume` is rejected because `completed_step_count` is
   telemetry, not server-owned continuation admission. Do not slice the old plan or create a new task ID to bypass that
   boundary. After definite cancellation, a user may submit an independent fresh workflow, which re-queries the fresh
   registry and uses new plan/task tokens. Success, failure, and unknown states do not authorize automatic continuation.

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
| Continuing after an unknown stop | `继续` needs a definite canceled terminal (5 + `SKILL_CANCELLED`). Otherwise refuse continuation and send no motion. |
| Slicing completed steps on `继续` | Reject breakpoint resume until the Gateway provides server-owned continuation admission. |
