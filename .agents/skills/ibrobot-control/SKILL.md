---
name: ibrobot-control
description: Use when a user asks Hermes or an Agent to discover, validate, execute, cancel, or stop an IB-Robot capability.
---

# IB-Robot Control

## Overview

Use only `robot-skill` and the ROS Capability Gateway. Discover before acting, require explicit user motion confirmation,
and report only what the CLI proves.

## Required Workflow

Run motion requests in this order. Replace placeholders with catalog or user-provided values.

1. Query the Gateway: `robot-skill --config-name NAME status`.
2. Discover capabilities: `robot-skill --config-name NAME list-skills`.
3. Read the selected contract: `robot-skill --config-name NAME describe SKILL`.
4. Validate parameters without motion: `robot-skill --config-name NAME validate SKILL ARGS`.
5. Show the resolved skill and parameters, then obtain explicit user motion confirmation. General permission, a prior
   confirmation, or schedule pressure does not count.
6. Use a fresh caller-supplied task ID once: `robot-skill --config-name NAME execute SKILL --task-id ID ARGS`.

Stop on any failure, unavailable/not-ready Gateway, unauthorized motion, rejected validation, or missing confirmation.
Do not invent parameters absent from `describe`.

## Cancellation

- For the currently running execute process, send SIGINT/SIGTERM and wait for its terminal JSONL `result`.
- Outside that process, use `robot-skill --config-name NAME cancel --task-id ID`, then wait for terminal task state.
- **Cancellation requested is not robot stopped.** `SKILL_CANCEL_TIMEOUT`, transport failure, or an unknown result means
  the stop state is unknown; dispatch no further motion.

## Hard Boundaries

- The Agent **must not launch or restart the pipeline**.
- The Agent **must not enable motion authorization**; only the operator may set `authorize_motion`.
- The Agent **must not modify ROS parameters**.
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
