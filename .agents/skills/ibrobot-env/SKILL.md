---
name: ibrobot-env
description: "Handles environment setup and command execution in the main IB_Robot workspace. Use BEFORE running any scripts or ROS 2 commands to 'setup environment', 'source .shrc_local', 'set ROS_DOMAIN_ID', '环境变量', '环境初始化', '初始化环境', 'fix import errors', 'ModuleNotFoundError', or 'PYTHONPATH issues'. Triggers whenever environment inheritance is required for correct execution. For commands inside a git worktree (shared venv, worktree PYTHONPATH, mixed main-repo/worktree paths), use ibrobot-worktree-env instead."
---

# IB_Robot Environment Skill

This skill ensures proper environment variable setup before executing Python or ROS 2 commands in the IB_Robot workspace.

## Core Requirement

**All commands that depend on project environment variables must include environment setup in the same shell call.**

Since the Bash tool runs in isolated subshells, environment variables set via `source` in one call are **not retained** across different calls. Therefore, `source .shrc_local` must be combined with the target command in a single call.

## When to Use ibrobot-worktree-env Instead

If the command will run inside a **git worktree** of IB_Robot (not the main checkout), use `ibrobot-worktree-env` instead of this skill. Worktree environments require reusing the main repo's venv while sourcing the worktree's `.shrc_local`; naively running `source .shrc_local` in a worktree produces a mixed environment that silently tests the wrong branch.

Routing signals that indicate worktree-env:
- User mentions "worktree", "git worktree", "worktree 环境"
- CWD is under a worktree path (e.g. `/tmp/ibrobot-*`), not the main repo root
- Symptoms: imports resolve to the main repo despite CWD being elsewhere, "测错分支" complaints

`ibrobot-worktree-env` defers to this skill for non-worktree scenarios; the boundary is bidirectional.

## Standard Execution Patterns

### Running Python Scripts
```bash
source .shrc_local && python3 <script.py>
```

### Running ROS 2 Commands
```bash
source .shrc_local && ros2 <command>
```

### Running Tests
```bash
source .shrc_local && pytest <args>
```

### Building the Project
When ROS 2 source code changes require recompilation:
```bash
source .shrc_local && ./scripts/build.sh
```

## Environment Provided by .shrc_local
- Activates Python virtual environment (`venv`).
- Sets `PYTHONPATH` including `libs/lerobot/src` and `src` directories.
- Sources ROS 2 Humble setup.
- Defines common aliases (`cb`, `cbp`, `src`, etc.).

## Common Error Resolution

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | `PYTHONPATH` not set | Prefix with `source .shrc_local &&` |
| `ros2: command not found` | ROS 2 environment not loaded | Prefix with `source .shrc_local &&` |
| `ImportError: lerobot` | venv not activated | Prefix with `source .shrc_local &&` |

## Quick Reference

Before executing any operation, always check if you need:
```bash
source .shrc_local && <your_command>
```
