# robot_mcp

MCP (Model Context Protocol) tool layer that exposes IB-Robot's **runtime**
skills and status to any MCP-compatible agent host (opencode / Claude Code /
Cursor / VS Code). This lets an external LLM agent read what the robot can do
and sample its state through a typed, cross-host interface — instead of being
bound to one IDE's private plugin API.

> **Phases 0 + 1 (this package).**
> *Phase 0 (read-only):* `list_skills`, `list_poses`, `get_status`.
> *Phase 1 (guarded commanding):* `validate_skill` (safety_guard dry-run) and
> `execute_skill` (real motion via `/embodied/execute_skill`).
>
> **Safety invariant:** `execute_skill` ONLY calls `/embodied/execute_skill`,
> which routes through `skill_executor` → `safety_guard` (white-list + workspace
> bounds). It never exposes `/task_executor/*` or MoveIt directly, so the agent
> cannot bypass the safety checks. The MCP layer adds no safety logic of its own.

## What it exposes

The catalog tools are derived from the **single source of truth** (`robot_config`
YAML). Changing the YAML changes what the agent sees — nothing is hard-coded.

| Tool (opencode prefix `robot_`) | Source | Purpose |
|---|---|---|
| `list_skills` | `robot_config` SSOT | Skill catalog + primitive decomposition |
| `list_poses` | `robot_config` SSOT | Named poses (`home`/`observe_table`/`zero` …) |
| `get_status` | ROS topics (`/embodied/task_status`, `/joint_states`) | Latest task state + joints (stale-flagged) |
| `validate_skill` | `safety_guard` service | Dry-run a skill (no motion): `{allowed, reason}` |
| `execute_skill` | `/embodied/execute_skill` action | Run a real skill, streaming feedback as progress |

opencode namespaces tools by the server name. With the server named `robot`
below, you get `robot_list_skills`, `robot_execute_skill`, etc.

## Prerequisites

```bash
# MCP Python SDK (one-time)
pip install mcp

# ROS 2 environment + workspace overlay (sets ROS_DOMAIN_ID, PYTHONPATH, …)
source .shrc_local
```

The active robot is selected exactly like the main launch:
`robot_config:=<name>` (name without `.yaml`), or `ROBOT_CONFIG=<path>` env,
or `ROBOT_NAME=<name>`. Default is `so101_single_arm`.

## Build

```bash
cbp robot_mcp          # colcon build --packages-select robot_mcp
```

## Run

### A. stdio — opencode launches it (local dev)

```jsonc
// opencode.jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "robot": {
      "type": "local",
      "command": ["bash", "-c", "source install/setup.bash && robot_mcp_server --config-name so101_single_arm"],
      "cwd": "${workspaceFolder}",
      "enabled": true,
      "timeout": 10000
    }
  }
}
```

Then in opencode:

```
list the robot's skills  use robot
```

### B. streamable-http — long-lived node (production)

Run on the robot host (lifecycle independent of the agent; survives reconnects):

```bash
ros2 launch robot_mcp robot_mcp.launch.py robot_config:=so101_single_arm port:=8080
# or directly:
robot_mcp_server --transport streamable-http --host 0.0.0.0 --port 8080
```

Point opencode at it:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "robot": { "type": "remote", "url": "http://<robot-ip>:8080/mcp", "enabled": true }
  }
}
```

## Smoke test (no ROS needed)

Catalog-only mode validates config loading + tool registration without a ROS
daemon:

```bash
robot_mcp_server --config-name so101_single_arm --no-ros
# in another terminal, against the http server, or use the stdio client of your host
```

`--no-ros` keeps `list_skills`/`list_poses` working; `get_status` reports
`ros_available: false`.

## Architecture

```
opencode ──MCP(stdio|http)──► robot_mcp_server ──rclpy──► existing ROS topics
                                  │                          (/embodied/task_status,
                                  └─ robot_config (SSOT)       /joint_states)
                                     list_skills/list_poses
```

The server is a thin client: it observes topics, reads config, and forwards
skill goals to `skill_executor` (which enforces safety). It never publishes
raw joint/pose commands itself.

## Tests

```bash
# Source the workspace overlay first (so robot_config / ibrobot_msgs import).
source install/setup.bash

# The ROS launch_testing pytest plugin conflicts with modern pytest; disable
# plugin autoload to run these pure rclpy tests cleanly:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest src/robot_mcp/test
```

* `test_catalog.py` — SSOT-derived skill/pose catalog (no ROS needed).
* `test_bridge.py` — Phase-1 path against a mock `skill_executor` + `safety_guard`:
  validates the validate-service call, the skill action goal, feedback streaming,
  and result handling end-to-end.
