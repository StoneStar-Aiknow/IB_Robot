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

`so101_single_arm` exposes the social gesture skills `wave_hello`, `nod_yes`,
`shake_no`, `celebrate`, `greet_observe_raise`, `act_cute`, and
`happy_spin_upright` through the same guarded `execute_skill` path.
The authoritative list is the `robot_config` SSOT YAML; call `list_skills`
to enumerate them at runtime. Use `wave_hello` for generic hello/goodbye waving;
`celebrate` and `greet_observe_raise` first move to `observe_table`, then perform
relative motions around that observation pose.

Templates with `disabled: true` are excluded by the shared enabled-template
boundary. They do not appear in `list_skills`; `validate_skill` and
`execute_skill` reject any name that is not a member of the current catalog
before calling the ROS bridge.

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
export ROS_DOMAIN_ID=49
ros2 launch robot_mcp robot_mcp.launch.py robot_config:=so101_single_arm port:=8080
# or directly:
robot_mcp_server --transport streamable-http --host 127.0.0.1 --port 8080
```

HTTP defaults to loopback. From a separate agent host, forward a local port over
SSH instead of exposing the MCP endpoint directly on the robot network:

```bash
ssh -N -L 8080:127.0.0.1:8080 robot@<robot-host>
```

`execute_skill` moves real hardware. The MCP HTTP endpoint does not provide
application-level authentication and must not be exposed to an untrusted
network. Use `host:=0.0.0.0` (or `--host 0.0.0.0`) only as an explicit override
behind an authenticated TLS proxy or on a controlled VPN.

> Note: `robot_mcp.launch.py` launches `robot_mcp_server` via `ExecuteProcess`
> against the installed binary path (`lib/robot_mcp/robot_mcp_server`) instead of
> a ros2 `Node` descriptor. This keeps the MCP stdio/HTTP transport isolated from
> ROS node lifecycle logging. The process therefore is not registered as a named
> ROS node, does not accept launch `parameters=[...]`, and requires the package to
> be installed (`colcon build`) so `FindPackagePrefix` can resolve. It still
> connects to ROS 2 via the inherited environment (`ROS_DOMAIN_ID`, `rmw`).

Point opencode at it:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "robot": { "type": "remote", "url": "http://127.0.0.1:8080/mcp", "enabled": true }
  }
}
```

### C. SO-101 real robot + Hermes

Terminal 1, start the real robot stack and guarded embodied skill runtime:

```bash
source .shrc_local
export ROS_DOMAIN_ID=49
ros2 launch embodied_bringup embodied_pipeline.launch.py \
  robot_config:=so101_single_arm \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false
```

Terminal 2, expose MCP over HTTP for Hermes:

```bash
source .shrc_local
export ROS_DOMAIN_ID=49
ros2 launch robot_mcp robot_mcp.launch.py robot_config:=so101_single_arm port:=8080
```

If Hermes runs on another host, create the same SSH tunnel there:

```bash
ssh -N -L 8080:127.0.0.1:8080 robot@<robot-host>
```

Configure Hermes to use `http://127.0.0.1:8080/mcp` as the `robot` MCP server.
Natural-language prompts should map to the guarded skill tools, for example
`和我打个招呼` -> `robot_execute_skill(skill_name="wave_hello")`.

## Smoke test (no ROS needed)

Catalog-only mode validates config loading + tool registration without a ROS
daemon:

```bash
robot_mcp_server --config-name so101_single_arm --transport streamable-http \
  --host 127.0.0.1 --port 8080 --no-ros
# in another terminal, connect to http://127.0.0.1:8080/mcp
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
