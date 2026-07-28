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

### Skill execution admission state machine

`execute_skill` maintains a thread-safe **single MCP SkillCommand admission**
slot so the robot never runs two skills in parallel. The slot has three states:

| State | Meaning | Concurrent request response |
|-------|---------|------------------------------|
| `idle` | No in-flight action | Dispatch a new goal immediately |
| `running` | Goal dispatched, awaiting terminal state | `SKILL_IN_PROGRESS` |
| `recovering` | Previous goal timed out, draining to terminal | `MOTION_RECOVERY_IN_PROGRESS` |

`validate_skill` is **not** gated by admission (it is a dry-run), only
`execute_skill` is.

### Error codes returned by `execute_skill`

| `error_code` | Trigger | Admission after return | Caller action |
|--------------|---------|--------------------------|---------------|
| `UNSUPPORTED_SKILL` | Skill name not in catalog (disabled or unknown) | Released immediately | Fix skill_name; retry |
| `BRIDGE_OFFLINE` / `ACTION_SERVER_UNAVAILABLE` | ROS bridge down or `/embodied/execute_skill` server not discovered | Released immediately | Start `skill_executor` / rclpy; retry |
| `SKILL_IN_PROGRESS` | Another `execute_skill` is still `running` | Stays `running` | Wait for previous call to return a terminal result; do not retry in a tight loop |
| `MOTION_RECOVERY_IN_PROGRESS` | Previous goal timed out, `cancel_and_drain_skill_goal` still running | Stays `recovering` | Wait; admission releases automatically when the result future reaches terminal state. If it never does, restart `robot_mcp_server` |
| `ACCEPT_TIMEOUT` | `send_goal_async` did not complete within `_ACCEPT_TIMEOUT_SEC` (5s) | Transitions to `recovering` | Same as `MOTION_RECOVERY_IN_PROGRESS` |
| `DISPATCH_FAILED` | `send_goal_async` / `get_result_async` raised (transport failure) | Released immediately | Retry; if persistent, restart `robot_mcp_server` |
| `REJECTED` | Server-side reject (`goal_handle.accepted == False`) | Released immediately | Check `skill_executor` logs |
| `RESULT_TIMEOUT` | Skill did not finish within `timeout_sec + _RESULT_GRACE_SEC` but cancel drained cleanly | Released immediately | Increase `timeout_sec`; retry |
| `CANCEL_CLEANUP_TIMEOUT` | Cancel requested but result future did not reach terminal within `_CANCEL_DRAIN_TIMEOUT_SEC` (5s) | Stays `recovering` until the future eventually terminates | Wait for `MOTION_RECOVERY_IN_PROGRESS` to clear; if it never does, restart `robot_mcp_server` |
| Server-side `error_code` (e.g. `SKILL_CANCELLED`, `PRIMITIVE_ARM_FAILED`, `SKILL_TIMEOUT`, `SKILL_CANCEL_CLEANUP_TIMEOUT`) | Goal completed terminally with non-success | Released immediately | Inspect `message` / `executed_primitives`; fix upstream config or retry |

The caller-visible pattern is:
- Admission is **claimed before** dispatch and **released in a `finally` block**
  once the goal reaches a tracked terminal state.
- Any path where the MCP side cannot prove the goal is terminal (late accepted
  goals, cancel-drain timeouts) keeps admission in `recovering` to fail closed.
- An LLM / Hermes orchestration loop should treat `SKILL_IN_PROGRESS`,
  `MOTION_RECOVERY_IN_PROGRESS`, and `CANCEL_CLEANUP_TIMEOUT` as
  "wait, do not retry" signals, and all other codes as "released, free to retry".

For a grasp-enabled robot config, Hermes can invoke the complete physical grasp
pipeline without receiving raw poses or MoveIt access:

```text
robot_execute_skill(skill_name="pick_object", target_name="banana", timeout_sec=0)
```

`target_name` is a runtime visual text query for `pick_object`. `timeout_sec=0`
uses the skill timeout declared in the robot YAML (240 seconds in the SO101
hand-eye grasp config).

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

For the calibrated SO101 grasp host, maintain hardware and calibration values in
`so101_handeye_realsense_grasp.yaml` and select that same config in both
`embodied_bringup` and `robot_mcp`.

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
      "command": ["bash", "-c", "source .shrc_local && ros2 run robot_mcp robot_mcp_server --config-name so101_handeye_realsense_grasp"],
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
ros2 launch robot_mcp robot_mcp.launch.py \
  robot_config:=so101_handeye_realsense_grasp \
  port:=8080
# or directly:
ros2 run robot_mcp robot_mcp_server \
  --config-name so101_handeye_realsense_grasp \
  --transport streamable-http --host 127.0.0.1 --port 8080
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
  robot_config:=so101_handeye_realsense_grasp \
  control_mode:=moveit_planning \
  use_sim:=false \
  moveit_display:=false
```

Terminal 2, expose MCP over HTTP for Hermes:

```bash
source .shrc_local
export ROS_DOMAIN_ID=49
ros2 launch robot_mcp robot_mcp.launch.py robot_config:=so101_handeye_realsense_grasp port:=8080
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
robot_mcp_server --config-name so101_handeye_realsense_grasp --transport streamable-http \
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
