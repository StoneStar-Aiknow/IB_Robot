# IB-Robot Tracing (ros2_tracing + LTTng)

Low-overhead tracing for the inference chain, following ROS 2 best practices.

## Setup

```bash
./scripts/setup.sh
```

`./scripts/setup.sh` now installs the tracing stack as part of the normal
workspace setup (LTTng, ros2_tracing, babeltrace2, and `tracetools-analysis`).

If the workspace is already set up and you only want to add tracing tools later,
you can still run:

```bash
bash scripts/tracing/setup_tracing.sh
```

## Usage

### Option A: Integrated launch (recommended)

```bash
ros2 launch robot_config robot.launch.py \
    robot_config:=so101_single_arm use_sim:=true \
    control_mode:=model_inference \
    enable_tracing:=true
```

This creates an LTTng session directly from the launch entrypoint, capturing
ROS 2 UST events (`ros2:*`) and Python business tracepoints (`ib_trace.*`).

If the default session name `ib_robot_trace` is already in use, launch will
auto-suffix a timestamp instead of overwriting the existing trace. Explicit
`trace_session_name:=...` values are never overwritten.

### Option B: Separate trace session

```bash
# Terminal 1
bash scripts/tracing/start_trace.sh

# Terminal 2
ros2 launch robot_config robot.launch.py ...

# When done
bash scripts/tracing/stop_trace.sh
```

In manual mode, `start_trace.sh` also auto-suffixes the default session name on
collision. If you explicitly pass a colliding session name, it fails fast rather
than clobbering the old trace.

### Analyze

```bash
python3 scripts/tracing/analyze_trace.py --trace-dir ~/.ros/tracing/ib_robot_trace
```

The analyzer now reports two views:
- **Request-level stage latencies**: `Obs sampling → Dispatch→Infer → Inference → Dispatch decode → Refill→Execute ...`
- **Observation summaries**: ingress transport (`transport_ms`) and sampling freshness (`age_ms`) per observation key

## How It Works

No separate tracing package needed — this follows the ROS 2 standard approach:

1. **Direct LTTng session management in `robot.launch.py`** — enables
   `ros2:*` UST events and the Python tracing domain `ib_trace.*` on startup,
   and stops/destroys the session on shutdown.

2. **`logging.getLogger('ib_trace.*')` + `lttngust`** in nodes — Python logging
   records are emitted through the LTTng Python handler, so business events are
   written to the trace as Python-domain events with low overhead.

3. **`ros2 trace` CLI / `lttng` CLI** — standard tools for manual session control.

4. **`babeltrace2` / Trace Compass / `tracetools_analysis`** — standard analysis.

## Business Tracepoints

Emitted as `logging.info("[event_name] key=value ...")` on `ib_trace.*` loggers:

| Event | Logger | Node |
|-------|--------|------|
| `dispatch_request` | `ib_trace.dispatch` | action_dispatcher_node |
| `dispatch_result` | `ib_trace.dispatch` | action_dispatcher_node |
| `dispatch_decode` | `ib_trace.dispatch` | action_dispatcher_node |
| `queue_refill` | `ib_trace.dispatch` | action_dispatcher_node |
| `action_execute` | `ib_trace.dispatch` | action_dispatcher_node |
| `action_topic_publish` | `ib_trace.execute` | topic_executor |
| `obs_receive` | `ib_trace.policy` | lerobot_policy_node |
| `obs_sample` | `ib_trace.policy` | lerobot_policy_node |
| `obs_frame` | `ib_trace.policy` | lerobot_policy_node |
| `preprocess_begin/end` | `ib_trace.policy` | lerobot_policy_node |
| `inference_begin/end` | `ib_trace.policy` / `ib_trace.inference` | both |
| `postprocess_begin/end` | `ib_trace.policy` | lerobot_policy_node |
| `action_chunk_publish` | `ib_trace.policy` | lerobot_policy_node |
| `edge_publish` | `ib_trace.policy` | lerobot_policy_node |
| `edge_receive` | `ib_trace.policy` | lerobot_policy_node |

## Files

```
robot.launch.py          ← Starts/stops LTTng session (enable_tracing:=true)
scripts/tracing/
├── setup_tracing.sh     ← Retrofit tracing tools into an existing workspace
├── start_trace.sh       ← Manual session start
├── stop_trace.sh        ← Manual session stop
├── analyze_trace.py     ← Latency analysis CLI
├── README.md            ← Chinese documentation
└── README.en.md         ← English documentation
```
