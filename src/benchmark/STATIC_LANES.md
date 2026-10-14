# Static LIBERO Lanes

## Launch Interface

This MVP supports static scheduling with fail-fast coordination only. It does
not create a dynamic queue, steal work, retry episodes on another lane, or
resume an existing run. Use a fresh shared run ID for every launch.

| Node | ROS parameters |
| --- | --- |
| `benchmark_environment_node` | Existing endpoint/config parameters, `lane_index=0`, `lane_count=1`, `run_id=""` |
| `benchmark_evaluator_node` | Existing endpoint/config parameters and the same global `run_id`; no lane index/count parameters needed |
| `benchmark_coordinator_node` | `robot_config_path`, global `run_id`, `lane_count`, optional `wait_timeout_sec=3600.0` |

`lane_id` remains declared for the existing launch interface but does not
control sharding or append a suffix to run identity. Environment and evaluator
must receive the same global `run_id` (or a shared explicit evaluation.run_id).
Each evaluator must connect to its own environment's plan/reset/finalize services.
The unified builder materializes lane-specific robot configs and endpoints.

The launch resolver accepts integer lane counts 1..20. The environment checks
that its lane_count matches active `benchmark.evaluation.parallel.lane_count`
and negotiates the actual global plan against provider capabilities, including
`max_lane_count`; it does not substitute a single-lane plan. LIBERO declares 20.
Supported configuration under `benchmark.evaluation`:

```yaml
enabled: true
parallel:
  enabled: true
  lane_count: 2
  scheduling: static
  fail_fast: true
output:
  write_canonical: true
inference:
  max_in_flight: 1
  queue_size: 64
```

Reject dynamic scheduling and fail_fast=false in parallel mode. The configured
lane count may exceed the per-lane episode count: every lane runs the complete
plan, with no modulo sharding or unassigned lanes. Keep the full evaluation task
list and episode count in every lane's configuration, not a sliced task list.

`inference.max_in_flight` is any positive int; `queue_size` is waiting
capacity and must be at least that limit. Defaults are 1 and 64 in both DDS and
RTP shared hosts. The concurrency level is model-neutral here; the shared owner
enforces the contracts it can host (policy interface, Torch backend,
request-scoped state, direct execution). These fields are validated in serial mode
too, but only consumed when `parallel.enabled=true` (even with lane_count=1).

## Work Identity

The adapter first resolves the entire suite and selected task order. Every
lane receives the complete configured task and episode plan:

```text
lane_<N> = selected tasks x configured episode indices
```

The plan keeps all suite-local task IDs, selected task order, original
episodes_per_task, seed, and initial-state selection policy. `lane_index=null`
means a global plan; an integer identifies an independent full-plan lane.
`planned_episodes` is the complete configured episode count on every lane.
`episode_indices(task_id)`, `assigned_task_ids`, and `episode_work` are the
execution views. No task or episode is renumbered.

For tasks=[0], episodes_per_task=2, lane_count=2:

```text
lane_000: task 0 / episode 0, episode 1 / seed S
lane_001: task 0 / episode 0, episode 1 / seed S
```

LIBERO's actual initial state remains original episode_index modulo that
task's native initial-state count. Supervisor, reset authorization, finalization
ledger, native task counts, and canonical completeness all use the full plan.
ROS episode_id is still local to each environment and may coincide across lanes;
canonical identity includes lane identity in metadata so repeated work from
different lanes remains distinct.

## Fan-In Lifecycle

For multiple lanes, native and canonical outputs share:

```text
<output.root>/<global_run_id>/
  lanes/lane_000/{run_manifest.json,episodes.jsonl,summary.json,native/}
  lanes/lane_001/{run_manifest.json,episodes.jsonl,summary.json,native/}
  run_manifest.json
  episodes.jsonl
  summary.json
```

Single-lane runs retain their existing flat run directory and need no coordinator.

The unified launch starts one coordinator with the lane processes,
and shuts down the graph on **coordinator exit**, not the first successful
evaluator exit. The coordinator's `wait_timeout_sec` defaults to 3600 seconds
for the whole run, not one episode. It is currently a ROS node parameter only:
the unified builder does not expose it through benchmark YAML or a launch CLI
argument, and uses the default. Adding `wait_timeout_sec:=...` to the unified
launch command does not configure the coordinator. The coordinator waits for
terminal lane manifests, written after durable lane summaries/records. It exits early on a partial/failed
lane, or at the deadline, after persisting a partial global report. Launch owns
stopping the remaining evaluators/environments/model when the coordinator exits.
Nonterminal lanes are excluded from that fail-fast snapshot and listed as errors;
their later shutdown reports do not mutate the already terminal global report.

Alternatively launch the coordinator after **all** evaluator exits, with
`wait_timeout_sec=0.0`. This immediately snapshots lane reports, including
missing/failed lanes. Do not shut down the graph until this coordinator exits.
This mode leaves live fail-fast cancellation to the launch owner.

For a custom launch owner that starts lanes separately and has not already
started a coordinator, a standalone invocation can set a longer timeout:

```bash
source .shrc_local && ros2 run benchmark_runtime benchmark_coordinator_node \
  --ros-args -p robot_config_path:=/absolute/path/generated/run.yaml \
  -p run_id:=libero_manual_run_001 -p lane_count:=5 -p wait_timeout_sec:=7200.0
```

The config must have parallel enabled, lane_count=5, static/fail_fast=true and
canonical output enabled. Its output root and the run ID must match the live
lanes; the run ID must not have a terminal global report. This is not an override
for an already running unified launch. The custom owner must stop the remaining
graph on coordinator exit. `lane_startup_timeout_sec`, `shutdown_timeout_sec`,
`output.lane_subdirectories` and `inference.session_idle_timeout_sec` are historical
design proposals, not shipped benchmark configuration keys.

The coordinator uses a lane's durable resolved global plan snapshot, validates
it against the evaluation config and every other lane plan, and does not import
or instantiate LIBERO or a model. If every lane dies before writing a plan, it
writes an explicit partial report with `counters=null` and
`metadata.plan_unavailable=true`, rather than inventing a resolved suite.

The pure `aggregate_lanes()` and durable `write_global_report()` functions accept
`run_root`, `run_id`, a global `BenchmarkPlan`, and `lane_count`. Fan-in validates
lane identity, complete per-lane episode coverage, duplicate work within a lane,
run/task/seed/init-state identities, terminal manifest/summary consistency, and
summary counters. Repeated task/episode identities across different lanes are
expected and remain separate through lane metadata. Invalid records are not
merged. Counters are recomputed from accepted canonical facts, not added from
lane summaries. Native model failures remain ordinary completed evaluation
results, not infrastructure errors. Artifact references remain absolute lane refs.

Coordinator exit code: 0 for complete, 1 for partial/error. Global writes are
exclusive; a second coordinator or reuse of an existing global output is rejected.

## Verification

Real RTP rollouts and the GPU trace are bounded to five lanes; the probe
evidence and the unsupported Nsight/driver caveat are recorded in the PR
review discussion.
The 1..20 configuration bound is not 20-lane capacity or long-duration certification.

## Capacity And Failure Limits

Environment and evaluator disk preflight check total demand for all replicated
lane plans: `ceil(aggregate_estimated_bytes * safety_factor) + reserve_bytes`.
Artifact estimates and report overhead scale with lane count; the free-space
reserve is added once. This is a free-space check, not disk reservation, and
does not prevent another run/process consuming space after the check.

Bounded model admission does not throttle simultaneous environment resets,
MuJoCo/EGL contexts, artifact writes, software RTP codecs, or DDS/RTP resources.
Measure these peaks before 8/20-lane or long-duration runs. Queue wait counts
toward request deadlines, but observations sampled before admission can age
while waiting; freshness at execution is not guaranteed.

A fatal shared-runtime/CUDA-context failure is a run-level infrastructure error
and triggers fail-fast for all lanes, not lane-local recovery. There is no
automatic shared-model restart or episode reassignment. Shared PI05 cancellation
is unsupported; see [shared inference limits](../inference_service/SHARED_BENCHMARK.md#isolation-and-limits).

## Focused Tests

```bash
source .shrc_local && ROS_DOMAIN_ID=88 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  pytest -q src/benchmark/benchmark_runtime/test
```

Plugin autoload is disabled because the installed ROS launch_testing plugin's
pytest hook signature is incompatible with the installed pytest. These tests
exercise pure planning/fan-in, full-plan per-lane supervisor/ledger progression, and actual
rclpy node parameters/timer-driven coordinator behavior without a simulator/model.
The new executable entry point requires refreshing the benchmark_runtime build
overlay before `ros2 run benchmark_runtime benchmark_coordinator_node` is used.
