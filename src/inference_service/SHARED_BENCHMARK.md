# Shared LIBERO Inference Processes

`benchmark_shared_policy_node` hosts independent legacy `PipelinePolicyNode`
frontends and exactly one `InferencePipelineManager` / shared policy weight load. It does
not alter the normal `pipeline_policy_node` distributed or scheduled executable.

## DDS Launch Interface

Pass a ROS string parameter `lane_configs_json`: a JSON array of complete
`PipelineNodeConfig` dictionaries, each with an additional required, unique
`node_name`. Optional dataclass fields retain their defaults. Example element:

```json
{
  "node_name": "inference_policy_lane_000",
  "pipeline_id": "policy",
  "model_path": "/absolute/path/models/pi05-libero-50k",
  "deployment": "torch-cuda",
  "execution_mode": "monolithic",
  "request_timeout": 60.0,
  "default_task": "",
  "runtime_options_json": "{}",
  "robot_config_path": "/absolute/path/generated/lane_000.yaml",
  "use_sim": false,
  "action_server": "/benchmark/lane_000/inference/policy/dispatch",
  "reset_service": "/benchmark/lane_000/inference/policy/reset",
  "health_topic": "/benchmark/lane_000/inference/policy/health",
  "action_topic": "/benchmark/lane_000/actions/policy",
  "request_topic": "",
  "result_topic": "",
  "heartbeat_topic": ""
}
```

The parameter contains JSON itself, not a filename. `robot_config_path` refers
to each lane's generated robot YAML. Topics in those contracts must already be
lane-specific; this process never derives lane identity from session-open order.
Pipeline IDs may be identical or different. All lanes must select the same
resolved bundle path, deployment, and runtime options. Action/reset/health/action
publication endpoints must be unique absolute names. Observation topics must be
disjoint across lane contracts. Node-global remaps are intentionally ignored by
frontends, so a launch `name=` cannot rename every frontend to the same name.

Launch one process, not one process per lane:

```python
Node(
    package="inference_service",
    executable="benchmark_shared_policy_node",
    parameters=[{
        "lane_configs_json": json.dumps(lane_configs),
        "max_in_flight": 1,
        "queue_size": 64,
    }],
)
```

Replace the original per-lane distributed edge/cloud inference processes with
this process. Keep the existing `ActionDispatcherNode`, `executor_type=benchmark`,
`scheduler_mode=wait_for_feedback`, evaluator, and environment nodes. Point each
dispatcher's `inference_action_server` and `inference_reset_service` to its entry
above, and the evaluator's health subscription to that entry's `health_topic`.
The public protocols remain `DispatchInfer`, `Trigger`, and `DiagnosticStatus`.
No scheduled-session APIs or distributed request/result/heartbeat are used here.

Before rebuilding the console entry point, the equivalent module invocation is:

```bash
source .shrc_local && python3 -m inference_service.benchmark_shared_policy_node \
  --ros-args --params-file /absolute/path/shared-ros-parameters.yaml
```

## RTP Shared Cloud Interface

For RTP + distributed execution, use one `benchmark_shared_cloud_node` instead
of the DDS host. It owns one `CloudBackendRuntime` / shared policy weight load and a
`PureInferenceNode` frontend (including receiver state) for each lane. Each lane
also has a distributed edge `pipeline_policy_node`; the environment publishes
RTP images and DDS state. There is no global scheduler or dynamic episode queue
in either shared benchmark topology.

Pass `cloud_configs_json` as a ROS string containing a non-empty JSON array of
`CloudNodeConfig` dictionaries, not a filename. Example element:

```json
{
  "pipeline_id": "policy",
  "model_path": "/absolute/path/models/pi05-libero-50k",
  "deployment": "torch-cuda",
  "request_timeout": 60.0,
  "runtime_options_json": "{}",
  "robot_config_path": "/absolute/path/generated/lane_000.yaml",
  "request_topic": "/benchmark/libero/lane_000/inference/policy/request",
  "result_topic": "/benchmark/libero/lane_000/inference/policy/result",
  "heartbeat_topic": "/benchmark/libero/lane_000/inference/policy/heartbeat",
  "video_descriptor_topic": "/benchmark/libero/lane_000/inference/policy/video/descriptor",
  "video_status_topic": "/benchmark/libero/lane_000/inference/policy/video/status"
}
```

All entries must use identical pipeline ID, model path, deployment, decoded
runtime options and request timeout. The five endpoint topics above must be
absolute and disjoint across all entries, and match the corresponding edge.
Each generated lane YAML supplies its own materialized transport contract,
RTP ports and stream identities. Do not pass DDS-only `node_name`,
`execution_mode` or per-entry `max_in_flight` to `CloudNodeConfig`.

```python
Node(
    package="inference_service",
    executable="benchmark_shared_cloud_node",
    parameters=[{
        "cloud_configs_json": json.dumps(cloud_configs),
        "max_in_flight": 1,
        "queue_size": 64,
    }],
)
```

Both hosts consume top-level ROS parameters `max_in_flight` (default 1) and
`queue_size` (default 64). `queue_size` is waiting capacity, not the active-slot
count, and must be at least `max_in_flight`. DDS entries must also set their
`max_in_flight` to the host value. `max_in_flight` is any positive int,
model-neutral at the resolver; the shared owner enforces the contracts it can
actually host.
The unified builder supplies both limits from `benchmark.evaluation.inference`.
Those YAML fields are validated even in serial mode but only take effect with
`parallel.enabled=true`, including a single lane with parallel enabled.

## Isolation And Limits

- The shared owner admits any Torch policy deployment with a `policy`
  interface, request-scoped state, and direct execution, regardless of model
  type. RTC processors are rejected (`select_action` is rejected with them);
  `predict_action_chunk` is the only supported action method.
- The generic Torch session currently treats a policy's `reset()` method as
  evidence of statefulness. This executable uses a dedicated session subclass
  that treats chunk prediction as request-only; `select_action` is rejected.
  Loaded capabilities explicitly declare `stateful=False`. Other executables
  retain their original capabilities.
- `max_in_flight` is any positive integer; the session default remains 1. Both
  hosts use the configured waiting capacity. Loaded session capacity and runtime
  admission must cover the requested limit. Concurrent sessions serialize the
  pre/postprocessors, retain thread-local execution metadata, and run
  model calls on synchronized worker-local CUDA streams. `stateful=False` alone
  is not evidence of thread safety. The tested scope (probe and rollout
  evidence recorded in the PR review discussion): CUDA isolation at 2/5, real
  five-lane RTP at inference limits 2/5.
  The real five-lane GPU trace records cross-lane kernel overlap, but Nsight
  Systems 2024.6.2.225 reports driver 13.2 as unsupported and uses 12.8 tracing
  libraries. This is bounded evidence, not supported-toolchain timing
  certification, sustained five-way execution, throughput or 20-lane validation.
- Deadlines include queue wait. Observations are sampled/assembled before
  admission; freshness checks at sampling do not guarantee fresh observations
  when model execution begins. There is no post-admission resampling guarantee.
  Capacity runs must measure queue wait and observation age at execution.
- Shared PI05 still declares `supports_cancellation=false`. Cloud `CANCEL`
  explicitly returns `unsupported_capability`. Cloud admission does not propagate
  cancellation. For other runtimes that support cancellation, queued/unknown
  targets and registration races still need protocol-level handling; a successful
  cancellation response does not establish that queued work was removed.
  FIFO waiter-cancellation unit tests do not establish
  end-to-end PI05 cancellation. Running CUDA is not preempted, and its slot remains
  occupied until execution returns. A deadline does not interrupt a running kernel.
- Reset drains only that frontend's operation lock and never resets or closes the
  shared runtime. It preserves the fresh observation delivered by ResetBenchmark
  before PreparePolicyEpisode, matching the existing benchmark lifecycle. Other
  lanes' observation buffers, in-flight goals, and model work are untouched.
- Policy queues are not shared because this API never uses them. Random noise
  still uses Torch's process RNG: per-lane deterministic seed streams and batching
  are not provided. Interleaving can change sampled trajectories.
- A fatal shared-runtime/CUDA-context failure is a run-level infrastructure
  failure: fail-fast stops all lanes. Lane reset cannot recover a broken model;
  no local recovery, automatic restart, episode retry or backend fallback is
  provided. Restart only as a new run after resolving the fault.
- The legacy shared executable is DDS-only. RTP uses
  `benchmark_shared_cloud_node`, with one distributed edge and RTP receiver per
  lane and one shared cloud/model owner.

## Request Trace Identity

Distributed inference forwards the raw wire request ID to the shared runtime.
With `IBROBOT_PROFILE_SHARED_POLICY=1`, the model NVTX label is
`shared_policy:<request_id>` (the 2026-09-16 nsys trace, recorded in the PR
review discussion, predates the rename and shows `shared_pi05:<request_id>`).
Join it to cloud logs while retaining session/generation/lane context and checking
for duplicate IDs. Neither a raw request ID nor a CUDA stream is a global lane
identity. Session-qualified runtime IDs are not implemented: concurrent reuse of
an ID across sessions can collide in runtime request-context bookkeeping. This
remains a cancellation/trace-isolation limitation, not a demonstrated result mix-up.

## Verification

The shared hosts are verified through the real parallel LIBERO evaluation on
the shipped profiles and the `models/pi05-libero-50k` Torch CUDA bundle:

```bash
source .shrc_local && ROS_DOMAIN_ID=<id> ros2 launch robot_config robot.launch.py \
  robot_config:=franka_libero_evaluation     # DDS topology, 5 lanes
source .shrc_local && ROS_DOMAIN_ID=<id> ros2 launch robot_config robot.launch.py \
  robot_config:=franka_libero_evaluation_rtp # RTP topology, 5 lanes + shared cloud owner
```

Reference runs on an RTX 3090 (five lanes, `max_in_flight=5`): the RTP run
`20261013_043111_libero_object_franka_libero_evaluation_rtp` and the DDS run
`20261013_042854_libero_object_franka_libero_evaluation` both completed 10/10
episodes with 0 infrastructure errors and zero RTP packet loss. PI05 CUDA
concurrency isolation evidence is recorded in the PR review discussion.
Dedicated pytest coverage for shared-session admission, FIFO deadline/cancel
and failed fan-in is not part of this PR; add it before relying on those
paths beyond the audited profiles.
