# inference_service

`inference_service` is IB-Robot's unified inference runtime. It selects one model bundle and one named deployment,
then runs the model through Torch, Ascend, Hisilicon, RKNN, or HMM. Policy models additionally use a stable pipeline
ID and support monolithic or distributed edge/cloud execution.

There is no compatibility layer for the removed runtime architecture. A backend is not selected with a launch
`device` argument. The runtime does not load per-backend sidecar manifests, scan directories for conventional
artifact names, or use environment variables to override artifacts.

Local requests use `ModelRuntimeHandle.execute(ModelRequest, ExecutionContext)` and return `ModelResult`.
Independent producer frames use `ModelRuntimeHandle.submit_frame(ModelRequest, ExecutionContext)` and return a
future. They share request registration, cancellation, failure reporting and control draining with ordinary execution.
Frame admission is bounded separately so a running producer can overlap an admitted Dispatch; diagnostics count both.
`ModelRuntimeFactory` is a registered construction surface, not the request boundary. A policy reaches the handle
through its `InferencePipeline` facade. A typed plugin constructs a `ModelSession`, places it in a
`RuntimeAssembly`, and transfers the assembly to a handle. The handle owns public lifecycle, admission, deadlines,
cancellation, health, and close draining; the session resource owns and releases vendor model objects, device
leases, buffers, and workers.

## Core Concepts

### Backend completion and cleanup evidence

`runtime_context_defaults` scopes the `priority_scheduling` and `managed_execution` defaults to one context.
Explicit values take precedence; leaving the context restores the previous values and other threads do not inherit them.
Pipeline and Manager factories inherit the scoped priority default when the argument is omitted; explicit False overrides it.
Standalone `AscendOmModelSession` construction reads the same priority default.
The existing monolithic policy node sets both defaults from `robot.runtime_scheduler.enable`:
scheduled policy Sessions use managed completion, while disabled scheduling loads direct Sessions.
Typed model services do not yet consume that switch; the policy interface does not install a perception scheduler.
`TorchModelSession` and `LeRobotTorchModelSession` share `torch_execution_scope()`. Failures preserve the original
exception and its execution, recovery and cleanup evidence. Synchronization supplements completion evidence for work
that has started; explicit pre-execution rejection stays NOT_STARTED. Successful inference adds a
completion wait only with `managed_execution=True`. Failure and close wait in both modes; LeRobot reset also waits
before mutating policy state. A direct result discarded on delivery timeout or cancellation also waits for device
completion; a failed wait retains an unknown outcome. The public Session API remains available.
Raw cancellation or deadline exceptions inside a callable still require a device fence; their default flags are not
native proof that work never started.
CUDA and NPU Sessions resolve a concrete device index during load; execution and Close fence that device even if the
calling thread later selects another current device.
LeRobot preprocessing and postprocessing may also enqueue device work and use the same completion helpers.
Processor completion does not imply that policy model state advanced. Request validation errors after
preprocessing retain device evidence without recording a persistent policy failure.
Managed processor success waits for completion; normal direct success adds no fence. If a later pipeline checkpoint
discards a direct Session result, the failure boundary fences it before reporting completion. Multiple fences combine
their evidence without erasing an unconfirmed operation. The frame tracks participating Sessions that still need a
completion fence; it reuses error completion evidence and never waits on idle components.
Managed CUDA/NPU success waits only for the current stream captured on callback entry. Callbacks must submit on that
stream or join child streams back to it before returning; unjoined background work is unsupported. Failure, Close
and fault recovery retain conservative device-wide waits. `force_completion_on_error` controls only exception handling;
`complete_on_success` independently controls successful-return completion. SpeechDirection FullSubNet Torch
also uses completion checks on execution failure and fences fault recovery/Close, retaining resources if completion fails.
A device stuck in a sticky error never reports completion; after 2 consecutive failed close drains the
executor abandons the wait, releases the references, and lets close terminate. The abandoned completion
state is unknown and the abandonment is logged.
Healthy FullSubNet reset reuses completion proved by its host output and does not wait for unrelated streams.
After reset initializes state, only the CUDA stream submitting that initialization is synchronized.
Execution errors retain secondary synchronization diagnostics. The Torch error boundary clears unwound inference
locals. Both direct and policy failure diagnostics keep a snapshot without an exception chain or traceback, so rejected request and result
payloads are not retained by the Handle.
FullSubNet Torch registers rollback before device construction.
It also pins the CUDA index at construction so models, state and completion fences use the same device.
ACL reports secondary release failures through `details.cleanup_error`, preserving both errors across normalization
and serialization. Failed D2H host-buffer release quarantines native pointers and clears unwound NumPy output frames.

ACL, HMM and RKNN owners retain resources after failed cleanup. ACL retries unfinished resources;
RKNN/HMM require explicit nonstart evidence before retrying an SDK release, as described under Close failures below.
Once JPEG codec resources are released, Close retries only the remaining ACL lease cleanup without rebinding a
destroyed context. HMM, Ascend, Hisilicon and ONNX pre-submission input errors report NOT_STARTED. ONNX reuses
Session role-progress merging so a later rejection preserves earlier execution. RKNN output validation after a
successful native inference retains COMPLETED evidence, including aggregation across multiple roles.
ONNX validates binding dtype, rank and static dimensions before the native call using the existing tensor validator.
Ascend input diagnostic capture is also inside the pre-submission input boundary.
ACL synchronous and shared-stream output errors retain completion after successful execution or synchronization.
Static dataset banks reuse ordinary ACL input preparation and host output reading. Rejected bank inputs
report NOT_STARTED; output errors after completion still require recovery of advanced model state.
HMM output extraction and conversion, and Ascend, Hisilicon and ONNX role output mapping also retain completion.
Host output failures preserve state mutation and recovery requirements without retaining idle device admission.
HMM, RKNN, ACL priority stream pool and speech ACL runner constructors accept an optional `cleanup.defer` owner. When supplied, that external
rollback owns partial construction. Without it, constructors attempt Close and attach `cleanup_owner` to the original
initialization error if cleanup remains pending. Constructor rollback errors use `rollback_errors` consistently.
ASR import errors preserve their cause and cleanup owner. Both failed startup and normal shutdown retain cleanup
ownership; normal shutdown also attempts executor and ROS cleanup. The ASR node retries failed construction cleanup,
including Handle load failures and post-load sample-rate or VAD initialization errors. VAD owners retain their
Handle and provider until release succeeds; partially initialized VAD runtimes remain owned by the node.
The ASR constructor also transfers the node through `cleanup_owner` after ROS interface or timer initialization failure.
Both the entry point and embedded callers can retry cleanup through that owner.
Hisilicon Close retains a worker that has not exited and rejects subsequent start/execute calls.
An execution-time worker exit preserves `worker_exited` and unknown-outcome evidence and requires runtime reload.
Hisilicon request writes and response waits observe deadlines, cancellation and protocol Close. Writes use a
nonblocking pipe and resume after short writes. Interrupted partial requests keep unknown outcomes, require reload,
and make the byte stream unusable; the existing process owner retains cleanup responsibility. Interrupted responses
and protocol errors also require reload. Response cancellation or timeout marks the protocol unusable before another
request can be sent, including direct Session/protocol calls. Once I/O exits, Close can terminate and reap the worker.
An unbounded Handle request still waits unless its caller cancels it.
Video codec probing treats FFmpeg driver errors as an unavailable device.

### Resource rollback and lifecycle budgets

`LoadRollback` releases resources in reverse order and retains a failing callback and its dependencies for retry.
`ModelSession` and the Factory share this rollback mechanism. Factory errors expose a retryable `cleanup_owner`.
`RuntimeProviders.retain_cleanup()` retains failed Factory and plugin construction owners until shutdown retries them.
Perception and ZipVoice plugins first close a failed Handle and retain it only when cleanup is unfinished.
ZipVoice marks itself closed only after Handle Close succeeds and keeps the original load error when rollback fails. Manager startup failures expose
their cleanup owner. Pipeline and Model Service nodes transfer pending shutdown to providers before dropping ownership.
If ROS entity construction fails after a model loads, the constructor runs the same node cleanup path.
Provider shutdown failures preserve close_pending=True and the original cause chain.
Service exit, including SpeechDirection, attempts provider cleanup and ROS shutdown even when node destruction or
provider cleanup fails.
SpeechDirection timer construction failure closes its already started runtime and transfers pending cleanup to
providers. The entry point also covers node and dependency construction failures.
These owners close before the admission provider and ACL provider. `process_resources` supplies the shared resource
list for process cleanup and Handle ownership filtering.
`InferencePipelineManager` retries unfinished cleanup in direct and priority modes alike.
Cleanup retry is unconditional; the manager has no retry toggle.
`lifecycle_budget` shares one deadline across nested cleanup; `lifecycle_scope` propagates a control call's deadline
and cancellation token. SpeechDirection thread joins poll this shared budget and cancellation token.
The inherited deadline covers startup checks, control-lock waits, ACL completion waits and Hisilicon process waits.
Control-lock and quiescence waits observe the control call's cancellation token.
Hisilicon EOF, terminate, kill and stderr-thread waits poll cancellation in slices of at most 50 ms,
retaining worker ownership for a later Close retry when interrupted.
Nested control calls without an explicit context inherit the parent token and use the stricter deadline.
Rollback, provider and ACL ownership lock waits use the same budget. HMM and RKNN check startup before later native
initialization.
ACL cleanup checks the remaining budget between individual release calls and retains unfinished resources for retry.
An already running SDK call without cancellation support must finish; the budget does not forcibly interrupt it.
Handle checks startup before loading, before each resource load and before publishing READY.
Derived runtime contexts preserve `managed_execution`. Multi-role requests and pipeline stages pass through existing
metadata. `StageTicket` is an optional identity utility; production scheduling does not create or consume it automatically.

### Handle execution policies and streams

`bind_execution_policy` installs one immutable `RuntimeExecutionPolicy` and returns a local dispatch authority.
Native Sessions must load with `managed_execution=True`. Binding a direct-loaded Session is rejected;
binding before load requires the same managed context at load time.
Public execute, stream, reset and close operations use that policy without falling back to direct execution after
failure. Policies can also support frame submission. `StageTicket` is an immutable handoff identity
with no scheduler dependency.
The base policy requires boolean `ready` plus `execute/reset/close/cancel`; it owns cancellation of queued requests
that have not entered local execution. Streaming policies also require
`open_stream/step/reset_stream/close_stream`. Binding validates the required interface and requires an idle Handle.
Stream runtime assembly also requires those four methods; missing control methods cannot count as successful cleanup.
A context provider must preserve the caller's request ID and cancellation token and may only shorten its deadline.
The authority also remains accepted by the compatible execute/step `_dispatch_key` argument.
Stages use the existing `StagedModelExecutor`; Handle does not maintain a second stage control interface.
Policy exceptions and returned frame Futures share failure normalization at completion.
`drain_managed` validates dispatch authority,
revokes admission, holds the control lock and propagates the lifecycle budget. Drain does not reopen readiness.
Drain also acquires existing stream control locks so it cannot overlap stream reset or close.
Every declared native Session and additional executable component must support drain; otherwise drain rejects
the composition before returning any capacity. Without declared Sessions, the executor itself must support drain.
Drain expands nested Factory assemblies, handles self links and deduplicates native Sessions by identity.
Orchestration wrappers are not required to prove device completion.
Exclusive stream reset uses the runtime control lock as well; drain revokes admission again after taking that lock
in case an earlier reset reopened readiness while drain was waiting.

Handles without a bound policy continue to use direct execution. `ready` includes policy
availability, while `local_ready` reports backend state. Deadlines and cancellation apply to local and control
operations. Ordinary requests, frames and stream steps share active request identities and cancellation registration;
policies and frame-capable executors require unique active IDs. Unbound synchronous direct execution continues to allow
duplicate IDs; cancel signals every matching active context, and completion removes only that execution's registration.
Cancellation callbacks run synchronously outside the Handle condition lock and must return promptly. They may signal
another owner but must not wait for execution completion or subsequent cancellation hooks. Arbitrary blocking Python
callbacks are unsupported and cannot be interrupted by the cancellation budget. Cancellation signals local tokens, notifies the policy,
then invokes the executor hook. A notification
exception does not suppress the other hook. Executor signature compatibility is checked before calling, so an
internal TypeError does not cause a duplicate call. Stream steps share instance and resource-domain
admission and respect the Handle execution limit. Known unstarted admission rejections do not count as model failures
at either request or stream entry points. Unknown native work retains its instance and shared-domain slots
until successful recovery, explicit drain, or Close, so another Handle cannot reuse unconfirmed capacity.
Request and stream failures use the same evidence conversion before leaving native admission; a raw control
exception after submission cannot prove completion. A successful per-stream Close must drain that bank before its
slots are returned. Runtime-exclusive logical Close and single-stream recovery cannot release runtime-fault slots.
`reset_stream()` rejects runtime recovery requirements; `reset()` must restore the runtime before admission reopens.
Voice ASR's VAD reset uses runtime reset and clears its host audio context only after recovery succeeds.
Host stream cleanup within runtime reset keeps capacity retained until all native Session resets succeed.
Without an independent native reset, an active stream uses its full reset_stream barrier.
If an exclusive stream is already closed and no native reset is available, recovery returns
reset_unsupported and retains the capacity. The caller must close and rebuild the runtime.
Unknown work in an exclusive bank retains runtime recovery even when stream Close is already pending.
Queued local requests check existing Handle/stream control state while waiting for instance or domain capacity,
so Close/reset can drain their own waiters without depending on those waiters acquiring retained slots.
Direct admission waits observe cancellation and check again after grant. Frame admission errors synchronously raise
`ExecutionFailure`; admitted frames deliver through a Future. Policy delivery rechecks deadline and cancellation for
execute, frame, step and open_stream. Completed Sessions and frames keep COMPLETED evidence on delivery
timeout/cancellation. These work entry points also reject expired or cancelled contexts before policy dispatch with
NOT_STARTED evidence; control entry points retain irreversible Close semantics. The Pipeline preserves legacy
exception types and carries normalized evidence to ROS.
Known execution with pending
cleanup also revokes admission. State-bank isolation sets the default recovery scope, but explicit runtime reset
or reload requirements remain runtime-wide, even during stream Close. Outcome certainty stays unchanged.
Failed delivery after a stateful step advances its bank requires recovery even with known completion. Completion
releases native capacity but does not prove that the caller advanced its matching host state.
Request failures also enforce explicit recovery for known outcomes and never downgrade RELOAD to reset.
The delivered exception, last failure snapshot and Handle state share the final recovery decision.
After reset completes, recovery finalization precedes the delivery deadline/cancellation check. A late control failure
keeps COMPLETED evidence and does not quarantine the recovered runtime again.
Stream-scoped CLOSE_STREAM and RELOAD block further steps and cannot be replaced by reset_stream. For unknown
runtime-exclusive work, these requirements escalate to runtime RELOAD. Explicit RELOAD after reset failure also
prevents another reset, regardless of the reset-retry configuration.
Failures observed while reset waits update the recovery state; a drain timeout cannot restore stale READY.
Stream reset rechecks stream state and recovery after waiting under the condition lock and preserves new CLOSE_STREAM
or RELOAD requirements. Local stream reset/close errors use the same recovery path as step, including runtime-wide
admission revocation. Full reset/close/drain_managed reentry from synchronous execute/step is rejected before waiting or changing state.
Opening threads participate in the same runtime control reentry check. Same-stream reset/close reentry is
rejected at the public boundary before policy dispatch or closing intent, leaving the stream usable.
Stateless executor errors without completion proof remain unknown, just like errors from the dispatch policy.
The Handle does not cache raised Close exceptions, so a successful retry does not retain their call stacks.
RKNN empty-output checks and array conversion preserve completed native execution. Public Session validation
does the same; direct Torch validation failures reuse the existing device completion fence.
Hisilicon cancellation or expiry before the protocol lock or the first request byte reports NOT_STARTED and
leaves the worker usable. Read/write polling supports pipe descriptors numbered 1024 or higher.
Role entry cancellation reports NOT_STARTED; the common Ascend role boundary preserves completed predecessor
evidence for both generic and host-orchestrated Sessions. Input size errors identify the role, input index, actual
bytes and expected bytes.
Sequential and staged executors accumulate frame evidence across roles and loop iterations. Visual snapshots do not
carry old-frame evidence into a new Action request. RequestDirectExecutor also preserves predecessor execution and
state mutations. Input validation before native submission reports NOT_STARTED. Synchronous frame submission failures
use native start evidence even before a Future exists, and open_stream preserves native rejection evidence.
Policy sync and Future failures use
the same control-error conversion; only an explicit native entry check proves non-start.
An interrupted SpeechDirection hop explicitly requires reset even when its device work is known complete.
An explicit native RELOAD requirement and its availability survive the stream adapter.
A later role's NOT_STARTED evidence cannot erase Host or predecessor progress. The implicit reset on open_stream
inherits the opening request's lifecycle context.
SpeechDirection reset uses runtime recovery while preserving the stream ID. The production stream layer clears Host
state while Handle resets each owned native Session once. Public reset_stream has no internal control-lock bypass;
stream-only reset still clears the complete stream state. Its exclusive composition supports one
shared device domain; mixed Torch FullSubNet and Ascend VAD still acquire the Ascend lease. Multiple domains or
conflicting limits are rejected at construction.
The shared-domain slot covers the whole audio hop, including Host and other-backend work. This safety boundary also
applies with runtime scheduling disabled; it is not per-role scheduling or a latency guarantee.
The Handle rejects later inputs until recovery. Stream reset records its intent before waiting for an active step.
Close never reopens admission and retains unfinished cleanup for retry.
Handle owns the whole assembly, including constructed components whose load was not reached. Policy close still
requires completion after local close finishes. Local and policy frame delivery share the same reentry guard.
Handle rejects reset/close from inline frame-result callbacks before
draining; an external owner performs those operations. Direct staged-executor Close from its worker still returns
pending. Worker joins obey the inherited lifecycle deadline and successful Close clears the final visual snapshot.
An ACL completion callback cannot wait
for queued tasks on its own worker. Normal D2H buffers remain local; only failed frees transfer to quarantine.
Stream close records irreversible intent before waiting or policy dispatch; reset cannot reopen that stream. Late open
results and open/Close races retain native cleanup ownership. Stream calls validate their owning
handle; successful Close releases native state and retains at most 128 closed records. Unconfirmed stream cleanup
keeps its resource ownership independently of that history limit.
SpeechDirection also bounds its native closed history to 128 entries while old owned closed objects remain idempotent.
SpeechDirection stop, compatibility wrappers and node shutdown preserve pending cleanup ownership.
Node shutdown also closes a Handle created before Runtime construction failed. A later pending cleanup cannot be
hidden by an earlier worker-wait error; final worker and Handle state determine whether cleanup needs retry.
SigLIP2, SAM2 automatic and GraspGen carry request priorities through each role to the actual OM execution stream.
SAM2 checks cancellation and deadlines before constructing decoder inputs and before final postprocessing.

### Policy Bundle

Every deployable policy directory contains the LeRobot semantic files and exactly one
`inference_manifest.json`:

```text
policy_bundle/
├── config.json
├── model.safetensors                         # when required by a Torch deployment
├── policy_preprocessor.json
├── policy_preprocessor_step_*.safetensors
├── policy_postprocessor.json
├── policy_postprocessor_step_*.safetensors
├── tokenizer/                                # when required by PI0.5 or SmolVLA
├── artifacts/
│   ├── ascend/<deployment>/...
│   ├── hisilicon/<deployment>/...
│   ├── rknn/<deployment>/...
│   └── hmm/<deployment>/...
└── inference_manifest.json
```

LeRobot owns `config.json`, processor JSON, processor state, tokenizer assets, and native weights. IB-Robot reads
those files without adding fields, removing fields, rewriting devices, or materializing a temporary policy
directory. All deployment metadata belongs in `inference_manifest.json`.

### Deployment

One manifest may declare multiple named deployments for the same policy, such as `cpu`, `cuda`, `rk3588`,
`ascend_310p3`, or `lq50`. A pipeline selects a deployment name, not a backend name.

A deployment uses an explicit v3 runtime profile for the backend, target, and instance fields:

```json
{
  "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
  "revision": 1,
  "execution_contract": {
    "state_scope": "request",
    "execution_structure": "direct",
    "cancellation_granularity": "request_boundary"
  },
  "runtime_profile": {
    "backend": "torch",
    "target": {"runtime": "torch"},
    "profile": {"device": "cpu"}
  }
}
```

A compiled deployment declares its target, artifacts, execution order, and complete runtime ABI bindings:

```json
{
  "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
  "revision": 3,
  "execution_contract": {
    "state_scope": "request",
    "execution_structure": "direct",
    "cancellation_granularity": "request_boundary"
  },
  "runtime_profile": {
    "backend": "rknn",
    "target": {
      "soc": "rk3588",
      "runtime": "rknn-lite2"
    },
    "profile": {
      "target_name": "rk3588",
      "core_mask": 7,
      "device_id": 0
    }
  },
  "artifacts": {
    "policy": {
      "path": "artifacts/rknn/rk3588/generations/<uuid>/policy.rknn",
      "format": "rknn"
    }
  },
  "execution": ["policy"],
  "bindings": {
    "policy": {
      "inputs": [
        {
          "semantic": "observation.state",
          "runtime_name": "observation.state",
          "index": 0,
          "dtype": "float32",
          "shape": [1, 6]
        },
        {
          "semantic": "observation.images.top",
          "runtime_name": "observation.images.top",
          "index": 1,
          "dtype": "float32",
          "shape": [1, 480, 640, 3],
          "layout": "NHWC"
        }
      ],
      "outputs": [
        {
          "semantic": "action",
          "runtime_name": "action",
          "index": 0,
          "dtype": "float32",
          "shape": [1, 100, 6]
        }
      ]
    }
  }
}
```

Every role in `execution` must have an artifact and a non-empty binding group. Image bindings must explicitly
declare `NCHW` or `NHWC`; non-image tensors are never transposed from rank alone. Multi-module deployments use
matching `internal.*` semantics or `device_links` that declare producer, consumer, device-pointer ownership, and
inference lifetime.

### Pipeline

The pipeline ID is the stable model-instance and ROS-routing identity. It must match
`^[a-z][a-z0-9_]{0,62}$`. Each pipeline independently owns:

- its policy bundle and named deployment
- its LeRobot preprocessor and postprocessor
- its policy codec and binding execution plan
- its `ModelRuntimeHandle`, admission state, and lifecycle
- its action, reset, health, action-output, and distributed transport endpoints

`InferencePipeline` is the policy facade that adapts the existing policy contract to `ModelRuntimeHandle` while
adding LeRobot processors, policy codecs, and action adaptation. The handle owns lifecycle, admission, deadlines,
cancellation, health, and diagnostics, and loads then releases the components transferred in its `RuntimeAssembly`.
Compiled models run through sequential or staged executors sharing `ComponentModelExecutor` helpers; iterative families
use `IterativeStage` to invoke roles. One `ExecutionContext` carries the request ID, deadline, and cancellation token
through every stage and `ModelSession` resource, while the session owns vendor resources shared across those roles.
Both strategies use `ModelSession.execute_role()` for role validation. Ascend selects isolated datasets internally,
while reusing semantic binding and diagnostic capture. The PI0.5 policy adapter validates the request-scoped
`vlm` / `action_expert` prefix-cache ABI, excluding state, iterative and reverse-link dependencies from the producer.
The manifest describes the model ABI; loaded runtime capabilities establish asynchronous execution and isolation.
The loaded backend must expose isolated async role execution and priority streams.
The executor owns triggers, immutable snapshots, prompt/freshness checks and generation fencing.
Private binding actions/status serve policy/VLA only. Deadline admission support comes from the actual executor,
not hardware, and is not a finish guarantee. Public priority uses one continuous upper bound.
Private Close retries use `(session_id, operation_id)` and still pass lifecycle/drain admission; successful drain
must advance the pipeline generation, except for explicit generation-zero cleanup requiring no work.
Perception, Echo, TTS, and other typed plugins use the direct
`ModelServicePlugin -> RuntimeAssembly/ModelRuntimeHandle -> ModelSession resource` composition path.

Default endpoints:

| Interface | Default |
| --- | --- |
| local node | `inference_<pipeline_id>` |
| cloud node | `inference_<pipeline_id>_cloud` |
| action server | `/inference/<pipeline_id>/dispatch` |
| reset service | `/inference/<pipeline_id>/reset` |
| health topic | `/inference/<pipeline_id>/health` |
| action output | `/actions/<pipeline_id>` |
| distributed request | `/inference/<pipeline_id>/request` |
| distributed result | `/inference/<pipeline_id>/result` |
| distributed heartbeat | `/inference/<pipeline_id>/heartbeat` |

### Scheduled Control Plane

`robot.runtime_scheduler.enable` is the only scheduler switch and defaults to `false`. When enabled,
the hardware launch graph uses Global Open/Dispatch/Close actions and the scheduled action dispatcher. Only monolithic pipelines
are supported; they use the whole-graph sequential executor by default, while a supported functional manifest may opt
into independent staged execution.
Simulation and benchmark always run directly. A disabled inference control mode stays disabled.
The existing policy scheduler, FIFO/EDF, profile schema and public session contracts remain unchanged.

When the switch is `false`, a complete scheduled configuration may remain in the SSOT as dormant configuration; the
launch graph, node parameters, endpoints, executor sizing, and backend execution still use the legacy path. When the
entire `runtime_scheduler` block is absent, scheduled fields remain unknown so legacy configurations keep strict typo checking.

A downstream Close with known `NOT_STARTED` preserves its outcome and recoverability. The session stays `CLOSING`
without quarantining the pipeline solely for that rejection. A retry uses new operation IDs and only visits bindings
that have not drained; Dispatch stays blocked. Mixed `UNKNOWN` results take precedence and remain quarantined;
executed failures still return unsuccessful `COMPLETED`. An identity-validated successful Close reclaims the drained
binding's reservations and operations. Operations with live waiters are reclaimed after detachment;
late results cannot restore released ownership.

Independent producers accept `scheduling.stages.<producer>.max_snapshot_age_ms` (positive integer, default `5000`).
Only the first manifest role may configure it; sequential and terminal stages reject the option. Snapshots must match
the generation and prompt and remain within the observation age limit. The node converts ROS capture age to a
monotonic anchor, including producer queueing and execution time. Dispatch refresh preserves the same capture time;
direct runtime calls without capture metadata age from initial admission. A refresh that is still stale fails instead
of repeatedly recomputing the same expired observation.

Two lifecycle behaviors apply only when scheduling is enabled: reset skips explicitly stateless,
non-resettable non-executor components; stateful policy late/completed failures or
failures after execution started make subsequent `infer()` calls report not-ready until a successful reset.
Reset support discovered during LeRobot Torch session loading always enables actual `policy.reset()`, regardless
of whether scheduling is enabled, disabled or absent. Reset is therefore no longer a harmless no-op: a drain or
`policy.reset()` error records a failure on the Handle. Handles with `retry_failed_reset` enabled enter
RESET_REQUIRED and may retry reset; other handles enter FAILED and require RELOAD. The reset service then
returns `success=false` and later requests are rejected with `recovery_required`; reset callers must handle
the failed return. With the scheduler disabled or absent,
the legacy topology and synchronous execution remain. Cleanup retries are independent of scheduling: manager Close
retries retained `close_pending` pipelines in both modes. Torch failure, reset and close confirm device completion;
successful direct inference adds no synchronization. Handle cancellation and irreversible
stream Close also apply to direct execution. Tests cover launch parameters, late results, reset and Close behavior.

Priority `0` is highest. Its resource ordering and fallback depend on the scheduler policy:

| `global_policy` | `priority_zero_deadline_admission.enable` | Behavior |
| --- | --- | --- |
| `fifo` (default) | `false` (default) | Dispatch the target directly without a resource queue; reject nonempty fallback chains |
| `fifo` | `true` | Profile-based finish admission, per-resource FIFO, and fallback; sequential pipelines only |
| `edf` | `false` | Per-resource deadline ordering, FIFO ties, and fallback; sequential pipelines only, no profile-based finish prediction |

EDF never preempts active work and does not guarantee completion before the deadline. Deadline-driven priority-0
(EDF or FIFO profile admission) is a sequential-only contract: independent-stage pipelines split one request across
overlapping workers, so per-request deadline ordering can be neither evaluated nor honored. robot_config rejects
such targets and fallback entries at configuration load, and Global skips independent candidates by serving-status
capability. Unsupported default FIFO/fallback combinations fail configuration loading when the scheduler is enabled.
Missing or invalid profiles affect
only FIFO with profile admission enabled, not readiness, EDF, or nonzero priorities. Only `NOT_STARTED` permits fallback;
`UNKNOWN` remains quarantined and must not be retried. Nonzero priorities dispatch only the target and are exempt from
the sequential-only constraint.

Offline measurements define a p99 admission SLA rather than an absolute completion guarantee. Action-generation
profiles must match the pipeline input-contract fingerprint and declare `prompt_bytes_max`; Global selects the smallest
profile bucket that covers the current prompt, adds the configured safety margins, and fails closed outside calibrated
coverage. Profile identity is independent of endpoint names and routing membership.

Serving readiness requires every required pipeline to report at least one generic priority level. If the configured
default priority is greater than zero, readiness additionally requires the configured default target pipeline to be
online and expose that priority. Other pipelines are not required to support multiple priorities. Backends without an
explicit generic-to-native mapping accept priority `0` only; Ascend currently maps generic priorities `[0, 7]` one to
one to ACL stream priorities.

## Robot Configuration

Inference is configured directly under `control_modes.<mode>.inference.pipelines`:

```yaml
control_modes:
  model_inference:
    inference:
      enabled: true
      pipelines:
        policy:
          model_path: models/so101_act
          deployment: rk3588
          execution_mode: monolithic
          request_timeout: 5.0
          default_task: pick up the banana
          runtime_options: {}
```

A relative `model_path` is resolved only against the `WORKSPACE` environment variable. If `WORKSPACE` is unset,
configuration fails without falling back to the current directory, YAML directory, or source tree. Source the
project environment before project or ROS commands:

```bash
source .shrc_local
```

Multiple models are multiple pipelines, not a generic `concurrency` value:

```yaml
pipelines:
  action_policy:
    model_path: models/so101_act
    deployment: rk3588
    execution_mode: monolithic
  auxiliary_policy:
    model_path: models/auxiliary_smolvla
    deployment: cpu
    execution_mode: monolithic
```

YAML remains the default configuration source. For development, override one explicitly named pipeline:

```bash
ros2 launch robot_config robot.launch.py \
    config_path:=/absolute/path/to/robot.yaml \
    control_mode:=model_inference \
    inference_pipeline:=policy \
    inference_execution_mode:=distributed
```

An empty `inference_execution_mode` preserves YAML configuration. A non-empty override requires
`inference_pipeline`, preventing accidental global changes in multi-pipeline configurations.

Each pipeline may override endpoints in a typed `transport` mapping. Node names, actions, services, and topics
must remain unique across pipelines. A monolithic pipeline cannot configure cloud-node, request, result, or
heartbeat overrides.

## Execution Modes

### Monolithic

`pipeline_policy_node` executes the complete path in one process:

```text
ROS observations
  -> contract adapter
  -> InferencePipeline policy facade
  -> ModelRuntimeHandle.execute(ModelRequest, ExecutionContext)
  -> SequentialModelExecutor stages
       -> LeRobot preprocessor + policy codec
       -> ModelSession execute / role execution
       -> action decode + LeRobot postprocessor
  -> DispatchInfer result and action topic
```

Before processors run, the node checks every observation required by the policy `input_features`. A buffered
sample must exist at or before the requested timestamp and satisfy its `align.strategy` (`hold`, `asof`, or `drop`).
When the contract configures `max_age_ms > 0` for that observation, it is an additional maximum live sample age,
separate from `tol_ms` used by `asof` alignment. Live age uses the node's local receipt clock, which request
timestamps cannot rewind; request timestamps only select aligned historical samples. Missing,
future-dated, or stale samples return the recoverable `observation_not_ready` error instead of silently running the model with zero
padding. The pipeline reset service resets the policy and LeRobot preprocessor/postprocessor, then clears
observation buffers so the next inference waits for inputs from the new episode. Inference, reset, and distributed
cancellation use the pipeline `request_timeout` as a cooperative deadline: lock and admission waits exit on time, while
backend/processor hook overruns are detected when the hook returns. An uncertain reset or cancellation outcome fails
the edge closed to prevent cloud and edge episode state from diverging.

Normal robot startup creates pipelines from robot YAML through the `robot_config` launch builder. To evaluate one
pipeline directly:

```bash
source .shrc_local
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:="$WORKSPACE/src/robot_config/config/robots/so101_single_arm.yaml" \
    model_path:="$WORKSPACE/models/ACT_1arm_2cam_banana_pick_v1_step_160000_distill_20260515" \
    deployment:=cpu \
    pipeline_id:=policy \
    action_server:=/inference/policy/dispatch \
    reset_service:=/inference/policy/reset
```

Trigger one request:

```bash
ros2 action send_goal /inference/policy/dispatch \
    ibrobot_msgs/action/DispatchInfer \
    "{obs_timestamp: {sec: 0, nanosec: 0}, prompt: '', inference_id: 'test-001'}"
```

### Distributed

For a distributed pipeline, the edge `pipeline_policy_node` retains observation sampling, robot-state unit
conversion, non-image tensor serialization, the action `TemporalSmoother`, and final robot-unit conversion. The
cloud `pure_inference_node` assembles the complete raw observation and runs the LeRobot preprocessor, selected
backend, and postprocessor in one process so processor state is not split across hosts. Before requests are
accepted, both sides must match:

- pipeline ID
- manifest schema version
- bundle digest
- deployment name
- selected deployment fingerprint
- policy input/output summary
- cloud backend `READY` state

Image observations may retain explicit `mode: dds` or use an H.264 RTP/UDP data plane. In RTP mode DDS carries
only descriptors, status/timestamp mappings, requests/results, and heartbeats; H.264 payloads never enter
`VariantsList`. Each camera needs a unique stream ID, SSRC, and even UDP port, with `port + 1` reserved during
collision validation. The cloud accepts requests only after every descriptor matches the protocol, session
generation, contract fingerprint, and deployment fingerprint and each stream has a keyframe and a fresh
RTP-to-capture timestamp mapping.

`encoder_backend` accepts `software`, `nvidia`, `ascend`, or `auto`; `nvidia` is currently encode-only and cannot
be selected as a decoder. The software backend uses PyAV 15 and probes its FFmpeg build for `libx264` and an H.264
decoder. The NVIDIA backend opens a real `h264_nvenc` session and uses ultra-low-latency, zero-delay, no-B-frame
H.264 with repeated SPS/PPS. RGB/BGR-to-NV12 conversion still occurs through FFmpeg and is not CUDA zero-copy.
The optional Ascend backend lazily discovers a
private FFmpeg `h264_ascend` installation through `IBROBOT_ASCEND_FFMPEG` or
`IBROBOT_ASCEND_FFMPEG_PREFIX`; the standard RPM entry point `/usr/bin/ffmpeg-ascend` is also detected. It neither
replaces system FFmpeg nor adds ACL/DVPP Python dependencies. Startup
logs (the "Video stream startup" lines) report configured and selected backends, endpoints, fingerprints,
lifecycle, and readiness.

RPM-installed ffmpeg-ascend runtimes (`/usr/bin/ffmpeg-ascend`, `/usr/local/bin/ffmpeg-ascend`, and the
versioned `/usr/local/ffmpeg-ascend-*/bin/ffmpeg` payload they dispatch to) start with an isolated
environment by default: the child process drops installation-path variables inherited from an unrelated
CANN toolkit (`ASCEND_TOOLKIT_HOME`, `ASCEND_HOME_PATH`, `ASCEND_AICPU_PATH`, `ASCEND_OPP_PATH`,
`ASCEND_NNRT_HOME`, `ASCEND_NNAE_HOME`, `TOOLCHAIN_HOME`) while preserving device runtime variables
(such as `ASCEND_RT_VISIBLE_DEVICES` and `ASCEND_DEVICE_ID`) so multi-NPU pinning keeps working.
Private builds are not isolated by default; `IBROBOT_ASCEND_FFMPEG_ISOLATE_ENV=1` forces isolation for
any binary and `=0` opts out of the RPM default. Probing the RPM payload without isolation is known to
hang on the first frame.

Ascend DVPP VENC channels are a per-device hardware resource; `DeviceVideoStreamManager`
assigns dense 1..N IDs (≤128) to Ascend streams by sorted observation key. `device_id`
is currently fixed at 0 (single-NPU scope); multi-NPU support needs a resource allocation
contract at the pipeline/resource layer, not a code change in the manager.

`auto` probes `ascend`, then `nvidia`, then `software`. Ascend boards retain DVPP priority, NVIDIA hosts select
NVENC when a real session opens, and other Linux hosts fall back to software. Explicit backend failure never falls
back.

RTP/UDP provides no authentication, confidentiality, or integrity and is restricted to a trusted robot network.
An interrupted stream, descriptor mismatch, unavailable explicit backend, stale timestamp mapping, or excessive
camera skew fails closed without an RTP-to-DDS fallback. Rollback requires a matching `mode: dds` contract on both
hosts. rosbag/MCAP recording remains DDS-image based; RTP-aware recording and untrusted-network security are
separate follow-up work.

Cloud example:

```bash
source .shrc_local
ros2 launch inference_service cloud_inference.launch.py \
    pipeline_id:=policy \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda
```

To debug a distributed edge process directly:

```bash
ros2 launch inference_service eval_inference.launch.py \
    robot_config_path:=/absolute/path/to/robot.yaml \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda \
    pipeline_id:=policy \
    inference_execution_mode:=distributed
```

To start edge and cloud together on one host, replacing the old implicit local-cloud switch:

```bash
ros2 launch inference_service local_distributed_inference.launch.py \
    robot_config_path:=/absolute/path/to/robot.yaml \
    model_path:=/absolute/path/to/policy_bundle \
    deployment:=cuda \
    pipeline_id:=policy
```

The edge can be created from robot YAML with `execution_mode: distributed` or through the explicit launch
override above. A successful handshake binds a unique
session ID and generation. Heartbeat expiry, cloud restart, fingerprint change, or the runtime handle leaving `READY`
immediately revokes readiness, rejects new requests, and fails in-flight requests with a structured unavailable
error. Responses from old sessions are discarded, and recovery requires a new handshake.

A new handshake is not sufficient to recover a stateful runtime. When replacing an existing distributed session,
the cloud first stops request admission for that session and drains its `ModelRuntimeHandle` operations. A stateless
runtime may then create the new generation directly. A stateful runtime must first reset successfully through the
handle and return to `READY` before the cloud publishes a new generation. Reset failure remains fail-closed and
subsequent heartbeats cannot bypass this recovery barrier. If a stateful runtime declares `resettable: false`,
session rollover cannot recover through
handshaking; the cloud runtime must be restarted or rebuilt before it can serve requests again.

## Backends And Support Matrix

The only canonical backend names are:

| Backend | Responsibility |
| --- | --- |
| `torch` | native LeRobot on `cpu`, `cuda`, `mps`, or `npu` |
| `ascend` | Ascend ACL execution of OM artifacts |
| `hisilicon` | Hisilicon worker runtime, initially targeting SoC `sd3403` |
| `rknn` | RKNNLite execution of RKNN artifacts |
| `hmm` | Houmo TCIM execution of HMM multi-module artifacts |

Native Torch policies use the LeRobot factory by default. Repository-owned
models may register a stable `(model_type, backend, device)` provider; PI0.5
Ascend310P uses `(pi05, torch, npu)`. The generic runtime does not maintain
model-specific platform branches; each provider owns its configuration,
platform validation, and runtime preparation.
The optional provider hook `preprocess_inputs(inputs, config=...)` runs
model-owned CPU input optimization before the LeRobot processor. Its cost
remains included in pipeline preprocessing; image rules stay out of the generic session.
The optional `configure_processors(preprocessor=..., postprocessor=..., config=...)`
hook configures model-specific LeRobot steps at load time, such as checking
language capacity before upload or keeping small action tensors on the CPU.
Deployment `extra_parameters` is a generic model-owned JSON object. The native
Torch session passes a copy to optional `resolve_load_options(config=...,
extra_parameters=...)` and uses the returned options for model construction.
The session does not interpret model fields. Non-empty parameters require a
provider consumer, which cannot override strict/local checkpoint loading.
Parameters affect deployment and latency identity without extra parameter files
or user-maintained content checksums.

Compiled PI0.5 and SmolVLA loops are owned by shared executor stages, not model-session resources. They execute through
`InferencePipeline -> ModelRuntimeHandle -> SequentialModelExecutor -> InferenceStage -> ModelSession resource`.
The handle owns control-plane state, while each session owns only its vendor runtime and model resources. The
following matrix is normative and enforced at startup:

| Policy family | `torch` | `ascend` | `hisilicon` | `rknn` | `hmm` |
| --- | --- | --- | --- | --- | --- |
| ACT | Runtime | Runtime | Runtime | Runtime | unsupported |
| Diffusion Policy | Runtime | unsupported | unsupported | unsupported | unsupported |
| PI0.5 | Runtime | Runtime | unsupported | unsupported | Runtime |
| SmolVLA | Runtime | unsupported | unsupported | Runtime | Runtime |

`Runtime` means execution through `ModelRequest` and `ExecutionContext`, with handle-owned lifecycle, admission,
health, cancellation, and recovery and session-owned vendor model resources. The registry-enforced perception
matrix is:

| Perception family | `torch` | `ascend` |
| --- | --- | --- |
| RAM++ | supported | supported |
| SAM2 | supported (automatic) | supported (prompt) |
| SigLIP2 | supported | supported |
| Grounding DINO | supported (combined) | supported (raw) |
| GraspGen | supported (CUDA) | supported |
| Dummy Echo | supported | unsupported |

The registry also requires matching `ConformanceEvidence` for every family/backend declaration; a family listed
without evidence still fails closed.

### PI0.5 Ascend Behavior

The optimized PI0.5 VLM combines all camera images into one temporary vision batch internally, then restores the
camera-major prefix before the handoff. This optimization does not change the external VLM ABI: runtime bindings,
per-camera observation semantics, raw image shapes, and ROS camera-topic contracts remain unchanged.

NPU export uses the accuracy-preserving `NPUGeglu` path for the Gemma text MLP by default. The explicit
`--fast-gelu-scope vision|vlm-text|ae|all` option limits approximate `NPUFastGelu` to one model region; the legacy
`--fast-gelu` option means global `all`. The approximate path may reduce action accuracy and must be validated
against an existing baseline.

A new Action Expert OM has a runtime output named `velocity` or `v_t`, while the Manifest still maps that tensor
to the policy `action` semantic. The policy runtime assembler reads strictly decreasing timesteps from the selected
deployment's `denoising_schedule` artifact, and the shared `IterativeStage` performs host-side Euler integration as
`x_next = x_t + (next_t - t) * velocity` before returning the final action. When export does not specify
`--schedule-file`, the exporter packages a uniform schedule derived from `config.num_inference_steps`; an explicit
file must be strict `pi05-denoising-schedule-v1` JSON.

`denoising_schedule` is a versioned, non-execution artifact. It is absent from `execution` and `bindings`, but its
artifact path and deployment revision are part of the deployment identity, so changing the schedule changes the selected deployment
fingerprint. Production runtime does not scan for a root `schedule.json` and does not accept schedule overrides.
`loss_compare` and the tuner inject a temporary schedule through an isolated diagnostic backend factory;
`curvature_log_path` only records diagnostics. The final schedule must be installed in the Manifest.

Compatibility is selected explicitly by the Action Expert runtime output. Existing legacy PI0.5 deployments with
an `action` output and no schedule artifact retain the old stepwise action-output behavior. A velocity deployment
without a schedule is rejected rather than given a guessed default. `hardware_mock` still validates only the raw
image/topic, joint, and action contracts and needs no PI0.5- or schedule-specific changes.

Native Torch Diffusion Policy samples observation history at the contract control rate according to the model's
`n_obs_steps`, and its nominal `predict_action_chunk()` length comes from `n_action_steps`. Missing startup history
is left-padded with each stream's first frame, while different sensor rates retain their configured `hold`, `asof`,
or `drop` alignment policy on a common time grid.

Optional SDKs are imported lazily. Importing the inference core does not require ACL, RKNNLite, TCIM, torch NPU,
or Hisilicon worker dependencies. A missing dependency fails only when its deployment is selected.

## Lifecycle, Health, And Capabilities

Public `ModelRuntimeHandle` states are `CREATED`, `LOADING`, `READY`, `RESET_REQUIRED`, `RESETTING`, `FAILED`,
`CLOSING`, and `CLOSED`. Only `READY` admits requests. The handle closes admission and drains active execution
before reset or close, records health and recovery requirements, and idempotently releases `RuntimeAssembly`
components in reverse ownership order.

`ModelSession` is a handle-owned resource, not the public lifecycle owner. It owns and releases contexts, vendor
model handles, device buffers, workers, tokenizers, and other model-specific assets, and implements
`execute(ModelRequest, ExecutionContext)` or role execution. The policy facade projects handle state into pipeline
diagnostics; distributed `HANDSHAKING` is a node-protocol state, not a model-resource lifecycle state.

Session and deployment capabilities let the handle determine:

- whether the backend is stateful, resettable, and thread-safe
- maximum in-flight requests per instance
- support for multiple instances
- shared resource-domain identity and limit
- attention and cancellation support

Defaults are conservative and serialized. A runtime may declare higher concurrency only after conformance tests
prove overlapping calls, output isolation, failure isolation, and deterministic cleanup. Different handles have
independent admission state, but a shared accelerator resource domain may still serialize them.

## Manifest Identity

Startup performs strict JSON/schema validation, deployment selection, UUID/revision and lightweight bundle-digest
validation, path-safety and regular-file checks, LeRobot metadata loading, and binding compatibility checks before
constructing the session resource, `RuntimeAssembly`, and `ModelRuntimeHandle`. Runtime does not read OM, RKNN,
HMM, or safetensors files to hash their contents.

`bundle.digest` is calculated as follows:

1. Normalize, deduplicate, and sort every `bundle.files` path.
2. Add the bundle UUID, revision, name, and a structure-format domain.
3. Serialize this small declaration as canonical UTF-8 JSON.
4. Calculate SHA-256 over the declaration bytes without reading the referenced files.

UUIDs, revisions, digests, and fingerprints provide version identity and distributed consistency, not tamper
protection. Production artifact updates must use the packager and publish a new revision; use signed read-only
images or verity when artifact authenticity is required.

The selected deployment fingerprint is SHA-256 over this canonical object:

```json
{
  "format": "ibrobot.deployment-structure-v3",
  "schema_version": 3,
  "bundle_digest": "...",
  "deployment_name": "rk3588",
  "deployment": {}
}
```

Paths cannot be absolute, use parent traversal, escape the bundle root through symlinks, or collide after
normalization.

### Identity Failures

Do not edit identities manually when startup reports:

- `Bundle digest mismatch`
- unsupported schema v1 (regeneration is required)
- missing or unexpected LeRobot semantic files
- an execution role missing an artifact or bindings
- runtime ABI incompatible with LeRobot feature shapes

Rerun the exporter or packaging workflow that owns the artifact. Exporters copy artifacts, read compiler/runtime
ABI metadata, generate bindings, update UUIDs/revisions and lightweight structural identities, and validate the
result through the production loader. Schema-v1 bundles and legacy artifacts are unsupported; regenerate a complete
schema-v3 bundle with the current exporter or packager.

## Exporter Entry Points

Generic compiled artifact packaging:

```bash
ros2 run model_utils package-compiled-deployment \
    --bundle-root /path/to/policy_bundle \
    --deployment rk3588 \
    --backend rknn \
    --target-soc rk3588 \
    --target-runtime rknn-lite2 \
    --spec /path/to/compiler-package-spec.json
```

For PI0.5 and SmolVLA HMM packaging:

```bash
ros2 run model_utils package-hmm-deployment --help
```

ACT Ascend, ACT RKNN, Hisilicon, and policy-specific multi-module exporters live in `model_utils`. Every tool must
finish through the shared `inference_manifest` writer. Artifact paths, bindings, and digests are exporter-owned,
not hand-maintained configuration.

## Serving Models From Another Host (`with_model_services`)

`with_model_services` is a launch argument declared by the `perception_models` launch entry,
default `true`. It decides whether this host starts the configured
`perception_services` nodes.

With `false`:

- The launch plan omits `perception_models`; this host starts no model service nodes.
- The YAML's service configuration is untouched, and callers here keep calling the
  configured endpoints.
- Config loading and the live binding validate enabled services by metadata only,
  reusing `load_inference_manifest_metadata()` (the same load the distributed edge
  uses): identity, endpoints, deployment names, and the semantic contract still
  validate, while bundle files and deployment artifacts are not checked on this host.
  The hosting machine keeps the full check.
- Contract readers validate by metadata as well: the semantic mapping parameter
  construction and `load_robot_config()` (the scheduled dispatcher, the recording
  node) consume endpoints and identities, never the weights. The validation tools
  (`validate_config.py`) explicitly demand this machine's artifacts.

Prerequisites for the consumer host:

- It must hold every enabled service's bundle directory and its
  `inference_manifest.json`; weights and deployment artifacts only need to exist on
  the host that runs the services.
- Both hosts use the same robot YAML; `bundle_path` resolves per host.

```bash
ros2 launch robot_config robot.launch.py config_path:=/absolute/path/to/robot.yaml \
    with_model_services:=false
```

## Verification

When running from source, prefer source package paths and disable unrelated external pytest plugins:

```bash
source .shrc_local
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src/inference_manifest:src/inference_service \
pytest -q src/inference_service/test
```

Check that removed identifiers have not re-entered active source, configuration, or tests:

```bash
source .shrc_local
python scripts/check_inference_legacy_identifiers.py
```

Run Ruff only on Python files changed by the current work. Always source `.shrc_local` before project or ROS
commands.

### Close failures and recovery limits

The Handle retains a stream and its capacity when Close fails. `retry_pending_stream_closes()` retries those
operations; runtime reset and ASR reopening also retry pending stream closure. Releasing quarantined capacity
requires native recovery proof, not an empty reset.

RKNN/HMM retain ownership and report `close_pending` after an SDK release exception without completion evidence.
Later Close calls do not invoke the same native release again or claim successful cleanup; a process restart is
required. Only explicit `NOT_STARTED` evidence permits retrying native release. Successful Close remains idempotent.

String stream IDs have idempotent Close receipts for the most recent 128 closures. Evicted IDs return
`stream_not_found`; an original closed `StreamHandle` remains idempotent. SpeechDirection has the same string-ID limit.

Direct stateless synchronous errors retain known completion unless native evidence says otherwise. LeRobot CUDA/NPU
execution pins the resolved device at load; resource domains remain conservative device-family groups such as
`torch:npu`, not per-device identifiers.
