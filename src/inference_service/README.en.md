# inference_service

`inference_service` is IB-Robot's unified inference runtime. It selects one policy bundle, one named deployment,
and one stable pipeline ID, then runs that pipeline through Torch, Ascend, Hisilicon, RKNN, or HMM in either
monolithic or distributed edge/cloud mode.

There is no compatibility layer for the removed runtime architecture. A backend is not selected with a launch
`device` argument. The runtime does not load per-backend sidecar manifests, scan directories for conventional
artifact names, or use environment variables to override artifacts.

## Core Concepts

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

A Torch deployment directly declares its runtime device:

```json
{
  "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
  "revision": 1,
  "backend": "torch",
  "device": "cpu"
}
```

A compiled deployment declares its target, artifacts, execution order, and complete runtime ABI bindings:

```json
{
  "uuid": "f9ebdcd5-1ce8-4b56-8860-4f32454fc209",
  "revision": 3,
  "backend": "rknn",
  "target": {
    "soc": "rk3588",
    "runtime": "rknn-lite2"
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
- its backend instance, admission state, and lifecycle
- its action, reset, health, action-output, and distributed transport endpoints

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
  -> LeRobot preprocessor
  -> semantic batch
  -> native policy or policy codec + bindings
  -> selected backend
  -> semantic action
  -> LeRobot postprocessor
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
private FFmpeg 4.4 `h264_ascend` installation through `IBROBOT_ASCEND_FFMPEG` or
`IBROBOT_ASCEND_FFMPEG_PREFIX`; it neither replaces system FFmpeg nor adds ACL/DVPP Python dependencies. Startup
logs and `/diagnostics` report configured and selected backends, endpoints, fingerprints, lifecycle, and readiness.

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
session ID and generation. Heartbeat expiry, cloud restart, fingerprint change, or a backend leaving `READY`
immediately revokes readiness, rejects new requests, and fails in-flight requests with a structured unavailable
error. Responses from old sessions are discarded, and recovery requires a new handshake.

A new handshake is not sufficient to recover a stateful backend. When replacing an existing session, the cloud
first stops admission for the old session and drains its runtime operations. A stateless backend may then create
the new generation directly. A stateful backend must first reset successfully and return to `READY` before the
cloud publishes a new generation. Reset failure remains fail-closed and subsequent heartbeats cannot bypass this
recovery barrier. If a stateful backend declares `resettable: false`, session rollover cannot recover through
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

The initial support matrix is normative and enforced at startup:

| Policy family | `torch` | `ascend` | `hisilicon` | `rknn` | `hmm` |
| --- | --- | --- | --- | --- | --- |
| ACT | supported | supported | supported | supported | unsupported |
| Diffusion Policy | supported | unsupported | unsupported | unsupported | unsupported |
| PI0.5 | supported | supported | unsupported | unsupported | supported |
| SmolVLA | supported | unsupported | unsupported | supported | supported |

### PI0.5 Ascend Behavior

The optimized PI0.5 VLM combines all camera images into one temporary vision batch internally, then restores the
camera-major prefix before the handoff. This optimization does not change the external VLM ABI: runtime bindings,
per-camera observation semantics, raw image shapes, and ROS camera-topic contracts remain unchanged.

NPU export uses the accuracy-preserving `NPUGeglu` path for the Gemma text MLP by default. Only the explicit
`--fast-gelu` export option replaces it with approximate `NPUFastGelu`; this may reduce action accuracy and must be
validated against an existing baseline.

A new Action Expert OM has a runtime output named `velocity` or `v_t`, while the Manifest still maps that tensor
to the policy `action` semantic. The Ascend backend reads strictly decreasing timesteps from the selected
deployment's `denoising_schedule` artifact and performs host-side Euler integration as
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

Backend states are `CREATED`, `LOADING`, `READY`, `DEGRADED`, `RECOVERING`, `FAILED`, `CLOSING`, and
`CLOSED`. Only `READY` admits requests. `close()` is idempotent, and partial startup failure releases every
already-created context, model handle, device buffer, or worker.

Pipeline states are `CREATED`, `LOADING`, `HANDSHAKING`, `READY`, `RESETTING`, `DEGRADED`, `FAILED`,
`CLOSING`, and `CLOSED`. Reset blocks new admission. `CLOSING` and `CLOSED` are terminal.

Backend capabilities report:

- whether the backend is stateful, resettable, and thread-safe
- maximum in-flight requests per instance
- support for multiple instances
- shared resource-domain identity and limit
- attention and cancellation support

Defaults are conservative and serialized. A backend may declare higher concurrency only after conformance tests
prove overlapping calls, output isolation, failure isolation, and deterministic cleanup. Different pipelines have
independent admission state, but a shared accelerator resource domain may still serialize them.

## Manifest Identity

Startup performs strict JSON/schema validation, deployment selection, UUID/revision and lightweight bundle-digest
validation, path-safety and regular-file checks, LeRobot metadata loading, and binding compatibility checks before
creating a backend runtime. Runtime does not read OM, RKNN, HMM, or safetensors files to hash their contents.

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
  "format": "ibrobot.deployment-structure-v2",
  "schema_version": 2,
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
schema-v2 bundle with the current exporter or packager.

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

## Verification

When running from source, prefer source package paths and disable unrelated external pytest plugins:

```bash
source .shrc_local
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src/inference_manifest:src/inference_service \
pytest -q src/inference_service/tests
```

Check that removed identifiers have not re-entered active source, configuration, or tests:

```bash
source .shrc_local
python scripts/check_inference_legacy_identifiers.py
```

Run Ruff only on Python files changed by the current work. Always source `.shrc_local` before project or ROS
commands.
