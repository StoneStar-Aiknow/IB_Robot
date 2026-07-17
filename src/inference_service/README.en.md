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
  "backend": "torch",
  "device": "cpu"
}
```

A compiled deployment declares its target, artifacts, execution order, and complete runtime ABI bindings:

```json
{
  "backend": "rknn",
  "target": {
    "soc": "rk3588",
    "runtime": "rknn-lite2"
  },
  "artifacts": {
    "policy": {
      "path": "artifacts/rknn/rk3588/policy.rknn",
      "format": "rknn",
      "sha256": "<64 lowercase hex>"
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

For a distributed pipeline, the edge `pipeline_policy_node` retains the observation adapter, processors, and
postprocessor. The cloud `pure_inference_node` owns the selected backend. Before tensors can be sent, both sides
must match:

- pipeline ID
- manifest schema version
- bundle digest
- deployment name
- selected deployment fingerprint
- policy input/output summary
- cloud backend `READY` state

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
| PI0.5 | supported | supported | unsupported | unsupported | supported |
| SmolVLA | supported | unsupported | unsupported | supported | supported |

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

## Manifest Integrity

Startup performs strict JSON/schema validation, deployment selection, path-safety validation, bundle-file
SHA-256 verification, bundle-digest verification, LeRobot metadata loading, compiled-artifact SHA-256 verification,
and binding compatibility checks before creating a backend runtime.

`bundle.digest` is calculated as follows:

1. Normalize every `bundle.files` path to a unique bundle-relative POSIX path.
2. Sort `{\"path\": ..., \"sha256\": ...}` entries by path.
3. Serialize the array as UTF-8 JSON without insignificant whitespace and with keys ordered `path`, `sha256`.
4. Calculate SHA-256 over the serialized bytes.

The selected deployment fingerprint is SHA-256 over this canonical object:

```json
{
  "schema_version": 1,
  "bundle_digest": "...",
  "deployment_name": "rk3588",
  "deployment": {}
}
```

Paths cannot be absolute, use parent traversal, escape the bundle root through symlinks, or collide after
normalization.

### Integrity Failures

Do not edit hashes manually when startup reports:

- `SHA-256 mismatch`
- `Bundle digest mismatch`
- missing or unexpected LeRobot semantic files
- an execution role missing an artifact or bindings
- runtime ABI incompatible with LeRobot feature shapes

Rerun the exporter or packaging workflow that owns the artifact. Exporters copy artifacts, read compiler/runtime
ABI metadata, generate bindings, calculate all SHA-256 values, update the bundle digest, and validate the result
through the production loader.

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
