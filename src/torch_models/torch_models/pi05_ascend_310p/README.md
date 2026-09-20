# PI0.5 Ascend310P

This directory owns the repository-native PI0.5 implementation for Ascend310P.
It contains the NPU-safe PI0.5 model, local SigLIP tower, configured fixed-length
prefix/denoise TorchAir graphs, internal-format preparation, persistent graph
cache, strict no-initialization checkpoint loading, and the runtime provider.

The provider is selected by the stable runtime identity:

```text
model_type: pi05
backend:    torch
device:     npu
```

No model-specific `architecture_class` or packaging flag is required. The
generic packager retains the legacy `torch-npu` deployment:

```bash
source .shrc_local
ros2 run model_utils package-torch-deployment \
  --bundle-root /path/to/pi05-bundle \
  --devices npu
```

The same bundle may contain CPU/CUDA native Torch deployments and compiled
Ascend OM deployments. CPU/CUDA use the existing LeRobot policy factory; OM
uses the existing Ascend session and its VLM/Action Expert artifacts. Only the
native `(pi05, torch, npu)` identity resolves this provider.

## Environment and Execution

On CANN 8.1, both `./scripts/setup.sh --profile full` and
`./scripts/setup.sh --profile inference` select the same compatibility runtime:
LeRobot 0.6.0, Torch/Torch-NPU 2.5.1, TorchVision 0.20.1, and Transformers
5.3.0. Full adds functionality without selecting a different Transformers.
The shared `requirements/lerobot-v0.6-cann-8.1-compat.txt` and CANN constraints
preserve this combination; the model provider still validates it before loading
weights. Newer Transformers compatibility is separate model-adaptation work.

After registering the named deployments below, select the fast preset in the
robot's inference pipeline (keep only one deployment line uncommented):

```yaml
model_path: models/pi05/pi05-doublecam-fp32/019200-torch-npu
deployment: torch-npu-fast
# deployment: torch-npu-fidelity
runtime_options:
  model_dtype: fp16
```

The `so101_pi05_ascend_310p_mock` profile supplies this pipeline and mock
observations, defaulting to fast. From the workspace root, after building and
packaging both presets:

```bash
source .shrc_local
TASK_QUEUE_ENABLE=0 ros2 launch robot_config robot.launch.py \
  robot_config:=so101_pi05_ascend_310p_mock \
  use_sim:=true control_mode:=model_inference
```

The provider validates the local tokenizer, model-owned inference settings,
Transformers `5.3.0`, and the physical Ascend310P SKU before loading weights.
On 310P, the default graph boundary is compiled vision plus compiled prefix
prefill and one fixed-length denoise graph. Legacy deployments without
`extra_parameters` keep the original ten-step contract and environment switches.

## Named Deployment Modes

Register both modes over the same native weights and processor assets:

```bash
source .shrc_local && ros2 run model_utils package-pi05-ascend-deployments \
  --bundle-root models/pi05/pi05-doublecam-fp32/019200-torch-npu
```

| Deployment | Solver | Network evaluations | Language capacity |
|---|---|---:|---:|
| `torch-npu-fidelity` | Uniform Euler | 10 | 200 |
| `torch-npu-fast` | Uniform AB2, first step Euler | 3 | 48 |

Both use FP16, compiled vision FFN, layout-aware CPU image preprocessing, and
CPU action output. AB2 keeps velocity history within one action request and
uses `1.5 * v_current - 0.5 * v_previous` after its Euler startup. The original
action shape, cameras, weights, and noise distribution are preserved.
AB2 currently rejects RTC and nonuniform schedules.

The preset capacity rule is `48 if mode == "fast" else 200` for supported
modes. Re-run the packager to update an existing bundle: the fidelity revision
and fingerprint change and select a separate graph cache; fast parameters stay
unchanged. The checkpoint tokenizer must support the selected capacity.

The packager writes inline, model-owned `extra_parameters` in each deployment:

```json
{
  "denoising": {"solver": "ab2", "schedule": "uniform", "steps": 3},
  "language_capacity": 48,
  "vision_ffn": true,
  "image_preprocessing": "layout_aware",
  "action_output_device": "cpu"
}
```

The manifest treats this as generic JSON; the PI05 provider parses it into an
immutable `PI05AscendInferenceConfig`. Defaults for individual omitted fields
are Euler/uniform/10, full checkpoint language length, no vision FFN, native
image preparation, and NPU action output. An entirely absent/empty object
selects legacy behavior. The two named presets always spell out all settings.
Unknown PI05 fields, incompatible types, unsupported precision, and valid tokens
beyond the capacity are rejected. There is no new parameter file or checksum.
For an explicit language capacity, the loaded tokenizer runs without truncation
and pads to the longest prompt in the batch. The model-owned capacity step checks
the complete token mask before device upload, rejects overflow, and right-pads
short inputs with the tokenizer's pad token to the configured static length.
This includes prompts longer than the checkpoint's original tokenizer maximum;
saved processor files are not rewritten.

Select the deployment in the existing robot YAML:

```yaml
model_path: models/pi05/pi05-doublecam-fp32/019200-torch-npu
deployment: torch-npu-fast  # or torch-npu-fidelity
execution_mode: monolithic
runtime_options:
  model_dtype: fp16
```

Use the single `so101_pi05_ascend_310p_mock.yaml` for both modes. It defaults to
`torch-npu-fast`; to select fidelity, comment out that line and uncomment the
adjacent `deployment: torch-npu-fidelity` line. After registering both deployments
and building the workspace, launch:

```bash
source .shrc_local && TASK_QUEUE_ENABLE=0 ros2 launch robot_config robot.launch.py \
  config_path:="$PWD/src/robot_config/config/robots/so101_pi05_ascend_310p_mock.yaml" \
  use_sim:=true control_mode:=model_inference
```

`TASK_QUEUE_ENABLE` is process-wide and must be set before Torch-NPU import;
the library does not mutate it. Conflicting legacy model environment switches
are rejected for explicit deployment settings. Matching switches are accepted.
The selected settings are returned as `pi05_inference_config` execution metadata.
Settings and solver/step count participate in graph and latency identities, so
switching modes cannot reuse the other mode's compiled denoise graph.
The graph cache ABI is bumped; the first load needs a fresh compilation.

Choose the mode at startup; changing the manifest does not mutate a live model.
Switching an active deployment requires draining requests, resetting action
queues, reloading and warming the selected model via the normal lifecycle.
The previous `torch-npu` deployment is preserved by the packager.

### Validation status and measurement scope

The latest review follow-up disables tokenizer truncation before capacity checks
and validates the loaded visual Normalizer configuration. Regression cases have
been added but have not been executed; tests, build, frozen precision and
hardware_mock validation are deferred until review feedback is consolidated.
The results below describe earlier revisions, not acceptance of these changes.

The configurable implementation at `b74fa550` was verified on Ascend310P1 /
CANN 8.1.RC1 / Torch-NPU 2.5.1 on 2026-09-22, with `TASK_QUEUE_ENABLE=0`.
Both deployments shared one checkpoint bundle and used 48 language slots.
Those fidelity measurements predate the restoration to 200 slots and do not
describe the current fidelity preset. Current-capacity validation is recorded
separately in the PR; do not relabel the historical 48-slot timings as 200-slot results.
Each workload ran in three fresh processes with no profiler or extra NPU
synchronization. The original 20 frozen batches, targets and noises were retained.

| Deployment | Frozen pipeline mean | hardware_mock pipeline mean | Unnormalized mean cosine |
|---|---:|---:|---:|
| `torch-npu-fidelity` (Euler10) | 195.38 ms | 194.63 ms | 0.99994864 |
| `torch-npu-fast` (AB2, three evaluations) | 142.71 ms | 154.65 ms | 0.99939797 |

Frozen means exclude the first request (19 warm samples per process); mock
means exclude the first three (20 warm samples per process). Per-run P95 was
199.27–205.85 ms / 145.14–153.15 ms for fidelity/fast frozen, and
195.72–197.18 ms / 155.25–164.48 ms for fidelity/fast mock (nearest rank).
Mean outer wall time was 197.41/144.76 ms frozen and 196.74/156.84 ms mock.
First cache-miss requests took 1282.67/509.67 seconds, including cold compilation;
these are excluded from steady latency. Mock inputs have no accuracy target;
cosine comes from the full frozen action chunks after unnormalization.

The board build and 151 targeted tests passed in that validation. Both modes
reported the expected solver, step count and capacity, reused separate caches,
and returned finite 50x6 chunks to the mock dispatcher. These figures identify
that tested revision; subsequent integration results and platform setup/build
evidence belong in the corresponding PR. Current package-level acceptance uses
`colcon test`, including the `torch_models` pytest suite.

Fidelity uses **200** slots; fast uses **48**. Revalidate performance changes with at least three
fresh-process frozen and hardware_mock runs; report steady mean/P95, outer wall
time and cold compilation separately. Use the original target and per-sample
noise files. The primary precision metric is unnormalized mean cosine over the
complete action chunk, with L1/W1 and first-frame metrics retained as diagnostics.
The fast mode is an approximation, and the fidelity mode is not bitwise lossless.
Mean latency is not a worst-case bound, and the figures are specific to the
tested Ascend310P1/checkpoint/task. No new 130 ms claim is made.

The existing OM schedule v1/parser/tuner remains Euler-only. These native Torch
parameters do not enable AB2 on an OM deployment. This change adds uniform
Euler/AB2 instance configuration without changing OM's schedule contract.

Diagnostic switches:

- `LEROBOT_PI05_COMPILE_VISION_EMBED=0` selects eager vision only for diagnosis.
- `IBROBOT_PI05_GRAPH_COMPILE=0` disables graph compilation for isolation tests.
- `IBROBOT_PI05_NPU_FUSED_OPS=0` disables NPU fused operations for isolation tests.
- `IBROBOT_PI05_STAGE_TIMING=1` adds synchronized prefix/denoise timings to result metadata.
- `IBROBOT_PI05_TORCHAIR_CACHE_HOME=/path/to/cache` isolates experimental graphs;
  confirm the actual cache path in the startup log before comparing variants.
- `IBROBOT_PI05_LANGUAGE_TOKEN_LIMIT=48` opts into a fixed shorter language
  capacity for deployments with short prompts and state text. Only masked tail
  tokens are removed; requests with valid tokens beyond the capacity fail with
  an explicit error. The default keeps the checkpoint's original padded length.
  The capacity has its own graph-cache namespace and requires accuracy validation
  for the deployment's tasks. It must include both task and discretized state tokens.
- `IBROBOT_PI05_CPU_IMAGE_RESIZE=1` resizes normalized float32 HWC-backed host images with
  CPU bilinear interpolation and zero padding before upload. Packed CHW inputs
  (including normal ROS observations) keep the NPU resize to avoid CPU contention. This opt-in path
  requires IDENTITY visual normalization both in model configuration and in the
  loaded Normalizer steps that process the model's images. A non-identity loaded
  Normalizer rejects this optimization during loading, even if config.json says
  IDENTITY. Its cost remains included in pipeline
  preprocessing. Validate outputs for the deployment before enabling it.
- `IBROBOT_PI05_VISION_FFN=1` uses the Ascend FFN operator for SigLIP vision MLPs
  inside the compiled graph. It requires FP16 and the compiled local vision
  tower, uses a separate graph-cache namespace, and keeps eager/training on
  the original MLP. The Gemma gate/up path is unaffected. Packed weights are
  derived after strict checkpoint loading; keep them fixed for the prepared
  runtime. This adds roughly 0.5 GiB of derived FP16 weights for the current model.
  Prepared models are read-only inference instances: training, replacing weights,
  or saving/reloading a prepared instance via `save_pretrained()` is not supported.
  Packaging and archiving use the original checkpoint files; the derived FFN
  layout copies are runtime-only data and do not require a separate save step.
- `IBROBOT_PI05_CPU_OUTPUT=1` reads the final action chunk back once and runs
  LeRobot's existing unnormalization on CPU. The copy stays inside inference
  timing. This avoids repeated device transfers during action validation and
  the small NPU postprocessing operations; it does not change the normalizer
  or its configured precision.

For the legacy `torch-npu` deployment, the optimized short-task configuration on
Ascend310P1 / CANN 8.1.RC1 / Torch-NPU 2.5.1 enables the four options together.
To reproduce that legacy configuration, first set the mock YAML's deployment
to `torch-npu` instead of either named preset. Set `TASK_QUEUE_ENABLE=0`
before starting the process to disable Torch-NPU's asynchronous host task
queue; the library does not change this process-wide setting:

```bash
source .shrc_local && \
  TASK_QUEUE_ENABLE=0 \
  IBROBOT_PI05_LANGUAGE_TOKEN_LIMIT=48 \
  IBROBOT_PI05_CPU_IMAGE_RESIZE=1 \
  IBROBOT_PI05_VISION_FFN=1 \
  IBROBOT_PI05_CPU_OUTPUT=1 \
  ros2 launch robot_config robot.launch.py \
    config_path:="$PWD/src/robot_config/config/robots/so101_pi05_ascend_310p_mock.yaml" \
    use_sim:=true control_mode:=model_inference
```

The 48-token capacity includes task text and discretized state. Validate the
capacity and numerical outputs for other tasks/checkpoints. Cold compilation
and the first request in each fresh process are excluded from steady latency;
sub-200 ms mean latency does not imply that every request finishes below 200 ms.
The launch example uses the mock YAML's default task. Frozen comparisons used
`Grasp banana and put it on the plate`, matching per-sample noises and the
preserved 20-batch reference; set that task in the benchmark configuration when
reproducing the measurements.

The graph/fused-op overrides are intended for diagnostics. Do not use
synchronized stage timing in formal latency measurements.

Evaluation mode is set at load/prepare time; predictions only switch mode if
the policy has explicitly returned to training. Repeated recursive `eval()`
traversals add measurable host overhead on Ascend310P1 without changing the graph.
