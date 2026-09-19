# PI0.5 Ascend310P

This directory owns the repository-native PI0.5 implementation for Ascend310P.
It contains the NPU-safe PI0.5 model, local SigLIP tower, fixed ten-step
prefix/denoise TorchAir graphs, internal-format preparation, persistent graph
cache, strict no-initialization checkpoint loading, and the runtime provider.

The provider is selected by the stable runtime identity:

```text
model_type: pi05
backend:    torch
device:     npu
```

No model-specific `architecture_class` or packaging flag is required. The
normal deployment name remains `torch-npu`:

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

Set the robot's inference pipeline to the normal bundle and deployment:

```yaml
model_path: models/pi05/pi05-doublecam-fp32/019200-torch-npu
deployment: torch-npu
runtime_options:
  model_dtype: fp16
```

The `so101_pi05_ascend_310p_mock` profile supplies this pipeline and mock
observations. From the workspace root, after building:

```bash
source .shrc_local
ros2 launch robot_config robot.launch.py \
  robot_config:=so101_pi05_ascend_310p_mock \
  use_sim:=true control_mode:=model_inference
```

The provider validates the local tokenizer, the fixed `num_inference_steps=10`
contract, Transformers `5.3.0`, and the physical Ascend310P SKU before loading
weights. On 310P, the default graph boundary is compiled vision plus compiled
prefix prefill and a fixed ten-step denoise graph.

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
  requires IDENTITY visual normalization. Its cost remains included in pipeline
  preprocessing. Validate outputs for the deployment before enabling it.
- `IBROBOT_PI05_VISION_FFN=1` uses the Ascend FFN operator for SigLIP vision MLPs
  inside the compiled graph. It requires FP16 and the compiled local vision
  tower, uses a separate graph-cache namespace, and keeps eager/training on
  the original MLP. The Gemma gate/up path is unaffected. Packed weights are
  derived after strict checkpoint loading; keep them fixed for the prepared
  runtime. This adds roughly 0.5 GiB of derived FP16 weights for the current model.
- `IBROBOT_PI05_CPU_OUTPUT=1` reads the final action chunk back once and runs
  LeRobot's existing unnormalization on CPU. The copy stays inside inference
  timing. This avoids repeated device transfers during action validation and
  the small NPU postprocessing operations; it does not change the normalizer
  or its configured precision.

The optimized short-task configuration on Ascend310P1 / CANN 8.1.RC1 /
Torch-NPU 2.5.1 enables the four options together. Set `TASK_QUEUE_ENABLE=0`
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
