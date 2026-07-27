---
name: pi05-om-convert
description: "Convert PI0.5/PI05 LeRobot policy bundles to Ascend OM deployments. Use when users mention 'PI05 OM', 'PI0.5 OM', 'convert pi05 to om', 'Ascend310P', 'ATC', 'pi05_export', 'PI05模型转换', '转OM', '生成OM', or want to export VLM and Action Expert ONNX/OM artifacts, save a reusable profile, and validate the bundle with hardware_mock. Before conversion, always ask about pipeline steps, FastGELU, and denoising schedule choice, in that order."
---

# PI05 Ascend OM Conversion Skill

Convert a local PI0.5 LeRobot policy into a strict IB-Robot Ascend deployment containing VLM and
Action Expert OM artifacts, compiler ABI metadata, and a named deployment in
`<bundle>/inference_manifest.json`.

Use the repository entry point `python3 -m model_utils.pi05_export`. Do not create a second export
script, hand-edit generated ABI bindings or identities, or deploy raw OM paths outside the policy bundle.

## Supported Scope

| Item | Support |
|------|---------|
| Policy | PI0.5 / PI05 only |
| Backend | Ascend ACL OM |
| Standard precision | FP16 VLM + FP16 Action Expert |
| Default NPU text MLP | Accuracy-preserving NPUGeglu |
| Optional acceleration | FastGELU override during NPU ONNX export; may reduce accuracy |
| Optional quantization | VLM and/or Action Expert W8A8 steps, with representative calibration data |
| Validation | Strict manifest loading, `loss_compare`, and ROS 2 `hardware_mock` |

For RKNN use `rknn-convert`. For Houmo HMM use `hmm-convert`. Do not route ACT or SmolVLA OM
requests through this skill.

## Mandatory Interaction Gate

Do not start export, quantization, or ATC before completing these questions. Ask concise questions
with the `question` tool when available.

### 1. Confirm Pipeline Steps

Always ask whether the user wants the default steps. Explain that a step is one independently
selectable stage of the conversion pipeline; explicitly selected steps are rebuilt even when their
output already exists.

Present the default as the recommended first option:

- `Default FP16 pipeline (Recommended)`: `vlm_onnx,ae_onnx,vlm_om,ae_om`
- `Custom steps`: the user chooses from the registry below

Explain the step meanings before asking:

| Step | Meaning |
|------|---------|
| `vlm_onnx` | Export the PI05 vision-language prefix model to ONNX and save VLM-to-AE handoff tensors. |
| `ae_onnx` | Export the Action Expert to ONNX; it consumes handoff tensors produced by `vlm_onnx`. |
| `vlm_om` | Compile the FP16 VLM ONNX to Ascend OM with ATC. |
| `ae_om` | Compile the FP16 Action Expert ONNX to Ascend OM with ATC. |
| `vlm_quant` | Quantize VLM ONNX to W8A8; requires representative `--batch-path`. |
| `ae_quant` | Quantize Action Expert ONNX to W8A8; calibration defaults to `runtime_save`. |
| `vlm_quant_om` | Compile the W8A8 VLM ONNX to OM. |
| `ae_quant_om` | Compile the W8A8 Action Expert ONNX to OM. |
| `verify` | Compare split ONNX behavior with the source policy; requires the exact task prompt. |

Dependency rules:

- `ae_onnx` needs the `vlm_onnx` handoff tensors, either produced in this run or already present.
- `*_om` needs its matching ONNX product.
- `*_quant` needs its matching FP ONNX product.
- `*_quant_om` needs its matching W8A8 ONNX product.
- Creating a deployable FP16 bundle for the first time needs both `vlm_om` and `ae_om`. Once the named
  deployment exists, rerunning one OM role inherits the unchanged counterpart artifact and bindings from
  that Manifest deployment; only the rebuilt role needs work-directory OM/ABI files.
- Quantization is not part of the default workflow. Warn that W8A8 accuracy depends on representative
  calibration and must be checked against an existing baseline profile.

If the user chooses custom steps, collect the exact comma-separated step list and validate the
dependencies before running anything.

### 2. Confirm FastGELU

Always ask whether to enable FastGELU. Present `Disabled (Recommended for accuracy)` first.

Explain the tradeoff:

- Disabled: uses the accuracy-preserving `NPUGeglu` fusion for the Gemma text MLP and is the default
  for NPU export.
- Enabled: exports Ascend `NPUFastGelu`, which can reduce inference latency and improve throughput,
  but it is an approximation and can reduce action accuracy.

Never silently enable FastGELU. When enabled, use `--fast-gelu`; when disabled, omit the flag or use
`--no-fast-gelu`. Record the selection in the saved profile and final report.

### 3. Confirm Denoising Schedule

Ask this after steps and FastGELU. Explain that new Action Expert OM files predict velocity and the
Ascend backend performs Euler integration using a strict schedule packaged in the Manifest.

Present these choices:

- `Default uniform schedule (Recommended for a first export)`: omit `--schedule-file`; the exporter
  generates a uniform schedule from `config.num_inference_steps` and packages it with the deployment.
- `Existing custom schedule`: collect the path to an existing strict
  `pi05-denoising-schedule-v1` JSON and pass `--schedule-file`. The saved export profile must record
  `schedule_file` so later OM finalization uses the same schedule.
- `Tune later`: start with the default uniform schedule, establish or reuse a `loss_compare` baseline,
  then run `pi05-tune-schedule`. Explain that tuning uses existing targets/noises and installs the
  selected schedule in the Manifest; it never generates targets.

Do not offer root `schedule.json` scanning or environment variables. A schedule is either packaged as
the named deployment's `denoising_schedule` artifact or is a transient diagnostic override. The
artifact is non-execution and versioned by the deployment revision; changing it changes the deployment
fingerprint. Existing legacy
deployments whose Action Expert output is `action` and which have no schedule artifact retain their old
behavior.

### 4. Collect Remaining Inputs

After the three mandatory choices, collect or derive:

| Input | Requirement |
|-------|-------------|
| Policy path | Existing local PI05 policy directory containing `config.json` and processor assets. |
| Experiment directory | New or existing directory for `onnx/`, `runtime_save/`, and `om/`. |
| SoC version | Obtain from `npu-smi info`; Ascend310P boards commonly use `Ascend310P1`. |
| Torch export device | Recommend `npu` on an Ascend310P conversion host; use `cpu` only when intended. |
| Deployment name | Named FP deployment written into `inference_manifest.json`, e.g. `ascend310p_fp16`. |
| Profile name | Reusable, descriptive name, e.g. `pi05-310p-fp16-npugeglu`. |
| Profile config path | Default `~/.config/model_utils/pi05_export.yaml`, unless the user requests another path. |
| Schedule file | Existing strict schedule selected above, or unset for generated uniform schedule. |
| Task | Required only when `verify` is selected and later useful as the mock pipeline `default_task`. |
| Calibration batch | Required for `vlm_quant`; must be representative real data. |
| Quant deployment | Required for quantized OM finalization, e.g. `ascend310p_w8a8`. |

Before running ATC on Ascend310P, inspect `npu-smi info`. If the toolkit cannot find standard C++
headers, use the installed compiler's actual include directories rather than hard-coding another
machine's paths. A known openEuler 310P environment may require:

```bash
export CPLUS_INCLUDE_PATH=/usr/include/c++/12:/usr/include/c++/12/aarch64-openEuler-linux${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
```

Only set this after confirming those directories exist.

## Environment

Every project or ROS command must load the workspace environment in the same shell:

```bash
source .shrc_local
```

After building code changes, also source the install space before runtime or ROS validation:

```bash
source .shrc_local
source install/setup.sh
```

## Conversion And Profile Creation

Use one explicit command and save the effective reusable parameters with `--save-as`.

Default FP16 example:

```bash
source .shrc_local
python3 -m model_utils.pi05_export \
    --config ~/.config/model_utils/pi05_export.yaml \
    --policy-path /absolute/path/to/pi05_bundle \
    --exp-dir /absolute/path/to/pi05_export_run \
    --soc-version Ascend310P1 \
    --device npu \
    --dtype fp16 \
    --deployment ascend310p_fp16 \
    --steps vlm_onnx,ae_onnx,vlm_om,ae_om \
    --no-fast-gelu \
    --save-as pi05-310p-fp16-npugeglu
```

For an existing custom schedule, add:

```text
--schedule-file /absolute/path/to/selected_schedule.json
```

The file must contain exactly `format`, `name`, `algorithm`, `model_output`, and `timesteps`, with
`format: pi05-denoising-schedule-v1`, `algorithm: euler`, `model_output: velocity`, and strictly
decreasing timesteps from `1.0` to `0.0`.

FastGELU example changes only the selection and profile name:

```bash
source .shrc_local
python3 -m model_utils.pi05_export \
    --config ~/.config/model_utils/pi05_export.yaml \
    --policy-path /absolute/path/to/pi05_bundle \
    --exp-dir /absolute/path/to/pi05_export_run-fastgelu \
    --soc-version Ascend310P1 \
    --device npu \
    --dtype fp16 \
    --deployment ascend310p_fp16_fastgelu \
    --steps vlm_onnx,ae_onnx,vlm_om,ae_om \
    --fast-gelu \
    --save-as pi05-310p-fp16-fastgelu
```

`--save-as` writes the profile before the long conversion starts. If conversion fails, report that
the profile exists but the bundle deployment may be incomplete; do not claim success.

The exporter treats `steps` as transient and does not persist it in profiles. This prevents a later
profile invocation from unexpectedly rebuilding expensive stages. Therefore, always tell the user
to pass `--steps` again when reusing a profile. In contrast, `schedule_file` is persistent: confirm it
appears in the saved profile whenever a custom schedule was selected. Its absence means the exporter
will generate the default uniform schedule from `config.num_inference_steps`.

After resolving arguments, quote the exact profile location printed by the tool. With the default it
is:

```text
~/.config/model_utils/pi05_export.yaml
```

Tell the user how to inspect and reuse it:

```bash
source .shrc_local
python3 -m model_utils.pi05_export \
    --config ~/.config/model_utils/pi05_export.yaml \
    --list-profiles
```

```bash
source .shrc_local
python3 -m model_utils.pi05_export \
    --config ~/.config/model_utils/pi05_export.yaml \
    --profile pi05-310p-fp16-npugeglu \
    --steps vlm_onnx,ae_onnx,vlm_om,ae_om
```

CLI flags override profile values. Use this for a new `--exp-dir`, deployment name, task, or selected
steps without duplicating the whole profile.

## Success Criteria And Bundle Location

Success requires all requested products and, for a complete deployment, both OM roles plus valid ABI
files. The exporter prints the exact artifact paths. Normally:

```text
<exp-dir>/onnx/
<exp-dir>/runtime_save/
<exp-dir>/om/
```

The deployable bundle remains the original `--policy-path`, now updated with:

```text
<policy-path>/inference_manifest.json
<policy-path>/artifacts/ascend/<deployment>/...
```

The `model_path` used by inference is the bundle root (`--policy-path`), not `<exp-dir>/om` and not an
individual `.om` file. The `deployment` is the named deployment supplied during conversion.

At the end, always report:

- Absolute bundle path.
- Named deployment.
- Absolute experiment directory and OM directory.
- VLM and Action Expert OM paths.
- Profile name and profile YAML path.
- Whether default/custom steps and NPUGeglu/FastGELU were used.
- Schedule name/source/step count and whether `schedule_file` is recorded in the profile.
- Any incomplete role, failed verification, or accuracy warning.

## Accuracy Validation

When an established loss baseline/profile exists, use its existing targets and noises. Do not
generate replacement targets from the new OM. Compare the new deployment with the matching historical
branch: NPUGeglu vs FastGELU, FP16 vs W8A8, and denoising schedule must all match.

Example:

```bash
source .shrc_local
source install/setup.sh
python3 src/model_utils/model_utils/loss_compare.py \
    --config /absolute/path/to/loss_compare.yaml \
    --profile pi05-baseline \
    --policy_path /absolute/path/to/pi05_bundle \
    --deployment ascend310p_fp16
```

Treat regressions in normalized Raw L1, Raw cosine, W1/std, or first-frame cosine as possible export
or runtime adaptation errors. FastGELU and W8A8 may trade accuracy for speed, but the tradeoff must be
measured rather than assumed.

### Schedule Diagnostics And Tuning

`loss_compare --metrics-json` writes machine-readable aggregate latency and accuracy metrics.
`--schedule-override-path` and `--curvature-log-path` are compute-only, CLI-only diagnostics; they are
not saved to a `loss_compare` profile and cannot be used with `--generate-target`. Overrides are for
tuning only and require a velocity/`v_t` Action Expert output. The final schedule must be installed as
the deployment's versioned Manifest artifact through the owning packager.

Use an existing `loss_compare` profile with its existing batches, targets, raw targets, and noises.
Never generate targets from the deployment being tuned:

```bash
source .shrc_local
source install/setup.sh
python3 src/model_utils/model_utils/loss_compare.py \
    --config /absolute/path/to/loss_compare.yaml \
    --profile pi05-baseline \
    --policy_path /absolute/path/to/pi05_bundle \
    --deployment ascend310p_fp16 \
    --schedule-override-path /absolute/path/to/dense_uniform_20.json \
    --curvature-log-path /absolute/path/to/tuning/curvature.jsonl \
    --metrics-json /absolute/path/to/tuning/dense_metrics.json
```

Generate strict candidates from an existing curvature log when manual inspection is useful:

```bash
ros2 run model_utils pi05-curvature-schedule \
    --log /absolute/path/to/tuning/curvature.jsonl \
    --num-steps 3 4 5 \
    --output-dir /absolute/path/to/tuning/schedules
```

For the complete workflow, use the same existing profile. It runs a dense uniform schedule, evaluates
uniform and curvature candidates with the profile's targets/noises, writes reports, and installs the
winner in the Manifest by default:

```bash
ros2 run model_utils pi05-tune-schedule \
    --config /absolute/path/to/loss_compare.yaml \
    --profile pi05-baseline \
    --policy-path /absolute/path/to/pi05_bundle \
    --deployment ascend310p_fp16 \
    --candidate-steps 3 4 5 \
    --metric raw_l1 \
    --artifacts-dir /absolute/path/to/tuning/run
```

Use `--no-install` only when the user explicitly wants reports without changing the bundle. In that
case, install `schedules/selected.json` later through `--schedule-file` or rerun the tuner without
`--no-install`; a diagnostic override is not a deployable final state.

## Hardware Mock Validation

Use `hardware_mock`, not Gazebo, to validate the generated bundle end to end without physical cameras
or a real SO-101 arm. The mock publishes contract-compatible images and joint states and consumes the
inference action topics.

The internal multi-camera vision batching optimization does not change this procedure. `hardware_mock`
validates the raw image/topic contract and needs no model-specific batching, velocity, or schedule
changes.

Do not edit the repository's shared robot YAML just for a test. Create a temporary copy below the
workspace `tmp/` directory and update only the policy pipeline:

```yaml
robot:
  simulation:
    platform: mock
  control_modes:
    model_inference:
      inference:
        enabled: true
        pipelines:
          policy:
            model_path: /absolute/path/to/pi05_bundle
            deployment: ascend310p_fp16
            execution_mode: monolithic
            request_timeout: 120.0
            default_task: "Grasp banana and put it on the plate"
      executor:
        inference_pipeline: policy
```

Preserve the rest of the source robot contract, joints, peripherals, and action configuration. Start
the unified stack with the temporary absolute config path:

```bash
source .shrc_local
source install/setup.sh
export ROS_DOMAIN_ID=42
ros2 launch robot_config robot.launch.py \
    config_path:=/absolute/path/to/tmp/so101_pi05_mock.yaml \
    control_mode:=model_inference \
    use_sim:=true \
    sim_platform:=mock
```

The first PI05/ACL initialization can take several minutes. Do not use a short fixed sleep as the
readiness criterion. Poll the inference action server until it reports one server:

```bash
ros2 action info /inference/policy/dispatch
```

Then verify the mock and inference graph:

```bash
ros2 node list
ros2 action list
ros2 topic list
ros2 topic echo /joint_states --once
ros2 topic hz /camera/top/image_raw
ros2 topic echo /action_dispatcher/queue_size --once
```

Expected evidence:

- `/contract_mock`, inference policy, and action dispatcher nodes are present.
- `/inference/policy/dispatch` has an action server.
- Mock camera topics publish continuously.
- `/joint_states` contains the configured joints.
- The action queue becomes non-empty and actions are consumed without fatal errors.

Terminate the launch cleanly with SIGINT after collecting logs. Report the absolute mock YAML and log
paths so the test can be reproduced.

## Troubleshooting

### Profile was saved but conversion failed

The profile is reusable configuration, not proof of successful compilation. Report the failing step,
retain successful artifacts, fix the cause, and rerun the required `--steps` with `--profile`.

### Deployment is missing after a single OM step

A new split PI05 deployment needs both VLM and Action Expert OM artifacts. Run the missing counterpart
step once to create it. For an existing named deployment, a single-role rerun reuses the counterpart
artifact and bindings from the Manifest and publishes the rebuilt role as a new generation.

### ATC cannot find `cstdint`

Inspect `g++ --version` and the actual C++ include directories. On the verified openEuler GCC 12
environment, exporting `CPLUS_INCLUDE_PATH` to `/usr/include/c++/12` and its platform subdirectory
resolved this. Do not assume those paths on another host.

### Mock launch has no inference action server yet

Check the launch log and NPU status, then continue polling. PI05 model and ACL initialization can still take
time, but schema-v2 Manifest loading does not hash large bundle files; node presence alone does not mean the
inference engine is ready.

### W8A8 result is much worse than the FP16 baseline

Confirm that calibration uses representative real batches and that the historical comparison branch
has the same quantized roles. A `VLM W8A8 + AE W8A8` deployment is not equivalent to a historical
`VLM W8A8 + AE FP16` branch.

## References

- `src/model_utils/model_utils/pi05_export/__main__.py`
- `src/model_utils/model_utils/pi05_export/_cli.py`
- `src/model_utils/model_utils/pi05_om_dump.py`
- `src/model_utils/model_utils/loss_compare.py`
- `src/hardware_mock/README.md`
- `src/robot_config/README.md`
