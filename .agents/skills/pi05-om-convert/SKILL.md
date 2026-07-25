---
name: pi05-om-convert
description: "Internal PI0.5/PI05-specific Ascend OM conversion workflow. Use only after om-convert has validated a PI05 bundle path and resolved the exact ATC soc_version. Exports VLM and Action Expert ONNX/OM artifacts, confirms steps, FastGELU and denoising schedule, saves a reusable profile, and validates with hardware_mock."
---

# PI05 Ascend OM Conversion Sub-Skill

Convert a local PI0.5 LeRobot policy into a strict IB-Robot Ascend deployment containing VLM and
Action Expert OM artifacts, compiler ABI metadata, and a named deployment in
`<bundle>/inference_manifest.json`.

Use the repository entry point `python3 -m model_utils.pi05_export`. Do not create a second export
script, hand-edit generated ABI bindings or identities, or deploy raw OM paths outside the policy bundle.

This is an internal model-specific executor. `om-convert` is the public entry point.

## Parent Handoff Contract

Before any PI05-specific question or command, require all of these values from `om-convert`:

| Field | Requirement |
|-------|-------------|
| `model_path` | Absolute existing bundle path |
| `model_type` | Exactly `pi05`, validated from `<model_path>/config.json` |
| `soc_version` | Exact ATC target value, including the full chip revision |
| Host evidence | OS, `npu-smi` result, and ATC version or availability |

If any value is absent, load `om-convert` and stop this workflow. Do not independently guess or ask
again for the model path, model family, or target SoC. If the config changes after handoff, stop on the
mismatch and return to the parent router.

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

For RKNN use `rknn-convert`. For Houmo HMM use `hmm-convert`. Return ACT requests to `om-convert`,
which delegates them to `act-om-convert`. Do not route SmolVLA OM requests through this skill.

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
| Policy path | Use the absolute `model_path` inherited from `om-convert`; do not ask again. |
| Experiment directory | New or existing directory for `onnx/`, `runtime_save/`, and `om/`. |
| SoC version | Use the exact `soc_version` inherited from `om-convert`; do not replace it with a family-level default. |
| Torch export device | Resolve explicitly: use `npu` when a compatible local Ascend device is available; otherwise use `cpu` for supported ONNX-only work. |
| Deployment name | Named FP deployment written into `inference_manifest.json`, e.g. `ascend_target_fp16`. |
| Profile name | Reusable, descriptive name, e.g. `pi05-target-fp16-npugeglu`. |
| Profile config path | Default `~/.config/model_utils/pi05_export.yaml`, unless the user requests another path. |
| Schedule file | Existing strict schedule selected above, or unset for generated uniform schedule. |
| Task | Required only when `verify` is selected and later useful as the mock pipeline `default_task`. |
| Calibration batch | Required for `vlm_quant`; must be representative real data. |
| Quant deployment | Required for quantized OM finalization, e.g. `ascend_target_w8a8`. |

The parent router has already inspected `npu-smi info`; include that evidence in the report. If the
toolkit cannot find standard C++ headers, use the installed compiler's actual include directories
rather than hard-coding another machine's paths. A known openEuler 310P environment may require:

```bash
export CPLUS_INCLUDE_PATH=/usr/include/c++/12:/usr/include/c++/12/aarch64-openEuler-linux${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}
```

Only set this after confirming those directories exist.

Before selecting any `*_om` step, verify that Python ACL can inspect OM descriptors on a compatible
local NPU. The PI05 exporter uses `write_acl_om_abi()` after ATC. On an Ubuntu cross-conversion host
without a compatible NPU, ONNX-only steps can proceed, but a new deployable OM bundle cannot be
finalized locally. Stop before the long OM conversion unless the user explicitly accepts this partial
result; never synthesize ABI metadata from ONNX.

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

Default FP16 template. Replace every `RESOLVED_*` token with the parent handoff values before running;
never execute placeholder text:

```bash
source .shrc_local
python3 -m model_utils.pi05_export \
    --config ~/.config/model_utils/pi05_export.yaml \
    --policy-path "RESOLVED_MODEL_PATH" \
    --exp-dir /absolute/path/to/pi05_export_run \
    --soc-version "RESOLVED_SOC_VERSION" \
    --device "RESOLVED_EXPORT_DEVICE" \
    --dtype fp16 \
    --deployment "RESOLVED_DEPLOYMENT" \
    --steps vlm_onnx,ae_onnx,vlm_om,ae_om \
    --no-fast-gelu \
    --save-as "RESOLVED_PROFILE_NAME"
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
    --policy-path "RESOLVED_MODEL_PATH" \
    --exp-dir /absolute/path/to/pi05_export_run-fastgelu \
    --soc-version "RESOLVED_SOC_VERSION" \
    --device "RESOLVED_EXPORT_DEVICE" \
    --dtype fp16 \
    --deployment "RESOLVED_DEPLOYMENT" \
    --steps vlm_onnx,ae_onnx,vlm_om,ae_om \
    --fast-gelu \
    --save-as "RESOLVED_PROFILE_NAME"
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
    --profile "RESOLVED_PROFILE_NAME" \
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
- Exact target `soc_version` and how `om-convert` resolved it.
- Local `npu-smi` and ATC evidence inherited from `om-convert`.
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
    --policy_path "RESOLVED_MODEL_PATH" \
    --deployment "RESOLVED_DEPLOYMENT"
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
    --policy_path "RESOLVED_MODEL_PATH" \
    --deployment "RESOLVED_DEPLOYMENT" \
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
    --policy-path "RESOLVED_MODEL_PATH" \
    --deployment "RESOLVED_DEPLOYMENT" \
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
            model_path: "RESOLVED_MODEL_PATH"
            deployment: "RESOLVED_DEPLOYMENT"
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

- `.agents/skills/om-convert/SKILL.md`
- `src/model_utils/model_utils/pi05_export/__main__.py`
- `src/model_utils/model_utils/pi05_export/_cli.py`
- `src/model_utils/model_utils/pi05_om_dump.py`
- `src/model_utils/model_utils/loss_compare.py`
- `src/hardware_mock/README.md`
- `src/robot_config/README.md`
