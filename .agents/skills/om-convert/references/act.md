# ACT Ascend OM Conversion Workflow

Convert a local ACT LeRobot policy into one static Ascend ACL `policy` OM and package it as a named
deployment in `<bundle>/inference_manifest.json`.

This is an internal workflow reference. `om-convert` is the only user-facing entry point.

## Parent Handoff Contract

Before any ACT-specific operation, require all of these values from `om-convert`:

| Field | Requirement |
|-------|-------------|
| `model_path` | Absolute existing bundle path |
| `model_type` | Exactly `act`, validated from `<model_path>/config.json` |
| `soc_version` | Exact ATC value, such as `Ascend310P3` |
| Host evidence | OS, `npu-smi` result, and ATC version or availability |

If any value is absent, return to the shared resolution steps in `om-convert`. Do not independently
guess or ask again for the model path, model family, or target SoC. If the config changes after
handoff, stop on the mismatch and return to the parent workflow.

## Supported Scope

| Item | Current implementation |
|------|------------------------|
| Policy | ACT only |
| Torch export | CPU, static batch 1, opset 14 |
| Runtime artifact | One `policy` OM |
| Runtime | Ascend ACL |
| Input shape | Derived from `observation.state` and ordered `observation.images.*` in `config.json` |
| Output | Exactly one action tensor |
| Manifest | Schema-v2 named deployment |

The repository does not currently implement ACT OM conversion profiles, dynamic shapes,
quantization controls, FastGELU, denoising schedules, or an ACT-specific ONNX equivalence verifier.
Do not offer PI0.5 options in this workflow.

## ACT-Specific Inputs

After accepting the parent handoff, collect only choices that are not already resolved:

1. Conversion mode:
   - `Fresh bundle export (Recommended)`: export ACT ONNX, compile OM, inspect ABI, and package.
   - `Existing ONNX`: compile the supplied ONNX, inspect ABI, and package.
   - `Resume compiled OM`: inspect and package an unchanged OM after copying its source ONNX and
     bundle to a compatible ACL host.
2. Work directory: default `<model_path>/model_utils_work/ascend`.
3. Deployment name: default to a descriptive target name such as `ascend_310p3`.
4. ACL device ID for ABI inspection: default `0`.
5. Optional ACL config path: unset unless the environment requires one.

Do not ask about defaults individually when the user accepts the recommended fresh workflow. State
the defaults once and continue.

## Preflight

Load the workspace environment before every project command:

```bash
source .shrc_local
```

Verify the bundle contains:

- `config.json` with `type: act`;
- `policy_preprocessor.json` and its referenced assets;
- `policy_postprocessor.json` and its referenced assets;
- `input_features` containing `observation.state` and/or `observation.images.*`;
- `output_features.action`.

`model.safetensors` is additionally required for `Fresh bundle export`. It is not required when the
user explicitly supplies an existing ONNX or resumes a compiled OM.

For fresh export or existing ONNX conversion, verify ATC. For every mode, verify Python ACL and a
compatible NPU because this workflow always inspects the exact OM before packaging:

```bash
source .shrc_local
command -v atc
atc --version
python3 -c "import acl; print(acl.__file__)"
```

The `soc_version` is inherited from `om-convert`; do not replace it with a local default.

## Mandatory ABI Rule

Packaging requires runtime ABI metadata read from the exact compiled OM. ATC does not create this
JSON. The repository helper `write_acl_om_abi()` loads the OM with Python ACL and records exact input
and output names, indices, dtypes, and shapes.

Never derive ABI JSON from ONNX, copy it from another build, or hand-edit compiler-mangled names.

If local Python ACL and a compatible NPU are available, generate ABI immediately after ATC. If the
Ubuntu conversion host has no compatible NPU:

1. compile the OM for the user-selected target;
2. report the conversion as incomplete and do not create the deployment;
3. move that unchanged OM to a compatible Ascend runtime host;
4. run `write_acl_om_abi()` there;
5. bring the exact OM and generated ABI together for packaging.

An OM file without its exact runtime-inspected ABI is a compiler artifact, not a deployable bundle.

## Fresh Conversion

The public `export_onnx_atc.py` CLI expects an ABI file but does not generate one. For a first-time
conversion, call the repository's existing functions in the required order; do not create another
export script.

Construct one self-contained command. Replace every `RESOLVED_*` token before execution; never run
placeholder text or reuse shell state from an earlier command:

```bash
source .shrc_local
python3 - <<'PY'
import json
import os
from pathlib import Path

from model_utils.export_onnx_atc import export_act_model, write_ascend_deployment
from model_utils.inference_manifest_export import write_acl_om_abi

bundle = Path("RESOLVED_MODEL_PATH").expanduser().resolve(strict=True)
work_dir = Path("RESOLVED_WORK_DIR").expanduser().resolve()
work_dir.mkdir(parents=True, exist_ok=True)
config = json.loads((bundle / "config.json").read_text(encoding="utf-8"))
onnx_path = work_dir / "model.onnx"
om_path = work_dir / "model.om"
abi_path = work_dir / "model.om.abi.json"
soc_version = "RESOLVED_SOC_VERSION"

if str(config.get("type", "")).strip().lower() != "act":
    raise ValueError(f"Expected ACT config, got {config.get('type')!r}")
if not export_act_model(str(bundle), config, str(onnx_path), str(om_path), soc_version):
    raise RuntimeError("ACT ONNX export or ATC conversion failed")

write_acl_om_abi(
    om_path,
    abi_path,
    device_id=int(os.environ.get("ASCEND_DEVICE_ID", "0")),
    acl_config_path=os.environ.get("ACL_CONFIG_PATH") or None,
)
manifest_path = write_ascend_deployment(
    str(bundle),
    config,
    str(onnx_path),
    str(om_path),
    soc_version,
    str(abi_path),
    "RESOLVED_DEPLOYMENT",
)
print(manifest_path)
PY
```

This uses only the current repository implementation:

- `export_act_model()` exports ONNX and invokes ATC;
- `write_acl_om_abi()` introspects the exact resulting OM;
- `write_ascend_deployment()` copies the OM into a managed generation and atomically updates the
  Manifest.

If any stage fails, retain useful work artifacts but do not claim that the deployment exists.

## Existing ONNX Or OM

For an existing ONNX, verify it belongs to the same ACT bundle. In the fresh workflow, replace the
default `onnx_path` assignment and `export_act_model()` call with:

```python
from model_utils.export_onnx_atc import convert_onnx_to_om

onnx_path = Path("RESOLVED_ONNX_PATH").expanduser().resolve(strict=True)
if not convert_onnx_to_om(config, str(onnx_path), str(om_path), soc_version):
    raise RuntimeError("ATC conversion failed")
```

Continue with ABI inspection and packaging in the same order.

To resume an OM compiled on a host without a compatible NPU, copy the unchanged OM together with its
source ONNX and complete bundle to the ACL host. In the fresh workflow, replace the default path
assignments and `export_act_model()` call with:

```python
onnx_path = Path("RESOLVED_ONNX_PATH").expanduser().resolve(strict=True)
om_path = Path("RESOLVED_OM_PATH").expanduser().resolve(strict=True)
abi_path = om_path.with_name(f"{om_path.name}.abi.json")
```

Continue directly with `write_acl_om_abi()` and `write_ascend_deployment()`. Do not rerun ATC between
ABI inspection and packaging.

Do not use the public `export_onnx_atc.py` CLI as an all-in-one workflow. It compiles before packaging
but does not generate ABI, so it cannot guarantee the required compile, inspect, package ordering by
itself.

## Success Criteria

Compilation and packaging succeed only when all of the following hold:

- the requested ONNX and OM files exist;
- ACL inspection produced ABI JSON for that OM;
- runtime input order exactly matches the ACT-consumed config features;
- the OM exposes exactly one output mapped to semantic `action`;
- `<model_path>/inference_manifest.json` contains the named `ascend` deployment;
- strict production Manifest loading succeeds.

The deployable `model_path` is the original bundle root, not the work directory or individual OM.
Normally the managed artifact is under:

```text
<model_path>/artifacts/ascend/<deployment>/generations/<generation>/policy.om
```

## Validation

First load the generated deployment with the production validator:

```bash
source .shrc_local
python3 - "RESOLVED_MODEL_PATH" "RESOLVED_DEPLOYMENT" <<'PY'
import sys

from inference_manifest import load_inference_manifest

validated = load_inference_manifest(sys.argv[1], sys.argv[2])
print(validated.policy.policy_type)
print(validated.deployment.backend)
print(validated.deployment.target)
print(validated.fingerprint)
PY
```

Expect policy type `act`, backend `ascend`, runtime `acl`, the resolved target SoC, and execution
order `policy`.

For numerical validation, generate a Torch baseline once and compare the Ascend deployment against
the same batches and targets:

```bash
source .shrc_local
source install/setup.sh
ros2 run model_utils package-torch-deployment \
    --bundle-root "RESOLVED_MODEL_PATH" \
    --devices cpu

python3 src/model_utils/model_utils/loss_compare.py \
    --policy_path "RESOLVED_MODEL_PATH" \
    --deployment torch-cpu \
    --batch_path /absolute/path/to/batches.json \
    --exp-dir /absolute/path/to/act_comparison \
    --generate-target

python3 src/model_utils/model_utils/loss_compare.py \
    --policy_path "RESOLVED_MODEL_PATH" \
    --deployment "RESOLVED_DEPLOYMENT" \
    --batch_path /absolute/path/to/batches.json \
    --exp-dir /absolute/path/to/act_comparison \
    --metrics-json /absolute/path/to/act_comparison/metrics.json
```

Do not regenerate targets from the OM being evaluated. `loss_compare` establishes numerical fidelity;
`hardware_mock` separately validates ROS topics, contract adaptation, and action flow. Use a temporary
robot YAML with the absolute bundle path and named deployment, following `src/hardware_mock/README.md`.

## Final Report

Always report:

- absolute ACT bundle path;
- exact target `soc_version` and how `om-convert` resolved it;
- local `npu-smi` and ATC evidence;
- conversion mode and work directory;
- ONNX, OM, and ABI paths;
- deployment name and managed OM path;
- strict Manifest result;
- numerical and hardware-mock validation performed or still pending;
- any incomplete ABI, packaging, or accuracy work.

## References

- `.agents/skills/om-convert/SKILL.md`
- `src/model_utils/model_utils/export_onnx_atc.py`
- `src/model_utils/model_utils/inference_manifest_export.py`
- `src/model_utils/model_utils/loss_compare.py`
- `src/model_utils/model_utils/README.md`
- `src/model_utils/test/test_export_onnx_atc.py`
- `src/hardware_mock/README.md`
