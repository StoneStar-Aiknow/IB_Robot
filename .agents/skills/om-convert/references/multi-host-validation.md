# Multi-Host Accuracy Workflow

Assume Torch target generation, conversion, and Ascend deployment may run on different machines.
Never assume paths or environments are shared.

## Topology Record

Create `reports/hosts.json` without secrets:

```json
{
  "torch": {
    "location": "local-or-user@ip",
    "workspace": "/absolute/IB_Robot",
    "bundle": "/absolute/policy",
    "deployment": "torch-device",
    "device": "cpu|cuda|npu"
  },
  "conversion": {
    "location": "local-or-user@ip",
    "workspace": "/absolute/IB_Robot",
    "bundle": "/absolute/policy",
    "experiment": "/absolute/experiment"
  },
  "ascend": {
    "location": "local-or-user@ip",
    "workspace": "/absolute/IB_Robot",
    "bundle": "/absolute/policy",
    "deployment": "ascend-name",
    "device_id": 0
  }
}
```

Ask once whether remote execution is allowed. Use the user's existing SSH configuration. Never ask for
passwords, private keys, access tokens, or secret file contents. If direct transfer is unavailable,
provide exact commands and stop at a clear handoff point.

Many lab hosts disable password authentication. Before starting a cross-host run, ask the user to
prepare key-based access from the controlling machine when needed:

```bash
ssh-copy-id "RESOLVED_USER@RESOLVED_HOST"
ssh -o BatchMode=yes "RESOLVED_USER@RESOLVED_HOST" true
```

If policy forbids `ssh-copy-id`, ask the user to use the organization's approved key provisioning
method. Do not change `sshd_config` or enable password authentication.

## Canonical Validation Package

Keep one package of immutable comparison inputs:

```text
validation-inputs/
├── observations.safetensors
├── target.json
├── target_raw.json
├── noises/
├── validation.json
└── SHA256SUMS
```

`validation.json` records policy type, source bundle, Torch deployment, observation provenance, task,
seed, sample count, generation command, IB-Robot SHA, and LeRobot SHA. Compute hashes after target
generation and verify them after every transfer.

This is the only authoritative action-target package for its recorded reference contract. Do not create
new target directories for exporter fixes, host preprocessing variants, dtype experiments, ONNX/OM
candidates, or optimization stages. Save their outputs under reports or candidate directories with names
such as `candidate_output.json`, `metrics.json`, or `stage_reference/`, never as another consumable
`target.json` package.

If multiple historical target directories already exist, do not choose by name or recency. Identify the
authoritative package by manifest and hashes, then prove it by rerunning the accepted baseline artifact and
reproducing its recorded metrics. Mark the others as historical diagnostic snapshots in the experiment
ledger and never mix files across them.

## Observation Batch

Prefer a real local LeRobot dataset:

```bash
source .shrc_local
source install/setup.sh
ros2 run model_utils observation-batch dataset \
    --dataset-root "RESOLVED_DATASET_ROOT" \
    --policy-path "RESOLVED_TORCH_BUNDLE" \
    --output "RESOLVED_INPUT_DIR/observations.safetensors" \
    --samples "RESOLVED_SAMPLE_COUNT" \
    --sampling episode-stratified \
    --seed "RESOLVED_SEED"
```

Inspect it:

```bash
ros2 run model_utils observation-batch inspect \
    "RESOLVED_INPUT_DIR/observations.safetensors"
```

An existing versioned batch is also valid. Deterministic random observations are allowed only for
smoke conversion when the user has no data; label resulting accuracy as non-representative. Random
observations are never valid calibration input.

## Torch Target Generation

On the Torch host, first prove IB-Robot loads and executes the original Torch deployment. Then run:

```bash
source .shrc_local
source install/setup.sh
python3 src/model_utils/model_utils/loss_compare.py \
    --policy_path "RESOLVED_TORCH_BUNDLE" \
    --deployment "RESOLVED_TORCH_DEPLOYMENT" \
    --batch_path "RESOLVED_INPUT_DIR/observations.safetensors" \
    --task "RESOLVED_TASK" \
    --seed "RESOLVED_SEED" \
    --exp-dir "RESOLVED_INPUT_DIR" \
    --generate-target
```

Use a new experiment directory. Do not add `--force` unless the user explicitly approves replacing a
baseline. For stochastic policies, target generation must persist `noises/`; missing noises is a hard
failure for cross-machine comparison.

Run target generation once per reference-contract version. After sealing, all later diagnostics and
optimization candidates reuse it. A failed candidate must not trigger target regeneration.

This applies to PI0, PI05, SmolVLA, Diffusion Policy, and any new policy whose production inference
accepts random noise or stochastic control inputs. Verify identical persisted noise reproduces
identical Torch actions, different noise changes actions, and the Ascend run consumes the transferred
noise rather than regenerating it.

The current `loss_compare` implements persisted noise only for PI05. Before validating any other
stochastic family, extend its codec/runtime and comparison harness with family-specific control-input
generation, versioned storage, replay, shape/dtype validation, and effectiveness tests. Do not assume
`--seed` alone reproduces a backend's internal random state.

## Seal The Package

After target generation, create `validation.json` with the resolved values. Use sorted JSON for a
stable record. Replace all placeholders before execution and keep numeric values numeric:

```bash
python3 - "RESOLVED_INPUT_DIR/validation.json" "RESOLVED_SEED" "RESOLVED_SAMPLE_COUNT" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "format": "ibrobot-om-validation-v1",
    "policy_type": "RESOLVED_POLICY_TYPE",
    "source_bundle": "RESOLVED_TORCH_BUNDLE",
    "torch_deployment": "RESOLVED_TORCH_DEPLOYMENT",
    "observation_batch": "observations.safetensors",
    "observation_provenance": "RESOLVED_PROVENANCE",
    "task": "RESOLVED_TASK",
    "seed": int(sys.argv[2]),
    "sample_count": int(sys.argv[3]),
    "generation_command": "RESOLVED_GENERATION_COMMAND",
    "ibrobot_revision": "RESOLVED_IBROBOT_SHA",
    "lerobot_revision": "RESOLVED_LEROBOT_SHA",
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
```

Then hash every package file except the checksum list itself:

```bash
python3 - "RESOLVED_INPUT_DIR" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve(strict=True)
files = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
lines = []
for path in files:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    lines.append(f"{digest}  {path.relative_to(root)}")
(root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
```

## Transfer

Transfer the complete validation package, not selected files. Example using an approved SSH target:

```bash
rsync -a --checksum "RESOLVED_INPUT_DIR/" \
    "RESOLVED_ASCEND_SSH:RESOLVED_ASCEND_INPUT_DIR/"
```

Quote paths. Verify `SHA256SUMS` on the destination. If `rsync` is unavailable, use `scp -r` and verify
hashes separately. Do not silently regenerate observations, targets, raw targets, or noises on another
host.

Use `sha256sum -c SHA256SUMS` when available, or a Python equivalent that parses each digest and
relative path. Do not include `SHA256SUMS` in itself.

Transfer the deployable bundle or managed generation separately. Preserve `inference_manifest.json`,
all referenced artifacts, processor assets, and relative paths.

## Ascend Compute-Loss

On the Ascend host, use the transferred package:

```bash
source .shrc_local
source install/setup.sh
python3 src/model_utils/model_utils/loss_compare.py \
    --policy_path "RESOLVED_ASCEND_BUNDLE" \
    --deployment "RESOLVED_ASCEND_DEPLOYMENT" \
    --batch_path "RESOLVED_ASCEND_INPUT_DIR/observations.safetensors" \
    --task "RESOLVED_TASK" \
    --seed "RESOLVED_SEED" \
    --exp-dir "RESOLVED_ASCEND_INPUT_DIR" \
    --metrics-json "RESOLVED_REPORT_DIR/torch-vs-om.json"
```

Never use `--generate-target` with the Ascend deployment under test. The task and seed must match
target generation exactly.

Before executing the command, verify that `--batch_path` and `--exp-dir`, plus any explicit target,
raw-target, or noise paths, all resolve to the same transferred package. Do not manually assemble a
comparison from similarly named files in separate directories.

## Handoff Report

Record commands, source/destination paths, hashes, host workspaces, IB-Robot/LeRobot revisions, and
whether the agent or user performed each transfer. A cross-host check with mismatched inputs is invalid.
