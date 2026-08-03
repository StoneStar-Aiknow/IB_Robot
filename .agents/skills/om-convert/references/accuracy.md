# Accuracy Gates

The workflow has exactly two numerical comparisons:

1. Torch vs portable ONNX;
2. Torch vs Ascend OM.

Do not add hardware-mock, tracing, or task-success gates here.

## Limits Collected Up Front

Before export, collect separate limits for both comparisons:

| Comparison | Limits |
|------------|--------|
| Torch vs ONNX | `max_abs_max`, `mean_l1_max`, and `cosine_min`. |
| Torch vs Ascend OM | `mean_l1_max` and `cosine_min`, matching the current `loss_compare` metrics. |

The user may select `report only`. In that mode compute all metrics but use verdict
`needs-user-acceptance`, not `pass`. Do not enter optional optimization until the user accepts the
baseline.

Record limits in `reports/accuracy-limits.json`. A later optimization may tighten limits but must not
silently relax them. Relaxation is a hard user-approval gate even in autonomous mode.

## Canonical Inputs

Both comparisons use the same:

- versioned observation batch;
- task;
- seed;
- persisted noise/control inputs;
- source Torch bundle and deployment;
- preprocessing and postprocessing semantics.

Persist stochastic control inputs as versioned artifacts with explicit canonical storage dtype and each
backend's consumer dtype. Do not assume that a policy's dominant weight dtype is also its noise input dtype;
inspect the actual consuming layer or manifest binding. When Torch and OM consume different dtypes, preserve
one canonical FP32 sample and independently cast it once at each backend boundary.

Generate targets only from the original Torch deployment as described in `multi-host-validation.md`.
Never regenerate targets from ONNX or OM.

### Single Authoritative Target Rule

For one fixed reference contract, generate native-Torch action targets exactly once. The reference
contract includes policy weights and revision, LeRobot revision, native Torch deployment and dtype,
observation batch, task, persisted noise/control inputs, preprocessing, postprocessing, schedule, step
count, and action semantics.

After the package is sealed:

- every portable ONNX, OM, ATC, runtime, and semantics-preserving optimization candidate uses the same
  `target.json`, `target_raw.json`, observations, and noises;
- exporter fixes, host resize/padding experiments, dtype probes, graph rewrites, kernel selection, and
  performance candidates do not justify generating new action targets;
- diagnostic runs may save `candidate_output*.json`, metrics, tensor dumps, or native intermediate-stage
  references, but must not name or treat them as alternate targets;
- native FP32 or wrapper outputs used to diagnose native BF16 behavior are candidate/control outputs,
  not new authoritative targets.

Create a new versioned target package only when the user intentionally changes the reference contract,
or when evidence proves the existing package was generated incorrectly. Never overwrite the prior
package. Record the old and new package identities, the exact contract change, and why regeneration was
necessary. A candidate disagreeing with the target is evidence to investigate, not permission to
regenerate the target around the candidate.

### Accuracy Preflight

Before every Torch-vs-ONNX or Torch-vs-OM comparison:

1. identify the authoritative package directory and its `validation.json`/`SHA256SUMS` identity;
2. verify observations, target, raw target, and every persisted noise/control file against that package;
3. ensure every CLI path resolves inside that one directory and task, seed, sample count, revisions, and
   inference contract match its manifest;
4. record the package path and identity in the candidate report.

Do not combine target files from one directory with noises or observations from another. Matching
filenames, observation hashes, or sample counts do not prove that independently generated targets and
noises belong together. If the package lacks sufficient provenance, first reproduce the accepted
baseline with the complete package; do not infer authority from a directory name.

## Torch Vs ONNX

Keep a portable ONNX without NPU-only custom operators for this comparison. If the deployable ONNX
uses custom NPU operators, validate the portable graph first, then prove any custom-op rewrite against
the same Torch baseline or an exact subgraph reference before ATC.

After replacing operators with an NPU custom domain, standard ONNX Runtime commonly cannot load the
deployment graph because no NPU ORT kernel is registered. Do not report this as an accuracy failure and
do not pretend ORT validated the custom graph. Preserve the last portable ONNX report, prove the
replacement subgraph in eager Torch or a focused reference where possible, then require Torch-vs-OM
for the custom-op candidate. Record that the post-rewrite ORT gate was `not_applicable` and why.

Compare:

- every external output;
- each role handoff for a split model;
- raw action before postprocessing;
- final postprocessed action.

For each output report shape, dtype, finite status, max absolute error, mean L1, and cosine.
Aggregate verdict is the worst result across all required outputs and samples.

The report must contain policy/bundle identity, observation hash, task/seed, ONNX and external-data
hashes, exporter command, opset, runtime versions, thresholds, metrics, and verdict.

If ONNX Runtime cannot execute because of environment limits, diagnose and repair the environment or
portable graph. Do not substitute successful `onnx.checker` for numerical equivalence.

For split or patched exporters, also prove native Torch policy vs split Torch wrapper equivalence before
interpreting Torch-vs-ONNX results. A pure refactor must be bit-identical at the same dtype. Test each exporter
monkey patch independently; do not let a global patch mutate the native reference in the same process.

## Torch Vs Ascend OM

Use `loss_compare` against the Torch-generated target package on the Ascend host:

```bash
source .shrc_local
source install/setup.sh
python3 src/model_utils/model_utils/loss_compare.py \
    --policy_path "RESOLVED_ASCEND_BUNDLE" \
    --deployment "RESOLVED_ASCEND_DEPLOYMENT" \
    --batch_path "RESOLVED_OBSERVATIONS" \
    --task "RESOLVED_TASK" \
    --seed "RESOLVED_SEED" \
    --exp-dir "RESOLVED_TARGET_DIR" \
    --metrics-json "RESOLVED_REPORT_DIR/torch-vs-om.json"
```

Evaluate the configured mean-L1 and cosine limits using `loss_compare --metrics-json`. Preserve both
normalized and unnormalized metrics and all policy-specific diagnostics. The current tool does not
emit a global max-absolute metric, so do not collect or claim a Torch-vs-OM max-absolute verdict unless
the implementation is extended to calculate it.

Record exact `soc_version`, NPU, driver, CANN, ATC, ACL, deployment fingerprint, role OM hashes, and
ABI hashes. A changed OM hash invalidates the previous OM accuracy result.

If several materially different candidates produce suspiciously similar failures, or a result conflicts
with an accepted baseline, stop candidate attribution. Rerun the accepted baseline artifact against the
same complete package. If the baseline does not reproduce its accepted metrics, classify the comparison
as a validation/provenance failure rather than a model regression and repair the experiment record before
continuing.

## Candidate Rule

Every performance candidate must rerun any comparison affected by its changes:

- exporter/graph changes: Torch vs ONNX and Torch vs OM;
- ATC-only/compiler options: Torch vs OM, while retaining the matching validated ONNX identity;
- runtime-only buffer or execution changes: Torch vs OM;
- preprocessing, action, schedule, or other semantic changes: hard user approval, then both gates.

A candidate that improves latency but fails accepted accuracy limits is rejected and rolled back.

## Troubleshooting

When a gate fails, follow `precision-troubleshooting.md`. It records proven failure modes involving image
resize/padding, static vision position IDs, FP16 Softmax, task/token alignment, noise provenance, opset
lowering, eval-mode audits, and invalid comparison harnesses.
