# Ascend OM Performance Baseline

The required performance check is `ais_bench --loop 50` for every OM executed by the deployment. Do
not run hardware mock, LTTng, tracing, or trace-summary checks.

## Preflight

On the Ascend host, record:

- deployment and Manifest fingerprint;
- exact role order and invocation count per generated action;
- OM and ABI absolute paths and SHA-256;
- input shapes and batch size;
- `npu-smi info`;
- CANN, ACL, driver, and ais_bench versions;
- schedule/step count or other values that determine role invocation count.

Only benchmark an OM that has exact ACL ABI and belongs to the deployment under test.

## Per-Role Command

Run each distinct role OM:

```bash
source .shrc_local
python3 -m ais_bench \
    --model "RESOLVED_ROLE_OM" \
    --loop 50 \
    --debug 0
```

If the model requires explicit inputs that ais_bench cannot generate correctly, create deterministic
inputs matching the ACL ABI and record the exact ais_bench arguments. Use the same input method for all
candidates. A role benchmark is invalid if its input shape or dtype differs from production ABI.

Preserve the raw log and parse:

- `NPU_compute_time` min;
- max;
- mean;
- median;
- p99;
- throughput.

Do not substitute wall-clock initialization time for NPU compute time.

## Weighted Total

Compute the primary total:

```text
total_mean_ms = sum(role_mean_ms * role_invocation_count)
```

Examples:

- single ACT policy: `policy_mean`;
- PI05: `vlm_mean + action_expert_mean * denoising_steps`.

Also report:

```text
total_median_estimate = sum(role_median_ms * invocation_count)
total_p99_upper_estimate = sum(role_p99_ms * invocation_count)
```

The latter values are weighted estimates, not measured end-to-end percentiles. Label them exactly as
estimates.

For conditionally executed roles, benchmark each path and report invocation assumptions. Do not hide
conditional behavior inside one total.

## Report

Write machine-readable JSON and a short Markdown summary under `reports/ais-bench/`. Include:

- environment and artifact identity;
- exact commands;
- raw log paths;
- per-role statistics;
- invocation counts and source of each count;
- weighted totals;
- failures or retries.

For optimization candidates, compare against the same baseline environment and input method. Accept a
performance candidate only when its accepted accuracy remains valid and `total_mean_ms` improves beyond
measurement noise. Record candidates that show no additive benefit; PI05 demonstrated that two local
optimizations can hit the same GEMM/memory floor and fail to stack.
