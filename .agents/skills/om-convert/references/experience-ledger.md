# Experience Ledger And Final Report

Use a stable catalog so conversion and optimization reports can show which accumulated experiences
actually helped. Create `reports/experience-ledger.json` at the start of work and update it as evidence
arrives.

## Catalog

Use these 12 experience IDs for every run:

| ID | Experience |
|----|------------|
| `E01` | Reuse an IB-Robot Torch-supported bundle, codec, processor, and vendored assets. |
| `E02` | Persist one canonical observation/task/seed/noise package across hosts and backends. |
| `E03` | Export a portable FP16 ONNX baseline with only correctness-required FP32 islands. |
| `E04` | Keep image resize, padding, and risky reshape logic on the host when ATC semantics are suspect. |
| `E05` | Use `precision_mode_v2=origin|default` and investigate precision before approximating math. |
| `E06` | Diagnose fusion-induced drift by compiling a no-fusion candidate. |
| `E07` | Detect high-priority-library misses and correct dtype/layout/operator eligibility. |
| `E08` | Eliminate exact glue, redundant materialization, and unnecessary host/device transfer. |
| `E09` | Replace proven exact subgraphs with supported NPU fused operators. |
| `E10` | Use `msprof` to rank measured bottlenecks before optimization. |
| `E11` | Scan valid static shapes/tiling empirically rather than guessing. |
| `E12` | Use explicit optional quantization only with representative real calibration data. |

Do not inflate the denominator with every sentence in the skill. The fixed catalog makes results
comparable between runs.

## Status

Each entry has one status:

- `hit`: applied to source/export/compiler/runtime behavior, passed required accuracy, and either
  enabled conversion or produced a measured accepted performance gain;
- `attempted_no_gain`: applicable and tested, but rejected for accuracy, performance, or complexity;
- `not_applicable`: inspected but the model/toolchain did not contain the prerequisite pattern;
- `not_evaluated`: outside the selected conversion/optimization scope.

Label the headline as hits among evaluated experiences. `x/y` uses:

```text
x = count(status == "hit")
y = count(status in {"hit", "attempted_no_gain", "not_applicable"})
```

Exclude `not_evaluated` because the user may choose conversion only and never enter optimization.
A hit requires evidence; merely following a default does not count unless it solved a real conversion
constraint or produced a measured gain.

Also report catalog coverage as `evaluated/12` so `1/1` cannot be mistaken for full experience
coverage.

## Entry Schema

```json
{
  "id": "E04",
  "status": "hit",
  "stage": "conversion",
  "evidence": ["torch-vs-onnx.json", "torch-vs-om.json"],
  "effect": "Moved resize/padding to host and restored accepted action cosine",
  "reproduce": "reports/reproduce.sh#host-image-preprocess"
}
```

## Final Report Section

End every conversion or optimization report with:

```text
Experience hit among evaluated: x/y
Experience coverage: y/12
Successful: E02, E04, E05
Attempted without accepted gain: E06
Not applicable: E08, E09
Not evaluated in this run: E10, E11, E12
```

List one-line evidence for each successful item. Keep the machine-readable ledger and do not report
private host credentials, personal note paths, or environment details that do not help reproduction.
