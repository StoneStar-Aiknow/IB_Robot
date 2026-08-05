# Experience Ledger And Final Report

Use a stable catalog so conversion and optimization reports can show which accumulated experiences
actually helped. Create `reports/experience-ledger.json` at the start of work and update it as evidence
arrives.

## Catalog

Use these 12 experience IDs for every run:

| ID | Experience |
|----|------------|
| `E01` | Reuse an IB-Robot Torch-supported bundle, codec, processor, and vendored assets. |
| `E02` | Persist one authoritative native-Torch target/observation/task/seed/noise package across all candidates. |
| `E03` | Prefer opset-17 portable FP16 ONNX on 310P, with only correctness-required FP32 islands. |
| `E04` | Keep image resize, padding, and risky reshape logic on the host when ATC semantics are suspect. |
| `E05` | Use `precision_mode_v2=origin|default` and investigate precision before approximating math. |
| `E06` | Diagnose fusion-induced drift by compiling a no-fusion candidate. |
| `E07` | Correct dtype/layout/operator eligibility, especially FP32 MatMul/BMM misses on 310P. |
| `E08` | Eliminate exact glue and host transfer, including shared ACL buffers between serial OM roles. |
| `E09` | Use fused operators only after parser/kernel proof; keep 310P PFA low priority. |
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

`attempted_no_gain` is scoped to the recorded policy, shapes, toolchain, and exact SoC. It lowers priority
for a close repeat but does not make the experience globally inapplicable. Re-evaluate it when those inputs
change, especially utilization-driven choices such as multi-camera batching between Ascend310P3 and P1.

Also report catalog coverage as `evaluated/12` so `1/1` cannot be mistaken for full experience
coverage.

## Experience Gap Review

Run this review whenever the user asks for more, other, or alternative optimization approaches. Review
all 12 entries instead of considering only the most recent experiments:

1. collect `not_evaluated` entries and test their prerequisites against the current graph, profile, ATC
   warnings, exact SoC, toolchain, and accepted accuracy constraints;
2. keep applicable entries as remaining candidates and rank them by measured bottleneck share, expected
   mechanism, implementation cost, accuracy risk, and approval requirements;
3. retain `not_applicable` unless its recorded prerequisite evidence has changed;
4. retain `attempted_no_gain` unless policy family, shape, dtype, compiler, exact SoC, or another recorded
   assumption has changed;
5. exclude `hit` from the remaining list unless profiling after that gain exposes a distinct follow-up
   opportunity.

Report the review before proposing experiments:

```text
Remaining experience-backed candidates: E07, E09
Reconsidered because assumptions changed: E11 (new exact SoC)
Still not applicable or no-gain: E04, E06
No remaining catalog candidate: yes|no
```

For each remaining candidate, cite the evidence that makes it applicable and the smallest experiment
that can reject it. If none remain, say so explicitly. New ideas may still be proposed, but label them
`speculative` and do not count them as catalog experience until the catalog is intentionally revised.

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
