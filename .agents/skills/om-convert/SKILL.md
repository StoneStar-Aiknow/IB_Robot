---
name: om-convert
description: "Unified Ascend OM conversion entry point for ACT and PI0.5/PI05. Use whenever users mention 'convert to OM', 'Ascend OM', 'ATC', 'OM model conversion', '转OM', '转换OM', or '昇腾模型转换'. Requires the user to choose a model type and explicitly supply its bundle path, validates config.json, probes npu-smi, resolves the exact ATC soc_version, and executes the matching internal workflow."
---

# Ascend OM Conversion Router

This is the public entry point for Ascend ACL OM conversion. It owns only the shared decisions:

1. model family;
2. model bundle path;
3. target Ascend `soc_version`;
4. selection of one internal model-specific workflow.

Do not improvise model-specific export, ATC, packaging, or validation steps in this router. Once the
shared context is complete, read exactly one internal workflow reference and follow it.

## Supported Routes

| User choice / `config.json.type` | Internal reference | Current artifact layout |
|--------------------|-------------|-------------------------|
| `act` | `references/act.md` | One static ACT `policy` OM |
| `pi05` | `references/pi05.md` | PI0.5 VLM + Action Expert OM files |

For RKNN use `rknn-convert`. For Houmo HMM use `hmm-convert`. Do not treat Hisilicon OM output as
Ascend ACL OM, and do not invent an Ascend route for an unsupported policy type.

## Mandatory Resolution Order

Complete these gates in order. Use the `question` tool for choices or missing user input when
available; otherwise ask the user directly.

### 1. Resolve The Model Family And Path

First extract any model family and path already present in the request.

| User input | Required action |
|------------|-----------------|
| Neither family nor path | Ask which model family to convert, offering `PI0.5 / PI05` and `ACT`; then ask for the model bundle path. |
| Family only | Ask for the model bundle path. |
| Path only | Read `<path>/config.json`, derive the candidate family from `type`, and ask the user to confirm that inferred family before routing. |
| Family and path | Read `<path>/config.json` and verify that `type` agrees with the user. |

The user must explicitly provide the model bundle path. Never search `models/`, choose the newest
checkpoint, infer a path from the current directory, or reuse a path from an unrelated prior task.
A relative path is acceptable only when the user explicitly supplied it; resolve it to an absolute
path before continuing and report the resolved value.

The path must be an existing directory containing a readable `config.json`. Read the JSON. The raw
`type` value must be exactly `act` or `pi05`, because the production metadata and codec loaders require
canonical values. Lowercase/trim normalization may be used only to explain a likely metadata typo;
never route or convert until the bundle itself contains a canonical value.

When a path supplies the only model-family evidence, phrase the required confirmation concretely,
for example: `config.json.type is act; convert this as ACT?` Offer confirmation or correction rather
than asking the user to repeat information already read from the bundle.

If `type` is absent or unsupported, stop and ask for the correct bundle. If the user-stated family and
`config.json.type` disagree, show both values and ask the user to correct the path or family. Never
silently override the mismatch based on directory names, weight names, or architecture heuristics.

### 2. Probe The Conversion Host

Before asking for or choosing a target, inspect the host. Read `/etc/os-release`, then run the
following in the workspace with the environment loaded:

```bash
source .shrc_local
command -v npu-smi
npu-smi info
command -v atc
atc --version
```

Always attempt `npu-smi info`; do not wait for the user to name a platform. Record whether the command
is unavailable, fails to query a device, or reports a concrete Ascend chip revision. A successful
`npu-smi` probe identifies the local device, not automatically the deployment target.

Also verify that `atc` is available. Missing `atc` blocks compilation but does not remove the need to
resolve the target platform.

### 3. Resolve The Exact ATC `soc_version`

The resolved platform is the exact value passed to ATC as `--soc_version` and recorded in the
deployment Manifest as `target.soc`. Apply these rules:

| Host/probe result | Required action |
|-------------------|-----------------|
| `/etc/os-release` identifies Ubuntu | Require an explicit user-selected target `soc_version`, even when `npu-smi` reports a local NPU. If the request already contains the exact target, use it and do not ask redundantly; otherwise ask and present the detected device only as a candidate because Ubuntu is commonly a cross-conversion host. |
| Non-Ubuntu and `npu-smi` reports a full chip revision | Derive the candidate ATC value and state it before continuing. For example, a full `310P3` revision maps to `Ascend310P3`. |
| `npu-smi` is missing, fails, or reports only a family such as `310P` | Ask the user for the exact target `soc_version`; do not guess the revision. |
| User says the target differs from the local device | Use the explicitly selected target value. |

Preserve the complete chip revision. Do not collapse `Ascend310P1` and `Ascend310P3` into
`Ascend310P`, and do not infer a SoC from the model type, deployment name, OS architecture, or CANN
version. If the detected text cannot be converted unambiguously to an ATC value, ask the user.

Before compilation, summarize the evidence and final decision:

```text
Host OS: <ID and VERSION_ID from /etc/os-release>
Local NPU: <npu-smi result or unavailable>
ATC: <version or unavailable>
Target soc_version: <exact resolved value>
Resolution source: <Ubuntu user choice | explicit cross-target | npu-smi detection>
```

### 4. Select One Internal Workflow

Build this handoff context:

| Field | Value |
|-------|-------|
| `model_path` | Absolute resolved bundle path |
| `model_type` | Validated `config.json.type` |
| `soc_version` | Exact ATC target value |
| `host_os` | `/etc/os-release` identity |
| `npu_smi_evidence` | Detected device or failure/unavailable state |
| `atc_version` | Detected version or unavailable state |

Then select the internal workflow:

- For `act`, read `.agents/skills/om-convert/references/act.md` and follow it.
- For `pi05`, read `.agents/skills/om-convert/references/pi05.md` and follow it.

The internal workflow must not ask for the model path, model family, or `soc_version` again. If any
handoff field is missing, return to this shared resolution flow instead of guessing. Keep the handoff
values in the final conversion report so the chosen target is auditable.

## Routing Examples

| Request | Router behavior |
|---------|-----------------|
| `帮我把模型转成 OM` | Ask model family, ask explicit bundle path, probe platform, resolve target, select the internal workflow. |
| `把 /models/policy 转成 OM` | Read `/models/policy/config.json`, ask the user to confirm the inferred supported family, probe platform, select the internal workflow. |
| `把 ACT 转成 Ascend OM` | Ask explicit ACT bundle path, validate `type: act`, probe platform, select the ACT workflow. |
| `把这个 PI05 模型转 OM，路径是 /models/pi05` | Validate `type: pi05`, probe platform, then ask the PI05-specific PaliGemma asset question. |
| `把模型转 OM，目标是 Ascend310P3` | Still resolve family and explicit path; probe `npu-smi` and record that the target was user-selected. |

## Failure Boundaries

- A model-family answer is not a substitute for a bundle path.
- A directory name is not a substitute for `config.json.type`.
- A local `npu-smi` device is not a target choice on Ubuntu.
- A generic `310P` label is not a valid substitute for a full ATC revision.
- A successful ONNX export is not a successful OM deployment.
- Never report conversion success from shared resolution alone; only the selected internal workflow
  can establish it.
