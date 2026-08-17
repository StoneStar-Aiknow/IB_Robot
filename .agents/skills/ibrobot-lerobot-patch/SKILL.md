---
name: ibrobot-lerobot-patch
description: "Manage the LeRobot patch stack after editing `libs/lerobot`. Use when user needs to 'export lerobot patch', 'make patch from libs/lerobot', 'update series.txt', 'register manifest', '导出 lerobot 补丁', '生成 patch', '更新 third_party/patches/lerobot', 'patch stack', '修改 libs/lerobot 后纳管', or '把 lerobot 改动做成 patch 提交回 IB_Robot'. Triggers for local `libs/lerobot` edits that must land as managed `third_party/patches/lerobot/<tag>/*.patch` files instead of a raw submodule pointer bump."
---

# IB-Robot LeRobot Patch Management Skill

This skill is the canonical workflow for turning local `libs/lerobot` changes
into managed patch files under `third_party/patches/lerobot/<tag>/`.

## Helper Script

This skill ships a helper script at:

`<project_root>/.agents/skills/ibrobot-lerobot-patch/scripts/export_lerobot_patch.py`

It automates four repetitive steps: resolve the active lerobot tag, compute the
next global patch number, run `git format-patch`, and append the new patch
filename(s) to the chosen `series*.txt`.

Typical usage:

```bash
python3 .agents/skills/ibrobot-lerobot-patch/scripts/export_lerobot_patch.py \
  --rev-range HEAD~1..HEAD \
  --series default
```

For the full set of variants (`master-parity-candidates`, `--dry-run`) and the
printed `manifest.yaml` snippet, see `references/helper-script.md`.

## Core Contract

- `third_party/patches/lerobot/INDEX.yaml` is the single source of truth for the
  active upstream lerobot tag.
- The real deliverable is usually:
  - one or more new mailbox patches in `third_party/patches/lerobot/<tag>/`
  - an updated `series*.txt`
  - an updated `manifest.yaml`
  - updated filter fixtures in `scripts/setup/tests/test_lerobot_filter.sh`
- The root repository should usually **not** commit a new `libs/lerobot` gitlink
  just because local lerobot code changed.
- `scripts/setup/lerobot_patches.sh` applies patches with `git am`, so exported
  patches **must** be mailbox patches produced by `git format-patch`, not plain
  `git diff` output.

## Choose The Right Series

- `series.txt`: default maintained stack used by normal setup flows.
- `series.master-parity-candidates.txt`: feature-scoped migration patches that
  are intentionally kept outside the default stack until validated.
- `series.openharmony-5.1.0-musl.txt`: OpenHarmony board-specific runtime stack.

Patch numbering is global inside one tag directory. If the highest file in
`third_party/patches/lerobot/v0.5.1/` is `0014-*.patch`, the next new patch is
`0015-*`, even if it only belongs to a non-default series file.

## Internal References

Read only the references needed for the current step:

| Purpose | Reference |
|---------|-----------|
| Full helper-script variants, `--dry-run`, printed `manifest.yaml` snippet | `references/helper-script.md` |
| Detailed bash commands for each of the 9 workflow steps | `references/workflow-details.md` |

Do not expose these references as separate skills.

## Mandatory Agent Workflow

### 1. Inspect The Current Stack

Resolve the active tag and read `manifest.yaml`, target `series*.txt`, and
submodule status before exporting anything.

### 2. Normalize Local Work Into Commit(s)

Commit any uncommitted `libs/lerobot` edits first; prefer one logical lerobot
commit per final managed patch.

### 3. Rebuild A Canonical Authoring Branch

Rebuild an authoring branch from the resolved upstream base commit plus the raw
target series file instead of exporting from a host-filtered branch.

### 4. Replay The New Work On Top Of The Canonical Stack

Cherry-pick the new local lerobot commit(s) onto the authoring branch and
resolve conflicts there, not in the managed patch files.

### 5. Export Mailbox Patch Files

Compute the next four-digit prefix and run `git format-patch` (one patch per
commit) into the active tag directory.

### 6. Register The New Patch In IB_Robot

Append the filename to the correct `series*.txt`, add a `patches[]` entry in
`manifest.yaml`, and extend `test_lerobot_filter.sh` if filter results changed.

### 7. Verify Before The Root Commit

Run `test_lerobot_filter.sh` at minimum; add build / docker verification if the
patch affects runtime or build behavior.

### 8. Clean The Submodule Checkout Before Committing IB_Robot

Restore `libs/lerobot` to the recorded gitlink and stage only the
patch-management artifacts (not the temporary authoring checkout).

### 9. Final Commit Handling

Hand off to `ibrobot-git-flow` so the root commit message and sign-off stay
compliant.

For the exact bash commands behind each step, see
`references/workflow-details.md`.

## Common Mistakes To Prevent

- Exporting from a host-filtered patched branch instead of rebuilding from the
  raw target series.
- Saving `git diff` output as `*.patch` and expecting `git am` to apply it.
- Reusing a patch number that already exists in the same tag directory.
- Broadening `profiles` too early in `manifest.yaml`.
- Forgetting to update `series*.txt` or `test_lerobot_filter.sh`.
- Accidentally staging `libs/lerobot` in the root repo when only patch files
  should be committed.

## When To Use This Skill

Invoke this skill when:

- `libs/lerobot` was modified and the result must be absorbed into
  `third_party/patches/lerobot/`
- A new lerobot compatibility / feature patch needs to be exported
- A patch should move between `series.txt` and an alternate series file
- The user asks to turn local lerobot work into an IB_Robot-managed patch

Do NOT invoke this skill for:

- plain workspace builds without lerobot patch changes
- generic root-repo commits that do not touch `libs/lerobot`
- AtomGit PR / Issue workflows
