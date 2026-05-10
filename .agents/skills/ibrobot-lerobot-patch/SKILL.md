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

It automates four repetitive steps:

- resolve the active lerobot tag via `INDEX.yaml`
- compute the next global patch number inside the active tag directory
- run `git format-patch` into `third_party/patches/lerobot/<tag>/`
- append the new patch filename(s) to the chosen `series*.txt`

Typical usage:

```bash
python3 .agents/skills/ibrobot-lerobot-patch/scripts/export_lerobot_patch.py \
  --rev-range HEAD~1..HEAD \
  --series default
```

Useful variants:

```bash
# Export to the master-parity candidate series.
python3 .agents/skills/ibrobot-lerobot-patch/scripts/export_lerobot_patch.py \
  --rev-range feature-base..HEAD \
  --series master-parity-candidates

# Preview numbering and target files without writing.
python3 .agents/skills/ibrobot-lerobot-patch/scripts/export_lerobot_patch.py \
  --rev-range HEAD~2..HEAD \
  --series default \
  --dry-run
```

After a real export, the script prints a ready-to-fill `manifest.yaml`
snippet for the new patch files.

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

## Mandatory Agent Workflow

### 1. Inspect The Current Stack

From `<project_root>`:

```bash
python3 scripts/setup/lerobot_resolve_active.py \
  --index third_party/patches/lerobot/INDEX.yaml
git status --short
git -C libs/lerobot status --short
git -C libs/lerobot branch --show-current
```

Read the resolved `LEROBOT_DIR`, `LEROBOT_BASE_COMMIT`, `LEROBOT_BRANCH_NAME`,
`manifest.yaml`, and the target `series*.txt` before exporting anything.

### 2. Normalize Local Work Into Commit(s)

- If `libs/lerobot` has uncommitted edits, commit them locally first.
- Prefer **one logical lerobot commit per final managed patch**.
- These submodule commits are intermediate authoring artifacts; the final
  IB_Robot submission is still the root-repo commit that adds patch files and
  metadata.

### 3. Rebuild A Canonical Authoring Branch

Do **not** export directly from whatever host-filtered branch happens to be
checked out locally. The default dev branch may only contain the subset of
patches that apply to the current machine, while `series.txt` may contain more.

Instead, rebuild an authoring branch from the resolved upstream base commit plus
the **raw target series file** you intend to extend.

Example for the default stack:

```bash
eval "$(python3 scripts/setup/lerobot_resolve_active.py \
  --index third_party/patches/lerobot/INDEX.yaml)"
TARGET_SERIES="${LEROBOT_DIR}/series.txt"
AUTHOR_BRANCH="${LEROBOT_BRANCH_NAME}-authoring-default"

git -C libs/lerobot checkout --detach "${LEROBOT_BASE_COMMIT}"
git -C libs/lerobot branch -D "${AUTHOR_BRANCH}" 2>/dev/null || true
git -C libs/lerobot checkout -b "${AUTHOR_BRANCH}"
while IFS= read -r patch; do
  git -C libs/lerobot am "${LEROBOT_DIR}/${patch}"
done < "${TARGET_SERIES}"
```

If the patch belongs to `master-parity-candidates` or OpenHarmony-only runtime
work, rebuild from that series file instead.

### 4. Replay The New Work On Top Of The Canonical Stack

- Cherry-pick the new local lerobot commit(s) onto the authoring branch.
- Resolve conflicts there, not in the managed patch files.
- If a patch must apply earlier than the current tail, rebase or renumber the
  affected tail explicitly; do not silently append a patch that depends on the
  wrong context.

### 5. Export Mailbox Patch Files

- Compute the next four-digit prefix from the highest existing patch number in
  the active tag directory.
- Run `git format-patch` from `libs/lerobot` into that tag directory.
- Export one patch per commit.

Example:

```bash
python3 .agents/skills/ibrobot-lerobot-patch/scripts/export_lerobot_patch.py \
  --rev-range HEAD~1..HEAD \
  --series default
```

For multiple new commits, export the whole range so numbering stays monotonic.

### 6. Register The New Patch In IB_Robot

After exporting the patch file:

1. Append the new filename to the correct `series*.txt`.
2. Add a matching `patches[]` entry in `manifest.yaml`.
3. Fill in `purpose` and the narrowest safe `applies_to` predicates.
4. If the new patch changes filter results, extend
   `scripts/setup/tests/test_lerobot_filter.sh` so pre-commit can catch scope
   drift.

Rules for `manifest.yaml` entries:

- Start with the narrowest proven `profiles` set.
- Add `python_min` / `python_max` only when the patch is version-gated.
- Keep the `patches[]` list in patch-number order.
- If the patch is not ready for the default stack, keep it out of `series.txt`
  and place it in the appropriate alternate series file instead.

### 7. Verify Before The Root Commit

Minimum verification:

```bash
scripts/setup/tests/test_lerobot_filter.sh
```

If the patch affects actual runtime or build behavior, also use the relevant
follow-up workflow:

- `ibrobot-build` for workspace build / import validation.
- `ibrobot-docker-verify` or `ibrobot-docker-verify-oee` after changing setup
  or platform-sensitive patch behavior.

### 8. Clean The Submodule Checkout Before Committing IB_Robot

Before the final root-repo commit, restore `libs/lerobot` to the superproject's
recorded gitlink unless you are intentionally bumping the upstream tag/base:

```bash
git submodule update --checkout libs/lerobot
```

Then stage the patch-management artifacts, not the temporary authoring checkout.

Typical root-repo staging set:

- `third_party/patches/lerobot/<tag>/*.patch`
- `third_party/patches/lerobot/<tag>/series*.txt`
- `third_party/patches/lerobot/<tag>/manifest.yaml`
- `scripts/setup/tests/test_lerobot_filter.sh`
- optional docs updates

Only stage `libs/lerobot` in the root repository when you are deliberately
changing the upstream base commit or moving to a new tag directory.

### 9. Final Commit Handling

When the user asks to create the final IB_Robot commit, hand off to
`ibrobot-git-flow` so the root commit message and sign-off stay compliant.

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
