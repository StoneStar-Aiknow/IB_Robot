# Mandatory Agent Workflow Details

## When to Read

- You are executing a specific step of the 9-step workflow and need the exact bash commands.
- You are rebuilding the canonical authoring branch (Step 3).
- You are exporting mailbox patches (Step 5) or registering them (Step 6).
- You need the verify / clean / commit handoff commands (Steps 7-9).

## Step 1. Inspect The Current Stack

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

## Step 3. Rebuild A Canonical Authoring Branch

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

## Step 5. Export Mailbox Patch Files

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

## Step 6. Register The New Patch In IB_Robot

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

## Step 7. Verify Before The Root Commit

Minimum verification:

```bash
scripts/setup/tests/test_lerobot_filter.sh
```

If the patch affects actual runtime or build behavior, also use the relevant
follow-up workflow:

- `ibrobot-build` for workspace build / import validation.
- `ibrobot-docker-verify` or `ibrobot-docker-verify-oee` after changing setup
  or platform-sensitive patch behavior.

## Step 8. Clean The Submodule Checkout Before Committing IB_Robot

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

## Step 9. Final Commit Handling

When the user asks to create the final IB_Robot commit, hand off to
`ibrobot-git-flow` so the root commit message and sign-off stay compliant.
