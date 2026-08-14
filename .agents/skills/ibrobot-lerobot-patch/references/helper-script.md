# Helper Script Usage

## When to Read

- You are about to run `export_lerobot_patch.py` and need the exact invocation.
- You want to export to a non-default series (`master-parity-candidates`).
- You want to preview numbering and target files before writing (`--dry-run`).
- You need to interpret the `manifest.yaml` snippet the script prints.

## Location

`<project_root>/.agents/skills/ibrobot-lerobot-patch/scripts/export_lerobot_patch.py`

The script automates four repetitive steps:

- resolve the active lerobot tag via `INDEX.yaml`
- compute the next global patch number inside the active tag directory
- run `git format-patch` into `third_party/patches/lerobot/<tag>/`
- append the new patch filename(s) to the chosen `series*.txt`

## Variants

### Default series export

```bash
python3 .agents/skills/ibrobot-lerobot-patch/scripts/export_lerobot_patch.py \
  --rev-range HEAD~1..HEAD \
  --series default
```

### Export to the master-parity candidate series

```bash
python3 .agents/skills/ibrobot-lerobot-patch/scripts/export_lerobot_patch.py \
  --rev-range feature-base..HEAD \
  --series master-parity-candidates
```

### Preview numbering and target files without writing

```bash
python3 .agents/skills/ibrobot-lerobot-patch/scripts/export_lerobot_patch.py \
  --rev-range HEAD~2..HEAD \
  --series default \
  --dry-run
```

## Printed `manifest.yaml` Snippet

After a real export, the script prints a ready-to-fill `manifest.yaml` snippet for
the new patch files. Fill in `purpose` and the narrowest safe `applies_to`
predicates before registering the patch in IB_Robot (see
`references/workflow-details.md` Step 6).
