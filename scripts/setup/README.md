# `scripts/setup/` — workspace bootstrap modules

This directory hosts the modular pieces sourced by `scripts/setup.sh`:

| File | Responsibility |
| --- | --- |
| `detect.sh` | OS / arch / Python detection; resolves `IBR_HOST_*` and `IBR_LEROBOT_PROFILES`. |
| `platforms/<id>.sh` | Per-platform package managers, ROS source, and `platform_lerobot_profiles()` defaults. |
| `lerobot_patches.sh` | Drives `git am` of the curated lerobot patch stack into `libs/lerobot`. |
| `lerobot_filter_series.py` | Reads `manifest.yaml` + host facts, prints the patch series that actually applies. |
| `tests/test_lerobot_filter.sh` | Regression fixtures pinned to the canonical 3-platform matrix. |

## LeRobot patch dispatch

`libs/lerobot` ships a single patch series under `third_party/patches/lerobot/v0.5.1/`,
but **not every patch applies to every host.** The applier filters the raw
`series.txt` through `lerobot_filter_series.py`, which gates each patch on
independent predicates declared in `manifest.yaml`:

- `python_min` / `python_max` — bounded interval (`>=3.10`, `<3.11`, ...).
- `profiles` — set intersection against the active profile set.

The active profile set is resolved with this precedence (highest first):

1. `--lerobot-profiles core,ascend,...` CLI flag on `setup.sh`.
2. `IBR_LEROBOT_PROFILES=core,ascend,...` environment variable.
3. `platform_lerobot_profiles` callback in the active platform script.
4. The fallback `core,ros,hardware,dev`.

### Common overrides for hardware bring-up

```bash
# Force the Ascend NPU patches on a desktop verifying inference parity:
./scripts/setup.sh --yes --lerobot-profiles core,ros,hardware,ascend

# Bring up an OpenHarmony device as if it were a vanilla core target:
./scripts/setup.sh --yes --lerobot-profiles core

# Explicitly bypass the filter (last-resort escape hatch — applies the raw
# series.txt verbatim, even patches that do not match host facts):
IBR_LEROBOT_FORCE_UNFILTERED=1 ./scripts/setup.sh --yes
```

### Diagnosing the filter

The applier prints a `[CTX] python=X.Y profiles=...` audit line and a
`KEEP` / `SKIP` decision for every patch. To preview without touching
`libs/lerobot`:

```bash
IBR_HOST_PYTHON_VERSION=3.11 \
IBR_LEROBOT_PROFILES=core,ros,hardware,openeuler \
  python3 scripts/setup/lerobot_filter_series.py \
    --manifest third_party/patches/lerobot/v0.5.1/manifest.yaml \
    --series   third_party/patches/lerobot/v0.5.1/series.txt
```

### When to add a new patch

1. Add the patch file under `third_party/patches/lerobot/v0.5.1/`.
2. Append it to `series.txt`.
3. Register a `patches[]` entry in `manifest.yaml` declaring **explicit**
   `python_min` / `python_max` / `profiles` predicates. Default to the
   narrowest set that is known to be safe; broaden later once verified on
   additional platforms.
4. Add a fixture line to `scripts/setup/tests/test_lerobot_filter.sh` so the
   pre-commit hook catches accidental scope changes.

### Failure modes

| Symptom | Cause | Fix |
| --- | --- | --- |
| `lerobot_filter_series.py failed with exit 1`, "PyYAML required but missing" | venv lacks `python3-yaml` | `pip install pyyaml` in the workspace venv, or set `IBR_LEROBOT_FORCE_UNFILTERED=1` to skip filtering. |
| `failed to parse manifest`, exit 1 | `manifest.yaml` is malformed | Lint the YAML; the applier fails closed by design. |
| `libs/lerobot is not at the expected upstream base commit` | submodule diverged | `git -C libs/lerobot fetch && git -C libs/lerobot checkout <base>`, then re-run setup. |
| Worktree dirty, applier aborts | local edits in `libs/lerobot` | Stash / commit your edits, or set `IBR_LEROBOT_FORCE_REBUILD=1` to discard them and rebuild the patched branch from scratch. |
