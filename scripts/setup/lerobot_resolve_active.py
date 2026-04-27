#!/usr/bin/env python3
"""Resolve the active lerobot patch tag from INDEX.yaml.

Invoked by ``scripts/setup/lerobot_patches.sh`` to translate the current
multi-tag layout under ``third_party/patches/lerobot/`` into a small set
of shell-eval'able assignments. Keeping the resolution logic in Python
(instead of grep/awk against YAML) lets us cross-validate INDEX.yaml
against the per-tag manifest in one pass and fail closed on mismatch.

CLI contract::

    lerobot_resolve_active.py --index <path/to/INDEX.yaml>

On success exits 0 and prints (one assignment per line) to stdout:

    LEROBOT_TAG=<tag>
    LEROBOT_DIR=<absolute-path-to-tag-dir>
    LEROBOT_BASE_COMMIT=<min-of-commit-range>
    LEROBOT_BASE_COMMIT_MIN=<min-of-commit-range>
    LEROBOT_BASE_COMMIT_MAX=<max-of-commit-range>
    LEROBOT_BRANCH_NAME=<branch-from-INDEX>
    LEROBOT_MANIFEST=<absolute-path-to-manifest.yaml>
    LEROBOT_SERIES=<absolute-path-to-series.txt>
    LEROBOT_UPSTREAM_REPO=<manifest-upstream-repo>

The shell consumer is expected to parse the ``KEY=value`` lines explicitly.

Exit codes:
    0  — Resolution succeeded.
    1  — INDEX/manifest unparsable, missing required fields, archived tag
         selected, INDEX vs manifest mismatch, or PyYAML missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _err(msg: str) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)


def _load_yaml(path: Path):
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via shell
        _err(
            f"PyYAML required but missing ({exc}). Install python3-yaml in the "
            "workspace venv or set IBR_LEROBOT_FORCE_UNFILTERED=1 to bypass "
            "filtering (note: bypass does NOT skip tag resolution)."
        )
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        _err(f"failed to parse {path}: {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    args = parser.parse_args(argv)

    index = _load_yaml(args.index)
    if index is None:
        return 1
    if not isinstance(index, dict):
        _err(f"INDEX root is not a mapping: {args.index}")
        return 1

    active_tag = index.get("active_tag")
    if not active_tag:
        _err(f"INDEX is missing required 'active_tag': {args.index}")
        return 1

    archived = index.get("archived_tags") or []
    archived_tags = {entry.get("tag") for entry in archived if isinstance(entry, dict)}
    if active_tag in archived_tags:
        _err(
            f"active_tag={active_tag!r} is listed under archived_tags. Promote it "
            "back to supported_tags or pick a different tag."
        )
        return 1

    supported = index.get("supported_tags") or []
    entry = next(
        (e for e in supported if isinstance(e, dict) and e.get("tag") == active_tag),
        None,
    )
    if entry is None:
        _err(
            f"active_tag={active_tag!r} not found in supported_tags. "
            f"Add an entry under {args.index} or fix the active_tag value."
        )
        return 1

    rel_dir = entry.get("dir")
    if not rel_dir:
        _err(f"supported_tags entry for {active_tag!r} is missing 'dir'")
        return 1
    upstream_commit = entry.get("upstream_commit")
    if not upstream_commit:
        _err(f"supported_tags entry for {active_tag!r} is missing 'upstream_commit'")
        return 1
    branch_name = entry.get("branch_name")
    if not branch_name:
        _err(f"supported_tags entry for {active_tag!r} is missing 'branch_name'")
        return 1

    base_dir = args.index.parent
    tag_dir = (base_dir / rel_dir).resolve()
    if not tag_dir.is_dir():
        _err(f"INDEX points active_tag={active_tag!r} at {tag_dir} but the directory does not exist.")
        return 1

    manifest_path = tag_dir / "manifest.yaml"
    series_path = tag_dir / "series.txt"
    if not manifest_path.is_file():
        _err(f"manifest.yaml missing for active_tag={active_tag!r}: {manifest_path}")
        return 1
    if not series_path.is_file():
        _err(f"series.txt missing for active_tag={active_tag!r}: {series_path}")
        return 1

    manifest = _load_yaml(manifest_path)
    if manifest is None:
        return 1
    if not isinstance(manifest, dict):
        _err(f"manifest root is not a mapping: {manifest_path}")
        return 1

    manifest_tag = manifest.get("lerobot_tag")
    if manifest_tag != active_tag:
        _err(
            f"INDEX active_tag={active_tag!r} does not match "
            f"manifest.lerobot_tag={manifest_tag!r} ({manifest_path}). "
            "Update one to match the other."
        )
        return 1

    commit_range = manifest.get("lerobot_commit_range") or {}
    if not isinstance(commit_range, dict):
        _err(f"manifest.lerobot_commit_range must be a mapping: {manifest_path}")
        return 1
    range_min = commit_range.get("min")
    range_max = commit_range.get("max")
    if not range_min or not range_max:
        _err(f"manifest.lerobot_commit_range must declare both 'min' and 'max' sha values: {manifest_path}")
        return 1
    if upstream_commit != range_min:
        _err(
            f"INDEX upstream_commit={upstream_commit!r} for tag {active_tag!r} "
            f"does not match manifest.lerobot_commit_range.min={range_min!r} "
            f"({manifest_path}). The two MUST agree."
        )
        return 1
    upstream = manifest.get("upstream") or {}
    if not isinstance(upstream, dict):
        _err(f"manifest.upstream must be a mapping when present: {manifest_path}")
        return 1
    upstream_repo = upstream.get("repo") or ""

    sys.stdout.write(f"LEROBOT_TAG={active_tag}\n")
    sys.stdout.write(f"LEROBOT_DIR={tag_dir}\n")
    sys.stdout.write(f"LEROBOT_BASE_COMMIT={range_min}\n")
    sys.stdout.write(f"LEROBOT_BASE_COMMIT_MIN={range_min}\n")
    sys.stdout.write(f"LEROBOT_BASE_COMMIT_MAX={range_max}\n")
    sys.stdout.write(f"LEROBOT_BRANCH_NAME={branch_name}\n")
    sys.stdout.write(f"LEROBOT_MANIFEST={manifest_path}\n")
    sys.stdout.write(f"LEROBOT_SERIES={series_path}\n")
    sys.stdout.write(f"LEROBOT_UPSTREAM_REPO={upstream_repo}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
