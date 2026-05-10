#!/usr/bin/env python3
"""Export committed libs/lerobot work into the managed IB_Robot patch stack."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

SERIES_ALIASES = {
    "default": "series.txt",
    "master-parity-candidates": "series.master-parity-candidates.txt",
    "openharmony-5.1.0-musl": "series.openharmony-5.1.0-musl.txt",
}

PATCH_RE = re.compile(r"^(\d{4})-.*\.patch$")


def run(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "command failed")
    return result.stdout.strip()


def get_project_root() -> Path:
    try:
        return Path(run(["git", "rev-parse", "--show-toplevel"], Path.cwd()))
    except RuntimeError:
        return Path.cwd()


def resolve_active_lerobot(root: Path) -> dict[str, str]:
    output = run(
        [
            "python3",
            "scripts/setup/lerobot_resolve_active.py",
            "--index",
            "third_party/patches/lerobot/INDEX.yaml",
        ],
        root,
    )
    data: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    required = ["LEROBOT_TAG", "LEROBOT_DIR", "LEROBOT_BASE_COMMIT", "LEROBOT_BRANCH_NAME"]
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise RuntimeError(f"resolver output missing keys: {', '.join(missing)}")
    return data


def resolve_series_file(patch_dir: Path, selector: str) -> Path:
    filename = SERIES_ALIASES.get(selector, selector)
    path = patch_dir / filename
    if not path.is_file():
        raise RuntimeError(f"series file not found: {path}")
    return path


def list_existing_patch_numbers(patch_dir: Path) -> list[int]:
    numbers: list[int] = []
    for path in patch_dir.iterdir():
        match = PATCH_RE.match(path.name)
        if match:
            numbers.append(int(match.group(1)))
    return sorted(numbers)


def next_patch_number(patch_dir: Path) -> int:
    numbers = list_existing_patch_numbers(patch_dir)
    return (numbers[-1] + 1) if numbers else 1


def count_commits(rev_range: str, lerobot_dir: Path) -> int:
    return int(run(["git", "rev-list", "--count", rev_range], lerobot_dir))


def commit_subjects(rev_range: str, lerobot_dir: Path) -> list[str]:
    output = run(["git", "log", "--reverse", "--format=%s", rev_range], lerobot_dir)
    return [line for line in output.splitlines() if line.strip()]


def is_dirty(lerobot_dir: Path) -> bool:
    return bool(run(["git", "status", "--short"], lerobot_dir))


def export_patches(rev_range: str, start_number: int, patch_dir: Path, lerobot_dir: Path) -> list[str]:
    output = run(
        [
            "git",
            "format-patch",
            rev_range,
            "--start-number",
            str(start_number),
            "--output-directory",
            str(patch_dir),
        ],
        lerobot_dir,
    )
    created: list[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        created.append(Path(line).name)
    if not created:
        raise RuntimeError("git format-patch did not report any created patch files")
    return created


def append_to_series(series_file: Path, patch_names: list[str]) -> None:
    existing = [line.strip() for line in series_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    duplicates = [name for name in patch_names if name in existing]
    if duplicates:
        raise RuntimeError(f"series file already contains: {', '.join(duplicates)}")

    updated = existing + patch_names
    series_file.write_text("\n".join(updated) + "\n", encoding="utf-8")


def render_manifest_template(patch_names: list[str]) -> str:
    lines = [
        "Add the following snippet under manifest.yaml patches[]:",
        "",
    ]
    for name in patch_names:
        lines.extend(
            [
                f"  - file: {name}",
                "    purpose:",
                "      - TODO: describe why this patch exists.",
                "    applies_to:",
                "      profiles: [TODO]",
                "",
            ]
        )
    lines.extend(
        [
            "Notes:",
            "- Keep patches[] sorted by patch number.",
            "- Add python_min/python_max only if the patch is version-gated.",
            "- Start with the narrowest safe profiles set.",
        ]
    )
    return "\n".join(lines)


def print_plan(
    *,
    tag: str,
    patch_dir: Path,
    series_file: Path,
    rev_range: str,
    start_number: int,
    commit_count: int,
    subjects: list[str],
) -> None:
    print("LeRobot patch export plan")
    print(f"  active tag   : {tag}")
    print(f"  patch dir    : {patch_dir}")
    print(f"  target series: {series_file.name}")
    print(f"  rev-range    : {rev_range}")
    print(f"  commits      : {commit_count}")
    print(f"  next number  : {start_number:04d}")
    if commit_count > 0:
        end_number = start_number + commit_count - 1
        print(f"  number range : {start_number:04d}-{end_number:04d}")
    if subjects:
        print("  subjects     :")
        for subject in subjects:
            print(f"    - {subject}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export libs/lerobot commits into third_party/patches/lerobot/<tag>/*.patch"
    )
    parser.add_argument(
        "--rev-range",
        required=True,
        help="Git revision range inside libs/lerobot, e.g. HEAD~1..HEAD",
    )
    parser.add_argument(
        "--series",
        default="default",
        help=(
            "Target series selector: default, master-parity-candidates, "
            "openharmony-5.1.0-musl, or a literal series filename"
        ),
    )
    parser.add_argument(
        "--no-append-series",
        action="store_true",
        help="Export patch files only; do not append filenames to the series file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the export plan without writing patch files or editing series",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = get_project_root()
    os.chdir(root)

    lerobot_dir = root / "libs/lerobot"
    if not lerobot_dir.is_dir():
        print(f"ERROR: missing lerobot submodule directory: {lerobot_dir}", file=sys.stderr)
        return 1

    try:
        active = resolve_active_lerobot(root)
        patch_dir = Path(active["LEROBOT_DIR"])
        series_file = resolve_series_file(patch_dir, args.series)
        commit_count = count_commits(args.rev_range, lerobot_dir)
        subjects = commit_subjects(args.rev_range, lerobot_dir)
        start_number = next_patch_number(patch_dir)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if commit_count <= 0:
        print(f"ERROR: rev-range produced no commits: {args.rev_range}", file=sys.stderr)
        return 1

    print_plan(
        tag=active["LEROBOT_TAG"],
        patch_dir=patch_dir,
        series_file=series_file,
        rev_range=args.rev_range,
        start_number=start_number,
        commit_count=commit_count,
        subjects=subjects,
    )

    if is_dirty(lerobot_dir):
        print(
            "WARNING: libs/lerobot has uncommitted changes; only committed revisions in --rev-range will be exported.",
            file=sys.stderr,
        )

    if args.dry_run:
        print("\nDry run only. No files were written.")
        print("Run without --dry-run to export patches and print the manifest snippet.")
        return 0

    try:
        created = export_patches(args.rev_range, start_number, patch_dir, lerobot_dir)
        if not args.no_append_series:
            append_to_series(series_file, created)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("\nCreated patch files:")
    for name in created:
        print(f"  - {name}")

    if args.no_append_series:
        print(f"\nSeries file left unchanged: {series_file}")
    else:
        print(f"\nUpdated series file: {series_file}")

    print()
    print(render_manifest_template(created))
    print()
    print("Next steps:")
    print("- Fill in manifest.yaml purpose/applies_to for the new patch(es).")
    print("- Update scripts/setup/tests/test_lerobot_filter.sh if scope changes.")
    print(
        "- Restore libs/lerobot to the recorded gitlink before the final root commit unless you are bumping the base tag."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
