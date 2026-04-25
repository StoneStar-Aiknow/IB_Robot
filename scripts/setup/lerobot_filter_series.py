#!/usr/bin/env python3
"""Filter the lerobot patch series against host facts and active profiles.

This helper is invoked by ``scripts/setup/lerobot_patches.sh`` from inside the
workspace virtualenv (which carries PyYAML). It reads:

* ``--manifest``: path to ``third_party/patches/lerobot/v0.5.1/manifest.yaml``.
* ``--series``: path to the series file consumed by ``git am`` (e.g.
  ``series.txt`` for the default profile).
* Environment variables ``IBR_HOST_PYTHON_VERSION`` and
  ``IBR_LEROBOT_PROFILES`` describing the active host facts.

It prints the filtered patch filenames to stdout (one per line, in the order
they appear in the series file) and an audit log to stderr listing every patch
considered together with the predicate that excluded it (if any).

Exit codes:
    0  — Filtering succeeded.
    1  — Manifest could not be parsed, PyYAML missing, or referenced patch is
         missing from the manifest.

The CLI surface is intentionally minimal so the same helper can be exercised
from the regression harness (``scripts/setup/tests/test_lerobot_filter.sh``).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_PYTHON_SPECIFIER_RE = re.compile(r"^\s*(?P<op><=|>=|<|>|==|!=|~=)\s*(?P<ver>\d+(?:\.\d+){0,2})\s*$")


def _log_stderr(message: str) -> None:
    """Emit an audit log line to stderr."""
    print(message, file=sys.stderr)


def _split_csv(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def _parse_version(text: str) -> tuple[int, ...]:
    """Parse a dotted numeric version like ``"3.11"`` into a tuple."""
    return tuple(int(part) for part in text.strip().split(".") if part)


def _python_satisfies(host: tuple[int, ...], specifier: str) -> bool:
    """Check whether ``host`` satisfies the PEP 440-style ``specifier``.

    Only the operators ``<=``, ``>=``, ``<``, ``>``, ``==`` and ``!=`` are
    supported; that covers the vocabulary used by the lerobot manifest.
    """
    match = _PYTHON_SPECIFIER_RE.match(specifier)
    if not match:
        raise ValueError(f"unrecognised python specifier: {specifier!r}")
    op = match.group("op")
    bound = _parse_version(match.group("ver"))
    if op == "<":
        return host < bound
    if op == "<=":
        return host <= bound
    if op == ">":
        return host > bound
    if op == ">=":
        return host >= bound
    if op == "==":
        return host == bound
    if op == "!=":
        return host != bound
    if op == "~=":
        # Compatible release: same prefix, no trailing components allowed
        # to drop below the bound. Approximated as >= bound and <
        # bound[:-1] + (bound[-1] + 1,).
        if not bound:
            return False
        upper = bound[:-2] + (bound[-2] + 1,) if len(bound) >= 2 else (bound[0] + 1,)
        return host >= bound and host < upper
    raise ValueError(f"unsupported python operator: {op}")


def _evaluate_patch(
    *,
    patch_file: str,
    applies_to: Mapping[str, object],
    host_python: tuple[int, ...] | None,
    active_profiles: set[str],
) -> str | None:
    """Return ``None`` when the patch matches; otherwise an exclusion reason."""

    python_min = applies_to.get("python_min")
    python_max = applies_to.get("python_max")
    profile_list = applies_to.get("profiles")

    if python_min:
        if host_python is None:
            return f"host python unknown but python_min={python_min!s}"
        if not _python_satisfies(host_python, str(python_min)):
            host_str = ".".join(str(part) for part in host_python)
            return f"host python {host_str} does not satisfy python_min {python_min!s}"

    if python_max:
        if host_python is None:
            return f"host python unknown but python_max={python_max!s}"
        if not _python_satisfies(host_python, str(python_max)):
            host_str = ".".join(str(part) for part in host_python)
            return f"host python {host_str} does not satisfy python_max {python_max!s}"

    if profile_list:
        if not active_profiles:
            return f"no active profiles but patch requires one of {list(profile_list)}"
        if not active_profiles.intersection(profile_list):
            return f"active profiles {sorted(active_profiles)} have no overlap with patch profiles {list(profile_list)}"

    return None


def _read_series(path: Path) -> list[str]:
    out: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _build_index(patches: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    return {str(entry["file"]): entry for entry in patches if "file" in entry}


def _resolve_host_python(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    try:
        return _parse_version(value)
    except ValueError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--series", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        import yaml  # intentional in-function import for fail-closed behaviour
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via shell
        _log_stderr(
            f"ERROR: PyYAML is required but missing ({exc}). "
            "Install python3-yaml in the workspace venv or set "
            "IBR_LEROBOT_FORCE_UNFILTERED=1 to bypass filtering."
        )
        return 1

    try:
        manifest = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        _log_stderr(f"ERROR: failed to parse manifest {args.manifest}: {exc}")
        return 1

    if not isinstance(manifest, dict) or "patches" not in manifest:
        _log_stderr(f"ERROR: manifest {args.manifest} missing required 'patches' key")
        return 1

    patches_index = _build_index(manifest.get("patches") or [])

    host_python = _resolve_host_python(os.environ.get("IBR_HOST_PYTHON_VERSION"))
    profiles_raw = os.environ.get("IBR_LEROBOT_PROFILES") or ""
    active_profiles = set(_split_csv(profiles_raw))

    _log_stderr(
        f"lerobot_filter: host_python={host_python or 'unknown'} active_profiles={sorted(active_profiles) or '[]'}"
    )

    series = _read_series(args.series)
    kept: list[str] = []
    for patch_file in series:
        entry = patches_index.get(patch_file)
        if entry is None:
            _log_stderr(f"ERROR: series entry {patch_file!r} not found in manifest patches list")
            return 1
        applies_to = entry.get("applies_to") or {}
        if not isinstance(applies_to, dict):
            _log_stderr(f"ERROR: patch {patch_file!r} has malformed applies_to (expected mapping)")
            return 1
        try:
            reason = _evaluate_patch(
                patch_file=patch_file,
                applies_to=applies_to,
                host_python=host_python,
                active_profiles=active_profiles,
            )
        except ValueError as exc:
            _log_stderr(f"ERROR: predicate evaluation failed for {patch_file}: {exc}")
            return 1
        if reason is None:
            _log_stderr(f"  KEEP   {patch_file}")
            kept.append(patch_file)
        else:
            _log_stderr(f"  SKIP   {patch_file} ({reason})")

    _log_stderr(f"lerobot_filter: kept {len(kept)}/{len(series)} patches")
    for patch_file in kept:
        sys.stdout.write(patch_file + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
