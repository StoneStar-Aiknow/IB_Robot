"""Where a measurement came from, and the one file format every tool writes.

Design §13 asks for a reproducible commit, dependency and asset digests and
the run matrix with every result. It also keeps private connection details and
the deployment mapping out of the repository, so a report records digests and
the commit, never a host name, an address or an absolute path.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "visual_interaction.phase0/1"

# Enough to see what made a tree dirty without dumping a whole status.
_MAX_DIRTY_PATHS = 20
_HASH_CHUNK = 1 << 20


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def digests(paths: Mapping[str, str | Path]) -> dict[str, str]:
    """``{label: sha256}`` for each file, keyed by a label rather than its path."""
    return {label: sha256_file(path) for label, path in sorted(paths.items())}


def _git(repo: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return completed.stdout


def source_state(repo: str | Path) -> dict[str, object]:
    """The commit a measurement ran on, and whether that commit is the whole story.

    A dirty tree is reported, not refused: an exploratory run on uncommitted
    code is still useful to its author. It is marked ``reproducible: false`` so
    nobody cites it as phase-zero evidence, which must come from a commit.
    """
    root = Path(repo)
    head = _git(root, "rev-parse", "HEAD")
    if head is None:
        return {"commit": None, "reproducible": False, "detail": "not a git checkout"}
    status = _git(root, "status", "--porcelain", "--untracked-files=no") or ""
    dirty = [line[3:] for line in status.splitlines() if line.strip()]
    return {
        "commit": head.strip(),
        "dirty_paths": dirty[:_MAX_DIRTY_PATHS],
        "dirty_count": len(dirty),
        "reproducible": not dirty,
    }


def environment() -> dict[str, str]:
    """Interpreter and platform; deliberately no host name."""
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "system": platform.system(),
        "machine": platform.machine(),
        "executable_prefix": Path(sys.prefix).name,
    }


def build_report(
    *,
    tool: str,
    conditions: Mapping[str, object],
    results: Mapping[str, object],
    repo: str | Path,
    assets: Mapping[str, str] | None = None,
    notes: Iterable[str] = (),
) -> dict[str, object]:
    return {
        "schema": SCHEMA,
        "tool": tool,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": source_state(repo),
        "environment": environment(),
        "assets": dict(assets or {}),
        "conditions": dict(conditions),
        "results": dict(results),
        "notes": list(notes),
    }


def write_report(path: str | Path, report: Mapping[str, object]) -> Path:
    """Write the report, refusing to overwrite: a measurement is not edited."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"{target} exists; a measurement is written once, pick a new path")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


__all__ = [
    "SCHEMA",
    "build_report",
    "digests",
    "environment",
    "sha256_file",
    "source_state",
    "write_report",
]
