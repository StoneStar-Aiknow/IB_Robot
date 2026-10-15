"""RoboTwin provider version probe.

Fail-closed identity check for the pinned ``libs/robotwin`` submodule.
The probe verifies:

1. ``libs/robotwin`` exists and is a git repository.
2. The directory contains the expected ``envs/_base_task.py`` source tree.
3. The checked-out commit HEAD equals the gitlink SHA recorded for
   ``libs/robotwin`` in the parent repository, so a locally modified or
   re-pinned provider checkout cannot silently pass identity checks
   (mirroring the fail-closed LIBERO provider probe).

Unlike LIBERO, RoboTwin has no pip-installed distribution; the probe checks
the on-disk source tree only.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


class ProviderProbeError(RuntimeError):
    """Raised when the RoboTwin provider cannot be verified."""


@dataclass(frozen=True, slots=True)
class ProviderIdentity:
    on_disk_root: str
    commit_sha: str


def default_robotwin_root() -> Path:
    """Locate the ``libs/robotwin`` submodule relative to this package.

    Single shared implementation: walk up from this file until a directory
    containing ``libs/robotwin`` is found. Other modules (plan_provider)
    must import this helper instead of re-implementing the walk.
    """

    root = Path(__file__).resolve().parent
    while root.parent != root and not (root / "libs" / "robotwin").exists():
        root = root.parent
    result = root / "libs" / "robotwin"
    if not result.is_dir():
        raise ProviderProbeError(f"libs/robotwin not found at {result}")
    return result.resolve()


def _verify_structure(root: Path) -> None:
    required = (
        Path("envs") / "_base_task.py",
        Path("envs") / "camera" / "camera.py",
        Path("envs") / "robot" / "robot.py",
        Path("env_cfg") / "eval" / "all_tasks.yml",
        Path("env_cfg") / "task_config" / "_eval_step_limit.yml",
    )
    for rel in required:
        full = root / rel
        if not full.is_file():
            raise ProviderProbeError(f"RoboTwin source missing expected file: {full}")


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001 - identity probe treats git as optional input
        pass
    return None


def _parent_repo_root(robotwin_root: Path) -> Path | None:
    for parent in robotwin_root.resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def _verify_pinned_commit(robotwin_root: Path) -> str:
    head = _git(robotwin_root, "rev-parse", "HEAD")
    if head is None:
        raise ProviderProbeError(f"libs/robotwin at {robotwin_root} is not a git repository")

    parent = _parent_repo_root(robotwin_root)
    if parent is None:
        raise ProviderProbeError(f"libs/robotwin at {robotwin_root} is not inside a parent git repository")

    ls_tree = _git(parent, "ls-tree", "HEAD", "--", "libs/robotwin")
    if ls_tree is None:
        raise ProviderProbeError(f"parent repository {parent} does not track libs/robotwin")
    fields = ls_tree.split()
    if len(fields) < 3 or fields[0] != "160000":
        raise ProviderProbeError(f"parent repository {parent} does not pin libs/robotwin as a submodule gitlink")
    expected = fields[2]
    if head != expected:
        raise ProviderProbeError(
            f"libs/robotwin HEAD {head} does not match the pinned gitlink {expected}; "
            "restore the pinned checkout (git submodule update libs/robotwin)"
        )
    return head


def probe_robotwin_provider(robotwin_root: str | Path | None = None) -> ProviderIdentity:
    root = Path(robotwin_root) if robotwin_root is not None else default_robotwin_root()
    root = root.resolve()
    if not root.is_dir():
        raise ProviderProbeError(f"libs/robotwin not found at {root}")
    _verify_structure(root)
    commit_sha = _verify_pinned_commit(root)
    return ProviderIdentity(
        on_disk_root=str(root),
        commit_sha=commit_sha,
    )
