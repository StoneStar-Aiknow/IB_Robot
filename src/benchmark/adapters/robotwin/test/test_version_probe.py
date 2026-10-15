"""Unit tests for benchmark_robotwin.version_probe.

The pinned-commit identity check needs a real parent repository with a
gitlink, so each test builds a minimal two-repository fixture with git.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from benchmark_robotwin.version_probe import ProviderProbeError, probe_robotwin_provider


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _write_structure(root: Path) -> None:
    (root / "envs" / "camera").mkdir(parents=True)
    (root / "envs" / "robot").mkdir(parents=True)
    (root / "envs" / "_base_task.py").write_text("# stub\n")
    (root / "envs" / "camera" / "camera.py").write_text("# stub\n")
    (root / "envs" / "robot" / "robot.py").write_text("# stub\n")
    (root / "env_cfg" / "eval").mkdir(parents=True)
    (root / "env_cfg" / "eval" / "all_tasks.yml").write_text("tasks: []\n")
    (root / "env_cfg" / "task_config").mkdir(parents=True)
    (root / "env_cfg" / "task_config" / "_eval_step_limit.yml").write_text("{}\n")


@pytest.fixture()
def pinned_fixture(tmp_path: Path) -> Path:
    """Parent repo + child repo wired as a gitlink at libs/robotwin."""

    _write_structure(tmp_path / "libs" / "robotwin")
    child = tmp_path / "libs" / "robotwin"
    _git(child, "init", "-q")
    _git(child, "config", "user.email", "t@example.com")
    _git(child, "config", "user.name", "t")
    _git(child, "add", "-A")
    _git(child, "commit", "-qm", "robotwin stub")
    child_sha = _git(child, "rev-parse", "HEAD")

    parent = tmp_path
    _git(parent, "init", "-q")
    _git(parent, "config", "user.email", "t@example.com")
    _git(parent, "config", "user.name", "t")
    _git(parent, "commit", "-q", "--allow-empty", "-m", "init")
    _git(parent, "update-index", "--add", "--cacheinfo", f"160000,{child_sha},libs/robotwin")
    _git(parent, "commit", "-qm", "pin robotwin")
    return tmp_path


def test_probe_passes_on_pinned_commit(pinned_fixture: Path) -> None:
    identity = probe_robotwin_provider(pinned_fixture / "libs" / "robotwin")
    assert identity.on_disk_root == str((pinned_fixture / "libs" / "robotwin").resolve())
    assert identity.commit_sha == _git(pinned_fixture / "libs" / "robotwin", "rev-parse", "HEAD")


def test_probe_rejects_drifted_commit(pinned_fixture: Path) -> None:
    child = pinned_fixture / "libs" / "robotwin"
    (child / "envs" / "extra.txt").write_text("drift\n")
    _git(child, "add", "envs/extra.txt")
    _git(child, "commit", "-qm", "drift")
    with pytest.raises(ProviderProbeError, match="does not match the pinned gitlink"):
        probe_robotwin_provider(child)


def test_probe_rejects_missing_structure(tmp_path: Path) -> None:
    root = tmp_path / "bare"
    root.mkdir()
    with pytest.raises(ProviderProbeError, match="missing expected file"):
        probe_robotwin_provider(root)


def test_probe_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(ProviderProbeError, match="not found"):
        probe_robotwin_provider(tmp_path / "nope")
