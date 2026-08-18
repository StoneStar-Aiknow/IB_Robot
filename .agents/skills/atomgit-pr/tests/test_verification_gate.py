import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import verification_gate
from verification_gate import (
    extract_verified_tree,
    file_triggers_dual_docker_gate,
    is_wip_title,
    normalize_pr_title,
    resolve_pr_head_tree,
    resolve_pr_stage,
    validate_verified_tree,
)


def test_gate_detects_global_and_package_dependency_changes():
    assert file_triggers_dual_docker_gate("scripts/setup.sh")
    assert file_triggers_dual_docker_gate("requirements/ubuntu.txt")
    assert file_triggers_dual_docker_gate("src/demo/package.xml", "+  <exec_depend>demo</exec_depend>")
    assert not file_triggers_dual_docker_gate("src/demo/package.xml", "+  <description>Updated</description>")
    assert file_triggers_dual_docker_gate("src/demo/package.xml")
    assert not file_triggers_dual_docker_gate("src/demo/setup.py")


def test_pr_creation_requires_description_file():
    result = subprocess.run(
        [sys.executable, str(Path(__file__).parents[1] / "scripts" / "pr_creation.py"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "--description-file" in result.stdout
    assert "--body" not in result.stdout


def test_wip_title_normalization_is_explicit_and_reversible():
    assert normalize_pr_title("agents: update workflow", "wip") == "[WIP] agents: update workflow"
    assert normalize_pr_title("[wip] [WIP] agents: update workflow", "review") == "agents: update workflow"
    assert is_wip_title("[WIP] agents: update workflow")
    assert is_wip_title(" [wip] agents: update workflow")
    assert not is_wip_title("agents: update [WIP] workflow")

    with pytest.raises(ValueError, match="must not be empty"):
        normalize_pr_title("[WIP]", "review")


def test_gate_requires_explicit_stage_and_wip_only_defers_docker():
    with pytest.raises(ValueError, match="explicit PR stage"):
        resolve_pr_stage("agents: update workflow", None, True)

    wip_title, wip_gate = resolve_pr_stage("agents: update workflow", "wip", True)
    review_title, review_gate = resolve_pr_stage(wip_title, "review", True)

    assert wip_title == "[WIP] agents: update workflow"
    assert not wip_gate
    assert review_title == "agents: update workflow"
    assert review_gate


def test_verified_tree_requires_one_full_matching_sha():
    tree = "a" * 40
    body = f"## Verification\n\n**Verified tree:** `{tree}`"

    assert extract_verified_tree(body) == tree
    assert validate_verified_tree(body, tree.upper()) == tree

    with pytest.raises(ValueError, match="exactly one"):
        validate_verified_tree("**Verified tree:** `abc123`", tree)
    with pytest.raises(ValueError, match="does not match"):
        validate_verified_tree(f"**Verified tree:** `{'b' * 40}`", tree)
    with pytest.raises(ValueError, match="exactly one"):
        validate_verified_tree(f"**Verified tree:** `{tree}`\n**Verified tree:** `{tree}`", tree)


def test_resolve_pr_head_tree_fetches_source_branch(monkeypatch):
    head = "b" * 40
    tree = "c" * 40
    calls = []

    def fake_run_git(args, cwd):
        calls.append(args)
        if args == ["rev-parse", "FETCH_HEAD"]:
            return head
        if args == ["rev-parse", "FETCH_HEAD^{tree}"]:
            return tree
        return ""

    monkeypatch.setattr(verification_gate, "_run_git", fake_run_git)
    pr = {
        "head": {
            "sha": head,
            "ref": "feature/example",
            "repo": {"html_url": "https://atomgit.com/example/IB_Robot.git"},
        }
    }

    assert resolve_pr_head_tree(pr) == tree
    assert [
        "fetch",
        "--no-tags",
        "--depth=1",
        pr["head"]["repo"]["html_url"],
        "refs/heads/feature/example",
    ] in calls


def test_resolve_pr_head_tree_rejects_racing_head(monkeypatch):
    monkeypatch.setattr(
        verification_gate,
        "_run_git",
        lambda args, cwd: "b" * 40 if args == ["rev-parse", "FETCH_HEAD"] else "",
    )
    pr = {
        "head": {
            "sha": "a" * 40,
            "ref": "feature/example",
            "repo": {"html_url": "https://atomgit.com/example/IB_Robot.git"},
        }
    }

    with pytest.raises(ValueError, match="head changed"):
        resolve_pr_head_tree(pr)


def test_resolve_pr_head_tree_rejects_non_https_repo():
    pr = {
        "head": {
            "sha": "a" * 40,
            "ref": "feature/example",
            "repo": {"html_url": "file:///tmp/IB_Robot"},
        }
    }

    with pytest.raises(ValueError, match="HTTPS repository URL"):
        resolve_pr_head_tree(pr)
