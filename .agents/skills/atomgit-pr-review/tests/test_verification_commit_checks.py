import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from pr_review import CodeReviewer


def gated_files() -> list[dict]:
    return [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@"}]


def test_verification_tree_check_accepts_current_pr_tree():
    head = "a" * 40
    tree = "c" * 40
    pr = {"head": {"sha": head}, "body": f"**Verified tree:** `{tree}`"}

    assert CodeReviewer._build_verification_tree_checks(pr, gated_files(), tree) == []


def test_verification_tree_check_blocks_missing_or_stale_sha():
    head = "a" * 40
    tree = "c" * 40
    missing = CodeReviewer._build_verification_tree_checks({"head": {"sha": head}, "body": ""}, gated_files(), tree)
    stale = CodeReviewer._build_verification_tree_checks(
        {"head": {"sha": head}, "body": f"**Verified tree:** `{'b' * 40}`"}, gated_files(), tree
    )

    assert [check["id"] for check in missing] == ["docker_verification_tree_missing"]
    assert [check["id"] for check in stale] == ["docker_verification_tree_mismatch"]


def test_wip_pr_skips_dual_docker_evidence_check():
    pr = {"title": "[WIP] agents: update workflow", "body": ""}

    assert CodeReviewer._build_verification_tree_checks(pr, gated_files(), None) == []


def test_verification_tree_survives_commit_message_rewrite():
    tree = "c" * 40
    rewritten_head = "b" * 40
    pr = {"head": {"sha": rewritten_head}, "body": f"**Verified tree:** `{tree}`"}

    assert CodeReviewer._build_verification_tree_checks(pr, gated_files(), tree) == []


def test_verification_commit_check_skips_non_gated_pr():
    files = [{"filename": "docs/README.md", "patch": "+text"}]

    assert CodeReviewer._build_verification_tree_checks({"head": {"sha": "a" * 40}, "body": ""}, files, None) == []
