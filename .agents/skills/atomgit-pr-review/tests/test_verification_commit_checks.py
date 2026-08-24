import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from pr_review import CodeReviewer
from verification_gate import format_verification_metadata


def gated_files() -> list[dict]:
    return [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-old\n+new"}]


def _inputs_sha() -> str:
    from verification_gate import compute_verification_inputs

    return compute_verification_inputs(gated_files())


def _env() -> str:
    return "ubuntu:ros-humble-desktop-full-jammy|openeuler:ibrobot-dev-env|policy:1"


def _full_block(tree: str, inputs: str) -> str:
    return format_verification_metadata("full", inputs, tree, _env())


def _reused_block(old_tree: str, inputs: str) -> str:
    return format_verification_metadata("reused-environment", inputs, old_tree, _env())


def test_full_verification_matches_current_tree():
    tree = "c" * 40
    inputs = _inputs_sha()
    pr = {"head": {"sha": "a" * 40}, "body": _full_block(tree, inputs)}

    assert CodeReviewer._build_verification_tree_checks(pr, gated_files(), tree) == []


def test_missing_verification_block_blocks():
    tree = "c" * 40
    pr = {"head": {"sha": "a" * 40}, "body": ""}

    checks = CodeReviewer._build_verification_tree_checks(pr, gated_files(), tree)
    assert [c["id"] for c in checks] == ["docker_verification_missing"]


def test_mismatched_tree_blocks():
    tree = "c" * 40
    inputs = _inputs_sha()
    pr = {"head": {"sha": "a" * 40}, "body": _full_block("d" * 40, inputs)}

    checks = CodeReviewer._build_verification_tree_checks(pr, gated_files(), tree)
    assert [c["id"] for c in checks] == ["docker_verification_mismatch"]


def test_reused_environment_accepts_old_tree_when_inputs_match():
    old_tree = "c" * 40
    new_tree = "d" * 40
    inputs = _inputs_sha()
    pr = {"head": {"sha": "b" * 40}, "body": _reused_block(old_tree, inputs)}

    assert CodeReviewer._build_verification_tree_checks(pr, gated_files(), new_tree) == []


def test_reused_environment_rejects_changed_inputs():
    old_tree = "c" * 40
    new_tree = "d" * 40
    inputs = "0" * 40
    pr = {"head": {"sha": "b" * 40}, "body": _reused_block(old_tree, inputs)}

    checks = CodeReviewer._build_verification_tree_checks(pr, gated_files(), new_tree)
    assert [c["id"] for c in checks] == ["docker_verification_mismatch"]


def test_wip_pr_skips_dual_docker_evidence_check():
    pr = {"title": "[WIP] agents: update workflow", "body": ""}

    assert CodeReviewer._build_verification_tree_checks(pr, gated_files(), None) == []


def test_non_gated_pr_skips_verification_check():
    files = [{"filename": "docs/README.md", "patch": "+text"}]

    assert CodeReviewer._build_verification_tree_checks({"head": {"sha": "a" * 40}, "body": ""}, files, None) == []
