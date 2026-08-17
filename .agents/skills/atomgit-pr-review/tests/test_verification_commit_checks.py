import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from pr_review import CodeReviewer


def gated_files() -> list[dict]:
    return [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@"}]


def test_verification_commit_check_accepts_current_pr_head():
    head = "a" * 40
    pr = {"head": {"sha": head}, "body": f"**Verified commit:** `{head}`"}

    assert CodeReviewer._build_verification_commit_checks(pr, gated_files()) == []


def test_verification_commit_check_blocks_missing_or_stale_sha():
    head = "a" * 40
    missing = CodeReviewer._build_verification_commit_checks({"head": {"sha": head}, "body": ""}, gated_files())
    stale = CodeReviewer._build_verification_commit_checks(
        {"head": {"sha": head}, "body": f"**Verified commit:** `{'b' * 40}`"}, gated_files()
    )

    assert [check["id"] for check in missing] == ["docker_verification_commit_missing"]
    assert [check["id"] for check in stale] == ["docker_verification_commit_mismatch"]


def test_verification_commit_check_skips_non_gated_pr():
    files = [{"filename": "docs/README.md", "patch": "+text"}]

    assert CodeReviewer._build_verification_commit_checks({"head": {"sha": "a" * 40}, "body": ""}, files) == []
