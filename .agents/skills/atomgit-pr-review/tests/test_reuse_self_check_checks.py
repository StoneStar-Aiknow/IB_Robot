import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from pr_review import CodeReviewer
from reuse_gate import format_reuse_self_check


def _large_files():
    return [{"filename": "src/big/new_module.py", "additions": 22707, "deletions": 11, "patch": ""}]


def _small_files():
    return [{"filename": "src/small/fix.py", "additions": 40, "deletions": 9, "patch": ""}]


def _complete_block():
    return format_reuse_self_check(
        "无：未重新发明现有流程",
        "复用 lerobot.benchmarks 与 inference_service 入口",
        "无（未重新发明现有流程）",
        "对齐既有推理服务的 bundle + manifest 架构",
    )


def test_large_pr_without_block_is_blocking():
    checks = CodeReviewer._build_reuse_self_check_checks({"body": ""}, _large_files())

    assert [c["id"] for c in checks] == ["large_pr_reuse_self_check_missing"]
    assert checks[0]["severity"] == "error"
    assert checks[0]["blocking_until_reviewed"] is True


def test_large_pr_with_incomplete_block_is_blocking():
    body = _complete_block().replace("**Reused components:** 复用 lerobot.benchmarks 与 inference_service 入口\n", "")

    checks = CodeReviewer._build_reuse_self_check_checks({"body": body}, _large_files())

    assert [c["id"] for c in checks] == ["large_pr_reuse_self_check_incomplete"]
    assert "Reused components" in checks[0]["message"]


def test_large_pr_with_ambiguous_block_is_blocking():
    checks = CodeReviewer._build_reuse_self_check_checks(
        {"body": _complete_block() + _complete_block()}, _large_files()
    )

    assert [c["id"] for c in checks] == ["large_pr_reuse_self_check_invalid"]


def test_large_pr_with_complete_block_passes():
    checks = CodeReviewer._build_reuse_self_check_checks({"body": _complete_block()}, _large_files())

    assert checks == []


def test_wip_does_not_defer_reuse_self_check():
    pr = {"title": "[WIP] big feature", "body": ""}

    checks = CodeReviewer._build_reuse_self_check_checks(pr, _large_files())

    assert [c["id"] for c in checks] == ["large_pr_reuse_self_check_missing"]


def test_small_pr_is_not_gated():
    checks = CodeReviewer._build_reuse_self_check_checks({"body": ""}, _small_files())

    assert checks == []
