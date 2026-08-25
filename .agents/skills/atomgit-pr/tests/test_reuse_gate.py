import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from reuse_gate import (
    REUSE_SELF_CHECK_THRESHOLD,
    count_changed_lines,
    extract_reuse_self_check,
    format_reuse_self_check,
    missing_reuse_fields,
    reuse_gate_required,
    reuse_self_check_status,
    validate_reuse_self_check,
)


def _large_files():
    return [{"filename": "src/big/new_module.py", "additions": 1500, "deletions": 501, "patch": ""}]


def _small_files():
    return [{"filename": "src/small/fix.py", "additions": 30, "deletions": 12, "patch": ""}]


def _complete_block():
    return format_reuse_self_check(
        "无：未重新发明现有流程",
        "复用 lerobot.datasets 与 robot_config SSOT",
        "无（未重新发明现有流程）",
        "对齐 inference_service 的 bundle + manifest 架构",
    )


def test_threshold_uses_additions_plus_deletions():
    boundary = [{"filename": "a.py", "additions": 1000, "deletions": 1000, "patch": ""}]
    over = [{"filename": "a.py", "additions": 1001, "deletions": 1000, "patch": ""}]

    assert count_changed_lines(boundary) == REUSE_SELF_CHECK_THRESHOLD
    assert not reuse_gate_required(boundary)
    assert reuse_gate_required(over)


def test_count_changed_lines_falls_back_to_patch_lines():
    files = [
        {"filename": "a.py", "patch": "@@ -1,2 +1,3 @@\n context\n-old\n+new\n+extra"},
        {"filename": "b.py", "patch": {"diff": "@@ -1 +1 @@\n-x\n+y"}},
        {"filename": "binary.png", "patch": ""},
    ]

    assert count_changed_lines(files) == 5


def test_format_extract_roundtrip():
    fields = extract_reuse_self_check(_complete_block())

    assert fields is not None
    assert not missing_reuse_fields(fields)
    assert fields["Reinvented workflows"] == "无：未重新发明现有流程"
    assert fields["Architecture conformance"].startswith("对齐 inference_service")


def test_extract_scopes_fields_to_section():
    description = (
        "## Background\n\n**Reused components:** decoy outside the section\n\n"
        + _complete_block()
        + "\n## Verification\n\npassed\n"
    )

    fields = extract_reuse_self_check(description)

    assert fields is not None
    assert fields["Reused components"] == "复用 lerobot.datasets 与 robot_config SSOT"


def test_extract_rejects_duplicate_headings_and_fields():
    with pytest.raises(ValueError, match="exactly one"):
        extract_reuse_self_check(_complete_block() + "\n" + _complete_block())
    with pytest.raises(ValueError, match="multiple 'Reinvented workflows'"):
        extract_reuse_self_check(
            "## Reuse Self-Check\n\n**Reinvented workflows:** a\n**Reinvented workflows:** b\n"
            "**Reused components:** c\n**Reinvention justification:** d\n**Architecture conformance:** e\n"
        )


def test_status_classification():
    assert reuse_self_check_status("") == "absent"
    assert reuse_self_check_status("## Changes\n\nonly a body\n") == "absent"
    assert reuse_self_check_status("## Reuse Self-Check\n\n**Reinvented workflows:** only one\n") == "incomplete"
    assert reuse_self_check_status(_complete_block()) == "complete"
    assert reuse_self_check_status(_complete_block() + _complete_block()) == "invalid"


def test_validate_requires_block_only_for_large_prs():
    with pytest.raises(ValueError, match="more than 2000 lines"):
        validate_reuse_self_check("## Changes\n\nbody only\n", _large_files())

    validate_reuse_self_check("## Changes\n\nbody only\n", _small_files())
    validate_reuse_self_check(_complete_block(), _large_files())


def test_validate_rejects_incomplete_block():
    block = _complete_block().replace("**Reused components:** 复用 lerobot.datasets 与 robot_config SSOT\n", "")

    with pytest.raises(ValueError, match="missing concrete answers for: Reused components"):
        validate_reuse_self_check(block, _large_files())


def test_validate_rejects_placeholder_empty_answers():
    block = _complete_block().replace(
        "**Reinvention justification:** 无（未重新发明现有流程）",
        "**Reinvention justification:** ",
    )

    with pytest.raises(ValueError, match="Reinvention justification"):
        validate_reuse_self_check(block, _large_files())
