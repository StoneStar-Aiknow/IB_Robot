import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from pr_review import CodeReviewer


def ai_pr(model: str = "gpt-5.6-sol") -> dict:
    return {
        "body": f"""### 当前PR是否有AI参与:
- [x] 是
Agent平台信息（Tool）: OpenCode 1.2.3
模型信息 (Model): {model}
Prompt摘要 (Prompt Summary): Apply the policy
人工审查情况: 已逐项审查
第三方材料及许可证: 无"""
    }


def ai_commit(model: str = "gpt-5.6-sol") -> dict:
    return {"commit": {"message": f"docs: update policy\n\nCo-Authored-By: {model}"}}


def test_ai_metadata_checks_accept_matching_disclosure():
    commit = ai_commit()
    commit["commit"]["message"] += "\nCo-Authored-By: Human Contributor <human@example.com>"

    assert CodeReviewer._build_ai_metadata_checks(ai_pr(), [commit]) == []


def test_ai_metadata_checks_reject_model_mismatch():
    checks = CodeReviewer._build_ai_metadata_checks(ai_pr(), [ai_commit("other-model")])

    assert [check["id"] for check in checks] == ["ai_model_metadata_mismatch"]


def test_ai_metadata_checks_require_complete_pr_disclosure():
    checks = CodeReviewer._build_ai_metadata_checks({"body": "- [x] 是\nModel: gpt-5.6-sol"}, [ai_commit()])

    assert "ai_disclosure_incomplete" in {check["id"] for check in checks}


def test_ai_metadata_checks_reject_provider_prefix():
    checks = CodeReviewer._build_ai_metadata_checks(ai_pr("xunxing/gpt-5.6-sol"), [ai_commit("xunxing/gpt-5.6-sol")])

    assert "ai_model_provider_prefix" in {check["id"] for check in checks}
