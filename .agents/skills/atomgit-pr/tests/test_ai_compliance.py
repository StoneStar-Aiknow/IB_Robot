import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from ai_compliance import DISCLOSURE_START, add_ai_disclosure, validate_commit_ai_model


def test_add_ai_disclosure_records_required_metadata():
    body = add_ai_disclosure(
        "## Changes\n\nFocused update.",
        agent_tool="OpenCode 1.2.3",
        ai_model="gpt-5.6-sol",
        prompt_summary="Apply the requested policy",
        third_party_materials="无",
    )

    assert body.count(DISCLOSURE_START) == 1
    assert "Agent平台信息（Tool）: OpenCode 1.2.3" in body
    assert "模型信息 (Model): gpt-5.6-sol" in body
    assert "人工审查情况" in body
    assert body.endswith("## Changes\n\nFocused update.")


def test_add_ai_disclosure_replaces_existing_block():
    first = add_ai_disclosure(
        "Description",
        agent_tool="OpenCode 1.0.0",
        ai_model="model-v1",
        prompt_summary="Initial prompt",
        third_party_materials="无",
    )
    updated = add_ai_disclosure(
        first,
        agent_tool="OpenCode 2.0.0",
        ai_model="model-v2",
        prompt_summary="Updated prompt",
        third_party_materials="无",
    )

    assert updated.count(DISCLOSURE_START) == 1
    assert "OpenCode 1.0.0" not in updated
    assert "模型信息 (Model): model-v2" in updated


def test_validate_commit_ai_model_accepts_matching_trailers():
    commits = [
        {"hash": "0" * 40, "subject": "docs: human authored context", "body": "No AI trailer."},
        {
            "hash": "a" * 40,
            "subject": "docs: update policy",
            "body": "Explain the policy.\n\nCo-Authored-By: gpt-5.6-sol\nSigned-off-by: User <user@example.com>",
        },
    ]

    validate_commit_ai_model(commits, "gpt-5.6-sol")


def test_validate_commit_ai_model_rejects_missing_or_mismatched_trailers():
    commits = [
        {"sha": "a" * 40, "commit": {"message": "docs: missing trailer"}},
        {"sha": "b" * 40, "commit": {"message": "docs: mismatch\n\nCo-Authored-By: other-model"}},
    ]

    with pytest.raises(ValueError, match="AI commit metadata check failed"):
        validate_commit_ai_model(commits, "gpt-5.6-sol")


def test_ai_model_rejects_provider_prefix():
    with pytest.raises(ValueError, match="must not include a provider prefix"):
        add_ai_disclosure(
            "Description",
            agent_tool="OpenCode 1.17.20",
            ai_model="xunxing/gpt-5.6-sol",
            prompt_summary="Apply the policy",
            third_party_materials="无",
        )
