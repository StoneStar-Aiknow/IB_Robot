import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import verification_gate
from verification_gate import (
    compute_legacy_verification_inputs,
    compute_verification_inputs,
    detect_verification_policy,
    file_triggers_dual_docker_gate,
    format_verification_metadata,
    is_wip_title,
    normalize_pr_title,
    prepare_update_verification,
    resolve_pr_head_tree,
    resolve_pr_stage,
    upsert_verification_metadata,
    validate_verification_metadata,
    verification_inputs_placeholder_tainted,
)

_ENV = "ubuntu:ros-humble-desktop-full-jammy|openeuler:ibrobot-dev-env|policy:1"


def _gated_files():
    return [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-old\n+new"}]


def _inputs_sha():
    return compute_verification_inputs(_gated_files())


def _full_block(tree, inputs):
    return format_verification_metadata("full", inputs, tree, _ENV)


def _reused_block(old_tree, inputs):
    return format_verification_metadata("reused-environment", inputs, old_tree, _ENV)


def test_verification_upsert_is_idempotent_and_preserves_results():
    tree, inputs = "a" * 40, _inputs_sha()
    description = (
        f"## Summary\n\nChange.\n\n{_full_block(tree, inputs)}\n\n### Ubuntu\n\nPassed.\n\n## Risks\n\nNone.\n"
    )
    updated = upsert_verification_metadata(description, "full", inputs, tree, _ENV)
    assert updated == description
    assert upsert_verification_metadata(updated, "full", inputs, tree, _ENV) == updated
    assert updated.count("## Docker Verification") == 1
    validate_verification_metadata(updated, inputs, tree)


def test_verification_upsert_reuses_heading_without_metadata():
    tree, inputs = "a" * 40, _inputs_sha()
    description = "## Docker Verification\n\n### Ubuntu\n\nPassed.\n"
    updated = upsert_verification_metadata(description, "full", inputs, tree, _ENV)
    assert updated == _full_block(tree, inputs) + "\n\n### Ubuntu\n\nPassed.\n"


def test_verification_rejects_duplicate_headings_with_only_one_field_set():
    tree, inputs = "a" * 40, _inputs_sha()
    description = "## Docker Verification\n\nPlatform results.\n\n" + _full_block(tree, inputs)
    with pytest.raises(ValueError, match="exactly one.*heading"):
        validate_verification_metadata(description, inputs, tree)
    with pytest.raises(ValueError, match="exactly one.*heading"):
        upsert_verification_metadata(description, "full", inputs, tree, _ENV)


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


def test_compute_verification_inputs_is_deterministic():
    sha1 = compute_verification_inputs(_gated_files())
    sha2 = compute_verification_inputs(_gated_files())
    assert sha1 == sha2
    assert len(sha1) == 40


def test_compute_verification_inputs_changes_with_patch():
    files_a = [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-old\n+new"}]
    files_b = [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-old\n+other"}]
    assert compute_verification_inputs(files_a) != compute_verification_inputs(files_b)


def test_compute_verification_inputs_ignores_non_gated_files():
    assert compute_verification_inputs([{"filename": "src/foo.py", "patch": "+x"}]) is None


def test_full_verification_metadata_validates():
    tree = "a" * 40
    inputs = _inputs_sha()
    body = _full_block(tree, inputs)
    metadata = validate_verification_metadata(body, inputs, tree)
    assert metadata["mode"] == "full"
    assert metadata["tested_tree"] == tree


def test_full_verification_rejects_mismatched_tree():
    inputs = _inputs_sha()
    body = _full_block("b" * 40, inputs)
    with pytest.raises(ValueError, match="does not match current tree"):
        validate_verification_metadata(body, inputs, "a" * 40)


def test_reused_environment_validates_with_prior_evidence():
    old_tree = "a" * 40
    new_tree = "b" * 40
    inputs = _inputs_sha()
    body = _reused_block(old_tree, inputs)
    metadata = validate_verification_metadata(body, inputs, new_tree, allow_reuse=True)
    assert metadata["mode"] == "reused-environment"
    assert metadata["tested_tree"] == old_tree


def test_reused_environment_rejects_without_prior_evidence():
    inputs = _inputs_sha()
    body = _reused_block("a" * 40, inputs)
    with pytest.raises(ValueError, match="only valid when prior evidence"):
        validate_verification_metadata(body, inputs, "b" * 40)


def test_prepare_update_carries_forward_reused_environment():
    old_tree = "a" * 40
    new_tree = "b" * 40
    inputs = _inputs_sha()
    previous = _reused_block(old_tree, inputs)
    description = "## Changes\n\nFix a typo.\n"
    updated, metadata = prepare_update_verification(description, previous, inputs, new_tree)
    assert metadata["mode"] == "reused-environment"
    assert metadata["tested_tree"] == old_tree
    assert "reused-environment" in updated


def test_prepare_update_requires_full_when_inputs_change():
    old_tree = "a" * 40
    new_tree = "b" * 40
    inputs_old = _inputs_sha()
    inputs_new = compute_verification_inputs(
        [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-old\n+brand-new"}]
    )
    previous = _full_block(old_tree, inputs_old)
    description = "## Changes\n\nNo Docker block yet.\n"
    with pytest.raises(ValueError, match="full Docker verification is required"):
        prepare_update_verification(description, previous, inputs_new, new_tree)


def test_prepare_update_accepts_draft_reused_block():
    old_tree = "a" * 40
    new_tree = "b" * 40
    inputs = _inputs_sha()
    previous = _reused_block(old_tree, inputs)
    draft = _reused_block(old_tree, inputs)
    updated, metadata = prepare_update_verification(draft, previous, inputs, new_tree)
    assert metadata["mode"] == "reused-environment"
    assert metadata["tested_tree"] == old_tree


def test_prepare_update_accepts_new_full_block():
    tree = "a" * 40
    inputs = _inputs_sha()
    body = _full_block(tree, inputs)
    previous = _full_block(tree, inputs)
    updated, metadata = prepare_update_verification(body, previous, inputs, tree)
    assert metadata["mode"] == "full"


def test_upsert_replaces_legacy_verified_tree():
    tree = "a" * 40
    inputs = _inputs_sha()
    legacy = f"## Verification\n\n**Verified tree:** `{tree}`\n"
    updated = upsert_verification_metadata(legacy, "full", inputs, tree, _ENV)
    assert "**Verified tree:**" not in updated
    assert "**Verified inputs:**" in updated


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


def test_extract_accepts_list_prefix_fields():
    """Regression: Agents naturally write Markdown bullet lists; the gate must
    accept '- **label:**' and upsert must strip it before appending canonical
    lines, otherwise the description ends up with duplicate blocks."""
    from verification_gate import extract_verification_metadata

    inputs = _inputs_sha()
    tree = "b" * 40
    description = (
        "## Docker Verification\n\n"
        f"- **Docker verification mode:** `full`\n"
        f"- **Verified inputs:** `{inputs}`\n"
        f"- **Tested source tree:** `{tree}`\n"
        f"- **Docker environment:** `ubuntu|openeuler`"
    )

    metadata = extract_verification_metadata(description)
    assert metadata == {
        "mode": "full",
        "verified_inputs": inputs,
        "tested_tree": tree,
        "environment": "ubuntu|openeuler",
    }

    # Upsert must recognise the list-format lines as the same block (not new
    # fields), so no duplicates remain afterwards.
    updated = upsert_verification_metadata(description, "full", inputs, tree, "ubuntu|openeuler")
    import re

    for label in (
        verification_gate.VERIFICATION_MODE_LABEL,
        verification_gate.VERIFIED_INPUTS_LABEL,
        verification_gate.TESTED_SOURCE_TREE_LABEL,
        verification_gate.VERIFICATION_ENVIRONMENT_LABEL,
    ):
        pattern = r"\*\*" + re.escape(label) + r":\*\*"
        total = len(re.findall(pattern, updated))
        list_form = len(re.findall(r"- " + pattern, updated))
        assert total == 1, f"{label} appears {total} times after upsert"
        assert list_form == 0, f"{label} still uses list format after upsert"


def test_mismatch_errors_report_expected_values():
    """Regression: mismatch errors used to hide the expected fingerprint/tree,
    forcing the Agent to monkey-patch internal helpers to discover them."""
    from verification_gate import validate_verification_metadata

    inputs = _inputs_sha()
    good_tree = "b" * 40
    bad_tree = "c" * 40
    block = format_verification_metadata("full", inputs, bad_tree, _ENV)

    with pytest.raises(ValueError, match=good_tree):
        validate_verification_metadata(block, inputs, good_tree)

    other_inputs = compute_verification_inputs(
        [{"filename": "scripts/setup.sh", "patch": "@@ -1 +1 @@\n-different\n+patch"}]
    )
    wrong_inputs_block = format_verification_metadata("full", other_inputs, good_tree, _ENV)
    with pytest.raises(ValueError, match=other_inputs):
        validate_verification_metadata(wrong_inputs_block, inputs, good_tree)


def test_duplicate_field_error_lists_actual_counts():
    from verification_gate import extract_verification_metadata

    inputs = _inputs_sha()
    duplicated = (
        format_verification_metadata("full", inputs, "b" * 40, _ENV) + "\n\n**Docker verification mode:** `full`\n"
    )

    with pytest.raises(ValueError) as excinfo:
        extract_verification_metadata(duplicated)
    message = str(excinfo.value)
    # Only the offending fields are listed, with their actual counts.
    assert "Docker verification mode=2" in message
    assert "Verified inputs" not in message


# --------------------------------------------------------------------------- #
# Policy-versioned fingerprints (task 3.2, reviews/round-3.md R3-01 option (b),
# design.md D3): complete v1 reusable, complete v2, placeholder v1, unknown
# hash and missing data are fixed fixtures.
# --------------------------------------------------------------------------- #


def test_detect_verification_policy_matches_complete_v1_and_v2():
    files = _gated_files()
    v2 = compute_verification_inputs(files)
    v1 = compute_legacy_verification_inputs(files)
    assert v1 and v2 and v1 != v2  # same records; the policy field is the only difference
    assert detect_verification_policy(v1, v2, v1) == "1"
    assert detect_verification_policy(v2, v2, v1) == "2"


def test_unrelated_file_changes_do_not_affect_either_fingerprint():
    """Task 3.2 review: v2 must stay scoped to dependency-relevant records.

    An unrelated doc change (or a rename touching no gate-relevant file)
    must not invalidate recorded evidence — the reverted file-list binding
    broke exactly this reuse semantic."""
    gated = _gated_files()
    with_unrelated = gated + [
        {"filename": "docs/notes.md", "patch": "@@ -1 +1 @@\n-old\n+docs change"},
        {"filename": "src/other.py", "patch": "@@ -1 +1 @@\n-old\n+unrelated"},
    ]
    assert compute_verification_inputs(with_unrelated) == compute_verification_inputs(gated)
    assert compute_legacy_verification_inputs(with_unrelated) == compute_legacy_verification_inputs(gated)


def test_truncated_requirements_patch_taints_inputs_and_keeps_gate():
    """Task 3.2 review: non-empty patches are never treated as complete.

    The hunk below declares 3 old/new lines but carries only one body line —
    a truncated patch must enter the fingerprint as a placeholder (gate stays
    triggered) and never match a hash recorded over complete inputs."""
    truncated = [{"filename": "requirements/ubuntu.txt", "patch": "@@ -1,3 +1,3 @@\n-old\n+new"}]
    assert verification_inputs_placeholder_tainted(truncated) is True
    truncated_v2 = compute_verification_inputs(truncated)
    truncated_v1 = compute_legacy_verification_inputs(truncated)
    assert truncated_v2 is not None and truncated_v1 is not None  # gate stays triggered
    assert detect_verification_policy(_inputs_sha(), truncated_v2, truncated_v1) is None


def test_flagged_truncation_taints_even_with_complete_hunks():
    """An entry that declares itself truncated taints even when the visible
    hunk structure looks complete — the platform says content was dropped."""
    flagged = [{"filename": "requirements/ubuntu.txt", "patch_truncated": True, "patch": "@@ -1 +1 @@\n-old\n+new"}]
    assert verification_inputs_placeholder_tainted(flagged) is True
    assert compute_verification_inputs(flagged) is not None


def test_truncated_package_xml_cannot_bypass_gate():
    """Task 3.2 review: a truncated package.xml patch whose visible part has
    no dependency lines must NOT skip the gate — the truncation is exactly
    where the depend lines could hide."""
    truncated = [{"filename": "package.xml", "patch": "@@ -1,5 +1,5 @@\n+  <description>Update</description>"}]
    assert verification_inputs_placeholder_tainted(truncated) is True
    # The placeholder record keeps the gate triggered instead of returning None.
    assert compute_verification_inputs(truncated) is not None
    assert compute_legacy_verification_inputs(truncated) is not None


def test_detect_verification_policy_rejects_unknown_hash_and_missing_data():
    files = _gated_files()
    v2 = compute_verification_inputs(files)
    v1 = compute_legacy_verification_inputs(files)
    # An unknown recorded hash matches no supported policy.
    assert detect_verification_policy("e" * 40, v2, v1) is None
    # A missing recorded hash is not evidence at all.
    assert detect_verification_policy(None, v2, v1) is None
    # Missing recomputation inputs: neither policy can be recomputed.
    assert detect_verification_policy(_inputs_sha(), None, None) is None
    # An ambiguous match (both policies recomputing to the same hash) is refused.
    ambiguous = _inputs_sha()
    assert detect_verification_policy(ambiguous, ambiguous, ambiguous) is None


def test_validate_recognizes_complete_v1_evidence_and_keeps_original_checks():
    files = _gated_files()
    v2 = compute_verification_inputs(files)
    v1 = compute_legacy_verification_inputs(files)
    tree = "a" * 40
    body = _full_block(tree, v1)  # a legacy four-field record carrying a v1 hash

    metadata = validate_verification_metadata(body, v2, tree, legacy_inputs=v1)
    assert metadata["detected_policy"] == "1"

    # Even a matched v1 record still passes through the original tree check.
    with pytest.raises(ValueError, match="does not match current tree"):
        validate_verification_metadata(_full_block("b" * 40, v1), v2, tree, legacy_inputs=v1)
    # Without the legacy recomputation the same v1 record requires renewed verification.
    with pytest.raises(ValueError, match="do not match the current inputs"):
        validate_verification_metadata(body, v2, tree)


def test_prepare_update_reuses_complete_v1_evidence_without_rewriting_hash():
    files = _gated_files()
    v2 = compute_verification_inputs(files)
    v1 = compute_legacy_verification_inputs(files)
    tree = "a" * 40
    previous = _full_block(tree, v1)  # v1-era evidence on the same inputs
    description = "## Changes\n\nFix a typo.\n"

    updated, metadata = prepare_update_verification(description, previous, v2, tree, legacy_inputs=v1)

    assert metadata["detected_policy"] == "1"
    # The carried-forward block keeps the recorded v1 hash — never rewritten as v2.
    assert v1 in updated
    assert v2 not in updated


def test_placeholder_tainted_inputs_keep_gate_but_never_support_reuse():
    placeholder_files = [{"filename": "scripts/setup.sh", "patch": ""}]  # missing patch
    assert verification_inputs_placeholder_tainted(placeholder_files) is True
    # A placeholder record keeps the gate triggered (conservative)...
    placeholder_v2 = compute_verification_inputs(placeholder_files)
    placeholder_v1 = compute_legacy_verification_inputs(placeholder_files)
    assert placeholder_v2 is not None and placeholder_v1 is not None
    # ...but a hash recorded over complete inputs never matches the placeholder
    # recomputation, so placeholder-derived evidence cannot be reused.
    complete_hash = _inputs_sha()
    assert detect_verification_policy(complete_hash, placeholder_v2, placeholder_v1) is None


def test_full_verification_without_legacy_recomputation_rejects_v1_hash():
    """Without complete inputs to recompute v1, an old-style record cannot be
    distinguished from an unknown hash and requires renewed verification."""
    files = _gated_files()
    v1 = compute_legacy_verification_inputs(files)
    tree = "a" * 40
    body = _full_block(tree, v1)

    with pytest.raises(ValueError, match="do not match the current inputs"):
        validate_verification_metadata(body, _inputs_sha(), tree, legacy_inputs=None)
