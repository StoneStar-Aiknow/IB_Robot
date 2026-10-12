import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import pr_gates
from pr_gates import (
    collect_diff_file_infos,
    compute_verification_inputs,
    file_triggers_dual_docker_gate,
    format_verification_metadata,
    is_wip_title,
    normalize_pr_title,
    prepare_update_verification,
    resolve_pr_stage,
    upsert_verification_metadata,
    validate_verification_metadata,
    verification_inputs_placeholder_tainted,
)

_ENV = "ubuntu:ros-humble-desktop-full-jammy|openeuler:ibrobot-dev-env"


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


def test_upsert_replaces_stale_verified_tree_field():
    tree = "a" * 40
    inputs = _inputs_sha()
    stale = f"## Verification\n\n**Verified tree:** `{tree}`\n"
    updated = upsert_verification_metadata(stale, "full", inputs, tree, _ENV)
    assert "**Verified tree:**" not in updated
    assert "**Verified inputs:**" in updated


def test_extract_accepts_list_prefix_fields():
    """Regression: Agents naturally write Markdown bullet lists; the gate must
    accept '- **label:**' and upsert must strip it before appending canonical
    lines, otherwise the description ends up with duplicate blocks."""
    from pr_gates import extract_verification_metadata

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
        pr_gates.VERIFICATION_MODE_LABEL,
        pr_gates.VERIFIED_INPUTS_LABEL,
        pr_gates.TESTED_SOURCE_TREE_LABEL,
        pr_gates.VERIFICATION_ENVIRONMENT_LABEL,
    ):
        pattern = r"\*\*" + re.escape(label) + r":\*\*"
        total = len(re.findall(pattern, updated))
        list_form = len(re.findall(r"- " + pattern, updated))
        assert total == 1, f"{label} appears {total} times after upsert"
        assert list_form == 0, f"{label} still uses list format after upsert"


def test_mismatch_errors_report_expected_values():
    """Regression: mismatch errors used to hide the expected fingerprint/tree,
    forcing the Agent to monkey-patch internal helpers to discover them."""
    from pr_gates import validate_verification_metadata

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
    from pr_gates import extract_verification_metadata

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
# Fingerprint policy: the single current v2 calculator is frozen, evidence
# recorded under the retired v1 policy no longer verifies, and incomplete
# patches never support reuse.
# --------------------------------------------------------------------------- #


def test_verification_inputs_hash_is_frozen():
    """The v2 calculator must stay byte-stable: changing record collection,
    serialization or hashing would invalidate every recorded Docker
    evidence hash (values cross-checked against the pre-cleanup baseline)."""
    assert compute_verification_inputs(_gated_files()) == "7394246cb24fc1ab3e91e961eda5cf61588d6619"
    assert (
        compute_verification_inputs([{"filename": "requirements/ubuntu.txt", "patch": "@@ -1 +1 @@\n-old\n+new"}])
        == "068e80ffcd1ef1aa9a8843d3c83a154d22201f7b"
    )
    assert compute_verification_inputs([{"filename": "requirements/ubuntu.txt", "patch": ""}]) == (
        "e184793d3fa66fa633a8ec99105ef8d63a6ab91e"
    )


def test_unrelated_file_changes_do_not_affect_the_fingerprint():
    """v2 must stay scoped to dependency-relevant records: an unrelated doc
    change must not invalidate recorded evidence."""
    gated = _gated_files()
    with_unrelated = gated + [
        {"filename": "docs/notes.md", "patch": "@@ -1 +1 @@\n-old\n+docs change"},
        {"filename": "src/other.py", "patch": "@@ -1 +1 @@\n-old\n+unrelated"},
    ]
    assert compute_verification_inputs(with_unrelated) == compute_verification_inputs(gated)


def test_truncated_requirements_patch_taints_inputs_and_keeps_gate():
    """Non-empty patches are never treated as complete.

    The hunk below declares 3 old/new lines but carries only one body line —
    a truncated patch must enter the fingerprint as a placeholder (gate stays
    triggered) and never match a hash recorded over complete inputs."""
    truncated = [{"filename": "requirements/ubuntu.txt", "patch": "@@ -1,3 +1,3 @@\n-old\n+new"}]
    assert verification_inputs_placeholder_tainted(truncated) is True
    truncated_v2 = compute_verification_inputs(truncated)
    assert truncated_v2 is not None  # gate stays triggered
    assert truncated_v2 != _inputs_sha()  # never matches a hash over complete inputs


def test_flagged_truncation_taints_even_with_complete_hunks():
    """An entry that declares itself truncated taints even when the visible
    hunk structure looks complete — the platform says content was dropped."""
    flagged = [{"filename": "requirements/ubuntu.txt", "patch_truncated": True, "patch": "@@ -1 +1 @@\n-old\n+new"}]
    assert verification_inputs_placeholder_tainted(flagged) is True
    assert compute_verification_inputs(flagged) is not None


def test_truncated_package_xml_cannot_bypass_gate():
    """A truncated package.xml patch whose visible part has no dependency
    lines must NOT skip the gate — the truncation is exactly where the
    depend lines could hide."""
    truncated = [{"filename": "package.xml", "patch": "@@ -1,5 +1,5 @@\n+  <description>Update</description>"}]
    assert verification_inputs_placeholder_tainted(truncated) is True
    # The placeholder record keeps the gate triggered instead of returning None.
    assert compute_verification_inputs(truncated) is not None


def _v1_style_hash(files: list[dict]) -> str:
    """Hash under the retired v1 policy, constructed inline for the regression
    tests only — production code no longer computes or accepts it."""
    records = pr_gates._collect_verification_inputs(files)
    payload = json.dumps({"policy": "1", "records": sorted(set(records))}, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(payload.encode()).hexdigest()


def test_v1_hash_evidence_requires_renewed_verification():
    """Evidence recorded under the retired v1 policy no longer verifies: the
    gate reports a mismatch and demands renewed verification instead of
    silently relabelling or reusing it."""
    files = _gated_files()
    v2 = compute_verification_inputs(files)
    v1 = _v1_style_hash(files)
    tree = "a" * 40
    assert v1 != v2

    with pytest.raises(ValueError, match="do not match the current inputs"):
        validate_verification_metadata(_full_block(tree, v1), v2, tree)

    # A previous body carrying v1-era evidence is not reusable either.
    with pytest.raises(ValueError, match="full Docker verification is required"):
        prepare_update_verification("## Changes\n\nFix a typo.\n", _full_block(tree, v1), v2, tree)


def test_reused_environment_cannot_swap_environment():
    """Reused evidence must keep the previously recorded environment: the same
    inputs and tree with a different environment string claims reuse over an
    environment that was never verified."""
    old_tree = "a" * 40
    new_tree = "b" * 40
    inputs = _inputs_sha()
    previous = _reused_block(old_tree, inputs)
    swapped = format_verification_metadata("reused-environment", inputs, old_tree, "never-verified-env")

    with pytest.raises(ValueError, match="does not match current environment"):
        prepare_update_verification(swapped, previous, inputs, new_tree)


def test_environment_mismatch_is_rejected():
    inputs = _inputs_sha()
    tree = "a" * 40
    body = _full_block(tree, inputs)
    with pytest.raises(ValueError, match="does not match current environment"):
        validate_verification_metadata(body, inputs, tree, expected_environment="other-env")


def test_placeholder_tainted_inputs_keep_gate_but_never_support_reuse():
    placeholder_files = [{"filename": "scripts/setup.sh", "patch": ""}]  # missing patch
    assert verification_inputs_placeholder_tainted(placeholder_files) is True
    # A placeholder record keeps the gate triggered (conservative)...
    placeholder_v2 = compute_verification_inputs(placeholder_files)
    assert placeholder_v2 is not None
    # ...but a hash recorded over complete inputs never matches the placeholder
    # recomputation, so placeholder-derived evidence cannot be reused.
    assert placeholder_v2 != _inputs_sha()


# --------------------------------------------------------------------------- #
# Deterministic local-git diff inputs (shared author/reviewer fingerprint
# source; platform files-API rename pairing diverges from local git).
# --------------------------------------------------------------------------- #


def _scratch_repo(tmp_path, base_files, moves=None, head_files=None):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"git {args}: {result.stderr}"
        return result.stdout

    def write(files):
        for path, content in files.items():
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    def commit(message):
        git("add", "-A")
        git("-c", "user.name=T", "-c", "user.email=t@example.com", "commit", "-qm", message)

    git("init", "-q")
    write(base_files)
    commit("chore: base")
    base = git("rev-parse", "HEAD").strip()
    for old, new in moves or []:
        (repo / new).parent.mkdir(parents=True, exist_ok=True)
        git("mv", "--", old, new)
    write(head_files or {})
    commit("feat: head")
    head = git("rev-parse", "HEAD").strip()
    return repo, base, head


def _collect_from(repo, base, head):
    def git(args):
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        return result.stdout

    return collect_diff_file_infos(git, base, head)


def test_collect_expands_pure_rename_to_full_delete_and_add(tmp_path):
    """With pairing disabled, a pure rename is a full delete of the old path
    plus a full add of the new path — complete content on both sides, never
    a similarity-trimmed rename diff."""
    repo, base, head = _scratch_repo(
        tmp_path,
        base_files={"requirements/ubuntu.txt": "old-a\nold-b\n"},
        moves=[("requirements/ubuntu.txt", "requirements/server.txt")],
    )
    infos = _collect_from(repo, base, head)
    by_name = {info["filename"]: info["patch"] for info in infos}
    assert set(by_name) == {"requirements/ubuntu.txt", "requirements/server.txt"}
    assert "+old-a\n+old-b" in by_name["requirements/server.txt"]
    assert "-old-a\n-old-b" in by_name["requirements/ubuntu.txt"]


def test_collect_keeps_gate_trigger_for_rename_out_of_gated_path(tmp_path):
    """requirements/ubuntu.txt -> docs/ubuntu.txt: the delete record of the
    old gated path keeps the gate triggered even though the new path is not
    gate-relevant and a rename-paired entry would list only docs/ubuntu.txt."""
    repo, base, head = _scratch_repo(
        tmp_path,
        base_files={"requirements/ubuntu.txt": "old-a\n"},
        moves=[("requirements/ubuntu.txt", "docs/ubuntu.txt")],
    )
    infos = _collect_from(repo, base, head)
    names = {info["filename"] for info in infos}
    assert "requirements/ubuntu.txt" in names  # old gated path stays visible as a delete
    assert compute_verification_inputs(infos) is not None


def test_collect_ignores_user_diff_configuration(tmp_path):
    """The fixed flag set must override user diff configuration so author and
    reviewer derive identical inputs from any clone."""
    repo, base, head = _scratch_repo(
        tmp_path,
        base_files={"requirements/ubuntu.txt": "old-a\nold-b\n"},
        moves=[("requirements/ubuntu.txt", "requirements/server.txt")],
        head_files={"requirements/server.txt": "old-a\nold-b\nserver-c\n"},
    )
    baseline = _collect_from(repo, base, head)
    for key, value in (
        ("diff.renames", "false"),
        ("diff.renames", "copies"),
        ("diff.algorithm", "histogram"),
        ("diff.indentHeuristic", "true"),
    ):
        subprocess.run(["git", "-C", str(repo), "config", key, value], check=True, timeout=60)
    assert _collect_from(repo, base, head) == baseline
