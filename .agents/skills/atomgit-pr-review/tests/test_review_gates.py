"""Behavior tests for the offline reviewer-side review_gates audit entry.

Inputs are plain JSON files (the shapes ``gitcode pr view --json`` and the
``api`` endpoints produce); the audit itself is offline. These tests cover
this project's own checks only — not the official CLI.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))

from pr_gates import add_ai_disclosure, compute_verification_inputs, format_verification_metadata

SCRIPT = Path(__file__).parents[1] / "scripts" / "review_gates.py"
ENV_TEXT = "ubuntu:ros-humble-desktop-full-jammy|openeuler:ibrobot-dev-env"
TREE = "a" * 40
HEAD_SHA = "b" * 40

DISCLOSED_BODY = add_ai_disclosure(
    "## 方案\n\n正文。",
    agent_tool="OpenCode 1.17.20",
    ai_model="glm-5.3",
    prompt_summary="Implement the feature",
    third_party_materials="无",
)


def gated_files() -> list[dict]:
    return [{"filename": "scripts/setup.sh", "additions": 1, "deletions": 1, "patch": "@@ -1 +1 @@\n-old\n+new"}]


def plain_files() -> list[dict]:
    return [{"filename": "src/feature.py", "additions": 10, "deletions": 2, "patch": "@@ -1,2 +1,10 @@\n ctx\n+new"}]


def ai_commits() -> list[dict]:
    return [
        {
            "sha": "c" * 40,
            "commit": {"message": "feat: add feature\n\nBody.\n\nCo-Authored-By: glm-5.3"},
        }
    ]


def make_pr(
    title: str = "feat: add feature",
    body: str = DISCLOSED_BODY,
    head_sha: str = HEAD_SHA,
    base_sha: str | None = None,
) -> dict:
    pr = {
        "number": 123,
        "title": title,
        "body": body,
        "head": {"sha": head_sha, "ref": "feat/change"},
        "changed_files": None,
        "commits": None,
    }
    if base_sha is not None:
        pr["base"] = {"sha": base_sha, "ref": "master"}
    return pr


def write_context(tmp_path: Path, pr: dict, files: list, commits: list) -> tuple[Path, Path, Path]:
    pr_json = tmp_path / "pr.json"
    files_json = tmp_path / "files.json"
    commits_json = tmp_path / "commits.json"
    pr_json.write_text(json.dumps(pr), encoding="utf-8")
    files_json.write_text(json.dumps(files), encoding="utf-8")
    commits_json.write_text(json.dumps(commits), encoding="utf-8")
    return pr_json, files_json, commits_json


def run_gates(tmp_path: Path, pr: dict, files: list, commits: list, *extra: str) -> subprocess.CompletedProcess:
    pr_json, files_json, commits_json = write_context(tmp_path, pr, files, commits)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--pr-json",
            str(pr_json),
            "--files-json",
            str(files_json),
            "--commits-json",
            str(commits_json),
            *extra,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )


def checks_of(result: subprocess.CompletedProcess) -> list[dict]:
    return json.loads(result.stdout)["checks"]


def ids_of(result: subprocess.CompletedProcess) -> set[str]:
    return {check["id"] for check in checks_of(result)}


def test_clean_pr_has_no_checks(tmp_path):
    repo, base_sha, head_sha, _tree = make_local_repo(tmp_path, head_files={"src/feature.py": "VALUE = 1\n"})
    result = run_gates(
        tmp_path, make_pr(base_sha=base_sha, head_sha=head_sha), plain_files(), ai_commits(), "--repo-root", str(repo)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert checks_of(result) == []


def test_lerobot_gitlink_is_blocking(tmp_path):
    files = plain_files() + [
        {
            "filename": "libs/lerobot",
            "mode": "160000",
            "patch": "-Subproject commit 1111111111111111111111111111111111111111\n+Subproject commit 2222222222222222222222222222222222222222\n",
        }
    ]
    # The local head commit also touches a gate file (lerobot patch stack),
    # so the docker checks fire alongside the gitlink finding on a non-WIP PR.
    repo, base_sha, head_sha, _tree = make_local_repo(
        tmp_path, head_files={"third_party/patches/lerobot/0001-demo.patch": "+demo\n"}
    )
    result = run_gates(
        tmp_path, make_pr(base_sha=base_sha, head_sha=head_sha), files, ai_commits(), "--repo-root", str(repo)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    checks = checks_of(result)
    lerobot = [c for c in checks if c["id"] == "lerobot_gitlink_changed"]
    assert len(lerobot) == 1 and lerobot[0]["blocking_until_reviewed"]
    assert "2222222222222222222222222222222222222222" in lerobot[0]["detected_commits"]
    assert "docker_verification_missing" in ids_of(result)


def make_local_repo(
    tmp_path: Path,
    base_files: dict[str, str] | None = None,
    head_files: dict[str, str] | None = None,
    moves: list[tuple[str, str]] | None = None,
) -> tuple[Path, str, str, str]:
    """Scratch repo with a base commit and a head commit.

    Returns ``(repo, base_sha, head_sha, head_tree)``. ``moves`` performs
    ``git mv`` pairs between the two commits so rename scenarios can be
    exercised against the deterministic diff source.
    """
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, f"git {args}: {result.stderr}"
        return result.stdout

    def write(files: dict[str, str]) -> None:
        for path, content in files.items():
            target = repo / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")

    git("init", "-q")
    write(base_files if base_files is not None else {"README.md": "base\n"})
    git("add", "-A")
    git("-c", "user.name=T", "-c", "user.email=t@example.com", "commit", "-qm", "chore: base")
    base_sha = git("rev-parse", "HEAD").strip()
    for old, new in moves or []:
        (repo / new).parent.mkdir(parents=True, exist_ok=True)
        git("mv", "--", old, new)
    write(head_files if head_files is not None else {"scripts/setup.sh": "set -e\n"})
    git("add", "-A")
    git("-c", "user.name=T", "-c", "user.email=t@example.com", "commit", "-qm", "feat: change")
    head_sha = git("rev-parse", "HEAD").strip()
    tree = git("rev-parse", "HEAD^{tree}").strip()
    return repo, base_sha, head_sha, tree


@pytest.mark.parametrize(
    "message_fields",
    [
        {"commit": {"message": "feat: more\n\nCo-Authored-By: gpt-9.1"}},
        {"message": "feat: more\n\nCo-Authored-By: gpt-9.1"},
        {"subject": "feat: more", "body": "Co-Authored-By: gpt-9.1"},
    ],
)
def test_ai_model_coverage_mismatch_is_blocking(tmp_path, message_fields):
    result = run_gates(
        tmp_path,
        make_pr(),
        plain_files(),
        ai_commits() + [{"sha": "d" * 40, **message_fields}],
    )
    assert result.returncode == 0
    assert "ai_model_metadata_mismatch" in ids_of(result)


def test_ai_commit_without_trailer_when_declared_is_blocking(tmp_path):
    commits = [{"sha": "c" * 40, "commit": {"message": "feat: add feature\n\nNo trailer."}}]
    result = run_gates(tmp_path, make_pr(), plain_files(), commits)
    assert result.returncode == 0
    assert "ai_commit_metadata_missing" in ids_of(result)


def test_ai_provider_prefix_is_blocking(tmp_path):
    result = run_gates(
        tmp_path,
        make_pr(),
        plain_files(),
        [{"sha": "c" * 40, "commit": {"message": "feat: x\n\nCo-Authored-By: zhongruan/glm-5.3"}}],
    )
    assert result.returncode == 0
    assert "ai_model_provider_prefix" in ids_of(result)


def test_ai_incomplete_disclosure_is_blocking(tmp_path):
    result = run_gates(tmp_path, make_pr(body="## 方案\n\n（无披露块，但 commit 有模型）"), plain_files(), ai_commits())
    assert result.returncode == 0
    assert "ai_disclosure_incomplete" in ids_of(result)


def test_large_pr_reuse_self_check_missing_and_incomplete(tmp_path):
    files = [{"filename": "src/big.py", "additions": 1500, "deletions": 501, "patch": ""}]
    result = run_gates(tmp_path, make_pr(), files, ai_commits())
    assert "large_pr_reuse_self_check_missing" in ids_of(result)

    body = DISCLOSED_BODY + "\n## Reuse Self-Check\n\n**Reinvented workflows:** 无\n"
    result = run_gates(tmp_path, make_pr(body=body), files, ai_commits())
    assert "large_pr_reuse_self_check_incomplete" in ids_of(result)


def test_docker_verification_missing_for_gated_non_wip_pr(tmp_path):
    repo, base_sha, head_sha, _tree = make_local_repo(tmp_path)
    result = run_gates(
        tmp_path, make_pr(base_sha=base_sha, head_sha=head_sha), gated_files(), ai_commits(), "--repo-root", str(repo)
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "docker_verification_missing" in ids_of(result)


def test_docker_verification_wip_title_defers_check_and_skips_local_inputs(tmp_path):
    # WIP defers the whole Docker branch: no local base/head needed, and the
    # unavailable-inputs check must not appear either.
    result = run_gates(
        tmp_path,
        make_pr(title="[WIP] feat: add feature", base_sha="1" * 40, head_sha="2" * 40),
        gated_files(),
        ai_commits(),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "docker_verification" not in " ".join(ids_of(result))


def test_docker_verification_inputs_unavailable_without_local_commits(tmp_path):
    """Non-WIP PR whose base/head commits are not resolvable locally: the
    audit blocks on the deterministic input source instead of silently
    falling back to the platform files list (whose rename pairing diverges
    from local git)."""
    result = run_gates(tmp_path, make_pr(base_sha="1" * 40, head_sha="2" * 40), gated_files(), ai_commits())
    assert result.returncode == 0, result.stdout + result.stderr
    ids = ids_of(result)
    assert "docker_verification_inputs_unavailable" in ids
    assert "docker_verification_missing" not in ids
    assert "deterministic local diff" in result.stdout


def test_docker_verification_mismatch_reports_expected_fingerprint(tmp_path):
    repo, base_sha, head_sha, tree = make_local_repo(tmp_path)
    # Fixed expectation: the head commit adds scripts/setup.sh with one line,
    # so the gate record is exactly ("scripts/setup.sh", "+set -e").
    expected = compute_verification_inputs([{"filename": "scripts/setup.sh", "patch": "+set -e"}])
    body = DISCLOSED_BODY + "\n" + format_verification_metadata("full", "0" * 40, tree, ENV_TEXT) + "\n"
    result = run_gates(
        tmp_path,
        make_pr(base_sha=base_sha, head_sha=head_sha, body=body),
        gated_files(),
        ai_commits(),
        "--repo-root",
        str(repo),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "docker_verification_mismatch" in ids_of(result)
    assert expected in result.stdout


def test_docker_verification_full_block_validates_against_local_tree(tmp_path):
    repo, base_sha, head_sha, tree = make_local_repo(tmp_path)
    inputs = compute_verification_inputs([{"filename": "scripts/setup.sh", "patch": "+set -e"}])
    body = DISCLOSED_BODY + "\n" + format_verification_metadata("full", inputs, tree, ENV_TEXT) + "\n"
    result = run_gates(
        tmp_path,
        make_pr(base_sha=base_sha, head_sha=head_sha, body=body),
        gated_files(),
        ai_commits(),
        "--repo-root",
        str(repo),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "docker_verification" not in " ".join(ids_of(result))


def test_docker_verification_reused_mode_checks_fingerprint_only(tmp_path):
    repo, base_sha, head_sha, _tree = make_local_repo(tmp_path)
    inputs = compute_verification_inputs([{"filename": "scripts/setup.sh", "patch": "+set -e"}])
    body = DISCLOSED_BODY + "\n" + format_verification_metadata("reused-environment", inputs, "e" * 40, ENV_TEXT) + "\n"
    result = run_gates(
        tmp_path,
        make_pr(base_sha=base_sha, head_sha=head_sha, body=body),
        gated_files(),
        ai_commits(),
        "--repo-root",
        str(repo),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "docker_verification" not in " ".join(ids_of(result))


def _load_review_gates():
    import importlib.util

    spec = importlib.util.spec_from_file_location("review_gates_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_incomplete_local_inputs_stay_blocked(tmp_path):
    """Small direct regression for the placeholder guard: real local git
    diffs are structurally complete, but the incomplete-input refusal is
    current policy (binary/attribute edge outputs) and stays enforced when
    such inputs reach the check layer."""
    module = _load_review_gates()
    incomplete = [{"filename": "scripts/setup.sh", "additions": 1, "deletions": 1, "patch": ""}]
    placeholder_hash = compute_verification_inputs(incomplete)
    pr = make_pr(
        body=DISCLOSED_BODY
        + "\n"
        + format_verification_metadata("reused-environment", placeholder_hash, TREE, ENV_TEXT)
        + "\n"
    )
    checks = module.build_verification_tree_checks(pr, TREE, incomplete, None)
    ids = {check["id"] for check in checks}
    assert "docker_verification_inputs_incomplete" in ids
    assert "docker_verification_mismatch" not in ids


def test_renamed_gate_file_fingerprint_matches_author_records(tmp_path):
    """Rename divergence regression (review side): platform files pair a
    rename into one entry with a content-only patch, the deterministic
    source expands it to a full delete of the old path plus a full add of
    the new path, and a body block quoting that fixed fingerprint is
    accepted exactly as the author gate records it."""
    repo, base_sha, head_sha, tree = make_local_repo(
        tmp_path,
        base_files={"requirements/ubuntu.txt": "old-a\nold-b\n"},
        moves=[("requirements/ubuntu.txt", "requirements/server.txt")],
        head_files={"requirements/server.txt": "old-a\nold-b\nserver-c\n"},
    )
    deterministic = compute_verification_inputs(
        [
            {"filename": "requirements/server.txt", "patch": "+old-a\n+old-b\n+server-c"},
            {"filename": "requirements/ubuntu.txt", "patch": "-old-a\n-old-b"},
        ]
    )
    platform_renamed = compute_verification_inputs(
        [{"filename": "requirements/server.txt", "previous_filename": "", "patch": " old-a\n old-b\n+server-c"}]
    )
    assert deterministic != platform_renamed  # the divergence this fix removes
    body = DISCLOSED_BODY + "\n" + format_verification_metadata("full", deterministic, tree, ENV_TEXT) + "\n"
    result = run_gates(
        tmp_path,
        make_pr(base_sha=base_sha, head_sha=head_sha, body=body),
        gated_files(),
        ai_commits(),
        "--repo-root",
        str(repo),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "docker_verification" not in " ".join(ids_of(result))


def test_non_object_entries_are_input_errors(tmp_path):
    result = run_gates(tmp_path, make_pr(), plain_files() + ["garbage"], ai_commits())
    assert result.returncode == 1
    assert "non-object entries" in result.stdout


def test_duplicate_identities_are_input_errors(tmp_path):
    result = run_gates(tmp_path, make_pr(), plain_files() + plain_files(), ai_commits())
    assert result.returncode == 1
    assert "duplicate" in result.stdout


def test_declared_total_mismatch_is_input_error(tmp_path):
    pr = make_pr()
    pr["changed_files"] = 5
    result = run_gates(tmp_path, pr, plain_files(), ai_commits())
    assert result.returncode == 1
    assert "declares 5" in result.stdout


def test_audit_output_is_pending_findings_not_a_pass(tmp_path):
    result = run_gates(tmp_path, make_pr(), gated_files(), ai_commits())
    assert result.returncode == 0
    assert "待处理" in result.stderr or "阻塞" in result.stderr
    summary = json.loads(result.stdout)
    assert summary["checks"], "non-empty checks must be visible in the JSON summary"
    assert summary["wip"] is False
