"""Entry-point behavior tests for the CLI-migrated link validator (task 3.3).

The real ``deepwiki-translator/scripts/link_validator.py`` runs as a
subprocess against the executable fake CLI (task 2.4), so every AtomGit
branch — repo, pull, commit, issues, milestones, blob/tree and user/org —
is exercised through the shared boundary exactly as an Agent drives it.
Required semantics (design.md D1/D5, task 3.3):

- CLI missing or not authenticated → every AtomGit link is explicit
  *inconclusive*, never ``broken``; local links keep validating and no
  platform probe is issued.
- Exit 3 → broken; exit 0 → valid. ``commit view`` has no ``--json`` flag
  in the verified help evidence, so its plain-text success must count as
  valid rather than crash the JSON decode.
- A probe timeout maps to inconclusive instead of escaping the worker
  thread and crashing the whole run.
- Legacy ``--config``/``--access-token`` are rejected with the CLI login
  guidance and exit 2 before any probe.

Third-party external links keep the urllib probe; they would hit real
networks from tests and are therefore not exercised here.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
SKILLS_DIR = TESTS_DIR.parents[1]
SHARED_TESTS_DIR = SKILLS_DIR / "_shared" / "tests"

sys.path.insert(0, str(SHARED_TESTS_DIR))

from entrypoint_fixture import AUTH_LOGIN_GUIDANCE, EntrypointFakeCLI  # noqa: E402

LINK_VALIDATOR_SCRIPT = SKILLS_DIR / "deepwiki-translator" / "scripts" / "link_validator.py"
AUTH_OK_RULE = {"argv_contains": ["auth", "status"], "stdout": "Logged in to atomgit.com\n", "exit": 0}
BLOB_CONTENT = base64.b64encode(b"line1\nline2\nline3\n").decode()


@pytest.fixture
def fake(tmp_path: Path) -> EntrypointFakeCLI:
    return EntrypointFakeCLI(tmp_path)


def unavailable_cli_env(tmp_path: Path) -> dict[str, str]:
    """Environment whose explicit CLI selection points at a missing file."""
    env = dict(os.environ)
    env["IBROBOT_ATOMGIT_CLI"] = str(tmp_path / "missing" / "gitcode")
    return env


def write_doc(tmp_path: Path, markdown: str) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "page.md").write_text(markdown, encoding="utf-8")
    return docs


def run_validator(docs: Path, report: Path, *extra_args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(LINK_VALIDATOR_SCRIPT),
            str(docs),
            "--root",
            str(docs),
            "--report",
            str(report),
            *extra_args,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(docs.parent),
        timeout=120,
    )


def load_report(report: Path) -> dict:
    return json.loads(report.read_text(encoding="utf-8"))


def reported(report: dict, url: str) -> dict:
    matches = [entry for entry in report["results"] if entry["url"] == url]
    assert len(matches) == 1, f"expected exactly one report entry for {url}, got {matches}"
    return matches[0]


def test_cli_unavailable_marks_atomgit_links_inconclusive(tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[repo](https://atomgit.com/acme/widget)\n[local](neighbor.md)\n")
    (docs / "neighbor.md").write_text("# Neighbor\n", encoding="utf-8")
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=unavailable_cli_env(tmp_path))

    assert proc.returncode == 0  # inconclusive without the strict flag is not a failure
    data = load_report(report)
    assert data["checked_summary"]["inconclusive"] == 1
    entry = reported(data, "https://atomgit.com/acme/widget")
    assert entry["status"] == "inconclusive"
    assert entry["detail"].startswith("IBROBOT_ATOMGIT_CLI points to a missing file")
    assert data["checked_summary"]["valid"] == 1  # the local link still validated


def test_fail_on_inconclusive_flag_returns_nonzero(tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[repo](https://atomgit.com/acme/widget)\n")
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, "--fail-on-inconclusive", env=unavailable_cli_env(tmp_path))

    assert proc.returncode == 1


def test_not_authenticated_gives_login_guidance_and_skips_probes(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[repo](https://atomgit.com/acme/widget)\n")
    fake.configure([{"argv_contains": ["auth", "status"], "exit": 4, "stderr": "not logged in"}])
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 0
    assert [invocation[:1] for invocation in fake.argvs()] == [["version"], ["auth"]]  # no platform probe
    entry = reported(load_report(report), "https://atomgit.com/acme/widget")
    assert entry["status"] == "inconclusive"
    assert AUTH_LOGIN_GUIDANCE in entry["detail"]


@pytest.mark.parametrize("retired", [["--config", "config.json"], ["--access-token", "secret"]])
def test_retired_token_flags_exit_before_any_probe(fake: EntrypointFakeCLI, tmp_path: Path, retired: list[str]) -> None:
    docs = write_doc(tmp_path, "[repo](https://atomgit.com/acme/widget)\n")
    fake.configure([AUTH_OK_RULE])
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, *retired, env=fake.env())

    assert proc.returncode == 2
    assert "retired" in proc.stderr
    assert AUTH_LOGIN_GUIDANCE in proc.stderr
    assert fake.argvs() == []  # rejected before even the version probe
    assert not report.exists()


def test_repo_link_exit3_is_broken(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[repo](https://atomgit.com/acme/widget)\n")
    fake.configure([AUTH_OK_RULE, {"argv_contains": ["repo", "view"], "exit": 3, "stderr": "not found"}])
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 1
    assert fake.argv_of(["repo", "view"]) == ["repo", "view", "acme/widget", "--json", "--no-interactive"]
    entry = reported(load_report(report), "https://atomgit.com/acme/widget")
    assert entry["status"] == "broken"
    assert entry["detail"] == "AtomGit target was not found"


def test_pull_link_uses_pr_view_and_pins_host(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[pull](https://atomgit.com/acme/widget/pull/7)\n")
    fake.configure([AUTH_OK_RULE, {"argv_contains": ["pr", "view"], "stdout": '{"number": 7}', "exit": 0}])
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 0
    expected_argv = ["pr", "view", "7", "-R", "acme/widget", "--json", "--no-interactive"]
    assert fake.argv_of(["pr", "view"]) == expected_argv
    assert fake.invocation_of(["pr", "view"])["env"]["GC_HOST"] == "atomgit.com"
    assert load_report(report)["checked_summary"]["valid"] == 1


def test_commit_link_valid_without_json_flag(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[commit](https://atomgit.com/acme/widget/commit/abc123def456)\n")
    fake.configure(
        [
            AUTH_OK_RULE,
            {"argv_contains": ["commit", "view"], "stdout": "commit abc123def456\nAuthor: someone\n", "exit": 0},
        ]
    )
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 0
    assert fake.argv_of(["commit", "view"]) == [
        "commit",
        "view",
        "abc123def456",
        "-R",
        "acme/widget",
        "--no-interactive",
    ]  # no --json: the verified help evidence has no such flag for commit view
    assert load_report(report)["checked_summary"]["valid"] == 1


def test_commit_link_exit3_is_broken(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[commit](https://atomgit.com/acme/widget/commit/abc123def456)\n")
    fake.configure([AUTH_OK_RULE, {"argv_contains": ["commit", "view"], "exit": 3}])
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 1
    assert reported(load_report(report), "https://atomgit.com/acme/widget/commit/abc123def456")["status"] == "broken"


def test_blob_link_line_fragment_within_file(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[guide](https://atomgit.com/acme/widget/blob/main/docs/guide.md#L2-L3)\n")
    fake.configure(
        [
            AUTH_OK_RULE,
            {"argv_contains": ["api"], "stdout": json.dumps({"type": "file", "content": BLOB_CONTENT}), "exit": 0},
        ]
    )
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 0
    assert fake.argv_of(["api"]) == ["api", "repos/acme/widget/contents/docs/guide.md?ref=main", "--no-interactive"]
    assert load_report(report)["checked_summary"]["valid"] == 1


def test_blob_link_fragment_beyond_file_length(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[guide](https://atomgit.com/acme/widget/blob/main/docs/guide.md#L99)\n")
    fake.configure(
        [
            AUTH_OK_RULE,
            {"argv_contains": ["api"], "stdout": json.dumps({"type": "file", "content": BLOB_CONTENT}), "exit": 0},
        ]
    )
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 1
    entry = reported(load_report(report), "https://atomgit.com/acme/widget/blob/main/docs/guide.md#L99")
    assert entry["status"] == "broken"
    assert entry["detail"] == "line fragment #L99 exceeds file length 3"


def test_tree_link_to_file_target_is_broken(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[tree](https://atomgit.com/acme/widget/tree/main/docs/guide.md)\n")
    fake.configure(
        [
            AUTH_OK_RULE,
            {"argv_contains": ["api"], "stdout": json.dumps({"type": "file", "content": BLOB_CONTENT}), "exit": 0},
        ]
    )
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 1
    entry = reported(load_report(report), "https://atomgit.com/acme/widget/tree/main/docs/guide.md")
    assert entry["status"] == "broken"
    assert entry["detail"] == "AtomGit file target should use /blob/ instead of /tree/"


def test_blob_link_to_directory_target_is_broken(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[blob](https://atomgit.com/acme/widget/blob/main/docs)\n")
    fake.configure(
        [
            AUTH_OK_RULE,
            {"argv_contains": ["api"], "stdout": json.dumps({"type": "dir"}), "exit": 0},
        ]
    )
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 1
    entry = reported(load_report(report), "https://atomgit.com/acme/widget/blob/main/docs")
    assert entry["status"] == "broken"
    assert entry["detail"] == "AtomGit directory target should use /tree/ instead of /blob/"


def test_user_link_valid_without_org_probe(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[user](https://atomgit.com/janedoe)\n")
    fake.configure([AUTH_OK_RULE, {"argv_contains": ["user", "view"], "stdout": '{"login": "janedoe"}', "exit": 0}])
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 0
    assert fake.argv_of(["user", "view"]) == ["user", "view", "janedoe", "--json", "--no-interactive"]
    assert not any("api" in argv for argv in fake.argvs())
    assert load_report(report)["checked_summary"]["valid"] == 1


def test_org_link_falls_back_to_orgs_endpoint(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[org](https://atomgit.com/acme-org)\n")
    fake.configure(
        [
            AUTH_OK_RULE,
            {"argv_contains": ["user", "view"], "exit": 3},
            {"argv_contains": ["api"], "stdout": '{"id": 1}', "exit": 0},
        ]
    )
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 0
    assert fake.argv_of(["api"]) == ["api", "orgs/acme-org", "--no-interactive"]
    assert load_report(report)["checked_summary"]["valid"] == 1


def test_org_link_missing_everywhere_is_broken(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[org](https://atomgit.com/acme-org)\n")
    fake.configure(
        [
            AUTH_OK_RULE,
            {"argv_contains": ["user", "view"], "exit": 3},
            {"argv_contains": ["api"], "exit": 3},
        ]
    )
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 1
    entry = reported(load_report(report), "https://atomgit.com/acme-org")
    assert entry["status"] == "broken"
    assert entry["detail"] == "AtomGit user or org target was not found"


def test_org_link_with_inconclusive_user_probe_never_broken(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[org](https://atomgit.com/acme-org)\n")
    fake.configure(
        [
            AUTH_OK_RULE,
            {"argv_contains": ["user", "view"], "exit": 1, "stderr": "server error"},
            {"argv_contains": ["api"], "exit": 3},
        ]
    )
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 0
    entry = reported(load_report(report), "https://atomgit.com/acme-org")
    assert entry["status"] == "inconclusive"
    assert entry["detail"]  # carries the inconclusive probe's reason


def test_issues_collection_route_probes_issue_list(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[issues](https://atomgit.com/acme/widget/issues)\n")
    fake.configure([AUTH_OK_RULE, {"argv_contains": ["issue", "list"], "stdout": "[]", "exit": 0}])
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 0
    assert fake.argv_of(["issue", "list"]) == [
        "issue",
        "list",
        "-R",
        "acme/widget",
        "--json",
        "-L",
        "1",
        "--no-interactive",
    ]
    assert load_report(report)["checked_summary"]["valid"] == 1


def test_issue_number_route_exit3_is_broken(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[issue](https://atomgit.com/acme/widget/issues/5)\n")
    fake.configure([AUTH_OK_RULE, {"argv_contains": ["issue", "view"], "exit": 3}])
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 1
    assert fake.argv_of(["issue", "view"]) == ["issue", "view", "5", "-R", "acme/widget", "--json", "--no-interactive"]
    assert reported(load_report(report), "https://atomgit.com/acme/widget/issues/5")["status"] == "broken"


def test_milestone_link_uses_api_endpoint(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[m](https://atomgit.com/acme/widget/milestones/3)\n")
    fake.configure([AUTH_OK_RULE, {"argv_contains": ["api"], "stdout": '{"number": 3}', "exit": 0}])
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=fake.env())

    assert proc.returncode == 0
    assert fake.argv_of(["api"]) == ["api", "repos/acme/widget/milestones/3", "--no-interactive"]
    assert load_report(report)["checked_summary"]["valid"] == 1


def test_probe_timeout_is_inconclusive_not_a_crash(fake: EntrypointFakeCLI, tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[repo](https://atomgit.com/acme/widget)\n")
    fake.configure([AUTH_OK_RULE, {"argv_contains": ["repo", "view"], "sleep": 3, "exit": 0}])
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, "--timeout", "1", env=fake.env())

    assert proc.returncode == 0
    entry = reported(load_report(report), "https://atomgit.com/acme/widget")
    assert entry["status"] == "inconclusive"
    assert "timed out after 1s" in entry["detail"]


def test_bolb_typo_is_broken_even_without_cli(tmp_path: Path) -> None:
    docs = write_doc(tmp_path, "[bad](https://atomgit.com/acme/widget/bolb/main/docs/guide.md)\n")
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=unavailable_cli_env(tmp_path))

    assert proc.returncode == 1
    entry = reported(load_report(report), "https://atomgit.com/acme/widget/bolb/main/docs/guide.md")
    assert entry["status"] == "broken"
    assert entry["detail"] == "AtomGit route typo: /bolb/ should be /blob/"


def test_incomplete_blob_url_is_broken_even_without_cli(tmp_path: Path) -> None:
    # Legacy pre-token-gate order: structurally incomplete blob/tree URLs are
    # offline-decidable and stay broken with the CLI missing.
    docs = write_doc(tmp_path, "[x](https://atomgit.com/acme/widget/blob/main)\n")
    report = tmp_path / "report.json"

    proc = run_validator(docs, report, env=unavailable_cli_env(tmp_path))

    assert proc.returncode == 1
    entry = reported(load_report(report), "https://atomgit.com/acme/widget/blob/main")
    assert entry["status"] == "broken"
    assert entry["detail"] == "AtomGit blob/tree URL is missing branch or path"
