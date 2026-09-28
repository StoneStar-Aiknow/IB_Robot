"""Entry-point behavior tests for the CLI-migrated Issue/intro skills (task 3.1).

These tests execute the real script entry points — ``atomgit-issue/scripts/
issue_management.py`` and ``intro/scripts/intro.py`` — as subprocesses against
the executable fake CLI from ``fake_cli.py`` (task 2.4), so the full path an
Agent drives is exercised for real: argument parsing → shared CLI boundary →
execve argv/stdin → exit-code classification → context/report output.

Scenarios required by the task 3.1 review: authentication exit 4, nonzero
exit with partial JSON, malformed list payloads, the intro sampling limit,
partial comment-read failures, no partial issue context on fetch failure, and
dry-run performing no platform writes. No real platform access, no
credentials.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from entrypoint_fixture import AUTH_LOGIN_GUIDANCE, EntrypointFakeCLI

TESTS_DIR = Path(__file__).resolve().parent
SKILLS_DIR = TESTS_DIR.parents[1]
ISSUE_SCRIPT = SKILLS_DIR / "atomgit-issue" / "scripts" / "issue_management.py"
INTRO_SCRIPT = SKILLS_DIR / "intro" / "scripts" / "intro.py"

ISSUE_TARGET_ARGS = ["--owner", "acme", "--repo", "widget"]


@pytest.fixture
def fake(tmp_path: Path) -> EntrypointFakeCLI:
    return EntrypointFakeCLI(tmp_path)


@pytest.fixture
def intro_cwd(tmp_path: Path) -> Path:
    cwd = tmp_path / "intro-cwd"
    cwd.mkdir()
    (cwd / "config.json").write_text(json.dumps({"atomgit": {"owner": "acme", "repo": "widget"}}), encoding="utf-8")
    return cwd


@pytest.fixture
def issue_cwd(tmp_path: Path) -> Path:
    cwd = tmp_path / "issue-cwd"
    cwd.mkdir()
    return cwd


def run_script(script: Path, args: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        env=env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


def load_intro_report(intro_cwd: Path) -> dict:
    return json.loads((intro_cwd / "tmp" / "intro_report.json").read_text(encoding="utf-8"))


def run_issue_script(fake: EntrypointFakeCLI, cwd: Path, args: list[str]) -> subprocess.CompletedProcess:
    return run_script(ISSUE_SCRIPT, [*args, *ISSUE_TARGET_ARGS], env=fake.env(), cwd=cwd)


# --------------------------------------------------------------------------- #
# intro.py entry point
# --------------------------------------------------------------------------- #


def test_intro_auth_exit4_marks_pr_check_unavailable(fake, intro_cwd):
    fake.configure([{"argv_contains": ["pr", "list"], "stderr": "not logged in to atomgit.com", "exit": 4}])
    result = run_script(INTRO_SCRIPT, [], env=fake.env(), cwd=intro_cwd)
    assert result.returncode == 0, result.stderr  # intro stays nonblocking
    prs = load_intro_report(intro_cwd)["prs"]
    assert prs["available"] is False
    assert AUTH_LOGIN_GUIDANCE in prs["error"]
    assert "跳过" in result.stdout
    assert len([argv for argv in fake.argvs() if argv[:2] == ["pr", "list"]]) == 1


def test_intro_nonzero_exit_with_partial_json_marks_unavailable(fake, intro_cwd):
    fake.configure([{"argv_contains": ["pr", "list"], "stdout": '[{"number": 1}]', "exit": 1}])
    result = run_script(INTRO_SCRIPT, [], env=fake.env(), cwd=intro_cwd)
    assert result.returncode == 0, result.stderr
    prs = load_intro_report(intro_cwd)["prs"]
    assert prs["available"] is False
    assert "exit code 1" in prs["error"]
    assert prs["prs_with_comments"] == []


def test_intro_non_list_json_marks_unavailable(fake, intro_cwd):
    fake.configure([{"argv_contains": ["pr", "list"], "stdout": '{"count": 0}', "exit": 0}])
    result = run_script(INTRO_SCRIPT, [], env=fake.env(), cwd=intro_cwd)
    assert result.returncode == 0, result.stderr
    prs = load_intro_report(intro_cwd)["prs"]
    assert prs["available"] is False
    assert "非列表 JSON" in prs["error"]


def test_intro_invalid_pr_entries_are_reported(fake, intro_cwd):
    fake.configure(
        [
            {
                "argv_contains": ["pr", "list"],
                "stdout": json.dumps([{"number": 1, "title": "valid"}, "garbage-entry", {"title": "missing-number"}]),
                "exit": 0,
            },
            {"argv_contains": ["pr", "comments"], "stdout": "[]", "exit": 0},
        ]
    )
    result = run_script(INTRO_SCRIPT, [], env=fake.env(), cwd=intro_cwd)
    assert result.returncode == 0, result.stderr
    prs = load_intro_report(intro_cwd)["prs"]
    assert prs["available"] is True
    assert prs["checked_prs"] == [1]
    assert len(prs["invalid_prs"]) == 2
    assert "无效记录" in result.stdout


def test_intro_sampling_limit_marks_truncated(fake, intro_cwd):
    prs_payload = json.dumps([{"number": number, "title": f"PR {number}"} for number in range(1, 6)])
    fake.configure(
        [
            {"argv_contains": ["pr", "list"], "stdout": prs_payload, "exit": 0},
            {"argv_contains": ["pr", "comments"], "stdout": "[]", "exit": 0},
        ]
    )
    result = run_script(INTRO_SCRIPT, [], env=fake.env(), cwd=intro_cwd)
    assert result.returncode == 0, result.stderr
    prs = load_intro_report(intro_cwd)["prs"]
    assert prs["available"] is True
    assert prs["sample_limit"] == 5
    assert prs["sample_truncated"] is True
    assert prs["checked_prs"] == [1, 2, 3, 4, 5]
    assert "抽样上限" in result.stdout
    assert fake.argv_of(["pr", "list"]) == [
        "pr",
        "list",
        "-R",
        "acme/widget",
        "--state",
        "open",
        "--json",
        "-L",
        "5",
        "--no-interactive",
    ]


def test_intro_partial_comment_failure_is_nonfatal(fake, intro_cwd):
    fake.configure(
        [
            {
                "argv_contains": ["pr", "list"],
                "stdout": json.dumps([{"number": 1, "title": "A"}, {"number": 2, "title": "B"}]),
                "exit": 0,
            },
            {"argv_contains": ["pr", "comments", "1"], "stdout": '[{"body": "please fix"}]', "exit": 0},
            {"argv_contains": ["pr", "comments", "2"], "stderr": "transient failure", "exit": 1},
        ]
    )
    result = run_script(INTRO_SCRIPT, [], env=fake.env(), cwd=intro_cwd)
    assert result.returncode == 0, result.stderr
    prs = load_intro_report(intro_cwd)["prs"]
    assert prs["available"] is True
    assert prs["checked_prs"] == [1]
    assert prs["unchecked_prs"] == [2]
    assert prs["prs_with_comments"] == [{"number": 1, "title": "A", "comment_count": 1}]
    assert "未检查" in result.stdout


# --------------------------------------------------------------------------- #
# issue_management.py entry point
# --------------------------------------------------------------------------- #


def test_issue_fetch_info_writes_context_with_valid_comments(fake, issue_cwd):
    fake.configure(
        [
            {
                "argv_contains": ["issue", "view"],
                "stdout": json.dumps({"number": 7, "title": "Fix wheels", "state": "open"}),
                "exit": 0,
            },
            {
                "argv_contains": ["issue", "comments"],
                "stdout": json.dumps([{"id": 11, "body": "first"}, {"id": 12, "body": "second"}]),
                "exit": 0,
            },
        ]
    )
    out_dir = issue_cwd / "out"
    result = run_issue_script(fake, issue_cwd, ["--fetch-info", "--issue", "7", "--output-dir", str(out_dir)])
    assert result.returncode == 0, result.stdout + result.stderr
    context = json.loads((out_dir / "widget_issue_7_context.json").read_text(encoding="utf-8"))
    assert context["title"] == "Fix wheels"
    assert context["comments_detail"] == [{"id": 11, "body": "first"}, {"id": 12, "body": "second"}]
    assert "Included 2 issue comments" in result.stdout
    assert fake.argv_of(["issue", "view"]) == [
        "issue",
        "view",
        "7",
        "-R",
        "acme/widget",
        "--json",
        "--no-interactive",
    ]
    assert fake.argv_of(["issue", "comments"]) == [
        "issue",
        "comments",
        "7",
        "-R",
        "acme/widget",
        "--json",
        "--limit",
        "0",
        "--order",
        "asc",
        "--no-interactive",
    ]


def test_issue_fetch_info_comment_auth_failure_writes_no_context(fake, issue_cwd):
    fake.configure(
        [
            {"argv_contains": ["issue", "view"], "stdout": '{"number": 7, "title": "Fix"}', "exit": 0},
            {"argv_contains": ["issue", "comments"], "stderr": "not logged in", "exit": 4},
        ]
    )
    out_dir = issue_cwd / "out"
    result = run_issue_script(fake, issue_cwd, ["--fetch-info", "--issue", "7", "--output-dir", str(out_dir)])
    assert result.returncode == 1
    assert AUTH_LOGIN_GUIDANCE in result.stdout
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_issue_fetch_info_comment_non_list_writes_no_context(fake, issue_cwd):
    fake.configure(
        [
            {"argv_contains": ["issue", "view"], "stdout": '{"number": 7}', "exit": 0},
            {"argv_contains": ["issue", "comments"], "stdout": '{"total": 2}', "exit": 0},
        ]
    )
    out_dir = issue_cwd / "out"
    result = run_issue_script(fake, issue_cwd, ["--fetch-info", "--issue", "7", "--output-dir", str(out_dir)])
    assert result.returncode == 1
    assert "instead of a JSON array" in result.stdout
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_issue_fetch_info_comment_non_object_items_write_no_context(fake, issue_cwd):
    fake.configure(
        [
            {"argv_contains": ["issue", "view"], "stdout": '{"number": 7}', "exit": 0},
            {"argv_contains": ["issue", "comments"], "stdout": '["plain string", {"id": 1}]', "exit": 0},
        ]
    )
    out_dir = issue_cwd / "out"
    result = run_issue_script(fake, issue_cwd, ["--fetch-info", "--issue", "7", "--output-dir", str(out_dir)])
    assert result.returncode == 1
    assert "non-object entries" in result.stdout
    assert "partial issue context" in result.stdout
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_issue_fetch_info_view_non_object_rejected(fake, issue_cwd):
    fake.configure(
        [
            {"argv_contains": ["issue", "view"], "stdout": '[{"number": 7}]', "exit": 0},
        ]
    )
    out_dir = issue_cwd / "out"
    result = run_issue_script(fake, issue_cwd, ["--fetch-info", "--issue", "7", "--output-dir", str(out_dir)])
    assert result.returncode == 1
    assert "instead of a JSON object" in result.stdout
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_issue_fetch_info_view_nonzero_partial_json_writes_no_context(fake, issue_cwd):
    fake.configure(
        [
            {"argv_contains": ["issue", "view"], "stdout": '{"number": 7}', "exit": 1},
        ]
    )
    out_dir = issue_cwd / "out"
    result = run_issue_script(fake, issue_cwd, ["--fetch-info", "--issue", "7", "--output-dir", str(out_dir)])
    assert result.returncode == 1
    assert "exit code 1" in result.stdout
    assert not out_dir.exists() or not any(out_dir.iterdir())


def test_issue_comment_auth_exit4_prints_guidance(fake, issue_cwd):
    fake.configure([{"argv_contains": ["issue", "comment"], "stderr": "not logged in", "exit": 4}])
    result = run_issue_script(fake, issue_cwd, ["--issue", "7", "--comment", "hello"])
    assert result.returncode == 1
    assert AUTH_LOGIN_GUIDANCE in result.stdout


def test_issue_comment_sends_body_via_stdin(fake, issue_cwd):
    fake.configure([{"argv_contains": ["issue", "comment"], "stdout": '{"id": 99}', "exit": 0}])
    body = 'line1\nline2 中文 "quoted" $VAR'
    result = run_issue_script(fake, issue_cwd, ["--issue", "7", "--comment", body])
    assert result.returncode == 0, result.stdout
    assert "Successfully commented on issue #7" in result.stdout
    assert "https://atomgit.com/acme/widget/issues/7" in result.stdout
    assert fake.argv_of(["issue", "comment"]) == [
        "issue",
        "comment",
        "7",
        "-R",
        "acme/widget",
        "--body-file",
        "-",
        "--json",
        "--no-interactive",
    ]
    assert fake.invocation_of(["issue", "comment"])["stdin"] == body


def test_issue_update_sends_body_via_stdin(fake, issue_cwd):
    fake.configure([{"argv_contains": ["issue", "edit"], "stdout": '{"number": 7}', "exit": 0}])
    result = run_issue_script(fake, issue_cwd, ["--issue", "7", "--title", "New Title", "--body", "b1\nb2"])
    assert result.returncode == 0, result.stdout
    assert "Successfully updated issue #7" in result.stdout
    assert fake.argv_of(["issue", "edit"]) == [
        "issue",
        "edit",
        "7",
        "-R",
        "acme/widget",
        "--title",
        "New Title",
        "--body-file",
        "-",
        "--json",
        "--no-interactive",
    ]
    assert fake.invocation_of(["issue", "edit"])["stdin"] == "b1\nb2"


def test_issue_update_state_adds_yes(fake, issue_cwd):
    fake.configure([{"argv_contains": ["issue", "edit"], "stdout": '{"number": 7}', "exit": 0}])
    result = run_issue_script(fake, issue_cwd, ["--issue", "7", "--state", "closed"])
    assert result.returncode == 0, result.stdout
    assert fake.argv_of(["issue", "edit"]) == [
        "issue",
        "edit",
        "7",
        "-R",
        "acme/widget",
        "--state",
        "closed",
        "--yes",
        "--json",
        "--no-interactive",
    ]


def test_issue_create_reports_number(fake, issue_cwd):
    fake.configure([{"argv_contains": ["issue", "create"], "stdout": '{"number": 42}', "exit": 0}])
    result = run_issue_script(fake, issue_cwd, ["--title", "Bug report", "--body", "description"])
    assert result.returncode == 0, result.stdout
    assert "Successfully created issue #42" in result.stdout
    assert "https://atomgit.com/acme/widget/issues/42" in result.stdout
    assert fake.argv_of(["issue", "create"]) == [
        "issue",
        "create",
        "-R",
        "acme/widget",
        "--title",
        "Bug report",
        "--body-file",
        "-",
        "--json",
        "--no-interactive",
    ]
    assert fake.invocation_of(["issue", "create"])["stdin"] == "description"


def test_issue_create_without_number_fails_without_retry(fake, issue_cwd):
    fake.configure([{"argv_contains": ["issue", "create"], "stdout": "{}", "exit": 0}])
    result = run_issue_script(fake, issue_cwd, ["--title", "Bug report"])
    assert result.returncode == 1
    assert "may already exist" in result.stdout
    assert len([argv for argv in fake.argvs() if argv[:2] == ["issue", "create"]]) == 1


def test_issue_dry_run_modes_write_nothing_to_platform(fake, issue_cwd):
    fake.configure([])
    for extra in (
        ["--issue", "7", "--comment", "draft comment", "--dry-run"],
        ["--issue", "7", "--title", "New", "--state", "closed", "--dry-run"],
        ["--title", "Draft issue", "--body", "text", "--dry-run"],
    ):
        result = run_issue_script(fake, issue_cwd, extra)
        assert result.returncode == 0, result.stdout
        assert "[DRY RUN]" in result.stdout
    assert fake.mutation_argv() == []
    # Each of the three runs only performed the local identity probe; no
    # platform command was attempted at all.
    assert fake.argvs() == [["version", "--no-interactive"]] * 3
