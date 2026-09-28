"""Entry-point behavior tests for the CLI-migrated review skills (task 4).

These tests execute the real script entry points — ``atomgit-pr-review/
scripts/pr_review.py`` and ``atomgit-pr-architecture-review/scripts/
architecture_review.py`` — as subprocesses against the executable fake CLI
(``fake_cli.py``, task 2.4), so the full path an Agent drives is exercised
for real: argument parsing → shared CLI boundary → ``review_helpers`` argv
building → execve argv/stdin → exit-code classification → context/comment
output.

Scenario coverage required by task 4: extract-info writing the review/arch
context through ``pr view`` / paginated ``api`` endpoints / ``pr comments
--limit 0`` / the percent-encoded contents API, the ``--no-comments`` skip,
comment-read failures never writing a partial context, submit-review posting
the summary plus only new-side-visible inline comments (``--position`` is the
new-side line number; findings off the diff are explicit per-comment failures
without a platform call), PR-level fallback comments for non-changed files in
the architecture skill, auth exit 4 printing the login guidance, and dry-run
performing zero platform mutations. No real platform access, no credentials.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest
from entrypoint_fixture import AUTH_LOGIN_GUIDANCE, EntrypointFakeCLI

TESTS_DIR = Path(__file__).resolve().parent
SKILLS_DIR = TESTS_DIR.parents[1]
REVIEW_SCRIPT = SKILLS_DIR / "atomgit-pr-review" / "scripts" / "pr_review.py"
ARCH_SCRIPT = SKILLS_DIR / "atomgit-pr-architecture-review" / "scripts" / "architecture_review.py"

TARGET_ARGS = ["--owner", "acme", "--repo", "widget"]
AI_MODEL_ARGS = ["--ai-model", "test-model"]
HEAD_SHA = "b" * 40
COMMITS_ENDPOINT = "repos/acme/widget/pulls/7/commits?page=1&per_page=100"
FILES_ENDPOINT = "repos/acme/widget/pulls/7/files?page=1&per_page=100"
FILE_CONTENT_ENDPOINT = f"repos/acme/widget/contents/src%2Ffoo.py?ref={HEAD_SHA}"
README_CONTENT_ENDPOINT = f"repos/acme/widget/contents/src%2FREADME.md?ref={HEAD_SHA}"
# New-side lines 1 and 2 are visible in this patch; line 9 is not.
PATCH = "@@ -1,2 +1,3 @@\n unchanged context\n+added line"


@pytest.fixture
def fake(tmp_path: Path) -> EntrypointFakeCLI:
    return EntrypointFakeCLI(tmp_path)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    cwd = tmp_path / "review-cwd"
    cwd.mkdir()
    # architecture_review.py takes no --owner/--repo flags: it resolves the
    # target from config.json (or git remotes). pr_review.py keeps explicit
    # flags that agree with these config defaults.
    (cwd / "config.json").write_text(
        json.dumps({"atomgit": {"owner": "acme", "repo": "widget"}}),
        encoding="utf-8",
    )
    return cwd


def run_script(script: Path, args: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(script), *args],
        env=env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=90,
    )


def encoded_content(text: str) -> str:
    return json.dumps({"content": base64.b64encode(text.encode("utf-8")).decode("ascii")})


def pr_view_record() -> dict:
    return {
        "number": 7,
        "title": "feat: small change",
        "state": "open",
        "user": {"login": "dev"},
        # A non-HTTPS web_url keeps any head-tree resolution offline and
        # unresolvable; the web_url→html_url mapping is applied by the helper.
        "head": {"ref": "feat/x", "sha": HEAD_SHA, "repo": {"web_url": "file:///tmp/not-a-repo"}},
        "base": {"ref": "master", "sha": "c" * 40, "repo": {}},
        "body": "## Summary\n\nSmall change.",
    }


def changed_files() -> list[dict]:
    return [
        {
            "filename": "src/foo.py",
            "status": "modified",
            "additions": 2,
            "deletions": 0,
            "patch": PATCH,
        }
    ]


def pr_comment_argv(path: str | None, position: int | None) -> list[str]:
    argv = ["pr", "comment", "7", "-R", "acme/widget", "--body-file", "-"]
    if path is not None:
        argv += ["--path", path]
    if position is not None:
        argv += ["--position", str(position)]
    return argv + ["--json", "--no-interactive"]


# --------------------------------------------------------------------------- #
# pr_review.py entry point
# --------------------------------------------------------------------------- #


def test_pr_review_extract_info_writes_review_context(fake, workdir):
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(pr_view_record()), "exit": 0},
            {
                "argv_contains": [COMMITS_ENDPOINT],
                "stdout": json.dumps(
                    [{"sha": "a" * 40, "commit": {"author": {"name": "Dev"}, "message": "feat: change things"}}]
                ),
                "exit": 0,
            },
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(changed_files()), "exit": 0},
            {
                "argv_contains": ["pr", "comments"],
                "stdout": json.dumps([{"id": 5, "path": "src/foo.py", "body": "existing"}]),
                "exit": 0,
            },
            {"argv_contains": [FILE_CONTENT_ENDPOINT], "stdout": encoded_content("print('hello')\n"), "exit": 0},
        ]
    )

    result = run_script(
        REVIEW_SCRIPT,
        ["--extract-info", "--pr", "7", *AI_MODEL_ARGS, *TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert fake.argv_of(["pr", "view"]) == ["pr", "view", "7", "-R", "acme/widget", "--json", "--no-interactive"]
    assert fake.argv_of(["pr", "comments"]) == [
        "pr",
        "comments",
        "7",
        "-R",
        "acme/widget",
        "--json",
        "--limit",
        "0",
        "--no-interactive",
    ]
    # The changed-file content is fetched at the PR head with the path
    # percent-encoded (safe="").
    assert fake.argv_of(["api", FILE_CONTENT_ENDPOINT]) == ["api", FILE_CONTENT_ENDPOINT, "--no-interactive"]

    context = json.loads((workdir / "tmp" / "widget_pr_7_info.json").read_text(encoding="utf-8"))
    assert context["pr"]["number"] == 7
    assert context["pr"]["head_sha"] == HEAD_SHA
    assert context["pr"]["stats"]["comments"] == 1
    assert context["pr"]["stats"]["inline_comments"] == 1
    assert context["pr"]["changed_files"][0]["content"].splitlines()[0] == "   1 | print('hello')"
    assert context["commits"][0]["sha"] == "a" * 40
    assert context["comments"][0]["id"] == 5


def test_pr_review_extract_info_no_comments_flag_skips_comment_fetch(fake, workdir):
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(pr_view_record()), "exit": 0},
            {"argv_contains": [COMMITS_ENDPOINT], "stdout": "[]", "exit": 0},
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(changed_files()), "exit": 0},
            {"argv_contains": [FILE_CONTENT_ENDPOINT], "stdout": encoded_content("x\n"), "exit": 0},
        ]
    )

    result = run_script(
        REVIEW_SCRIPT,
        ["--extract-info", "--no-comments", "--pr", "7", *AI_MODEL_ARGS, *TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not any("comments" in argv for argv in fake.argvs())
    context = json.loads((workdir / "tmp" / "widget_pr_7_info.json").read_text(encoding="utf-8"))
    assert context["pr"]["stats"]["comments"] == 0
    assert context["comments"] == []


def test_pr_review_extract_info_comment_failure_writes_no_context(fake, workdir):
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(pr_view_record()), "exit": 0},
            {"argv_contains": [COMMITS_ENDPOINT], "stdout": "[]", "exit": 0},
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(changed_files()), "exit": 0},
            {"argv_contains": ["pr", "comments"], "stderr": "transient failure", "exit": 1},
        ]
    )

    result = run_script(
        REVIEW_SCRIPT,
        ["--extract-info", "--pr", "7", *AI_MODEL_ARGS, *TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode != 0
    assert not (workdir / "tmp" / "widget_pr_7_info.json").exists()


def test_pr_review_submit_review_posts_summary_and_visible_inline_comments(fake, workdir):
    issues_file = workdir / "issues.json"
    issues_file.write_text(
        json.dumps(
            [
                {
                    "file": "src/foo.py",
                    "line": 2,
                    "type": "bug",
                    "severity": "warning",
                    "confidence": 100,
                    "title": "Wrong default",
                    "description": "The default breaks the contract.",
                    "context_code": "added line",
                    "fix": {"code": "added line = safe", "explanation": "Use the safe default."},
                },
                {
                    "file": "src/foo.py",
                    "line": 9,
                    "type": "bug",
                    "severity": "warning",
                    "confidence": 100,
                    "title": "Off-diff finding",
                    "description": "Line not visible on the new side.",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(pr_view_record()), "exit": 0},
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(changed_files()), "exit": 0},
            {"argv_contains": ["pr", "comment"], "stdout": '{"id": 321}', "exit": 0},
        ]
    )

    result = run_script(
        REVIEW_SCRIPT,
        ["--submit-review", str(issues_file), "--pr", "7", *AI_MODEL_ARGS, *TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "已提交摘要评论" in result.stdout
    # One visible inline comment + the summary; the off-diff finding never
    # reaches the platform.
    assert fake.mutation_argv() == [
        pr_comment_argv(None, None),
        pr_comment_argv("src/foo.py", 2),
    ]
    summary_stdin = fake.invocations()[fake.argvs().index(pr_comment_argv(None, None))]["stdin"]
    assert "代码审查摘要" in summary_stdin
    assert "generated by ai@test-model" in summary_stdin
    inline = fake.invocations()[fake.argvs().index(pr_comment_argv("src/foo.py", 2))]
    assert "Wrong default" in inline["stdin"]
    assert "Use the safe default." in inline["stdin"]
    # The off-diff finding is an explicit per-comment failure, and the posted
    # comment link carries the fake comment id.
    assert "提交 1/2 条评论" in result.stdout
    assert "not visible on the new side" in result.stdout
    assert "https://atomgit.com/acme/widget/pull/7#comment-321" in result.stdout


def test_pr_review_submit_review_auth_exit4_prints_login_guidance(fake, workdir):
    issues_file = workdir / "issues.json"
    issues_file.write_text(
        json.dumps([{"file": "src/foo.py", "line": 2, "title": "Wrong default", "confidence": 100}]),
        encoding="utf-8",
    )
    fake.configure([{"argv_contains": ["pr", "view"], "stderr": "not logged in to atomgit.com", "exit": 4}])

    result = run_script(
        REVIEW_SCRIPT,
        ["--submit-review", str(issues_file), "--pr", "7", *AI_MODEL_ARGS, *TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode != 0
    assert AUTH_LOGIN_GUIDANCE in result.stdout + result.stderr
    assert fake.mutation_argv() == []


def test_pr_review_submit_review_dry_run_makes_no_platform_calls(fake, workdir):
    issues_file = workdir / "issues.json"
    issues_file.write_text(
        json.dumps([{"file": "src/foo.py", "line": 2, "title": "Wrong default", "confidence": 100}]),
        encoding="utf-8",
    )
    fake.configure([])

    result = run_script(
        REVIEW_SCRIPT,
        ["--submit-review", str(issues_file), "--dry-run", "--pr", "7", *AI_MODEL_ARGS, *TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dry run" in result.stdout
    assert "Wrong default" in result.stdout
    assert fake.mutation_argv() == []
    # Only the local CLI identity probe ran; no platform command at all.
    assert fake.argvs() == [["version", "--no-interactive"]]


# --------------------------------------------------------------------------- #
# architecture_review.py entry point
# --------------------------------------------------------------------------- #


def test_arch_extract_info_writes_arch_context_with_readme(fake, workdir):
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(pr_view_record()), "exit": 0},
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(changed_files()), "exit": 0},
            {"argv_contains": [FILE_CONTENT_ENDPOINT], "stdout": encoded_content("print('hello')\n"), "exit": 0},
            {"argv_contains": [README_CONTENT_ENDPOINT], "stdout": encoded_content("# Package\n"), "exit": 0},
        ]
    )

    result = run_script(
        ARCH_SCRIPT,
        ["--extract-info", "--pr", "7", *AI_MODEL_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # The arch context reads no PR comments at all.
    assert not any("comments" in argv for argv in fake.argvs())
    context = json.loads((workdir / "tmp" / "widget_pr_7_arch_info.json").read_text(encoding="utf-8"))
    changed = context["pr"]["changed_files"]
    assert changed[0]["filename"] == "src/foo.py"
    assert changed[0]["content"].splitlines()[0] == "   1 | print('hello')"
    readme = changed[0]["readme_context"]
    assert readme["path"] == "src/README.md"
    assert readme["content"].splitlines()[0] == "   1 | # Package"


def test_arch_submit_review_posts_inline_and_pr_level_fallback(fake, workdir):
    issues_file = workdir / "arch_issues.json"
    issues_file.write_text(
        json.dumps(
            [
                {
                    "file": "src/foo.py",
                    "line": 1,
                    "title": "SSOT violation",
                    "description": "Configuration is hardcoded.",
                    "severity": "error",
                    "pillar": "ssot",
                    "fix": "Read the value from robot_config.",
                },
                {
                    "file": "docs/other.md",
                    "line": 3,
                    "title": "Off-diff architecture note",
                    "description": "File is not part of this change.",
                    "severity": "warning",
                    "pillar": "docs",
                    "fix": "Update the document.",
                },
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    fake.configure(
        [
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(changed_files()), "exit": 0},
            {"argv_contains": ["pr", "comment"], "stdout": '{"id": 55}', "exit": 0},
        ]
    )

    result = run_script(
        ARCH_SCRIPT,
        ["--submit-review", str(issues_file), "--pr", "7", *AI_MODEL_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # The visible inline finding goes out as one positioned comment; the
    # non-changed file degrades to one PR-level comment without --path.
    assert fake.mutation_argv() == [
        pr_comment_argv("src/foo.py", 1),
        pr_comment_argv(None, None),
    ]
    pr_level_stdin = fake.invocations()[fake.argvs().index(pr_comment_argv(None, None))]["stdin"]
    assert "Off-diff architecture note" in pr_level_stdin
    assert "SSOT violation" not in pr_level_stdin
    assert "降级为 PR 级评论" in result.stdout
    assert "提交 1/1 条行内评论" in result.stdout


def test_arch_submit_review_no_issues_posts_pass_summary(fake, workdir):
    issues_file = workdir / "arch_issues.json"
    issues_file.write_text("[]", encoding="utf-8")
    fake.configure([{"argv_contains": ["pr", "comment"], "stdout": '{"id": 56}', "exit": 0}])

    result = run_script(
        ARCH_SCRIPT,
        ["--submit-review", str(issues_file), "--pr", "7", *AI_MODEL_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert fake.mutation_argv() == [pr_comment_argv(None, None)]
    summary_stdin = fake.invocation_of(["pr", "comment"])["stdin"]
    assert "架构审查完成" in summary_stdin
    assert "未发现架构违背问题" in summary_stdin
    assert "generated by ai@test-model" in summary_stdin
    assert "提交架构审查通过评论" in result.stdout


def test_arch_extract_info_auth_exit4_prints_login_guidance(fake, workdir):
    fake.configure([{"argv_contains": ["pr", "view"], "stderr": "not logged in to atomgit.com", "exit": 4}])

    result = run_script(
        ARCH_SCRIPT,
        ["--extract-info", "--pr", "7", *AI_MODEL_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode != 0
    assert AUTH_LOGIN_GUIDANCE in result.stdout + result.stderr
    assert not (workdir / "tmp" / "widget_pr_7_arch_info.json").exists()
