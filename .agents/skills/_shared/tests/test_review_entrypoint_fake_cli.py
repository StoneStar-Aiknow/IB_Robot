"""Project workflow regressions; CLI responses are supplied by the shared fake."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
from pathlib import Path

import pytest
from entrypoint_fixture import EntrypointFakeCLI

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


def pr_view_record(*, head_repo: dict | str | None = None) -> dict:
    """The live ``pr view`` record; ``head_repo`` overrides the head repository
    (``"missing"`` removes it entirely) for current-source refusal shapes."""
    record = {
        "number": 7,
        "title": "feat: small change",
        "state": "open",
        "user": {"login": "dev"},
        # ``full_name`` proves the head repo identity (cross-fork gap fix:
        # content reads bind to head.repo at the complete head SHA); a
        # non-HTTPS web_url keeps any head-tree resolution offline and
        # unresolvable; the web_url→html_url mapping is applied by the helper.
        "head": {
            "ref": "feat/x",
            "sha": HEAD_SHA,
            "repo": {"full_name": "acme/widget", "web_url": "file:///tmp/not-a-repo"},
        },
        "base": {"ref": "master", "sha": "c" * 40, "repo": {}},
        "body": "## Summary\n\nSmall change.",
    }
    if head_repo == "missing":
        record["head"].pop("repo")
    elif head_repo is not None:
        record["head"]["repo"] = head_repo
    return record


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


def write_review_context(
    workdir: Path,
    *,
    head_sha: str = HEAD_SHA,
    head_source_sha: str | None = None,
    head_incomplete_reason: str | None = None,
    source_repo: str | None = None,
    content_error: str | None = None,
    target_owner: str = "acme",
    target_pr_number: int = 7,
    drop_target: bool = False,
    drop_head_source: bool = False,
    drop_changed_files: bool = False,
    arch: bool = False,
) -> Path:
    """Write the extract-info context that binds a submit to its review.

    ``content_error`` marks one changed file's head content unreadable;
    ``head_incomplete_reason`` marks the whole head source unprovable;
    ``drop_*`` remove structural keys entirely (malformed/legacy contexts);
    ``target_owner``/``target_pr_number``/``head_source_sha``/``source_repo``
    build identity-mismatch contexts (e.g. a same-named fork or a wrong source
    repository at the same SHA).
    """
    changed = {
        "filename": "src/foo.py",
        "status": "modified",
        "patch": PATCH,
        "content": None if content_error else "   1 | print('hello')",
    }
    if content_error:
        changed["content_error"] = content_error
    pr_block = {
        "number": 7,
        "head_sha": head_sha,
        "head_source": {
            "repo": None if head_incomplete_reason else (source_repo or "acme/widget"),
            "sha": None if head_incomplete_reason else (head_source_sha or head_sha),
            "incomplete_reason": head_incomplete_reason,
        },
        "changed_files": [changed],
    }
    if drop_head_source:
        pr_block.pop("head_source")
    if drop_changed_files:
        pr_block.pop("changed_files")
    context = {
        "fetch_time": "2026-09-29T00:00:00",
        "target": {
            "owner": target_owner,
            "repo": "widget",
            "pr_number": target_pr_number,
        },
        "pr": pr_block,
    }
    if drop_target:
        context.pop("target")
    output_dir = workdir / "tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    name = "widget_pr_7_arch_info.json" if arch else "widget_pr_7_info.json"
    path = output_dir / name
    path.write_text(json.dumps(context, ensure_ascii=False), encoding="utf-8")
    return path


# pr_review.py entry point


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
    # The context records its target identity so a submit can prove the
    # review was extracted for exactly this owner/repo/PR.
    assert context["target"] == {"owner": "acme", "repo": "widget", "pr_number": 7}
    assert context["pr"]["number"] == 7
    assert context["pr"]["head_sha"] == HEAD_SHA
    assert context["pr"]["stats"]["comments"] == 1
    assert context["pr"]["stats"]["inline_comments"] == 1
    assert context["pr"]["changed_files"][0]["content"].splitlines()[0] == "   1 | print('hello')"
    assert context["commits"][0]["sha"] == "a" * 40
    assert context["comments"][0]["id"] == 5


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


def test_pr_review_extract_info_unprovable_head_repo_marks_content_unavailable(fake, workdir):
    record = pr_view_record()
    # No full_name / owner.login+name: the cross-fork content source cannot be
    # proven, so content must be reported unavailable — never read from the
    # base repo at an ambiguous ref.
    record["head"]["repo"] = {"web_url": "file:///tmp/not-a-repo"}
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(record), "exit": 0},
            {"argv_contains": [COMMITS_ENDPOINT], "stdout": "[]", "exit": 0},
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(changed_files()), "exit": 0},
        ]
    )

    result = run_script(
        REVIEW_SCRIPT,
        ["--extract-info", "--no-comments", "--pr", "7", *AI_MODEL_ARGS, *TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    # No contents API call was attempted at all (commits/files pages are fine).
    assert not any("contents" in part for argv in fake.argvs() for part in argv)
    context = json.loads((workdir / "tmp" / "widget_pr_7_info.json").read_text(encoding="utf-8"))
    head_source = context["pr"]["head_source"]
    assert head_source["repo"] is None
    assert head_source["sha"] is None
    assert "cannot prove the cross-fork content source" in head_source["incomplete_reason"]
    changed = context["pr"]["changed_files"][0]
    assert changed["content"] is None
    assert changed["content_error"] == head_source["incomplete_reason"]


def test_pr_review_extract_info_incomplete_head_sha_marks_content_unavailable(fake, workdir):
    record = pr_view_record()
    # A short SHA is an ambiguous ref; content reads must not bind to it.
    record["head"]["sha"] = "b" * 7
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(record), "exit": 0},
            {"argv_contains": [COMMITS_ENDPOINT], "stdout": "[]", "exit": 0},
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(changed_files()), "exit": 0},
        ]
    )

    result = run_script(
        REVIEW_SCRIPT,
        ["--extract-info", "--no-comments", "--pr", "7", *AI_MODEL_ARGS, *TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not any("contents" in part for argv in fake.argvs() for part in argv)
    context = json.loads((workdir / "tmp" / "widget_pr_7_info.json").read_text(encoding="utf-8"))
    head_source = context["pr"]["head_source"]
    assert head_source["repo"] is None
    assert head_source["sha"] is None
    assert "not a complete 40-hex SHA" in head_source["incomplete_reason"]
    changed = context["pr"]["changed_files"][0]
    assert changed["content"] is None
    assert changed["content_error"] == head_source["incomplete_reason"]


def test_pr_review_submit_review_rejects_whole_batch_before_platform_writes(fake, workdir):
    issues_file = workdir / "issues.json"
    issues_file.write_text(
        json.dumps(
            [
                {
                    "file": "src/foo.py",
                    "line": 2,
                    "title": "Wrong default",
                    "description": "The default breaks the contract.",
                    "confidence": 100,
                },
                # line 0 is invalid: the whole batch is rejected before any
                # platform write, including the valid finding above.
                {
                    "file": "src/foo.py",
                    "line": 0,
                    "title": "Bad line",
                    "description": "Line zero is not a valid position.",
                    "confidence": 100,
                },
            ]
        ),
        encoding="utf-8",
    )
    fake.configure([])

    result = run_script(
        REVIEW_SCRIPT,
        ["--submit-review", str(issues_file), "--pr", "7", *AI_MODEL_ARGS, *TARGET_ARGS],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode != 0
    assert "rejected before any platform write" in result.stdout + result.stderr
    assert "item 1" in result.stdout + result.stderr
    # Zero mutations: the invalid batch never reaches the platform.
    assert fake.mutation_argv() == []


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
    write_review_context(workdir)
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


REFUSAL_SHAPES = [
    # (context kwargs, expected refusal reason, live pr-view head.repo override)
    pytest.param({"absent": True}, "no review context", None, id="missing-context"),
    pytest.param({"head_incomplete_reason": "head repo identity unreadable"}, "incomplete", None, id="incomplete-head"),
    pytest.param(
        {"content_error": "contents payload is not usable: truncated"},
        "unreadable file content",
        None,
        id="unreadable-content",
    ),
    pytest.param({"head_sha": "d" * 40}, "PR head moved", None, id="stale-revision"),
    pytest.param({"drop_target": True}, "predates target-identity binding", None, id="legacy-no-target"),
    pytest.param({"drop_head_source": True}, "missing the pr.head_source object", None, id="missing-head-source"),
    pytest.param({"drop_changed_files": True}, "missing the pr.changed_files list", None, id="missing-changed-files"),
    # Same repo name, same head SHA, different owner: the context filename
    # alone cannot distinguish the fork — only the recorded identity can.
    pytest.param({"target_owner": "other-org"}, "not interchangeable", None, id="same-repo-other-owner"),
    pytest.param({"target_pr_number": 8}, "was extracted for PR #8", None, id="wrong-pr-number"),
    pytest.param({"head_source_sha": "e" * 40}, "internally inconsistent", None, id="head-source-sha-mismatch"),
    # Round 3: the recorded source repository must match the *live* PR head
    # source — a wrong source repo at the same SHA must not publish.
    pytest.param(
        {"source_repo": "other-org/widget"},
        "against head repository other-org/widget",
        None,
        id="wrong-source-repo-same-sha",
    ),
    # The live PR record carries no head repository at all: refusing beats a
    # bare SHA comparison against an unprovable source.
    pytest.param({}, "current PR head content source is unprovable", "missing", id="current-head-repo-missing"),
    pytest.param({"source_repo": "not-a-slug"}, "not a valid owner/repo slug", None, id="invalid-source-slug"),
]


def write_issues_file(workdir: Path) -> Path:
    issues_file = workdir / "issues.json"
    issues_file.write_text(
        json.dumps(
            [
                {
                    "file": "src/foo.py",
                    "line": 2,
                    "title": "Wrong default",
                    "description": "The default breaks the contract.",
                    "confidence": 100,
                }
            ]
        ),
        encoding="utf-8",
    )
    return issues_file


@pytest.mark.parametrize("arch", [False, True], ids=["code", "architecture"])
@pytest.mark.parametrize("context_setup,expected_reason,current_head_repo", REFUSAL_SHAPES)
def test_submit_refuses_without_complete_bound_context(
    fake, workdir, context_setup, expected_reason, current_head_repo, arch
):
    issues_file = write_arch_issues_file(workdir) if arch else write_issues_file(workdir)
    if not context_setup.get("absent"):
        write_review_context(workdir, arch=arch, **context_setup)
    # A fully-working platform: if the binding validation wrongly passed, the
    # submission would publish and the zero-write assertion below would fail.
    fake.configure(
        [
            {
                "argv_contains": ["pr", "view"],
                "stdout": json.dumps(pr_view_record(head_repo=current_head_repo)),
                "exit": 0,
            },
            {"argv_contains": [FILES_ENDPOINT], "stdout": json.dumps(changed_files()), "exit": 0},
            {"argv_contains": ["pr", "comment"], "stdout": '{"id": 321}', "exit": 0},
        ]
    )

    result = run_script(
        ARCH_SCRIPT if arch else REVIEW_SCRIPT,
        ["--submit-review", str(issues_file), "--pr", "7", *AI_MODEL_ARGS, *([] if arch else TARGET_ARGS)],
        env=fake.env(),
        cwd=workdir,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert "拒绝提交审查结果" in output
    if arch and context_setup.get("absent"):
        expected_reason = "no architecture review context"
    assert expected_reason in output
    # Zero mutations: an unbound review never reaches the platform.
    assert fake.mutation_argv() == []


def test_pr_review_submit_review_accepts_digit_string_pr_number_in_context(fake, workdir):
    # Authenticated evidence (task 1.4): CLI payloads mix int and digit-string
    # numbers (PR numbers are ints, issue numbers are strings). A context
    # whose recorded target.pr_number is the digit string "7" is the same PR —
    # normalization must not falsely reject it.
    issues_file = write_issues_file(workdir)
    write_review_context(workdir, target_pr_number="7")
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
    # The bound submission published the summary comment.
    assert fake.mutation_argv() != []


def test_pr_review_submit_review_dry_run_makes_no_platform_calls(fake, workdir):
    issues_file = workdir / "issues.json"
    issues_file.write_text(
        json.dumps(
            [
                {
                    "file": "src/foo.py",
                    "line": 2,
                    "title": "Wrong default",
                    "description": "The default breaks the contract.",
                    "confidence": 100,
                }
            ]
        ),
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


# architecture_review.py entry point


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
    assert context["target"] == {"owner": "acme", "repo": "widget", "pr_number": 7}
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
    write_review_context(workdir, arch=True)
    fake.configure(
        [
            {"argv_contains": ["pr", "view"], "stdout": json.dumps(pr_view_record()), "exit": 0},
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


def write_arch_issues_file(workdir: Path) -> Path:
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
                }
            ]
        ),
        encoding="utf-8",
    )
    return issues_file
