"""Unit behavior tests for the review CLI helpers (tasks 4.1-4.3).

The entrypoint tests in ``test_review_entrypoint_fake_cli.py`` drive the full
subprocess path; these tests pin the helper contracts the entrypoints build
on: new-side diff positioning for added/modified/renamed/deleted files, the
line-numbering port, ``web_url``→``html_url`` mapping, contents-API
percent-encoding, pagination hard-failure/dedup rules, and the per-comment
partial-failure semantics of ``submit_inline_comments`` (missing position,
nonzero exit, raised timeout — each attempted exactly once, never retried,
never aborting the batch). Platform access is faked by monkeypatching the
shared boundary; no real CLI, no credentials.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

import atomgit_cli
import review_helpers

TARGET = atomgit_cli.RepoTarget("acme", "widget", "test", None)


def cli_result(data, *, returncode: int = 0) -> atomgit_cli.CLIResult:
    return atomgit_cli.CLIResult(
        argv=["fake"],
        returncode=returncode,
        stdout=json.dumps(data) if data is not None else "",
        stderr="",
        data=data,
    )


# --------------------------------------------------------------------------- #
# diff positioning (task 4.3): added / modified / renamed / deleted
# --------------------------------------------------------------------------- #


def test_diff_new_side_lines_added_file():
    patch = "@@ -0,0 +1,3 @@\n+a\n+b\n+c"
    assert review_helpers.diff_new_side_lines(patch) == {1, 2, 3}


def test_diff_new_side_lines_modified_hunk_skips_deleted_lines():
    patch = "@@ -1,4 +1,4 @@\n keep1\n-old\n+new\n keep2\n"
    # New side: keep1=1, new=2, keep2=3; the deleted line has no new-side number.
    assert review_helpers.diff_new_side_lines(patch) == {1, 2, 3}


def test_diff_new_side_lines_deleted_file_yields_nothing():
    patch = "@@ -1,2 +0,0 @@\n-gone1\n-gone2"
    assert review_helpers.diff_new_side_lines(patch) == set()


def test_diff_new_side_lines_renamed_file_ignores_rename_headers():
    patch = (
        "similarity index 90%\nrename from src/old.py\nrename to src/new.py\n@@ -5,3 +5,4 @@\n ctx\n+renamed addition\n"
    )
    # Rename headers before the first hunk are not diff lines; numbering
    # follows the new-side start of the hunk.
    assert review_helpers.diff_new_side_lines(patch) == {5, 6}


def test_diff_new_side_lines_multi_hunk_numbering():
    patch = "@@ -1,2 +1,2 @@\n a\n b\n@@ -10,2 +10,2 @@\n c\n+d\n"
    assert review_helpers.diff_new_side_lines(patch) == {1, 2, 10, 11}


def test_diff_new_side_lines_empty_patch():
    assert review_helpers.diff_new_side_lines("") == set()
    assert review_helpers.diff_new_side_lines("no hunks here\n") == set()


def test_add_line_numbers_prefixes_one_based_numbers():
    assert review_helpers.add_line_numbers("a\nb") == "   1 | a\n   2 | b"


def test_add_line_numbers_empty():
    assert review_helpers.add_line_numbers("") == ""


# --------------------------------------------------------------------------- #
# PR record and file content reads (task 4.1)
# --------------------------------------------------------------------------- #


def test_fetch_pr_record_maps_web_url_to_html_url(monkeypatch):
    record = {
        "number": 7,
        "head": {"repo": {"web_url": "https://atomgit.com/acme/widget"}},
        "base": {"repo": {"web_url": "https://atomgit.com/acme/base"}},
    }
    captured = {}

    def fake_run_json(args, **kwargs):
        captured["argv"] = args
        return cli_result(record)

    monkeypatch.setattr(atomgit_cli, "run_json", fake_run_json)
    pr = review_helpers.fetch_pr_record("gitcode", TARGET, 7)

    assert captured["argv"] == ["pr", "view", "7", "-R", "acme/widget", "--json"]
    assert pr["head"]["repo"]["html_url"] == "https://atomgit.com/acme/widget"
    assert pr["base"]["repo"]["html_url"] == "https://atomgit.com/acme/base"


def test_fetch_pr_record_rejects_non_object_payload(monkeypatch):
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, **kwargs: cli_result([1, 2]))
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_pr_record("gitcode", TARGET, 7)


def test_fetch_file_content_percent_encodes_path_and_ref(monkeypatch):
    captured = {}

    def fake_run_json(args, **kwargs):
        captured["argv"] = args
        return cli_result({"content": base64.b64encode(b"print('hi')\n").decode("ascii")})

    monkeypatch.setattr(atomgit_cli, "run_json", fake_run_json)
    text = review_helpers.fetch_file_content("gitcode", TARGET, "src/foo.py", "bbbb")

    assert text == "print('hi')\n"
    assert captured["argv"] == ["api", "repos/acme/widget/contents/src%2Ffoo.py?ref=bbbb"]


def test_fetch_file_content_missing_content_decodes_empty(monkeypatch):
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, **kwargs: cli_result({}))
    assert review_helpers.fetch_file_content("gitcode", TARGET, "a", "b") == ""


# --------------------------------------------------------------------------- #
# paginated collections (task 4.1): dedup + hard failure on partial data
# --------------------------------------------------------------------------- #


def paginated(items, *, truncated: bool = False, stop_reason: str = "complete"):
    return atomgit_cli.PaginatedResult(
        items=items,
        pages=1,
        truncated=truncated,
        stop_reason=stop_reason,
        last_result=cli_result([]),
    )


def test_fetch_api_pages_dedups_by_key_preserving_order(monkeypatch):
    items = [{"filename": "a.py"}, {"filename": "b.py"}, {"filename": "a.py"}]
    monkeypatch.setattr(atomgit_cli, "run_paginated", lambda build, **kwargs: paginated(items))
    assert review_helpers.fetch_pr_files("gitcode", TARGET, 7) == [{"filename": "a.py"}, {"filename": "b.py"}]


def test_fetch_api_pages_truncated_is_a_hard_failure(monkeypatch):
    monkeypatch.setattr(
        atomgit_cli,
        "run_paginated",
        lambda build, **kwargs: paginated([{"filename": "a.py"}], truncated=True, stop_reason="failed-page"),
    )
    with pytest.raises(atomgit_cli.CLIExecutionError, match="partial data"):
        review_helpers.fetch_pr_files("gitcode", TARGET, 7)


def test_fetch_api_pages_rejects_malformed_entries(monkeypatch):
    monkeypatch.setattr(atomgit_cli, "run_paginated", lambda build, **kwargs: paginated(["not-an-object"]))
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_pr_files("gitcode", TARGET, 7)

    monkeypatch.setattr(atomgit_cli, "run_paginated", lambda build, **kwargs: paginated([{"status": "modified"}]))
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_pr_files("gitcode", TARGET, 7)


# --------------------------------------------------------------------------- #
# inline comment submission (tasks 4.2/4.3): partial failure, no retry
# --------------------------------------------------------------------------- #


def test_submit_inline_comments_missing_position_fails_without_platform_call(monkeypatch):
    def unexpected(args, **kwargs):
        raise AssertionError("no platform call expected for a positionless comment")

    monkeypatch.setattr(atomgit_cli, "run_json", unexpected)
    results = review_helpers.submit_inline_comments(
        "gitcode", TARGET, 7, [{"path": "src/foo.py", "line": 9, "body": "off diff", "position": None}]
    )

    assert results == [
        {
            "success": False,
            "comment": {"path": "src/foo.py", "line": 9, "body": "off diff", "position": None},
            "error": results[0]["error"],
            "comment_url": None,
        }
    ]
    assert "not visible on the new side" in results[0]["error"]


def test_submit_inline_comments_success_and_partial_failure(monkeypatch):
    responses = [
        cli_result({"id": 11}),
        atomgit_cli.CLIResult(argv=["pr", "comment"], returncode=1, stdout="", stderr="boom"),
    ]
    calls = []

    def fake_run_json(args, **kwargs):
        calls.append(args)
        return responses[len(calls) - 1]

    monkeypatch.setattr(atomgit_cli, "run_json", fake_run_json)
    comments = [
        {"path": "src/a.py", "line": 2, "body": "ok", "position": 2},
        {"path": "src/b.py", "line": 3, "body": "bad", "position": 3},
    ]
    results = review_helpers.submit_inline_comments("gitcode", TARGET, 7, comments)

    assert [result["success"] for result in results] == [True, False]
    assert results[0]["comment_url"] == "https://atomgit.com/acme/widget/pull/7#comment-11"
    assert "exit code 1" in results[1]["error"]
    assert results[1]["comment_url"] is None
    # Each comment attempted exactly once — no retry of the failed one.
    assert len(calls) == 2


def test_submit_inline_comments_timeout_is_a_per_comment_failure(monkeypatch):
    calls = []

    def fake_run_json(args, **kwargs):
        calls.append(args)
        if len(calls) == 1:
            raise atomgit_cli.CLITimeoutError("AtomGit CLI invocation timed out after 30s", argv=args, timeout=30.0)
        return cli_result({"id": 12})

    monkeypatch.setattr(atomgit_cli, "run_json", fake_run_json)
    comments = [
        {"path": "src/a.py", "line": 1, "body": "slow", "position": 1},
        {"path": "src/b.py", "line": 2, "body": "fast", "position": 2},
    ]
    results = review_helpers.submit_inline_comments("gitcode", TARGET, 7, comments)

    assert [result["success"] for result in results] == [False, True]
    assert "timed out" in results[0]["error"]
    assert results[1]["comment_url"] == "https://atomgit.com/acme/widget/pull/7#comment-12"
    # The batch continued past the timeout; the second comment was still attempted.
    assert len(calls) == 2


def test_submit_inline_comments_posts_position_as_new_side_line(monkeypatch):
    captured = {}

    def fake_run_json(args, **kwargs):
        captured.update(argv=args, stdin=kwargs.get("stdin_text"))
        return cli_result({"id": 13})

    monkeypatch.setattr(atomgit_cli, "run_json", fake_run_json)
    review_helpers.submit_inline_comments(
        "gitcode", TARGET, 7, [{"path": "src/a.py", "line": 2, "body": "body text", "position": 2}]
    )

    assert captured["argv"] == [
        "pr",
        "comment",
        "7",
        "-R",
        "acme/widget",
        "--body-file",
        "-",
        "--path",
        "src/a.py",
        "--position",
        "2",
        "--json",
    ]
    assert captured["stdin"] == "body text"
