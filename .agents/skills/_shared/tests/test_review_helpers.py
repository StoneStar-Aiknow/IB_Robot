"""Project workflow regressions; CLI responses are supplied by the shared fake."""

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


# diff positioning (task 4.3): added / modified / renamed / deleted


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


# PR record and file content reads (task 4.1)


def test_fetch_pr_record_rejects_non_object_payload(monkeypatch):
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, **kwargs: cli_result([1, 2]))
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_pr_record("gitcode", TARGET, 7)


def test_fetch_file_content_missing_content_proven_empty_by_size(monkeypatch):
    # Missing content is a genuinely empty file only when size == 0 proves it.
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, **kwargs: cli_result({"size": 0}))
    assert review_helpers.fetch_file_content("gitcode", TARGET, "a", "b") == ""


def test_fetch_file_content_missing_content_without_size_proof_fails(monkeypatch):
    # Withheld content must never be reported as a fake empty file.
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, **kwargs: cli_result({}))
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_file_content("gitcode", TARGET, "a", "b")


def test_fetch_file_content_missing_content_with_nonzero_size_fails(monkeypatch):
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, **kwargs: cli_result({"size": 42}))
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_file_content("gitcode", TARGET, "a", "b")


def test_fetch_file_content_truncated_payload_fails(monkeypatch):
    monkeypatch.setattr(
        atomgit_cli,
        "run_json",
        lambda args, **kwargs: cli_result({"content": base64.b64encode(b"x").decode(), "truncated": True}),
    )
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_file_content("gitcode", TARGET, "a", "b")


def test_fetch_file_content_unsupported_encoding_fails(monkeypatch):
    payload = {"encoding": "utf-8", "content": "plain text"}
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, **kwargs: cli_result(payload))
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_file_content("gitcode", TARGET, "a", "b")


def test_fetch_file_content_strict_base64_rejects_invalid_chars(monkeypatch):
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, **kwargs: cli_result({"content": "not!base64!!"}))
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_file_content("gitcode", TARGET, "a", "b")


def test_fetch_file_content_size_mismatch_fails(monkeypatch):
    payload = {"content": base64.b64encode(b"hello").decode(), "size": 3}
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, **kwargs: cli_result(payload))
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_file_content("gitcode", TARGET, "a", "b")


def test_fetch_file_content_non_utf8_bytes_fail(monkeypatch):
    payload = {"content": base64.b64encode(b"\xff\xfe\x00").decode()}
    monkeypatch.setattr(atomgit_cli, "run_json", lambda args, **kwargs: cli_result(payload))
    with pytest.raises(atomgit_cli.CLIJSONError):
        review_helpers.fetch_file_content("gitcode", TARGET, "a", "b")


# PR head source resolution (review gap 1): cross-fork content identity


def pr_record_with_head(head) -> dict:
    return {"number": 7, "head": head}


def test_resolve_head_source_full_name_proves_cross_fork_identity():
    pr = pr_record_with_head({"ref": "feat/x", "sha": "b" * 40, "repo": {"full_name": "acme-dev/widget"}})
    head_source, reason = review_helpers.resolve_head_source(pr)
    assert reason is None
    assert head_source is not None
    assert head_source.target.owner == "acme-dev"
    assert head_source.target.repo == "widget"
    assert head_source.sha == "b" * 40
    assert head_source.repo_slug == "acme-dev/widget"


def test_resolve_head_source_owner_login_and_name_prove_identity():
    pr = pr_record_with_head(
        {"ref": "feat/x", "sha": "b" * 40, "repo": {"owner": {"login": "acme-dev"}, "name": "widget"}}
    )
    head_source, reason = review_helpers.resolve_head_source(pr)
    assert reason is None
    assert head_source.target.slug == "acme-dev/widget"


def test_resolve_head_source_missing_head_branch_is_unprovable():
    head_source, reason = review_helpers.resolve_head_source({"number": 7})
    assert head_source is None
    assert "no head branch metadata" in reason


def test_resolve_head_source_short_sha_is_refused():
    pr = pr_record_with_head({"ref": "feat/x", "sha": "b" * 7, "repo": {"full_name": "acme-dev/widget"}})
    head_source, reason = review_helpers.resolve_head_source(pr)
    assert head_source is None
    assert "not a complete 40-hex SHA" in reason


def test_resolve_head_source_missing_sha_is_refused():
    pr = pr_record_with_head({"ref": "feat/x", "repo": {"full_name": "acme-dev/widget"}})
    head_source, reason = review_helpers.resolve_head_source(pr)
    assert head_source is None
    assert "not a complete 40-hex SHA" in reason


def test_resolve_head_source_unreadable_repo_identity_is_unprovable():
    pr = pr_record_with_head({"ref": "feat/x", "sha": "b" * 40, "repo": {"web_url": "https://atomgit.com/x"}})
    head_source, reason = review_helpers.resolve_head_source(pr)
    assert head_source is None
    assert "cannot prove the cross-fork content source" in reason


def test_resolve_head_source_missing_repo_metadata_is_unprovable():
    pr = pr_record_with_head({"ref": "feat/x", "sha": "b" * 40})
    head_source, reason = review_helpers.resolve_head_source(pr)
    assert head_source is None
    assert "no head repository metadata" in reason


# paginated collections (task 4.1): dedup + hard failure on partial data


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


@pytest.mark.parametrize("total", [127, 128, None, "127"])
def test_repeated_pr_files_need_matching_independent_total(monkeypatch, total):
    full = [{"filename": f"f{i}.py"} for i in range(127)]
    requests = []

    def fake_run_json(args, **kwargs):
        requests.append(list(args))
        if args[:2] == ["pr", "view"]:
            return cli_result({"number": 7, "changed_files": total})
        return cli_result(full)

    monkeypatch.setattr(atomgit_cli, "run_json", fake_run_json)
    if type(total) is int and total == 127:
        assert review_helpers.fetch_pr_files("gitcode", TARGET, 7) == full
    else:
        with pytest.raises(atomgit_cli.CLIExecutionError, match="cannot prove completeness"):
            review_helpers.fetch_pr_files("gitcode", TARGET, 7)
    assert len(requests) == 3


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


# inline comment submission (tasks 4.2/4.3): partial failure, no retry


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
