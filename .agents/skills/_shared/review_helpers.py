"""PR review CLI helpers for the review and architecture-review skills (tasks 4.1-4.3).

Both review skills reach the platform exclusively through the shared CLI
boundary (``atomgit_cli``); this module owns the review-specific argv
building that used to live in the retired ``atomgit_sdk`` client. Every
command below is bound to the v0.14.1 capability-matrix evidence:

- PR record: ``pr view <n> -R <slug> --json``. CLI Repository objects expose
  the web URL under ``web_url`` (Go json tag) while the verification gate
  consumes ``html_url``; the mapping is applied here, mirroring
  ``atomgit-pr/scripts/pr_management.py``.
- Commits / files: ``api repos/{owner}/{repo}/pulls/{n}/{commits,files}``
  traversed with the boundary's page loop (per-file patches are not part of
  ``pr view`` output). A failed or capped page is a hard failure: review
  context must never be written from a partial collection.
- Comments read: ``pr comments <n> -R <slug> --json --limit 0`` (0 = all).
- File content at a ref: ``api repos/{owner}/{repo}/contents/{path}?ref=``
  with base64 ``content`` decoding (SDK-compatible wrapper semantics, matrix
  row A17 remote path).
- Comment writes: ``pr comment <n> -R <slug> --body-file - [--path --position]
  --json``. ``--position`` is the new-side file line number, NOT a diff-hunk
  offset (pr-comment help evidence), so inline comments pass the review line
  number directly. Findings on lines not visible on the new side of the diff
  are reported as explicit per-comment failures without a platform call
  (task 4.3); batch submission keeps the SDK's partial-failure semantics and
  never retries (design.md D3/D5).

The diff utilities are local ports of the retired SDK helpers:
``add_line_numbers`` is byte-identical, and ``diff_new_side_lines`` re-derives
the SDK's ``calculate_diff_position`` hunk walk under the CLI's position
semantics. They are pure functions with no platform access.
"""

from __future__ import annotations

import base64
import re
from urllib.parse import quote

import atomgit_cli

_HUNK_HEADER_RE = re.compile(r"^@@\s+-\d+,?\d*\s+\+(\d+),?\d*\s+@@")


def pr_url(target: atomgit_cli.RepoTarget, pr_number: int) -> str:
    """Construct the web URL locally (registered web host only; no platform call)."""
    return f"https://{atomgit_cli.DEFAULT_WEB_HOST}/{target.owner}/{target.repo}/pull/{pr_number}"


def fetch_pr_record(cli: str, target: atomgit_cli.RepoTarget, pr_number: int) -> dict:
    """Read one PR via ``pr view --json`` into the project-shaped record."""
    result = atomgit_cli.run_json(
        ["pr", "view", str(pr_number), "-R", target.slug, "--json"],
        cli=cli,
    )
    pr = result.require_data()
    if not isinstance(pr, dict):
        raise atomgit_cli.CLIJSONError(
            f"pr view returned {type(pr).__name__} instead of a JSON object",
            argv=result.argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    for branch_key in ("head", "base"):
        branch = pr.get(branch_key)
        if not isinstance(branch, dict):
            continue
        repo = branch.get("repo")
        if isinstance(repo, dict) and not repo.get("html_url") and repo.get("web_url"):
            repo["html_url"] = repo["web_url"]
    return pr


def fetch_api_pages(cli: str, endpoint: str, *, what: str, key: str) -> list[dict]:
    """Traverse every page of a v5 GET endpoint through ``gitcode api``.

    The ``api`` command has no ``--paginate`` flag (capability matrix), so
    the shared boundary's page loop is used. Window-shift duplicates are
    dropped by ``key`` while preserving first-seen order. Incomplete
    collections (failed page or page cap) are hard failures: a review context
    computed over partial files/commits could silently skip gate-relevant
    changes.
    """

    def build_page_args(page: int) -> list[str]:
        separator = "&" if "?" in endpoint else "?"
        return ["api", f"{endpoint}{separator}page={page}&per_page=100"]

    result = atomgit_cli.run_paginated(build_page_args, cli=cli)
    if result.truncated:
        last = result.last_result
        raise atomgit_cli.CLIExecutionError(
            f"could not read the complete {what} collection (stop reason: {result.stop_reason}); "
            "refusing to proceed on partial data",
            argv=last.argv if last else None,
            returncode=last.returncode if last else None,
            stdout=last.stdout if last else "",
            stderr=last.stderr if last else "",
        )
    items: list[dict] = []
    seen: set = set()
    for entry in result.items:
        if not isinstance(entry, dict):
            last = result.last_result
            raise atomgit_cli.CLIJSONError(
                f"{what} page contained a non-object entry; refusing to proceed on partial data",
                argv=last.argv if last else None,
                returncode=last.returncode if last else None,
                stdout=last.stdout if last else "",
                stderr=last.stderr if last else "",
            )
        identity = entry.get(key)
        if not identity:
            raise atomgit_cli.CLIJSONError(
                f"{what} page contained an entry without a {key!r} field; refusing to proceed on partial data",
                argv=result.last_result.argv if result.last_result else None,
                returncode=result.last_result.returncode if result.last_result else None,
                stdout=result.last_result.stdout if result.last_result else "",
                stderr=result.last_result.stderr if result.last_result else "",
            )
        if identity in seen:
            continue
        seen.add(identity)
        items.append(entry)
    return items


def fetch_pr_files(cli: str, target: atomgit_cli.RepoTarget, pr_number: int) -> list[dict]:
    return fetch_api_pages(
        cli,
        f"repos/{target.owner}/{target.repo}/pulls/{pr_number}/files",
        what="PR files",
        key="filename",
    )


def fetch_pr_commits(cli: str, target: atomgit_cli.RepoTarget, pr_number: int) -> list[dict]:
    return fetch_api_pages(
        cli,
        f"repos/{target.owner}/{target.repo}/pulls/{pr_number}/commits",
        what="PR commits",
        key="sha",
    )


def fetch_pr_comments(cli: str, target: atomgit_cli.RepoTarget, pr_number: int) -> list[dict]:
    """All PR comments via ``pr comments --limit 0`` (help: 0 = all)."""
    result = atomgit_cli.run_json(
        ["pr", "comments", str(pr_number), "-R", target.slug, "--json", "--limit", "0"],
        cli=cli,
    )
    comments = result.require_data()
    if not isinstance(comments, list):
        raise atomgit_cli.CLIJSONError(
            f"pr comments returned {type(comments).__name__} instead of a JSON array",
            argv=result.argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    non_object_entries = [comment for comment in comments if not isinstance(comment, dict)]
    if non_object_entries:
        raise atomgit_cli.CLIJSONError(
            f"pr comments array contains {len(non_object_entries)} non-object entries; "
            "refusing to write a partial PR context",
            argv=result.argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    return comments


def fetch_file_content(cli: str, target: atomgit_cli.RepoTarget, file_path: str, ref: str) -> str:
    """Read one file at ``ref`` via the contents API and decode the base64 wrapper.

    Byte-compatible with the retired SDK ``get_file_content``: the path is
    percent-encoded with ``safe=""`` and an empty/missing ``content`` field
    decodes to an empty string.
    """
    endpoint = f"repos/{target.owner}/{target.repo}/contents/{quote(file_path, safe='')}?ref={quote(str(ref), safe='')}"
    result = atomgit_cli.run_json(["api", endpoint], cli=cli)
    data = result.require_data()
    if not isinstance(data, dict):
        raise atomgit_cli.CLIJSONError(
            f"contents endpoint returned {type(data).__name__} instead of a JSON object",
            argv=result.argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    content = data.get("content")
    if not content:
        return ""
    return base64.b64decode(content).decode("utf-8")


def submit_pr_comment(cli: str, target: atomgit_cli.RepoTarget, pr_number: int, body: str) -> atomgit_cli.CLIResult:
    """Post one PR-level comment through stdin (``--body-file -``); raises on failure."""
    result = atomgit_cli.run_json(
        ["pr", "comment", str(pr_number), "-R", target.slug, "--body-file", "-", "--json"],
        cli=cli,
        stdin_text=body,
    )
    failure = result.failure()
    if failure is not None:
        raise failure
    return result


def submit_inline_comments(
    cli: str,
    target: atomgit_cli.RepoTarget,
    pr_number: int,
    comments: list[dict],
) -> list[dict]:
    """Submit inline findings as one ``pr comment`` per comment (task 4.2/4.3).

    ``--position`` is the new-side file line number, so the formatter-attached
    position is posted verbatim. Comments without a position (line not
    visible on the new side of the diff) are reported as explicit failures
    without a platform call. Every comment is attempted exactly once; a
    per-comment failure — including a timeout, whose message carries the
    boundary's reconciliation guidance — never aborts the batch (SDK
    partial-failure semantics).
    """
    results = []
    for comment in comments:
        position = comment.get("position")
        if position is None:
            results.append(
                {
                    "success": False,
                    "comment": comment,
                    "error": (
                        f"line {comment.get('line')} of {comment.get('path')} is not visible on the new "
                        "side of the PR diff; the AtomGit CLI only supports inline comments on "
                        "new-side diff lines"
                    ),
                    "comment_url": None,
                }
            )
            continue
        try:
            result = atomgit_cli.run_json(
                [
                    "pr",
                    "comment",
                    str(pr_number),
                    "-R",
                    target.slug,
                    "--body-file",
                    "-",
                    "--path",
                    str(comment.get("path") or ""),
                    "--position",
                    str(position),
                    "--json",
                ],
                cli=cli,
                stdin_text=comment.get("body") or "",
            )
            failure = result.failure()
        except atomgit_cli.CLIError as error:
            # Raised boundary errors (e.g. CLITimeoutError) are per-comment
            # failures too; the batch continues with the rest (SDK semantics).
            failure = error
        if failure is not None:
            results.append({"success": False, "comment": comment, "error": str(failure), "comment_url": None})
            continue
        data = result.data if isinstance(result.data, dict) else {}
        comment_id = data.get("id")
        comment_url = pr_url(target, pr_number) + (f"#comment-{comment_id}" if comment_id else "")
        results.append({"success": True, "comment": comment, "result": data, "comment_url": comment_url})
    return results


def add_line_numbers(content: str) -> str:
    """Prefix every line with its 1-based number (``   1 | …``), SDK-identical port."""
    if not content:
        return ""
    lines = content.split("\n")
    return "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))


def diff_new_side_lines(patch: str) -> set[int]:
    """New-side file line numbers visible in the patch hunks.

    Port of the retired SDK ``calculate_diff_position`` hunk walk under the
    CLI's position semantics: ``pr comment --position`` takes the new-side
    file line number, and only lines shown on the new side of the diff
    (added or context lines) can carry inline comments. Deleted-only lines
    and lines outside the hunks are not visible and yield no entry.
    """
    if not patch:
        return set()
    visible: set[int] = set()
    current_new_line = 0
    in_hunk = False
    for line in patch.split("\n"):
        hunk_match = _HUNK_HEADER_RE.match(line)
        if hunk_match:
            in_hunk = True
            current_new_line = int(hunk_match.group(1)) - 1
            continue
        if not in_hunk:
            continue
        first_char = line[0] if line else ""
        # Tuple membership, not substring: ``"" in "+ "`` is True, and the
        # trailing empty string of a newline-terminated patch would otherwise
        # mark the line after the last hunk line as commentable.
        if first_char in ("+", " "):
            current_new_line += 1
            visible.add(current_new_line)
    return visible
