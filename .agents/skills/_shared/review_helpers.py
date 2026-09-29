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
  with base64 ``content`` decoding (matrix row A17 remote path). Integrity is
  proven from the payload: truncation markers, missing content on a nonzero
  size, non-base64 encodings, strict-decode failures and byte-length
  mismatches are hard failures — an unavailable file is never reported as an
  empty one, and errors are never stringified into the file body (design.md
  D3: incomplete reads never masquerade as data).
- PR head content source: ``resolve_head_source`` proves where the head
  content lives from the PR record itself (CLI v0.14.1 ``PRBranch``/``Repository``
  JSON: ``head.sha`` + ``head.repo.full_name`` or ``owner.login``+``name``).
  Cross-fork PRs keep head content in the *source* repository, so content
  reads bind to the head repo at the complete head SHA; unverifiable metadata
  yields ``(None, reason)`` and callers mark the content unavailable instead
  of silently reading from the base repository at an ambiguous ref.
- Comment writes: ``pr comment <n> -R <slug> --body-file - [--path --position]
  --json``. ``--position`` is the new-side file line number, NOT a diff-hunk
  offset (pr-comment help evidence), so inline comments pass the review line
  number directly. Findings on lines not visible on the new side of the diff
  are reported as explicit per-comment failures without a platform call
  (task 4.3); batch submission keeps the SDK's partial-failure semantics and
  never retries (design.md D3/D5).
- Manual submit binding: ``validate_extract_context_file`` refuses a findings
  submission unless its extract-info context carries a matching target
  identity (owner/repo/pr_number), a structurally complete head source object
  (valid ``owner/repo`` slug + 40/64-hex SHA agreeing with the recorded head
  SHA), complete per-file head content, and — via ``resolve_head_source`` on
  the live PR record — a provable current head source whose SHA and repository
  slug both match the recorded context (a missing current head repo or a
  wrong source repo at the same SHA is a zero-write refusal; task 4 gap fix,
  round 3).

The diff utilities are local ports of the retired SDK helpers:
``add_line_numbers`` is byte-identical, and ``diff_new_side_lines`` re-derives
the SDK's ``calculate_diff_position`` hunk walk under the CLI's position
semantics. They are pure functions with no platform access.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import atomgit_cli

_HUNK_HEADER_RE = re.compile(r"^@@\s+-\d+,?\d*\s+\+(\d+),?\d*\s+@@")
# Complete Git SHA-1 object name (40 hex). Short SHAs are ambiguous refs:
# binding PR head content to one cannot prove which commit is being read.
_FULL_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
# Complete object names accepted in recorded review contexts: SHA-1 (40 hex)
# or SHA-256 (64 hex), case-insensitive.
_CONTEXT_SHA_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")


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


@dataclass(frozen=True)
class HeadSource:
    """Proven PR head content source: head repository identity + full head SHA.

    Cross-fork PRs keep their head content in the *source* repository; reading
    the base repo at the head SHA works only when the platform happens to make
    fork objects reachable and silently reads wrong content when a short/branch
    ref resolves on both sides. The target therefore always comes from the PR
    record's own ``head.repo`` identity, never from the reviewed base target.
    """

    target: atomgit_cli.RepoTarget
    sha: str

    @property
    def repo_slug(self) -> str:
        return self.target.slug


def _head_repo_identity(repo: dict) -> tuple[str, str] | None:
    """Owner/repo identity from a CLI Repository object, without guessing.

    CLI v0.14.1 ``Repository`` JSON (api/queries_repo.go): ``full_name`` plus
    ``owner`` (User object) and ``name``. Either spelling is accepted; anything
    else yields ``None`` so unprovable identities stay unprovable.
    """
    full_name = repo.get("full_name")
    if isinstance(full_name, str) and full_name.strip():
        owner, _, name = full_name.strip().strip("/").partition("/")
        if owner and name:
            return owner, name
    owner = repo.get("owner")
    if isinstance(owner, dict):
        login = owner.get("login")
    elif isinstance(owner, str):
        login = owner
    else:
        login = None
    name = repo.get("name") or repo.get("path")
    if isinstance(login, str) and login.strip() and isinstance(name, str) and name.strip():
        return login.strip(), name.strip()
    return None


def resolve_head_source(pr: dict) -> tuple[HeadSource | None, str | None]:
    """Prove where PR head content lives; incomplete metadata is reported, never guessed.

    Returns ``(head_source, incomplete_reason)`` with exactly one ``None``.
    A missing head branch, a missing/non-complete SHA (the old code silently
    fell back to ``HEAD``, which is not the PR head once the PR updates), or a
    head repo whose identity cannot be derived from the record all yield
    ``(None, reason)``; the caller must mark head content unavailable instead
    of reading from the base target.
    """
    head = pr.get("head")
    if not isinstance(head, dict):
        return None, "pr view returned no head branch metadata; cannot locate PR head content"
    sha = head.get("sha")
    if not isinstance(sha, str) or not _FULL_SHA_RE.fullmatch(sha.strip()):
        return None, (
            "PR head SHA is missing or not a complete 40-hex SHA; refusing to bind content reads to an ambiguous ref"
        )
    repo = head.get("repo")
    if not isinstance(repo, dict):
        return None, "pr view returned no head repository metadata; cannot locate PR head content"
    identity = _head_repo_identity(repo)
    if identity is None:
        return None, (
            "PR head repository identity (full_name or owner.login+name) is unreadable; "
            "cannot prove the cross-fork content source"
        )
    owner, name = identity
    source = HeadSource(
        target=atomgit_cli.RepoTarget(owner=owner, repo=name, origin="source"),
        sha=sha.strip(),
    )
    return source, None


def fetch_api_pages(cli: str, endpoint: str, *, what: str, key: str) -> list[dict]:
    """Read API collections without treating a repeated page as completeness.

    The verified PR files endpoint can repeat a full collection. Accept that
    case only when the unique record count matches independent PR metadata.
    Other repeated collections or unknown/mismatched totals fail closed.
    """

    def build_page_args(page: int) -> list[str]:
        separator = "&" if "?" in endpoint else "?"
        return ["api", f"{endpoint}{separator}page={page}&per_page=100"]

    result = atomgit_cli.run_paginated(build_page_args, cli=cli)
    if result.truncated and result.stop_reason != "repeated-page":
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
    if result.stop_reason == "repeated-page":
        match = re.fullmatch(r"repos/([^/]+)/([^/]+)/pulls/([0-9]+)/(files|commits)", endpoint)
        if match is None:
            raise atomgit_cli.CLIExecutionError("repeated page without an independent total; partial data")
        owner, repo, number, collection = match.groups()
        pr = fetch_pr_record(cli, atomgit_cli.RepoTarget(owner, repo, "explicit"), int(number))
        expected = pr.get("changed_files" if collection == "files" else "commits")
        if type(expected) is not int or expected < 0 or len(items) != expected:
            raise atomgit_cli.CLIExecutionError(
                f"repeated {collection} page cannot prove completeness: {len(items)} unique records, "
                f"expected total {expected!r}; refusing partial data"
            )
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

    Integrity is proven from the payload, not assumed (matrix row A17: the
    remote wrapper shape is pending authenticated verification, so every
    distinction the review requires is explicit):

    - a ``truncated`` payload is a hard failure — never a partial read;
    - a non-base64 ``encoding`` is a hard failure (no silent passthrough);
    - missing/None ``content`` is only accepted as a genuinely empty file
      when ``size == 0`` proves it; a nonzero size means the platform
      withheld the content (too large/truncated), which is a hard failure
      rather than a fake empty file;
    - base64 is decoded strictly (whitespace-stripped, then validated):
      decode and UTF-8 failures are hard failures, never partial text;
    - when ``size`` is present, the decoded byte length must match it,
      catching truncated transport before the text is consumed.

    Failures propagate as ``atomgit_cli.CLIJSONError`` (payload integrity)
    or the boundary's execution/timeout errors; callers must record them as
    unavailable content instead of stringifying them into the file body.
    The percent-encoding of the path (``safe=""``) matches the retired SDK
    ``get_file_content``.
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

    def integrity(problem: str) -> atomgit_cli.CLIJSONError:
        # Message carries only the integrity problem; the endpoint/argv stays
        # in the structured attributes (CLIError messages never embed paths).
        return atomgit_cli.CLIJSONError(
            f"contents payload is not usable: {problem}",
            argv=result.argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    if data.get("truncated"):
        raise integrity("payload is marked truncated")
    encoding = data.get("encoding")
    if encoding not in (None, "", "base64"):
        raise integrity(f"unsupported encoding {encoding!r}")
    content = data.get("content")
    size = data.get("size")
    size_known = isinstance(size, int) and not isinstance(size, bool)
    if content is None:
        # Missing content proves an empty file only when size says zero;
        # anything else is withheld content, not an empty file.
        if not size_known or size != 0:
            raise integrity("content is missing and the payload does not prove an empty file")
        return ""
    if not isinstance(content, str):
        raise integrity("content field is not a string")
    if content == "":
        if size_known and size != 0:
            raise integrity("content is empty but the payload reports nonzero size")
        return ""
    try:
        decoded = base64.b64decode("".join(content.split()), validate=True)
    except (binascii.Error, ValueError):
        raise integrity("base64 content failed strict decoding") from None
    if size_known and len(decoded) != size:
        raise integrity(f"decoded {len(decoded)} bytes but the payload reports {size}")
    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError:
        raise integrity("decoded bytes are not valid UTF-8") from None


def validate_extract_context_file(
    context_path: str | Path,
    *,
    cli: str,
    target: atomgit_cli.RepoTarget,
    pr_number: int,
    what: str,
) -> dict:
    """Bind a manual submit to a complete extract-info context and revision.

    A findings JSON alone is not publication evidence. The recorded context
    must carry an explicit target identity (owner/repo/pr_number) matching the
    submit target — two forks with the same repo name at the same head SHA are
    NOT interchangeable — plus a structurally complete ``pr.head_source``
    object (a valid ``owner/repo`` source slug and a complete 40/64-hex SHA
    consistent with ``pr.head_sha``) and a ``pr.changed_files`` list whose
    entries all carry a filename and complete head content. Finally the
    recorded head source is matched against the *live* PR head resolved via
    :func:`resolve_head_source`: the current source must be provable and both
    its SHA and its repository slug must agree with the recorded context —
    a missing current head repo or a wrong source repo at the same SHA refuses
    the submit. Every failure is a zero-write refusal; contexts predating
    identity binding are rejected and must be re-extracted.
    """
    path = Path(context_path)
    if not path.is_file():
        raise ValueError(
            f"no {what} context found at {path}; run --extract-info for PR {pr_number} first. "
            "submit-review refuses to publish findings that are not bound to a complete reviewed context"
        )
    try:
        with open(path, encoding="utf-8") as context_file:
            context = json.load(context_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"the {what} context {path} is unreadable: {error}") from error
    if not isinstance(context, dict):
        raise ValueError(f"the {what} context {path} is not a JSON object; re-run --extract-info")

    context_target = context.get("target")
    if not isinstance(context_target, dict) or not all(
        isinstance(context_target.get(key), str) and context_target[key].strip() for key in ("owner", "repo")
    ):
        raise ValueError(
            f"the {what} context {path} predates target-identity binding (no target.owner/repo); "
            "re-run --extract-info before submitting"
        )
    # PR numbers arrive as ints from our own argparse, but recorded contexts
    # (and some CLI payloads, e.g. issue numbers) may carry digit strings —
    # normalize for the comparison so a string recording never causes a false
    # rejection (authenticated evidence: PR numbers are ints, issue numbers
    # are strings; the context records whatever its extractor wrote).
    context_pr_number = context_target.get("pr_number")
    if not (
        (isinstance(context_pr_number, int) and not isinstance(context_pr_number, bool))
        or (isinstance(context_pr_number, str) and context_pr_number.strip().isdigit())
    ):
        raise ValueError(
            f"the {what} context target.pr_number {context_pr_number!r} is not a PR number; "
            "re-run --extract-info before submitting"
        )
    if str(context_pr_number).strip() != str(pr_number).strip():
        raise ValueError(
            f"the {what} context was extracted for PR #{context_pr_number}, not PR #{pr_number}; "
            "re-run --extract-info for this PR before submitting"
        )
    if context_target["owner"] != target.owner or context_target["repo"] != target.repo:
        raise ValueError(
            f"the {what} context was extracted for {context_target['owner']}/{context_target['repo']}, "
            f"not {target.slug}; a same-named repository under a different owner (e.g. a fork at the "
            "same head SHA) is not interchangeable — re-run --extract-info against the intended target"
        )

    pr_context = context.get("pr")
    if not isinstance(pr_context, dict):
        raise ValueError(f"the {what} context {path} has no pr object; re-run --extract-info")

    head_source = pr_context.get("head_source")
    if not isinstance(head_source, dict):
        raise ValueError(
            f"the {what} context is missing the pr.head_source object; its head content source is "
            "unprovable — re-run --extract-info before submitting"
        )
    if head_source.get("incomplete_reason"):
        raise ValueError(
            f"the recorded {what} context is incomplete ({head_source['incomplete_reason']}); "
            "re-run --extract-info and re-review before submitting"
        )
    head_source_repo = head_source.get("repo")
    if not isinstance(head_source_repo, str) or not head_source_repo.strip():
        raise ValueError(
            "the recorded pr.head_source.repo is missing or empty; the head content source is "
            "unprovable — re-run --extract-info before submitting"
        )
    head_source_repo = head_source_repo.strip()
    repo_slug_parts = head_source_repo.split("/")
    if len(repo_slug_parts) != 2 or not all(part.strip() for part in repo_slug_parts):
        raise ValueError(
            f"the recorded pr.head_source.repo {head_source_repo!r} is not a valid owner/repo slug; "
            "the head content source is unprovable — re-run --extract-info before submitting"
        )
    head_source_sha = head_source.get("sha")
    if not isinstance(head_source_sha, str) or _CONTEXT_SHA_RE.fullmatch(head_source_sha) is None:
        raise ValueError(
            "the recorded pr.head_source.sha is not a complete 40/64-hex object name; refusing to "
            f"bind the {what} to an ambiguous revision — re-run --extract-info"
        )
    context_head = pr_context.get("head_sha")
    if context_head != head_source_sha:
        raise ValueError(
            f"the {what} context is internally inconsistent: pr.head_sha {context_head!r} does not "
            f"match pr.head_source.sha {head_source_sha!r}; re-run --extract-info"
        )

    changed_files = pr_context.get("changed_files")
    if not isinstance(changed_files, list):
        raise ValueError(
            f"the {what} context is missing the pr.changed_files list; re-run --extract-info before submitting"
        )
    for index, changed in enumerate(changed_files):
        if not isinstance(changed, dict) or not isinstance(changed.get("filename"), str) or not changed["filename"]:
            raise ValueError(
                f"the {what} context changed_files[{index}] is malformed (no filename); re-run --extract-info"
            )
        filename = changed["filename"]
        if changed.get("content_error"):
            raise ValueError(
                f"the recorded {what} context has unreadable file content for: {filename}; "
                "re-run --extract-info and re-review before submitting"
            )
        if not isinstance(changed.get("content"), str):
            raise ValueError(
                f"the {what} context changed_files[{index}] ({filename}) carries no head content; "
                "re-run --extract-info and re-review before submitting"
            )

    # Bind to the *current* PR head source, not just its SHA: two forks can
    # share the same commit SHA, so a same-SHA context recorded against the
    # wrong source repository must not publish. resolve_head_source proves the
    # current source from the live PR record; an unprovable or missing current
    # head repo refuses the submit instead of falling back to a bare SHA
    # comparison.
    current = fetch_pr_record(cli, target, pr_number)
    current_source, current_incomplete_reason = resolve_head_source(current)
    if current_source is None:
        raise ValueError(
            f"the current PR head content source is unprovable ({current_incomplete_reason}); "
            f"refusing to bind the {what} to an ambiguous revision — re-run --extract-info after "
            "the PR head metadata is readable"
        )
    if current_source.sha.lower() != head_source_sha.lower():
        raise ValueError(
            f"PR head moved since the {what} context was extracted (context {context_head} vs current "
            f"{current_source.sha}); re-run --extract-info and re-review the updated revision before submitting"
        )
    if head_source_repo != current_source.repo_slug:
        raise ValueError(
            f"the {what} context was extracted against head repository {head_source_repo}, but the PR "
            f"head now lives at {current_source.repo_slug} (the same SHA does not make same-named forks "
            "interchangeable); re-run --extract-info against the intended source"
        )
    return context


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
