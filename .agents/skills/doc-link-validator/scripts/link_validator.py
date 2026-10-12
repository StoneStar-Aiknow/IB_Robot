#!/usr/bin/env python3
"""
Documentation Link Validator

Validates local, external, and AtomGit links in generated docs. AtomGit
platform lookups go through the official CLI: every former
``api.atomgit.com`` REST v5 branch — blob/tree, pull, commit, issues,
milestones, repo, user/org — is probed read-only through ``gitcode``. The
CLI access layer is self-contained below (this skill carries no cross-skill
shared module): an argv-only ``subprocess`` call with a bounded timeout and
``GC_HOST=atomgit.com`` pinned, plus a one-shot ``gitcode version``
identity probe so a same-name non-official executable can never have its
exit codes mistaken for link verdicts. Authentication belongs to the CLI
(``gitcode auth login --hostname atomgit.com --with-token``); the script
never reads tokens and accepts no token arguments.

Result semantics are preserved: the CLI exit code maps to three states —
exit 0 → valid, exit 3 (not found) → broken, everything else (CLI
unavailable, not authenticated, no permission, timeout, exec failure,
other failures, unusable output) is explicit inconclusive and is never
misreported as broken. Third-party HTTP links keep the plain urllib probe
and are not routed through the AtomGit platform API.
"""

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

DOC_EXTENSIONS = {".md", ".rst", ".txt"}
HTTP_TIMEOUT = 10
DEFAULT_MAX_WORKERS = 16
USER_AGENT = "deepwiki-link-validator/1.0"
ATOMGIT_HOSTS = frozenset({"atomgit.com", "gitcode.com"})
BROKEN_STATUSES = {"broken", "auth_error", "error"}

# --- minimal read-only AtomGit CLI access (self-contained) ----------------- #

CLI_ENV_VAR = "IBROBOT_ATOMGIT_CLI"
CLI_ENTRY_NAME = "gitcode"
CLI_HOST = "atomgit.com"
AUTH_LOGIN_GUIDANCE = "gitcode auth login --hostname atomgit.com --with-token"
CLI_EXIT_NOT_FOUND = 3
CLI_EXIT_AUTH = 4
IDENTITY_TIMEOUT = 30
_VERSION_RE = re.compile(r"gitcode version (\d+)\.(\d+)\.(\d+)")


@dataclass
class CliOutcome:
    """Outcome of one CLI invocation; ``run_cli`` never raises.

    ``returncode`` is None when the process never produced an exit code
    (timeout or exec failure); ``run_error`` then carries the reason.
    """

    returncode: int | None = None
    stdout: str = ""
    data: object = None
    json_error: str | None = None
    run_error: str | None = None


def run_cli(args: list[str], cli: str, timeout: float) -> CliOutcome:
    """Execute the CLI once read-only: argv list, no shell, no retry.

    stdin is /dev/null so prompts cannot hang, ``GC_HOST`` is pinned to the
    platform host, and stdout is decoded as JSON when possible.
    """
    env = dict(os.environ)
    env["GC_HOST"] = CLI_HOST
    try:
        completed = subprocess.run(
            [cli, *args, "--no-interactive"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CliOutcome(run_error=f"request timed out after {timeout}s")
    except OSError as exc:
        # ENOEXEC ("Exec format error") and friends: the CLI never ran, so
        # the platform was never asked — inconclusive, never a verdict.
        return CliOutcome(run_error=f"executable could not be run: {exc.strerror or exc}")
    outcome = CliOutcome(returncode=completed.returncode, stdout=completed.stdout)
    stripped = completed.stdout.strip()
    if stripped:
        try:
            outcome.data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            outcome.json_error = f"invalid JSON: {exc}"
    else:
        outcome.json_error = "empty stdout"
    return outcome


def resolve_cli_access() -> tuple[str | None, str | None]:
    """Resolve the official CLI and verify its identity once.

    Returns ``(cli, unavailable_reason)`` with exactly one side None. An
    explicit ``IBROBOT_ATOMGIT_CLI`` absolute path wins over ``gitcode`` on
    PATH; ``gc`` (Graphviz) is never considered. The one-shot identity
    probe runs ``gitcode version`` and requires the official output shape,
    so a same-name non-official executable is rejected before its exit
    codes can be mistaken for link verdicts; any release number passes.
    When the CLI is unavailable, every AtomGit link is reported
    inconclusive with the reason while local and external links keep being
    checked. Authentication is diagnosed per probe instead: exit 4 is
    inconclusive with login guidance, never broken.
    """
    explicit = os.environ.get(CLI_ENV_VAR, "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            return None, f"{CLI_ENV_VAR} must be an absolute executable path, got: {explicit!r}"
        if not path.is_file():
            return None, (
                f"{CLI_ENV_VAR} points to a missing file: {explicit}; "
                "fix the variable or unset it to use 'gitcode' from PATH"
            )
        if not os.access(path, os.X_OK):
            return None, f"{CLI_ENV_VAR} points to a non-executable file: {explicit}"
    else:
        found = shutil.which(CLI_ENTRY_NAME)
        if not found:
            return None, (
                f"official AtomGit CLI '{CLI_ENTRY_NAME}' was not found on PATH and "
                f"{CLI_ENV_VAR} is not set; install the official CLI or set {CLI_ENV_VAR} "
                "to its absolute path (do not use Graphviz 'gc')"
            )
        path = Path(found)
    outcome = run_cli(["version"], str(path), IDENTITY_TIMEOUT)
    if outcome.run_error is not None:
        return None, f"AtomGit CLI at {path} {outcome.run_error}"
    if outcome.returncode != 0 or not _VERSION_RE.search(outcome.stdout):
        return None, (
            f"executable at {path} did not report 'gitcode version X.Y.Z' and is not the "
            f"official AtomGit CLI (same-name tools such as Graphviz 'gc' are rejected on "
            f"purpose); install the official CLI or set {CLI_ENV_VAR}"
        )
    return str(path), None


@dataclass
class LinkOccurrence:
    file: str
    line: int
    url: str
    kind: str


@dataclass
class ValidationResult:
    file: str
    line: int
    url: str
    kind: str
    status: str
    detail: str
    http_status: int | None = None


class HeadingParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if "id" in attrs_dict:
            self.ids.add(attrs_dict["id"])


def strip_code_and_mermaid_blocks(text: str) -> str:
    lines = text.splitlines(keepends=True)
    output = []
    in_fence = False
    fence_marker = ""
    in_mermaid = False
    mermaid_indent = None

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if in_fence:
            output.append("\n")
            if stripped.startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            continue

        if in_mermaid:
            if (
                mermaid_indent is None
                or stripped.strip()
                and indent <= mermaid_indent
                and not stripped.startswith((":", "#"))
            ):
                in_mermaid = False
            else:
                output.append("\n")
                continue

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = True
            fence_marker = stripped[:3]
            output.append("\n")
            continue

        if stripped.startswith(".. mermaid::"):
            in_mermaid = True
            mermaid_indent = indent
            output.append("\n")
            continue

        output.append(line)

    return "".join(output)


def iter_doc_files(paths: Iterable[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_file() and path.suffix.lower() in DOC_EXTENSIONS:
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in DOC_EXTENSIONS)
    return sorted(set(files))


def line_number(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def normalize_url(raw: str) -> str:
    return raw.strip().strip("<>").rstrip(".,;:")


def extract_links(file_path: Path, root: Path) -> list[LinkOccurrence]:
    try:
        raw = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = file_path.read_text(encoding="gbk")
    text = strip_code_and_mermaid_blocks(raw)
    rel_file = file_path.relative_to(root).as_posix() if file_path.is_relative_to(root) else file_path.as_posix()
    links = []
    seen = set()

    patterns = [
        re.compile(r"!?\[[^\]\n]*\]\(([^)\s]*)(?:\s+\"[^\"]*\")?\)"),
        re.compile(r"(?<![\w])(https?://[^\s<>)\]]+)"),
    ]

    for pattern in patterns:
        for match in pattern.finditer(text):
            url = normalize_url(match.group(1))
            if not url:
                key = (match.start(1), url)
                if key in seen:
                    continue
                seen.add(key)
                links.append(LinkOccurrence(rel_file, line_number(text, match.start(1)), url, "empty"))
                continue
            if not should_collect_url(url):
                continue
            key = (match.start(1), url)
            if key in seen:
                continue
            seen.add(key)
            links.append(LinkOccurrence(rel_file, line_number(text, match.start(1)), url, classify_kind(url)))

    return sorted(links, key=lambda item: (item.line, item.url))


def should_collect_url(url: str) -> bool:
    if not url:
        return False
    lower = url.lower()
    return not lower.startswith(("mailto:", "tel:", "javascript:", "data:"))


def classify_kind(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc.lower() in ATOMGIT_HOSTS:
            return "atomgit"
        return "external"
    return "local"


def validate_links(
    paths: Iterable[Path],
    root: Path,
    cli: str | None,
    cli_unavailable: str | None,
    timeout: int,
    max_workers: int,
) -> list[ValidationResult]:
    occurrences = []
    for file_path in iter_doc_files(paths):
        occurrences.extend(extract_links(file_path, root))

    remote_occurrences = {}
    for occurrence in occurrences:
        if occurrence.kind in {"local", "empty"}:
            continue
        cache_key = (occurrence.kind, occurrence.url)
        remote_occurrences.setdefault(cache_key, occurrence)

    cache = validate_remote_links(remote_occurrences, cli, cli_unavailable, timeout, max_workers)
    results = []
    for occurrence in occurrences:
        cache_key = (occurrence.kind, occurrence.url)
        if occurrence.kind == "empty":
            result = make_result(occurrence, "broken", "empty markdown link target")
        elif occurrence.kind == "local":
            result = validate_local_link(occurrence, root)
        else:
            cached = cache[cache_key]
            result = ValidationResult(
                occurrence.file,
                occurrence.line,
                occurrence.url,
                occurrence.kind,
                cached.status,
                cached.detail,
                cached.http_status,
            )
        results.append(result)
    return results


def validate_remote_links(
    remote_occurrences: dict[tuple[str, str], LinkOccurrence],
    cli: str | None,
    cli_unavailable: str | None,
    timeout: int,
    max_workers: int,
) -> dict[tuple[str, str], ValidationResult]:
    if not remote_occurrences:
        return {}

    workers = max(1, min(max_workers, len(remote_occurrences)))
    cache = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(validate_remote_link, occurrence, cli, cli_unavailable, timeout): cache_key
            for cache_key, occurrence in remote_occurrences.items()
        }
        for future in as_completed(futures):
            cache[futures[future]] = future.result()
    return cache


def validate_remote_link(
    occurrence: LinkOccurrence,
    cli: str | None,
    cli_unavailable: str | None,
    timeout: int,
) -> ValidationResult:
    if occurrence.kind == "atomgit":
        offline = atomgit_offline_result(occurrence)
        if offline is not None:
            return offline
        if cli is None:
            # CLI missing, failing identity, or not authenticated: explicit
            # inconclusive for every remaining AtomGit link (never broken,
            # D1/D5). Offline-decidable classifications were already
            # returned above, mirroring the legacy pre-token-gate order.
            return make_result(occurrence, "inconclusive", cli_unavailable or "AtomGit CLI is unavailable")
        return validate_atomgit_link(occurrence, cli, timeout)
    return validate_external_link(occurrence, timeout)


def atomgit_offline_result(link: LinkOccurrence) -> ValidationResult | None:
    """Offline-decidable AtomGit classification, or None when a probe is needed.

    Route typos and structurally incomplete blob/tree URLs were reported
    broken *before* the token gate in the pre-migration code; they stay
    broken with the CLI missing instead of degrading to inconclusive.
    """
    segments = [unquote(segment) for segment in urlparse(link.url).path.split("/") if segment]
    if any(segment == "bolb" for segment in segments):
        return make_result(link, "broken", "AtomGit route typo: /bolb/ should be /blob/")
    if len(segments) >= 3 and segments[2] in {"blob", "tree"} and len(segments) < 5:
        return make_result(link, "broken", "AtomGit blob/tree URL is missing branch or path")
    return None


def cli_probe(
    link: LinkOccurrence,
    args: list[str],
    cli: str,
    timeout: int,
    valid_detail: str,
    *,
    require_json: bool = False,
) -> tuple[ValidationResult, object | None]:
    """One read-only CLI probe mapped to the three-state semantics.

    Exit 0 → valid (JSON is additionally required only when the caller
    consumes the response body); exit 3 → broken; everything else — CLI
    unavailable, not authenticated or lacking permission (exit 4), timeout,
    exec failure, other nonzero exits and unusable output — is explicit
    inconclusive, never misreported as broken.
    """
    outcome = run_cli(args, cli, timeout)
    if outcome.run_error is not None:
        return make_result(link, "inconclusive", f"AtomGit CLI {outcome.run_error}"), None
    if outcome.returncode == 0 and (not require_json or outcome.data is not None):
        return make_result(link, "valid", valid_detail), outcome.data
    if outcome.returncode == CLI_EXIT_NOT_FOUND:
        return make_result(link, "broken", "AtomGit target was not found"), None
    if outcome.returncode == CLI_EXIT_AUTH:
        detail = f"AtomGit CLI is not authenticated or lacks permission for this target; run: {AUTH_LOGIN_GUIDANCE}"
    elif outcome.returncode != 0:
        detail = f"AtomGit CLI failed with exit code {outcome.returncode}"
    else:
        detail = "AtomGit CLI produced no usable JSON output"
    return make_result(link, "inconclusive", detail), None


def validate_local_link(link: LinkOccurrence, root: Path) -> ValidationResult:
    parsed = urlparse(link.url)
    source = root / link.file
    if not parsed.path and parsed.fragment:
        target = source
    else:
        target = (source.parent / unquote(parsed.path)).resolve()

    try:
        target.relative_to(root.resolve())
    except ValueError:
        return make_result(link, "broken", "local link escapes validation root")

    if not target.exists():
        return make_result(link, "broken", f"local target does not exist: {target}")
    if parsed.fragment and target.suffix.lower() in {".md", ".rst", ".html"}:
        return validate_local_fragment(link, target, parsed.fragment)
    return make_result(link, "valid", "local target exists")


def validate_local_fragment(link: LinkOccurrence, target: Path, fragment: str) -> ValidationResult:
    try:
        text = target.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = target.read_text(encoding="gbk")
    anchors = markdown_anchors(text)
    if target.suffix.lower() == ".html":
        parser = HeadingParser()
        parser.feed(text)
        anchors.update(parser.ids)
    if fragment in anchors:
        return make_result(link, "valid", "local target and fragment exist")
    return make_result(link, "inconclusive", f"local target exists but fragment was not found: #{fragment}")


def markdown_anchors(text: str) -> set[str]:
    anchors = set()
    duplicates = {}
    for line in text.splitlines():
        line = line.lstrip("\ufeff")
        explicit = re.search(r"\{#([A-Za-z0-9_.:-]+)\}\s*$", line)
        if explicit:
            anchors.add(explicit.group(1))
        match = re.match(r"^#{1,6}\s+(.+?)\s*$", line)
        if not match:
            continue
        title = re.sub(r"<[^>]+>", "", match.group(1))
        title = re.sub(r"[`*_\[\](){}]", "", title).strip().lower()
        anchor = re.sub(r"[^\w\u4e00-\u9fff\-\s]", "", title)
        anchor = re.sub(r"\s+", "-", anchor)
        count = duplicates.get(anchor, 0)
        duplicates[anchor] = count + 1
        anchors.add(anchor if count == 0 else f"{anchor}-{count}")
    return anchors


def validate_external_link(link: LinkOccurrence, timeout: int) -> ValidationResult:
    status, detail = request_status(link.url, "HEAD", timeout)
    if status in {405, 501, 403} or status is None:
        status, detail = request_status(link.url, "GET", timeout)
    return status_to_result(link, status, detail)


def validate_atomgit_link(link: LinkOccurrence, cli: str, timeout: int) -> ValidationResult:
    parsed = urlparse(link.url)
    segments = [unquote(segment) for segment in parsed.path.split("/") if segment]
    if len(segments) < 2:
        return validate_atomgit_user_or_org(link, cli, timeout)

    route = segments[2] if len(segments) >= 3 else ""
    if route in {"blob", "tree"}:
        return validate_atomgit_blob_tree(link, parsed, segments, cli, timeout)
    if route in {"pull", "pulls", "merge_requests", "commit", "issues", "milestones"}:
        return validate_atomgit_issue_like(link, segments, cli, timeout)
    if len(segments) == 2:
        return validate_atomgit_repo(link, segments, cli, timeout)
    return validate_external_link(link, timeout)


def validate_atomgit_blob_tree(
    link: LinkOccurrence,
    parsed,
    segments: list[str],
    cli: str,
    timeout: int,
) -> ValidationResult:
    namespace, repo, route, branch = segments[:4]
    file_path = "/".join(segments[4:])
    # The contents endpoint is only reachable through the registered generic
    # API entry (no dedicated CLI command); query parameters ride the endpoint
    # string, auto-prefixed with /api/v5 by the CLI.
    endpoint = f"repos/{namespace}/{repo}/contents/{quote(file_path, safe='/')}?ref={branch}"
    result, body = cli_probe(link, ["api", endpoint], cli, timeout, f"AtomGit {route} target exists", require_json=True)
    if result.status != "valid":
        return result
    target_type = atomgit_content_type(body)
    if target_type == "file" and route == "tree":
        return make_result(link, "broken", "AtomGit file target should use /blob/ instead of /tree/")
    if target_type == "directory" and route == "blob":
        return make_result(link, "broken", "AtomGit directory target should use /tree/ instead of /blob/")
    if target_type == "unknown":
        return make_result(link, "inconclusive", "AtomGit API response did not identify file or directory target")
    if parsed.fragment:
        if target_type != "file":
            return make_result(link, "broken", f"line fragment #{parsed.fragment} can only target a file")
        line_result = validate_atomgit_line_fragment(link, parsed, body)
        if line_result:
            return line_result
    return result


def atomgit_content_type(body: object | None) -> str:
    if isinstance(body, list):
        return "directory"
    if isinstance(body, dict):
        if body.get("content") is not None:
            return "file"
        if body.get("type") in {"file", "dir", "directory"}:
            return "directory" if body.get("type") in {"dir", "directory"} else "file"
    return "unknown"


def validate_atomgit_line_fragment(link: LinkOccurrence, parsed, body: object | None) -> ValidationResult | None:
    line_match = re.fullmatch(r"L(\d+)(?:-L?(\d+))?", parsed.fragment)
    if not line_match:
        return None
    end_line = int(line_match.group(2) or line_match.group(1))
    content = body.get("content") if isinstance(body, dict) else None
    if not content:
        return make_result(link, "inconclusive", "AtomGit API did not return file content for line check")
    try:
        decoded = base64.b64decode("".join(content.split())).decode("utf-8", errors="replace")
    except ValueError:
        return make_result(link, "inconclusive", "AtomGit API content could not be decoded")
    line_count = len(decoded.splitlines())
    if end_line <= line_count:
        return make_result(link, "valid", f"AtomGit target exists and line fragment is within {line_count} lines")
    return make_result(link, "broken", f"line fragment #{parsed.fragment} exceeds file length {line_count}")


def validate_atomgit_issue_like(link: LinkOccurrence, segments: list[str], cli: str, timeout: int) -> ValidationResult:
    namespace, repo, route = segments[:3]
    if route in {"pull", "pulls", "merge_requests"}:
        if len(segments) < 4:
            return make_result(link, "broken", "pull URL is missing pull request number")
        if not segments[3].isdigit():
            return validate_external_link(link, timeout)
        result, _ = cli_probe(
            link,
            ["pr", "view", segments[3], "-R", f"{namespace}/{repo}", "--json"],
            cli,
            timeout,
            "AtomGit pull target exists",
        )
        return result
    if route == "commit":
        if len(segments) < 4:
            return make_result(link, "broken", "commit URL is missing commit id")
        result, _ = cli_probe(
            link,
            ["commit", "view", segments[3], "-R", f"{namespace}/{repo}"],
            cli,
            timeout,
            "AtomGit commit target exists",
        )
        return result
    if route == "milestones":
        if len(segments) < 4:
            return make_result(link, "broken", "milestones URL is missing milestone number")
        # ``milestone view`` repository targeting is not documented in the
        # verified help evidence (its examples carry no -R), so the original
        # REST endpoint stays on the registered generic API entry.
        result, _ = cli_probe(
            link,
            ["api", f"repos/{namespace}/{repo}/milestones/{segments[3]}"],
            cli,
            timeout,
            "AtomGit milestone target exists",
        )
        return result
    if len(segments) == 3:
        result, _ = cli_probe(
            link,
            ["issue", "list", "-R", f"{namespace}/{repo}", "--json", "-L", "1"],
            cli,
            timeout,
            "AtomGit issues collection is readable",
        )
        return result
    result, _ = cli_probe(
        link,
        ["issue", "view", segments[3], "-R", f"{namespace}/{repo}", "--json"],
        cli,
        timeout,
        "AtomGit issue target exists",
    )
    return result


def validate_atomgit_user_or_org(link: LinkOccurrence, cli: str, timeout: int) -> ValidationResult:
    parsed = urlparse(link.url)
    segments = [unquote(segment) for segment in parsed.path.split("/") if segment]
    if not segments:
        return make_result(link, "valid", "AtomGit host root")
    namespace = segments[0]
    user_result, _ = cli_probe(link, ["user", "view", namespace, "--json"], cli, timeout, "AtomGit user exists")
    if user_result.status == "valid":
        return user_result
    org_result, _ = cli_probe(link, ["api", f"orgs/{namespace}"], cli, timeout, "AtomGit org exists")
    if org_result.status == "valid":
        return org_result
    if user_result.status == "broken" and org_result.status == "broken":
        return make_result(link, "broken", "AtomGit user or org target was not found")
    # At least one probe was inconclusive: never misreport broken (D1/D5).
    detail = org_result.detail if org_result.status != "broken" else user_result.detail
    return make_result(link, "inconclusive", detail)


def validate_atomgit_repo(link: LinkOccurrence, segments: list[str], cli: str, timeout: int) -> ValidationResult:
    namespace, repo = segments[:2]
    result, _ = cli_probe(
        link, ["repo", "view", f"{namespace}/{repo}", "--json"], cli, timeout, "AtomGit repository exists"
    )
    return result


def encode_url(url: str) -> str:
    """Percent-encode non-ASCII characters so http.client can send the request line.

    Raw non-ASCII characters (e.g. Chinese paths) crash http.client with UnicodeEncodeError;
    existing percent-escapes are preserved verbatim.
    """
    return quote(url, safe=":/?#[]@!$&'()*+,;=%")


def request_status(url: str, method: str, timeout: int) -> tuple[int | None, str]:
    request = Request(encode_url(url), method=method, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.reason
    except HTTPError as error:
        return error.code, error.reason
    except URLError as error:
        return None, str(error.reason)
    except TimeoutError:
        return None, "timeout"


def status_to_result(
    link: LinkOccurrence, status: int | None, detail: str, valid_detail: str = "URL is reachable"
) -> ValidationResult:
    if status is None:
        return make_result(link, "error", detail)
    if 200 <= status < 400:
        return make_result(link, "valid", valid_detail, status)
    if status in {401, 403}:
        return make_result(link, "auth_error", detail, status)
    if status == 404:
        return make_result(link, "broken", detail, status)
    if 400 <= status < 600:
        return make_result(link, "broken", detail, status)
    return make_result(link, "inconclusive", detail, status)


def make_result(link: LinkOccurrence, status: str, detail: str, http_status: int | None = None) -> ValidationResult:
    return ValidationResult(link.file, link.line, link.url, link.kind, status, detail, http_status)


def write_report(results: list[ValidationResult], report_path: Path) -> None:
    summary = {}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    reported_results = [result for result in results if result.status != "valid"]
    reported_summary = {}
    for result in reported_results:
        reported_summary[result.status] = reported_summary.get(result.status, 0) + 1
    payload = {
        "checked_summary": summary,
        "reported_summary": reported_summary,
        "results": [asdict(result) for result in reported_results],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(results: list[ValidationResult], report_path: Path) -> None:
    summary = {}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    total = len(results)
    print(f"Checked {total} links. Report written to {report_path}")
    for status in sorted(summary):
        print(f"  {status}: {summary[status]}")
    for result in results:
        if result.status in BROKEN_STATUSES or result.status == "inconclusive":
            print(f"{result.status.upper()}: {result.file}:{result.line} {result.url} ({result.detail})")


def default_report_path(paths: list[Path]) -> Path:
    base_path = paths[0]
    if base_path.is_file():
        base_path = base_path.parent
    report_dir = base_path.parent / "reports"
    return report_dir / "link_validation.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate local, external, and AtomGit links in generated docs")
    parser.add_argument("paths", nargs="+", help="Files or directories to scan")
    parser.add_argument(
        "--root", default=".", help="Root directory for resolving local links and relative report paths"
    )
    parser.add_argument(
        "--report",
        default=None,
        help="JSON report path (default: <first scanned directory parent>/reports/link_validation.json)",
    )
    parser.add_argument("--timeout", type=int, default=HTTP_TIMEOUT, help="HTTP and CLI timeout in seconds")
    parser.add_argument(
        "--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Maximum concurrent remote URL checks"
    )
    parser.add_argument(
        "--fail-on-inconclusive", action="store_true", help="Return non-zero when inconclusive links are found"
    )
    args = parser.parse_args(argv)

    cli, cli_unavailable = resolve_cli_access()

    root = Path(args.root).resolve()
    paths = [Path(path).resolve() for path in args.paths]
    results = validate_links(paths, root, cli, cli_unavailable, args.timeout, args.max_workers)
    report_path = Path(args.report) if args.report else default_report_path(paths)
    write_report(results, report_path)
    print_summary(results, report_path)

    has_broken = any(result.status in BROKEN_STATUSES for result in results)
    has_inconclusive = any(result.status == "inconclusive" for result in results)
    if has_broken or (args.fail_on_inconclusive and has_inconclusive):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
