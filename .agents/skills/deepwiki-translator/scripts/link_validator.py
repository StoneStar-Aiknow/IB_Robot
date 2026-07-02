import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


DOC_EXTENSIONS = {".md", ".rst", ".txt"}
HTTP_TIMEOUT = 10
DEFAULT_MAX_WORKERS = 16
USER_AGENT = "deepwiki-link-validator/1.0"
ATOMGIT_HOSTS = {"atomgit.com"}
BROKEN_STATUSES = {"broken", "auth_error", "error"}
TOKEN_ENV_VARS = ("ATOMGIT_TOKEN",)


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
            if mermaid_indent is None:
                in_mermaid = False
            elif stripped.strip() and indent <= mermaid_indent and not stripped.startswith((":", "#")):
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
        re.compile(r"(?<![\w])(https?://[^\s<>)\]]+)")
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


def validate_links(paths: Iterable[Path], root: Path, access_token: str | None, timeout: int, max_workers: int) -> list[ValidationResult]:
    occurrences = []
    for file_path in iter_doc_files(paths):
        occurrences.extend(extract_links(file_path, root))

    remote_occurrences = {}
    for occurrence in occurrences:
        if occurrence.kind in {"local", "empty"}:
            continue
        cache_key = (occurrence.kind, occurrence.url)
        remote_occurrences.setdefault(cache_key, occurrence)

    cache = validate_remote_links(remote_occurrences, access_token, timeout, max_workers)
    results = []
    for occurrence in occurrences:
        cache_key = (occurrence.kind, occurrence.url)
        if occurrence.kind == "empty":
            result = make_result(occurrence, "broken", "empty markdown link target")
        elif occurrence.kind == "local":
            result = validate_local_link(occurrence, root)
        else:
            cached = cache[cache_key]
            result = ValidationResult(occurrence.file, occurrence.line, occurrence.url, occurrence.kind, cached.status, cached.detail, cached.http_status)
        results.append(result)
    return results


def validate_remote_links(remote_occurrences: dict[tuple[str, str], LinkOccurrence], access_token: str | None, timeout: int, max_workers: int) -> dict[tuple[str, str], ValidationResult]:
    if not remote_occurrences:
        return {}

    workers = max(1, min(max_workers, len(remote_occurrences)))
    cache = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(validate_remote_link, occurrence, access_token, timeout): cache_key
            for cache_key, occurrence in remote_occurrences.items()
        }
        for future in as_completed(futures):
            cache[futures[future]] = future.result()
    return cache


def validate_remote_link(occurrence: LinkOccurrence, access_token: str | None, timeout: int) -> ValidationResult:
    if occurrence.kind == "atomgit":
        return validate_atomgit_link(occurrence, access_token, timeout)
    return validate_external_link(occurrence, timeout)


def resolve_access_token(cli_token: str | None, config_path: str | None) -> str | None:
    if cli_token:
        return expand_env_var(cli_token)

    if config_path:
        config_token = read_config_token(Path(config_path))
        if config_token:
            return config_token

    for env_var in TOKEN_ENV_VARS:
        token = os.environ.get(env_var)
        if token:
            return token
    return None


def read_config_token(config_path: Path) -> str | None:
    if not config_path.exists():
        return None
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read config file {config_path}: {exc}") from exc

    atomgit = config.get("atomgit", {})
    if not isinstance(atomgit, dict):
        return None
    token = atomgit.get("token")
    if not isinstance(token, str) or not token.strip():
        return None
    return expand_env_var(token.strip())


def expand_env_var(value: str) -> str:
    pattern = re.compile(r"^\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?$")
    match = pattern.match(value)
    if not match:
        return value
    env_var = match.group(1)
    token = os.environ.get(env_var)
    if not token:
        raise ValueError(f"environment variable '{env_var}' referenced by token is not set")
    return token


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


def validate_atomgit_link(link: LinkOccurrence, access_token: str | None, timeout: int) -> ValidationResult:
    parsed = urlparse(link.url)
    segments = [unquote(segment) for segment in parsed.path.split("/") if segment]
    if any(segment == "bolb" for segment in segments):
        return make_result(link, "broken", "AtomGit route typo: /bolb/ should be /blob/")
    if len(segments) < 2:
        return validate_atomgit_user_or_org(link, access_token, timeout)

    route = segments[2] if len(segments) >= 3 else ""
    if route in {"blob", "tree"}:
        return validate_atomgit_blob_tree(link, parsed, segments, access_token, timeout)
    if route in {"pull", "commit", "issues", "milestones"}:
        return validate_atomgit_issue_like(link, parsed, segments, access_token, timeout)
    if len(segments) == 1 and segments[0] != "mindspore":
        return validate_atomgit_user_or_org(link, access_token, timeout)
    if len(segments) == 2:
        return validate_atomgit_repo(link, parsed, segments, access_token, timeout)
    return validate_external_link(link, timeout)


def validate_atomgit_blob_tree(link: LinkOccurrence, parsed, segments: list[str], access_token: str | None, timeout: int) -> ValidationResult:
    if len(segments) < 5:
        return make_result(link, "broken", "AtomGit blob/tree URL is missing branch or path")
    if not access_token:
        return make_result(link, "inconclusive", "AtomGit API validation requires --access-token or atomgit.token in --config")
    namespace, repo, route, branch = segments[:4]
    file_path = "/".join(segments[4:])
    status, detail, body = request_json(atomgit_api_url(parsed, f"repos/{namespace}/{repo}/contents/{quote(file_path, safe='/')}", {"ref": branch, "access_token": access_token}), timeout)
    if status != 200:
        return status_to_result(link, status, detail)
    target_type = atomgit_content_type(body)
    if target_type == "file" and route == "tree":
        return make_result(link, "broken", "AtomGit file target should use /blob/ instead of /tree/", status)
    if target_type == "directory" and route == "blob":
        return make_result(link, "broken", "AtomGit directory target should use /tree/ instead of /blob/", status)
    if target_type == "unknown":
        return make_result(link, "inconclusive", "AtomGit API response did not identify file or directory target", status)
    if parsed.fragment:
        if target_type != "file":
            return make_result(link, "broken", f"line fragment #{parsed.fragment} can only target a file", status)
        line_result = validate_atomgit_line_fragment(link, parsed, body, status)
        if line_result:
            return line_result
    return make_result(link, "valid", f"AtomGit {route} target exists", status)


def atomgit_content_type(body: object | None) -> str:
    if isinstance(body, list):
        return "directory"
    if isinstance(body, dict):
        if body.get("content") is not None:
            return "file"
        if body.get("type") in {"file", "dir", "directory"}:
            return "directory" if body.get("type") in {"dir", "directory"} else "file"
    return "unknown"


def validate_atomgit_line_fragment(link: LinkOccurrence, parsed, body: object | None, status: int) -> ValidationResult | None:
    line_match = re.fullmatch(r"L(\d+)(?:-L?(\d+))?", parsed.fragment)
    if not line_match:
        return None
    end_line = int(line_match.group(2) or line_match.group(1))
    content = body.get("content") if isinstance(body, dict) else None
    if not content:
        return make_result(link, "inconclusive", "AtomGit API did not return file content for line check", status)
    import base64
    try:
        decoded = base64.b64decode("".join(content.split())).decode("utf-8", errors="replace")
    except ValueError:
        return make_result(link, "inconclusive", "AtomGit API content could not be decoded", status)
    line_count = len(decoded.splitlines())
    if end_line <= line_count:
        return make_result(link, "valid", f"AtomGit target exists and line fragment is within {line_count} lines", status)
    return make_result(link, "broken", f"line fragment #{parsed.fragment} exceeds file length {line_count}", status)


def validate_atomgit_issue_like(link: LinkOccurrence, parsed, segments: list[str], access_token: str | None, timeout: int) -> ValidationResult:
    if not access_token:
        return make_result(link, "inconclusive", "AtomGit API validation requires --access-token or atomgit.token in --config")
    namespace, repo, route = segments[:3]
    if route == "pull":
        if len(segments) < 4:
            return make_result(link, "broken", "pull URL is missing pull request number")
        endpoint = f"repos/{namespace}/{repo}/pulls/{segments[3]}"
    elif route == "commit":
        if len(segments) < 4:
            return make_result(link, "broken", "commit URL is missing commit id")
        endpoint = f"repos/{namespace}/{repo}/commits/{segments[3]}"
    elif route == "milestones":
        if len(segments) < 4:
            return make_result(link, "broken", "milestones URL is missing milestone number")
        endpoint = f"repos/{namespace}/{repo}/milestones/{segments[3]}"
    else:
        endpoint = f"repos/{namespace}/{repo}/issues" if len(segments) == 3 else f"repos/{namespace}/{repo}/issues/{segments[3]}"
    status, detail, _ = request_json(atomgit_api_url(parsed, endpoint, {"access_token": access_token}), timeout)
    return status_to_result(link, status, detail, valid_detail=f"AtomGit {route} target exists")


def validate_atomgit_user_or_org(link: LinkOccurrence, access_token: str | None, timeout: int) -> ValidationResult:
    if not access_token:
        return make_result(link, "inconclusive", "AtomGit API validation requires --access-token or atomgit.token in --config")
    parsed = urlparse(link.url)
    segments = [unquote(segment) for segment in parsed.path.split("/") if segment]
    if not segments:
        return make_result(link, "valid", "AtomGit host root")
    namespace = segments[0]
    user_status, user_detail, _ = request_json(atomgit_api_url(parsed, f"users/{namespace}", {"access_token": access_token}), timeout)
    if user_status == 200:
        return make_result(link, "valid", "AtomGit user exists", user_status)
    org_status, org_detail, _ = request_json(atomgit_api_url(parsed, f"orgs/{namespace}", {"access_token": access_token}), timeout)
    if org_status == 200:
        return make_result(link, "valid", "AtomGit org exists", org_status)
    return status_to_result(link, org_status or user_status, org_detail or user_detail)


def validate_atomgit_repo(link: LinkOccurrence, parsed, segments: list[str], access_token: str | None, timeout: int) -> ValidationResult:
    if not access_token:
        return make_result(link, "inconclusive", "AtomGit API validation requires --access-token or atomgit.token in --config")
    namespace, repo = segments[:2]
    status, detail, _ = request_json(atomgit_api_url(parsed, f"repos/{namespace}/{repo}", {"access_token": access_token}), timeout)
    return status_to_result(link, status, detail, valid_detail="AtomGit repository exists")


def atomgit_api_url(parsed, endpoint: str, params: dict[str, str]) -> str:
    query = urlencode(params)
    return urlunparse((parsed.scheme, "api.atomgit.com", f"/api/v5/{endpoint}", "", query, ""))


def request_status(url: str, method: str, timeout: int) -> tuple[int | None, str]:
    request = Request(url, method=method, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.reason
    except HTTPError as error:
        return error.code, error.reason
    except URLError as error:
        return None, str(error.reason)
    except TimeoutError:
        return None, "timeout"


def request_json(url: str, timeout: int) -> tuple[int | None, str, object | None]:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            try:
                return response.status, response.reason, json.loads(raw) if raw else None
            except json.JSONDecodeError:
                return response.status, "non-JSON API response", None
    except HTTPError as error:
        return error.code, error.reason, None
    except URLError as error:
        return None, str(error.reason), None
    except TimeoutError:
        return None, "timeout", None


def status_to_result(link: LinkOccurrence, status: int | None, detail: str, valid_detail: str = "URL is reachable") -> ValidationResult:
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
    parser.add_argument("--root", default=".", help="Root directory for resolving local links and relative report paths")
    parser.add_argument("--report", default=None, help="JSON report path (default: <first scanned directory parent>/reports/link_validation.json)")
    parser.add_argument("--config", default="config.json", help="Optional config.json containing atomgit.token; supports $ATOMGIT_TOKEN placeholders")
    parser.add_argument("--access-token", default=None, help="AtomGit access token for API validation")
    parser.add_argument("--timeout", type=int, default=HTTP_TIMEOUT, help="HTTP timeout in seconds")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Maximum concurrent remote URL checks")
    parser.add_argument("--fail-on-inconclusive", action="store_true", help="Return non-zero when inconclusive links are found")
    args = parser.parse_args(argv)

    root = Path(args.root).resolve()
    paths = [Path(path).resolve() for path in args.paths]
    try:
        access_token = resolve_access_token(args.access_token, args.config)
    except ValueError as exc:
        print(f"Token configuration error: {exc}", file=sys.stderr)
        return 2
    results = validate_links(paths, root, access_token, args.timeout, args.max_workers)
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
