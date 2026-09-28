"""Shared AtomGit CLI boundary for IB-Robot collaboration skills.

OpenSpec change ``migrate-atomgit-skills-to-cli`` (tasks 2.1-2.3). All
AtomGit collaboration scripts call the official CLI through this thin
boundary; it owns process execution, repository/URL targeting, JSON
decoding, pagination and error classification only (design.md D1). It is
deliberately NOT an SDK replacement: scripts build command argv from the
version-bound capability matrix evidence (CLI v0.14.1, commit 21e19618).

Verified facts baked into this module (evidence/cli/0.14.1/linux-x64):

- Entry point: ``gitcode``. The resolution order is the explicit absolute
  path in ``IBROBOT_ATOMGIT_CLI``, then ``gitcode`` on PATH. ``gc`` is never
  resolved implicitly: on dev hosts it is Graphviz (design.md D2). An invalid
  explicit selection fails without falling back to another install.
- Host selection: ``GC_HOST=atomgit.com`` is pinned for every invocation. The
  only registered web->API service mapping is atomgit.com -> api.atomgit.com
  (design.md D2). Other hosts (including gitcode.com) are rejected before
  access instead of being silently redirected.
- Exit codes (documented on ``pr view`` and verified for the auth gate):
  401/403 -> 4 (auth), 404 -> 3 (not found), 409 -> 5 (conflict),
  other failure -> 1. Unauthenticated API access exits 4 before any request.
- ``--no-interactive`` (global flag) is appended to every invocation so
  agents/scripts never block on prompts.
- Commands ``issue list``/``api`` have no ``--paginate``: callers use
  :func:`run_paginated` for them. ``pr list`` and ``pr comments --limit 0``
  fetch server-side; completeness still has to be judged by callers.

Migration semantics that this module intentionally does NOT implement:

- No token handling: legacy ``atomgit.token`` in config.json is never read,
  validated, printed or migrated (design.md D2). Authentication belongs to
  the CLI (``gitcode auth login --hostname atomgit.com --with-token``).
- No retry of any kind: writes whose outcome is unknown (timeout, nonzero
  exit after server acceptance) are surfaced with full argv context so
  callers can reconcile before any retry (design.md D3/D5).
- No owner/repo case folding for platform targeting: canonical identity is
  only provable via the platform itself (design.md D2); unresolved case
  differences are reported as conflicts.
- ``config.json``'s ``atomgit.baseUrl`` never routes requests (host selection
  is pinned through ``GC_HOST`` to the registered mapping above), but its host
  is validated against that mapping whenever a config participates in
  targeting: an unregistered service is rejected before any request instead
  of being silently ignored.

Consumers import this module by putting the ``_shared`` directory on
``sys.path`` (mirroring the existing ``lib`` convention), e.g. from a skill
``scripts/foo.py``::

    sys.path.insert(0, str(Path(__file__).parents[2] / "_shared"))
    import atomgit_cli
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Service mapping (design.md D2): the only registered web->API host pair.
DEFAULT_WEB_HOST = "atomgit.com"
WEB_HOST_TO_API_HOST = {"atomgit.com": "api.atomgit.com"}

# CLI selection (design.md D2).
CLI_ENV_VAR = "IBROBOT_ATOMGIT_CLI"
CLI_ENTRY_NAME = "gitcode"
AUTH_LOGIN_GUIDANCE = "gitcode auth login --hostname atomgit.com --with-token"

# CLI exit codes, verified on v0.14.1 (pr view help + auth-gate evidence).
EXIT_ERROR = 1
EXIT_NOT_FOUND = 3
EXIT_AUTH = 4
EXIT_CONFLICT = 5

DEFAULT_TIMEOUT = 120.0
MAX_PAGES = 1000
# Capability matrix evidence is bound to v0.14.1 (help-audit.md); older
# binaries are unverified for the mapped command surface.
MIN_CLI_VERSION = (0, 14, 1)

_VERSION_RE = re.compile(r"gitcode version (\d+)\.(\d+)\.(\d+)")
_COMMIT_RE = re.compile(r"^\s*commit:\s*([0-9a-f]{7,40})\s*$", re.MULTILINE)


def _command_summary(argv: list[str]) -> str:
    """Build a redacted command shape for error message text.

    Message text must never carry bodies, file paths or raw CLI output:
    the CLI may echo them on stderr, and the boundary does not control
    what the platform writes back. Only the executable name and the
    leading positional tokens (up to the first flag and at most four
    words) are exposed. The full argv remains in ``CLIError.argv`` as a
    controlled structure so callers can log it deliberately.
    """
    if not argv:
        return "<not invoked>"
    parts = [Path(argv[0]).name]
    for arg in argv[1:]:
        if arg.startswith("-") or len(parts) >= 4:
            parts.append("[arguments omitted]")
            break
        parts.append(arg)
    return " ".join(parts)


class CLIError(RuntimeError):
    """Base class for every error raised by this boundary.

    Message text is sanitized: it never embeds full argv, comment bodies
    or raw CLI output (stderr may echo bodies or credentials, and the
    boundary cannot assume otherwise). Transaction context that callers
    need for reconciliation lives in the structured attributes
    (``argv``/``stdout``/``stderr``/``returncode``/``timeout``) and is
    logged only under caller control, never via ``str(error)``.
    """

    def __init__(
        self,
        message: str,
        *,
        argv: list[str] | None = None,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        timeout: float | None = None,
    ) -> None:
        super().__init__(message)
        self.argv = list(argv or [])
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timeout = timeout


class CLISelectionError(CLIError):
    """The CLI executable is missing, invalid or not the official tool."""


class CLIExecutionError(CLIError):
    """The CLI failed with a nonzero exit (unclassified other than below)."""


class CLIAuthError(CLIExecutionError):
    """Exit code 4: not authenticated, or lacking permission for the request."""


class CLINotFoundError(CLIExecutionError):
    """Exit code 3: requested platform resource was not found."""


class CLIConflictError(CLIExecutionError):
    """Exit code 5: the request conflicts with platform state."""


class CLITimeoutError(CLIError):
    """The invocation exceeded the bounded timeout; outcome is unknown.

    For write commands the server may already have accepted the mutation:
    reconcile via read-back or listing before any retry (design.md D3/D5).
    The boundary itself never retries.
    """


class CLIJSONError(CLIError):
    """Structured output was required but could not be decoded."""


class TargetResolutionError(ValueError):
    """Repository targeting failed before any platform access."""


class TargetConflictError(TargetResolutionError):
    """Explicit arguments and a resource URL identify different repositories."""


class MissingTargetError(TargetResolutionError):
    """No source supplied an unambiguous target repository."""


class AtomgitURLError(TargetResolutionError):
    """A URL, remote or configured baseUrl does not belong to a registered AtomGit service."""


# --------------------------------------------------------------------------- #
# CLI resolution and identity (task 2.1)
# --------------------------------------------------------------------------- #


def resolve_cli(env: Mapping[str, str] | None = None) -> str:
    """Return the absolute CLI executable path per design.md D2.

    Resolution order: explicit ``IBROBOT_ATOMGIT_CLI`` absolute path, else
    ``gitcode`` on PATH. ``gc`` is never considered (system Graphviz owns
    that name). An invalid explicit selection raises instead of falling
    back to PATH. Callers resolve once and pass ``cli=...`` to every run.
    """
    environ = os.environ if env is None else env
    explicit = environ.get(CLI_ENV_VAR, "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            raise CLISelectionError(f"{CLI_ENV_VAR} must be an absolute executable path, got: {explicit!r}")
        if not path.is_file():
            raise CLISelectionError(
                f"{CLI_ENV_VAR} points to a missing file: {explicit}; "
                "fix the variable or unset it to use 'gitcode' from PATH"
            )
        if not os.access(path, os.X_OK):
            raise CLISelectionError(f"{CLI_ENV_VAR} points to a non-executable file: {explicit}")
        return str(path)
    found = shutil.which(CLI_ENTRY_NAME, path=environ.get("PATH", ""))
    if not found:
        raise CLISelectionError(
            "official AtomGit CLI 'gitcode' was not found on PATH "
            f"and {CLI_ENV_VAR} is not set; install the verified CLI release "
            "(see .agents/skills/README.md) or set "
            f"{CLI_ENV_VAR} to its absolute path (do not use Graphviz 'gc')"
        )
    return str(Path(found).resolve())


@dataclass(frozen=True)
class CLIIdentity:
    executable: str
    version: tuple[int, ...]
    version_text: str
    commit: str | None
    raw_output: str


def check_cli_identity(
    cli: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    min_version: tuple[int, ...] = MIN_CLI_VERSION,
    timeout: float = 30.0,
) -> CLIIdentity:
    """Verify the executable really is the official CLI (task 2.1).

    Runs ``gitcode version`` and parses the verified output shape. This is
    the identity check that rejects same-name Graphviz binaries and stale
    versions; ``version`` does not hit the network, so it works unauthenticated.
    """
    executable = cli or resolve_cli(env)
    result = run(["version"], cli=executable, env=env, timeout=timeout)
    if not result.ok:
        raise CLISelectionError(
            f"CLI at {executable} failed 'version' probing (exit {result.returncode}); "
            "refusing to use it for AtomGit access",
            argv=result.argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    match = _VERSION_RE.search(result.stdout)
    if not match:
        raise CLISelectionError(
            f"executable at {executable} did not report 'gitcode version X.Y.Z'; "
            "it is not the official AtomGit CLI (Graphviz 'gc' and other same-name "
            "tools are rejected on purpose); install the verified release or set "
            f"{CLI_ENV_VAR}",
            argv=result.argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    version = tuple(int(part) for part in match.groups())
    if version < min_version:
        raise CLISelectionError(
            f"AtomGit CLI version {version_text(version)} at {executable} is older than the "
            f"verified minimum {version_text(min_version)} (capability evidence is bound to it); "
            "upgrade the CLI before running collaboration workflows",
            argv=result.argv,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    commit_match = _COMMIT_RE.search(result.stdout)
    return CLIIdentity(
        executable=executable,
        version=version,
        version_text=version_text(version),
        commit=commit_match.group(1) if commit_match else None,
        raw_output=result.stdout,
    )


def version_text(version: tuple[int, ...]) -> str:
    return ".".join(str(part) for part in version)


@dataclass(frozen=True)
class CLIAuthStatus:
    logged_in: bool
    returncode: int
    hostname: str = DEFAULT_WEB_HOST
    stdout: str = ""
    stderr: str = ""
    guidance: str = AUTH_LOGIN_GUIDANCE


def check_auth_status(
    cli: str | None = None,
    env: Mapping[str, str] | None = None,
    *,
    timeout: float = 30.0,
) -> CLIAuthStatus:
    """Diagnose CLI authentication for the pinned host (task 2.1).

    Uses ``auth status`` (never ``--show-token``). Exit 4 means "not logged
    in" and is a normal diagnostic result, not an error. Any other nonzero
    exit is unexpected and raises a classified error.
    """
    result = run(["auth", "status"], cli=cli, env=env, timeout=timeout)
    if result.returncode == 0:
        return CLIAuthStatus(True, 0, stdout=result.stdout, stderr=result.stderr)
    if result.returncode == EXIT_AUTH:
        return CLIAuthStatus(False, EXIT_AUTH, stdout=result.stdout, stderr=result.stderr)
    raise CLIExecutionError(
        f"unexpected 'auth status' exit code {result.returncode}",
        argv=result.argv,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )


# --------------------------------------------------------------------------- #
# Execution boundary (task 2.3)
# --------------------------------------------------------------------------- #


@dataclass
class CLIResult:
    """Outcome of one CLI invocation.

    ``ok`` reflects the exit code only. A nonzero exit that still printed
    JSON keeps ``data`` populated for caller inspection but ``ok`` stays
    False: nonzero exits are never treated as complete success (spec
    atomgit-cli-access, structured reads).
    """

    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    data: Any = None
    json_error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def failure(self) -> CLIExecutionError | None:
        """Classified error for a nonzero exit, or None when ``ok``.

        Message text is sanitized (see :func:`_command_summary`): raw CLI
        output and flag values never enter the message, because stderr may
        echo bodies or credentials we do not control. Full argv/stdout/
        stderr stay in the structured attributes for caller-controlled
        reconciliation logging.
        """
        if self.ok:
            return None
        message = f"AtomGit CLI failed with exit code {self.returncode}: {_command_summary(self.argv)}"
        if self.returncode == EXIT_AUTH:
            message = (
                f"{message}\nnot authenticated or lacking permission; if not logged in, run: {AUTH_LOGIN_GUIDANCE}"
            )
            return CLIAuthError(
                message, argv=self.argv, returncode=self.returncode, stdout=self.stdout, stderr=self.stderr
            )
        error_class: type[CLIExecutionError] = {
            EXIT_NOT_FOUND: CLINotFoundError,
            EXIT_CONFLICT: CLIConflictError,
        }.get(self.returncode, CLIExecutionError)
        return error_class(message, argv=self.argv, returncode=self.returncode, stdout=self.stdout, stderr=self.stderr)

    def require_data(self) -> Any:
        """Return decoded JSON only for a complete successful read (task 2.3).

        Enforces the spec completeness constraint: a nonzero exit or an
        undecodable payload is never accepted as complete success, even
        when partial JSON parsed. Partial data remains on ``self.data``
        for explicit diagnostics: callers needing it must inspect the
        result object directly instead of going through this entry point.
        """
        failure = self.failure()
        if failure is not None:
            raise failure
        if self.data is None:
            reason = self.json_error or "empty stdout"
            raise CLIJSONError(
                f"AtomGit CLI produced no usable JSON ({reason}): {_command_summary(self.argv)}",
                argv=self.argv,
                returncode=self.returncode,
                stdout=self.stdout,
                stderr=self.stderr,
            )
        return self.data


def run(
    args: list[str],
    *,
    cli: str | None = None,
    env: Mapping[str, str] | None = None,
    stdin_text: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    cwd: str | Path | None = None,
    no_interactive: bool = True,
) -> CLIResult:
    """Execute the CLI once with an argument array — no shell, no retry.

    - Arguments pass through argv untouched (``shell=False``): multiline
      bodies, quotes, backticks and ``$``-expressions reach the CLI verbatim.
    - Bodies that support it should use ``--body-file -`` with ``stdin_text``.
    - Without ``stdin_text`` stdin is /dev/null so prompts cannot hang.
    - ``GC_HOST=atomgit.com`` is pinned in the child environment.
    - Timeouts raise :class:`CLITimeoutError` carrying full argv; for writes
      the outcome is unknown and must be reconciled, never blindly retried.
    """
    executable = cli or resolve_cli(env)
    argv = [executable, *[str(arg) for arg in args]]
    if no_interactive:
        argv.append("--no-interactive")
    child_env = dict(os.environ if env is None else env)
    child_env["GC_HOST"] = DEFAULT_WEB_HOST
    try:
        completed = subprocess.run(
            argv,
            input=stdin_text,
            stdin=None if stdin_text is not None else subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd,
            env=child_env,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _safe_text(exc.stdout)
        stderr = _safe_text(exc.stderr)
        raise CLITimeoutError(
            f"AtomGit CLI invocation timed out after {timeout}s "
            f"(outcome unknown — reconcile before any retry): {_command_summary(argv)}",
            argv=argv,
            stdout=stdout,
            stderr=stderr,
            timeout=timeout,
        ) from exc
    return CLIResult(
        argv=argv,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _safe_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def run_json(args: list[str], **kwargs: Any) -> CLIResult:
    """Run and decode stdout as JSON when possible (task 2.3).

    Never raises for a nonzero exit or undecodable output: callers gate on
    ``result.ok`` plus ``result.data`` (or :meth:`CLIResult.require_data`).
    """
    result = run(args, **kwargs)
    stdout = result.stdout.strip()
    if stdout:
        try:
            result.data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            result.json_error = f"invalid JSON: {exc}"
    else:
        result.json_error = "empty stdout"
    return result


@dataclass
class PaginatedResult:
    """Complete-or-truncated result of a page loop (task 2.3).

    ``truncated`` distinguishes incomplete reads from empty collections:
    an empty first page yields ``items == []`` with ``truncated == False``.
    """

    items: list[Any] = field(default_factory=list)
    pages: int = 0
    truncated: bool = False
    stop_reason: str = "complete"  # complete | max-pages | failed-page
    last_result: CLIResult | None = None

    @property
    def complete(self) -> bool:
        return not self.truncated


def run_paginated(
    build_page_args: Callable[[int], list[str]],
    *,
    per_page: int = 100,
    max_pages: int = MAX_PAGES,
    cli: str | None = None,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> PaginatedResult:
    """Loop CLI pages for commands without native ``--paginate`` (task 2.3).

    ``build_page_args(page)`` returns the argv for one page (the caller adds
    ``--page``/``--per-page`` or query parameters). The loop stops on a short
    page (complete), on ``max_pages`` (truncated, guards runaway loops the
    SDK also capped), or on a failed page (truncated, items collected so far
    preserved so callers can mark the collection incomplete and stop before
    dependent side effects instead of reporting a false "no comments").
    """
    items: list[Any] = []
    result: CLIResult | None = None
    for page in range(1, max_pages + 1):
        result = run_json(build_page_args(page), cli=cli, env=env, timeout=timeout)
        if not result.ok or result.data is None:
            return PaginatedResult(items, page, truncated=True, stop_reason="failed-page", last_result=result)
        batch = result.data if isinstance(result.data, list) else [result.data]
        items.extend(batch)
        if len(batch) < per_page:
            return PaginatedResult(items, page, truncated=False, stop_reason="complete", last_result=result)
    return PaginatedResult(items, max_pages, truncated=True, stop_reason="max-pages", last_result=result)


# --------------------------------------------------------------------------- #
# Repository/URL/remote targeting (task 2.2)
# --------------------------------------------------------------------------- #


def normalize_host(host: str) -> str:
    """Normalize a web hostname: lowercase, no leading ``www.`` (design.md D2)."""
    host = (host or "").strip().lower()
    if host.startswith("www."):
        host = host[len("www.") :]
    return host


@dataclass(frozen=True)
class RepoTarget:
    owner: str
    repo: str
    origin: str  # explicit | url | config | upstream | source
    url_info: dict[str, Any] | None = None

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"


def parse_atomgit_url(url: str) -> dict[str, Any]:
    """Parse an AtomGit web URL into repository/resource context.

    Keeps the SDK-supported forms (contracts.md §4): repo, ``pull(s)``/
    ``merge_request(s)`` with number, ``issue(s)`` with number, and
    ``tree``/``commits``/``blob``/``src`` with branch. Unlike the SDK,
    only the registered project host ``atomgit.com`` is accepted: other
    services (e.g. gitcode.com) are rejected before access instead of being
    silently redirected (design.md D2).
    """
    if not url or not url.strip():
        raise AtomgitURLError("cannot parse an empty URL")
    normalized = url.strip() if "://" in url.strip() else f"https://{url.strip()}"

    parsed = urlparse(normalized)
    host = normalize_host(parsed.hostname or "")
    if host not in WEB_HOST_TO_API_HOST:
        raise AtomgitURLError(
            f"unsupported URL host {host!r}: known AtomGit web hosts are "
            f"{sorted(WEB_HOST_TO_API_HOST)} (service mappings must be verified before registration)"
        )
    path_parts = [part for part in parsed.path.split("/") if part]
    if len(path_parts) < 2:
        raise AtomgitURLError(f"URL does not identify an owner/repository: {url}")
    owner = path_parts[0]
    repo = path_parts[1].removesuffix(".git")
    info: dict[str, Any] = {"owner": owner, "repo": repo, "host": host}
    if len(path_parts) == 2:
        return info
    resource = path_parts[2]
    if resource in {"pull", "pulls", "merge_request", "merge_requests"}:
        if len(path_parts) >= 4 and path_parts[3].isdigit():
            info["pr_number"] = int(path_parts[3])
            return info
        raise AtomgitURLError(f"PR URL is missing a numeric PR number: {url}")
    if resource in {"issue", "issues"}:
        if len(path_parts) >= 4 and path_parts[3].isdigit():
            info["issue_number"] = int(path_parts[3])
            return info
        raise AtomgitURLError(f"Issue URL is missing a numeric issue number: {url}")
    if resource in {"tree", "commits", "blob", "src"} and len(path_parts) >= 4:
        info["branch"] = "/".join(path_parts[3:])
        return info
    return info


def parse_git_remote_url(url: str) -> dict[str, Any] | None:
    """Parse an https or ssh Git remote URL; None when unparseable.

    Hosts are checked against the registered AtomGit mapping by the caller
    (upstream tier), not here: this function only extracts the components.
    """
    text = (url or "").strip()
    if not text:
        return None
    if "://" in text:
        parsed = urlparse(text)
        host = normalize_host(parsed.hostname or "")
        path = parsed.path or ""
    else:
        match = re.match(r"^git@([^:/]+)[:/](.+)$", text)
        if not match:
            return None
        host = normalize_host(match.group(1))
        path = f"/{match.group(2)}"
    parts = [part for part in path.split("/") if part]
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1].removesuffix(".git")
    if not owner or not repo:
        return None
    return {"owner": owner, "repo": repo, "host": host}


def read_git_remotes(cwd: str | Path | None = None) -> dict[str, list[str]]:
    """Read local Git remotes for the ``upstream`` targeting tier (task 2.2).

    Runs ``git remote -v`` and maps each remote name to its unique URLs
    (fetch and push collapsed). Local Git reads stay outside the platform
    API requirement, so this is a plain local subprocess — no CLI, no
    network. It is a best-effort input to the *last* targeting tier: when
    Git is not installed, ``cwd`` is not a repository, or the command
    fails, an empty mapping is returned instead of raising, so explicit,
    URL and config targets keep working in non-repository contexts
    (design.md D2: “缺 upstream 但有明确/config 目标时可工作”). URL forms
    are preserved verbatim; host/owner/repo interpretation belongs to
    :func:`parse_git_remote_url` and :func:`resolve_target`.
    """
    try:
        completed = subprocess.run(
            ["git", "remote", "-v"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            cwd=cwd,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if completed.returncode != 0:
        return {}
    remotes: dict[str, list[str]] = {}
    for line in completed.stdout.splitlines():
        if not line.strip() or "\t" not in line:
            continue
        name, rest = line.split("\t", 1)
        # Each line is "<name>\t<url> (fetch|push)"; strip the parenthesized kind.
        url = rest.rsplit("(", 1)[0].strip() if rest.rstrip().endswith(")") else rest.strip()
        entries = remotes.setdefault(name.strip(), [])
        if url and url not in entries:
            entries.append(url)
    return remotes


def load_config(config_path: str | Path = "config.json", *, required: bool = False) -> dict[str, Any] | None:
    """Load config.json for non-sensitive owner/repo defaults only.

    The legacy ``atomgit.token`` is never read, validated or printed. A
    missing explicit config path (``required=True``) is an error; a missing
    default config falls through to the next targeting tier (design.md D2).
    Unparseable JSON is an error in both cases.
    """
    path = Path(config_path)
    if not path.is_file():
        if required:
            raise TargetResolutionError(f"config file not found: {path}")
        return None
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TargetResolutionError(f"invalid JSON in config file {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise TargetResolutionError(f"config file {path} must contain a JSON object")
    return config


def config_repo_defaults(config: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Extract atomgit.owner/atomgit.repo defaults; token is never touched."""
    atomgit = config.get("atomgit")
    if not isinstance(atomgit, Mapping):
        return None, None
    owner = atomgit.get("owner")
    repo = atomgit.get("repo")
    if (owner is None) != (repo is None):
        raise TargetResolutionError(
            "config atomgit section must define both owner and repo, not only one "
            f"(owner={'set' if owner else 'missing'}, repo={'set' if repo else 'missing'})"
        )
    return owner, repo


def validate_config_base_url(config: Mapping[str, Any]) -> None:
    """Validate ``atomgit.baseUrl`` against the registered service mapping.

    The boundary never routes requests through ``baseUrl`` (``GC_HOST`` pins
    the web host), but a config pointing at an unregistered service disagrees
    with the service every command will actually access, so it is rejected
    before any request instead of being silently ignored (design.md D2). Both
    sides of the registered mapping are accepted: the API host
    ``api.atomgit.com`` and the web-host alias ``atomgit.com``. A missing,
    null or empty ``baseUrl`` is not an error.
    """
    atomgit = config.get("atomgit")
    if not isinstance(atomgit, Mapping):
        return
    base_url = atomgit.get("baseUrl")
    if base_url is None or not str(base_url).strip():
        return
    text = str(base_url).strip()
    if "://" not in text:
        text = f"https://{text}"
    host = normalize_host(urlparse(text).hostname or "")
    registered = set(WEB_HOST_TO_API_HOST) | set(WEB_HOST_TO_API_HOST.values())
    if host not in registered:
        raise AtomgitURLError(
            f"config atomgit.baseUrl host {host!r} is not a registered AtomGit service "
            f"(registered: {sorted(registered)}); requests are pinned to the atomgit.com "
            "service via GC_HOST and never redirected — fix or remove baseUrl"
        )


def resolve_target(
    *,
    owner: str | None = None,
    repo: str | None = None,
    url: str | None = None,
    config_path: str | Path = "config.json",
    explicit_config: bool = False,
    remotes: Mapping[str, list[str]] | None = None,
) -> RepoTarget:
    """Resolve an unambiguous target repository (task 2.2, design.md D2).

    Priority: explicit owner/repo and resource URL (same tier — a mismatch
    is a rejected conflict, not a silent override) → config defaults → a
    unique ``upstream`` remote. ``origin`` is never used as the target: it
    only identifies the fork source. When nothing resolves both fields the
    error explains what to supply instead of guessing.

    Whenever a config file participates, its ``atomgit.baseUrl`` host must
    match the registered service mapping (:func:`validate_config_base_url`),
    even when another tier supplies the owner/repo.

    Comparisons are exact strings: canonical owner/repo identity is only
    provable via the platform (design.md D2), so case-only differences are
    reported as conflicts rather than folded away.
    """
    url_info = parse_atomgit_url(url) if url else None
    if url_info is not None:
        for field_name, explicit_value in (("owner", owner), ("repo", repo)):
            url_value = url_info.get(field_name)
            if explicit_value and url_value and explicit_value != url_value:
                raise TargetConflictError(
                    f"explicit {field_name}={explicit_value!r} conflicts with URL "
                    f"{field_name}={url_value!r} ({url}); canonical identity cannot be "
                    "verified locally, so supply one unambiguous target"
                )

    config = load_config(config_path, required=explicit_config)
    config_owner = config_repo = None
    if config is not None:
        validate_config_base_url(config)
        config_owner, config_repo = config_repo_defaults(config)

    resolved_owner = owner or (url_info or {}).get("owner") or config_owner
    resolved_repo = repo or (url_info or {}).get("repo") or config_repo

    origin = "explicit" if (owner or repo) else ("url" if url_info else ("config" if config_owner else "upstream"))

    if resolved_owner and resolved_repo:
        return RepoTarget(resolved_owner, resolved_repo, origin, url_info)

    if resolved_owner or resolved_repo:
        missing = "repo" if resolved_owner else "owner"
        raise MissingTargetError(
            f"target repository is missing {missing!r} "
            f"(owner={resolved_owner!r}, repo={resolved_repo!r}); supply both explicitly, "
            "via --url, or through config defaults"
        )

    upstream_urls = (remotes or {}).get("upstream") or []
    if len(upstream_urls) == 0:
        raise MissingTargetError(
            "missing target repository: no explicit owner/repo, no resource URL, "
            "no config defaults, and no 'upstream' remote (origin is a fork source "
            "and is never used as the target)"
        )
    if len(upstream_urls) > 1:
        raise TargetResolutionError(f"'upstream' remote has {len(upstream_urls)} URLs; cannot pick one unambiguously")
    remote_info = parse_git_remote_url(upstream_urls[0])
    if remote_info is None:
        raise TargetResolutionError(f"cannot parse 'upstream' remote URL: {upstream_urls[0]!r}")
    if remote_info["host"] not in WEB_HOST_TO_API_HOST:
        raise AtomgitURLError(
            f"'upstream' remote points at {remote_info['host']!r}, which is not a registered "
            f"AtomGit web host ({sorted(WEB_HOST_TO_API_HOST)}); refusing to guess a service mapping"
        )
    return RepoTarget(remote_info["owner"], remote_info["repo"], "upstream", None)


def resolve_source(fork_owner: str, target: RepoTarget) -> RepoTarget:
    """Resolve the cross-fork source repository, kept distinct from target.

    The fork shares the target repository name under the fork owner
    (SDK semantics, contracts.md §5.3): ``head`` is ``fork_owner:branch``.
    The target stays untouched for ``-R``/``--repo`` arguments.
    """
    if not fork_owner or not fork_owner.strip():
        raise TargetResolutionError("fork owner must be a non-empty string")
    if fork_owner.strip() == target.owner:
        raise TargetConflictError(
            f"fork owner {fork_owner!r} equals target owner {target.owner!r}; "
            "a cross-fork PR needs distinct source/target owners"
        )
    return RepoTarget(fork_owner.strip(), target.repo, "source", None)
