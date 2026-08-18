import os
import re
import subprocess
import tempfile
from urllib.parse import urlparse

_VERIFIED_TREE_RE = re.compile(
    r"\*\*Verified tree:\*\*\s*`?([0-9a-f]{40})(?![0-9a-f])`?",
    re.IGNORECASE,
)
_WIP_PREFIX_RE = re.compile(r"^(?:\s*\[WIP\]\s*)+", re.IGNORECASE)
_PACKAGE_DEPENDENCY_RE = re.compile(
    r"^[+-](?![+-])\s*</?(?:depend|build_depend|build_export_depend|buildtool_depend|buildtool_export_depend|exec_depend|test_depend|doc_depend|group_depend)\b",
    re.MULTILINE,
)
_GLOBAL_GATE_FILES = {
    "CMakeLists.txt",
    "pyproject.toml",
    "scripts/build.sh",
    "scripts/install_ros.sh",
    "scripts/setup.sh",
    "scripts/setup/python_venv.sh",
    "scripts/setup/verify_env.sh",
}


def file_triggers_dual_docker_gate(filename: str, patch: str = "") -> bool:
    if filename in _GLOBAL_GATE_FILES:
        return True
    if filename.startswith("scripts/setup/platforms/") and filename.endswith(".sh"):
        return True
    if filename.startswith("requirements/") and filename.endswith(".txt"):
        return True
    if not filename.endswith("/package.xml"):
        return False
    # Missing patches are treated as gated so API truncation cannot bypass the check.
    return not patch or bool(_PACKAGE_DEPENDENCY_RE.search(patch))


def is_wip_title(title: str) -> bool:
    return bool(_WIP_PREFIX_RE.match(title or ""))


def extract_verified_tree(description: str) -> str | None:
    matches = [match.lower() for match in _VERIFIED_TREE_RE.findall(description or "")]
    if len(matches) != 1:
        return None
    return matches[0]


def _run_git(args: list[str], cwd: str) -> str:
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise ValueError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def resolve_pr_head_tree(pr: dict) -> str:
    head = pr.get("head") or {}
    head_sha = (head.get("sha") or "").lower()
    head_ref = head.get("ref") or ""
    repo_url = (head.get("repo") or {}).get("html_url") or ""
    parsed_url = urlparse(repo_url)
    if len(head_sha) != 40 or not head_ref or parsed_url.scheme != "https" or not parsed_url.netloc:
        raise ValueError("PR head commit, branch, or HTTPS repository URL is unavailable")

    with tempfile.TemporaryDirectory(prefix="ibrobot-pr-head-") as git_dir:
        _run_git(["init", "--bare"], git_dir)
        _run_git(["fetch", "--no-tags", "--depth=1", repo_url, f"refs/heads/{head_ref}"], git_dir)
        fetched_sha = _run_git(["rev-parse", "FETCH_HEAD"], git_dir).lower()
        if fetched_sha != head_sha:
            raise ValueError(f"PR head changed during tree resolution: API={head_sha}, fetched={fetched_sha}")
        return _run_git(["rev-parse", "FETCH_HEAD^{tree}"], git_dir).lower()
