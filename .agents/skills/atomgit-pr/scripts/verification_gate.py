import re

VERIFIED_COMMIT_LABEL = "Verified commit"
_VERIFIED_COMMIT_RE = re.compile(
    rf"\*\*{VERIFIED_COMMIT_LABEL}:\*\*\s*`?([0-9a-f]{{40}})(?![0-9a-f])`?",
    re.IGNORECASE,
)
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


def extract_verified_commit(description: str) -> str | None:
    matches = [match.lower() for match in _VERIFIED_COMMIT_RE.findall(description or "")]
    if len(matches) != 1:
        return None
    return matches[0]


def validate_verified_commit(description: str, expected_head: str) -> str:
    verified_commit = extract_verified_commit(description)
    if verified_commit is None:
        raise ValueError(
            f"dual Docker verification requires exactly one '**{VERIFIED_COMMIT_LABEL}:** `<40-character SHA>`' field"
        )
    expected_head = expected_head.lower()
    if verified_commit != expected_head:
        raise ValueError(
            f"Docker verification commit {verified_commit} does not match the latest PR commit {expected_head}"
        )
    return verified_commit
