"""The standard launcher keeps exactly the environment the RoboDojo client reads.

``run_eval_client.sh`` re-execs itself with an allowlisted environment so RoboDojo's own
interpreters do not import the ROS (Python 3.10) packages. A setting the client reads from
the environment but the allowlist drops silently falls back to its default.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

CLIENT = Path(__file__).resolve().parents[1] / "robodojo_client"
SCRIPT = CLIENT / "run_eval_client.sh"
CLEAN_GUARD = 'if [[ "${IBROBOT_CLIENT_ENV_CLEAN:-}" != 1 ]]; then'


def _env_read_by_client() -> set[str]:
    names: set[str] = set()
    for source in CLIENT.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        names |= set(re.findall(r"os\.(?:environ\.get|getenv)\(\s*[\"']([A-Z0-9_]+)[\"']", text))
        names |= set(re.findall(r"os\.environ\[\s*[\"']([A-Z0-9_]+)[\"']\s*\]", text))
    return names


def _allowlist() -> list[str]:
    block = re.search(r"keep=\(([^)]*)\)", SCRIPT.read_text(encoding="utf-8"))
    assert block, "run_eval_client.sh has no keep=(...) allowlist"
    return block.group(1).split()


def _clean_block() -> str:
    """The script's re-exec block, ending in ``env`` instead of the client (prints what survives)."""
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    start = lines.index(CLEAN_GUARD)
    end = next(i for i in range(start, len(lines)) if lines[i] == "fi")
    block = "\n".join(lines[start : end + 1])
    reexec = re.search(r'exec env -i "\$\{args\[@\]\}" .*', block)
    assert reexec, "the re-exec line changed; update this test"
    return block.replace(reexec.group(0), 'exec env -i "${args[@]}" env')


def test_every_variable_the_client_reads_is_allowlisted_and_none_is_stale():
    read = {name for name in _env_read_by_client() if name.startswith(("IBROBOT_", "ROBODOJO_"))}
    keep = set(_allowlist())
    assert "IBROBOT_RUNTIME_RECONNECT_S" in read
    assert read <= keep, f"read by the client but dropped by run_eval_client.sh: {sorted(read - keep)}"
    stale = {name for name in keep if name.startswith("IBROBOT_RUNTIME_")} - read
    assert not stale, f"allowlisted but read by nothing: {sorted(stale)}"


def test_the_launcher_passes_the_reconnect_budget_and_drops_ros_paths():
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/tmp"),
        "IBROBOT_RUNTIME_RECONNECT_S": "7",
        "PYTHONPATH": "/opt/ros/humble/lib/python3.10/site-packages",
        "AMENT_PREFIX_PATH": "/opt/ros/humble",
    }
    out = subprocess.run(
        ["bash", "-c", _clean_block()], env=env, capture_output=True, text=True, timeout=30, check=True
    ).stdout
    survived = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    assert survived["IBROBOT_RUNTIME_RECONNECT_S"] == "7"
    assert survived["IBROBOT_CLIENT_ENV_CLEAN"] == "1"
    assert "PYTHONPATH" not in survived and "AMENT_PREFIX_PATH" not in survived
