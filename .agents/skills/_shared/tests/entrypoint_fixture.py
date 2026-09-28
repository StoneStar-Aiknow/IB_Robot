"""Executable fake-CLI fixture shared by entrypoint behavior tests.

Extracted from ``test_entrypoint_fake_cli.py`` (task 3.1) so later migration
tasks can drive their real script entry points against the same harness
without duplicating the launcher logic. The launcher points its ``sys.path``
at this directory so the subprocess imports the shared ``fake_cli.py``
executable (task 2.4); everything else — scenario configuration, environment
shaping and invocation recording — is identical for every skill.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

VERSION_STDOUT = (
    "gitcode version 0.14.1\n"
    "  commit: 21e19618984ec0628192f6283bca066c76e12511\n"
    "  built:  2026-09-24T05:07:54Z\n"
    "https://gitcode.com/gitcode-cli/cli\n"
)
AUTH_LOGIN_GUIDANCE = "gitcode auth login --hostname atomgit.com --with-token"


class EntrypointFakeCLI:
    """Executable fake ``gitcode`` reusing the shared tests/fake_cli.py."""

    def __init__(self, tmp_path: Path, *, tests_dir: Path = TESTS_DIR) -> None:
        self.bin_dir = tmp_path / "bin"
        self.bin_dir.mkdir()
        self.exe = self.bin_dir / "gitcode"
        self.exe.write_text(
            f"#!{sys.executable}\n"
            f"import sys\n"
            f"sys.path.insert(0, {str(tests_dir)!r})\n"
            f"import fake_cli\n"
            f"fake_cli.main()\n"
        )
        self.exe.chmod(0o755)
        self.scenario_path = tmp_path / "scenario.json"
        self.record_path = tmp_path / "record.jsonl"

    def configure(self, rules: list[dict], default: dict | None = None) -> None:
        scenario = {"rules": [{"argv_contains": ["version"], "stdout": VERSION_STDOUT, "exit": 0}, *rules]}
        if default is not None:
            scenario["default"] = default
        self.scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

    def env(self, **extra: str) -> dict[str, str]:
        env = dict(os.environ)
        env["PATH"] = os.pathsep.join([str(self.bin_dir), env.get("PATH", "")])
        env["PYTHONIOENCODING"] = "utf-8"
        env["FAKE_CLI_SCENARIO"] = str(self.scenario_path)
        env["FAKE_CLI_RECORD"] = str(self.record_path)
        env["IBROBOT_ATOMGIT_CLI"] = str(self.exe)
        env.update(extra)
        return env

    def invocations(self) -> list[dict]:
        if not self.record_path.is_file():
            return []
        return [json.loads(line) for line in self.record_path.read_text(encoding="utf-8").splitlines() if line]

    def argvs(self) -> list[list[str]]:
        return [invocation["argv"] for invocation in self.invocations()]

    def mutation_argv(self) -> list[list[str]]:
        mutation_tokens = {"comment", "edit", "create"}
        return [argv for argv in self.argvs() if mutation_tokens & set(argv)]

    def argv_of(self, command: list[str]) -> list[str]:
        matches = [argv for argv in self.argvs() if argv[: len(command)] == command]
        assert len(matches) == 1, f"expected exactly one {command} invocation, got {matches}"
        return matches[0]

    def invocation_of(self, command: list[str]) -> dict:
        matches = [invocation for invocation in self.invocations() if invocation["argv"][: len(command)] == command]
        assert len(matches) == 1, f"expected exactly one {command} invocation, got {matches}"
        return matches[0]
