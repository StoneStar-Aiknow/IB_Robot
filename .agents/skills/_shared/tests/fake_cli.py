"""Fake AtomGit CLI used by the shared-boundary behavior tests (task 2.4).

This module is executed through a generated launcher script (shebang points
at the current Python interpreter) so the boundary treats it exactly like a
real executable: argv arrives via execve, stdin is a real pipe/DEVNULL, and
the exit code is a real process exit. Every invocation is appended as one
JSON line to ``$FAKE_CLI_RECORD`` for assertions.

Responses are driven by ``$FAKE_CLI_SCENARIO`` (JSON document)::

    {
      "rules": [
        {"argv_contains": ["auth", "status"], "stdout": "...", "stderr": "",
         "exit": 4},
        {"argv_contains": ["--page", "2"], "stdout": "[{...}]", "exit": 0},
        {"argv_contains": ["pr", "view"], "sleep": 3, "exit": 0}
      ],
      "default": {"stdout": "", "stderr": "unsupported in scenario", "exit": 64}
    }

The first rule whose ``argv_contains`` elements all appear in argv wins.
``sleep`` simulates a hanging CLI for timeout tests; ``consume_stdin`` forces
a stdin read so tests can prove the boundary never leaves stdin on a TTY.
"""

from __future__ import annotations

import json
import os
import sys
import time

RECORD_ENV = "FAKE_CLI_RECORD"
SCENARIO_ENV = "FAKE_CLI_SCENARIO"
RECORDED_ENV_KEYS = ("GC_HOST", "IBROBOT_ATOMGIT_CLI")
_JSON_ERRORS = (TypeError, ValueError)


def main() -> None:
    argv = sys.argv[1:]
    stdin_text = ""
    if os.environ.get("FAKE_CLI_CONSUME_STDIN", "1") == "1":
        stdin_text = sys.stdin.read()

    rule = _match_rule(argv)
    _record(argv, stdin_text)

    if rule.get("sleep"):
        time.sleep(float(rule["sleep"]))
    stdout = rule.get("stdout", "")
    stderr = rule.get("stderr", "")
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(int(rule.get("exit", 0)))


def _match_rule(argv: list[str]) -> dict:
    scenario = _load_scenario()
    for rule in scenario.get("rules", []):
        needles = rule.get("argv_contains") or []
        if all(element in argv for element in needles):
            return rule
    default = scenario.get("default")
    if isinstance(default, dict):
        return default
    return {"stdout": "", "stderr": "no rule matched", "exit": 64}


def _load_scenario() -> dict:
    path = os.environ.get(SCENARIO_ENV, "")
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as handle:
            scenario = json.load(handle)
        return scenario if isinstance(scenario, dict) else {}
    except _JSON_ERRORS:
        return {}


def _record(argv: list[str], stdin_text: str) -> None:
    path = os.environ.get(RECORD_ENV, "")
    if not path:
        return
    entry = {
        "argv": argv,
        "stdin": stdin_text,
        "env": {key: os.environ.get(key, "") for key in RECORDED_ENV_KEYS},
    }
    try:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


if __name__ == "__main__":
    main()
