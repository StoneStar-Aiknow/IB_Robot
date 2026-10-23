"""Guard against reintroducing old framework import paths.

The registration framework lives in ``robot_config.launch_registry`` and
the old ``robot_config.launch_builders.<framework module>`` paths were
removed with the aliases: the modules only ever existed inside this
repository's own history, so no published consumer can be importing
them. This test keeps it that way — no in-repo code may mention the old
paths, and importing them must fail instead of silently resolving.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

SRC_ROOT = Path(__file__).resolve().parents[2]

OLD_PATH_RE = re.compile(
    r"robot_config\.launch_builders\.(?:registry|preparation|orchestrator|launch_options|builtin_registry|builder_support|import_budget)\b"
)

FRAMEWORK_MODULES = (
    "registry",
    "preparation",
    "orchestrator",
    "launch_options",
    "builtin_registry",
    "builder_support",
    "import_budget",
)


def test_no_source_file_uses_the_old_framework_paths():
    offenders = []
    for path in sorted(SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(SRC_ROOT).as_posix()
        if "__pycache__" in path.parts:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if OLD_PATH_RE.search(line):
                offenders.append(f"{relative}:{line_number}: {line.strip()}")
    assert offenders == [], (
        "old framework import paths found in src/ (use robot_config.launch_registry.<module> instead):\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("module_name", FRAMEWORK_MODULES)
def test_the_old_framework_paths_are_not_importable(module_name):
    """The removed alias modules must fail to import, not resolve silently."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"robot_config.launch_builders.{module_name}")
