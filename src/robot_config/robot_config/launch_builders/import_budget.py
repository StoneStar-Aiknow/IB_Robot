"""Shared import-budget helpers for launch-builder registry tests.

Openspec change ``migrate-business-launch-builders``, design decision 6:
the subprocess probing and core-budget computation used by robot_config's
import-budget test are reusable so every package that contributes launch
builders through the ``ibrobot.launch_builders`` entry point group can pin
the same rule — importing that package's ``launch_builders`` modules must
not pull in non-stdlib top-level packages beyond the registry core budget
and the package itself.

This module deliberately depends on the standard library only: it is
imported by tests of several packages and must never contribute to any
import budget itself.
"""

from __future__ import annotations

import json
import subprocess
import sys

# Fresh-interpreter snippet: print the sorted non-stdlib top-level packages
# present in ``sys.modules`` (ignoring underscore-prefixed private modules
# and the stdlib list, with the two compatibility exceptions).
_COLLECT_SET = (
    "import sys\n"
    "STDLIB = set(sys.stdlib_module_names) | {'__future__', 'typing_extensions'}\n"
    "tops = {n.split('.')[0] for n in sys.modules"
    " if n.split('.')[0] not in STDLIB and not n.startswith('_')}\n"
    "print(json.dumps(sorted(tops)))\n"
)

# Packages allowed on top of the core budget: ``ament_index_python`` backs
# the ament share-path lookups used by a few domain modules, and
# ``robot_runtime`` is a HARD robot_config dependency (``package.xml``
# declares ``<depend>robot_runtime</depend>``) that the pre-migration
# declaration chain already imported through ``robot_config.utils``.
HARD_DEPENDENCY_ALLOWANCE = frozenset({"ament_index_python", "robot_runtime"})


def imported_top_level_packages(statement: str, *, timeout: float = 120) -> set[str]:
    """Execute ``statement`` in a fresh interpreter, return its top-level packages.

    The statement runs before the collection snippet, so the result is the
    set of non-stdlib top-level packages present after executing it. Running
    in a subprocess keeps the measurement independent of whatever this
    process imported for unrelated reasons (same pattern as the facade
    tests). A non-zero exit raises with the captured output.
    """
    code = "import json\n" + statement + _COLLECT_SET
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise AssertionError(f"statement failed: {statement!r}\nstdout={result.stdout}\nstderr={result.stderr}")
    return set(json.loads(result.stdout))


def core_import_budget() -> set[str]:
    """The registry core budget: importing the registry core + launch + launch_ros.

    This is the allowed baseline for anything executed during launch
    description generation (registry construction, argument declaration):
    ``launch`` / ``launch_ros`` pull launch, rclpy, message packages, yaml,
    ...; anything beyond this set plus :data:`HARD_DEPENDENCY_ALLOWANCE`
    means a domain module regressed to a top-level cross-package import.
    """
    return imported_top_level_packages(
        "import launch\nimport launch_ros\nimport robot_config.launch_builders.registry\n"
    )
