"""Import-budget test for building the launch-builder registry.

Openspec change ``decentralize-launch-builder-registration``, design
decision 2: building the registry now imports every builtin domain module
(the ``BUILTIN_BUILDERS`` reference table), so the domain modules must keep
their cross-package run dependencies function-local. In a fresh subprocess,
the non-stdlib top-level packages present after ``build_builtin_registry()``
must be a subset of what importing the registry core + ``launch`` +
``launch_ros`` pulls in, plus ``ament_index_python`` and robot_config's
hard dependency ``robot_runtime``.

Both sets are collected in subprocesses so the assertions cannot be
polluted by this pytest process importing business modules for other test
files (same pattern as ``test_launch_builders_facade.py``).
"""

from __future__ import annotations

import json
import subprocess
import sys

_COLLECT_SET = (
    "import sys\n"
    "STDLIB = set(sys.stdlib_module_names) | {'__future__', 'typing_extensions'}\n"
    "tops = {n.split('.')[0] for n in sys.modules"
    " if n.split('.')[0] not in STDLIB and not n.startswith('_')}\n"
    "print(json.dumps(sorted(tops)))\n"
)


def _collect_tops(prepare: str) -> set[str]:
    code = "import json\n" + prepare + _COLLECT_SET
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    return set(json.loads(result.stdout))


def test_builtin_registry_stays_within_the_core_import_budget():
    """build_builtin_registry() imports no optional business run dependencies.

    The allowed set is what the registry core itself needs (``launch`` /
    ``launch_ros`` pull launch, rclpy, message packages, yaml, ...), plus
    ``ament_index_python`` (used by ament share-path lookups in a few
    domain modules) and ``robot_runtime``: a HARD robot_config dependency
    (``package.xml`` declares ``<depend>robot_runtime</depend>``) that the
    pre-migration declaration chain already imported through
    ``robot_config.utils``, so it belongs to the core budget, not to the
    optional business packages. Anything beyond that — torch, tensormsg,
    lerobot, robot_teleop, embodied_common, xacro, inference_manifest,
    ... — means a domain module regressed to a top-level cross-package
    import.
    """
    core = _collect_tops("import launch\nimport launch_ros\nimport robot_config.launch_builders.registry\n")
    registry = _collect_tops(
        "from robot_config.launch_builders.builtin_registry import build_builtin_registry\n"
        "registry = build_builtin_registry()\n"
    )
    overflow = sorted(registry - core - {"ament_index_python", "robot_runtime"})
    assert overflow == [], (
        f"building the builtin registry pulled non-core top-level packages: {overflow}; "
        "move these imports inside the build functions (design decision 2)"
    )


def test_builtin_registry_does_not_import_known_business_packages():
    """Pin the known run dependencies to function-local imports.

    Even though the packages are installed in this environment, building
    the registry must not import them: an environment without them (the
    spec scenario: an optional business package is absent) must still be
    able to enumerate arguments and generate plans.
    """
    forbidden = {
        "robot_teleop",  # teleop.py (teleop groups)
        "embodied_common",  # mirror_session.py (session contract)
        "xacro",  # description.py (URDF processing)
        "inference_manifest",  # execution.py / benchmark.py via inference_config
        "tensormsg",  # benchmark.py via contract_utils
        "torch",  # benchmark.py via contract_utils
        "robot_navigation",  # navigation.py sub-builders
    }
    code = (
        "import json, sys\n"
        "from robot_config.launch_builders.builtin_registry import build_builtin_registry\n"
        "build_builtin_registry()\n"
        "loaded = sorted(n.split('.')[0] for n in sys.modules if n.split('.')[0] in " + repr(sorted(forbidden)) + ")\n"
        "print(json.dumps(loaded))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    loaded = set(json.loads(result.stdout))
    assert loaded == set(), (
        f"building the builtin registry imported business packages {sorted(loaded)}; "
        "their imports must stay inside the build functions"
    )
