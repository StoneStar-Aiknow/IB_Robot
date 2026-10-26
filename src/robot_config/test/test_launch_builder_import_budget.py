"""Import-budget test for building the launch-builder registry.

Openspec change ``decentralize-launch-builder-registration``, design
decision 2: building the registry now imports every builtin domain module
(the ``BUILTIN_BUILDERS`` reference table), so the domain modules must keep
their cross-package run dependencies function-local. In a fresh subprocess,
the non-stdlib top-level packages present after ``build_builtin_registry()``
(and, for Phase 2, ``build_launch_registry(entry_points=[])``) must be a
subset of what importing the registry core + ``launch`` + ``launch_ros``
pulls in, plus ``ament_index_python`` and robot_config's hard dependency
``robot_runtime``.

The subprocess probing and the core-budget computation live in
``robot_config.launch_registry.import_budget`` (openspec change
``migrate-business-launch-builders``, design decision 6) so every package
contributing launch builders can pin the same rule; this test keeps its
original assertions unchanged.
"""

from __future__ import annotations

import json
import subprocess
import sys

from robot_config.launch_registry.import_budget import (
    HARD_DEPENDENCY_ALLOWANCE,
    core_import_budget,
    imported_top_level_packages,
)

# The builtin entry modules that ``robot_config.launch_builders`` keeps
# owning after the framework split (openspec change
# ``split-launch-registry-package``): importing the registration framework
# must never load any of them.
BUILTIN_ENTRY_MODULES = (
    "robot_config.launch_builders.control",
    "robot_config.launch_builders.simulation",
    "robot_config.launch_builders.readiness",
    "robot_config.launch_builders.perception",
    "robot_config.launch_builders.benchmark",
    "robot_config.launch_builders.audio_io",
    "robot_config.launch_builders.execution",
    "robot_config.launch_builders.task_execution",
    "robot_config.launch_builders.embodied",
)


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
    core = core_import_budget()
    registry = imported_top_level_packages(
        "from robot_config.launch_registry.builtin_registry import build_builtin_registry\n"
        "registry = build_builtin_registry()\n"
    )
    overflow = sorted(registry - core - HARD_DEPENDENCY_ALLOWANCE)
    assert overflow == [], (
        f"building the builtin registry pulled non-core top-level packages: {overflow}; "
        "move these imports inside the build functions (design decision 2)"
    )


def test_launch_registry_with_extensions_stays_within_the_core_import_budget():
    """build_launch_registry(entry_points=[]) adds no imports of its own.

    Phase 2 (design decision 3) wraps the builtin table with entry point
    discovery; the discovery/validation machinery itself (importlib.metadata
    is stdlib) must stay within the same budget as the builtin table. Real
    discovery would import the DECLARED extension's own module — that is the
    point of registration — so the budget pins the machinery with the test
    injection seam (no extensions) instead of the ambient environment.
    """
    core = core_import_budget()
    registry = imported_top_level_packages(
        "from robot_config.launch_registry.builtin_registry import build_launch_registry\n"
        "registry = build_launch_registry(entry_points=[])\n"
    )
    overflow = sorted(registry - core - HARD_DEPENDENCY_ALLOWANCE)
    assert overflow == [], (
        f"building the launch registry pulled non-core top-level packages: {overflow}; "
        "the entry point discovery path must stay import-light (design decisions 2 and 3)"
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
        "embodied_common",  # business run dependency (session contract); post-migration the mirror_session builder lives in manipulation_execution, which must not be imported here either
        "xacro",  # description.py (URDF processing)
        "inference_manifest",  # execution.py / benchmark.py via inference_config
        "tensormsg",  # benchmark.py via contract_utils
        "torch",  # benchmark.py via contract_utils
        "robot_navigation",  # navigation.py sub-builders
    }
    loaded = imported_top_level_packages(
        "from robot_config.launch_registry.builtin_registry import build_builtin_registry\nbuild_builtin_registry()\n"
    )
    assert not (loaded & forbidden), (
        f"building the builtin registry imported business packages {sorted(loaded & forbidden)}; "
        "their imports must stay inside the build functions"
    )


def test_registry_core_import_loads_no_builtin_entry_modules():
    """Importing the framework core stays framework-only (design decision 3).

    ``robot_config.launch_registry.registry`` is the import-light entry of
    the split registration framework (openspec change
    ``split-launch-registry-package``): a fresh interpreter importing it may
    load framework modules and the infrastructure helpers they legitimately
    need (``launch_builders.tracing`` through launch_options), but never a
    builtin entry module (control, perception, execution, ...) — those load
    only when the builtin table or an extension factory is resolved.
    """
    code = (
        "import json\n"
        "import sys\n"
        "import robot_config.launch_registry.registry\n"
        "loaded = sorted(n for n in sys.modules if n.startswith('robot_config.'))\n"
        "print(json.dumps(loaded))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    loaded = json.loads(result.stdout)
    forbidden = [name for name in BUILTIN_ENTRY_MODULES if name in loaded]
    assert forbidden == [], (
        f"importing robot_config.launch_registry.registry loaded builtin entry modules {forbidden}; "
        "the framework core must stay import-light (design decision 3)"
    )
    allowed = {
        "robot_config",
        "robot_config.config",
        "robot_config.observation_transport",
        "robot_config.launch_registry",
        "robot_config.launch_registry.registry",
        "robot_config.launch_registry.launch_options",
        "robot_config.launch_builders",
        "robot_config.launch_builders.tracing",
        "robot_config.logger_utils",
        "robot_config.runtime_target",
    }
    unexpected = [name for name in loaded if name not in allowed]
    assert unexpected == [], f"registry core imported unexpected robot_config modules: {unexpected}"


def test_real_discovery_stays_within_the_core_budget_plus_discovered_packages():
    """Real entry-point discovery (openspec change
    ``migrate-business-launch-builders``, design decision 6): with the
    extensions that happen to be installed, ``build_launch_registry()`` may
    only import non-stdlib top-level packages from the core budget plus the
    top-level packages owning the discovered entry points — every extension
    module (and its package ``__init__``) must stay import-light because
    this runs on every ``ros2 launch robot_config robot.launch.py``
    description generation."""
    from importlib.metadata import entry_points

    from robot_config.launch_registry.builtin_registry import LAUNCH_BUILDER_ENTRY_POINT_GROUP

    discovered = list(entry_points(group=LAUNCH_BUILDER_ENTRY_POINT_GROUP))
    owning_packages = {str(entry.value).partition(":")[0].partition(".")[0] for entry in discovered}
    assert owning_packages, "no launch-builder entry points discovered in this environment"

    core = core_import_budget()
    registry = imported_top_level_packages(
        "from robot_config.launch_registry.builtin_registry import build_launch_registry\nbuild_launch_registry()\n"
    )
    overflow = sorted(registry - core - HARD_DEPENDENCY_ALLOWANCE - owning_packages)
    assert overflow == [], (
        f"real discovery imported non-core top-level packages {overflow}; "
        "extension modules (and their package __init__) must stay import-light (design decision 6)"
    )
    # Every discovered entry point's owning package was indeed imported by
    # the registry construction (the resolution itself), nothing more.
    assert owning_packages <= registry, (
        f"discovered entry point packages missing from the import set: {sorted(owning_packages - registry)}"
    )
